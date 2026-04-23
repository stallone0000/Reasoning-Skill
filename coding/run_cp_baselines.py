#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cp_baselines.py

Baseline experiments for competitive programming (CP):
  3 models × 4 prompt conditions = 12 experiments

Models:
  qiniu/gpt-oss-120b          (oss120b)
  volcengine/doubao-seed-2-0-pro  (doubao)
  cloudsway/gemini-3-flash-preview (gemini)

Prompt conditions:
  direct : 直接调用, 仅 CP 指令 + 问题 (无额外 prompt 策略)
  nowait : CP 指令 + NoWait prompt (禁止等待类词汇)
  cod    : CP 指令 + CoD prompt (Chain of Draft, 最小草稿)
  tale   : CP 指令 + TALE prompt (先预估 budget，再约束 tokens 解题)

所有条件均包含 CP 指令:
  "Solve this competitive programming problem using Python or C++."

输出格式 (JSONL, 每行):
  {
    "unique_key": "...",
    "llm_output": "...",       # 模型最终输出 (content only, 无 reasoning_content 拼接)
    "completion_tokens": 123,  # 从 API usage 中取
    "prompt_tokens": 456
  }

特性:
  - 使用新端点 api.zhinao.qihoo.net (无需代理)
  - 只取 content，不拼 reasoning_content (正常调用场景)
  - 断点续传
  - 指数退避重试
  - tqdm 进度条
  - 多线程并发

用法:
  python run_cp_baselines.py --model qiniu/gpt-oss-120b --prompt direct
  python run_cp_baselines.py --model volcengine/doubao-seed-2-0-pro --prompt nowait --workers 200
  python run_cp_baselines.py --model cloudsway/gemini-3-flash-preview --prompt tale --workers 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

RM_ROOT = None
for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "LATEST_STATUS.md").exists() and (_candidate / "README.md").exists():
        RM_ROOT = _candidate
        break
if RM_ROOT is None:
    raise RuntimeError("Cannot locate reasoning_memory root")
if str(RM_ROOT) not in sys.path:
    sys.path.insert(0, str(RM_ROOT))

from rm_runtime import DEFAULT_SAMPLING_PAYLOAD, QihooChatClient, resolve_api_key

# ============================================================
# 模型简称映射
# ============================================================
MODEL_SHORTS = {
    "qiniu/gpt-oss-120b": "oss120b",
    "volcengine/doubao-seed-2-0-pro": "doubao",
    "cloudsway/gemini-3-flash-preview": "gemini",
}

# ============================================================
# Prompt 模板路径
# ============================================================
PROMPT_DIR = "/home/jovyan/zhaoguangxiang-data/shiqilong/prompt_bank/DeepMath-103K/cot-bank"
NOWAIT_SOLVE_PATH = os.path.join(PROMPT_DIR, "NoWait", "solve.txt")
COD_SOLVE_PATH    = os.path.join(PROMPT_DIR, "CoD", "solve.txt")
TALE_BUDGET_PATH  = os.path.join(PROMPT_DIR, "TALE", "budget.txt")
TALE_SOLVE_PATH   = os.path.join(PROMPT_DIR, "TALE", "solve.txt")

# 所有条件共用的 CP 指令 (必须出现在每个 prompt 中)
CP_INSTRUCTION = "Solve this competitive programming problem using Python or C++."

# 直接调用 prompt: CP 指令 + 问题 (无额外策略)
DIRECT_PROMPT_TEMPLATE = (
    CP_INSTRUCTION + "\n\n"
    "Problem:\n{question}"
)

# ============================================================
# 默认路径
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # reasoning_memory/
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "cp_test_1000.jsonl")
DEFAULT_CASES = os.path.join(PARENT_DIR, "nemotron_cp_cases_34799_v1.jsonl")
CLIENT: Optional[QihooChatClient] = None


# ============================================================
# JSONL 工具
# ============================================================

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_done_keys(output_path: str) -> set:
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    uk = obj.get("unique_key")
                    if uk:
                        done.add(uk)
                except Exception:
                    pass
    return done


# ============================================================
# API 调用 (返回 content + usage)
# ============================================================

def call_api_with_usage(
    model: str,
    messages: List[Dict[str, str]],
    max_retries: int = 6,
    initial_retry_delay: float = 5,
    timeout: float = 900,
) -> Tuple[Optional[str], int, int]:
    """
    调用 360 API，返回 (content, completion_tokens, prompt_tokens)。

    正常调用模式:
      - 只取 content，不拼 reasoning_content
      - extra_body={} (显式置空，不给 DeepSeek 开 thinking)
      - 不带 assistant prompt

    失败返回 (None, 0, 0)。
    """
    if CLIENT is None:
        raise RuntimeError("Qihoo chat client is not initialized")

    result = CLIENT.chat(
        model=model,
        messages=messages,
        timeout=timeout,
        max_retries=max_retries,
        initial_retry_delay=initial_retry_delay,
        base_payload=DEFAULT_SAMPLING_PAYLOAD,
    )
    if result is None:
        print(f"[FAIL] model={model}, max retries ({max_retries}) exceeded")
        return None, 0, 0

    comp_tokens = int(result.usage.get("completion_tokens", 0) or 0)
    prompt_tokens = int(result.usage.get("prompt_tokens", 0) or 0)
    return result.content, comp_tokens, prompt_tokens


# ============================================================
# Prompt 构造
# ============================================================

def build_direct_prompt(question: str) -> str:
    """直接调用: CP 指令 + 问题 (无额外推理策略)"""
    return DIRECT_PROMPT_TEMPLATE.format(question=question)


def build_template_prompt(template: str, question: str) -> str:
    """
    NoWait / CoD / TALE 模板: 在 [Question Here] 中嵌入 CP 指令 + 原题。
    这样所有实验条件都包含相同的 CP 指令，仅推理策略不同。
    """
    enriched_question = f"{CP_INSTRUCTION}\n\n{question}"
    return template.replace("[Question Here]", enriched_question)


def extract_budget(text: str) -> int:
    """从 TALE budget 模型输出中提取数字。"""
    m = re.search(r'\[\[(\d+)\]\]', text)
    if m:
        return int(m.group(1))
    m = re.search(r'Budget:\s*\[\[(\d+)\]\]', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)', text)
    if m:
        return int(m.group(1))
    return 512  # 默认值


# ============================================================
# 单条处理函数
# ============================================================

def process_direct(uk: str, question: str, model: str) -> dict:
    """直接调用: CP 指令 + 问题, 无额外推理策略"""
    prompt = build_direct_prompt(question)
    messages = [{"role": "user", "content": prompt}]
    content, comp_toks, prompt_toks = call_api_with_usage(model, messages)
    return {
        "unique_key": uk,
        "llm_output": content,
        "completion_tokens": comp_toks,
        "prompt_tokens": prompt_toks,
    }


def process_template(uk: str, question: str, model: str, template: str) -> dict:
    """NoWait / CoD 使用。"""
    prompt = build_template_prompt(template, question)
    messages = [{"role": "user", "content": prompt}]
    content, comp_toks, prompt_toks = call_api_with_usage(model, messages)
    return {
        "unique_key": uk,
        "llm_output": content,
        "completion_tokens": comp_toks,
        "prompt_tokens": prompt_toks,
    }


def process_tale(
    uk: str, question: str, model: str,
    budget_template: str, solve_template: str,
) -> dict:
    """TALE: 2 步 (budget → solve with budget)。"""
    # Step 1: 获取 budget
    budget_prompt = build_template_prompt(budget_template, question)
    messages_budget = [{"role": "user", "content": budget_prompt}]
    budget_content, b_comp, b_prompt = call_api_with_usage(
        model, messages_budget, timeout=180
    )
    budget = extract_budget(budget_content or "")

    # Step 2: 用 budget 约束解题
    solve_prompt = build_template_prompt(solve_template, question)
    solve_prompt = solve_prompt.replace("[Budget Here]", str(budget))
    messages_solve = [{"role": "user", "content": solve_prompt}]
    content, s_comp, s_prompt = call_api_with_usage(model, messages_solve)

    return {
        "unique_key": uk,
        "llm_output": content,
        "completion_tokens": b_comp + s_comp,
        "prompt_tokens": b_prompt + s_prompt,
        "tale_budget": budget,
    }


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CP Baseline: 生成模型输出 (JSONL, 含 completion_tokens)"
    )
    parser.add_argument(
        "--model", required=True,
        choices=list(MODEL_SHORTS.keys()),
        help="模型名称"
    )
    parser.add_argument(
        "--prompt", required=True,
        choices=["direct", "nowait", "cod", "tale"],
        help="Prompt 条件 (direct=直接调用, nowait/cod/tale=带策略prompt)"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="输入 JSONL")
    parser.add_argument("--output", default=None, help="输出 JSONL (默认自动命名)")
    parser.add_argument("--workers", type=int, default=200, help="并发线程数")
    parser.add_argument("--limit", type=int, default=None, help="限制处理数量")
    parser.add_argument("--max_retries", type=int, default=6, help="API 最大重试次数")
    parser.add_argument("--timeout", type=int, default=900, help="单次 API 超时秒数")
    parser.add_argument("--api_key", default=None, help="API key（默认从环境变量读取）")
    parser.add_argument("--api_url", default=None, help="API URL（默认从环境变量或共享默认值读取）")
    parser.add_argument("--api_host", default=None, help="Host 头（默认从环境变量或共享默认值读取）")
    args = parser.parse_args()

    global CLIENT
    try:
        api_key = resolve_api_key(args.api_key, required=True)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return
    CLIENT = QihooChatClient(api_key=api_key, api_url=args.api_url, api_host=args.api_host)

    model_short = MODEL_SHORTS[args.model]

    # 自动命名输出文件
    if args.output is None:
        args.output = os.path.join(SCRIPT_DIR, f"gen_{args.prompt}_{model_short}.jsonl")

    print("=" * 60)
    print(f"CP Baseline Generation")
    print(f"  Model  : {args.model} ({model_short})")
    print(f"  Prompt : {args.prompt}")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Workers: {args.workers}")
    print("=" * 60)

    # 加载 prompt 模板
    templates: Dict[str, str] = {}
    if args.prompt == "nowait":
        with open(NOWAIT_SOLVE_PATH, "r") as f:
            templates["solve"] = f.read()
        print(f"  NoWait solve template loaded from {NOWAIT_SOLVE_PATH}")
    elif args.prompt == "cod":
        with open(COD_SOLVE_PATH, "r") as f:
            templates["solve"] = f.read()
        print(f"  CoD solve template loaded from {COD_SOLVE_PATH}")
    elif args.prompt == "tale":
        with open(TALE_BUDGET_PATH, "r") as f:
            templates["budget"] = f.read()
        with open(TALE_SOLVE_PATH, "r") as f:
            templates["solve"] = f.read()
        print(f"  TALE budget template loaded from {TALE_BUDGET_PATH}")
        print(f"  TALE solve template loaded from {TALE_SOLVE_PATH}")

    # 断点续传
    done_keys = load_done_keys(args.output)
    if done_keys:
        print(f"  Resume: {len(done_keys)} items already done, will skip.")

    # 加载待处理的题目
    print(f"\n读取输入文件: {args.input}")
    todo_items: List[Tuple[str, str]] = []
    total_in_file = 0
    for row in iter_jsonl(args.input):
        total_in_file += 1
        uk = row.get("unique_key", "")
        if uk in done_keys:
            continue
        question = row.get("question", "")
        if not question.strip():
            continue
        todo_items.append((uk, question))

    print(f"  输入文件共 {total_in_file} 条, 待处理 {len(todo_items)} 条")

    if args.limit is not None and args.limit > 0:
        todo_items = todo_items[:args.limit]
        print(f"  限制处理数量: {len(todo_items)} 条")

    if not todo_items:
        print("\n没有需要处理的条目，退出。")
        return

    # 线程安全写入
    write_lock = threading.Lock()
    fout = open(args.output, "a", encoding="utf-8")
    n_success = 0
    n_fail = 0
    total_comp_tokens = 0

    def write_result(result: dict):
        nonlocal n_success, n_fail, total_comp_tokens
        content = result.get("llm_output")
        if content is None or not str(content).strip():
            n_fail += 1
            return
        line = json.dumps(result, ensure_ascii=False) + "\n"
        with write_lock:
            fout.write(line)
            fout.flush()
            n_success += 1
            total_comp_tokens += result.get("completion_tokens", 0)

    # 提交函数
    def submit_one(uk: str, q: str) -> dict:
        if args.prompt == "direct":
            return process_direct(uk, q, args.model)
        elif args.prompt == "nowait":
            return process_template(uk, q, args.model, templates["solve"])
        elif args.prompt == "cod":
            return process_template(uk, q, args.model, templates["solve"])
        elif args.prompt == "tale":
            return process_tale(
                uk, q, args.model, templates["budget"], templates["solve"]
            )
        else:
            raise ValueError(f"Unknown prompt: {args.prompt}")

    # 并发执行
    t0 = time.time()
    print(f"\n开始生成, 模型={args.model}, prompt={args.prompt}, 并发={args.workers}")
    print("-" * 60)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(submit_one, uk, q): uk
            for uk, q in todo_items
        }

        desc = f"{args.prompt}/{model_short}"
        with tqdm(total=len(futures), desc=desc, unit="item") as pbar:
            for fut in as_completed(futures):
                uk = futures[fut]
                try:
                    result = fut.result()
                    write_result(result)
                except Exception as e:
                    print(f"[EXCEPTION] uk={uk}, error={e}")
                    n_fail += 1

                pbar.update(1)

                # 定期刷新
                if pbar.n % 500 == 0:
                    with write_lock:
                        fout.flush()

    fout.close()

    elapsed = time.time() - t0
    total_done = n_success + n_fail
    avg_comp = total_comp_tokens / n_success if n_success > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"生成完成: {total_done} 条, 用时 {elapsed:.1f}s ({total_done/max(elapsed,0.1):.1f} items/s)")
    print(f"  成功: {n_success} 条")
    print(f"  失败: {n_fail} 条")
    print(f"  总 completion_tokens: {total_comp_tokens:,}")
    print(f"  平均 completion_tokens: {avg_comp:,.0f}")
    print(f"  输出文件: {args.output}")
    print(f"{'=' * 60}")

    # 写一份摘要到 .summary.json
    summary_path = args.output.replace(".jsonl", ".summary.json")
    summary = {
        "model": args.model,
        "model_short": model_short,
        "prompt": args.prompt,
        "input_file": args.input,
        "output_file": args.output,
        "total_in_file": total_in_file,
        "total_processed": total_done,
        "success": n_success,
        "fail": n_fail,
        "total_completion_tokens": total_comp_tokens,
        "avg_completion_tokens": round(avg_comp, 1),
        "elapsed_seconds": round(elapsed, 1),
        "workers": args.workers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  摘要: {summary_path}")


if __name__ == "__main__":
    main()
