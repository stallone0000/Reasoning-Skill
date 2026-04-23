# -*- coding: utf-8 -*-
"""
gen_qiniu_oss.py

批量调用 qiniu/gpt-oss-120b 为 nemotron_cp_questions 生成代码。

输入:  nemotron_cp_questions_34799_v1.jsonl
       每行: {"unique_key": ..., "question": ..., ...}

输出:  qiniu_oss_cp_34799_v1.jsonl
       每行: {"unique_key": ..., "llm_output": "..."}

特性:
  - 500 线程并发
  - 断点续传（已有输出自动跳过）
  - 指数退避重试
  - tqdm 进度条
  - 支持 reasoning_content（thinking token）

用法:
  python gen_qiniu_oss.py
  python gen_qiniu_oss.py --input xxx.jsonl --output yyy.jsonl --workers 500
"""

import os
import json
import time
import argparse
import threading
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from rm_runtime import QihooChatClient, resolve_api_key


# =========================
# 默认配置
# =========================

CLIENT = None

DEFAULT_MODEL = "qiniu/gpt-oss-120b"
DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "nemotron_cp_questions_34799_v1.jsonl")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "qiniu_oss_cp_34799_v1_with_prompt.jsonl")
DEFAULT_WORKERS = 500


# =========================
# JSONL 工具
# =========================

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_done_keys(output_path):
    """加载已完成的 unique_key 用于断点续传。"""
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


# =========================
# API 调用
# =========================

def call_qiniu_api(model, messages, max_retries=6, initial_retry_delay=5, timeout=900):
    """
    通用 API 调用封装：
    - 最多 max_retries 次（这是 API 层面的重试）
    - 指数退避：initial_retry_delay * (2 ** attempt)
    - 成功返回文本；失败返回 None
    """
    if CLIENT is None:
        raise RuntimeError("Qihoo chat client is not initialized")

    result = CLIENT.chat(
        model=model,
        messages=messages,
        timeout=timeout,
        max_retries=max_retries,
        initial_retry_delay=initial_retry_delay,
        include_reasoning=True,
        base_payload={"content_filter": False},
    )
    if result is None:
        print(f"[ERROR] model={model} 超过最大重试次数，返回 None")
        return None
    return result.content


# =========================
# 单条处理
# =========================

def process_one(unique_key, question, model):
    """
    对一道题调用模型生成，返回 (unique_key, llm_output)。
    在题目前添加英文描述，要求生成能AC通过的C++和Python代码。
    """
    # 添加英文描述
    prompt_prefix = """This is a competitive programming problem. Please solve it and provide both C++ and Python code that can pass all test cases (AC).

Please provide:
1. Your solution approach and reasoning
2. C++ code that can compile and pass all test cases
3. Python code that can pass all test cases

The code should be complete, correct, and ready to submit.

Problem:
"""
    
    full_prompt = prompt_prefix + question
    
    messages = [
        {"role": "user", "content": full_prompt},
    ]
    output = call_qiniu_api(model, messages)
    return unique_key, output


# =========================
# 主流程
# =========================

def main():
    parser = argparse.ArgumentParser(description="批量调用 qiniu/gpt-oss-120b 生成 coding 题解")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="输入 JSONL 文件路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 JSONL 文件路径")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并发线程数")
    parser.add_argument("--max_retries", type=int, default=6, help="单次 API 最大重试次数")
    parser.add_argument("--timeout", type=int, default=900, help="单次 API 超时秒数")
    parser.add_argument("--api_key", type=str, default=None, help="API Key（覆盖环境变量）")
    parser.add_argument("--api_url", type=str, default=None, help="API URL（覆盖环境变量）")
    parser.add_argument("--api_host", type=str, default=None, help="Host 头（覆盖环境变量）")
    parser.add_argument("--limit", type=int, default=None, help="限制处理的题目数量（用于测试）")
    args = parser.parse_args()

    global CLIENT
    try:
        api_key = resolve_api_key(args.api_key, required=True)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return
    CLIENT = QihooChatClient(api_key=api_key, api_url=args.api_url, api_host=args.api_host)

    # 1) 加载已完成的 key（断点续传）
    done_keys = load_done_keys(args.output)
    if done_keys:
        print(f"[INFO] 断点续传：已有 {len(done_keys)} 条结果，将跳过这些。")

    # 2) 加载待处理的题目
    print(f"[INFO] 读取输入文件: {args.input}")
    todo_items = []
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

    print(f"[INFO] 输入文件共 {total_in_file} 条，待处理 {len(todo_items)} 条")
    
    # 如果设置了 limit，只处理前 limit 条
    if args.limit is not None and args.limit > 0:
        todo_items = todo_items[:args.limit]
        print(f"[INFO] 限制处理数量: {len(todo_items)} 条")

    if not todo_items:
        print("[INFO] 没有需要处理的条目，退出。")
        return

    # 3) 线程安全写入
    write_lock = threading.Lock()
    fout = open(args.output, "a", encoding="utf-8")
    n_success = 0
    n_fail = 0

    def write_result(unique_key, llm_output):
        nonlocal n_success, n_fail
        # 如果输出为空或 None，不写入记录（避免产生空输出）
        # 这些失败的记录可以在后续重试时重新处理
        if llm_output is None or not llm_output.strip():
            n_fail += 1
            return
        
        row = {
            "unique_key": unique_key,
            "llm_output": llm_output,
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with write_lock:
            fout.write(line)
            n_success += 1

    # 4) 并发调用
    t0 = time.time()
    print(f"[INFO] 开始生成，模型={args.model}，并发={args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, uk, q, args.model): uk
            for uk, q in todo_items
        }

        with tqdm(total=len(futures), desc="Generating", unit="item") as pbar:
            for fut in as_completed(futures):
                uk = futures[fut]
                try:
                    unique_key, llm_output = fut.result()
                    write_result(unique_key, llm_output)
                except Exception as e:
                    print(f"[EXCEPTION] unique_key={uk}, error={e}")
                    # 不写入失败记录，避免产生空输出
                    n_fail += 1

                pbar.update(1)

                # 定期 flush
                if (pbar.n) % 200 == 0:
                    with write_lock:
                        fout.flush()

    fout.close()

    elapsed = time.time() - t0
    total_done = n_success + n_fail
    print(f"\n{'='*60}")
    print(f"生成完成: {total_done} 条, 用时 {elapsed:.1f}s ({total_done/elapsed:.1f} items/s)")
    print(f"  成功: {n_success} 条")
    print(f"  失败: {n_fail} 条")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
