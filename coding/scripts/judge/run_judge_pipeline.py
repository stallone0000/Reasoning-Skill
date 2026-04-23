# -*- coding: utf-8 -*-
"""
run_judge_pipeline.py

批量 judge Gemini Pro / Flash 生成的 coding 题结果。

流程:
  1. 流式读取 Pro/Flash JSON array 文件
  2. 用正则匹配 thought 结束 tag，拆分 think / answer
  3. 从 answer 部分提取代码
  4. 用 cases JSONL 中的测试数据 judge
  5. 每条记录写出 JSONL，包含完整的 judge 信息

输出 JSONL 每行格式:
  {
    "unique_key": str,
    "split_ok": bool,
    "end_token": str|null,
    "judge": {
      "status": str,       # AC / WA / TLE / RE / CE / NO_TESTS / NO_CASES / SPLIT_FAIL / NO_OUTPUT / ...
      "lang": str|null,    # python / cpp
      "passed": int,       # 通过的 test case 数
      "total_tests": int,  # 总 test case 数
      "case_results": [    # 每个 test case 的详细结果
        {"case": int, "status": str, ...},  # AC / WA / TLE / RE / SKIPPED
      ],
      "detail": str|null,  # CE 编译错误 / RE stderr / WA 详情
    }
  }

用法:
  python run_judge_pipeline.py \
    --input  nemotron_cp_unique_questions_34729_withimages_pro.json \
    --cases  nemotron_cp_cases_34799_v1.jsonl \
    --output judge_results_pro.jsonl \
    --workers 4 \
    --timeout 10

安全:
  这不是沙箱环境。脚本会在本机编译/执行用户代码。
  请在 docker/nsjail/bwrap 等隔离环境中运行。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

# ── 导入 judge_cp（同目录下） ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_cp  # noqa: E402

# ──────────────────────────────────────
# Thought end token regex (同 notebook)
# ──────────────────────────────────────

THOUGHT_END_RE = re.compile(
    r'<\s*/\s*(?:thought|think|analysis|reasoning)\s*>'
    r'|<\s*(?:thought|think)\s*/\s*>'
    r'|<\s*(?:thought|think)\s+/\s*>'
    r'|<\s*end_?(?:thought|think)s?\s*>'
    r'|<\s*(?:thought|think)_?ends?\s*>'
    r'|<\s*(?:thought|think)_process_end\s*>',
    re.IGNORECASE,
)

OUTPUT_KEYS = ['gemini_outputs', 'correct_gemini_outputs']


# ──────────────────────────────────────
# 流式 JSON array 解析器
# ──────────────────────────────────────

def iter_json_array(path: Path) -> Iterator[dict]:
    """流式逐条解析 JSON array 文件。"""
    with path.open('r', encoding='utf-8') as f:
        ch = f.read(1)
        while ch and ch.isspace():
            ch = f.read(1)
        if ch != '[':
            raise ValueError(f'{path} is not a JSON array file')

        started = False
        depth = 0
        in_str = False
        esc = False
        buf = ''

        while True:
            ch = f.read(1)
            if not ch:
                break

            if not started:
                if ch.isspace() or ch == ',':
                    continue
                if ch == ']':
                    break
                if ch != '{':
                    continue
                started = True
                depth = 1
                in_str = False
                esc = False
                buf = '{'
                continue

            buf += ch
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    yield json.loads(buf)
                    started = False
                    buf = ''


# ──────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────

def first_output_text(rec: dict) -> Optional[str]:
    for k in OUTPUT_KEYS:
        v = rec.get(k)
        if isinstance(v, list) and v:
            if isinstance(v[0], str) and v[0].strip():
                return v[0]
    return None


def split_think_answer(text: str) -> Optional[Tuple[str, str, str]]:
    """拆分 think / answer。返回 (think, answer, end_token) 或 None。"""
    if not text:
        return None
    matches = list(THOUGHT_END_RE.finditer(text))
    if not matches:
        return None
    best = max(matches, key=lambda m: m.start())
    think = text[:best.start()].strip()
    answer = text[best.end():].strip()
    end_tok = best.group(0)
    if not answer:
        return None
    return think, answer, end_tok


def flatten_judge_result(raw: dict) -> dict:
    """
    把 judge_one 的返回整理成统一格式的 judge 信息字典。
    judge_one 现在返回 case_results: List[dict]，每个 case 有独立状态。
    """
    status = raw.get("status", "UNKNOWN")
    lang = raw.get("lang")
    passed = raw.get("passed", 0)
    total_tests = raw.get("total_tests", 0)
    case_results = raw.get("case_results", [])

    # 组装 top-level 详情
    detail_parts = []
    if raw.get("compile_error"):
        detail_parts.append(f"[compile_error] {raw['compile_error']}")
    if raw.get("detail"):
        detail_parts.append(f"[detail] {raw['detail']}")
    if raw.get("error"):
        detail_parts.append(f"[error] {raw['error']}")
    detail = "\n".join(detail_parts) if detail_parts else None

    return {
        "status": status,
        "lang": lang,
        "passed": passed,
        "total_tests": total_tests,
        "case_results": case_results,
        "detail": detail,
    }


# ──────────────────────────────────────
# 单条 judge 任务
# ──────────────────────────────────────

def judge_one_item(
    rec: dict,
    cases_idx: Dict[str, dict],
    cfg: judge_cp.JudgeConfig,
) -> dict:
    """
    对单条记录做完整的拆分 + judge。

    返回一个 dict，包含:
      unique_key, split_ok, end_token, response (标准化后), judge (judge 信息)
    """
    uk = rec.get("unique_key", "")

    # 1) 获取模型输出
    text = first_output_text(rec)
    if not text:
        return {
            "unique_key": uk,
            "split_ok": False,
            "end_token": None,
            "response": None,
            "judge": {
                "status": "NO_OUTPUT",
                "lang": None,
                "passed": 0,
                "total_tests": 0,
                "case_results": [],
                "detail": None,
            },
        }

    # 2) 拆分 think / answer
    sp = split_think_answer(text)
    if sp is None:
        # 拆分失败 → 尝试从全文提取代码 fence 作为 answer
        hint, code = judge_cp.extract_code(text)
        if hint is not None and code.strip():
            # 找到了有语言标记的代码 fence，用它做 answer
            think_part = ""
            answer_part = text  # 仍传完整文本，让 judge_one 内部 extract_code 来定位代码
            end_tok = None
            split_ok = False
        else:
            think_part = ""
            answer_part = text
            end_tok = None
            split_ok = False
    else:
        think_part, answer_part, end_tok = sp
        split_ok = True

    # 3) 标准化 response
    if split_ok and think_part:
        response = f"<think>\n{think_part}\n</think>\n{answer_part}"
    else:
        response = answer_part

    # 4) 查找 cases
    prob = cases_idx.get(uk)
    if not prob:
        return {
            "unique_key": uk,
            "split_ok": split_ok,
            "end_token": end_tok,
            "response": response,
            "judge": {
                "status": "NO_CASES",
                "lang": None,
                "passed": 0,
                "total_tests": 0,
                "case_results": [],
                "detail": f"unique_key '{uk}' not found in cases file",
            },
        }

    # 5) judge
    try:
        raw = judge_cp.judge_one(prob, answer_part, cfg)
    except Exception as e:
        raw = {"status": "JUDGE_ERROR", "error": str(e)[:2000], "case_results": []}

    return {
        "unique_key": uk,
        "split_ok": split_ok,
        "end_token": end_tok,
        "response": response,
        "judge": flatten_judge_result(raw),
    }


# ──────────────────────────────────────
# 批量 pipeline
# ──────────────────────────────────────

def load_cases_index(path: Path) -> Dict[str, dict]:
    print(f"Loading cases index from {path} ...")
    idx: Dict[str, dict] = {}
    for row in judge_cp.iter_jsonl(str(path)):
        k = row.get("unique_key")
        if k:
            idx[k] = row
    print(f"  loaded {len(idx)} unique cases")
    return idx


def load_done_keys(out_path: Path) -> set:
    """如果输出文件已存在，加载已完成的 key 用于断点续传。"""
    done = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    uk = json.loads(line).get("unique_key")
                    if uk:
                        done.add(uk)
                except Exception:
                    pass
    return done


def run_pipeline(
    input_path: Path,
    cases_path: Path,
    output_path: Path,
    workers: int = 4,
    timeout_s: float = 10.0,
    mem_mb: int = 1024,
    include_private: bool = False,
    save_response: bool = False,
    limit: Optional[int] = None,
):
    cfg = judge_cp.JudgeConfig(
        float_tol=1e-6,
        include_private=include_private,
        default_timeout_s=timeout_s,
        default_mem_mb=mem_mb,
    )

    cases_idx = load_cases_index(cases_path)
    done_keys = load_done_keys(output_path)
    if done_keys:
        print(f"Resume: {len(done_keys)} items already judged, will skip them.")

    # 收集待 judge 的记录
    print(f"Streaming input from {input_path} ...")
    records = []
    n_skipped = 0
    for rec in iter_json_array(input_path):
        uk = rec.get("unique_key", "")
        if uk in done_keys:
            n_skipped += 1
            continue
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    print(f"  loaded {len(records)} records to judge (skipped {n_skipped} already done)")

    if not records:
        print("Nothing to judge.")
        return

    # 统计
    status_counter: Counter = Counter()
    t0 = time.time()
    n_done = 0

    # 打开输出文件（追加模式），线程锁保护写入
    fout = output_path.open("a", encoding="utf-8")
    import threading
    _write_lock = threading.Lock()

    def process_result(result: dict):
        nonlocal n_done
        # 如果不保存完整 response，只保留前 200 字预览
        if not save_response and result.get("response"):
            result["response_preview"] = (result["response"] or "")[:200]
            del result["response"]
        line = json.dumps(result, ensure_ascii=False) + "\n"
        with _write_lock:
            fout.write(line)
            status_counter[result["judge"]["status"]] += 1
            n_done += 1

    # 并行 judge
    if workers <= 1:
        for i, rec in enumerate(records):
            result = judge_one_item(rec, cases_idx, cfg)
            process_result(result)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(records) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(records)}] {rate:.1f} items/s  ETA {eta:.0f}s  | {dict(status_counter)}")
                fout.flush()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(judge_one_item, rec, cases_idx, cfg): rec.get("unique_key", "")
                for rec in records
            }
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except Exception as e:
                    uk = futures[fut]
                    result = {
                        "unique_key": uk,
                        "split_ok": False,
                        "end_token": None,
                        "judge": {
                            "status": "JUDGE_ERROR",
                            "lang": None,
                            "passed": 0,
                            "total_tests": 0,
                            "case_results": [],
                            "detail": str(e)[:2000],
                        },
                    }
                process_result(result)
                if n_done % 100 == 0:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    remaining = len(records) - n_done
                    eta = remaining / rate if rate > 0 else 0
                    print(f"  [{n_done}/{len(records)}] {rate:.1f} items/s  ETA {eta:.0f}s  | {dict(status_counter)}")
                    fout.flush()

    fout.close()

    # 最终统计
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Judge 完成: {n_done} items in {elapsed:.1f}s ({n_done/elapsed:.1f} items/s)")
    print(f"输出文件: {output_path}")
    print(f"\n状态分布:")
    for st, c in status_counter.most_common():
        pct = c / n_done * 100 if n_done else 0
        print(f"  {st:<30s} {c:>6d}  ({pct:5.1f}%)")


# ──────────────────────────────────────
# CLI
# ──────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Batch judge Gemini coding outputs")
    ap.add_argument("--input", required=True, help="Gemini output JSON array file (pro or flash)")
    ap.add_argument("--cases", required=True, help="Cases JSONL file (nemotron_cp_cases_34799_v1.jsonl)")
    ap.add_argument("--output", required=True, help="Output JSONL file for judge results")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel judge workers (default: 4)")
    ap.add_argument("--timeout", type=float, default=10.0, help="Timeout per test case in seconds (default: 10)")
    ap.add_argument("--mem_mb", type=int, default=1024, help="Memory limit per run in MB (default: 1024)")
    ap.add_argument("--include_private", action="store_true", help="Include private test cases")
    ap.add_argument("--save_response", action="store_true", help="Save full standardized response (large!)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of items to judge (for testing)")
    args = ap.parse_args()

    run_pipeline(
        input_path=Path(args.input),
        cases_path=Path(args.cases),
        output_path=Path(args.output),
        workers=args.workers,
        timeout_s=args.timeout,
        mem_mb=args.mem_mb,
        include_private=args.include_private,
        save_response=args.save_response,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
