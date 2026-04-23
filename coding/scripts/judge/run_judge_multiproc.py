# -*- coding: utf-8 -*-
"""
run_judge_multiproc.py

高性能多进程 judge pipeline, 专为 Linux 多核服务器优化。

核心优化:
  1. 使用 multiprocessing.Pool (fork) 替代 ThreadPoolExecutor
     - 每个进程有独立 GIL, 无争抢
     - fork 模式下 cases_idx 通过 copy-on-write 共享, 内存高效
  2. 使用 imap_unordered 流式获取结果, 不需要一次创建所有 Future
  3. 每完成 1 条立即写入, 每 10 条 flush 一次
  4. 详细的实时进度和 ETA

用法:
  # 标准用法 (32 进程)
  python3 run_judge_multiproc.py \
    --input  nemotron_cp_unique_questions_34729_withimages_pro.json \
    --cases  nemotron_cp_cases_34799_v1.jsonl \
    --output judge_results_pro.jsonl \
    --workers 32

  # 测试用法
  python3 run_judge_multiproc.py \
    --input  nemotron_cp_unique_questions_34729_withimages_pro.json \
    --cases  nemotron_cp_cases_34799_v1.jsonl \
    --output /tmp/test.jsonl \
    --workers 4 --limit 20

安全:
  这不是沙箱环境。脚本会在本机编译/执行用户代码。
  请在 docker/nsjail/bwrap 等隔离环境中运行。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

# ── 导入 judge_cp（同目录下） ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_cp  # noqa: E402

# ── 全局变量 (fork 后子进程共享) ──
_CASES_IDX: Dict[str, dict] = {}
_CFG: judge_cp.JudgeConfig = judge_cp.JudgeConfig()
_MAX_JUDGE_ERROR_RETRIES: int = 0

# ──────────────────────────────────────
# Thought end token regex
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
    status = raw.get("status", "UNKNOWN")
    lang = raw.get("lang")
    passed = raw.get("passed", 0)
    total_tests = raw.get("total_tests", 0)
    case_results = raw.get("case_results", [])

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
# Worker 函数 (在子进程中执行)
# ──────────────────────────────────────

def _worker_judge(rec: dict) -> dict:
    """
    在子进程中对单条记录做完整的拆分 + judge。
    使用全局 _CASES_IDX 和 _CFG (fork 后共享, copy-on-write)。
    """
    uk = rec.get("unique_key", "")

    # 1) 获取模型输出
    text = first_output_text(rec)
    if not text:
        return {
            "unique_key": uk,
            "split_ok": False,
            "end_token": None,
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
        hint, code = judge_cp.extract_code(text)
        if hint is not None and code.strip():
            answer_part = text
            end_tok = None
            split_ok = False
        else:
            answer_part = text
            end_tok = None
            split_ok = False
    else:
        _, answer_part, end_tok = sp
        split_ok = True

    # 3) 查找 cases
    prob = _CASES_IDX.get(uk)
    if not prob:
        return {
            "unique_key": uk,
            "split_ok": split_ok,
            "end_token": end_tok,
            "judge": {
                "status": "NO_CASES",
                "lang": None,
                "passed": 0,
                "total_tests": 0,
                "case_results": [],
                "detail": f"unique_key '{uk}' not found in cases file",
            },
        }

    # 4) judge
    try:
        raw = judge_cp.judge_one(prob, answer_part, _CFG)
    except Exception as e:
        raw = {"status": "JUDGE_ERROR", "error": str(e)[:2000], "case_results": []}

    return {
        "unique_key": uk,
        "split_ok": split_ok,
        "end_token": end_tok,
        "judge": flatten_judge_result(raw),
    }


def _worker_judge_with_retry(rec: dict) -> dict:
    """
    对 JUDGE_ERROR 自动重试，避免瞬时异常导致结果抖动。
    """
    attempts = max(0, int(_MAX_JUDGE_ERROR_RETRIES))
    result = _worker_judge(rec)
    retry_count = 0
    while retry_count < attempts and result.get("judge", {}).get("status") == "JUDGE_ERROR":
        retry_count += 1
        result = _worker_judge(rec)
    result["judge_retry_count"] = retry_count
    return result


# ──────────────────────────────────────
# 加载与工具
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


# ──────────────────────────────────────
# 主 pipeline
# ──────────────────────────────────────

def run_pipeline(
    input_path: Path,
    cases_path: Path,
    output_path: Path,
    workers: int = 32,
    timeout_s: float = 10.0,
    mem_mb: int = 1024,
    include_private: bool = False,
    save_response: bool = False,
    limit: Optional[int] = None,
    judge_error_retries: int = 2,
):
    global _CASES_IDX, _CFG, _MAX_JUDGE_ERROR_RETRIES

    _CFG = judge_cp.JudgeConfig(
        float_tol=1e-6,
        include_private=include_private,
        default_timeout_s=timeout_s,
        default_mem_mb=mem_mb,
    )

    _CASES_IDX = load_cases_index(cases_path)
    _MAX_JUDGE_ERROR_RETRIES = max(0, int(judge_error_retries))
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
    n_total = len(records)

    # 打开输出文件（追加模式）
    fout = output_path.open("a", encoding="utf-8")

    print(f"\n开始 judge: {n_total} items, {workers} workers, judge_error_retries={_MAX_JUDGE_ERROR_RETRIES}")
    print(f"{'='*60}")

    # 使用 multiprocessing.Pool + fork
    # fork 后子进程共享 _CASES_IDX (copy-on-write), 内存高效
    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(processes=workers, maxtasksperchild=200) as pool:
            for result in pool.imap_unordered(_worker_judge_with_retry, records, chunksize=1):
                # 处理结果
                if not save_response and result.get("response"):
                    result["response_preview"] = (result["response"] or "")[:200]
                    del result["response"]

                line = json.dumps(result, ensure_ascii=False) + "\n"
                fout.write(line)
                status_counter[result["judge"]["status"]] += 1
                n_done += 1

                # 每 10 条 flush 一次, 每 50 条打印进度
                if n_done % 10 == 0:
                    fout.flush()

                if n_done % 50 == 0 or n_done == n_total:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    remaining = n_total - n_done
                    eta = remaining / rate if rate > 0 else 0
                    eta_h = eta / 3600
                    pct = n_done / n_total * 100

                    # 状态摘要
                    top_statuses = status_counter.most_common(5)
                    status_str = ", ".join(f"{s}:{c}" for s, c in top_statuses)

                    print(
                        f"  [{n_done:>6d}/{n_total}] {pct:5.1f}%  "
                        f"{rate:5.1f} items/s  "
                        f"ETA {eta:.0f}s ({eta_h:.1f}h)  "
                        f"| {status_str}"
                    )
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n中断! 正在保存已完成的结果...")
        fout.flush()
        fout.close()
        print(f"已保存 {n_done} 条结果到 {output_path}")
        print("下次运行可从断点继续。")
        return

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
    ap = argparse.ArgumentParser(description="高性能多进程 Judge Pipeline")
    ap.add_argument("--input", required=True, help="Gemini output JSON array file (pro or flash)")
    ap.add_argument("--cases", required=True, help="Cases JSONL file")
    ap.add_argument("--output", required=True, help="Output JSONL file for judge results")
    ap.add_argument("--workers", type=int, default=32, help="Number of worker processes (default: 32)")
    ap.add_argument("--timeout", type=float, default=10.0, help="Timeout per test case in seconds (default: 10)")
    ap.add_argument("--mem_mb", type=int, default=1024, help="Memory limit per run in MB (default: 1024)")
    ap.add_argument("--include_private", action="store_true", help="Include private test cases")
    ap.add_argument("--save_response", action="store_true", help="Save full standardized response")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of items to judge")
    ap.add_argument("--judge_error_retries", type=int, default=2, help="Auto retry count when status is JUDGE_ERROR")
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
        judge_error_retries=args.judge_error_retries,
    )


if __name__ == "__main__":
    main()
