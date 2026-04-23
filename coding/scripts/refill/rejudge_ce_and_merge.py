# -*- coding: utf-8 -*-
"""
Rejudge CE rows from an existing judge result file and merge back.

Use case:
  python3 rejudge_ce_and_merge.py \
    --result judge_results_pro_rerun.jsonl \
    --model_json nemotron_cp_unique_questions_34729_withimages_pro.json \
    --cases nemotron_cp_cases_34799_v1.jsonl \
    --output judge_results_pro_final_fixed.jsonl
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import judge_cp

THOUGHT_END_RE = re.compile(
    r'<\s*/\s*(?:thought|think|analysis|reasoning)\s*>'
    r'|<\s*(?:thought|think)\s*/\s*>'
    r'|<\s*(?:thought|think)\s+/\s*>'
    r'|<\s*end_?(?:thought|think)s?\s*>'
    r'|<\s*(?:thought|think)_?ends?\s*>'
    r'|<\s*(?:thought|think)_process_end\s*>'
    r'|<\s*/\s*redacted_reasoning\s*>',
    re.IGNORECASE,
)

OUTPUT_KEYS = ["gemini_outputs", "correct_gemini_outputs"]

_CASES_IDX: Dict[str, dict] = {}
_REC_BY_KEY: Dict[str, dict] = {}
_CFG = judge_cp.JudgeConfig()


def iter_json_array(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        ch = f.read(1)
        while ch and ch.isspace():
            ch = f.read(1)
        if ch != "[":
            raise ValueError(f"{path} is not a JSON array file")

        started = False
        depth = 0
        in_str = False
        esc = False
        buf = ""

        while True:
            ch = f.read(1)
            if not ch:
                break
            if not started:
                if ch.isspace() or ch == ",":
                    continue
                if ch == "]":
                    break
                if ch != "{":
                    continue
                started = True
                depth = 1
                in_str = False
                esc = False
                buf = "{"
                continue

            buf += ch
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield json.loads(buf)
                    started = False
                    buf = ""


def first_output_text(rec: dict) -> Optional[str]:
    for k in OUTPUT_KEYS:
        v = rec.get(k)
        if isinstance(v, list) and v and isinstance(v[0], str) and v[0].strip():
            return v[0]
    return None


def split_think_answer(text: str) -> Tuple[bool, Optional[str], str]:
    if not text:
        return False, None, ""
    matches = list(THOUGHT_END_RE.finditer(text))
    if not matches:
        return False, None, text
    best = max(matches, key=lambda m: m.start())
    answer = text[best.end():].strip()
    if not answer:
        return False, best.group(0), text
    return True, best.group(0), answer


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


def _worker_rejudge_ce(uk: str) -> Tuple[str, dict]:
    rec = _REC_BY_KEY.get(uk)
    prob = _CASES_IDX.get(uk)

    if not rec or not prob:
        return uk, {
            "status": "JUDGE_ERROR",
            "lang": None,
            "passed": 0,
            "total_tests": 0,
            "case_results": [],
            "detail": "missing rec or case",
        }

    text = first_output_text(rec)
    if not text:
        return uk, {
            "status": "NO_OUTPUT",
            "lang": None,
            "passed": 0,
            "total_tests": 0,
            "case_results": [],
            "detail": None,
        }

    split_ok, end_tok, answer_part = split_think_answer(text)
    if not split_ok:
        hint, code = judge_cp.extract_code(text)
        if hint is not None and code.strip():
            answer_part = text

    try:
        raw = judge_cp.judge_one(prob, answer_part, _CFG)
        return uk, flatten_judge_result(raw)
    except Exception as e:
        return uk, {
            "status": "JUDGE_ERROR",
            "lang": None,
            "passed": 0,
            "total_tests": 0,
            "case_results": [],
            "detail": str(e)[:2000],
        }


def load_cases_idx(path: Path) -> Dict[str, dict]:
    idx = {}
    for row in judge_cp.iter_jsonl(str(path)):
        uk = row.get("unique_key")
        if uk:
            idx[uk] = row
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--model_json", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--mem_mb", type=int, default=1024)
    ap.add_argument("--include_private", action="store_true")
    args = ap.parse_args()

    result_path = Path(args.result)
    model_path = Path(args.model_json)
    cases_path = Path(args.cases)
    output_path = Path(args.output)

    # load result rows + ce keys
    rows = []
    ce_keys = set()
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            rows.append(d)
            if d.get("judge", {}).get("status") == "CE":
                uk = d.get("unique_key")
                if uk:
                    ce_keys.add(uk)

    print(f"loaded rows={len(rows)}, CE keys={len(ce_keys)}")

    global _CASES_IDX, _REC_BY_KEY, _CFG
    _CFG = judge_cp.JudgeConfig(
        float_tol=1e-6,
        include_private=args.include_private,
        default_timeout_s=args.timeout,
        default_mem_mb=args.mem_mb,
    )
    _CASES_IDX = load_cases_idx(cases_path)

    _REC_BY_KEY = {}
    for rec in iter_json_array(model_path):
        uk = rec.get("unique_key")
        if uk in ce_keys:
            _REC_BY_KEY[uk] = rec
            if len(_REC_BY_KEY) >= len(ce_keys):
                break

    print(f"loaded model records for CE keys: {len(_REC_BY_KEY)}")

    # rejudge CE keys
    rejudged = {}
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers, maxtasksperchild=200) as pool:
        done = 0
        total = len(ce_keys)
        stat = Counter()
        for uk, j in pool.imap_unordered(_worker_rejudge_ce, sorted(ce_keys), chunksize=1):
            rejudged[uk] = j
            stat[j.get("status", "UNKNOWN")] += 1
            done += 1
            if done % 100 == 0 or done == total:
                dt = time.time() - t0
                rate = done / dt if dt > 0 else 0
                print(f"  [{done}/{total}] {rate:.1f} items/s | {stat.most_common(5)}")

    # merge
    before = Counter(d.get("judge", {}).get("status", "UNKNOWN") for d in rows)
    merged = []
    for d in rows:
        uk = d.get("unique_key")
        if uk in rejudged:
            d = dict(d)
            d["judge"] = rejudged[uk]
        merged.append(d)

    after = Counter(d.get("judge", {}).get("status", "UNKNOWN") for d in merged)

    with output_path.open("w", encoding="utf-8") as f:
        for d in merged:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    n = len(merged)
    print("\n=== summary ===")
    print(f"rows={n}")
    print(f"AC before={before['AC']} ({before['AC']/n*100:.2f}%)")
    print(f"AC after ={after['AC']} ({after['AC']/n*100:.2f}%)")
    print(f"CE before={before['CE']} -> after={after['CE']}")


if __name__ == "__main__":
    main()
