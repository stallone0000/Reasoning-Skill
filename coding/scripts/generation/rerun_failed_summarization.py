#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rerun only failed/missing summarize tasks using existing outputs in output_dir.

This script reuses:
- summarize_tasks.jsonl
- task_results.jsonl
- cards_raw.jsonl

It does NOT reload flash/doubao source datasets, so reruns are much faster.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set

from run_flash_experience_summarization import (
    build_prompt,
    extract_json_obj,
    iter_jsonl,
    load_case_inputs_for_needed_indices,
    run_task_with_retries,
    summarize_judge_for_prompt,
    validate_cards_obj,
    write_jsonl,
)

INVALID_ESCAPE_RE = re.compile(r'(?<!\\)\\(?!["\\/bfnrtu])')


def sanitize_invalid_backslashes(text: str) -> str:
    # Convert invalid JSON escapes such as "\in" into "\\in".
    return INVALID_ESCAPE_RE.sub(r"\\\\", text)


def _to_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        parts = [p.strip(" \t-") for p in re.split(r"[;\n]+", s) if p.strip()]
        return parts or [s]
    return []


def tolerant_validate_cards_obj(obj) -> tuple[Optional[List[dict]], Optional[str]]:
    if isinstance(obj, dict):
        cards = obj.get("cards")
    elif isinstance(obj, list):
        cards = obj
    else:
        return None, "root_not_object_or_list"

    if not isinstance(cards, list) or not cards:
        return None, "cards_missing_or_empty"

    out: List[dict] = []
    for i, card in enumerate(cards):
        if not isinstance(card, dict):
            return None, f"card_{i}_not_object"

        row = {}
        for k in ("trigger", "tags", "do", "avoid", "check"):
            vals = _to_list(card.get(k))
            if not vals:
                return None, f"card_{i}_{k}_empty_after_coerce"
            row[k] = vals

        row["type"] = str(card.get("type") or "").strip().lower() or "unknown"
        row["detours"] = _to_list(card.get("detours"))
        row["risk"] = str(card.get("risk") or "").strip() or "Potential overfitting risk."
        row["complexity"] = str(card.get("complexity") or "").strip()
        out.append(row)

    return out, None


def recover_cards_from_raw(raw: str) -> tuple[Optional[List[dict]], Optional[str]]:
    raw = (raw or "").strip()
    if not raw:
        return None, "empty_raw_content"

    attempts = [raw]
    patched = sanitize_invalid_backslashes(raw)
    if patched != raw:
        attempts.append(patched)

    last_err = "no_json_object"
    for cand in attempts:
        obj, parse_err = extract_json_obj(cand)
        if parse_err or obj is None:
            last_err = parse_err or "no_json_object"
            continue

        cards, valid_err = validate_cards_obj(obj)
        if not valid_err and cards:
            return cards, None
        last_err = valid_err or last_err

        cards2, valid_err2 = tolerant_validate_cards_obj(obj)
        if not valid_err2 and cards2:
            return cards2, None
        last_err = valid_err2 or last_err

    return None, last_err


def clip_text(text: str, max_chars: int) -> tuple[str, bool]:
    s = str(text or "")
    if max_chars <= 0 or len(s) <= max_chars:
        return s, False
    return s[:max_chars] + "\n...[TRUNCATED_FOR_RERUN]", True


def trim_task_for_prompt(task: dict, max_question_chars: int, max_chain_chars: int) -> tuple[dict, bool]:
    t = dict(task)
    truncated = False

    q, q_tr = clip_text(t.get("question") or "", max_question_chars)
    t["question"] = q
    truncated = truncated or q_tr

    for k in ("thinking_chain", "success_chain", "failed_chain"):
        if k in t:
            v, v_tr = clip_text(t.get(k) or "", max_chain_chars)
            t[k] = v
            truncated = truncated or v_tr
    return t, truncated


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun failed summarize tasks from existing output_dir")
    parser.add_argument("--output_dir", required=True, help="Existing summarize output directory")
    parser.add_argument("--model", default="cloudsway/gemini-3-flash-preview")
    parser.add_argument("--api_url", default="http://api.360.cn/v1/chat/completions")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--cases", default="nemotron_cp_cases_34799_v1.jsonl")
    parser.add_argument("--workers", type=int, default=120)
    parser.add_argument("--timeout_s", type=int, default=120)
    parser.add_argument("--max_tokens", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--backoff_base_s", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_case_chars", type=int, default=900)
    parser.add_argument("--max_question_chars", type=int, default=1800)
    parser.add_argument("--max_chain_chars", type=int, default=12000)
    parser.add_argument("--use_env_proxy", action="store_true", help="Use proxy from env (default: disabled)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    output_dir = (base / args.output_dir).resolve()
    cases_path = (base / args.cases).resolve()

    tasks_path = output_dir / "summarize_tasks.jsonl"
    task_results_path = output_dir / "task_results.jsonl"
    cards_raw_path = output_dir / "cards_raw.jsonl"
    problem_to_cards_path = output_dir / "problem_to_cards.jsonl"
    rerun_report_path = output_dir / "rerun_failed_report.json"

    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing tasks file: {tasks_path}")
    if not task_results_path.exists():
        raise FileNotFoundError(f"Missing task results file: {task_results_path}")
    if not cards_raw_path.exists():
        cards_raw_path.touch()

    print(f"Loading tasks from {tasks_path} ...")
    tasks: Dict[str, dict] = {}
    for row in iter_jsonl(tasks_path):
        tid = row.get("task_id")
        if tid:
            tasks[tid] = row
    print(f"  tasks total={len(tasks)}")

    done_ok: Set[str] = set()
    latest_by_task: Dict[str, dict] = {}
    for row in iter_jsonl(task_results_path):
        tid = row.get("task_id")
        if not tid:
            continue
        latest_by_task[tid] = row
        if row.get("ok"):
            done_ok.add(tid)

    pending_ids = [tid for tid in tasks.keys() if tid not in done_ok]
    pending_tasks = [tasks[tid] for tid in pending_ids]
    print(f"  done_ok={len(done_ok)}, pending={len(pending_tasks)}")
    if not pending_tasks:
        print("No pending task.")
        return

    needed_indices: Dict[str, Set[int]] = defaultdict(set)
    for task in pending_tasks:
        if task.get("task_type") not in ("failure", "contrast"):
            continue
        key = task["unique_key"]
        for cr in task.get("selected_fails", []):
            try:
                needed_indices[key].add(int(cr.get("case")))
            except Exception:
                pass
    print(f"Loading case inputs for {len(needed_indices)} keys ...")
    case_inputs = load_case_inputs_for_needed_indices(cases_path, needed_indices, include_private=False)
    print(f"  loaded case inputs for {len(case_inputs)} keys")

    prompt_map: Dict[str, str] = {}
    truncated_task_count = 0
    for task in pending_tasks:
        prompt_task, was_truncated = trim_task_for_prompt(task, args.max_question_chars, args.max_chain_chars)
        if was_truncated:
            truncated_task_count += 1
        judge_info = None
        if prompt_task["task_type"] in ("failure", "contrast"):
            judge_info = summarize_judge_for_prompt(
                prompt_task["failed_judge"],
                prompt_task.get("selected_fails", []),
                case_inputs.get(prompt_task["unique_key"], {}),
                max_case_chars=args.max_case_chars,
            )
        prompt_map[prompt_task["task_id"]] = build_prompt(prompt_task, judge_info=judge_info)
    print(
        f"  prompt_trim: max_question_chars={args.max_question_chars}, "
        f"max_chain_chars={args.max_chain_chars}, truncated_tasks={truncated_task_count}"
    )

    write_lock = threading.Lock()
    stats = Counter()
    t0 = time.time()

    def worker(task: dict) -> dict:
        result = run_task_with_retries(
            task=task,
            prompt=prompt_map[task["task_id"]],
            url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            use_env_proxy=args.use_env_proxy,
            max_retries=args.max_retries,
            backoff_base_s=args.backoff_base_s,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        recovered_from = None
        recovery_error = None
        if not result.get("ok"):
            recovered_cards, recovery_error = recover_cards_from_raw(result.get("raw_content") or "")
            if recovered_cards:
                recovered_from = result.get("error")
                result = {
                    "ok": True,
                    "attempts": int(result.get("attempts") or 0),
                    "usage": result.get("usage") or {},
                    "raw_content": result.get("raw_content") or "",
                    "cards": recovered_cards,
                    "finish_reason": result.get("finish_reason"),
                    "error": None,
                }
        return {
            "task": task,
            "result": result,
            "recovered_from": recovered_from,
            "recovery_error": recovery_error,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, task): task for task in pending_tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            payload = fut.result()
            task = payload["task"]
            result = payload["result"]
            usage = result.get("usage") or {}

            row = {
                "task_id": task["task_id"],
                "unique_key": task["unique_key"],
                "task_type": task["task_type"],
                "category": task["category"],
                "source_model": task["source_model"],
                "ok": bool(result.get("ok")),
                "attempts": int(result.get("attempts") or 0),
                "error": result.get("error"),
                "recovered_from": payload.get("recovered_from"),
                "recovery_error": payload.get("recovery_error"),
                "finish_reason": result.get("finish_reason"),
                "raw_content_head": (result.get("raw_content") or "")[:2000],
                "usage": usage,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            card_rows: List[dict] = []
            if result.get("ok"):
                stats["ok"] += 1
                if payload.get("recovered_from"):
                    stats["ok_recovered"] += 1
                cards = result.get("cards") or []
                for idx, card in enumerate(cards):
                    cid = f"{task['task_id']}_rerun_c{idx}_{int(time.time())}"
                    card_rows.append(
                        {
                            "card_id": cid,
                            "task_id": task["task_id"],
                            "problem_id": task["unique_key"],
                            "task_type": task["task_type"],
                            "category": task["category"],
                            "source_model": task["source_model"],
                            "source_status": {
                                "flash": task.get("flash_status"),
                                "doubao": task.get("doubao_status"),
                                "failed_model": task.get("failed_model"),
                            },
                            **card,
                        }
                    )
                stats["cards"] += len(card_rows)
            else:
                stats["fail"] += 1
                stats[f"err::{result.get('error')}"] += 1

            with write_lock:
                write_jsonl(task_results_path, [row], mode="a")
                if card_rows:
                    write_jsonl(cards_raw_path, card_rows, mode="a")

            if i % 20 == 0 or i == len(pending_tasks):
                elapsed = time.time() - t0
                speed = i / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{i}/{len(pending_tasks)}] ok={stats['ok']} recovered={stats['ok_recovered']} fail={stats['fail']} "
                    f"cards={stats['cards']} speed={speed:.2f} task/s"
                )

    # rebuild problem_to_cards
    p2c: Dict[str, List[str]] = defaultdict(list)
    card_count = 0
    for row in iter_jsonl(cards_raw_path):
        pid = row.get("problem_id")
        cid = row.get("card_id")
        if not pid or not cid:
            continue
        p2c[pid].append(cid)
        card_count += 1
    with problem_to_cards_path.open("w", encoding="utf-8") as f:
        for pid, cids in sorted(p2c.items()):
            f.write(json.dumps({"problem_id": pid, "card_ids": cids}, ensure_ascii=False) + "\n")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pending_input": len(pending_tasks),
        "rerun_ok": stats["ok"],
        "rerun_fail": stats["fail"],
        "cards_added": stats["cards"],
        "cards_total": card_count,
        "error_counts": {k[5:]: v for k, v in stats.items() if k.startswith("err::")},
        "config": {
            "workers": args.workers,
            "timeout_s": args.timeout_s,
            "max_tokens": args.max_tokens,
            "max_retries": args.max_retries,
            "backoff_base_s": args.backoff_base_s,
            "use_env_proxy": args.use_env_proxy,
        },
    }
    with rerun_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Done rerun.")
    print(f"  rerun report: {rerun_report_path}")


if __name__ == "__main__":
    main()
