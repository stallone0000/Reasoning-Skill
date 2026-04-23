#!/usr/bin/env python3
"""Rebuild a report JSON from existing gens/judge artifacts.

This is for repair/repro work when inference and judge are already complete,
but the saved report is stale or incorrect.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path


def load_jsonl_by_key(path: Path) -> dict[str, dict]:
    data: dict[str, dict] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("unique_key")
            if key:
                data[key] = rec
    return data


def get_completion_tokens(rec: dict) -> int | None:
    usage = rec.get("usage") or {}
    value = usage.get("completion_tokens")
    if value is not None:
        return int(value)

    value = rec.get("completion_tokens")
    if value is not None:
        return int(value)

    reasoning = rec.get("reasoning_tokens")
    output = rec.get("output_tokens")
    if reasoning is not None and output is not None:
        return int(reasoning) + int(output)
    return None


def get_total_tokens(rec: dict) -> int | None:
    usage = rec.get("usage") or {}
    value = usage.get("total_tokens")
    if value is not None:
        return int(value)

    value = rec.get("total_tokens")
    if value is not None:
        return int(value)

    prompt = rec.get("prompt_tokens")
    completion = get_completion_tokens(rec)
    if prompt is not None and completion is not None:
        return int(prompt) + int(completion)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild report JSON from gens/judge JSONL")
    parser.add_argument("--gens", required=True, help="Path to gens JSONL")
    parser.add_argument("--judge", required=True, help="Path to judge JSONL")
    parser.add_argument("--report", required=True, help="Path to output report JSON")
    parser.add_argument("--experiment", default=None, help="Override experiment name")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument(
        "--backup-suffix",
        default=None,
        help="If set and report exists, copy old report to <report><suffix> before overwrite",
    )
    args = parser.parse_args()

    gens_path = Path(args.gens).resolve()
    judge_path = Path(args.judge).resolve()
    report_path = Path(args.report).resolve()

    all_gens = load_jsonl_by_key(gens_path)
    ok_gens = {k: v for k, v in all_gens.items() if v.get("status") == "OK"}
    judge_results = load_jsonl_by_key(judge_path)

    missing_judge = sorted(set(ok_gens) - set(judge_results))
    if missing_judge:
        print(
            f"ERROR: judge incomplete: missing {len(missing_judge)} keys "
            f"(have {len(judge_results)}, need {len(ok_gens)})",
            file=sys.stderr,
        )
        return 1

    status_counter: Counter[str] = Counter()
    completion_tokens: list[int] = []
    total_tokens: list[int] = []

    first_gen = next(iter(all_gens.values()), {})
    experiment = args.experiment or first_gen.get("experiment") or "unknown"
    model = args.model or first_gen.get("model") or "unknown"

    for key, gen in ok_gens.items():
        jr = judge_results.get(key, {})
        status_counter[jr.get("status", "NOT_JUDGED")] += 1

        ct = get_completion_tokens(gen)
        tt = get_total_tokens(gen)
        if ct is not None:
            completion_tokens.append(ct)
        if tt is not None:
            total_tokens.append(tt)

    n_total = len(ok_gens)
    n_ac = status_counter.get("AC", 0)
    ac_rate = round(n_ac / n_total, 4) if n_total else 0.0
    avg_ct = round(statistics.mean(completion_tokens)) if completion_tokens else 0
    avg_tt = round(statistics.mean(total_tokens)) if total_tokens else 0
    median_ct = round(statistics.median(completion_tokens)) if completion_tokens else 0
    tokens_per_ac = round(sum(total_tokens) / n_ac) if n_ac > 0 and total_tokens else None

    report = {
        "experiment": experiment,
        "model": model,
        "n_problems": n_total,
        "n_ac": n_ac,
        "ac_rate": ac_rate,
        "status_distribution": dict(status_counter),
        "avg_completion_tokens": avg_ct,
        "median_completion_tokens": median_ct,
        "avg_total_tokens": avg_tt,
        "tokens_per_ac": tokens_per_ac,
        "n_api_fail": len(all_gens) - len(ok_gens),
    }

    if report_path.exists() and args.backup_suffix:
        backup_path = report_path.with_name(report_path.name + args.backup_suffix)
        shutil.copy2(report_path, backup_path)
        print(f"Backed up existing report to {backup_path}")

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        json.dumps(
            {
                "report": str(report_path),
                "n_problems": n_total,
                "n_ac": n_ac,
                "ac_rate": ac_rate,
                "status_distribution": dict(status_counter),
                "avg_total_tokens": avg_tt,
                "tokens_per_ac": tokens_per_ac,
                "n_api_fail": len(all_gens) - len(ok_gens),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
