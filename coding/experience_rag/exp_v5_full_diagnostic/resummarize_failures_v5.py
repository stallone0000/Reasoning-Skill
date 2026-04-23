#!/usr/bin/env python3
"""
resummarize_failures_v5.py — "Full Diagnostic" failure re-summarization.

Two-tier approach:
  Tier A (partial-pass): Code + ALL failing test cases → edge_fix cards
  Tier B (zero-pass):    Code + 8 selected cases + brief chain → wrong_approach cards

Key differences from V3:
  1. Extracts CODE from thinking chain (not just narrative)
  2. Shows ALL failing test cases with full I/O (not 4-8 truncated)
  3. Removes thinking chain for Tier A (uses code + tests instead)
  4. Shorter, more concrete cards (~150 chars inject vs ~770)

Usage:
    python resummarize_failures_v5.py [--workers 64] [--pilot 100] [--output_dir .]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

RM_ROOT = None
for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "LATEST_STATUS.md").exists() and (_candidate / "README.md").exists():
        RM_ROOT = _candidate
        break
if RM_ROOT is None:
    raise RuntimeError("Cannot locate reasoning_memory root")
if str(RM_ROOT) not in sys.path:
    sys.path.insert(0, str(RM_ROOT))

from rm_runtime import clear_proxy_env, resolve_api_key

# ─── paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # experience_rag/
SUMMARY_DIR = ROOT.parent / "experiments" / "experience_summary" / "full_v2_nomax"
CASES_PATH = ROOT.parent / "data" / "questions" / "nemotron_cp_cases_34799_v1.jsonl"

# ─── API config ─────────────────────────────────────────────────────
DEFAULT_MODEL = "cloudsway/gemini-3-flash-preview"
DEFAULT_URL = "http://api.360.cn/v1/chat/completions"
NO_ENV_SESSION = requests.Session()
NO_ENV_SESSION.trust_env = False

# ─── Prompts ─────────────────────────────────────────────────────────

TIER_A_PROMPT = """You are an expert competitive programmer performing EDGE CASE FORENSICS.

This code passes {pass_pct}% of test cases but fails on specific inputs. Identify the EXACT failure pattern and produce a concise, actionable card.

CRITICAL RULES:
A) Focus on the GAP between what the code does and what the failing cases expect.
B) "trigger" must be PROBLEM-LEVEL patterns (e.g., "output requires specific ordering", "modular arithmetic with large N"), NOT judge outputs like "WA" or "Expected: X Got: Y".
C) "edge_pattern" must describe the SPECIFIC input characteristic that breaks this code (e.g., "all elements equal", "N=1", "output order differs from computation order").
D) "fix_hint" must be a ONE-LINE actionable instruction (e.g., "sort output by second element desc", "use int64 for intermediate products").
E) Keep the card EXTREMELY concise. Total text under 150 words.
F) Generalize: no problem-specific variable names, constants, or story details.

[Problem]
{question}

[Submitted Code]
```{lang}
{code}
```

[Test Results: {passed}/{total} passed ({pass_pct}%)]
Error distribution: {error_dist}

[All Failing Cases ({n_fail} cases)]
{failing_cases_block}

Output ONLY valid JSON (no markdown, no explanation):
{{"cards": [{{
  "type": "edge_fix",
  "trigger": ["1-2 problem-level pattern strings"],
  "tags": ["1-4 algorithm/topic tags"],
  "edge_pattern": "specific input pattern that breaks the code",
  "fix_hint": "one-line actionable fix",
  "do": ["1-2 specific diagnostic actions"],
  "avoid": ["1 specific wrong assumption from the code"],
  "check": ["1 pre-submit verification step"]
}}]}}
"""

TIER_B_PROMPT = """You are an expert competitive programmer diagnosing a FUNDAMENTALLY WRONG solution.

This code fails ALL {total} test cases. Identify the wrong algorithm/approach and the correct direction.

CRITICAL RULES:
A) "trigger" must be PROBLEM-LEVEL patterns that signal this type of problem.
B) "wrong_approach" must name the SPECIFIC algorithm/technique tried (e.g., "greedy pair matching without considering global constraints", "BFS without tracking visited states").
C) "correct_direction" must name the CORRECT approach family. Only state what you can confidently infer from the test case I/O patterns (e.g., "dynamic programming on intervals", "model as bipartite matching"). If uncertain, say "verify with small cases before committing to approach".
D) Keep the card under 120 words total.
E) Generalize: no problem-specific variable names, constants, or story details.

[Problem]
{question}

[Submitted Code]
```{lang}
{code}
```

[Test Results: 0/{total} passed]
Error distribution: {error_dist}

[Sample Failing Cases ({n_shown} of {total})]
{failing_cases_block}

[Approach Summary]
{approach_summary}

Output ONLY valid JSON (no markdown, no explanation):
{{"cards": [{{
  "type": "wrong_approach",
  "trigger": ["1-2 problem-level pattern strings"],
  "tags": ["1-4 algorithm/topic tags"],
  "wrong_approach": "specific wrong algorithm/technique tried",
  "correct_direction": "correct approach family or diagnostic check",
  "do": ["1-2 specific actions"],
  "avoid": ["1 specific anti-pattern"],
  "check": ["1 diagnostic question to ask before committing"]
}}]}}
"""

# ─── code extraction (adapted from judge_cp.py) ────────────────────

FENCE_RE = re.compile(
    r"```[ \t]*(\w+)?[ \t]*\n(.*?)```",
    re.DOTALL,
)
CODE_BLOCK_RE = re.compile(
    r"<code_block>(.*?)</code_block>",
    re.DOTALL,
)
LANG_ALIASES = {
    "py": "python", "python3": "python", "py3": "python",
    "c++": "cpp", "c": "cpp", "cxx": "cpp", "cplusplus": "cpp",
}
UNSUPPORTED_LANGS = {"rust", "rs", "java", "javascript", "js", "go", "php", "ruby", "scala", "kotlin", "swift"}


def extract_code_from_text(text: str) -> Tuple[str, str]:
    """Extract code from LLM output. Returns (lang, code)."""
    blocks = []

    for lang, code in FENCE_RE.findall(text):
        lang = (lang or "").strip().lower()
        if lang in UNSUPPORTED_LANGS:
            continue
        lang = LANG_ALIASES.get(lang, lang) if lang else None
        c = (code or "").strip("\n")
        if c.strip():
            blocks.append((lang if lang in ("python", "cpp") else None, c))

    for m in CODE_BLOCK_RE.finditer(text):
        code = m.group(1).strip()
        if code.strip():
            lang_hint = None
            if "#include" in code or "using namespace std" in code:
                lang_hint = "cpp"
            elif re.search(r"^\s*def\s+\w+\(", code, re.M):
                lang_hint = "python"
            blocks.append((lang_hint, code))

    if blocks:
        def score(b):
            lang, code = b
            s = len(code)
            if lang in ("python", "cpp"):
                s += 10_000_000
            return s
        lang, code = max(blocks, key=score)
        return lang or "python", code

    return "python", text.strip()[:3000]  # fallback


# ─── reuse imports from main script ─────────────────────────────────
sys.path.insert(0, str(ROOT.parent / "scripts" / "generation"))
from run_flash_experience_summarization import (
    extract_json_obj,
    iter_jsonl,
    flatten_tests_for_inputs,
    extract_input_from_test,
    select_diagnostic_failures,
    _shorten,
)


def validate_v5_cards(obj: Any) -> Tuple[Optional[List[dict]], Optional[str]]:
    """Validate V5 card format (edge_fix or wrong_approach)."""
    if isinstance(obj, dict):
        cards = obj.get("cards")
    elif isinstance(obj, list):
        cards = obj
    else:
        return None, "root_not_object_or_list"
    if not isinstance(cards, list) or not cards:
        return None, "cards_missing_or_empty"

    def norm_list(val):
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return []
            parts = [p.strip(" \t-") for p in re.split(r"[;\n]+", s) if p.strip()]
            return parts or [s]
        return []

    normalized = []
    for i, card in enumerate(cards):
        if not isinstance(card, dict):
            return None, f"card_{i}_not_object"

        ctype = str(card.get("type", "")).strip().lower()
        if ctype not in ("edge_fix", "wrong_approach"):
            ctype = "edge_fix"  # default

        # Required fields for both types
        trigger = norm_list(card.get("trigger"))
        tags = norm_list(card.get("tags"))
        do = norm_list(card.get("do"))
        avoid = norm_list(card.get("avoid"))
        check = norm_list(card.get("check"))

        if not trigger:
            return None, f"card_{i}_trigger_empty"
        if not tags:
            return None, f"card_{i}_tags_empty"

        out = {
            "type": ctype,
            "trigger": trigger[:4],
            "tags": tags[:6],
            "do": do[:3],
            "avoid": avoid[:3],
            "check": check[:3],
        }

        # Type-specific fields
        if ctype == "edge_fix":
            out["edge_pattern"] = str(card.get("edge_pattern", "")).strip()
            out["fix_hint"] = str(card.get("fix_hint", "")).strip()
        else:
            out["wrong_approach"] = str(card.get("wrong_approach", "")).strip()
            out["correct_direction"] = str(card.get("correct_direction", "")).strip()

        normalized.append(out)

    return normalized, None


# ─── test case loading ──────────────────────────────────────────────

def load_all_case_io(cases_path: Path, needed_keys: Set[str]) -> Dict[str, List[dict]]:
    """Load ALL test case inputs and expected outputs for needed problems.
    Returns {problem_key: [{input: str, output: str, idx: int}, ...]}
    """
    out = {}
    for row in iter_jsonl(cases_path):
        key = row.get("unique_key")
        if key not in needed_keys:
            continue
        cases = row.get("cases") or {}
        if not isinstance(cases, dict):
            continue
        tests = flatten_tests_for_inputs(cases, include_private=False)
        test_list = []
        for idx, t in enumerate(tests):
            if not isinstance(t, dict):
                continue
            inp = extract_input_from_test(t)
            out_text = t.get("output", t.get("expected_output", ""))
            if isinstance(out_text, list):
                out_text = "\n".join(str(x) for x in out_text)
            test_list.append({
                "idx": idx,
                "input": str(inp) if inp else "",
                "output": str(out_text) if out_text else "",
            })
        out[key] = test_list
        if len(out) >= len(needed_keys):
            break
    return out


def build_failing_cases_block(
    judge: dict,
    all_case_io: List[dict],
    max_cases: int = 30,
    max_chars_per_field: int = 2000,
) -> str:
    """Build text block of ALL failing test cases with full I/O."""
    case_results = judge.get("case_results", [])
    fails = [cr for cr in case_results if isinstance(cr, dict)
             and (cr.get("status") or "").upper() != "AC"]

    # Build index from case_io for lookup
    io_by_idx = {tc["idx"]: tc for tc in all_case_io}

    lines = []
    for cr in fails[:max_cases]:
        idx = cr.get("case", -1)
        status = (cr.get("status") or "UNKNOWN").upper()
        lines.append(f"--- Case {idx}: {status} ---")

        # Input from full test case data
        tc = io_by_idx.get(idx, {})
        inp = tc.get("input", "")
        if inp:
            lines.append(f"Input:\n{_shorten(inp, max_chars_per_field)}")

        if status in ("WA", "WA_CHECKER"):
            # Expected from test case data (more reliable) or from judge
            exp = tc.get("output", "") or str(cr.get("exp", ""))
            got = str(cr.get("got", ""))
            if exp:
                lines.append(f"Expected:\n{_shorten(exp, max_chars_per_field)}")
            if got:
                lines.append(f"Got:\n{_shorten(got, max_chars_per_field)}")
        elif status == "TLE":
            lines.append("(Time Limit Exceeded)")
        elif status in ("RE", "RUNTIME_ERROR"):
            detail = cr.get("detail") or cr.get("stderr") or ""
            if detail:
                lines.append(f"Error:\n{_shorten(str(detail), max_chars_per_field)}")
        lines.append("")

    return "\n".join(lines)


def build_selected_cases_block(
    judge: dict,
    all_case_io: List[dict],
    max_cases: int = 8,
    max_chars_per_field: int = 1500,
    seed: int = 42,
) -> str:
    """Build block with selected diagnostic cases (for Tier B)."""
    case_results = judge.get("case_results", [])
    selected = select_diagnostic_failures(case_results, max_cases, seed)
    io_by_idx = {tc["idx"]: tc for tc in all_case_io}

    lines = []
    for cr in selected:
        idx = cr.get("case", -1)
        status = (cr.get("status") or "UNKNOWN").upper()
        lines.append(f"--- Case {idx}: {status} ---")

        tc = io_by_idx.get(idx, {})
        inp = tc.get("input", "")
        if inp:
            lines.append(f"Input:\n{_shorten(inp, max_chars_per_field)}")

        if status in ("WA", "WA_CHECKER"):
            exp = tc.get("output", "") or str(cr.get("exp", ""))
            got = str(cr.get("got", ""))
            if exp:
                lines.append(f"Expected:\n{_shorten(exp, max_chars_per_field)}")
            if got:
                lines.append(f"Got:\n{_shorten(got, max_chars_per_field)}")
        elif status == "TLE":
            lines.append("(Time Limit Exceeded)")
        elif status in ("RE", "RUNTIME_ERROR"):
            detail = cr.get("detail") or cr.get("stderr") or ""
            if detail:
                lines.append(f"Error:\n{_shorten(str(detail), max_chars_per_field)}")
        lines.append("")

    return "\n".join(lines)


# ─── API call ───────────────────────────────────────────────────────

def post_api(url, api_key, model, prompt, timeout_s=900):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "content_filter": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Host": "api.360.cn",
    }
    try:
        resp = NO_ENV_SESSION.post(url, headers=headers,
                                   data=json.dumps(payload), timeout=timeout_s)
        resp.raise_for_status()
        j = resp.json()
    except Exception as e:
        return None, f"http_error:{type(e).__name__}:{e}"
    if "error" in j:
        return None, f"api_error:{j['error']}"
    return j, None


def run_with_retries(prompt, *, url, api_key, model, max_retries=6, backoff=5):
    """Call API with retries + JSON repair on parse failure."""
    last_err = None
    current_prompt = prompt
    for attempt in range(max_retries):
        resp_json, err = post_api(url, api_key, model, current_prompt)
        if err:
            last_err = err
            if attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))
            continue

        usage = resp_json.get("usage", {})
        try:
            content = resp_json["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = f"bad_schema:{e}"
            if attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))
            continue

        obj, parse_err = extract_json_obj(content)
        if parse_err:
            last_err = parse_err
        else:
            cards, valid_err = validate_v5_cards(obj)
            if not valid_err:
                return {"ok": True, "attempts": attempt + 1, "usage": usage,
                        "raw_content": content, "cards": cards, "error": None}
            last_err = valid_err

        # repair prompt
        if attempt < max_retries - 1:
            current_prompt = (
                "Convert the text below into strict JSON only.\n"
                "Output exactly one JSON object with top-level key `cards`.\n"
                "Each card must include keys: type, trigger, tags, do, avoid, check.\n"
                "For edge_fix type: also include edge_pattern and fix_hint.\n"
                "For wrong_approach type: also include wrong_approach and correct_direction.\n"
                "No markdown, no explanations.\n\n"
                "[YOUR PREVIOUS OUTPUT]\n" + (content or "")
            )
            time.sleep(backoff * (2 ** attempt))
            continue

    return {"ok": False, "attempts": max_retries, "usage": {},
            "raw_content": None, "cards": [], "error": last_err}


# ─── main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--pilot", type=int, default=0,
                        help="Only process first N tasks (0=all)")
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).parent))
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()
    args.api_key = resolve_api_key(args.api_key, required=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cards_path = output_dir / "cards_v5_diagnostic.jsonl"
    results_path = output_dir / "task_results_v5.jsonl"
    report_path = output_dir / "resummarize_v5_report.json"

    # unset proxy
    clear_proxy_env()

    # ─── load failure tasks ─────────────────────────────────────────
    print("[1/6] Loading failure tasks ...")
    tasks_path = SUMMARY_DIR / "summarize_tasks.jsonl"
    all_tasks = list(iter_jsonl(tasks_path))
    failure_tasks = [t for t in all_tasks if t["task_type"] == "failure"]
    print(f"  Total tasks: {len(all_tasks)}, failure tasks: {len(failure_tasks)}")

    if args.pilot > 0:
        failure_tasks = failure_tasks[:args.pilot]
        print(f"  Pilot mode: using first {args.pilot} tasks")

    # ─── split into tiers ───────────────────────────────────────────
    print("[2/6] Splitting into tiers ...")
    tier_a = []  # partial pass
    tier_b = []  # zero pass
    for t in failure_tasks:
        judge = t.get("failed_judge", {})
        passed = judge.get("passed", 0)
        if passed > 0:
            tier_a.append(t)
        else:
            tier_b.append(t)
    print(f"  Tier A (partial pass): {len(tier_a)}")
    print(f"  Tier B (zero pass): {len(tier_b)}")

    # ─── load ALL test case I/O ─────────────────────────────────────
    print("[3/6] Loading all test case I/O ...")
    needed_keys = {t["unique_key"] for t in failure_tasks}
    all_case_io = load_all_case_io(CASES_PATH, needed_keys)
    print(f"  Loaded test cases for {len(all_case_io)} / {len(needed_keys)} problems")

    # ─── extract code & build prompts ───────────────────────────────
    print("[4/6] Extracting code and building prompts ...")
    prompt_map = {}
    tier_map = {}  # task_id -> "A" or "B"
    code_extracted = 0
    code_fallback = 0

    for t in failure_tasks:
        task_id = t["task_id"]
        key = t["unique_key"]
        judge = t.get("failed_judge", {})
        chain = t.get("thinking_chain", "")

        # Extract code from thinking chain
        lang, code = extract_code_from_text(chain)
        if len(code) > 10:
            code_extracted += 1
        else:
            code_fallback += 1
            code = chain[:3000] if chain else "(no code found)"
            lang = "text"

        # Get case I/O
        case_io = all_case_io.get(key, [])
        passed = judge.get("passed", 0)
        total = judge.get("total_tests", 0)

        # Error distribution
        case_results = judge.get("case_results", [])
        err_dist = Counter((cr.get("status") or "?").upper() for cr in case_results
                           if isinstance(cr, dict) and (cr.get("status") or "").upper() != "AC")

        if passed > 0:
            # Tier A: edge case forensics
            tier_map[task_id] = "A"
            pass_pct = int(passed / max(total, 1) * 100)
            n_fail = sum(1 for cr in case_results if isinstance(cr, dict)
                         and (cr.get("status") or "").upper() != "AC")

            failing_block = build_failing_cases_block(judge, case_io, max_cases=30)

            prompt = TIER_A_PROMPT.format(
                question=t["question"],
                lang=lang,
                code=code,
                passed=passed,
                total=total,
                pass_pct=pass_pct,
                error_dist=dict(err_dist),
                n_fail=n_fail,
                failing_cases_block=failing_block,
            )
        else:
            # Tier B: algorithm diagnosis
            tier_map[task_id] = "B"

            # Use selected cases (8 max)
            failing_block = build_selected_cases_block(
                judge, case_io, max_cases=8, seed=hash(key) & 0xFFFFFF
            )
            n_shown = min(8, sum(1 for cr in case_results if isinstance(cr, dict)
                                 and (cr.get("status") or "").upper() != "AC"))

            # Brief approach summary (first 2KB of chain)
            approach_summary = chain[:2000] if chain else "(no reasoning chain)"

            prompt = TIER_B_PROMPT.format(
                question=t["question"],
                lang=lang,
                code=code,
                total=total,
                error_dist=dict(err_dist),
                n_shown=n_shown,
                failing_cases_block=failing_block,
                approach_summary=approach_summary,
            )

        prompt_map[task_id] = prompt

    print(f"  Code extracted: {code_extracted}, fallback: {code_fallback}")

    # ─── check for existing checkpoint ──────────────────────────────
    done_ids = set()
    if results_path.exists():
        for rec in iter_jsonl(results_path):
            if rec.get("ok"):
                done_ids.add(rec.get("task_id"))
        print(f"  Checkpoint: {len(done_ids)} tasks already done")

    remaining = [t for t in failure_tasks if t["task_id"] not in done_ids]
    print(f"  Remaining tasks: {len(remaining)}")

    if not remaining:
        print("  All tasks done!")
    else:
        # ─── run API calls ──────────────────────────────────────────
        print(f"[5/6] Running {len(remaining)} tasks with {args.workers} workers ...")
        lock = threading.Lock()
        done_count = [len(done_ids)]
        total_count = len(failure_tasks)

        def worker(task):
            task_id = task["task_id"]
            prompt = prompt_map[task_id]
            result = run_with_retries(
                prompt, url=args.url, api_key=args.api_key, model=args.model
            )
            result["task_id"] = task_id
            result["unique_key"] = task["unique_key"]
            result["category"] = task["category"]
            result["tier"] = tier_map.get(task_id, "?")

            with lock:
                with open(results_path, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                if result["ok"]:
                    for ci, card in enumerate(result["cards"]):
                        card_rec = {
                            "card_id": f"{task_id}_v5_c{ci}",
                            "task_id": task_id,
                            "problem_id": task["unique_key"],
                            "task_type": "failure",
                            "category": task["category"],
                            "source_model": task.get("source_model", "flash"),
                            "tier": tier_map.get(task_id, "?"),
                            **card,
                        }
                        with open(cards_path, "a") as f:
                            f.write(json.dumps(card_rec, ensure_ascii=False) + "\n")

                done_count[0] += 1
                if done_count[0] % 100 == 0 or done_count[0] == total_count:
                    print(f"  Progress: {done_count[0]}/{total_count} "
                          f"({done_count[0] / total_count * 100:.1f}%)")

            return result

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(worker, t): t["task_id"] for t in remaining}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  ERROR in {futures[fut]}: {e}")

    # ─── generate report ────────────────────────────────────────────
    print("[6/6] Generating report ...")
    all_results = list(iter_jsonl(results_path))
    all_cards = list(iter_jsonl(cards_path))

    ok_count = sum(1 for r in all_results if r.get("ok"))
    fail_count = sum(1 for r in all_results if not r.get("ok"))
    type_dist = Counter(c.get("type", "?") for c in all_cards)
    tier_dist = Counter(c.get("tier", "?") for c in all_cards)

    # Card quality stats
    edge_patterns = [c.get("edge_pattern", "") for c in all_cards if c.get("type") == "edge_fix"]
    fix_hints = [c.get("fix_hint", "") for c in all_cards if c.get("type") == "edge_fix"]
    wrong_approaches = [c.get("wrong_approach", "") for c in all_cards if c.get("type") == "wrong_approach"]

    report = {
        "total_tasks": len(failure_tasks),
        "tier_a_tasks": len(tier_a),
        "tier_b_tasks": len(tier_b),
        "ok": ok_count,
        "failed": fail_count,
        "total_cards": len(all_cards),
        "cards_per_task": len(all_cards) / max(ok_count, 1),
        "type_distribution": dict(type_dist),
        "tier_distribution": dict(tier_dist),
        "edge_fix_with_pattern": sum(1 for p in edge_patterns if len(p) > 5),
        "edge_fix_with_hint": sum(1 for h in fix_hints if len(h) > 5),
        "wrong_approach_with_approach": sum(1 for w in wrong_approaches if len(w) > 5),
        "model": args.model,
        "prompt_version": "v5_full_diagnostic",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n=== Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nCards written to: {cards_path}")


if __name__ == "__main__":
    main()
