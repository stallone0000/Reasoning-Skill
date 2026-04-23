#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_flash_experience_summarization.py

Build and run Gemini 3 Flash experience summarization pipeline:
1) Align flash/doubao outputs with final judge files
2) Keep only clearly judged samples and strictly parseable CoTs
3) Stratified split into train + holdout(1000)
4) Build success/failure/contrast summarize tasks from train split
5) Call cloudsway/gemini-3-flash-preview concurrently
6) Validate JSON outputs and write:
   - cards_raw.jsonl
   - problem_to_cards.jsonl
   - summarize_run_report.json

Supports a pilot mode to run a small number of tasks first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import requests


DEFAULT_MODEL = "cloudsway/gemini-3-flash-preview"
DEFAULT_URL = "http://api.360.cn/v1/chat/completions"
NO_ENV_SESSION = requests.Session()
NO_ENV_SESSION.trust_env = False
RM_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = RM_ROOT / "data"

SUCCESS_PROMPT_TEMPLATE = """You are an expert competitive programming coach.
Given a coding problem and a full successful reasoning chain, output 1-2 reusable lessons as strict JSON.

Output requirements:
1) Output ONLY valid JSON object with a top-level field: "cards".
2) cards must be a non-empty array. Each card must include:
   - type (must be "success")
   - trigger (array of 1-4 short strings)
   - tags (array of 1-6 short strings)
   - do (array of 1-5 concise actionable steps)
   - avoid (array of 1-4 concise anti-patterns)
   - check (array of 1-4 quick checks)
   - detours (array of 0-3 observed wrong turns from the chain)
   - risk (string, <= 25 words, when this lesson should NOT be applied)
   - complexity (string, optional but preferred)
3) Generalize aggressively: do not include concrete constants, variable names, examples, or story details from this specific problem.
4) Make each item short and operational.

[Problem]
{question}

[Full Successful Reasoning Chain]
{thinking_chain}
"""

FAILURE_PROMPT_TEMPLATE = """You are an expert competitive programming debugging coach.
Given a coding problem, a full failed reasoning chain, and judge diagnostics, output 1-2 reusable lessons as strict JSON.

Output requirements:
1) Output ONLY valid JSON object with a top-level field: "cards".
2) cards must be a non-empty array. Each card must include:
   - type (must be "failure")
   - trigger (array of 1-4 short strings)
   - tags (array of 1-6 short strings)
   - do (array of 1-5 concise corrective actions)
   - avoid (array of 1-4 concise anti-patterns that caused failure)
   - check (array of 1-4 pre-submit checks)
   - detours (array of 0-3 observed wrong turns from the chain)
   - risk (string, <= 25 words)
   - complexity (string, optional)
3) Use the diagnostics to identify failure-triggering input patterns in abstract form.
4) Generalize: do not include problem-specific constants, variable names, or sample literals.

[Problem]
{question}

[Full Failed Reasoning Chain]
{thinking_chain}

[Judge Diagnostics]
{judge_info}
"""

ASYMMETRIC_FAILURE_PROMPT_TEMPLATE = """You are an expert competitive programming debugging coach.
Given a coding problem, a failed reasoning chain, a successful reasoning chain for the same problem, and judge diagnostics,
output 1-2 reusable failure lessons as strict JSON.

Output requirements:
1) Output ONLY valid JSON object with a top-level field: "cards".
2) cards must be a non-empty array. Each card must include:
   - type (must be "failure")
   - trigger (array of 1-4 short strings)
   - tags (array of 1-6 short strings)
   - do (array of 1-5 concise corrective actions)
   - avoid (array of 1-4 concise anti-patterns that caused failure)
   - check (array of 1-4 pre-submit checks)
   - detours (array of 0-3 observed wrong turns from the chain)
   - risk (string, <= 25 words)
   - complexity (string, optional)
3) Use the successful chain only as a counterexample signal: isolate what the failed chain missed or mis-modeled.
4) Focus on reusable failure patterns, not story details or judge wording.
5) Generalize: do not include problem-specific constants, variable names, or sample literals.

[Problem]
{question}

[Failed Reasoning Chain]
{thinking_chain}

[Successful Counterexample Chain]
{success_chain}

[Judge Diagnostics]
{judge_info}
"""

CONTRAST_PROMPT_TEMPLATE = """You are an expert competitive programming coach.
Given one successful chain and one failed chain for the same problem, plus failed judge diagnostics,
extract 1-2 reusable contrast lessons as strict JSON.

Output requirements:
1) Output ONLY valid JSON object with top-level field: "cards".
2) cards must be a non-empty array. Each card must include:
   - type (must be "contrast")
   - trigger (array of 1-4 short strings)
   - tags (array of 1-6 short strings)
   - do (array of 1-5 concise actions choosing the successful path)
   - avoid (array of 1-4 concise anti-patterns seen in failed path)
   - check (array of 1-4 checks to distinguish paths early)
   - detours (array of 0-3 observed wrong turns)
   - risk (string, <= 25 words)
   - complexity (string, optional)
3) Focus on actionable decision splits, not retrospective narrative.
4) Generalize: do not include problem-specific constants, variable names, or sample literals.

[Problem]
{question}

[Successful Reasoning Chain]
{success_chain}

[Failed Reasoning Chain]
{failed_chain}

[Failed Judge Diagnostics]
{judge_info}
"""


@dataclass
class JoinedSample:
    unique_key: str
    question: str
    flash_output: str
    doubao_output: str
    flash_thought: str
    doubao_thought: str
    flash_judge: dict
    doubao_judge: dict
    flash_status: str
    doubao_status: str
    category: str


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def iter_json_array(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
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
                    try:
                        yield json.loads(buf)
                    except Exception:
                        pass
                    started = False
                    buf = ""


def first_flash_output(rec: dict) -> Optional[str]:
    for key in ("gemini_outputs", "correct_gemini_outputs"):
        val = rec.get(key)
        if isinstance(val, list) and val and isinstance(val[0], str) and val[0].strip():
            return val[0]
    return None


FLASH_END_RE = re.compile(r"</\s*thought\s*>", re.IGNORECASE)
DOUBAO_THINK_RE = re.compile(r"<\s*think\s*>(.*?)</\s*think\s*>", re.IGNORECASE | re.DOTALL)


def extract_flash_thought(text: str) -> Optional[str]:
    if not text:
        return None
    m = FLASH_END_RE.search(text)
    if not m:
        return None
    thought = text[: m.start()].strip()
    return thought if thought else None


def extract_doubao_thought(text: str) -> Optional[str]:
    if not text:
        return None
    m = DOUBAO_THINK_RE.search(text)
    if not m:
        return None
    thought = (m.group(1) or "").strip()
    return thought if thought else None


def normalize_status(status: Any) -> str:
    if isinstance(status, str):
        return status.strip().upper()
    return ""


def load_judge_map(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in iter_jsonl(path):
        key = row.get("unique_key")
        if key:
            out[key] = row
    return out


def load_flash_records(path: Path, keys: Set[str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for rec in iter_json_array(path):
        key = rec.get("unique_key")
        if key not in keys:
            continue
        question = rec.get("question")
        output = first_flash_output(rec)
        if isinstance(question, str) and isinstance(output, str):
            out[key] = {"question": question, "output": output}
    return out


def load_doubao_records(path: Path, keys: Set[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rec in iter_jsonl(path):
        key = rec.get("unique_key")
        if key not in keys:
            continue
        output = rec.get("llm_output")
        if isinstance(output, str):
            out[key] = output
    return out


def classify_category(f_status: str, d_status: str, ac_set: Set[str], wrong_set: Set[str]) -> Optional[str]:
    if f_status in ac_set and d_status in ac_set:
        return "both_ac"
    if f_status in ac_set and d_status in wrong_set:
        return "flash_ac_doubao_wrong"
    if f_status in wrong_set and d_status in ac_set:
        return "flash_wrong_doubao_ac"
    if f_status in wrong_set and d_status in wrong_set:
        return "both_wrong"
    return None


def stratified_holdout(
    records: List[JoinedSample],
    holdout_size: int,
    seed: int,
) -> Tuple[Set[str], Set[str], Dict[str, int]]:
    by_cat: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        by_cat[rec.category].append(rec.unique_key)

    rng = random.Random(seed)
    total = len(records)
    if total <= holdout_size:
        holdout = {r.unique_key for r in records}
        train = set()
        cat_counts = {k: len(v) for k, v in by_cat.items()}
        return train, holdout, cat_counts

    target: Dict[str, int] = {}
    for cat, keys in by_cat.items():
        target[cat] = int(round(holdout_size * len(keys) / total))
    # ensure at least 1 from each non-empty category when possible
    non_empty = [c for c, ks in by_cat.items() if ks]
    for cat in non_empty:
        if target[cat] == 0:
            target[cat] = 1

    def total_target() -> int:
        return sum(target.values())

    # adjust to exact holdout_size
    while total_target() > holdout_size:
        cat = max(non_empty, key=lambda c: target[c])
        if target[cat] > 1:
            target[cat] -= 1
        else:
            break
    while total_target() < holdout_size:
        cat = max(non_empty, key=lambda c: len(by_cat[c]) - target[c])
        if target[cat] < len(by_cat[cat]):
            target[cat] += 1
        else:
            break

    holdout: Set[str] = set()
    for cat, keys in by_cat.items():
        n = min(target.get(cat, 0), len(keys))
        if n <= 0:
            continue
        sampled = rng.sample(keys, n)
        holdout.update(sampled)

    # final adjust for exact size
    all_keys = [r.unique_key for r in records]
    if len(holdout) > holdout_size:
        holdout = set(rng.sample(list(holdout), holdout_size))
    elif len(holdout) < holdout_size:
        remaining = [k for k in all_keys if k not in holdout]
        extra_n = min(holdout_size - len(holdout), len(remaining))
        holdout.update(rng.sample(remaining, extra_n))

    train = set(all_keys) - holdout
    cat_counts = {k: len(v) for k, v in by_cat.items()}
    return train, holdout, cat_counts


def _shorten(text: str, limit: int = 1200) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [truncated, total {len(text)} chars] ...\n" + text[-half:]


def select_diagnostic_failures(case_results: List[dict], max_fail_cases: int, seed: int) -> List[dict]:
    fails = [cr for cr in case_results if (cr.get("status") or "").upper() != "AC"]
    if len(fails) <= max_fail_cases:
        return fails

    used: Set[Any] = set()
    chosen: List[dict] = []

    def add(cr: dict) -> None:
        idx = cr.get("case")
        if idx in used:
            return
        used.add(idx)
        chosen.append(cr)

    add(fails[0])
    add(fails[-1])

    for cr in fails:
        if (cr.get("status") or "").upper() == "TLE":
            add(cr)
            break

    wa = [cr for cr in fails if (cr.get("status") or "").upper() in ("WA", "WA_CHECKER")]
    if wa:
        def diff_score(cr: dict) -> float:
            got = cr.get("got") or ""
            exp = cr.get("exp") or ""
            return float(abs(len(got) - len(exp)) + 0.1 * (len(got) + len(exp)))
        add(max(wa, key=diff_score))

    rng = random.Random(seed)
    rest = [cr for cr in fails if cr.get("case") not in used]
    rng.shuffle(rest)
    for cr in rest:
        if len(chosen) >= max_fail_cases:
            break
        add(cr)

    return chosen[:max_fail_cases]


def flatten_tests_for_inputs(cases: dict, include_private: bool = False) -> List[dict]:
    kind = (cases or {}).get("kind")
    if kind == "stdio":
        return list(cases.get("tests") or [])
    if kind == "code_contests":
        tests = list(cases.get("public") or []) + list(cases.get("generated") or [])
        if include_private:
            tests += list(cases.get("private") or [])
        return tests
    if kind == "codeforces":
        return list(cases.get("official") or [])
    if kind == "function":
        return list(cases.get("tests") or [])
    return []


def extract_input_from_test(test: dict) -> Optional[str]:
    if not isinstance(test, dict):
        return None
    if "input" in test:
        inp = test.get("input")
        if isinstance(inp, str):
            return inp
        try:
            return json.dumps(inp, ensure_ascii=False)
        except Exception:
            return str(inp)
    if "args_json" in test:
        val = test.get("args_json")
        if isinstance(val, str):
            return val
    return None


def load_case_inputs_for_needed_indices(
    cases_path: Path,
    needed: Dict[str, Set[int]],
    include_private: bool = False,
) -> Dict[str, Dict[int, str]]:
    out: Dict[str, Dict[int, str]] = defaultdict(dict)
    if not needed:
        return out

    target_keys = set(needed.keys())
    seen = 0
    for row in iter_jsonl(cases_path):
        key = row.get("unique_key")
        if key not in target_keys:
            continue
        tests = flatten_tests_for_inputs(row.get("cases") or {}, include_private=include_private)
        for idx in needed[key]:
            if idx < 0 or idx >= len(tests):
                continue
            inp = extract_input_from_test(tests[idx])
            if inp is not None:
                out[key][idx] = inp
        seen += 1
        if seen >= len(target_keys):
            break
    return out


def summarize_judge_for_prompt(
    judge: dict,
    selected_fails: List[dict],
    case_inputs: Dict[int, str],
    max_case_chars: int,
) -> str:
    status = judge.get("status")
    passed = judge.get("passed", 0)
    total = judge.get("total_tests", 0)
    case_results = list(judge.get("case_results") or [])
    dist = Counter((cr.get("status") or "UNKNOWN") for cr in case_results)

    lines: List[str] = []
    lines.append(f"Judge status: {status}; passed={passed}/{total}")
    lines.append(f"Case status distribution: {dict(dist)}")
    if not selected_fails:
        return "\n".join(lines)

    lines.append("")
    lines.append(f"Selected failing cases ({len(selected_fails)}):")
    for cr in selected_fails:
        idx_raw = cr.get("case")
        try:
            idx = int(idx_raw)
        except Exception:
            idx = -1
        st = (cr.get("status") or "UNKNOWN").upper()
        lines.append(f"- Case {idx_raw}: {st}")
        inp = case_inputs.get(idx)
        if inp is not None:
            lines.append("  Input:")
            lines.append("  " + _shorten(inp, max_case_chars).replace("\n", "\n  "))

        if st in ("WA", "WA_CHECKER"):
            exp = cr.get("exp") or ""
            got = cr.get("got") or ""
            lines.append("  Expected:")
            lines.append("  " + _shorten(exp, max_case_chars).replace("\n", "\n  "))
            lines.append("  Got:")
            lines.append("  " + _shorten(got, max_case_chars).replace("\n", "\n  "))
        elif st == "TLE":
            lines.append("  Note: TLE on this input.")
        else:
            detail = cr.get("detail") or cr.get("stderr") or ""
            if detail:
                lines.append("  Detail:")
                lines.append("  " + _shorten(str(detail), max_case_chars).replace("\n", "\n  "))
    return "\n".join(lines)


def build_problem_tasks(
    records_by_key: Dict[str, JoinedSample],
    train_keys: Set[str],
    max_fail_cases: int,
    seed: int,
    emit_asymmetric_failure_tasks: bool,
) -> Tuple[List[dict], Dict[str, Set[int]]]:
    tasks: List[dict] = []
    needed_indices: Dict[str, Set[int]] = defaultdict(set)

    for key in sorted(train_keys):
        rec = records_by_key[key]
        cat = rec.category
        base = {
            "unique_key": key,
            "category": cat,
            "question": rec.question,
            "flash_status": rec.flash_status,
            "doubao_status": rec.doubao_status,
        }

        if cat == "both_ac":
            task = {
                **base,
                "task_type": "success",
                "source_model": "flash",
                "thinking_chain": rec.flash_thought,
                "failed_model": None,
                "selected_fails": [],
            }
            tasks.append(task)
            continue

        if cat == "both_wrong":
            selected = select_diagnostic_failures(
                list(rec.flash_judge.get("case_results") or []),
                max_fail_cases=max_fail_cases,
                seed=seed,
            )
            for cr in selected:
                try:
                    needed_indices[key].add(int(cr.get("case")))
                except Exception:
                    pass
            task = {
                **base,
                "task_type": "failure",
                "source_model": "flash",
                "thinking_chain": rec.flash_thought,
                "failed_model": "flash",
                "failed_judge": rec.flash_judge,
                "selected_fails": selected,
            }
            tasks.append(task)
            continue

        if cat == "flash_ac_doubao_wrong":
            selected = select_diagnostic_failures(
                list(rec.doubao_judge.get("case_results") or []),
                max_fail_cases=max_fail_cases,
                seed=seed,
            )
            for cr in selected:
                try:
                    needed_indices[key].add(int(cr.get("case")))
                except Exception:
                    pass
            task = {
                **base,
                "task_type": "contrast",
                "source_model": "both",
                "success_chain": rec.flash_thought,
                "failed_chain": rec.doubao_thought,
                "failed_model": "doubao",
                "failed_judge": rec.doubao_judge,
                "selected_fails": selected,
            }
            tasks.append(task)
            if emit_asymmetric_failure_tasks:
                failure_task = {
                    **base,
                    "task_type": "failure",
                    "source_model": "doubao",
                    "thinking_chain": rec.doubao_thought,
                    "paired_success_chain": rec.flash_thought,
                    "paired_success_model": "flash",
                    "failed_model": "doubao",
                    "failed_judge": rec.doubao_judge,
                    "selected_fails": selected,
                    "failure_origin": "asymmetric_with_counterexample",
                }
                tasks.append(failure_task)
            continue

        if cat == "flash_wrong_doubao_ac":
            selected = select_diagnostic_failures(
                list(rec.flash_judge.get("case_results") or []),
                max_fail_cases=max_fail_cases,
                seed=seed,
            )
            for cr in selected:
                try:
                    needed_indices[key].add(int(cr.get("case")))
                except Exception:
                    pass
            task = {
                **base,
                "task_type": "contrast",
                "source_model": "both",
                "success_chain": rec.doubao_thought,
                "failed_chain": rec.flash_thought,
                "failed_model": "flash",
                "failed_judge": rec.flash_judge,
                "selected_fails": selected,
            }
            tasks.append(task)
            if emit_asymmetric_failure_tasks:
                failure_task = {
                    **base,
                    "task_type": "failure",
                    "source_model": "flash",
                    "thinking_chain": rec.flash_thought,
                    "paired_success_chain": rec.doubao_thought,
                    "paired_success_model": "doubao",
                    "failed_model": "flash",
                    "failed_judge": rec.flash_judge,
                    "selected_fails": selected,
                    "failure_origin": "asymmetric_with_counterexample",
                }
                tasks.append(failure_task)
            continue

    for i, task in enumerate(tasks):
        content_hash = hashlib.md5(
            f"{task['unique_key']}|{task['task_type']}|{task.get('source_model')}".encode("utf-8")
        ).hexdigest()[:12]
        task["task_id"] = f"sum_{i:08d}_{content_hash}"
    return tasks, needed_indices


def build_prompt(task: dict, judge_info: Optional[str] = None) -> str:
    t = task["task_type"]
    if t == "success":
        return SUCCESS_PROMPT_TEMPLATE.format(
            question=task["question"],
            thinking_chain=task["thinking_chain"],
        )
    if t == "failure":
        if task.get("paired_success_chain"):
            return ASYMMETRIC_FAILURE_PROMPT_TEMPLATE.format(
                question=task["question"],
                thinking_chain=task["thinking_chain"],
                success_chain=task["paired_success_chain"],
                judge_info=judge_info or "",
            )
        return FAILURE_PROMPT_TEMPLATE.format(
            question=task["question"],
            thinking_chain=task["thinking_chain"],
            judge_info=judge_info or "",
        )
    if t == "contrast":
        return CONTRAST_PROMPT_TEMPLATE.format(
            question=task["question"],
            success_chain=task["success_chain"],
            failed_chain=task["failed_chain"],
            judge_info=judge_info or "",
        )
    raise ValueError(f"unknown task_type={t}")


def _extract_json_candidates(raw: str) -> List[str]:
    """
    Extract possible JSON substrings (object/list) via bracket stack matching.
    """
    cands: List[str] = []
    n = len(raw)
    i = 0
    while i < n:
        start_ch = raw[i]
        if start_ch not in "{[":
            i += 1
            continue
        start = i
        stack = [start_ch]
        i += 1
        in_str = False
        esc = False
        while i < n and stack:
            ch = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                top = stack[-1]
                if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                    stack.pop()
                else:
                    break
            i += 1
        if not stack:
            cands.append(raw[start:i])
    return cands


def extract_json_obj(text: str) -> Tuple[Optional[Any], Optional[str]]:
    if not isinstance(text, str):
        return None, "content_not_string"
    raw = text.strip()
    if not raw:
        return None, "empty_content"

    for candidate in (raw, raw.replace("```json", "").replace("```", "").strip()):
        try:
            obj = json.loads(candidate)
            return obj, None
        except Exception:
            pass

    for mid in _extract_json_candidates(raw):
        try:
            obj = json.loads(mid)
            return obj, None
        except Exception:
            continue
    return None, "no_json_object"


def validate_cards_obj(obj: Any) -> Tuple[Optional[List[dict]], Optional[str]]:
    if isinstance(obj, dict):
        cards = obj.get("cards")
    elif isinstance(obj, list):
        cards = obj
    else:
        return None, "root_not_object_or_list"
    if not isinstance(cards, list) or not cards:
        return None, "cards_missing_or_empty"

    required = {"type", "trigger", "tags", "do", "avoid", "check"}
    normalized: List[dict] = []

    def norm_list(val: Any) -> List[str]:
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return []
            parts = [p.strip(" \t-") for p in re.split(r"[;\n]+", s) if p.strip()]
            return parts or [s]
        return []

    def norm_scalar(val: Any) -> str:
        if isinstance(val, list):
            parts = [str(x).strip() for x in val if str(x).strip()]
            return " ; ".join(parts)
        return str(val or "").strip()

    for i, card in enumerate(cards):
        if not isinstance(card, dict):
            return None, f"card_{i}_not_object"
        missing = [k for k in required if k not in card]
        if missing:
            return None, f"card_{i}_missing:{','.join(missing)}"

        out_card = dict(card)
        out_card.setdefault("detours", [])
        out_card.setdefault("risk", "")
        out_card.setdefault("complexity", "")
        for arr_key in ("trigger", "tags", "do", "avoid", "check", "detours"):
            cleaned = norm_list(out_card.get(arr_key))
            if arr_key in ("trigger", "tags", "do", "avoid", "check") and not cleaned:
                return None, f"card_{i}_{arr_key}_empty_after_coerce"
            out_card[arr_key] = cleaned

        out_card["type"] = str(out_card.get("type") or "").strip().lower()
        out_card["risk"] = norm_scalar(out_card.get("risk"))
        out_card["complexity"] = norm_scalar(out_card.get("complexity"))
        normalized.append(out_card)
    return normalized, None


def post_chat_completion(
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout_s: int,
    max_tokens: int,
    use_env_proxy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[Optional[dict], Optional[str]]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "content_filter": False,
    }
    if max_tokens > 0:
        payload["max_tokens"] = int(max_tokens)
    if "gemini" in model.lower():
        payload["temperature"] = float(temperature)
        payload["top_p"] = float(top_p)
        payload["top_k"] = int(top_k)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Host": "api.360.cn",
    }
    try:
        client = requests if use_env_proxy else NO_ENV_SESSION
        resp = client.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
        resp.raise_for_status()
        j = resp.json()
    except Exception as e:
        return None, f"http_or_json_error:{type(e).__name__}:{e}"

    if "error" in j:
        return None, f"api_error:{j['error']}"
    return j, None


def run_task_with_retries(
    task: dict,
    prompt: str,
    *,
    url: str,
    api_key: str,
    model: str,
    timeout_s: int,
    max_tokens: int,
    use_env_proxy: bool,
    max_retries: int,
    backoff_base_s: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict:
    last_err = None
    current_prompt = prompt
    attempts_made = 0
    last_raw_content: Optional[str] = None
    last_usage: Dict[str, Any] = {}
    last_finish_reason: Optional[str] = None

    for attempt in range(max_retries):
        attempts_made = attempt + 1
        response_json, err = post_chat_completion(
            url=url,
            api_key=api_key,
            model=model,
            prompt=current_prompt,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            use_env_proxy=use_env_proxy,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        if err:
            last_err = err
            if attempt < max_retries - 1:
                time.sleep(backoff_base_s * (2 ** attempt))
                continue
            break

        usage = response_json.get("usage") or {}
        last_usage = usage
        content = ""
        try:
            content = response_json["choices"][0]["message"]["content"]
            last_raw_content = content
            last_finish_reason = response_json["choices"][0].get("finish_reason")
        except Exception as e:
            last_err = f"bad_response_schema:{type(e).__name__}:{e}"
            if attempt < max_retries - 1:
                time.sleep(backoff_base_s * (2 ** attempt))
                continue
            break

        obj, parse_err = extract_json_obj(content)
        if parse_err:
            last_err = parse_err
        else:
            cards, valid_err = validate_cards_obj(obj)
            if not valid_err:
                return {
                    "ok": True,
                    "attempts": attempts_made,
                    "usage": usage,
                    "raw_content": content,
                    "cards": cards,
                    "finish_reason": last_finish_reason,
                    "error": None,
                }
            last_err = valid_err

        # retry with one-shot repair prompt
        if attempt < max_retries - 1:
            repair = (
                "Convert the text below into strict JSON only.\n"
                "Output exactly one JSON object with top-level key `cards`.\n"
                "Each card must include keys: type, trigger, tags, do, avoid, check, detours, risk, complexity.\n"
                "No markdown, no explanations.\n\n"
                "[YOUR PREVIOUS OUTPUT]\n"
                f"{content}\n"
            )
            current_prompt = repair
            time.sleep(backoff_base_s * (2 ** attempt))
            continue

    return {
        "ok": False,
        "attempts": attempts_made,
        "usage": last_usage,
        "raw_content": last_raw_content,
        "finish_reason": last_finish_reason,
        "cards": [],
        "error": last_err or "unknown_error",
    }


def write_jsonl(path: Path, rows: Iterable[dict], mode: str = "a") -> None:
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_input_path(raw: str, candidates: List[Path]) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    for root in candidates:
        candidate = (root / raw).resolve()
        if candidate.exists():
            return candidate
    return (candidates[0] / raw).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini flash experience summarization pipeline")
    parser.add_argument("--flash_input", default="nemotron_cp_unique_questions_34729_withimages_flash.json")
    parser.add_argument("--doubao_input", default="doubao_seed_cp_34799_v1.jsonl")
    parser.add_argument("--flash_judge", default="judge_results_flash_final_fixed.jsonl")
    parser.add_argument("--doubao_judge", default="judge_results_doubao_final_fixed.jsonl")
    parser.add_argument("--cases", default="nemotron_cp_cases_34799_v1.jsonl")
    parser.add_argument("--output_dir", default="experience_summary_outputs")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_url", default=DEFAULT_URL)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--workers", type=int, default=500)
    parser.add_argument("--timeout_s", type=int, default=300)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=0,
        help="max completion tokens for summarize call (0 means do not set max_tokens)",
    )
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--backoff_base_s", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument(
        "--use_env_proxy",
        action="store_true",
        help="Use proxy settings from environment (default: disabled).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout_size", type=int, default=1000)
    parser.add_argument("--max_fail_cases", type=int, default=8)
    parser.add_argument("--max_case_chars", type=int, default=900)
    parser.add_argument("--pilot_tasks", type=int, default=0, help="Only run first N tasks after shuffling")
    parser.add_argument("--shuffle_tasks", action="store_true", help="Shuffle tasks before pilot slicing")
    parser.add_argument("--write_tasks_only", action="store_true", help="Prepare datasets and tasks but do not call API")
    parser.add_argument(
        "--disable_asymmetric_failure_tasks",
        action="store_true",
        help="Keep old behavior: asymmetric wrong categories only emit contrast tasks",
    )
    parser.add_argument(
        "--clear_wrong_statuses",
        default="WA,TLE,WA_CHECKER",
        help="Comma-separated statuses treated as clear wrong",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    flash_input = resolve_input_path(args.flash_input, [base, DATA_ROOT / "questions"])
    doubao_input = resolve_input_path(args.doubao_input, [base, DATA_ROOT / "model_outputs"])
    flash_judge = resolve_input_path(args.flash_judge, [base, DATA_ROOT / "judge_results"])
    doubao_judge = resolve_input_path(args.doubao_judge, [base, DATA_ROOT / "judge_results"])
    cases_path = resolve_input_path(args.cases, [base, DATA_ROOT / "questions"])
    output_dir = (base / args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    valid_samples_path = output_dir / "valid_samples.jsonl"
    split_path = output_dir / "split_keys.json"
    tasks_path = output_dir / "summarize_tasks.jsonl"
    prompt_manifest_path = output_dir / "prompt_manifest.jsonl"
    task_results_path = output_dir / "task_results.jsonl"
    cards_raw_path = output_dir / "cards_raw.jsonl"
    problem_to_cards_path = output_dir / "problem_to_cards.jsonl"
    report_path = output_dir / "summarize_run_report.json"

    print("Resolved inputs ...")
    print(f"  flash_input={flash_input}")
    print(f"  doubao_input={doubao_input}")
    print(f"  flash_judge={flash_judge}")
    print(f"  doubao_judge={doubao_judge}")
    print(f"  cases={cases_path}")

    print("Loading judge maps ...")
    flash_jmap = load_judge_map(flash_judge)
    doubao_jmap = load_judge_map(doubao_judge)
    common_keys = set(flash_jmap.keys()) & set(doubao_jmap.keys())
    print(f"  flash_judge={len(flash_jmap)}, doubao_judge={len(doubao_jmap)}, common={len(common_keys)}")

    ac_set = {"AC"}
    wrong_set = {s.strip().upper() for s in args.clear_wrong_statuses.split(",") if s.strip()}

    clear_keys: Set[str] = set()
    clear_category_counter = Counter()
    for key in common_keys:
        fs = normalize_status((flash_jmap[key].get("judge") or {}).get("status"))
        ds = normalize_status((doubao_jmap[key].get("judge") or {}).get("status"))
        cat = classify_category(fs, ds, ac_set, wrong_set)
        if cat:
            clear_keys.add(key)
            clear_category_counter[cat] += 1
    print(f"  clear_keys={len(clear_keys)} with categories={dict(clear_category_counter)}")

    print("Loading flash records for clear keys ...")
    flash_map = load_flash_records(flash_input, clear_keys)
    print(f"  flash records loaded={len(flash_map)}")

    print("Loading doubao records for clear keys ...")
    doubao_map = load_doubao_records(doubao_input, clear_keys)
    print(f"  doubao records loaded={len(doubao_map)}")

    print("Building strict valid samples (both thought formats required) ...")
    records: List[JoinedSample] = []
    drop_counter = Counter()
    for key in sorted(clear_keys):
        f = flash_map.get(key)
        d_out = doubao_map.get(key)
        if not f or not d_out:
            drop_counter["missing_model_output"] += 1
            continue

        f_out = f["output"]
        q = f["question"]
        f_th = extract_flash_thought(f_out)
        d_th = extract_doubao_thought(d_out)
        if not f_th:
            drop_counter["flash_thought_parse_fail"] += 1
            continue
        if not d_th:
            drop_counter["doubao_thought_parse_fail"] += 1
            continue

        f_status = normalize_status((flash_jmap[key].get("judge") or {}).get("status"))
        d_status = normalize_status((doubao_jmap[key].get("judge") or {}).get("status"))
        cat = classify_category(f_status, d_status, ac_set, wrong_set)
        if not cat:
            drop_counter["category_not_clear"] += 1
            continue

        records.append(
            JoinedSample(
                unique_key=key,
                question=q,
                flash_output=f_out,
                doubao_output=d_out,
                flash_thought=f_th,
                doubao_thought=d_th,
                flash_judge=flash_jmap[key].get("judge") or {},
                doubao_judge=doubao_jmap[key].get("judge") or {},
                flash_status=f_status,
                doubao_status=d_status,
                category=cat,
            )
        )

    print(f"  valid_samples={len(records)}, dropped={dict(drop_counter)}")
    write_jsonl(
        valid_samples_path,
        (
            {
                "unique_key": r.unique_key,
                "category": r.category,
                "flash_status": r.flash_status,
                "doubao_status": r.doubao_status,
            }
            for r in records
        ),
        mode="w",
    )

    if not records:
        raise RuntimeError("No valid samples after strict filtering.")

    rec_by_key = {r.unique_key: r for r in records}
    train_keys, holdout_keys, valid_cat_counts = stratified_holdout(records, args.holdout_size, args.seed)
    print(f"Split done: train={len(train_keys)}, holdout={len(holdout_keys)}")

    with split_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "holdout_size": args.holdout_size,
                "valid_total": len(records),
                "valid_category_counts": valid_cat_counts,
                "train_size": len(train_keys),
                "holdout_size_actual": len(holdout_keys),
                "train_keys": sorted(train_keys),
                "holdout_keys": sorted(holdout_keys),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Building summarize tasks from train split ...")
    tasks, needed_indices = build_problem_tasks(
        rec_by_key,
        train_keys,
        max_fail_cases=args.max_fail_cases,
        seed=args.seed,
        emit_asymmetric_failure_tasks=not args.disable_asymmetric_failure_tasks,
    )
    print(f"  tasks_total={len(tasks)}; keys needing case inputs={len(needed_indices)}")

    if args.shuffle_tasks:
        random.Random(args.seed).shuffle(tasks)

    if args.pilot_tasks > 0:
        tasks = tasks[: args.pilot_tasks]
        needed_subset: Dict[str, Set[int]] = defaultdict(set)
        for t in tasks:
            key = t["unique_key"]
            for cr in t.get("selected_fails", []):
                try:
                    needed_subset[key].add(int(cr.get("case")))
                except Exception:
                    pass
        needed_indices = needed_subset
        print(f"  pilot mode enabled: tasks={len(tasks)}, case_input_keys={len(needed_indices)}")

    write_jsonl(tasks_path, tasks, mode="w")

    print("Loading needed case inputs ...")
    case_inputs = load_case_inputs_for_needed_indices(cases_path, needed_indices, include_private=False)
    print(f"  loaded case inputs for {len(case_inputs)} keys")

    api_key = args.api_key or os.environ.get("API_KEY_360") or os.environ.get("QINIU_API_KEY")
    if not api_key and not args.write_tasks_only:
        raise RuntimeError("Missing API key. Set API_KEY_360/QINIU_API_KEY or pass --api_key.")

    # prebuild prompts for deterministic hashing and task export
    prompt_map: Dict[str, str] = {}
    prompt_manifest_rows: List[dict] = []
    for task in tasks:
        judge_info = None
        if task["task_type"] in ("failure", "contrast"):
            key = task["unique_key"]
            judge_info = summarize_judge_for_prompt(
                task["failed_judge"],
                task.get("selected_fails", []),
                case_inputs.get(key, {}),
                max_case_chars=args.max_case_chars,
            )
        prompt = build_prompt(task, judge_info=judge_info)
        prompt_map[task["task_id"]] = prompt
        prompt_manifest_rows.append(
            {
                "task_id": task["task_id"],
                "unique_key": task["unique_key"],
                "task_type": task["task_type"],
                "category": task["category"],
                "source_model": task.get("source_model"),
                "failed_model": task.get("failed_model"),
                "failure_origin": task.get("failure_origin"),
                "paired_success_model": task.get("paired_success_model"),
                "prompt_len": len(prompt),
                "prompt": prompt,
            }
        )
    write_jsonl(prompt_manifest_path, prompt_manifest_rows, mode="w")

    if args.write_tasks_only:
        print("write_tasks_only enabled; skipping API calls.")
        task_type_counts = Counter(task["task_type"] for task in tasks)
        failure_origin_counts = Counter(
            task.get("failure_origin") or "none" for task in tasks if task["task_type"] == "failure"
        )
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "write_tasks_only",
            "valid_samples": len(records),
            "train_size": len(train_keys),
            "holdout_size": len(holdout_keys),
            "tasks_prepared": len(tasks),
            "task_type_counts": dict(task_type_counts),
            "failure_origin_counts": dict(failure_origin_counts),
            "asymmetric_failure_tasks_enabled": not args.disable_asymmetric_failure_tasks,
            "output_dir": str(output_dir),
            "prompt_manifest": str(prompt_manifest_path),
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return

    # Load existing completed tasks for resume
    done_task_ids: Set[str] = set()
    if task_results_path.exists():
        for row in iter_jsonl(task_results_path):
            if row.get("ok") and row.get("task_id"):
                done_task_ids.add(row["task_id"])
    print(f"Resume: done_task_ids={len(done_task_ids)}")

    pending = [t for t in tasks if t["task_id"] not in done_task_ids]
    print(f"Pending tasks={len(pending)}")
    if not pending:
        print("No pending task. Building card mapping/report only.")

    write_lock = threading.Lock()
    stats = Counter()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_total_tokens = 0
    total_attempts = 0
    error_counter = Counter()

    # ensure files exist
    ensure_parent(task_results_path)
    ensure_parent(cards_raw_path)
    if not task_results_path.exists():
        task_results_path.touch()
    if not cards_raw_path.exists():
        cards_raw_path.touch()

    def worker(task: dict) -> dict:
        prompt = prompt_map[task["task_id"]]
        result = run_task_with_retries(
            task,
            prompt,
            url=args.api_url,
            api_key=api_key,
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
        usage = result.get("usage") or {}
        return {
            "task": task,
            "result": result,
            "usage": usage,
        }

    if pending:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(worker, task): task for task in pending}
            for i, fut in enumerate(as_completed(futs), 1):
                payload = fut.result()
                task = payload["task"]
                result = payload["result"]
                usage = payload["usage"]

                attempts = int(result.get("attempts") or 0)
                total_attempts += attempts
                stats["processed"] += 1
                stats["attempts_total"] += attempts

                p_tok = int(usage.get("prompt_tokens") or 0)
                c_tok = int(usage.get("completion_tokens") or 0)
                t_tok = int(usage.get("total_tokens") or 0)
                total_prompt_tokens += p_tok
                total_completion_tokens += c_tok
                total_total_tokens += t_tok

                task_row = {
                    "task_id": task["task_id"],
                    "unique_key": task["unique_key"],
                    "task_type": task["task_type"],
                    "category": task["category"],
                    "source_model": task["source_model"],
                    "ok": bool(result.get("ok")),
                    "attempts": attempts,
                    "error": result.get("error"),
                    "finish_reason": result.get("finish_reason"),
                    "raw_content_head": (result.get("raw_content") or "")[:2000],
                    "usage": usage,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                card_rows: List[dict] = []
                if result.get("ok"):
                    stats["ok"] += 1
                    cards = result.get("cards") or []
                    for idx, card in enumerate(cards):
                        card_id = f"{task['task_id']}_c{idx}"
                        card_row = {
                            "card_id": card_id,
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
                        card_rows.append(card_row)
                    stats["cards_written"] += len(card_rows)
                    stats[f"task_type_{task['task_type']}"] += 1
                else:
                    stats["failed"] += 1
                    error_counter[str(result.get("error") or "unknown_error")] += 1

                with write_lock:
                    write_jsonl(task_results_path, [task_row], mode="a")
                    if card_rows:
                        write_jsonl(cards_raw_path, card_rows, mode="a")

                if i % 20 == 0 or i == len(pending):
                    elapsed = time.time() - t0
                    speed = i / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{i}/{len(pending)}] ok={stats['ok']} fail={stats['failed']} "
                        f"cards={stats['cards_written']} speed={speed:.2f} task/s"
                    )

    # rebuild problem_to_cards from cards_raw
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

    # compute overall success summary from task_results
    overall = Counter()
    overall_errors = Counter()
    token_prompt_sum = 0
    token_completion_sum = 0
    token_total_sum = 0
    attempts_sum = 0
    rows_total = 0
    for row in iter_jsonl(task_results_path):
        rows_total += 1
        if row.get("ok"):
            overall["ok"] += 1
            overall[f"task_type_{row.get('task_type')}"] += 1
        else:
            overall["failed"] += 1
            overall_errors[str(row.get("error") or "unknown_error")] += 1
        usage = row.get("usage") or {}
        token_prompt_sum += int(usage.get("prompt_tokens") or 0)
        token_completion_sum += int(usage.get("completion_tokens") or 0)
        token_total_sum += int(usage.get("total_tokens") or 0)
        attempts_sum += int(row.get("attempts") or 0)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model": args.model,
            "api_url": args.api_url,
            "workers": args.workers,
            "timeout_s": args.timeout_s,
            "max_tokens": args.max_tokens,
            "max_retries": args.max_retries,
            "backoff_base_s": args.backoff_base_s,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "use_env_proxy": args.use_env_proxy,
            "seed": args.seed,
            "holdout_size": args.holdout_size,
            "max_fail_cases": args.max_fail_cases,
            "clear_wrong_statuses": sorted(wrong_set),
            "pilot_tasks": args.pilot_tasks,
            "asymmetric_failure_tasks_enabled": not args.disable_asymmetric_failure_tasks,
        },
        "data_stats": {
            "common_judge_keys": len(common_keys),
            "clear_keys": len(clear_keys),
            "clear_category_counts": dict(clear_category_counter),
            "drop_counts": dict(drop_counter),
            "valid_samples": len(records),
            "valid_category_counts": dict(Counter(r.category for r in records)),
            "train_size": len(train_keys),
            "holdout_size_actual": len(holdout_keys),
            "tasks_prepared": len(tasks),
            "task_type_counts": dict(Counter(task["task_type"] for task in tasks)),
            "failure_origin_counts": dict(
                Counter(task.get("failure_origin") or "none" for task in tasks if task["task_type"] == "failure")
            ),
        },
        "run_stats": {
            "task_rows_total": rows_total,
            "ok": overall["ok"],
            "failed": overall["failed"],
            "error_counts": dict(overall_errors),
            "cards_total": card_count,
            "problem_with_cards": len(p2c),
            "avg_attempts_per_task": (attempts_sum / rows_total) if rows_total else 0.0,
            "token_usage": {
                "prompt_tokens": token_prompt_sum,
                "completion_tokens": token_completion_sum,
                "total_tokens": token_total_sum,
                "avg_total_tokens_per_task": (token_total_sum / rows_total) if rows_total else 0.0,
            },
        },
        "outputs": {
            "valid_samples": str(valid_samples_path),
            "split_keys": str(split_path),
            "summarize_tasks": str(tasks_path),
            "prompt_manifest": str(prompt_manifest_path),
            "task_results": str(task_results_path),
            "cards_raw": str(cards_raw_path),
            "problem_to_cards": str(problem_to_cards_path),
            "report": str(report_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"  cards_raw: {cards_raw_path}")
    print(f"  problem_to_cards: {problem_to_cards_path}")
    print(f"  summarize_run_report: {report_path}")


if __name__ == "__main__":
    main()
