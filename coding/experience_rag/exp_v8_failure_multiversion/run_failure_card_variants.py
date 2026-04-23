#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run multiple failure-card prompt variants on the same failure task set.

This script is designed for parallel prompt experimentation on wrong-answer cases.
It supports:
1. Multi-variant prompt generation on the same task pool
2. `--write_prompts_only` mode for no-API environments
3. Parallel variant execution when API keys are available
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

RM_ROOT = None
for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "LATEST_STATUS.md").exists() and (_candidate / "README.md").exists():
        RM_ROOT = _candidate
        break
if RM_ROOT is None:
    raise RuntimeError("Cannot locate reasoning_memory root")
if str(RM_ROOT) not in sys.path:
    sys.path.insert(0, str(RM_ROOT))

from rm_runtime import DEFAULT_SAMPLING_PAYLOAD, QihooChatClient, clear_proxy_env, resolve_api_key

EXP_RAG_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_RAG_ROOT))

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from card_quality import analyze_cards, split_cards_by_quality, summarize_issues
from exp_v5_full_diagnostic.resummarize_failures_v5 import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_URL,
    ROOT,
    build_failing_cases_block,
    build_selected_cases_block,
    extract_code_from_text,
    extract_json_obj,
    iter_jsonl,
    load_all_case_io,
    validate_v5_cards,
)

EXPECTED_VARIANTS = (
    "counterexample_first",
    "invariant_guardrail",
    "retrieval_ready",
    "counterexample_delta",
    "success_gap_checklist",
)

SUMMARY_DIR_CANDIDATES = (
    ROOT.parent / "experiments" / "experience_summary" / "runs" / "full_v2_nomax",
    ROOT.parent / "experiments" / "experience_summary" / "full_v2_nomax",
)
CASES_PATH = ROOT.parent / "data" / "questions" / "nemotron_cp_cases_34799_v1.jsonl"
DEFAULT_MAX_PROMPT_CHARS = 28000

TIER_A_PROMPTS: Dict[str, str] = {
    "counterexample_first": """You are an expert competitive programming debugger extracting reusable edge-case memory.

This code passes some tests but fails on specific inputs.
Your job is to identify the SMALLEST reusable counterexample pattern and the missing guardrail.

Rules:
1. `trigger` must describe problem-level signals, not judge words.
2. `edge_pattern` must describe the exact family of inputs that breaks the code.
3. `fix_hint` must be a concise repair instruction future solvers can apply before resubmitting.
4. Prefer one sharp card over multiple vague cards.
5. Generalize aggressively: no story details, no copied literals, no variable names.

[Problem]
{question}

[Submitted Code]
```{lang}
{code}
```

[Test Results: {passed}/{total} passed ({pass_pct}%)]
Error distribution: {error_dist}

[Failing Cases]
{failing_cases_block}

{success_counterexample_block}

Output ONLY valid JSON:
{{"cards": [{{
  "type": "edge_fix",
  "trigger": ["1-3 reusable trigger strings"],
  "tags": ["1-4 algorithm/topic tags"],
  "edge_pattern": "minimal failure family",
  "fix_hint": "short repair hint",
  "do": ["1-2 actions"],
  "avoid": ["1 wrong assumption"],
  "check": ["1 pre-submit guardrail"]
}}]}}
""",
    "invariant_guardrail": """You are an expert competitive programming debugger extracting invariant-focused failure memory.

This code is close but violates a hidden invariant on some inputs.
Your job is to state what input pattern exposes that invariant break and what guardrail would catch it.

Rules:
1. `trigger` should point to when this invariant matters.
2. `edge_pattern` should name the failing structure, boundary, ordering, or state configuration.
3. `fix_hint` should tell a future solver how to preserve or verify the invariant.
4. Rewrite sample values into the invariant or structure they reveal.
5. Avoid judge wording and avoid retrospective narrative.
6. Keep the card retrieval-friendly and concrete.

[Problem]
{question}

[Submitted Code]
```{lang}
{code}
```

[Test Results: {passed}/{total} passed ({pass_pct}%)]
Error distribution: {error_dist}

[Failing Cases]
{failing_cases_block}

{success_counterexample_block}

Output ONLY valid JSON:
{{"cards": [{{
  "type": "edge_fix",
  "trigger": ["1-3 trigger strings"],
  "tags": ["1-4 tags"],
  "edge_pattern": "input/state pattern where invariant breaks",
  "fix_hint": "how to preserve or verify the invariant",
  "do": ["1-2 actions"],
  "avoid": ["1 brittle shortcut"],
  "check": ["1 invariant check"]
}}]}}
""",
    "retrieval_ready": """You are building a retrieval memory bank for competitive programming wrong answers.

Produce a short, high-utility failure card future solvers can immediately use.

Rules:
1. Optimize for retrieval value: trigger must be discriminative, not generic.
2. `edge_pattern` and `fix_hint` should be short enough to inject into a solve prompt.
3. Keep all fields concrete and reusable.
4. Rewrite sample-specific values into structural conditions or constraints.
5. No judge jargon, no story details, no copied literals, no sample-number triggers.

[Problem]
{question}

[Submitted Code]
```{lang}
{code}
```

[Test Results: {passed}/{total} passed ({pass_pct}%)]
Error distribution: {error_dist}

[Failing Cases]
{failing_cases_block}

{success_counterexample_block}

Output ONLY valid JSON:
{{"cards": [{{
  "type": "edge_fix",
  "trigger": ["1-2 strong trigger strings"],
  "tags": ["1-4 tags"],
  "edge_pattern": "short breaking pattern",
  "fix_hint": "short reusable fix",
  "do": ["1-2 actions"],
  "avoid": ["1 anti-pattern"],
  "check": ["1 fast check"]
}}]}}
""",
    "counterexample_delta": """You are building a retrieval memory bank for competitive programming wrong answers.

Use the failed attempt plus any successful counterexample chain to isolate the smallest decisive gap.

Rules:
1. If a successful counterexample chain is provided, use it to identify the missing observation, invariant, or branch in the failed attempt.
2. `trigger` must describe when this exact gap appears.
3. `edge_pattern` should capture the smallest failure family exposed by that gap.
4. `fix_hint` should say what to test or restore before resubmitting.
5. Abstract away sample values: name the structural condition, not the literal example.
6. No judge jargon, no copied literals, no retrospective storytelling.

[Problem]
{question}

[Submitted Code]
```{lang}
{code}
```

[Test Results: {passed}/{total} passed ({pass_pct}%)]
Error distribution: {error_dist}

[Failing Cases]
{failing_cases_block}

{success_counterexample_block}

Output ONLY valid JSON:
{{"cards": [{{
  "type": "edge_fix",
  "trigger": ["1-2 discriminative trigger strings"],
  "tags": ["1-4 tags"],
  "edge_pattern": "smallest failure family",
  "fix_hint": "missing observation or correction",
  "do": ["1-2 actions"],
  "avoid": ["1 anti-pattern"],
  "check": ["1 decisive check"]
}}]}}
""",
    "success_gap_checklist": """You are building a retrieval memory bank for competitive programming wrong answers.

Turn this failed attempt into a short reusable checklist card.

Rules:
1. If a successful counterexample chain is provided, compress the gap into one missed checklist item.
2. `trigger` should tell future solvers when to run this checklist.
3. `edge_pattern` and `fix_hint` must stay short enough for prompt injection.
4. Prefer one precise checklist-style lesson over any narrative summary.
5. Rewrite sample-specific numbers or strings into the reusable condition they represent.
6. No judge jargon or copied literals.

[Problem]
{question}

[Submitted Code]
```{lang}
{code}
```

[Test Results: {passed}/{total} passed ({pass_pct}%)]
Error distribution: {error_dist}

[Failing Cases]
{failing_cases_block}

{success_counterexample_block}

Output ONLY valid JSON:
{{"cards": [{{
  "type": "edge_fix",
  "trigger": ["1-2 short trigger strings"],
  "tags": ["1-4 tags"],
  "edge_pattern": "short failure pattern",
  "fix_hint": "one missing checklist item",
  "do": ["1-2 checklist actions"],
  "avoid": ["1 brittle shortcut"],
  "check": ["1 pre-submit checklist question"]
}}]}}
""",
}

TIER_B_PROMPTS: Dict[str, str] = {
    "counterexample_first": """You are an expert competitive programming debugger extracting reusable memory from a totally wrong solution.

This code fails all tests. Infer the wrong approach and the better direction using the failing cases.

Rules:
1. `trigger` must describe the kind of problem where this failure mode appears.
2. `wrong_approach` should name the mistaken algorithmic move or mental shortcut.
3. `correct_direction` should give the next approach family or a diagnostic fork to test before committing.
4. Prefer reusable strategy language, not retrospective explanation.
5. No judge jargon or copied literals.

[Problem]
{question}

[Submitted Code]
```{lang}
{code}
```

[Test Results: 0/{total} passed]
Error distribution: {error_dist}

[Selected Failing Cases]
{failing_cases_block}

[Approach Summary]
{approach_summary}

{success_counterexample_block}

Output ONLY valid JSON:
{{"cards": [{{
  "type": "wrong_approach",
  "trigger": ["1-3 trigger strings"],
  "tags": ["1-4 tags"],
  "wrong_approach": "specific mistaken approach",
  "correct_direction": "better direction or decision fork",
  "do": ["1-2 actions"],
  "avoid": ["1 anti-pattern"],
  "check": ["1 diagnostic question"]
}}]}}
""",
    "invariant_guardrail": """You are an expert competitive programming debugger extracting invariant-level memory from a fully failing solution.

This code fails all tests, which usually means the state model or core invariant is wrong.

Rules:
1. `wrong_approach` should name the broken model, not just say "greedy" or "brute force" unless that is the real issue.
2. `correct_direction` should say what state, invariant, or decomposition the solver should verify next.
3. `trigger` must describe when this class of mistake is likely.
4. Convert sample-specific values into the underlying invariant or structure.
5. Keep the card concise and reusable.

[Problem]
{question}

[Submitted Code]
```{lang}
{code}
```

[Test Results: 0/{total} passed]
Error distribution: {error_dist}

[Selected Failing Cases]
{failing_cases_block}

[Approach Summary]
{approach_summary}

{success_counterexample_block}

Output ONLY valid JSON:
{{"cards": [{{
  "type": "wrong_approach",
  "trigger": ["1-3 trigger strings"],
  "tags": ["1-4 tags"],
  "wrong_approach": "broken state model or wrong algorithmic commitment",
  "correct_direction": "invariant or state representation to verify next",
  "do": ["1-2 actions"],
  "avoid": ["1 anti-pattern"],
  "check": ["1 invariant diagnostic question"]
}}]}}
""",
    "retrieval_ready": """You are building a retrieval memory bank for failed competitive programming attempts.

Produce one short, high-utility wrong-approach card.

Rules:
1. `trigger` must be short but discriminative.
2. `wrong_approach` must name the exact bad move future solvers should avoid.
3. `correct_direction` must be a short next-step hint, not a full tutorial.
4. Keep the card injection-friendly.
5. Convert sample-specific values into structural conditions before writing the trigger.
6. No judge jargon, no copied literals, no long narratives.

[Problem]
{question}

[Submitted Code]
```{lang}
{code}
```

[Test Results: 0/{total} passed]
Error distribution: {error_dist}

[Selected Failing Cases]
{failing_cases_block}

[Approach Summary]
{approach_summary}

{success_counterexample_block}

Output ONLY valid JSON:
{{"cards": [{{
  "type": "wrong_approach",
  "trigger": ["1-2 strong trigger strings"],
  "tags": ["1-4 tags"],
  "wrong_approach": "short wrong move",
  "correct_direction": "short better direction",
  "do": ["1-2 actions"],
  "avoid": ["1 anti-pattern"],
  "check": ["1 quick question"]
}}]}}
""",
    "counterexample_delta": """You are building a retrieval memory bank for failed competitive programming attempts.

Use the failed attempt and any successful counterexample chain to identify the decisive wrong commitment.

Rules:
1. If a successful counterexample chain is provided, compare it against the failed attempt and isolate the first decision gap.
2. `wrong_approach` must name the exact mistaken commitment.
3. `correct_direction` must give the smallest better direction or diagnostic fork.
4. Keep the card concise and injection-friendly.
5. Abstract sample values into the underlying condition or invariant gap.
6. No judge jargon, no copied literals, no narrative recap.

[Problem]
{question}

[Submitted Code]
```{lang}
{code}
```

[Test Results: 0/{total} passed]
Error distribution: {error_dist}

[Selected Failing Cases]
{failing_cases_block}

[Approach Summary]
{approach_summary}

{success_counterexample_block}

Output ONLY valid JSON:
{{"cards": [{{
  "type": "wrong_approach",
  "trigger": ["1-2 strong trigger strings"],
  "tags": ["1-4 tags"],
  "wrong_approach": "decisive wrong commitment",
  "correct_direction": "small better direction or diagnostic fork",
  "do": ["1-2 actions"],
  "avoid": ["1 anti-pattern"],
  "check": ["1 decisive diagnostic question"]
}}]}}
""",
    "success_gap_checklist": """You are building a retrieval memory bank for failed competitive programming attempts.

Turn this failed attempt into a short reusable checklist card.

Rules:
1. If a successful counterexample chain is provided, compress the gap into one missed checklist item.
2. `wrong_approach` should name the mistaken checklist miss, assumption, or state model.
3. `correct_direction` should be a short next-step checklist item.
4. Keep the card short, specific, and reusable.
5. Rewrite any sample-specific value into a reusable trigger condition.
6. No judge jargon or retrospective narration.

[Problem]
{question}

[Submitted Code]
```{lang}
{code}
```

[Test Results: 0/{total} passed]
Error distribution: {error_dist}

[Selected Failing Cases]
{failing_cases_block}

[Approach Summary]
{approach_summary}

{success_counterexample_block}

Output ONLY valid JSON:
{{"cards": [{{
  "type": "wrong_approach",
  "trigger": ["1-2 short trigger strings"],
  "tags": ["1-4 tags"],
  "wrong_approach": "missed checklist item or wrong commitment",
  "correct_direction": "short corrective checklist item",
  "do": ["1-2 actions"],
  "avoid": ["1 anti-pattern"],
  "check": ["1 pre-submit checklist question"]
}}]}}
""",
}


def resolve_summary_dir(explicit: str = "") -> Path:
    if explicit:
        candidate = Path(explicit).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"summary_dir does not exist: {candidate}")
        return candidate
    for candidate in SUMMARY_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate full_v2_nomax summary run directory")


def build_failure_tasks(summary_dir: Path, pilot: int, shuffle_seed: int) -> List[dict]:
    tasks_path = summary_dir / "summarize_tasks.jsonl"
    all_tasks = list(iter_jsonl(tasks_path))
    failure_tasks = [t for t in all_tasks if t.get("task_type") == "failure"]
    if shuffle_seed >= 0:
        random.Random(shuffle_seed).shuffle(failure_tasks)
    if pilot > 0:
        failure_tasks = failure_tasks[:pilot]
    return failure_tasks


def stable_seed(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_success_counterexample_block(task: dict, max_chars: int = 2500) -> str:
    success_chain = str(task.get("paired_success_chain") or "").strip()
    if not success_chain:
        return ""
    success_model = str(task.get("paired_success_model") or "peer").strip() or "peer"
    return (
        f"[Successful Counterexample Chain: {success_model}]\n"
        + trim_text(success_chain, max_chars)
    )


def trim_text(text: str, max_chars: int) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len("\n... [truncated]"))
    return text[:keep].rstrip() + "\n... [truncated]"


def fit_prompt_to_budget(
    template: str,
    values: Dict[str, Any],
    *,
    max_prompt_chars: int,
    shrink_fields: List[tuple[str, int]],
) -> tuple[str, Dict[str, int]]:
    current = dict(values)
    prompt = template.format(**current)
    truncation_meta: Dict[str, int] = {}
    if len(prompt) <= max_prompt_chars:
        return prompt, truncation_meta

    for field, floor in shrink_fields:
        field_value = str(current.get(field, ""))
        if not field_value:
            continue
        while len(prompt) > max_prompt_chars and len(field_value) > floor:
            overflow = len(prompt) - max_prompt_chars
            target = max(floor, len(field_value) - overflow - 256)
            if target >= len(field_value):
                break
            field_value = trim_text(field_value, target)
            current[field] = field_value
            truncation_meta[field] = len(field_value)
            prompt = template.format(**current)
        if len(prompt) <= max_prompt_chars:
            break
    return prompt, truncation_meta


def render_tier_prompt(
    *,
    template: str,
    success_counterexample_block: str,
    question: str,
    lang: str,
    code: str,
    passed: int,
    total: int,
    error_dist: Dict[str, int],
    judge: dict,
    case_io: List[dict],
    key: str,
    chain: str,
    tier: str,
    max_prompt_chars: int,
) -> tuple[str, dict]:
    if tier == "A":
        budgets = (
            (30, 2000),
            (20, 1500),
            (12, 900),
            (8, 600),
            (5, 400),
        )
        last_prompt = ""
        last_meta = {}
        for max_cases, max_chars_per_field in budgets:
            failing_block = build_failing_cases_block(
                judge,
                case_io,
                max_cases=max_cases,
                max_chars_per_field=max_chars_per_field,
            )
            prompt, truncation_meta = fit_prompt_to_budget(
                template,
                {
                    "question": question,
                    "lang": lang,
                    "code": code,
                    "passed": passed,
                    "total": total,
                    "pass_pct": int(passed / max(total, 1) * 100),
                    "error_dist": error_dist,
                    "failing_cases_block": failing_block,
                    "success_counterexample_block": success_counterexample_block,
                },
                max_prompt_chars=max_prompt_chars,
                shrink_fields=[("code", 3000), ("success_counterexample_block", 900), ("question", 2000)],
            )
            last_prompt = prompt
            last_meta = {
                "max_cases": max_cases,
                "max_chars_per_field": max_chars_per_field,
                "truncation": truncation_meta,
            }
            if len(prompt) <= max_prompt_chars:
                return prompt, last_meta
        return last_prompt, last_meta

    budgets = (
        (8, 1500),
        (8, 900),
        (6, 700),
        (4, 500),
    )
    last_prompt = ""
    last_meta = {}
    for max_cases, max_chars_per_field in budgets:
        failing_block = build_selected_cases_block(
            judge,
            case_io,
            max_cases=max_cases,
            max_chars_per_field=max_chars_per_field,
            seed=stable_seed(key),
        )
        prompt, truncation_meta = fit_prompt_to_budget(
            template,
            {
                "question": question,
                "lang": lang,
                "code": code,
                "total": total,
                "error_dist": error_dist,
                "failing_cases_block": failing_block,
                "approach_summary": (chain[:2000] if chain else "(no reasoning chain)"),
                "success_counterexample_block": success_counterexample_block,
            },
            max_prompt_chars=max_prompt_chars,
            shrink_fields=[("code", 3000), ("success_counterexample_block", 900), ("approach_summary", 800), ("question", 2000)],
        )
        last_prompt = prompt
        last_meta = {
            "max_cases": max_cases,
            "max_chars_per_field": max_chars_per_field,
            "truncation": truncation_meta,
        }
        if len(prompt) <= max_prompt_chars:
            return prompt, last_meta
    return last_prompt, last_meta


def build_prompt_map(failure_tasks: List[dict], variant: str, max_prompt_chars: int) -> Dict[str, dict]:
    needed_keys = {t["unique_key"] for t in failure_tasks}
    all_case_io = load_all_case_io(CASES_PATH, needed_keys)
    prompt_map: Dict[str, dict] = {}

    for task in failure_tasks:
        task_id = task["task_id"]
        key = task["unique_key"]
        judge = task.get("failed_judge", {})
        chain = task.get("thinking_chain", "")
        lang, code = extract_code_from_text(chain)
        if not code.strip():
            code = "(no code found)"
            lang = "text"

        case_io = all_case_io.get(key, [])
        success_counterexample_block = build_success_counterexample_block(task)
        passed = int(judge.get("passed", 0) or 0)
        total = int(judge.get("total_tests", 0) or 0)
        case_results = judge.get("case_results", []) or []
        err_dist = Counter(
            (cr.get("status") or "?").upper()
            for cr in case_results
            if isinstance(cr, dict) and (cr.get("status") or "").upper() != "AC"
        )

        if passed > 0:
            tier = "A"
        else:
            tier = "B"
        template = TIER_A_PROMPTS[variant] if tier == "A" else TIER_B_PROMPTS[variant]
        prompt, budget_meta = render_tier_prompt(
            template=template,
            success_counterexample_block=success_counterexample_block,
            question=task["question"],
            lang=lang,
            code=code,
            passed=passed,
            total=total,
            error_dist=dict(err_dist),
            judge=judge,
            case_io=case_io,
            key=key,
            chain=chain,
            tier=tier,
            max_prompt_chars=max_prompt_chars,
        )

        prompt_map[task_id] = {
            "prompt": prompt,
            "tier": tier,
            "passed": passed,
            "total": total,
            "budget_profile": budget_meta,
        }
    return prompt_map


def parse_variant_arg(raw: str) -> List[str]:
    if raw == "all":
        return list(EXPECTED_VARIANTS)
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in variants if item not in EXPECTED_VARIANTS]
    if invalid:
        raise ValueError(f"Unknown variants: {invalid}")
    return variants


def run_variant_with_retries(
    client: QihooChatClient,
    *,
    prompt: str,
    model: str,
    max_retries: int,
    backoff: float,
    quality_gate: bool,
) -> dict:
    current_prompt = prompt
    last_err = None
    for attempt in range(max_retries):
        result = None
        result = client.chat(
            model=model,
            messages=[{"role": "user", "content": current_prompt}],
            timeout=900,
            max_retries=1,
            initial_retry_delay=backoff,
            base_payload=DEFAULT_SAMPLING_PAYLOAD,
        )
        if result is None:
            last_err = "api_call_failed"
        else:
            obj, parse_err = extract_json_obj(result.content)
            if parse_err:
                last_err = parse_err
            else:
                cards, valid_err = validate_v5_cards(obj)
                if not valid_err:
                    rejected_cards = []
                    quality_report = analyze_cards(cards)
                    if quality_gate:
                        cards, rejected_cards = split_cards_by_quality(cards)
                        if not cards:
                            last_err = "quality_gate:" + summarize_issues(rejected_cards)
                        else:
                            return {
                                "ok": True,
                                "attempts": attempt + 1,
                                "usage": result.usage,
                                "raw_content": result.content,
                                "cards": cards,
                                "rejected_cards": rejected_cards,
                                "quality_report": quality_report,
                                "error": None,
                            }
                    else:
                        return {
                            "ok": True,
                            "attempts": attempt + 1,
                            "usage": result.usage,
                            "raw_content": result.content,
                            "cards": cards,
                            "rejected_cards": rejected_cards,
                            "quality_report": quality_report,
                            "error": None,
                        }
                else:
                    last_err = valid_err

        if attempt + 1 < max_retries:
            current_prompt = (
                "Rewrite the output below as strict JSON only.\n"
                "Output exactly one JSON object with top-level key `cards`.\n"
                "Each card must include keys: type, trigger, tags, do, avoid, check.\n"
                "For edge_fix type: also include edge_pattern and fix_hint.\n"
                "For wrong_approach type: also include wrong_approach and correct_direction.\n"
                "Fix these issues before returning JSON: "
                + str(last_err)
                + ".\n"
                "Triggers must be problem-level and discriminative.\n"
                "Do not copy sample values, literal I/O pairs, or quoted example strings into triggers.\n"
                "Rewrite any example-specific trigger as an abstract structural condition.\n"
                "Remove judge words like WA/TLE/expected/got/passed/failed.\n"
                "Keep the card short and retrieval-ready.\n"
                "No markdown, no explanations.\n\n"
                "[YOUR PREVIOUS OUTPUT]\n"
                + (result.content if result else "")
            )
            time.sleep(backoff * (2 ** attempt))

    return {
        "ok": False,
        "attempts": max_retries,
        "usage": {},
        "raw_content": None,
        "cards": [],
        "rejected_cards": [],
        "quality_report": {},
        "error": last_err,
    }


def run_one_variant(
    *,
    variant: str,
    failure_tasks: List[dict],
    args: argparse.Namespace,
    summary_dir: Path,
) -> dict:
    out_dir = Path(args.output_dir) / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "prompt_manifest.jsonl"
    results_path = out_dir / "task_results.jsonl"
    cards_path = out_dir / "cards.jsonl"
    rejected_path = out_dir / "rejected_cards.jsonl"
    report_path = out_dir / "report.json"

    prompt_map = build_prompt_map(failure_tasks, variant, args.max_prompt_chars)

    if args.write_prompts_only:
        with manifest_path.open("w", encoding="utf-8") as f:
            for task in failure_tasks:
                meta = prompt_map[task["task_id"]]
                rec = {
                    "variant": variant,
                    "task_id": task["task_id"],
                    "unique_key": task["unique_key"],
                    "tier": meta["tier"],
                    "prompt_len": len(meta["prompt"]),
                    "budget_profile": meta["budget_profile"],
                    "prompt": meta["prompt"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tier_counter = Counter(meta["tier"] for meta in prompt_map.values())
        budget_counter = Counter(
            f"{meta['tier']}:cases={meta['budget_profile']['max_cases']},field_chars={meta['budget_profile']['max_chars_per_field']}"
            for meta in prompt_map.values()
        )
        prompt_lens = [len(meta["prompt"]) for meta in prompt_map.values()]
        report = {
            "variant": variant,
            "mode": "write_prompts_only",
            "summary_dir": str(summary_dir),
            "total_tasks": len(failure_tasks),
            "tier_distribution": dict(tier_counter),
            "budget_profiles": dict(budget_counter),
            "avg_prompt_len": round(sum(prompt_lens) / max(len(prompt_lens), 1), 2),
            "max_prompt_len": max(prompt_lens) if prompt_lens else 0,
            "max_prompt_chars": args.max_prompt_chars,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    api_key = resolve_api_key(args.api_key, required=True)
    client = QihooChatClient(api_key=api_key, api_url=args.url, api_host="api.360.cn")

    done_ids = set()
    if results_path.exists():
        for rec in iter_jsonl(results_path):
            if rec.get("ok"):
                done_ids.add(rec.get("task_id"))

    remaining = [task for task in failure_tasks if task["task_id"] not in done_ids]
    lock = threading.Lock()
    done_count = len(done_ids)

    def worker(task: dict) -> dict:
        nonlocal done_count
        meta = prompt_map[task["task_id"]]
        result = run_variant_with_retries(
            client,
            prompt=meta["prompt"],
            model=args.model,
            max_retries=args.max_retries,
            backoff=args.backoff,
            quality_gate=not args.disable_quality_gate,
        )
        result["task_id"] = task["task_id"]
        result["unique_key"] = task["unique_key"]
        result["tier"] = meta["tier"]
        result["variant"] = variant
        result["budget_profile"] = meta["budget_profile"]
        result["failure_origin"] = task.get("failure_origin")
        result["paired_success_model"] = task.get("paired_success_model")

        with lock:
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if result["ok"]:
                with cards_path.open("a", encoding="utf-8") as f:
                    for idx, card in enumerate(result["cards"]):
                        card_rec = {
                            "card_id": f"{task['task_id']}_{variant}_c{idx}",
                            "task_id": task["task_id"],
                            "problem_id": task["unique_key"],
                            "task_type": "failure",
                            "category": task["category"],
                            "variant": variant,
                            "tier": meta["tier"],
                            "failure_origin": task.get("failure_origin"),
                            "paired_success_model": task.get("paired_success_model"),
                            **card,
                        }
                        f.write(json.dumps(card_rec, ensure_ascii=False) + "\n")
                if result["rejected_cards"]:
                    with rejected_path.open("a", encoding="utf-8") as f:
                        for idx, card in enumerate(result["rejected_cards"]):
                            rejected_rec = {
                                "card_id": f"{task['task_id']}_{variant}_r{idx}",
                                "task_id": task["task_id"],
                                "problem_id": task["unique_key"],
                                "task_type": "failure",
                                "category": task["category"],
                                "variant": variant,
                                "tier": meta["tier"],
                                "failure_origin": task.get("failure_origin"),
                                "paired_success_model": task.get("paired_success_model"),
                                **card,
                            }
                            f.write(json.dumps(rejected_rec, ensure_ascii=False) + "\n")
            done_count += 1
            if done_count % 50 == 0 or done_count == len(failure_tasks):
                print(f"[{variant}] progress {done_count}/{len(failure_tasks)}")
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, task): task["task_id"] for task in remaining}
        for future in as_completed(futures):
            future.result()

    all_results = list(iter_jsonl(results_path))
    all_cards = list(iter_jsonl(cards_path)) if cards_path.exists() else []
    rejected_cards = list(iter_jsonl(rejected_path)) if rejected_path.exists() else []
    report = {
        "variant": variant,
        "mode": "api_run",
        "summary_dir": str(summary_dir),
        "total_tasks": len(failure_tasks),
        "ok": sum(1 for row in all_results if row.get("ok")),
        "failed": sum(1 for row in all_results if not row.get("ok")),
        "cards_total": len(all_cards),
        "rejected_cards_total": len(rejected_cards),
        "type_distribution": dict(Counter(card.get("type", "?") for card in all_cards)),
        "tier_distribution": dict(Counter(card.get("tier", "?") for card in all_cards)),
        "quality_gate_enabled": not args.disable_quality_gate,
        "quality_summary": analyze_cards(all_cards),
        "rejected_issue_summary": analyze_cards(rejected_cards) if rejected_cards else {},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple failure-card variants in parallel")
    parser.add_argument("--variants", default="all", help="Comma-separated variant list or 'all'")
    parser.add_argument("--pilot", type=int, default=0, help="Only process first N failure tasks")
    parser.add_argument("--summary_dir", default="", help="Optional explicit summary run directory")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Shuffle failure tasks before pilot slicing; set <0 to disable")
    parser.add_argument("--workers", type=int, default=16, help="Per-variant task workers")
    parser.add_argument("--variant_workers", type=int, default=3, help="Number of variants to execute in parallel")
    parser.add_argument("--write_prompts_only", action="store_true", help="Only write prompt manifests; no API calls")
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--backoff", type=float, default=5.0)
    parser.add_argument("--max_prompt_chars", type=int, default=DEFAULT_MAX_PROMPT_CHARS)
    parser.add_argument("--disable_quality_gate", action="store_true")
    args = parser.parse_args()

    clear_proxy_env()
    variants = parse_variant_arg(args.variants)
    summary_dir = resolve_summary_dir(args.summary_dir)
    failure_tasks = build_failure_tasks(summary_dir, args.pilot, args.shuffle_seed)
    if not failure_tasks:
        raise RuntimeError("No failure tasks found")

    print(f"summary_dir={summary_dir}")
    print(f"failure_tasks={len(failure_tasks)}")
    print(f"variants={variants}")
    print(f"mode={'write_prompts_only' if args.write_prompts_only else 'api_run'}")

    reports = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.variant_workers, len(variants)))) as pool:
        futures = {
            pool.submit(
                run_one_variant,
                variant=variant,
                failure_tasks=failure_tasks,
                args=args,
                summary_dir=summary_dir,
            ): variant
            for variant in variants
        }
        for future in as_completed(futures):
            variant = futures[future]
            report = future.result()
            reports.append(report)
            print(f"[done] {variant}: {report}")

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
