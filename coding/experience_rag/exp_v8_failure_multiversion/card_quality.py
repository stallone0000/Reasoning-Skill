#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared quality heuristics for failure-card analysis and gating."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

ALLOWED_CP_TYPES = {"success", "failure", "contrast", "trap", "edge_fix", "wrong_approach"}
JUDGE_LEAK_RE = re.compile(
    r"\b(?:wa|tle|re|ce|wrong answer|runtime error|compile error|expected|got|stderr|judge|pass(?:ed)?\b|fail(?:ed)?\b)\b",
    re.IGNORECASE,
)
GENERIC_TRIGGER_RES = (
    re.compile(r"^(?:edge ?cases?|boundary cases?)$", re.IGNORECASE),
    re.compile(r"^(?:wrong answer|hidden tests?|fails? (?:some|all) tests?)$", re.IGNORECASE),
    re.compile(r"^(?:implementation bug|logic bug|buggy implementation)$", re.IGNORECASE),
    re.compile(r"^(?:be careful|check constraints|watch out)$", re.IGNORECASE),
)
LITERAL_EXAMPLE_RES = (
    re.compile(r"\b\d{3,}\b\s*(?:to|->|=>|→)\s*\b\d{3,}\b", re.IGNORECASE),
    re.compile(r"\b(?:sample|example)\b", re.IGNORECASE),
    re.compile(r"[\"'`][^\"'`\n]{1,24}[\"'`]"),
)


def as_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    return []


def card_text(card: dict) -> str:
    pieces: List[str] = []
    for key in (
        "type",
        "edge_pattern",
        "fix_hint",
        "wrong_approach",
        "correct_direction",
        "risk",
        "complexity",
    ):
        value = str(card.get(key, "")).strip()
        if value:
            pieces.append(value)
    for key in ("trigger", "tags", "do", "avoid", "check", "detours"):
        pieces.extend(as_list(card.get(key)))
    return "\n".join(pieces)


def is_type_specific_complete(card: dict) -> bool:
    ctype = str(card.get("type", "")).strip().lower()
    if ctype == "edge_fix":
        return bool(str(card.get("edge_pattern", "")).strip() and str(card.get("fix_hint", "")).strip())
    if ctype == "wrong_approach":
        return bool(str(card.get("wrong_approach", "")).strip() and str(card.get("correct_direction", "")).strip())
    return bool(as_list(card.get("do")) or as_list(card.get("check")))


def _is_low_signal_trigger(text: str) -> bool:
    text = " ".join(str(text).strip().split())
    if not text:
        return True
    if JUDGE_LEAK_RE.search(text):
        return True
    return any(pattern.fullmatch(text) for pattern in GENERIC_TRIGGER_RES)


def _looks_literal_example_trigger(text: str) -> bool:
    text = " ".join(str(text).strip().split())
    if not text:
        return False
    if any(pattern.search(text) for pattern in LITERAL_EXAMPLE_RES):
        return True

    long_numbers = re.findall(r"\b\d{3,}\b", text)
    if len(long_numbers) >= 2:
        return True

    if re.search(r"\b[A-Za-z_]\w*\s*=\s*\d+\b", text) and re.search(r"\b(?:to|->|=>|→)\b", text):
        return True

    return False


def card_issues(card: dict) -> List[str]:
    issues: List[str] = []
    ctype = str(card.get("type", "")).strip().lower()
    triggers = as_list(card.get("trigger"))
    trigger_lengths = [len(item) for item in triggers]
    full_text = card_text(card)

    if ctype not in ALLOWED_CP_TYPES:
        issues.append("unexpected_type")
    if not triggers:
        issues.append("trigger_empty")
    elif all(_is_low_signal_trigger(item) for item in triggers):
        issues.append("trigger_low_signal")
    if triggers and any(_looks_literal_example_trigger(item) for item in triggers):
        issues.append("trigger_literal_example")
    if trigger_lengths and max(trigger_lengths) > 120:
        issues.append("trigger_too_long")
    if JUDGE_LEAK_RE.search(full_text):
        issues.append("judge_leak")
    if not as_list(card.get("do")) and not as_list(card.get("check")):
        issues.append("action_gap")
    if not is_type_specific_complete(card):
        issues.append("type_specific_gap")

    key_fields = []
    for key in ("edge_pattern", "fix_hint", "wrong_approach", "correct_direction"):
        value = str(card.get(key, "")).strip()
        if value:
            key_fields.append(value)
    if len(full_text) > 900 or any(len(value) > 280 for value in key_fields):
        issues.append("long_form")

    return issues


def is_retrieval_ready(card: dict) -> bool:
    return not card_issues(card)


def split_cards_by_quality(cards: Iterable[dict]) -> Tuple[List[dict], List[dict]]:
    accepted: List[dict] = []
    rejected: List[dict] = []
    for card in cards:
        issues = card_issues(card)
        if issues:
            rejected.append({**card, "_quality_issues": issues})
        else:
            accepted.append(card)
    return accepted, rejected


def summarize_issues(rejected_cards: Iterable[dict], limit: int = 5) -> str:
    counter: Counter[str] = Counter()
    for card in rejected_cards:
        counter.update(as_list(card.get("_quality_issues")))
    parts = [f"{label}={count}" for label, count in counter.most_common(limit)]
    return ", ".join(parts) if parts else "unknown_quality_issue"


def analyze_cards(cards: List[dict]) -> Dict[str, object]:
    total = len(cards)
    type_counter = Counter(str(card.get("type", "?")).strip() or "?" for card in cards)
    all_trigger_items: List[str] = []
    issue_counter: Counter[str] = Counter()
    retrieval_ready = 0

    for card in cards:
        all_trigger_items.extend(as_list(card.get("trigger")))
        issues = card_issues(card)
        issue_counter.update(issues)
        if not issues:
            retrieval_ready += 1

    unique_trigger_ratio = round(len(set(all_trigger_items)) / max(len(all_trigger_items), 1), 4)
    return {
        "total_cards": total,
        "type_distribution": dict(type_counter),
        "unexpected_type_cards": issue_counter.get("unexpected_type", 0),
        "empty_trigger_cards": issue_counter.get("trigger_empty", 0),
        "low_signal_trigger_cards": issue_counter.get("trigger_low_signal", 0),
        "literal_example_trigger_cards": issue_counter.get("trigger_literal_example", 0),
        "trigger_too_long_cards": issue_counter.get("trigger_too_long", 0),
        "judge_leak_cards": issue_counter.get("judge_leak", 0),
        "action_gap_cards": issue_counter.get("action_gap", 0),
        "type_specific_gap_cards": issue_counter.get("type_specific_gap", 0),
        "long_form_cards": issue_counter.get("long_form", 0),
        "retrieval_ready_cards": retrieval_ready,
        "retrieval_ready_pct": round(retrieval_ready / max(total, 1), 4),
        "rejected_cards": total - retrieval_ready,
        "unique_trigger_ratio": unique_trigger_ratio,
        "issue_distribution": dict(issue_counter),
        "top_triggers": Counter(all_trigger_items).most_common(20),
    }
