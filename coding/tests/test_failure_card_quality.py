#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUALITY_ROOT = ROOT / "experience_rag" / "exp_v8_failure_multiversion"
for candidate in (ROOT, QUALITY_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from card_quality import analyze_cards, card_issues, split_cards_by_quality


class FailureCardQualityTests(unittest.TestCase):
    def test_judge_leak_and_low_signal_trigger_are_rejected(self) -> None:
        card = {
            "type": "edge_fix",
            "trigger": ["Wrong answer on hidden tests"],
            "tags": ["implementation"],
            "edge_pattern": "single-item input",
            "fix_hint": "handle the base case first",
            "do": ["trace the base case"],
            "avoid": ["assuming the loop runs twice"],
            "check": ["test N=1"],
        }
        issues = card_issues(card)
        self.assertIn("judge_leak", issues)
        self.assertIn("trigger_low_signal", issues)

    def test_clean_retrieval_ready_card_passes(self) -> None:
        card = {
            "type": "wrong_approach",
            "trigger": ["global parity constraint with local greedy temptation"],
            "tags": ["greedy", "parity"],
            "wrong_approach": "commit locally without preserving future parity feasibility",
            "correct_direction": "track parity feasibility before locking each move",
            "do": ["build a two-step counterexample", "verify parity after each transition"],
            "avoid": ["greedy commitment without feasibility state"],
            "check": ["can a locally best move make the remaining parity impossible?"],
        }
        accepted, rejected = split_cards_by_quality([card])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])

    def test_literal_example_trigger_is_rejected(self) -> None:
        card = {
            "type": "wrong_approach",
            "trigger": ["45654 to 35753"],
            "tags": ["graph"],
            "wrong_approach": "memorize the sample path instead of the target constraint",
            "correct_direction": "rewrite the trigger as the structural condition that made the sample fail",
            "do": ["replace sample values with the underlying invariant"],
            "avoid": ["copying literal sample numbers into retrieval keys"],
            "check": ["would this still be useful on a different input instance?"],
        }
        issues = card_issues(card)
        self.assertIn("trigger_literal_example", issues)

    def test_analyze_cards_reports_issue_breakdown(self) -> None:
        cards = [
            {
                "type": "edge_fix",
                "trigger": ["edge case"],
                "tags": ["math"],
                "edge_pattern": "overflow on large product",
                "fix_hint": "cast before multiplication",
                "do": [],
                "avoid": [],
                "check": [],
            },
            {
                "type": "edge_fix",
                "trigger": ["prefix sum with negative updates"],
                "tags": ["prefix-sum"],
                "edge_pattern": "negative update can break monotonic assumption",
                "fix_hint": "store the true signed delta before accumulation",
                "do": ["replay one negative update by hand"],
                "avoid": ["assuming deltas stay non-negative"],
                "check": ["run the smallest negative-update case"],
            },
        ]
        report = analyze_cards(cards)
        self.assertEqual(report["total_cards"], 2)
        self.assertEqual(report["retrieval_ready_cards"], 1)
        self.assertEqual(report["low_signal_trigger_cards"], 1)
        self.assertEqual(report["action_gap_cards"], 1)
        self.assertEqual(report["issue_distribution"]["trigger_low_signal"], 1)
        self.assertEqual(report["literal_example_trigger_cards"], 0)


if __name__ == "__main__":
    unittest.main()
