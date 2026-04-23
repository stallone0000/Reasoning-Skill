#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rm_policy_audit import find_hardcoded_keys
from rm_runtime import (
    API_KEY_ENV_VARS,
    MODEL_MAX_TOKENS,
    build_chat_payload,
    clear_proxy_env,
    find_reasoning_memory_root,
    resolve_api_key,
)


class RuntimeTests(unittest.TestCase):
    def test_clear_proxy_env_removes_all_supported_keys(self) -> None:
        original = {
            "http_proxy": "a",
            "https_proxy": "b",
            "all_proxy": "c",
            "HTTP_PROXY": "d",
            "HTTPS_PROXY": "e",
            "ALL_PROXY": "f",
            "OTHER": "keep",
        }
        env = dict(original)
        removed = clear_proxy_env(env)
        expected_removed = [key for key in original if key != "OTHER"]
        self.assertEqual(sorted(removed), sorted(expected_removed))
        self.assertEqual(env, {"OTHER": "keep"})

    def test_resolve_api_key_prefers_explicit_then_env(self) -> None:
        env = {"API_KEY": "from_env"}
        self.assertEqual(resolve_api_key(" from_arg ", env=env), "from_arg")
        self.assertEqual(resolve_api_key(None, env=env, env_vars=("API_KEY",)), "from_env")

    def test_resolve_api_key_respects_standard_env_priority(self) -> None:
        env = {
            "API_KEY": "fallback",
            "QIHOO_API_KEY": "secondary",
            "API_KEY_360": "primary",
        }
        self.assertEqual(resolve_api_key(None, env=env, env_vars=API_KEY_ENV_VARS), "primary")

    def test_build_chat_payload_only_forces_gptoss_default(self) -> None:
        base_messages = [{"role": "user", "content": "hi"}]
        gptoss_payload = build_chat_payload("qiniu/gpt-oss-120b", base_messages)
        gemini_payload = build_chat_payload("cloudsway/gemini-3-flash-preview", base_messages)

        self.assertEqual(gptoss_payload["max_tokens"], MODEL_MAX_TOKENS["qiniu/gpt-oss-120b"])
        self.assertNotIn("max_tokens", gemini_payload)

    def test_find_reasoning_memory_root(self) -> None:
        found = find_reasoning_memory_root(Path(__file__))
        self.assertEqual(found, ROOT)

    def test_active_tree_has_no_hardcoded_qihoo_api_keys(self) -> None:
        findings = find_hardcoded_keys(ROOT)
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
