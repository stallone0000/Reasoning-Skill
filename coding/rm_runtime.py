#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared runtime helpers for reasoning_memory scripts.

This module centralizes the project-wide 360 API rules:
1. Always clear proxy env vars before calling the API.
2. Resolve API keys from environment instead of hardcoding them.
3. Only force `max_tokens=32768` for `qiniu/gpt-oss-120b`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence
from urllib.parse import urlparse

import requests

PROXY_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)

API_KEY_ENV_VARS = (
    "API_KEY_360",
    "QINIU_API_KEY",
    "QIHOO_API_KEY",
    "API_KEY",
)

DEFAULT_CHAT_URL = os.environ.get(
    "QIHOO_API_URL",
    "http://api.zhinao.qihoo.net/v1/chat/completions",
)
DEFAULT_CHAT_HOST = os.environ.get("QIHOO_API_HOST", "api.360.cn")

DEFAULT_SAMPLING_PAYLOAD: Dict[str, Any] = {
    "content_filter": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
}

MODEL_MAX_TOKENS: Dict[str, int] = {
    "qiniu/gpt-oss-120b": 32768,
}


def find_reasoning_memory_root(anchor: Optional[Path | str] = None) -> Path:
    """Locate the reasoning_memory root directory from a file or directory anchor."""
    current = Path(anchor or __file__).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "LATEST_STATUS.md").exists() and (candidate / "README.md").exists():
            return candidate

    raise FileNotFoundError(f"Cannot locate reasoning_memory root from: {current}")


def clear_proxy_env(env: Optional[MutableMapping[str, str]] = None) -> list[str]:
    """Remove all proxy-related env vars and return the removed keys."""
    target = os.environ if env is None else env
    removed: list[str] = []
    for key in PROXY_ENV_KEYS:
        if key in target:
            removed.append(key)
            target.pop(key, None)
    return removed


def build_no_proxy_session() -> requests.Session:
    """Create a requests session that never reads system proxy config."""
    clear_proxy_env()
    session = requests.Session()
    session.trust_env = False
    return session


def resolve_api_key(
    explicit: Optional[str] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    env_vars: Sequence[str] = API_KEY_ENV_VARS,
    required: bool = False,
) -> str:
    """Resolve an API key from an explicit value or known environment variables."""
    if explicit and explicit.strip():
        return explicit.strip()

    source = os.environ if env is None else env
    for name in env_vars:
        value = str(source.get(name, "")).strip()
        if value:
            return value

    if required:
        joined = ", ".join(env_vars)
        raise RuntimeError(f"Missing API key. Set one of: {joined}")
    return ""


def resolve_api_url(explicit: Optional[str] = None) -> str:
    value = explicit or os.environ.get("QIHOO_API_URL") or DEFAULT_CHAT_URL
    return value.strip()


def resolve_api_host(explicit: Optional[str] = None, *, api_url: Optional[str] = None) -> str:
    value = explicit or os.environ.get("QIHOO_API_HOST") or DEFAULT_CHAT_HOST
    value = value.strip()
    if value:
        return value
    parsed = urlparse(api_url or DEFAULT_CHAT_URL)
    return parsed.netloc or DEFAULT_CHAT_HOST


def build_headers(api_key: str, *, host: Optional[str] = None) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8",
        "Host": resolve_api_host(host),
    }


def resolve_model_max_tokens(
    model: str,
    *,
    extra_map: Optional[Mapping[str, int]] = None,
    include_default_map: bool = True,
) -> Optional[int]:
    mapping: Dict[str, int] = {}
    if include_default_map:
        mapping.update(MODEL_MAX_TOKENS)
    if extra_map:
        mapping.update(extra_map)
    return mapping.get(model)


def build_chat_payload(
    model: str,
    messages: Iterable[Dict[str, Any]],
    *,
    base_payload: Optional[Mapping[str, Any]] = None,
    extra_payload: Optional[Mapping[str, Any]] = None,
    model_max_tokens: Optional[Mapping[str, int]] = None,
    include_default_model_max_tokens: bool = True,
) -> Dict[str, Any]:
    """Build a chat/completions payload with shared model-specific token policy."""
    payload: Dict[str, Any] = dict(base_payload or {})
    payload["model"] = model
    payload["messages"] = list(messages)

    max_tokens = resolve_model_max_tokens(
        model,
        extra_map=model_max_tokens,
        include_default_map=include_default_model_max_tokens,
    )
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if extra_payload:
        payload.update(extra_payload)
    return payload


def extract_message_text(response_json: Mapping[str, Any], *, include_reasoning: bool = False) -> str:
    """Extract `content` or `reasoning_content + content` from a response payload."""
    msg = response_json["choices"][0]["message"]
    content = msg.get("content", "") or ""
    if not include_reasoning:
        return content

    reasoning = msg.get("reasoning_content", "") or ""
    if reasoning.strip():
        return f"<think>\n{reasoning}\n</think>\n\n{content}"
    return content


@dataclass
class ChatResult:
    content: str
    usage: Dict[str, Any]
    raw_json: Dict[str, Any]


class QihooChatClient:
    """Small synchronous client for 360 chat/completions."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        api_host: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = resolve_api_key(api_key, required=True)
        self.api_url = resolve_api_url(api_url)
        self.api_host = resolve_api_host(api_host, api_url=self.api_url)
        self.session = session or build_no_proxy_session()

    def chat(
        self,
        *,
        model: str,
        messages: Iterable[Dict[str, Any]],
        timeout: float = 900,
        max_retries: int = 6,
        initial_retry_delay: float = 5,
        include_reasoning: bool = False,
        base_payload: Optional[Mapping[str, Any]] = None,
        extra_payload: Optional[Mapping[str, Any]] = None,
        model_max_tokens: Optional[Mapping[str, int]] = None,
        include_default_model_max_tokens: bool = True,
    ) -> Optional[ChatResult]:
        payload = build_chat_payload(
            model,
            messages,
            base_payload=base_payload,
            extra_payload=extra_payload,
            model_max_tokens=model_max_tokens,
            include_default_model_max_tokens=include_default_model_max_tokens,
        )
        headers = build_headers(self.api_key, host=self.api_host)
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    self.api_url,
                    headers=headers,
                    data=payload_bytes,
                    timeout=timeout,
                )
                response_json = response.json()
                if "error" in response_json:
                    print(f"[API ERROR] model={model}, attempt={attempt+1}: {response_json['error']}")
                else:
                    return ChatResult(
                        content=extract_message_text(response_json, include_reasoning=include_reasoning),
                        usage=dict(response_json.get("usage", {})),
                        raw_json=dict(response_json),
                    )
            except Exception as exc:
                print(f"[EXCEPTION] model={model}, attempt={attempt+1}: {exc}")

            if attempt + 1 < max_retries:
                delay = initial_retry_delay * (2 ** attempt)
                print(f"[RETRY] model={model}, attempt={attempt+1}, wait {delay:.0f}s")
                time.sleep(delay)

        return None

