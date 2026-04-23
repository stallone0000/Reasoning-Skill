#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import gzip
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import aiohttp


DEFAULT_API_BASE_URL = "https://api.openai.com/v1/chat/completions"
API_KEY_ENV = "TRS_API_KEY"
API_BASE_URL_ENV = "TRS_API_BASE_URL"
HTTP_PROXY_ENV = "TRS_HTTP_PROXY"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if "question_id" not in row:
                row["question_id"] = row.get("id", f"q_{idx}")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content)


def render_prompt(
    template: str,
    *,
    problem: str = "",
    trace: str = "",
    answer: str = "",
    hints: str = "",
) -> str:
    prompt = template
    replacements = {
        "[INSERT PROBLEM HERE]": problem,
        "[INSERT CHAIN OF THOUGHT HERE]": trace,
        "[INSERT CORRECT ANSWER HERE]": answer,
        "{PROBLEM}": problem,
        "{SOLVING_HINTS}": hints,
        "{TRACE}": trace,
        "{ANSWER}": answer,
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def require_api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is empty. Set it for online runs or pass --mock for an offline smoke test."
        )
    return key


def api_base_url() -> str:
    return os.environ.get(API_BASE_URL_ENV, DEFAULT_API_BASE_URL).strip() or DEFAULT_API_BASE_URL


def make_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {require_api_key()}",
        "Content-Type": "application/json; charset=utf-8",
    }


async def call_chat_completion(
    session: aiohttp.ClientSession,
    prompt: str,
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout_tag: str = "trs_release",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
        "user": timeout_tag,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens

    proxy = os.environ.get(HTTP_PROXY_ENV, "").strip() or None
    async with session.post(api_base_url(), headers=make_headers(), json=payload, proxy=proxy) as response:
        body_text = await response.text()
        if response.status >= 400:
            return {"status": "error", "error": f"HTTP {response.status}: {body_text}"}
        try:
            return {"status": "success", "response": json.loads(body_text)}
        except json.JSONDecodeError:
            return {"status": "error", "error": f"non-json response: {body_text[:500]}"}


def response_message(response_payload: dict[str, Any]) -> dict[str, Any]:
    choices = response_payload.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def response_texts(response_payload: dict[str, Any]) -> tuple[str, str]:
    message = response_message(response_payload)
    visible = normalize_text_content(message.get("content", ""))
    reasoning = normalize_text_content(
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("thinking")
        or ""
    )
    return reasoning, visible


def usage_counts(response_payload: dict[str, Any], *, prompt_text: str = "", response_text: str = "") -> dict[str, int]:
    usage = response_payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or estimate_tokens(prompt_text))
    completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(response_text))
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def build_reasoning_trace(model_think: str, model_response: str) -> str:
    model_think = model_think.strip()
    model_response = model_response.strip()
    if model_think and model_response:
        return f"<think>\n{model_think}\n</think>\n\n{model_response}"
    return model_think or model_response


def extract_xml_content(text: str, tag: str) -> str:
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def mock_source_response(question: str, answer: str) -> tuple[str, str]:
    short_question = re.sub(r"\s+", " ", question).strip()
    model_think = (
        "Identify the structure of the problem, translate it into equations or algebraic operations, "
        "solve for the requested quantity, and check that the final form matches the question."
    )
    model_response = (
        f"For the problem: {short_question}\n"
        f"A compact solution follows the relevant algebraic structure and gives the final answer {answer}.\n\n"
        f"Final answer: {answer}"
    )
    return model_think, model_response


def mock_skill_response(question: str, trace: str, answer: str) -> str:
    keywords = ", ".join(sorted(set(tokenize(question)))[:14])
    return f"""<learned_heuristic>
1. When encountering a problem with this surface form, first identify the invariant or algebraic structure because it determines the shortest path to the requested quantity. Be cautious of reporting an intermediate variable instead of the final requested answer.
2. When a trace reaches the correct answer, preserve the transferable operation sequence rather than the specific constants because future retrieval should generalize across similar instances.
3. Before finalizing, compare the output form against the prompt and the known answer because answer-format mismatches can look like reasoning errors.
</learned_heuristic>

<retrieval_keywords>
{keywords}
</retrieval_keywords>""".strip()


def mock_summary(question: str, trace: str, answer: str) -> str:
    return (
        "Identify the reusable structure in the problem, solve only for the requested quantity, "
        f"and verify that the final answer format matches the target answer ({answer})."
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


class BM25:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.tokenized = [tokenize(doc) for doc in documents]
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in self.tokenized]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.idf: dict[str, float] = {}
        self.freqs: list[dict[str, int]] = []
        self._index()

    def _index(self) -> None:
        doc_counts: dict[str, int] = {}
        for doc in self.tokenized:
            freqs: dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            self.freqs.append(freqs)
            for token in freqs:
                doc_counts[token] = doc_counts.get(token, 0) + 1
        n_docs = max(len(self.tokenized), 1)
        for token, count in doc_counts.items():
            self.idf[token] = math.log(1 + (n_docs - count + 0.5) / (count + 0.5))

    def scores(self, query: str) -> list[float]:
        query_tokens = tokenize(query)
        out: list[float] = []
        for idx, freqs in enumerate(self.freqs):
            score = 0.0
            doc_len = self.doc_len[idx] or 1
            for token in query_tokens:
                freq = freqs.get(token, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += self.idf.get(token, 0.0) * (freq * (self.k1 + 1) / max(denom, 1e-9))
            out.append(score)
        return out

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        scored = [(idx, score) for idx, score in enumerate(self.scores(query)) if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


def build_retrieval_document(row: dict[str, Any]) -> str:
    structured_parts = []
    for name in ["trigger", "tags", "do", "avoid", "check", "risk", "complexity"]:
        value = row.get(name)
        if isinstance(value, list):
            structured_parts.append(" ".join(str(item) for item in value))
        elif value:
            structured_parts.append(str(value))
    parts = [
        row.get("question", ""),
        row.get("topic", ""),
        row.get("heuristic", ""),
        row.get("skill_text", ""),
        row.get("inject_text", ""),
        row.get("keywords", ""),
        row.get("free_summary", ""),
        " ".join(structured_parts),
    ]
    return "\n".join(str(part).strip() for part in parts if str(part).strip())


def find_last_boxed(text: str) -> str:
    marker = "\\boxed"
    idx = text.rfind(marker)
    if idx < 0:
        return ""
    brace_idx = text.find("{", idx)
    if brace_idx < 0:
        return ""
    depth = 0
    chars: list[str] = []
    for ch in text[brace_idx:]:
        if ch == "{":
            depth += 1
            if depth > 1:
                chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(ch)
        else:
            chars.append(ch)
    return ""


def extract_final_answer(text: str) -> str:
    boxed = find_last_boxed(text)
    if boxed:
        return boxed
    patterns = [
        r"(?im)^\s*final\s+answer\s*[:：]\s*(.+?)\s*$",
        r"(?im)^\s*answer\s*[:：]\s*(.+?)\s*$",
        r"(?is)\bthe\s+final\s+answer\s+is\s+(.+?)(?:\n|$)",
        r"(?is)\bthe\s+answer\s+is\s+(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            return matches[-1].group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\\boxed\s*{([^{}]*)}", r"\1", text)
    text = re.sub(r"\\text\s*{([^{}]*)}", r"\1", text)
    for token in ["$", "\\(", "\\)", "\\[", "\\]", "\\left", "\\right", "`", "*"]:
        text = text.replace(token, "")
    text = text.replace(" ", "")
    text = text.rstrip(".;,")
    return text


def answers_match(candidate: str, gold: str) -> bool:
    candidate_norm = normalize_answer(candidate)
    gold_norm = normalize_answer(gold)
    if not candidate_norm or not gold_norm:
        return False
    return candidate_norm == gold_norm
