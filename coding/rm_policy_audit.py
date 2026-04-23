#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit active reasoning_memory files for hardcoded 360 API keys."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

from rm_runtime import find_reasoning_memory_root

HARD_CODED_QIHOO_KEY_RE = re.compile(r"fk\d+\.[A-Za-z0-9_\-]+")
SCAN_EXTENSIONS = {".py", ".sh", ".md", ".txt"}
ACTIVE_TOP_LEVELS = ("baselines", "docs", "experience_rag", "experiments", "scripts")
SKIP_DIRS = {"archive", "data", "history", "logs", "notebooks"}


def iter_active_files(root: Path) -> Iterable[Path]:
    for name in ACTIVE_TOP_LEVELS:
        base = root / name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SCAN_EXTENSIONS:
                continue
            if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
                continue
            yield path


def find_hardcoded_keys(root: Path) -> list[tuple[Path, int, str]]:
    findings: list[tuple[Path, int, str]] = []
    for path in iter_active_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if HARD_CODED_QIHOO_KEY_RE.search(line):
                findings.append((path, lineno, line.strip()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit reasoning_memory for hardcoded 360 API keys")
    parser.add_argument(
        "--root",
        default=None,
        help="Optional reasoning_memory root path",
    )
    args = parser.parse_args()

    root = find_reasoning_memory_root(args.root)
    findings = find_hardcoded_keys(root)
    if findings:
        print("Found hardcoded 360 API keys:")
        for path, lineno, line in findings:
            print(f"{path}:{lineno}: {line}")
        return 1

    print(f"No hardcoded 360 API keys found in active files under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

