#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_baseline_stats.py

统计 baseline 实验结果: completion tokens 和 AC 率。

输入:
  - gen_*.jsonl     : 生成输出 (含 completion_tokens)
  - judge_*.jsonl   : Judge 结果 (含 status: AC / WA / CE / ...)

输出:
  - 控制台表格 + baseline_stats.json

用法:
  # 统计所有结果
  python compute_baseline_stats.py

  # 指定目录
  python compute_baseline_stats.py --dir /path/to/baselines
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def compute_gen_stats(gen_path: str) -> Dict[str, Any]:
    """从 gen JSONL 计算 completion_tokens 统计。"""
    total_comp = 0
    total_prompt = 0
    count = 0
    empty = 0

    for row in iter_jsonl(gen_path):
        content = row.get("llm_output", "")
        if not content or not str(content).strip():
            empty += 1
            continue
        count += 1
        total_comp += row.get("completion_tokens", 0)
        total_prompt += row.get("prompt_tokens", 0)

    avg_comp = total_comp / count if count > 0 else 0
    avg_prompt = total_prompt / count if count > 0 else 0

    return {
        "gen_count": count,
        "gen_empty": empty,
        "total_completion_tokens": total_comp,
        "avg_completion_tokens": round(avg_comp, 1),
        "total_prompt_tokens": total_prompt,
        "avg_prompt_tokens": round(avg_prompt, 1),
    }


def compute_judge_stats(judge_path: str) -> Dict[str, Any]:
    """从 judge JSONL 计算 AC 率和状态分布。"""
    status_counter: Counter = Counter()
    total = 0

    for row in iter_jsonl(judge_path):
        judge = row.get("judge", {})
        status = judge.get("status", "UNKNOWN")
        status_counter[status] += 1
        total += 1

    ac_count = status_counter.get("AC", 0)
    ac_rate = ac_count / total if total > 0 else 0

    return {
        "judge_total": total,
        "ac_count": ac_count,
        "ac_rate": round(ac_rate, 4),
        "ac_rate_pct": f"{ac_rate*100:.2f}%",
        "status_distribution": dict(status_counter.most_common()),
    }


def parse_filename(filename: str):
    """
    从文件名解析 (type, prompt, model_short)。
    例如: gen_bare_oss120b.jsonl → ('gen', 'bare', 'oss120b')
          judge_nowait_doubao.jsonl → ('judge', 'nowait', 'doubao')
    """
    stem = filename.replace(".jsonl", "")
    parts = stem.split("_", 2)
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Baseline 统计: completion tokens + AC 率")
    parser.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="baselines 目录")
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    basedir = args.dir
    print(f"扫描目录: {basedir}\n")

    # 收集所有 gen 和 judge 文件
    gen_files: Dict[str, str] = {}   # key: "prompt_model" → path
    judge_files: Dict[str, str] = {}

    for fname in sorted(os.listdir(basedir)):
        if not fname.endswith(".jsonl"):
            continue
        ftype, prompt, model_short = parse_filename(fname)
        if ftype is None:
            continue
        key = f"{prompt}_{model_short}"
        full = os.path.join(basedir, fname)
        if ftype == "gen":
            gen_files[key] = full
        elif ftype == "judge":
            judge_files[key] = full

    # 定义展示顺序
    models_order = ["oss120b", "doubao", "gemini"]
    prompts_order = ["direct", "nowait", "cod", "tale"]
    model_names = {
        "oss120b": "qiniu/gpt-oss-120b",
        "doubao": "volcengine/doubao-seed-2-0-pro",
        "gemini": "cloudsway/gemini-3-flash-preview",
    }

    all_stats: Dict[str, Dict] = {}

    print(f"{'Experiment':<28s} {'Gen':>6s} {'Avg CompTok':>12s} {'Judge':>6s} {'AC':>6s} {'AC Rate':>8s}")
    print("-" * 72)

    for prompt in prompts_order:
        for model_short in models_order:
            key = f"{prompt}_{model_short}"
            label = f"{prompt}/{model_short}"

            stats: Dict[str, Any] = {
                "prompt": prompt,
                "model_short": model_short,
                "model": model_names.get(model_short, model_short),
            }

            # Gen stats
            if key in gen_files:
                gs = compute_gen_stats(gen_files[key])
                stats.update(gs)
            else:
                gs = {}

            # Judge stats
            if key in judge_files:
                js = compute_judge_stats(judge_files[key])
                stats.update(js)
            else:
                js = {}

            all_stats[key] = stats

            gen_n = gs.get("gen_count", "-")
            avg_ct = gs.get("avg_completion_tokens", "-")
            judge_n = js.get("judge_total", "-")
            ac_n = js.get("ac_count", "-")
            ac_r = js.get("ac_rate_pct", "-")

            if isinstance(avg_ct, (int, float)):
                avg_ct_str = f"{avg_ct:,.0f}"
            else:
                avg_ct_str = str(avg_ct)

            print(f"{label:<28s} {str(gen_n):>6s} {avg_ct_str:>12s} {str(judge_n):>6s} {str(ac_n):>6s} {str(ac_r):>8s}")

    print("-" * 72)

    # 保存
    output_path = args.output or os.path.join(basedir, "baseline_stats.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n统计结果已保存: {output_path}")

    # 额外输出 Markdown 表格
    md_path = os.path.join(basedir, "baseline_stats.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# CP Baseline 实验结果\n\n")
        f.write(f"统计时间: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Prompt | Model | Gen Count | Avg Completion Tokens | Judge Count | AC Count | AC Rate |\n")
        f.write("|--------|-------|-----------|-----------------------|-------------|----------|---------|\n")
        for prompt in prompts_order:
            for model_short in models_order:
                key = f"{prompt}_{model_short}"
                s = all_stats.get(key, {})
                f.write(
                    f"| {prompt} | {model_short} "
                    f"| {s.get('gen_count', '-')} "
                    f"| {s.get('avg_completion_tokens', '-')} "
                    f"| {s.get('judge_total', '-')} "
                    f"| {s.get('ac_count', '-')} "
                    f"| {s.get('ac_rate_pct', '-')} |\n"
                )
    print(f"Markdown 表格已保存: {md_path}")


if __name__ == "__main__":
    main()
