#!/usr/bin/env python3
"""
analyze_results.py

Comprehensive analysis of Experience-RAG experiments.
Compares our RAG methods (bm25-5, hybrid_m3-5) against baselines
across 3 models (gemini, doubao, gptoss).

Outputs:
  - Main comparison table (AC Rate, Avg CT, Tokens/AC)
  - Per-category breakdown
  - Error distribution
  - Bootstrap confidence intervals
  - Markdown + console output
"""

import json
import os
import sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_DIR = Path(__file__).resolve().parent / "data"

# ── Baseline results (from user, temperature=1.0, top_p=1.0, top_k=0) ──
BASELINE_RESULTS = {
    ("direct", "oss120b"):  {"n_done": 983,  "avg_ct": 3988,  "ac_rate": 0.4873},
    ("direct", "doubao"):   {"n_done": 1000, "avg_ct": 7500,  "ac_rate": 0.6360},
    ("direct", "gemini"):   {"n_done": 1000, "avg_ct": 20206, "ac_rate": 0.7200},
    ("nowait", "oss120b"):  {"n_done": 990,  "avg_ct": 4075,  "ac_rate": 0.5333},
    ("nowait", "doubao"):   {"n_done": 1000, "avg_ct": 6902,  "ac_rate": 0.6300},
    ("nowait", "gemini"):   {"n_done": 1000, "avg_ct": 17361, "ac_rate": 0.7210},
    ("cod", "oss120b"):     {"n_done": 989,  "avg_ct": 3512,  "ac_rate": 0.4671},
    ("cod", "doubao"):      {"n_done": 1000, "avg_ct": 6197,  "ac_rate": 0.5990},
    ("cod", "gemini"):      {"n_done": 1000, "avg_ct": 15799, "ac_rate": 0.6780},
    ("tale", "oss120b"):    {"n_done": 994,  "avg_ct": 3893,  "ac_rate": 0.4457},
    ("tale", "doubao"):     {"n_done": 1000, "avg_ct": 10015, "ac_rate": 0.6260},
    ("tale", "gemini"):     {"n_done": 1000, "avg_ct": 16777, "ac_rate": 0.6850},
}

MODEL_SHORT = {
    "cloudsway/gemini-3-flash-preview": "gemini",
    "volcengine/doubao-seed-2-0-pro": "doubao",
    "qiniu/gpt-oss-120b": "oss120b",
}


def load_gens(tag: str):
    """Load generation results. Returns list of dicts."""
    path = RESULTS_DIR / f"gens_{tag}.jsonl"
    if not path.exists():
        return []
    results = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_judge(tag: str):
    """Load judge results. Returns dict: unique_key -> judge_result."""
    path = RESULTS_DIR / f"judge_{tag}.jsonl"
    if not path.exists():
        return {}
    results = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                results[d["unique_key"]] = d
    return results


def load_categories():
    """Load category mapping for holdout keys."""
    path = DATA_DIR / "holdout_categories.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {}


def bootstrap_ci(values, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for mean."""
    if len(values) == 0:
        return 0, 0, 0
    values = np.array(values)
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return np.mean(values), lower, upper


def analyze_experiment(tag: str, categories: dict):
    """Analyze one experiment, return summary dict."""
    gens = load_gens(tag)
    judges = load_judge(tag)
    
    if not gens:
        return None
    
    ok_gens = [g for g in gens if g.get("status") == "OK"]
    n_total = len(ok_gens)
    n_api_fail = len(gens) - len(ok_gens)
    
    # Token stats
    cts = []
    pts = []
    tts = []
    for g in ok_gens:
        u = g.get("usage", {})
        ct = u.get("completion_tokens", 0)
        pt = u.get("prompt_tokens", 0)
        tt = u.get("total_tokens", ct + pt)
        cts.append(ct)
        pts.append(pt)
        tts.append(tt)
    
    # Judge stats
    status_counter = Counter()
    ac_per_cat = Counter()
    total_per_cat = Counter()
    ac_cts = []
    
    for g in ok_gens:
        key = g["unique_key"]
        jr = judges.get(key, {})
        status = jr.get("status", "NOT_JUDGED")
        status_counter[status] += 1
        
        cat = categories.get(key, "unknown")
        total_per_cat[cat] += 1
        if status == "AC":
            ac_per_cat[cat] += 1
            u = g.get("usage", {})
            ac_cts.append(u.get("completion_tokens", 0))
    
    n_ac = status_counter.get("AC", 0)
    ac_rate = n_ac / n_total if n_total > 0 else 0
    
    # Bootstrap CI for AC rate
    ac_binary = [1 if judges.get(g["unique_key"], {}).get("status") == "AC" else 0 
                 for g in ok_gens]
    ac_mean, ac_lower, ac_upper = bootstrap_ci(ac_binary)
    
    return {
        "tag": tag,
        "n_total": n_total,
        "n_api_fail": n_api_fail,
        "n_ac": n_ac,
        "ac_rate": ac_rate,
        "ac_ci_lower": ac_lower,
        "ac_ci_upper": ac_upper,
        "avg_ct": np.mean(cts) if cts else 0,
        "median_ct": np.median(cts) if cts else 0,
        "avg_pt": np.mean(pts) if pts else 0,
        "avg_tt": np.mean(tts) if tts else 0,
        "tokens_per_ac": sum(tts) / n_ac if n_ac > 0 else None,
        "status_dist": dict(status_counter),
        "ac_per_cat": dict(ac_per_cat),
        "total_per_cat": dict(total_per_cat),
        "avg_ac_ct": np.mean(ac_cts) if ac_cts else 0,
    }


def print_main_table(experiments, baselines):
    """Print the main comparison table."""
    print("\n" + "=" * 100)
    print("  RAG Experiments vs Baselines — Main Results (1000 holdout problems)")
    print("=" * 100)
    
    # Group by model
    # Note: experiment tags use "gptoss" but baselines use "oss120b"
    models = ["gemini", "doubao", "gptoss"]
    model_labels = {"gemini": "Gemini Flash", "doubao": "Doubao Seed 2.0", "gptoss": "GPT-OSS 120B"}
    # Map our model names to baseline model names
    model_to_baseline = {"gemini": "gemini", "doubao": "doubao", "gptoss": "oss120b"}
    
    for model in models:
        print(f"\n{'─' * 90}")
        print(f"  Model: {model_labels[model]}")
        print(f"{'─' * 90}")
        print(f"  {'Method':<22} {'Done':>5} {'AC':>5} {'AC Rate':>10} {'95% CI':>16} {'Avg CT':>8} {'Avg TT':>8} {'Tok/AC':>8}")
        print(f"  {'─'*22} {'─'*5} {'─'*5} {'─'*10} {'─'*16} {'─'*8} {'─'*8} {'─'*8}")
        
        # Baselines
        bl_model = model_to_baseline.get(model, model)
        for prompt in ["direct", "nowait", "cod", "tale"]:
            key = (prompt, bl_model)
            b = baselines.get(key)
            if b:
                print(f"  {prompt + ' (baseline)':<22} {b['n_done']:>5} "
                      f"{'—':>5} {b['ac_rate']:>9.1%} {'—':>16} {b['avg_ct']:>8,} {'—':>8} {'—':>8}")
        
        # Our experiments
        for exp_name in ["bm25-5", "hybrid_m3-5"]:
            tag = f"{exp_name}_{model}"
            if tag in experiments and experiments[tag] is not None:
                e = experiments[tag]
                ci_str = f"[{e['ac_ci_lower']:.1%},{e['ac_ci_upper']:.1%}]"
                tok_ac = f"{e['tokens_per_ac']:>8,.0f}" if e['tokens_per_ac'] else "—"
                
                # Compare with direct baseline
                direct_b = baselines.get(("direct", bl_model))
                delta = ""
                if direct_b:
                    d_ac = e['ac_rate'] - direct_b['ac_rate']
                    d_ct = e['avg_ct'] - direct_b['avg_ct']
                    delta_ac = f"({'+'if d_ac>=0 else ''}{d_ac:.1%})"
                    delta_ct = f"({'+'if d_ct>=0 else ''}{d_ct:,.0f})"
                
                label = f"Ours: {exp_name}"
                print(f"  {label:<22} {e['n_total']:>5} {e['n_ac']:>5} "
                      f"{e['ac_rate']:>9.1%} {ci_str:>16} {e['avg_ct']:>8,.0f} "
                      f"{e['avg_tt']:>8,.0f} {tok_ac}")
                if direct_b:
                    print(f"  {'  vs direct baseline':<22} {'':>5} {'':>5} "
                          f"{delta_ac:>10} {'':>16} {delta_ct:>8}")
    
    print()


def print_category_table(experiments, categories):
    """Print per-category AC rates."""
    print("\n" + "=" * 100)
    print("  Per-Category AC Rate Breakdown")
    print("=" * 100)
    
    cat_order = ["both_ac", "flash_ac_doubao_wrong", "flash_wrong_doubao_ac", "both_wrong"]
    cat_labels = {
        "both_ac": "Both AC (easy)",
        "flash_ac_doubao_wrong": "Flash AC / Doubao Wrong",
        "flash_wrong_doubao_ac": "Flash Wrong / Doubao AC",
        "both_wrong": "Both Wrong (hard)",
    }
    
    # Count per category
    cat_counts = Counter(categories.values())
    
    models = ["gemini", "doubao", "gptoss"]
    for model in models:
        print(f"\n  Model: {model}")
        header = f"  {'Category':<28} {'N':>5}"
        for exp in ["bm25-5", "hybrid_m3-5"]:
            header += f"  {exp:>15}"
        print(header)
        print(f"  {'─'*28} {'─'*5}" + f"  {'─'*15}" * 2)
        
        for cat in cat_order:
            n = cat_counts.get(cat, 0)
            row = f"  {cat_labels.get(cat, cat):<28} {n:>5}"
            for exp in ["bm25-5", "hybrid_m3-5"]:
                tag = f"{exp}_{model}"
                if tag in experiments and experiments[tag]:
                    e = experiments[tag]
                    ac_cat = e["ac_per_cat"].get(cat, 0)
                    tot_cat = e["total_per_cat"].get(cat, 0)
                    rate = ac_cat / tot_cat if tot_cat > 0 else 0
                    row += f"  {ac_cat:>3}/{tot_cat:<3} ({rate:>5.1%})"
                else:
                    row += f"  {'—':>15}"
            print(row)
    print()


def print_error_distribution(experiments):
    """Print error distribution for each experiment."""
    print("\n" + "=" * 100)
    print("  Error Distribution (Judge Status)")
    print("=" * 100)
    
    all_statuses = set()
    for tag, e in experiments.items():
        if e:
            all_statuses.update(e["status_dist"].keys())
    
    status_order = ["AC", "WA", "TLE", "RE", "CE", "WA_CHECKER", "NO_OUTPUT", "JUDGE_ERROR", "NO_CASES", "NOT_JUDGED"]
    status_order = [s for s in status_order if s in all_statuses]
    other_statuses = sorted(all_statuses - set(status_order))
    status_order += other_statuses
    
    header = f"  {'Experiment':<25}"
    for s in status_order:
        header += f" {s:>8}"
    print(header)
    print(f"  {'─'*25}" + f" {'─'*8}" * len(status_order))
    
    for tag in sorted(experiments.keys()):
        e = experiments[tag]
        if not e:
            continue
        row = f"  {tag:<25}"
        for s in status_order:
            n = e["status_dist"].get(s, 0)
            row += f" {n:>8}"
        print(row)
    print()


def print_token_analysis(experiments):
    """Print detailed token analysis."""
    print("\n" + "=" * 100)
    print("  Token Usage Analysis")
    print("=" * 100)
    
    print(f"\n  {'Experiment':<25} {'Avg PT':>8} {'Avg CT':>8} {'Med CT':>8} {'Avg TT':>8} {'Avg AC CT':>10}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    
    for tag in sorted(experiments.keys()):
        e = experiments[tag]
        if not e:
            continue
        print(f"  {tag:<25} {e['avg_pt']:>8,.0f} {e['avg_ct']:>8,.0f} "
              f"{e['median_ct']:>8,.0f} {e['avg_tt']:>8,.0f} {e['avg_ac_ct']:>10,.0f}")
    print()


def generate_markdown_report(experiments, baselines, categories):
    """Generate a markdown report file."""
    report_path = RESULTS_DIR / "experiment_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Experience-RAG 实验结果报告\n\n")
        f.write(f"> 测试集: 1000 holdout problems (分层抽样)\n")
        f.write(f"> Category: both_ac={Counter(categories.values())['both_ac']}, "
                f"flash_ac_doubao_wrong={Counter(categories.values()).get('flash_ac_doubao_wrong',0)}, "
                f"flash_wrong_doubao_ac={Counter(categories.values()).get('flash_wrong_doubao_ac',0)}, "
                f"both_wrong={Counter(categories.values()).get('both_wrong',0)}\n\n")
        
        # Main results table
        f.write("## 1. 主结果对比\n\n")
        
        models = ["gemini", "doubao", "gptoss"]
        model_labels = {"gemini": "Gemini Flash", "doubao": "Doubao Seed 2.0", "gptoss": "GPT-OSS 120B"}
        model_to_baseline = {"gemini": "gemini", "doubao": "doubao", "gptoss": "oss120b"}
        
        for model in models:
            bl_model = model_to_baseline.get(model, model)
            f.write(f"\n### {model_labels[model]}\n\n")
            f.write("| Method | Done | AC | AC Rate | Δ vs direct | Avg CT | Δ CT |\n")
            f.write("|--------|------|----|---------|-------------|--------|------|\n")
            
            direct_b = baselines.get(("direct", bl_model))
            
            for prompt in ["direct", "nowait", "cod", "tale"]:
                b = baselines.get((prompt, bl_model))
                if b:
                    d_ac = ""
                    d_ct = ""
                    if direct_b and prompt != "direct":
                        da = b['ac_rate'] - direct_b['ac_rate']
                        dc = b['avg_ct'] - direct_b['avg_ct']
                        d_ac = f"{'+'if da>=0 else ''}{da:.1%}"
                        d_ct = f"{'+'if dc>=0 else ''}{dc:,}"
                    f.write(f"| {prompt} (baseline) | {b['n_done']} | — | {b['ac_rate']:.1%} | {d_ac} | {b['avg_ct']:,} | {d_ct} |\n")
            
            for exp_name in ["bm25-5", "hybrid_m3-5"]:
                tag = f"{exp_name}_{model}"
                if tag in experiments and experiments[tag]:
                    e = experiments[tag]
                    d_ac = ""
                    d_ct = ""
                    if direct_b:
                        da = e['ac_rate'] - direct_b['ac_rate']
                        dc = e['avg_ct'] - direct_b['avg_ct']
                        d_ac = f"{'+'if da>=0 else ''}{da:.1%}"
                        d_ct = f"{'+'if dc>=0 else ''}{dc:,.0f}"
                    f.write(f"| **Ours: {exp_name}** | {e['n_total']} | {e['n_ac']} | "
                            f"**{e['ac_rate']:.1%}** | {d_ac} | {e['avg_ct']:,.0f} | {d_ct} |\n")
            
            f.write("\n")
        
        # Per-category breakdown
        f.write("## 2. 按题目类别 AC 率\n\n")
        cat_order = ["both_ac", "flash_ac_doubao_wrong", "flash_wrong_doubao_ac", "both_wrong"]
        cat_counts = Counter(categories.values())
        
        models_cat = ["gemini", "doubao", "gptoss"]
        for model in models_cat:
            f.write(f"\n### {model_labels.get(model, model)}\n\n")
            f.write("| Category | N |")
            for exp in ["bm25-5", "hybrid_m3-5"]:
                f.write(f" {exp} |")
            f.write("\n|----------|---|")
            for _ in ["bm25-5", "hybrid_m3-5"]:
                f.write("---------|")
            f.write("\n")
            
            for cat in cat_order:
                n = cat_counts.get(cat, 0)
                row = f"| {cat} | {n} |"
                for exp in ["bm25-5", "hybrid_m3-5"]:
                    tag = f"{exp}_{model}"
                    if tag in experiments and experiments[tag]:
                        e = experiments[tag]
                        ac_cat = e["ac_per_cat"].get(cat, 0)
                        tot_cat = e["total_per_cat"].get(cat, 0)
                        rate = ac_cat / tot_cat if tot_cat > 0 else 0
                        row += f" {ac_cat}/{tot_cat} ({rate:.1%}) |"
                    else:
                        row += " — |"
                f.write(row + "\n")
            f.write("\n")
        
        # Token analysis
        f.write("## 3. Token 使用分析\n\n")
        f.write("| Experiment | Avg Prompt T | Avg Completion T | Median CT | Tokens/AC |\n")
        f.write("|------------|-------------|-----------------|-----------|----------|\n")
        for tag in sorted(experiments.keys()):
            e = experiments[tag]
            if not e:
                continue
            tok_ac = f"{e['tokens_per_ac']:,.0f}" if e['tokens_per_ac'] else "—"
            f.write(f"| {tag} | {e['avg_pt']:,.0f} | {e['avg_ct']:,.0f} | "
                    f"{e['median_ct']:,.0f} | {tok_ac} |\n")
        f.write("\n")
        
        # Error distribution
        f.write("## 4. 错误分布\n\n")
        f.write("| Experiment | AC | WA | TLE | RE | CE | Other |\n")
        f.write("|------------|----|----|-----|----|----|-------|\n")
        for tag in sorted(experiments.keys()):
            e = experiments[tag]
            if not e:
                continue
            sd = e["status_dist"]
            other = sum(v for k, v in sd.items() if k not in ("AC", "WA", "TLE", "RE", "CE"))
            f.write(f"| {tag} | {sd.get('AC',0)} | {sd.get('WA',0)} | "
                    f"{sd.get('TLE',0)} | {sd.get('RE',0)} | {sd.get('CE',0)} | {other} |\n")
        f.write("\n")
        
        # Bootstrap CIs
        f.write("## 5. Bootstrap 95% 置信区间\n\n")
        f.write("| Experiment | AC Rate | 95% CI |\n")
        f.write("|------------|---------|--------|\n")
        for tag in sorted(experiments.keys()):
            e = experiments[tag]
            if not e:
                continue
            f.write(f"| {tag} | {e['ac_rate']:.1%} | [{e['ac_ci_lower']:.1%}, {e['ac_ci_upper']:.1%}] |\n")
        f.write("\n")
    
    print(f"  Report saved to {report_path}")
    return report_path


def main():
    categories = load_categories()
    
    # Analyze all experiments
    exp_tags = []
    for f in sorted(RESULTS_DIR.glob("gens_*.jsonl")):
        tag = f.stem.replace("gens_", "")
        # Only analyze our RAG experiments (not old/baseline)
        if tag.startswith("bm25-5_") or tag.startswith("hybrid_m3-5_"):
            exp_tags.append(tag)
    
    print(f"Found {len(exp_tags)} experiments to analyze")
    print(f"Categories loaded: {len(categories)} keys")
    
    experiments = {}
    for tag in exp_tags:
        print(f"  Analyzing: {tag}...", end=" ")
        result = analyze_experiment(tag, categories)
        if result:
            print(f"OK (n={result['n_total']}, AC={result['n_ac']}, "
                  f"rate={result['ac_rate']:.1%}, avg_ct={result['avg_ct']:.0f})")
            experiments[tag] = result
        else:
            print("SKIP (no data)")
    
    if not experiments:
        print("\nNo experiments with data found!")
        return
    
    # Print all analysis sections
    print_main_table(experiments, BASELINE_RESULTS)
    print_category_table(experiments, categories)
    print_error_distribution(experiments)
    print_token_analysis(experiments)
    
    # Generate markdown report
    report_path = generate_markdown_report(experiments, BASELINE_RESULTS, categories)
    
    print("\n" + "=" * 60)
    print("  Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
