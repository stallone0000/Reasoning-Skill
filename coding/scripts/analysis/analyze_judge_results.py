# -*- coding: utf-8 -*-
"""
analyze_judge_results.py

分析 judge 结果并生成统计报告。

用法:
  python analyze_judge_results.py --input judge_results_qiniu_oss.jsonl --output judge_report_qiniu_oss.md
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def iter_jsonl(path: Path):
    """流式读取 JSONL 文件。"""
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def analyze_judge_results(input_path: Path) -> Dict:
    """分析 judge 结果并返回统计信息。"""
    status_counter = Counter()
    lang_counter = Counter()
    split_ok_count = 0
    split_fail_count = 0
    total_items = 0
    total_passed = 0
    total_tests = 0
    
    # 按状态统计
    status_details = {
        "AC": {"count": 0, "passed": 0, "total": 0},
        "WA": {"count": 0, "passed": 0, "total": 0},
        "TLE": {"count": 0, "passed": 0, "total": 0},
        "RE": {"count": 0, "passed": 0, "total": 0},
        "CE": {"count": 0, "passed": 0, "total": 0},
        "NO_OUTPUT": {"count": 0, "passed": 0, "total": 0},
        "NO_CASES": {"count": 0, "passed": 0, "total": 0},
        "JUDGE_ERROR": {"count": 0, "passed": 0, "total": 0},
    }
    
    # 语言分布
    lang_status = {}
    
    for record in iter_jsonl(input_path):
        total_items += 1
        judge = record.get("judge", {})
        status = judge.get("status", "UNKNOWN")
        lang = judge.get("lang")
        passed = judge.get("passed", 0)
        total = judge.get("total_tests", 0)
        split_ok = record.get("split_ok", False)
        
        status_counter[status] += 1
        if lang:
            lang_counter[lang] += 1
            if lang not in lang_status:
                lang_status[lang] = Counter()
            lang_status[lang][status] += 1
        
        if split_ok:
            split_ok_count += 1
        else:
            split_fail_count += 1
        
        total_passed += passed
        total_tests += total
        
        if status in status_details:
            status_details[status]["count"] += 1
            status_details[status]["passed"] += passed
            status_details[status]["total"] += total
    
    # 计算通过率
    ac_rate = status_counter.get("AC", 0) / total_items * 100 if total_items > 0 else 0
    test_pass_rate = total_passed / total_tests * 100 if total_tests > 0 else 0
    split_rate = split_ok_count / total_items * 100 if total_items > 0 else 0
    
    return {
        "total_items": total_items,
        "status_counter": dict(status_counter),
        "lang_counter": dict(lang_counter),
        "lang_status": {k: dict(v) for k, v in lang_status.items()},
        "split_ok_count": split_ok_count,
        "split_fail_count": split_fail_count,
        "split_rate": split_rate,
        "total_passed": total_passed,
        "total_tests": total_tests,
        "test_pass_rate": test_pass_rate,
        "ac_rate": ac_rate,
        "status_details": status_details,
    }


def generate_report(stats: Dict, output_path: Path):
    """生成 Markdown 格式的报告。"""
    lines = []
    lines.append("# Judge 结果统计报告\n")
    lines.append(f"## 总体统计\n")
    lines.append(f"- **总题目数**: {stats['total_items']}")
    lines.append(f"- **AC 率**: {stats['ac_rate']:.2f}% ({stats['status_counter'].get('AC', 0)}/{stats['total_items']})")
    lines.append(f"- **测试用例通过率**: {stats['test_pass_rate']:.2f}% ({stats['total_passed']}/{stats['total_tests']})")
    lines.append(f"- **拆分成功率**: {stats['split_rate']:.2f}% ({stats['split_ok_count']}/{stats['total_items']})")
    lines.append("")
    
    lines.append("## 状态分布\n")
    lines.append("| 状态 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for status, count in sorted(stats['status_counter'].items(), key=lambda x: -x[1]):
        pct = count / stats['total_items'] * 100 if stats['total_items'] > 0 else 0
        lines.append(f"| {status} | {count} | {pct:.2f}% |")
    lines.append("")
    
    if stats['lang_counter']:
        lines.append("## 语言分布\n")
        lines.append("| 语言 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        for lang, count in sorted(stats['lang_counter'].items(), key=lambda x: -x[1]):
            pct = count / stats['total_items'] * 100 if stats['total_items'] > 0 else 0
            lines.append(f"| {lang} | {count} | {pct:.2f}% |")
        lines.append("")
        
        lines.append("## 语言-状态交叉统计\n")
        for lang in sorted(stats['lang_status'].keys()):
            lines.append(f"### {lang}\n")
            lines.append("| 状态 | 数量 |")
            lines.append("|------|------|")
            lang_stats = stats['lang_status'][lang]
            for status, count in sorted(lang_stats.items(), key=lambda x: -x[1]):
                lines.append(f"| {status} | {count} |")
            lines.append("")
    
    lines.append("## 详细状态统计\n")
    lines.append("| 状态 | 题目数 | 通过测试数 | 总测试数 | 测试通过率 |")
    lines.append("|------|--------|------------|----------|------------|")
    for status in ["AC", "WA", "TLE", "RE", "CE", "NO_OUTPUT", "NO_CASES", "JUDGE_ERROR"]:
        detail = stats['status_details'].get(status, {"count": 0, "passed": 0, "total": 0})
        count = detail["count"]
        passed = detail["passed"]
        total = detail["total"]
        rate = passed / total * 100 if total > 0 else 0
        lines.append(f"| {status} | {count} | {passed} | {total} | {rate:.2f}% |")
    lines.append("")
    
    with output_path.open('w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description="分析 judge 结果并生成统计报告")
    parser.add_argument("--input", required=True, help="Judge 结果 JSONL 文件")
    parser.add_argument("--output", required=True, help="输出 Markdown 报告文件")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return
    
    print(f"分析 judge 结果: {input_path}")
    stats = analyze_judge_results(input_path)
    
    print(f"生成报告: {output_path}")
    generate_report(stats, output_path)
    
    print("\n统计摘要:")
    print(f"  总题目数: {stats['total_items']}")
    print(f"  AC 率: {stats['ac_rate']:.2f}%")
    print(f"  测试用例通过率: {stats['test_pass_rate']:.2f}%")
    print(f"  拆分成功率: {stats['split_rate']:.2f}%")
    print(f"\n状态分布:")
    for status, count in sorted(stats['status_counter'].items(), key=lambda x: -x[1]):
        pct = count / stats['total_items'] * 100 if stats['total_items'] > 0 else 0
        print(f"  {status}: {count} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
