#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成最终统计报告
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent

def get_file_lines(filepath):
    """获取文件行数"""
    if not filepath.exists():
        return 0
    try:
        result = subprocess.run(
            ["wc", "-l", str(filepath)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return int(result.stdout.split()[0])
    except:
        pass
    return 0

def get_file_size(filepath):
    """获取文件大小（人类可读格式）"""
    if not filepath.exists():
        return "0B"
    try:
        result = subprocess.run(
            ["du", "-h", str(filepath)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.split()[0]
    except:
        pass
    return "0B"

def analyze_gen_results():
    """分析生成结果"""
    gen_output = SCRIPT_DIR / "qiniu_oss_cp_34799_v1_with_prompt.jsonl"
    if not gen_output.exists():
        return {}
    
    stats = {
        "total": 0,
        "has_output": 0,
        "empty_output": 0,
        "cpp_blocks": 0,
        "python_blocks": 0,
        "rust_blocks": 0,
    }
    
    import re
    try:
        with open(gen_output, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    stats["total"] += 1
                    output = rec.get("llm_output", "")
                    if output and output.strip():
                        stats["has_output"] += 1
                        # 统计代码块
                        stats["cpp_blocks"] += len(re.findall(r'```(?:cpp|c\+\+)', output, re.IGNORECASE))
                        stats["python_blocks"] += len(re.findall(r'```python', output, re.IGNORECASE))
                        stats["rust_blocks"] += len(re.findall(r'```(?:rust|rs)', output, re.IGNORECASE))
                    else:
                        stats["empty_output"] += 1
                except:
                    pass
    except:
        pass
    
    return stats

def analyze_judge_results():
    """分析 judge 结果"""
    judge_output = SCRIPT_DIR / "judge_results_qiniu_oss_with_prompt.jsonl"
    if not judge_output.exists():
        return {}
    
    stats = {
        "total": 0,
        "status_counter": Counter(),
        "lang_counter": Counter(),
        "split_ok": 0,
        "split_fail": 0,
    }
    
    try:
        with open(judge_output, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    stats["total"] += 1
                    judge = rec.get("judge", {})
                    status = judge.get("status", "UNKNOWN")
                    lang = judge.get("lang")
                    split_ok = rec.get("split_ok", False)
                    
                    stats["status_counter"][status] += 1
                    if lang:
                        stats["lang_counter"][lang] += 1
                    if split_ok:
                        stats["split_ok"] += 1
                    else:
                        stats["split_fail"] += 1
                except:
                    pass
    except:
        pass
    
    return {
        "total": stats["total"],
        "status_distribution": dict(stats["status_counter"]),
        "lang_distribution": dict(stats["lang_counter"]),
        "split_ok": stats["split_ok"],
        "split_fail": stats["split_fail"],
    }

def generate_final_report():
    """生成最终统计报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 收集信息
    gen_output = SCRIPT_DIR / "qiniu_oss_cp_34799_v1_with_prompt.jsonl"
    judge_output = SCRIPT_DIR / "judge_results_qiniu_oss_with_prompt.jsonl"
    gen_log = SCRIPT_DIR / "gen_qiniu_oss_with_prompt.log"
    judge_log = SCRIPT_DIR / "judge_qiniu_oss_with_prompt.log"
    
    gen_lines = get_file_lines(gen_output)
    judge_lines = get_file_lines(judge_output)
    
    # 分析生成结果
    gen_stats = analyze_gen_results()
    
    # 分析 judge 结果
    judge_stats = analyze_judge_results()
    
    # 生成报告
    report = {
        "timestamp": timestamp,
        "generation": {
            "total_lines": gen_lines,
            "total_expected": 34721,
            "completion_rate": gen_lines / 34721 * 100 if gen_lines > 0 else 0,
            "output_size": get_file_size(gen_output),
            "log_size": get_file_size(gen_log),
            "statistics": gen_stats,
        },
        "judge": {
            "total_lines": judge_lines,
            "output_size": get_file_size(judge_output),
            "log_size": get_file_size(judge_log),
            "statistics": judge_stats,
        },
    }
    
    # 保存 JSON 报告
    report_file = SCRIPT_DIR / "qiniu_oss_final_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 生成 Markdown 报告
    md_lines = []
    md_lines.append("# Qiniu GPT-OSS-120B 最终统计报告\n")
    md_lines.append(f"**生成时间**: {timestamp}\n")
    md_lines.append("")
    
    # 生成任务统计
    md_lines.append("## 生成任务统计\n")
    md_lines.append(f"- **总题目数**: {gen_lines} / 34721")
    md_lines.append(f"- **完成度**: {gen_lines / 34721 * 100:.2f}%")
    md_lines.append(f"- **输出文件大小**: {get_file_size(gen_output)}")
    md_lines.append(f"- **日志文件大小**: {get_file_size(gen_log)}")
    
    if gen_stats:
        md_lines.append("\n### 生成结果分析\n")
        md_lines.append(f"- **有输出**: {gen_stats.get('has_output', 0)} 条")
        md_lines.append(f"- **空输出**: {gen_stats.get('empty_output', 0)} 条")
        md_lines.append(f"- **C++ 代码块总数**: {gen_stats.get('cpp_blocks', 0)}")
        md_lines.append(f"- **Python 代码块总数**: {gen_stats.get('python_blocks', 0)}")
        md_lines.append(f"- **Rust 代码块总数**: {gen_stats.get('rust_blocks', 0)}")
    
    md_lines.append("")
    
    # Judge 任务统计
    md_lines.append("## Judge 任务统计\n")
    md_lines.append(f"- **已 judge**: {judge_lines} 条")
    md_lines.append(f"- **输出文件大小**: {get_file_size(judge_output)}")
    md_lines.append(f"- **日志文件大小**: {get_file_size(judge_log)}")
    
    if judge_stats and judge_stats.get("total", 0) > 0:
        md_lines.append("\n### Judge 结果分析\n")
        total = judge_stats["total"]
        md_lines.append(f"- **总题目数**: {total}")
        
        status_dist = judge_stats.get("status_distribution", {})
        if status_dist:
            md_lines.append("\n#### 状态分布\n")
            md_lines.append("| 状态 | 数量 | 占比 |")
            md_lines.append("|------|------|------|")
            for status, count in sorted(status_dist.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total > 0 else 0
                md_lines.append(f"| {status} | {count} | {pct:.2f}% |")
        
        lang_dist = judge_stats.get("lang_distribution", {})
        if lang_dist:
            md_lines.append("\n#### 语言分布\n")
            md_lines.append("| 语言 | 数量 | 占比 |")
            md_lines.append("|------|------|------|")
            for lang, count in sorted(lang_dist.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total > 0 else 0
                md_lines.append(f"| {lang} | {count} | {pct:.2f}% |")
        
        split_ok = judge_stats.get("split_ok", 0)
        split_fail = judge_stats.get("split_fail", 0)
        if split_ok + split_fail > 0:
            split_total = split_ok + split_fail
            md_lines.append(f"\n#### 拆分统计\n")
            md_lines.append(f"- **拆分成功**: {split_ok} ({split_ok/split_total*100:.2f}%)")
            md_lines.append(f"- **拆分失败**: {split_fail} ({split_fail/split_total*100:.2f}%)")
    
    md_lines.append("")
    
    # 保存 Markdown 报告
    md_file = SCRIPT_DIR / "qiniu_oss_final_report.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"✓ 最终报告已生成:")
    print(f"  - JSON: {report_file}")
    print(f"  - Markdown: {md_file}")
    
    return report

if __name__ == '__main__':
    generate_final_report()
