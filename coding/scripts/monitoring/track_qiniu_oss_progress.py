#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持续跟踪 qiniu_oss 任务进度，并保存统计信息
"""

import json
import time
import subprocess
import os
from pathlib import Path
from datetime import datetime
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
GEN_LOG = SCRIPT_DIR / "gen_qiniu_oss_with_prompt.log"
GEN_OUTPUT = SCRIPT_DIR / "qiniu_oss_cp_34799_v1_with_prompt.jsonl"
JUDGE_LOG = SCRIPT_DIR / "judge_qiniu_oss_with_prompt.log"
JUDGE_OUTPUT = SCRIPT_DIR / "judge_results_qiniu_oss_with_prompt.jsonl"
STATS_FILE = SCRIPT_DIR / "qiniu_oss_progress_stats.json"
SUMMARY_FILE = SCRIPT_DIR / "qiniu_oss_progress_summary.md"

def check_process_running(pattern):
    """检查进程是否在运行"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False

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

def parse_gen_log():
    """解析生成日志，获取最新进度"""
    if not GEN_LOG.exists():
        return {}
    
    try:
        with open(GEN_LOG, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 查找最后几行中的进度信息
            for line in reversed(lines[-50:]):
                if "Generating:" in line and "%" in line:
                    # 提取进度信息
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "/" in part and part.replace("/", "").replace("34721", "").isdigit():
                            current, total = part.split("/")
                            try:
                                current = int(current)
                                total = int(total)
                                percent = current / total * 100 if total > 0 else 0
                                return {
                                    "current": current,
                                    "total": total,
                                    "percent": percent,
                                    "line": line.strip()
                                }
                            except:
                                pass
                elif "生成完成" in line or "完成:" in line:
                    # 查找完成信息
                    if "成功:" in line and "失败:" in line:
                        parts = line.split()
                        success = 0
                        fail = 0
                        for i, part in enumerate(parts):
                            if part == "成功:" and i + 1 < len(parts):
                                try:
                                    success = int(parts[i+1].replace("条", ""))
                                except:
                                    pass
                            elif part == "失败:" and i + 1 < len(parts):
                                try:
                                    fail = int(parts[i+1].replace("条", ""))
                                except:
                                    pass
                        return {"completed": True, "success": success, "fail": fail}
    except Exception as e:
        pass
    return {}

def parse_judge_log():
    """解析 judge 日志，获取最新进度"""
    if not JUDGE_LOG.exists():
        return {}
    
    try:
        with open(JUDGE_LOG, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 查找最后几行中的进度信息
            for line in reversed(lines[-50:]):
                if "[" in line and "/" in line and "%" in line:
                    # 提取进度信息
                    import re
                    match = re.search(r'\[(\d+)/(\d+)\]\s+(\d+\.\d+)%', line)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        percent = float(match.group(3))
                        # 提取状态分布
                        status_match = re.search(r'\|\s+(.+)', line)
                        status_str = status_match.group(1) if status_match else ""
                        return {
                            "current": current,
                            "total": total,
                            "percent": percent,
                            "status": status_str,
                            "line": line.strip()
                        }
                elif "Judge 完成" in line or "完成:" in line:
                    # 查找完成信息
                    import re
                    match = re.search(r'(\d+)\s+items', line)
                    if match:
                        return {"completed": True, "total": int(match.group(1))}
    except Exception as e:
        pass
    return {}

def analyze_judge_results():
    """分析 judge 结果"""
    if not JUDGE_OUTPUT.exists():
        return {}
    
    stats = {
        "total": 0,
        "status_counter": Counter(),
        "lang_counter": Counter(),
    }
    
    try:
        with open(JUDGE_OUTPUT, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    stats["total"] += 1
                    judge = rec.get("judge", {})
                    status = judge.get("status", "UNKNOWN")
                    lang = judge.get("lang")
                    stats["status_counter"][status] += 1
                    if lang:
                        stats["lang_counter"][lang] += 1
                except:
                    pass
    except:
        pass
    
    return {
        "total": stats["total"],
        "status_distribution": dict(stats["status_counter"]),
        "lang_distribution": dict(stats["lang_counter"]),
    }

def collect_stats():
    """收集当前统计信息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    stats = {
        "timestamp": timestamp,
        "generation": {
            "running": check_process_running("gen_qiniu_oss.py.*with_prompt"),
            "output_lines": get_file_lines(GEN_OUTPUT),
            "output_size": get_file_size(GEN_OUTPUT),
            "log_size": get_file_size(GEN_LOG),
            "progress": parse_gen_log(),
        },
        "judge": {
            "running": check_process_running("run_judge_qiniu_oss.*with_prompt"),
            "output_lines": get_file_lines(JUDGE_OUTPUT),
            "output_size": get_file_size(JUDGE_OUTPUT),
            "log_size": get_file_size(JUDGE_LOG),
            "progress": parse_judge_log(),
        },
    }
    
    # 如果有 judge 结果，分析统计
    if stats["judge"]["output_lines"] > 0:
        judge_analysis = analyze_judge_results()
        stats["judge"]["analysis"] = judge_analysis
    
    return stats

def save_stats(stats):
    """保存统计信息到 JSON 文件"""
    # 读取历史统计
    history = []
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []
    
    # 添加新统计
    history.append(stats)
    
    # 保存
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def generate_summary():
    """生成 Markdown 格式的摘要"""
    if not STATS_FILE.exists():
        return
    
    try:
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if not history:
            return
        
        latest = history[-1]
        
        lines = []
        lines.append("# Qiniu GPT-OSS-120B 任务进度统计\n")
        lines.append(f"**最后更新时间**: {latest['timestamp']}\n")
        lines.append("")
        
        # 生成任务状态
        lines.append("## 生成任务\n")
        gen = latest["generation"]
        if gen["running"]:
            lines.append("- **状态**: 🟢 运行中")
        else:
            lines.append("- **状态**: 🔴 已停止")
        lines.append(f"- **已生成**: {gen['output_lines']} / 34721 条")
        if gen["output_lines"] > 0:
            percent = gen["output_lines"] / 34721 * 100
            lines.append(f"- **完成度**: {percent:.2f}%")
        lines.append(f"- **输出文件大小**: {gen['output_size']}")
        lines.append(f"- **日志文件大小**: {gen['log_size']}")
        if gen.get("progress"):
            prog = gen["progress"]
            if "completed" in prog:
                lines.append(f"- **完成**: 成功 {prog.get('success', 0)} 条, 失败 {prog.get('fail', 0)} 条")
            elif "current" in prog:
                lines.append(f"- **当前进度**: {prog['current']}/{prog['total']} ({prog.get('percent', 0):.2f}%)")
        lines.append("")
        
        # Judge 任务状态
        lines.append("## Judge 任务\n")
        judge = latest["judge"]
        if judge["running"]:
            lines.append("- **状态**: 🟢 运行中")
        else:
            lines.append("- **状态**: 🔴 已停止")
        lines.append(f"- **已 judge**: {judge['output_lines']} 条")
        lines.append(f"- **输出文件大小**: {judge['output_size']}")
        lines.append(f"- **日志文件大小**: {judge['log_size']}")
        if judge.get("progress"):
            prog = judge["progress"]
            if "completed" in prog:
                lines.append(f"- **完成**: {prog.get('total', 0)} 条")
            elif "current" in prog:
                lines.append(f"- **当前进度**: {prog['current']}/{prog['total']} ({prog.get('percent', 0):.2f}%)")
                if "status" in prog:
                    lines.append(f"- **状态分布**: {prog['status']}")
        lines.append("")
        
        # Judge 结果分析
        if judge.get("analysis"):
            analysis = judge["analysis"]
            lines.append("## Judge 结果分析\n")
            lines.append(f"- **总题目数**: {analysis.get('total', 0)}")
            
            status_dist = analysis.get("status_distribution", {})
            if status_dist:
                lines.append("\n### 状态分布\n")
                lines.append("| 状态 | 数量 | 占比 |")
                lines.append("|------|------|------|")
                total = analysis.get("total", 1)
                for status, count in sorted(status_dist.items(), key=lambda x: -x[1]):
                    pct = count / total * 100 if total > 0 else 0
                    lines.append(f"| {status} | {count} | {pct:.2f}% |")
            
            lang_dist = analysis.get("lang_distribution", {})
            if lang_dist:
                lines.append("\n### 语言分布\n")
                lines.append("| 语言 | 数量 | 占比 |")
                lines.append("|------|------|------|")
                total = analysis.get("total", 1)
                for lang, count in sorted(lang_dist.items(), key=lambda x: -x[1]):
                    pct = count / total * 100 if total > 0 else 0
                    lines.append(f"| {lang} | {count} | {pct:.2f}% |")
        
        # 历史记录
        if len(history) > 1:
            lines.append("\n## 历史记录\n")
            lines.append(f"共记录了 {len(history)} 次检查\n")
            lines.append("| 时间 | 生成进度 | Judge 进度 |")
            lines.append("|------|----------|------------|")
            for stat in history[-10:]:  # 只显示最后10条
                gen_lines = stat["generation"]["output_lines"]
                judge_lines = stat["judge"]["output_lines"]
                lines.append(f"| {stat['timestamp']} | {gen_lines} | {judge_lines} |")
        
        # 保存摘要
        with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ 摘要已保存到: {SUMMARY_FILE}")
        
    except Exception as e:
        print(f"生成摘要时出错: {e}")

def main():
    """主函数：持续跟踪并保存统计"""
    print("=" * 60)
    print("Qiniu GPT-OSS-120B 任务进度跟踪")
    print("=" * 60)
    print(f"统计文件: {STATS_FILE}")
    print(f"摘要文件: {SUMMARY_FILE}")
    print("按 Ctrl+C 停止跟踪\n")
    
    check_interval = 300  # 每5分钟检查一次
    
    try:
        while True:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 收集统计信息...")
            
            stats = collect_stats()
            save_stats(stats)
            generate_summary()
            
            # 显示当前状态
            gen = stats["generation"]
            judge = stats["judge"]
            
            print(f"生成任务: {'运行中' if gen['running'] else '已停止'} | "
                  f"已生成: {gen['output_lines']} / 34721 ({gen['output_lines']/34721*100:.2f}%)")
            print(f"Judge任务: {'运行中' if judge['running'] else '已停止'} | "
                  f"已 judge: {judge['output_lines']} 条")
            
            if not gen['running'] and not judge['running']:
                print("\n所有任务已完成！")
                break
            
            print(f"\n等待 {check_interval} 秒后再次检查...")
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n跟踪已停止")
        # 最后一次保存
        stats = collect_stats()
        save_stats(stats)
        generate_summary()
        print(f"\n最终统计已保存:")
        print(f"  - {STATS_FILE}")
        print(f"  - {SUMMARY_FILE}")

if __name__ == '__main__':
    main()
