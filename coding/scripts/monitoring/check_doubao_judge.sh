#!/bin/bash
# 检查 Doubao Seed 2.0 Pro judge 进度

cd "$(dirname "$0")"

echo "=== Doubao Seed 2.0 Pro Judge 进度 ==="
echo ""

# 检查进程是否在运行
if pgrep -f "run_judge_doubao.py" > /dev/null; then
    echo "✅ Judge 进程正在运行"
else
    echo "❌ Judge 进程未运行"
fi
echo ""

# 显示最新日志
echo "--- 最新进度 (最后 10 行) ---"
tail -10 judge_doubao_run.log 2>/dev/null || echo "日志文件不存在"
echo ""

# 显示结果文件统计
if [ -f judge_results_doubao.jsonl ]; then
    LINES=$(wc -l < judge_results_doubao.jsonl)
    SIZE=$(ls -lh judge_results_doubao.jsonl | awk '{print $5}')
    echo "--- 结果文件统计 ---"
    echo "已完成: $LINES 条"
    echo "文件大小: $SIZE"
    
    # 统计状态分布
    if [ $LINES -gt 0 ]; then
        echo ""
        echo "--- 状态分布 (前 10) ---"
        python3 << 'PYEOF'
import json
from collections import Counter

status_counter = Counter()
total = 0

try:
    with open('judge_results_doubao.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                status = data.get('judge', {}).get('status', 'UNKNOWN')
                status_counter[status] += 1
                total += 1
            except:
                pass
    
    for status, count in status_counter.most_common(10):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {status:<30s} {count:>6d}  ({pct:5.1f}%)")
except Exception as e:
    print(f"  错误: {e}")
PYEOF
    fi
else
    echo "结果文件尚未创建"
fi

echo ""
echo "--- 监控命令 ---"
echo "实时查看日志: tail -f judge_doubao_run.log"
echo "查看结果: wc -l judge_results_doubao.jsonl"
