#!/bin/bash
# 检查重新 judge 的进度

cd "$(dirname "$0")"

echo "=== 重新 Judge 进度检查 ==="
echo ""

# 检查进程
if pgrep -f "rejudge_pro_fast.py" > /dev/null; then
    echo "✅ 重新 judge 进程正在运行"
else
    echo "❌ 重新 judge 进程未运行"
fi
echo ""

# 显示日志
echo "--- 最新日志 (最后 20 行) ---"
tail -20 rejudge_pro_fast.log 2>/dev/null || echo "日志文件不存在或为空"
echo ""

# 显示结果文件
if [ -f judge_results_pro_rejudged.jsonl ]; then
    LINES=$(wc -l < judge_results_pro_rejudged.jsonl)
    SIZE=$(ls -lh judge_results_pro_rejudged.jsonl | awk '{print $5}')
    echo "--- 结果文件统计 ---"
    echo "已完成: $LINES / 2194 条"
    echo "文件大小: $SIZE"
    echo "进度: $(echo "scale=1; $LINES * 100 / 2194" | bc)%"
    
    if [ $LINES -gt 0 ]; then
        echo ""
        echo "--- 状态分布 ---"
        python3 << 'PYEOF'
import json
from collections import Counter

status_counter = Counter()
total = 0

try:
    with open('judge_results_pro_rejudged.jsonl', 'r', encoding='utf-8') as f:
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
    
    ac_count = status_counter.get('AC', 0)
    print(f"\n  AC 率: {ac_count}/{total} = {ac_count/total*100:.1f}%")
except Exception as e:
    print(f"  错误: {e}")
PYEOF
    fi
else
    echo "结果文件尚未创建"
fi

echo ""
echo "--- 监控命令 ---"
echo "实时查看日志: tail -f rejudge_pro_fast.log"
echo "查看结果: wc -l judge_results_pro_rejudged.jsonl"
