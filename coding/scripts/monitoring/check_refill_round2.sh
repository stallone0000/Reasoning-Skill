#!/bin/bash
# check_refill_round2.sh
# 检查第二轮重新生成空输出的进度

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "第二轮重新生成空输出进度检查"
echo "=========================================="
echo ""

# 检查进程
if pgrep -f "refill_empty_outputs.*round2" > /dev/null || pgrep -f "refill_empty_outputs.py.*empty_output_keys_round2" > /dev/null; then
    echo "✅ 重新生成任务: 运行中"
else
    echo "❌ 重新生成任务: 已停止"
fi

# 检查日志
if [ -f "refill_empty_outputs_round2.log" ]; then
    echo ""
    echo "最新日志 (最后10行):"
    echo "---"
    tail -10 refill_empty_outputs_round2.log
    echo "---"
    
    # 统计成功和失败
    if grep -q "重新生成完成" refill_empty_outputs_round2.log; then
        echo ""
        echo "任务完成信息:"
        tail -5 refill_empty_outputs_round2.log | grep -E "成功|失败|完成"
    fi
else
    echo "日志文件不存在"
fi

# 检查当前空输出数量
echo ""
echo "当前空输出统计:"
python3 << 'EOF'
import json

total = 0
empty = 0
has_output = 0

try:
    with open('qiniu_oss_cp_34799_v1_with_prompt.jsonl', 'r') as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
                output = rec.get('llm_output', '')
                if output and output.strip():
                    has_output += 1
                else:
                    empty += 1
            except:
                pass
    
    print(f"  总记录数: {total}")
    print(f"  有输出: {has_output} ({has_output/total*100:.2f}%)")
    print(f"  空输出: {empty} ({empty/total*100:.2f}%)")
    
    if empty < 100:
        print(f"  ✅ 空输出已降到很低水平！可以开始 Judge 了")
    elif empty < 500:
        print(f"  ⚠️  空输出较少，建议再运行一轮")
    else:
        print(f"  ⏳ 空输出仍较多，继续等待...")
except Exception as e:
    print(f"  错误: {e}")
EOF

echo ""
echo "=========================================="
