#!/bin/bash
# monitor_and_judge.sh
# 监控空输出数量，当降到很低时自动开始 Judge 任务

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EMPTY_THRESHOLD=100  # 空输出低于这个数量时开始 Judge
CHECK_INTERVAL=300   # 每5分钟检查一次
MAX_WAIT_HOURS=24    # 最多等待24小时

echo "=========================================="
echo "空输出监控与自动 Judge 任务"
echo "=========================================="
echo "目标空输出阈值: ${EMPTY_THRESHOLD} 条"
echo "检查间隔: ${CHECK_INTERVAL} 秒"
echo "最多等待: ${MAX_WAIT_HOURS} 小时"
echo "=========================================="
echo ""

START_TIME=$(date +%s)
MAX_WAIT_SECONDS=$((MAX_WAIT_HOURS * 3600))

while true; do
    # 检查是否超时
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED -gt $MAX_WAIT_SECONDS ]; then
        echo "⚠️  已达到最大等待时间 (${MAX_WAIT_HOURS} 小时)"
        break
    fi
    
    # 检查当前空输出数量
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检查空输出数量..."
    
    EMPTY_COUNT=$(python3 << 'PYEOF'
import json
empty = 0
has_output = 0
total = 0
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

print(f"{empty}|{has_output}|{total}")
PYEOF
)
    
    EMPTY=$(echo $EMPTY_COUNT | cut -d'|' -f1)
    HAS_OUTPUT=$(echo $EMPTY_COUNT | cut -d'|' -f2)
    TOTAL=$(echo $EMPTY_COUNT | cut -d'|' -f3)
    EMPTY_PCT=$(python3 -c "print(f'{int($EMPTY)/int($TOTAL)*100:.2f}')")
    
    echo "  总记录数: ${TOTAL}"
    echo "  有输出: ${HAS_OUTPUT} ($(python3 -c "print(f'{int($HAS_OUTPUT)/int($TOTAL)*100:.2f}')")%)"
    echo "  空输出: ${EMPTY} (${EMPTY_PCT}%)"
    
    # 检查是否达到阈值
    if [ "$EMPTY" -lt "$EMPTY_THRESHOLD" ]; then
        echo ""
        echo "✅ 空输出已降到阈值以下 (${EMPTY} < ${EMPTY_THRESHOLD})"
        echo "开始 Judge 任务..."
        echo ""
        
        # 检查是否已有 Judge 任务在运行
        if pgrep -f "run_judge_qiniu_oss.*with_prompt" > /dev/null; then
            echo "⚠️  Judge 任务已在运行，跳过"
        else
            # 启动 Judge 任务
            echo "启动 Judge 任务..."
            nohup python3 run_judge_qiniu_oss.py \
                --input qiniu_oss_cp_34799_v1_with_prompt.jsonl \
                --cases nemotron_cp_cases_34799_v1.jsonl \
                --output judge_results_qiniu_oss_with_prompt.jsonl \
                --workers 96 \
                > judge_qiniu_oss_with_prompt.log 2>&1 &
            
            JUDGE_PID=$!
            echo "Judge 任务已启动，PID: ${JUDGE_PID}"
            echo "日志文件: judge_qiniu_oss_with_prompt.log"
            echo ""
            echo "=========================================="
            echo "监控任务完成，Judge 任务已启动"
            echo "=========================================="
            break
        fi
    else
        echo "  空输出仍较多，继续等待..."
        
        # 检查重新生成任务是否还在运行
        if pgrep -f "refill_empty_outputs" > /dev/null; then
            echo "  ✅ 重新生成任务运行中"
        else
            echo "  ⚠️  重新生成任务已停止"
            echo "  检查是否需要启动新一轮..."
            
            # 如果空输出还很多，可以考虑启动新一轮
            if [ "$EMPTY" -gt 500 ]; then
                echo "  空输出仍较多 (${EMPTY} > 500)，建议手动启动新一轮重新生成"
            fi
        fi
    fi
    
    echo ""
    echo "等待 ${CHECK_INTERVAL} 秒后再次检查..."
    echo ""
    sleep $CHECK_INTERVAL
done
