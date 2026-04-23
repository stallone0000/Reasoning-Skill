#!/bin/bash
# auto_refill_until_ready.sh
# 自动循环重新生成空输出，直到空输出降到很低水平

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/qihoo_env.sh"
unset_qihoo_proxies
require_qihoo_api_key

cd "$SCRIPT_DIR"

MAX_EMPTY_THRESHOLD=100  # 空输出低于这个数量时停止
MAX_ROUNDS=5  # 最多运行几轮
ROUND=1

echo "=========================================="
echo "自动减少空输出任务"
echo "=========================================="
echo "目标: 将空输出降到 ${MAX_EMPTY_THRESHOLD} 条以下"
echo "最多运行: ${MAX_ROUNDS} 轮"
echo "=========================================="
echo ""

while [ $ROUND -le $MAX_ROUNDS ]; do
    echo "=========================================="
    echo "第 ${ROUND} 轮重新生成"
    echo "=========================================="
    
    # 检查当前空输出数量
    echo "检查当前空输出..."
    EMPTY_COUNT=$(python3 << 'PYEOF'
import json
empty = 0
with open('qiniu_oss_cp_34799_v1_with_prompt.jsonl', 'r') as f:
    for line in f:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            output = rec.get('llm_output', '')
            if not output or not output.strip():
                empty += 1
        except:
            pass
print(empty)
PYEOF
)
    
    echo "当前空输出: ${EMPTY_COUNT} 条"
    
    if [ "$EMPTY_COUNT" -lt "$MAX_EMPTY_THRESHOLD" ]; then
        echo ""
        echo "✅ 空输出已降到目标水平以下 (${EMPTY_COUNT} < ${MAX_EMPTY_THRESHOLD})"
        echo "可以开始 Judge 任务了！"
        break
    fi
    
    echo "空输出仍较多，开始第 ${ROUND} 轮重新生成..."
    
    # 生成空输出 key 列表
    echo "生成空输出 key 列表..."
    python3 << 'PYEOF'
import json
empty_keys = []
with open('qiniu_oss_cp_34799_v1_with_prompt.jsonl', 'r') as f:
    for line in f:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            output = rec.get('llm_output', '')
            if not output or not output.strip():
                empty_keys.append(rec.get('unique_key'))
        except:
            pass

with open('empty_output_keys_current.txt', 'w') as f:
    for key in empty_keys:
        f.write(key + '\n')
print(f"找到 {len(empty_keys)} 个空输出")
PYEOF
    
    EMPTY_KEYS_COUNT=$(wc -l < empty_output_keys_current.txt)
    echo "需要重新生成: ${EMPTY_KEYS_COUNT} 条"
    
    if [ "$EMPTY_KEYS_COUNT" -eq 0 ]; then
        echo "没有空输出了，退出"
        break
    fi
    
    # 启动重新生成任务
    echo "启动重新生成任务..."
    python3 refill_empty_outputs.py \
        --input qiniu_oss_cp_34799_v1_with_prompt.jsonl \
        --output qiniu_oss_cp_34799_v1_with_prompt.jsonl \
        --questions nemotron_cp_questions_34799_v1.jsonl \
        --empty_keys empty_output_keys_current.txt \
        --workers 200 \
        --api_key "$API_KEY_VALUE" \
        > "refill_round_${ROUND}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ 第 ${ROUND} 轮完成"
    else
        echo "❌ 第 ${ROUND} 轮失败，检查日志: refill_round_${ROUND}.log"
    fi
    
    echo ""
    ROUND=$((ROUND + 1))
    
    # 等待一下再继续
    if [ $ROUND -le $MAX_ROUNDS ]; then
        echo "等待 10 秒后开始下一轮..."
        sleep 10
    fi
done

if [ $ROUND -gt $MAX_ROUNDS ]; then
    echo ""
    echo "⚠️  已达到最大轮数 (${MAX_ROUNDS})，停止"
fi

echo ""
echo "=========================================="
echo "最终统计:"
python3 << 'PYEOF'
import json
total = 0
empty = 0
has_output = 0
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

print(f"总记录数: {total}")
print(f"有输出: {has_output} ({has_output/total*100:.2f}%)")
print(f"空输出: {empty} ({empty/total*100:.2f}%)")
PYEOF
echo "=========================================="
