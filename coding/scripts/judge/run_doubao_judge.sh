#!/bin/bash
# 运行 Doubao Seed 2.0 Pro 的 judge

cd "$(dirname "$0")"

INPUT="doubao_seed_cp_34799_v1.jsonl"
CASES="nemotron_cp_cases_34799_v1.jsonl"
OUTPUT="judge_results_doubao.jsonl"
WORKERS=96
LOG="judge_doubao_run.log"

# 检查输入文件是否存在且有内容
if [ ! -s "$INPUT" ]; then
    echo "错误: 输入文件 $INPUT 不存在或为空"
    echo "请先运行 gen_doubao_seed.py 生成答案"
    exit 1
fi

echo "开始 judge Doubao Seed 2.0 Pro 结果..."
echo "输入: $INPUT"
echo "输出: $OUTPUT"
echo "Workers: $WORKERS"
echo "日志: $LOG"
echo ""

# 运行 judge，同时输出到日志和终端
python3 run_judge_doubao.py \
    --input "$INPUT" \
    --cases "$CASES" \
    --output "$OUTPUT" \
    --workers "$WORKERS" \
    2>&1 | tee "$LOG"

echo ""
echo "Judge 完成！结果保存在: $OUTPUT"
echo "日志保存在: $LOG"
