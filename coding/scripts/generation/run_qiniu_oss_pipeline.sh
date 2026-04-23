#!/bin/bash
# run_qiniu_oss_pipeline.sh
#
# 完整的 qiniu/gpt-oss-120b 模型调用和 judge 流程
#
# 用法:
#   bash run_qiniu_oss_pipeline.sh
#
# 环境变量:
#   QINIU_API_KEY: API Key
#   QINIU_API_URL: API URL (可选，有默认值)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置
INPUT_QUESTIONS="nemotron_cp_questions_34799_v1.jsonl"
CASES_FILE="nemotron_cp_cases_34799_v1.jsonl"
OUTPUT_GEN="qiniu_oss_cp_34799_v1.jsonl"
OUTPUT_JUDGE="judge_results_qiniu_oss.jsonl"
OUTPUT_REPORT="judge_report_qiniu_oss.md"

MODEL="qiniu/gpt-oss-120b"
WORKERS_GEN=500
WORKERS_JUDGE=96

echo "=========================================="
echo "Qiniu GPT-OSS-120B 完整流程"
echo "=========================================="
echo ""

# 步骤 1: 生成代码
echo "步骤 1: 调用模型生成代码..."
echo "  输入: $INPUT_QUESTIONS"
echo "  输出: $OUTPUT_GEN"
echo "  模型: $MODEL"
echo "  并发: $WORKERS_GEN"
echo ""

if [ -z "$QINIU_API_KEY" ]; then
    echo "警告: 未设置 QINIU_API_KEY 环境变量"
    echo "请设置: export QINIU_API_KEY=your_api_key"
    exit 1
fi

python3 gen_qiniu_oss.py \
    --input "$INPUT_QUESTIONS" \
    --output "$OUTPUT_GEN" \
    --model "$MODEL" \
    --workers "$WORKERS_GEN" \
    --api_key "$QINIU_API_KEY" \
    ${QINIU_API_URL:+--api_url "$QINIU_API_URL"}

if [ $? -ne 0 ]; then
    echo "错误: 代码生成失败"
    exit 1
fi

echo ""
echo "步骤 1 完成!"
echo ""

# 步骤 2: Judge
echo "步骤 2: 对生成结果进行 judge..."
echo "  输入: $OUTPUT_GEN"
echo "  Cases: $CASES_FILE"
echo "  输出: $OUTPUT_JUDGE"
echo "  并发: $WORKERS_JUDGE"
echo ""

python3 run_judge_qiniu_oss.py \
    --input "$OUTPUT_GEN" \
    --cases "$CASES_FILE" \
    --output "$OUTPUT_JUDGE" \
    --workers "$WORKERS_JUDGE"

if [ $? -ne 0 ]; then
    echo "错误: Judge 失败"
    exit 1
fi

echo ""
echo "步骤 2 完成!"
echo ""

# 步骤 3: 生成报告
echo "步骤 3: 生成 judge 统计报告..."
echo "  输入: $OUTPUT_JUDGE"
echo "  输出: $OUTPUT_REPORT"
echo ""

python3 analyze_judge_results.py \
    --input "$OUTPUT_JUDGE" \
    --output "$OUTPUT_REPORT"

if [ $? -ne 0 ]; then
    echo "错误: 报告生成失败"
    exit 1
fi

echo ""
echo "步骤 3 完成!"
echo ""

echo "=========================================="
echo "所有步骤完成!"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - 模型输出: $OUTPUT_GEN"
echo "  - Judge 结果: $OUTPUT_JUDGE"
echo "  - 统计报告: $OUTPUT_REPORT"
echo ""
