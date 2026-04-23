#!/bin/bash
# ============================================================
# run_all_experiments.sh
#
# 并行启动 12 个 experience-RAG 实验:
#   2 models (gemini, doubao) × 6 strategies
#     baseline, bm25-5, embed_m3-5, embed_large-5, hybrid_m3-5, hybrid_large-5
#
# 用法:
#   bash run_all_experiments.sh          # 默认 50 题
#   bash run_all_experiments.sh 1000     # 全量 1000 题
#   bash run_all_experiments.sh 10       # 测试 10 题
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../scripts/common/qihoo_env.sh"
unset_qihoo_proxies
require_qihoo_api_key

cd "$SCRIPT_DIR"

N_PROBLEMS="${1:-50}"
WORKERS=8

MODELS=("gemini" "doubao")
EXPERIMENTS=("baseline" "bm25-5" "embed_m3-5" "embed_large-5" "hybrid_m3-5" "hybrid_large-5")

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "============================================="
echo "  Experience-RAG 实验"
echo "  题目数: $N_PROBLEMS"
echo "  模型: ${MODELS[*]}"
echo "  策略: ${EXPERIMENTS[*]}"
echo "  Workers: $WORKERS"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# Phase 1: 启动所有推理实验 (12 个并行)
PIDS=()

for model in "${MODELS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        log="$LOG_DIR/${exp}_${model}.log"
        echo "  启动: ${exp}/${model}"
        
        python3 step4_inference_and_judge.py \
            --experiment "$exp" \
            --model "$model" \
            --n_problems "$N_PROBLEMS" \
            --workers "$WORKERS" \
            --api_key "$API_KEY_VALUE" \
            --skip_judge \
            > "$log" 2>&1 &
        
        PIDS+=($!)
    done
done

echo ""
echo "  已启动 ${#PIDS[@]} 个推理进程"
echo "  等待所有推理完成..."

# Wait and track
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "  推理完成! 失败: $FAIL / ${#PIDS[@]}"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"

# Phase 2: Judge (串行, 因为 judge 本身是多进程)
echo ""
echo "============================================="
echo "  Phase 2: Judge"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

for model in "${MODELS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        log="$LOG_DIR/judge_${exp}_${model}.log"
        echo "  Judge: ${exp}/${model}"
        
        python3 step4_inference_and_judge.py \
            --experiment "$exp" \
            --model "$model" \
            --n_problems "$N_PROBLEMS" \
            --workers "$WORKERS" \
            --api_key "$API_KEY_VALUE" \
            > "$log" 2>&1 || echo "    [WARN] judge ${exp}/${model} exit non-0"
    done
done

echo ""
echo "============================================="
echo "  全部完成! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# Print all reports
echo ""
echo "  === 实验报告汇总 ==="
for f in results/report_*.json; do
    if [ -f "$f" ]; then
        echo ""
        echo "--- $(basename $f) ---"
        cat "$f"
    fi
done
