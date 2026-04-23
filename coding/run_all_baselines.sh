#!/bin/bash
# ============================================================
# run_all_baselines.sh
#
# 并行启动 3 个模型的实验 (每个模型内的 4 prompt 串行执行)
# 这样每个模型同时只有 200 并发请求, 避免 API 限流
#
# 架构:
#   model_oss120b (4 prompts 串行) ─┐
#   model_doubao  (4 prompts 串行) ─┼─ 并行
#   model_gemini  (4 prompts 串行) ─┘
#
# Prompt 条件 (所有条件均包含 CP 指令):
#   direct : 直接调用 (CP 指令 + 问题, 无额外推理策略)
#   nowait : NoWait (禁止等待词)
#   cod    : CoD (Chain of Draft)
#   tale   : TALE (budget 预估 + 约束解题)
#
# 用法:
#   bash run_all_baselines.sh           # 跑全量 34721 题
#   bash run_all_baselines.sh 100       # 测试: 每个实验只跑 100 题
#   bash run_all_baselines.sh 0 judge   # 只跑 judge (生成已完成)
#   bash run_all_baselines.sh 0 stats   # 只跑统计
# ============================================================

set -euo pipefail

# 清除代理 (360 API 关键规则)
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON=python3

# 参数
LIMIT="${1:-}"          # 第一个参数: 限制数量 (空 = 全量)
ACTION="${2:-all}"      # 第二个参数: all / gen / judge / stats

# 统一 200 并发 (1000 条够用)
WORKERS=200
INPUT="${SCRIPT_DIR}/cp_test_1000.jsonl"
CASES="${PARENT_DIR}/nemotron_cp_cases_34799_v1.jsonl"
JUDGE_SCRIPT="${PARENT_DIR}/run_judge_qiniu_oss.py"
JUDGE_WORKERS=96

MODELS=(
    "qiniu/gpt-oss-120b"
    "volcengine/doubao-seed-2-0-pro"
    "cloudsway/gemini-3-flash-preview"
)

PROMPTS=(
    "direct"
    "nowait"
    "cod"
    "tale"
)

MODEL_SHORTS=(
    "oss120b"
    "doubao"
    "gemini"
)

LIMIT_ARG=""
if [[ -n "$LIMIT" && "$LIMIT" != "0" ]]; then
    LIMIT_ARG="--limit $LIMIT"
    echo "============================================="
    echo "  测试模式: 每个实验只跑 $LIMIT 题"
    echo "============================================="
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# ============================================================
# 单模型生成函数: 对该模型串行跑 4 个 prompt
# ============================================================
run_model_gen() {
    local model="$1"
    local short="$2"
    echo "[${short}] 开始生成 (workers=${WORKERS}), $(date '+%H:%M:%S')"
    for prompt in "${PROMPTS[@]}"; do
        local log="${LOG_DIR}/gen_${prompt}_${short}.log"
        local out="${SCRIPT_DIR}/gen_${prompt}_${short}.jsonl"

        # 检查是否已完成
        if [[ -f "$out" ]]; then
            local existing
            existing=$(wc -l < "$out" 2>/dev/null || echo 0)
            if [[ "$existing" -ge 950 ]]; then
                echo "[${short}] 跳过 ${prompt} (已有 ${existing} 条)"
                continue
            fi
        fi

        echo "[${short}] 开始 ${prompt} (workers=${WORKERS}), $(date '+%H:%M:%S')"
        $PYTHON "${SCRIPT_DIR}/run_cp_baselines.py" \
            --model "$model" \
            --prompt "$prompt" \
            --input "$INPUT" \
            --workers "$WORKERS" \
            $LIMIT_ARG \
            > "$log" 2>&1 || echo "[${short}] WARN: ${prompt} 退出码非 0"

        local done_n
        done_n=$(wc -l < "$out" 2>/dev/null || echo 0)
        echo "[${short}] 完成 ${prompt}: ${done_n} 条, $(date '+%H:%M:%S')"
    done
    echo "[${short}] 所有生成完成! $(date '+%H:%M:%S')"
}

# ============================================================
# Phase 1: 生成 (3 个模型并行, 每个模型内 4 prompt 串行)
# ============================================================
if [[ "$ACTION" == "all" || "$ACTION" == "gen" ]]; then
    echo ""
    echo "============================================="
    echo "  Phase 1: 生成 (3 models × 4 prompts)"
    echo "  每模型 workers=${WORKERS}, 4 prompt 串行"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================="

    GEN_PIDS=()
    for i in "${!MODELS[@]}"; do
        model="${MODELS[$i]}"
        short="${MODEL_SHORTS[$i]}"
        model_log="${LOG_DIR}/model_${short}.log"

        run_model_gen "$model" "$short" > "$model_log" 2>&1 &
        GEN_PIDS+=($!)
        echo "  启动模型: ${short} (PID: ${GEN_PIDS[-1]})"
    done

    echo ""
    echo "  已启动 ${#GEN_PIDS[@]} 个模型生成流程 (每个内部串行 4 prompt)"
    echo "  等待所有生成完成..."
    echo ""

    FAIL_COUNT=0
    for pid in "${GEN_PIDS[@]}"; do
        if ! wait "$pid"; then
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done

    echo ""
    echo "  生成完成! 失败: $FAIL_COUNT / ${#GEN_PIDS[@]}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
fi

# ============================================================
# Phase 2: Judge (12 个串行执行, 因为 judge 本身是多进程)
# ============================================================
if [[ "$ACTION" == "all" || "$ACTION" == "judge" ]]; then
    echo ""
    echo "============================================="
    echo "  Phase 2: Judge (12 experiments)"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================="

    JUDGE_LIMIT_ARG=""
    if [[ -n "$LIMIT" && "$LIMIT" != "0" ]]; then
        JUDGE_LIMIT_ARG="--limit $LIMIT"
    fi

    for prompt in "${PROMPTS[@]}"; do
        for i in "${!MODELS[@]}"; do
            short="${MODEL_SHORTS[$i]}"
            gen_file="${SCRIPT_DIR}/gen_${prompt}_${short}.jsonl"
            judge_file="${SCRIPT_DIR}/judge_${prompt}_${short}.jsonl"
            log="${LOG_DIR}/judge_${prompt}_${short}.log"

            if [[ ! -f "$gen_file" ]]; then
                echo "  跳过: $gen_file 不存在"
                continue
            fi

            # 检查是否已有 judge 结果
            if [[ -f "$judge_file" ]]; then
                existing=$(wc -l < "$judge_file" 2>/dev/null || echo 0)
                gen_count=$(wc -l < "$gen_file" 2>/dev/null || echo 0)
                if [[ "$existing" -ge "$gen_count" && "$gen_count" -gt 0 ]]; then
                    echo "  跳过: judge_${prompt}_${short} 已完成 ($existing/$gen_count)"
                    continue
                fi
            fi

            echo "  Judge: ${prompt}/${short} (workers=$JUDGE_WORKERS)"

            $PYTHON "$JUDGE_SCRIPT" \
                --input "$gen_file" \
                --cases "$CASES" \
                --output "$judge_file" \
                --workers "$JUDGE_WORKERS" \
                $JUDGE_LIMIT_ARG \
                > "$log" 2>&1 || echo "  [WARN] judge_${prompt}_${short} 退出码非 0"

            echo "    完成: $(wc -l < "$judge_file" 2>/dev/null || echo 0) 条"
        done
    done

    echo ""
    echo "  Judge 全部完成!"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
fi

# ============================================================
# Phase 3: 统计
# ============================================================
if [[ "$ACTION" == "all" || "$ACTION" == "stats" ]]; then
    echo ""
    echo "============================================="
    echo "  Phase 3: 统计"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================="

    $PYTHON "${SCRIPT_DIR}/compute_baseline_stats.py" --dir "$SCRIPT_DIR"

    echo ""
    echo "  统计完成!"
fi

echo ""
echo "============================================="
echo "  全部完成! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
