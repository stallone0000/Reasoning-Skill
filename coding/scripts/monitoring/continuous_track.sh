#!/bin/bash
# continuous_track.sh
# 持续跟踪任务进度，每5分钟更新一次统计信息

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Qiniu GPT-OSS-120B 持续跟踪"
echo "=========================================="
echo "每5分钟更新一次统计信息"
echo "按 Ctrl+C 停止"
echo ""

CHECK_INTERVAL=300  # 5分钟
CHECK_COUNT=0

cleanup() {
    echo ""
    echo "跟踪已停止，生成最终统计..."
    python3 generate_final_stats.py
    echo ""
    echo "最终统计文件:"
    echo "  - qiniu_oss_final_report.json"
    echo "  - qiniu_oss_final_report.md"
    echo "  - qiniu_oss_progress_stats.json"
    exit 0
}

trap cleanup SIGINT SIGTERM

while true; do
    CHECK_COUNT=$((CHECK_COUNT + 1))
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$TIMESTAMP] 检查 #$CHECK_COUNT"
    
    # 更新统计
    python3 generate_final_stats.py > /dev/null 2>&1
    
    # 显示当前状态
    GEN_LINES=$(wc -l < qiniu_oss_cp_34799_v1_with_prompt.jsonl 2>/dev/null || echo "0")
    JUDGE_LINES=$(wc -l < judge_results_qiniu_oss_with_prompt.jsonl 2>/dev/null || echo "0")
    GEN_RUNNING=$(pgrep -f "gen_qiniu_oss.py.*with_prompt" > /dev/null && echo "是" || echo "否")
    JUDGE_RUNNING=$(pgrep -f "run_judge_qiniu_oss.*with_prompt" > /dev/null && echo "是" || echo "否")
    
    GEN_PCT=$(echo "scale=2; $GEN_LINES * 100 / 34721" | bc)
    
    echo "  生成: $GEN_LINES / 34721 ($GEN_PCT%) - 运行中: $GEN_RUNNING"
    echo "  Judge: $JUDGE_LINES 条 - 运行中: $JUDGE_RUNNING"
    
    # 检查是否都完成了
    if [ "$GEN_RUNNING" = "否" ] && [ "$JUDGE_RUNNING" = "否" ]; then
        echo ""
        echo "所有任务已完成！"
        python3 generate_final_stats.py
        break
    fi
    
    echo "  等待 $CHECK_INTERVAL 秒后再次检查..."
    echo ""
    sleep $CHECK_INTERVAL
done
