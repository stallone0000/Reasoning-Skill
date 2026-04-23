#!/bin/bash
# check_qiniu_oss_with_prompt_progress.sh
# 检查带 prompt 的 qiniu_oss 生成进度

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Qiniu GPT-OSS-120B (带 Prompt) 任务进度"
echo "=========================================="
echo ""

# 检查生成任务
echo "1. 生成任务状态:"
if pgrep -f "gen_qiniu_oss.py.*with_prompt" > /dev/null; then
    echo "   ✓ 生成任务正在运行"
    echo ""
    echo "   生成进度:"
    if [ -f "qiniu_oss_cp_34799_v1_with_prompt.jsonl" ]; then
        GENERATED=$(wc -l < qiniu_oss_cp_34799_v1_with_prompt.jsonl)
        TOTAL=34721
        PERCENT=$(echo "scale=2; $GENERATED * 100 / $TOTAL" | bc)
        echo "   已生成: $GENERATED / $TOTAL ($PERCENT%)"
        echo "   文件大小: $(du -h qiniu_oss_cp_34799_v1_with_prompt.jsonl | cut -f1)"
    else
        echo "   输出文件尚未生成"
    fi
    echo ""
    echo "   最近日志 (最后 5 行):"
    tail -5 gen_qiniu_oss_with_prompt.log 2>/dev/null | sed 's/^/   /'
else
    echo "   ✗ 生成任务未运行"
    if [ -f "qiniu_oss_cp_34799_v1_with_prompt.jsonl" ]; then
        GENERATED=$(wc -l < qiniu_oss_cp_34799_v1_with_prompt.jsonl)
        echo "   已完成生成: $GENERATED 条"
    fi
fi
echo ""

# 检查 judge 任务
echo "2. Judge 任务状态:"
if pgrep -f "run_judge_qiniu_oss.*with_prompt" > /dev/null; then
    echo "   ✓ Judge 任务正在运行"
    if [ -f "judge_results_qiniu_oss_with_prompt.jsonl" ]; then
        JUDGED=$(wc -l < judge_results_qiniu_oss_with_prompt.jsonl)
        echo "   已 judge: $JUDGED 条"
        echo "   文件大小: $(du -h judge_results_qiniu_oss_with_prompt.jsonl | cut -f1)"
    fi
else
    echo "   ✗ Judge 任务未运行"
    if [ -f "judge_results_qiniu_oss_with_prompt.jsonl" ]; then
        JUDGED=$(wc -l < judge_results_qiniu_oss_with_prompt.jsonl)
        echo "   已完成 judge: $JUDGED 条"
    fi
fi
echo ""

# 检查报告
echo "3. 报告文件:"
if [ -f "judge_report_qiniu_oss_with_prompt.md" ]; then
    echo "   ✓ 报告已生成: judge_report_qiniu_oss_with_prompt.md"
    echo "   文件大小: $(du -h judge_report_qiniu_oss_with_prompt.md | cut -f1)"
else
    echo "   ✗ 报告尚未生成"
fi
echo ""

echo "=========================================="
echo "查看完整日志:"
echo "  生成日志: tail -f gen_qiniu_oss_with_prompt.log"
echo "  Judge日志: tail -f judge_qiniu_oss_with_prompt.log (如果存在)"
echo "=========================================="
