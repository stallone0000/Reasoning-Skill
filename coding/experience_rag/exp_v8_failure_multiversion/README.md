# exp_v8_failure_multiversion

目标：专门解决“错题上没总结出有用经验”这个问题。

这条分支不直接再赌一个 prompt，而是同时维护多个 failure-card 版本，用统一任务集、统一 schema、统一报告来比较。

## 版本

- `counterexample_first`
  - 先抓最小反例模式、错误假设、提交前 guardrail。
- `invariant_guardrail`
  - 强调被破坏的状态/不变量，以及如何在推理早期识别。
- `retrieval_ready`
  - 更偏检索注入友好，尽量短、准、可复用。

## 文件

- `run_failure_card_variants.py`
  - 生成多版本 failure-card 结果。
  - 支持 `--write_prompts_only`，即使没有 API key 也能先生成 prompt manifest。
  - 内置稳定 seed、prompt 长度预算、quality gate。
- `analyze_failure_card_quality.py`
  - 对已有卡片文件做离线质量评估，重点看：
    - judge 泄漏
    - 类型污染
    - type-specific 字段完整性
    - retrieval-ready 比例
  - 可直接导出 `accepted/rejected` 子集。
- `card_quality.py`
  - 统一 quality gate 规则，供在线生成和离线分析复用。

## 推荐用法

```bash
# 先离线生成多个版本的 prompt manifest
python run_failure_card_variants.py --pilot 64 --write_prompts_only

# 有 API key 后再真跑
export API_KEY_360=...
python run_failure_card_variants.py --pilot 128 --variant_workers 3 --workers 16

# 对现有/新版本卡片做离线比较
python analyze_failure_card_quality.py \
  --input v3=../exp_v3_trap_detector/cards_v3_trap.jsonl \
  --input v5=../exp_v5_full_diagnostic/cards_v5_diagnostic.jsonl

# 顺手产出可直接给检索用的干净子集
python analyze_failure_card_quality.py \
  --input v5=../exp_v5_full_diagnostic/cards_v5_diagnostic.jsonl \
  --write_filtered_dir ../../../user_tasks/20260319_reasoning_memory_project_improvement/filtered_failure_cards
```

## 当前观察

- 旧 `v5` 卡片离线看比 `v3` 干净很多，但按更严格的 retrieval-ready 规则，仍有大量卡片会被 judge 泄漏或长文本拖累。
- `run_failure_card_variants.py` 现在会优先把 prompt 压到预算内，再对返回卡片做质量门，不再只检查“是不是合法 JSON”。
