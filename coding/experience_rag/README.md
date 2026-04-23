# experience_rag/ — 主实验工作区

这个目录是 `reasoning_memory` 里最核心、也最容易变乱的子树。这里同时放了：

- 当前主线实验代码
- 多轮失败卡片/检索迭代分支
- 检索索引数据包
- 各类结果与日志

为了降低“看不出来哪个还在用”的成本，建议按下面的心智模型导航。

## 当前主线

- 当前推荐主线：`exp_v5_full_diagnostic/`
- 当前结论入口：`../LATEST_STATUS.md`
- 主要公共入口脚本：
  - `step4_inference_and_judge.py`
  - `step4b_inference_v2.py`
  - `run_gating_ablation.py`
  - `run_ablation_1k300.sh`

## 目录分层

### Core

- `retrieval.py`
- `step*.py`
- `run_*.py`

说明：核心推理、索引构建、实验入口。

### Current Variant Branches

- `exp_v5_full_diagnostic/`
- `exp_v7_v5_dynamic_gate/`
- `exp_v8_failure_multiversion/`

说明：当前还值得继续迭代或对照的分支。
其中 `exp_v8_failure_multiversion/` 不是新的单一路线结论，而是“并行试多个 failure-card 版本 + quality gate”的实验工作台。

### Legacy Variant Branches

- `exp_v3_trap_detector/`
- `exp_v4_no_failure/`
- `exp_v6_additive_edgefix/`

说明：历史中间版，保留以便回溯，不应默认当成当前主线。

### Index/Data Packs

- `data/`
- `data_v2/`
- `data_ablation/`
- `exp_v3_trap_detector/data_v3/`
- `exp_v5_full_diagnostic/data_v5/`
- `exp_v6_additive_edgefix/data_v6/`

说明：不同版本的检索索引和持久化卡片数据。

### Results

- `results/`
- `results_v2/`
- `results_newmodel/`
- `results_gating_ablation/`
- `results_ablation_1k300/`

说明：正式输出。smoke/临时结果优先放到所属结果树下面，而不是继续在本层散落。

### Logs

- `logs/`
- `logs/gating_ablation/`

说明：运行日志统一放日志树，避免和代码目录并列散开。

## 当前整理约定

- 新实验分支尽量使用 `exp_vX_*` 命名，并在目录内放 `README.md`。
- 新 smoke 结果尽量挂到正式结果目录下面。
- 需要保留旧路径兼容时，优先用软链接而不是复制一份新旧产物。
- 新的 wrong-answer 经验分支应优先输出 `accepted/rejected` 两类卡片，避免把低质量失败经验直接混进检索库。
