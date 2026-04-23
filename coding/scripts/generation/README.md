# scripts/generation/ — 模型输出生成

调用 LLM API 生成解题输出和经验总结。

## 文件说明

| 文件 | 说明 |
|------|------|
| `_run_gen.py` | 通用生成入口脚本 |
| `gen_doubao_seed.py` | Doubao Seed 2.0 Pro 批量解题生成 |
| `gen_qiniu_oss.py` | Qiniu OSS (gpt-oss-120b) 批量解题生成 |
| `run_qiniu_oss_pipeline.sh` | Qiniu OSS 完整流水线（生成 → Judge → 统计） |
| `run_flash_experience_summarization.py` | 用 Gemini Flash 批量总结经验卡片 |
| `rerun_failed_summarization.py` | 重跑失败的经验总结任务 |

Note: the original private deployment helper is intentionally omitted from this sanitized release.

## 关键参数

- **并发数**: 通常 200 worker
- **超时**: 300-600s（不截断思维链时需更长超时）
- **Checkpoint**: 每 1000-2000 条保存一次
