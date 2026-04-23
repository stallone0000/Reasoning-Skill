# scripts/refill/ — 补填与修复

处理模型生成中的异常情况：空输出补填、CE 错误修复、重新 Judge。

## 文件说明

| 文件 | 说明 |
|------|------|
| `refill_empty_outputs.py` | 重新调用 API 补填空输出（主要针对 Doubao NO_OUTPUT） |
| `auto_refill_until_ready.sh` | 自动循环补填直到所有空输出都补完 |
| `rejudge_ce_and_merge.py` | 对 CE 错误重新解析代码块 + 重新 Judge + 合并结果 |
| `rejudge_code_block_errors.py` | 修复代码块提取错误后重新 Judge |
| `rejudge_pro_codeblock.py` | 专门针对 Pro 模型的代码块修复 |
| `rejudge_pro_fast.py` | Pro 模型快速重 Judge |

## 典型场景

1. **Doubao 空输出**: API 返回空 → `refill_empty_outputs.py` 重新请求 → 重新 Judge
2. **CE 修复**: 代码提取逻辑导致的假 CE → `rejudge_ce_and_merge.py` 重新解析 → 合并到最终结果
3. **代码块错误**: markdown 格式导致代码提取失败 → `rejudge_code_block_errors.py` 修复
