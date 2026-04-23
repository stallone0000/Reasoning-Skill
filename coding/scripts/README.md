# scripts/ — 流水线脚本

本目录按功能分类存放项目的所有 Python/Shell 脚本。

## 子目录

| 目录 | 说明 | 文件数 |
|------|------|--------|
| `generation/` | 模型输出生成（调 API 跑题、经验总结生成） | 7 |
| `judge/` | 代码评判（本地编译运行 + 测试用例对比） | 11 |
| `refill/` | 补填空输出、修复 CE、重新 Judge | 6 |
| `analysis/` | 结果分析（统计、格式检查） | 5 |
| `monitoring/` | 进度监控与自动化跟踪 | 9 |
| `data_prep/` | 数据准备（导出、格式转换） | 2 |

## 典型工作流

```
data_prep/ → generation/ → judge/ → refill/ (如有问题) → analysis/
                                        ↑                    │
                                        └── monitoring/ ─────┘
```

## ⚠️ 注意

- 脚本中的文件路径可能仍指向旧的根目录位置，如需运行请先检查路径引用。
- 调 360 API 前务必 `unset http_proxy https_proxy all_proxy`。
- Python 侧公共运行时已收敛到 `../rm_runtime.py`。
- Shell 侧公共环境处理已收敛到 `common/qihoo_env.sh`。
