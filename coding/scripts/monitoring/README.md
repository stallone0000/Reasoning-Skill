# scripts/monitoring/ — 进度监控

实验运行过程中的进度跟踪、自动监控和持续检查脚本。

## 文件说明

| 文件 | 说明 |
|------|------|
| `monitor_and_judge.sh` | 综合监控：生成完成后自动触发 Judge |
| `continuous_track.sh` | 持续跟踪脚本（循环检查进度） |
| `track_qiniu_oss_progress.py` | Qiniu OSS 实验进度跟踪 |

## 检查脚本（check_*.sh）

| 文件 | 检查对象 |
|------|---------|
| `check_doubao_judge.sh` | Doubao Judge 进度 |
| `check_qiniu_oss_progress.sh` | Qiniu OSS 生成进度 |
| `check_qiniu_oss_with_prompt_progress.sh` | Qiniu OSS（含 prompt 版本）进度 |
| `check_refill_progress.sh` | 补填进度（第 1 轮） |
| `check_refill_round2.sh` | 补填进度（第 2 轮） |
| `check_rejudge.sh` | 重新 Judge 进度 |

## 用法

```bash
# 持续跟踪（后台运行）
nohup bash continuous_track.sh &

# 手动检查某个任务进度
bash check_doubao_judge.sh
```
