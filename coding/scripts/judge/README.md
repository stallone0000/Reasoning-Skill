# scripts/judge/ — 代码评判

本地编译运行模型生成的代码，与测试用例对比输出，判定 AC/WA/CE/TLE/RE 等状态。

## 核心文件

| 文件 | 说明 |
|------|------|
| `judge_cp.py` | **核心 Judge 逻辑**：代码提取、编译、运行、结果对比 |
| `judge_cp_multiproc_gens.py` | 多进程 Judge（处理大规模生成结果） |
| `run_judge_pipeline.py` | Judge 完整流水线（从 JSONL 读取 → 评判 → 输出） |
| `run_judge_multiproc.py` | 多进程 Judge 运行器 |
| `run_judge_linux.py` | Linux 环境适配的 Judge 运行器 |

## 各模型 Judge 入口

| 文件 | 说明 |
|------|------|
| `_run_flash_judge.py` | Gemini Flash 结果评判 |
| `_run_pro_judge.py` | Gemini Pro 结果评判 |
| `run_judge_doubao.py` | Doubao 结果评判（高性能多进程） |
| `run_doubao_judge.sh` | Doubao Judge Shell 启动器 |
| `run_judge_qiniu_oss.py` | Qiniu OSS 结果评判 |

## 测试

| 文件 | 说明 |
|------|------|
| `test_judge_lang_detection.py` | Judge 语言检测单元测试 |

## Judge 状态说明

| 状态 | 含义 |
|------|------|
| `AC` | Accepted — 全部测试通过 |
| `WA` | Wrong Answer — 输出不匹配 |
| `CE` | Compile Error — 编译失败 |
| `RE` | Runtime Error — 运行时错误 |
| `TLE` | Time Limit Exceeded — 超时 |
| `WA_CHECKER` | 自定义 Checker 判定错误 |
| `NO_TESTS` | 无可用测试用例 |
| `NO_OUTPUT` | 模型未产出有效代码 |
