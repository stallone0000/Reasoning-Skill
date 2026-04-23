# scripts/data_prep/ — 数据准备

从原始数据中提取和格式化实验所需的数据文件。

## 文件说明

| 文件 | 说明 |
|------|------|
| `export_nemotron_cp_unique.py` | 从 Nemotron 数据集导出去重后的题目和测试用例 |
| `get_nemotron_cp_unique_questions_34729_withimages.py` | 提取 34,729 道唯一题目（含图片信息） |

## 数据流

```
Nemotron 原始数据
    ↓ export_nemotron_cp_unique.py
nemotron_cp_questions_34799_v1.jsonl    (题目文本)
nemotron_cp_cases_34799_v1.jsonl        (测试用例)
    ↓ get_nemotron_cp_unique_questions_34729_withimages.py
nemotron_cp_unique_questions_34729_withimages.json (去重 + 图片)
```

输出文件存放于 `data/questions/`。
