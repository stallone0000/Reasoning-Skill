# -*- coding: utf-8 -*-
"""
rejudge_code_block_errors.py

重新 judge 那些因为 <code_block> 标签导致编译错误的案例。
这些案例在修复代码提取逻辑后应该能正确提取代码。

用法:
  python3 rejudge_code_block_errors.py --input judge_results_pro.jsonl --output judge_results_pro_fixed.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_cp

def load_cases_index(cases_path: Path) -> Dict[str, dict]:
    """加载测试用例索引。"""
    print(f"Loading cases index from {cases_path} ...")
    idx: Dict[str, dict] = {}
    for row in judge_cp.iter_jsonl(str(cases_path)):
        k = row.get("unique_key")
        if k:
            idx[k] = row
    print(f"  loaded {len(idx)} unique cases")
    return idx

def load_original_data(input_path: Path) -> Dict[str, dict]:
    """加载原始 Gemini 输出数据。"""
    print(f"Loading original data from {input_path} ...")
    data = {}
    
    # 尝试作为 JSON array 读取
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.strip().startswith('['):
                items = json.loads(content)
                for item in items:
                    uk = item.get('unique_key')
                    if uk:
                        data[uk] = item
    except Exception as e:
        print(f"  Error loading as JSON array: {e}")
    
    print(f"  loaded {len(data)} original records")
    return data

def find_code_block_errors(results_path: Path) -> Set[str]:
    """找出所有因为 <code_block> 标签导致错误的 unique_key。"""
    print(f"Scanning {results_path} for code_block errors...")
    error_keys = set()
    
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                judge = data.get('judge', {})
                status = judge.get('status', '')
                detail = judge.get('detail', '')
                
                if status in ('CE', 'RE') and ('<code_block>' in detail or 'code_block' in detail.lower()):
                    uk = data.get('unique_key')
                    if uk:
                        error_keys.add(uk)
            except:
                pass
    
    print(f"  found {len(error_keys)} cases with code_block errors")
    return error_keys

def rejudge_one(uk: str, original_data: Dict[str, dict], cases_idx: Dict[str, dict], cfg: judge_cp.JudgeConfig) -> dict:
    """重新 judge 一条记录。"""
    rec = original_data.get(uk)
    if not rec:
        return {
            "unique_key": uk,
            "judge": {
                "status": "NO_ORIGINAL_DATA",
                "lang": None,
                "passed": 0,
                "total_tests": 0,
                "case_results": [],
                "detail": "Original data not found",
            },
        }
    
    # 获取输出文本
    text = None
    for key in ['gemini_outputs', 'correct_gemini_outputs']:
        v = rec.get(key)
        if isinstance(v, list) and v:
            if isinstance(v[0], str) and v[0].strip():
                text = v[0]
                break
    
    if not text:
        return {
            "unique_key": uk,
            "judge": {
                "status": "NO_OUTPUT",
                "lang": None,
                "passed": 0,
                "total_tests": 0,
                "case_results": [],
                "detail": None,
            },
        }
    
    # 拆分 think/answer
    import re
    THOUGHT_END_RE = re.compile(
        r'<\s*/\s*(?:thought|think|analysis|reasoning)\s*>'
        r'|<\s*(?:thought|think)\s*/\s*>'
        r'|<\s*/\s*redacted_reasoning\s*>',
        re.IGNORECASE,
    )
    
    matches = list(THOUGHT_END_RE.finditer(text))
    if matches:
        best = max(matches, key=lambda m: m.start())
        answer_part = text[best.end():].strip()
    else:
        answer_part = text
    
    # 查找 cases
    prob = cases_idx.get(uk)
    if not prob:
        return {
            "unique_key": uk,
            "judge": {
                "status": "NO_CASES",
                "lang": None,
                "passed": 0,
                "total_tests": 0,
                "case_results": [],
                "detail": f"unique_key '{uk}' not found in cases file",
            },
        }
    
    # judge
    try:
        raw = judge_cp.judge_one(prob, answer_part, cfg)
        status = raw.get("status", "UNKNOWN")
        lang = raw.get("lang")
        passed = raw.get("passed", 0)
        total_tests = raw.get("total_tests", 0)
        case_results = raw.get("case_results", [])
        
        detail_parts = []
        if raw.get("compile_error"):
            detail_parts.append(f"[compile_error] {raw['compile_error']}")
        if raw.get("detail"):
            detail_parts.append(f"[detail] {raw['detail']}")
        if raw.get("error"):
            detail_parts.append(f"[error] {raw['error']}")
        detail = "\n".join(detail_parts) if detail_parts else None
        
        return {
            "unique_key": uk,
            "judge": {
                "status": status,
                "lang": lang,
                "passed": passed,
                "total_tests": total_tests,
                "case_results": case_results,
                "detail": detail,
            },
        }
    except Exception as e:
        return {
            "unique_key": uk,
            "judge": {
                "status": "JUDGE_ERROR",
                "lang": None,
                "passed": 0,
                "total_tests": 0,
                "case_results": [],
                "detail": str(e)[:2000],
            },
        }

def main():
    parser = argparse.ArgumentParser(description="重新 judge code_block 错误案例")
    parser.add_argument("--results", required=True, help="原始 judge 结果文件")
    parser.add_argument("--original", required=True, help="原始 Gemini 输出文件 (JSON array)")
    parser.add_argument("--cases", required=True, help="测试用例文件")
    parser.add_argument("--output", required=True, help="输出文件")
    parser.add_argument("--workers", type=int, default=32, help="并行 worker 数")
    args = parser.parse_args()
    
    # 加载数据
    cases_idx = load_cases_index(Path(args.cases))
    original_data = load_original_data(Path(args.original))
    error_keys = find_code_block_errors(Path(args.results))
    
    if not error_keys:
        print("没有找到需要重新 judge 的案例。")
        return
    
    # 配置
    cfg = judge_cp.JudgeConfig(
        float_tol=1e-6,
        include_private=False,
        default_timeout_s=10.0,
        default_mem_mb=1024,
    )
    
    # 重新 judge
    print(f"\n重新 judge {len(error_keys)} 条记录...")
    results = []
    for i, uk in enumerate(sorted(error_keys)):
        result = rejudge_one(uk, original_data, cases_idx, cfg)
        results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"  完成 {i+1}/{len(error_keys)}")
    
    # 保存结果
    print(f"\n保存结果到 {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 统计
    from collections import Counter
    status_counter = Counter()
    for r in results:
        status_counter[r['judge']['status']] += 1
    
    print(f"\n重新 judge 结果统计:")
    for status, count in status_counter.most_common():
        pct = count / len(results) * 100 if results else 0
        print(f"  {status:<30s} {count:>6d}  ({pct:5.1f}%)")

if __name__ == "__main__":
    main()
