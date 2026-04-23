# -*- coding: utf-8 -*-
"""
rejudge_pro_fast.py

高效重新 judge Gemini Pro 中因为 <code_block> 标签导致失败的案例。
只加载需要的记录，使用多进程加速。

用法:
  python3 rejudge_pro_fast.py
"""

import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_cp

# ── 全局变量 (fork 后子进程共享) ──
_CASES_IDX: Dict[str, dict] = {}
_CFG: judge_cp.JudgeConfig = judge_cp.JudgeConfig()
_ORIGINAL_DATA: Dict[str, dict] = {}

THOUGHT_END_RE = re.compile(
    r'<\s*/\s*(?:thought|think|analysis|reasoning)\s*>'
    r'|<\s*(?:thought|think)\s*/\s*>'
    r'|<\s*/\s*redacted_reasoning\s*>',
    re.IGNORECASE,
)

OUTPUT_KEYS = ['gemini_outputs', 'correct_gemini_outputs']


def first_output_text(rec: dict) -> Optional[str]:
    """从记录中提取输出文本。"""
    for k in OUTPUT_KEYS:
        v = rec.get(k)
        if isinstance(v, list) and v:
            if isinstance(v[0], str) and v[0].strip():
                return v[0]
    return None


def _worker_rejudge(uk: str) -> dict:
    """在子进程中重新 judge 一条记录。"""
    rec = _ORIGINAL_DATA.get(uk)
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
    
    # 1) 获取模型输出
    text = first_output_text(rec)
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
    
    # 2) 拆分 think / answer
    matches = list(THOUGHT_END_RE.finditer(text))
    if matches:
        best = max(matches, key=lambda m: m.start())
        answer_part = text[best.end():].strip()
    else:
        answer_part = text
    
    # 3) 查找 cases
    prob = _CASES_IDX.get(uk)
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
    
    # 4) judge (使用修复后的代码提取逻辑)
    try:
        raw = judge_cp.judge_one(prob, answer_part, _CFG)
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


def load_original_data_selective(input_path: Path, needed_keys: set) -> Dict[str, dict]:
    """只加载需要的记录。"""
    print(f"Loading original data from {input_path} (selective)...")
    data = {}
    count = 0
    
    # 流式读取 JSON array
    with input_path.open('r', encoding='utf-8') as f:
        ch = f.read(1)
        while ch and ch.isspace():
            ch = f.read(1)
        if ch != '[':
            raise ValueError(f'{input_path} is not a JSON array file')
        
        started = False
        depth = 0
        in_str = False
        esc = False
        buf = ''
        
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not started:
                if ch.isspace() or ch == ',':
                    continue
                if ch == ']':
                    break
                if ch != '{':
                    continue
                started = True
                depth = 1
                in_str = False
                esc = False
                buf = '{'
                continue
            
            buf += ch
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        item = json.loads(buf)
                        uk = item.get('unique_key')
                        if uk in needed_keys:
                            data[uk] = item
                            count += 1
                            if count % 100 == 0:
                                print(f"  loaded {count}/{len(needed_keys)} records...", end='\r')
                            if len(data) >= len(needed_keys):
                                break
                    except:
                        pass
                    started = False
                    buf = ''
    
    print(f"\n  loaded {len(data)} original records")
    return data


def main():
    # 配置路径
    results_file = Path('judge_results_pro.jsonl')
    original_file = Path('nemotron_cp_unique_questions_34729_withimages_pro.json')
    cases_file = Path('nemotron_cp_cases_34799_v1.jsonl')
    output_file = Path('judge_results_pro_rejudged.jsonl')
    keys_file = Path('rejudge_keys.txt')
    workers = 96
    
    global _CASES_IDX, _CFG, _ORIGINAL_DATA
    
    # 加载需要重新 judge 的 key
    print("Loading keys to rejudge...")
    needed_keys = set()
    if keys_file.exists():
        with keys_file.open('r') as f:
            for line in f:
                uk = line.strip()
                if uk:
                    needed_keys.add(uk)
    else:
        # 如果没有 keys 文件，从结果文件中提取
        print("Extracting keys from results file...")
        with results_file.open('r', encoding='utf-8') as f:
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
                            needed_keys.add(uk)
                except:
                    pass
    
    print(f"Found {len(needed_keys)} keys to rejudge")
    
    if not needed_keys:
        print("没有找到需要重新 judge 的案例。")
        return
    
    # 加载 cases
    print("Loading cases index...")
    _CASES_IDX = {}
    for row in judge_cp.iter_jsonl(str(cases_file)):
        k = row.get("unique_key")
        if k:
            _CASES_IDX[k] = row
    print(f"Loaded {len(_CASES_IDX)} cases")
    
    # 只加载需要的原始数据
    _ORIGINAL_DATA = load_original_data_selective(original_file, needed_keys)
    
    if len(_ORIGINAL_DATA) < len(needed_keys):
        print(f"Warning: 只找到 {len(_ORIGINAL_DATA)}/{len(needed_keys)} 条原始数据")
    
    # 配置
    _CFG = judge_cp.JudgeConfig(
        float_tol=1e-6,
        include_private=False,
        default_timeout_s=10.0,
        default_mem_mb=1024,
    )
    
    # 统计
    status_counter: Counter = Counter()
    t0 = time.time()
    n_done = 0
    n_total = len(needed_keys)
    
    # 打开输出文件
    fout = output_file.open("w", encoding="utf-8")
    
    print(f"\n开始重新 judge: {n_total} items, {workers} workers")
    print(f"{'='*60}")
    
    # 使用 multiprocessing.Pool + fork
    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(processes=workers, maxtasksperchild=200) as pool:
            for result in pool.imap_unordered(_worker_rejudge, list(needed_keys), chunksize=1):
                line = json.dumps(result, ensure_ascii=False) + "\n"
                fout.write(line)
                status_counter[result["judge"]["status"]] += 1
                n_done += 1
                
                # 每 10 条 flush 一次, 每 50 条打印进度
                if n_done % 10 == 0:
                    fout.flush()
                
                if n_done % 50 == 0 or n_done == n_total:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    remaining = n_total - n_done
                    eta = remaining / rate if rate > 0 else 0
                    eta_h = eta / 3600
                    pct = n_done / n_total * 100
                    
                    # 状态摘要
                    top_statuses = status_counter.most_common(5)
                    status_str = ", ".join(f"{s}:{c}" for s, c in top_statuses)
                    
                    print(
                        f"  [{n_done:>6d}/{n_total}] {pct:5.1f}%  "
                        f"{rate:5.1f} items/s  "
                        f"ETA {eta:.0f}s ({eta_h:.1f}h)  "
                        f"| {status_str}"
                    )
                    sys.stdout.flush()
    
    except KeyboardInterrupt:
        print("\n\n中断! 正在保存已完成的结果...")
        fout.flush()
        fout.close()
        print(f"已保存 {n_done} 条结果到 {output_file}")
        return
    
    fout.flush()
    fout.close()
    
    # 最终统计
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"重新 judge 完成: {n_done} items in {elapsed:.1f}s ({n_done/elapsed:.1f} items/s)")
    print(f"输出文件: {output_file}")
    print(f"\n状态分布:")
    for st, c in status_counter.most_common():
        pct = c / n_done * 100 if n_done else 0
        print(f"  {st:<30s} {c:>6d}  ({pct:5.1f}%)")
    
    # 对比原始结果
    print(f"\n{'='*60}")
    print("对比原始结果 (这些案例之前都是 CE/RE):")
    ac_count = status_counter.get('AC', 0)
    print(f"  AC (修复后通过): {ac_count} ({ac_count/n_done*100:.1f}%)")
    print(f"  其他状态: {n_done - ac_count} ({100 - ac_count/n_done*100:.1f}%)")


if __name__ == "__main__":
    main()
