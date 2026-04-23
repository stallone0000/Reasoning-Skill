#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 CE 错误的原因
"""

import json
import re
from collections import Counter

def analyze_ce_errors():
    # 加载 judge 结果
    judge_results = {}
    with open('judge_results_qiniu_oss.jsonl', 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                uk = rec.get('unique_key')
                judge = rec.get('judge', {})
                if judge.get('status') == 'CE':
                    judge_results[uk] = judge
            except:
                pass
    
    print(f"找到 {len(judge_results)} 个 CE 错误")
    
    # 加载原始输出
    outputs = {}
    with open('qiniu_oss_cp_34799_v1.jsonl', 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                uk = rec.get('unique_key')
                if uk in judge_results:
                    outputs[uk] = rec.get('llm_output', '')
            except:
                pass
    
    print(f"找到 {len(outputs)} 个对应的原始输出")
    
    # 分析问题
    rust_code_count = 0
    wrong_lang_tags = Counter()
    no_code_blocks = 0
    has_rust_syntax = 0
    
    for uk, output in list(outputs.items())[:200]:  # 检查前 200 个
        # 检查是否有 Rust 语法
        if 'use std::' in output or 'fn main()' in output or re.search(r'for\s+_\s+in\s+0\.\.', output):
            has_rust_syntax += 1
        
        # 检查代码块标记
        code_blocks = re.findall(r'```(\w+)', output, re.IGNORECASE)
        if not code_blocks:
            no_code_blocks += 1
        else:
            for lang in code_blocks:
                lang_lower = lang.lower()
                if lang_lower not in ['cpp', 'c++', 'cc', 'cxx', 'python', 'py']:
                    wrong_lang_tags[lang_lower] += 1
                    if lang_lower == 'rust' or lang_lower == 'rs':
                        rust_code_count += 1
    
    print(f"\n分析结果 (前 200 个 CE 样例):")
    print(f"  包含 Rust 语法: {has_rust_syntax}")
    print(f"  标记为 Rust 代码块: {rust_code_count}")
    print(f"  没有代码块标记: {no_code_blocks}")
    print(f"\n错误的语言标记分布:")
    for lang, count in wrong_lang_tags.most_common(10):
        print(f"  {lang}: {count}")
    
    # 展示几个具体例子
    print(f"\n具体例子:")
    count = 0
    for uk, output in outputs.items():
        if count >= 3:
            break
        if '```rust' in output.lower() or '```rs' in output.lower():
            print(f"\n样例 {count + 1}: {uk}")
            # 找到 Rust 代码块
            rust_match = re.search(r'```(?:rust|rs)\s*\n(.*?)```', output, re.DOTALL | re.IGNORECASE)
            if rust_match:
                rust_code = rust_match.group(1)[:500]
                print(f"Rust 代码预览:\n{rust_code}")
            count += 1

if __name__ == '__main__':
    analyze_ce_errors()
