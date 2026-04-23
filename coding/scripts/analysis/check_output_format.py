#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查输出格式"""

import json
import re
import sys

if len(sys.argv) < 2:
    print("用法: python3 check_output_format.py <jsonl_file>")
    sys.exit(1)

with open(sys.argv[1], 'r') as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        output = rec.get('llm_output', '')
        
        print(f"unique_key: {rec.get('unique_key')}")
        print(f"输出长度: {len(output)}")
        
        # 统计代码块
        cpp_blocks = len(re.findall(r'```(?:cpp|c\+\+)', output, re.IGNORECASE))
        python_blocks = len(re.findall(r'```python', output, re.IGNORECASE))
        rust_blocks = len(re.findall(r'```(?:rust|rs)', output, re.IGNORECASE))
        
        print(f"代码块统计:")
        print(f"  C++: {cpp_blocks}")
        print(f"  Python: {python_blocks}")
        print(f"  Rust: {rust_blocks}")
        
        # 查找第一个 C++ 代码块
        cpp_match = re.search(r'```(?:cpp|c\+\+)\s*\n(.*?)```', output, re.DOTALL | re.IGNORECASE)
        if cpp_match:
            cpp_code = cpp_match.group(1).strip()
            print(f"\n第一个 C++ 代码块长度: {len(cpp_code)}")
            print(f"前 200 字符:")
            print(cpp_code[:200])
        
        # 查找第一个 Python 代码块
        py_match = re.search(r'```python\s*\n(.*?)```', output, re.DOTALL | re.IGNORECASE)
        if py_match:
            py_code = py_match.group(1).strip()
            print(f"\n第一个 Python 代码块长度: {len(py_code)}")
            print(f"前 200 字符:")
            print(py_code[:200])
        
        break  # 只检查第一个
