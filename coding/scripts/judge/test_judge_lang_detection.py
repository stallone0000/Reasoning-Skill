#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 judge_cp.py 的语言检测功能
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_cp

def test_rust_detection():
    """测试 Rust 代码检测"""
    print("测试 Rust 代码检测...")
    
    # 测试 1: Rust 代码块
    rust_code1 = """```rust
use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    for _ in 0..10 {
        println!("Hello");
    }
}
```"""
    
    hint, code = judge_cp.extract_code(rust_code1)
    lang = judge_cp.guess_language(code, hint)
    print(f"  测试 1: hint={hint}, lang={lang}")
    assert lang == "unsupported" or hint is None, f"应该检测为不支持的语言，但得到 lang={lang}, hint={hint}"
    
    # 测试 2: Rust 代码但没有代码块标记
    rust_code2 = """use std::io::{self, Read};

fn main() {
    for _ in 0..10 {
        println!("Hello");
    }
}"""
    
    hint, code = judge_cp.extract_code(rust_code2)
    lang = judge_cp.guess_language(code, hint)
    print(f"  测试 2: hint={hint}, lang={lang}")
    assert lang == "unsupported", f"应该检测为不支持的语言，但得到 lang={lang}"
    
    # 测试 3: C++ 代码（应该正常识别）
    cpp_code = """```cpp
#include <iostream>
using namespace std;

int main() {
    for (int i = 0; i < 10; i++) {
        cout << "Hello" << endl;
    }
    return 0;
}
```"""
    
    hint, code = judge_cp.extract_code(cpp_code)
    lang = judge_cp.guess_language(code, hint)
    print(f"  测试 3: hint={hint}, lang={lang}")
    assert lang == "cpp", f"应该检测为 C++，但得到 lang={lang}"
    
    # 测试 4: Python 代码（应该正常识别）
    python_code = """```python
def main():
    for i in range(10):
        print("Hello")

if __name__ == "__main__":
    main()
```"""
    
    hint, code = judge_cp.extract_code(python_code)
    lang = judge_cp.guess_language(code, hint)
    print(f"  测试 4: hint={hint}, lang={lang}")
    assert lang == "python", f"应该检测为 Python，但得到 lang={lang}"
    
    print("✓ 所有测试通过！")

def test_judge_one_with_rust():
    """测试 judge_one 对 Rust 代码的处理"""
    print("\n测试 judge_one 对 Rust 代码的处理...")
    
    problem = {
        "unique_key": "test/rust",
        "cases": {
            "kind": "stdio",
            "tests": [
                {"input": "1", "output": "1"}
            ]
        },
        "judge": {
            "mode": "stdio"
        }
    }
    
    rust_output = """```rust
use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!("{}", input.trim());
}
```"""
    
    cfg = judge_cp.JudgeConfig()
    result = judge_cp.judge_one(problem, rust_output, cfg)
    
    print(f"  状态: {result.get('status')}")
    print(f"  语言: {result.get('lang')}")
    print(f"  详情: {result.get('detail', '')[:100]}")
    
    assert result.get('status') == 'UNSUPPORTED_LANG', f"应该返回 UNSUPPORTED_LANG，但得到 {result.get('status')}"
    print("✓ 测试通过！")

if __name__ == '__main__':
    test_rust_detection()
    test_judge_one_with_rust()
    print("\n所有测试完成！")
