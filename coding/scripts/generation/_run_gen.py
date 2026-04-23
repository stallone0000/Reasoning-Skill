# -*- coding: utf-8 -*-
"""临时启动脚本，避免 PowerShell 中文路径编码问题。"""
import os
import sys

# 切换到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 直接 import 并运行
sys.argv = [
    "gen_doubao_seed.py",
    "--input", os.path.join(script_dir, "nemotron_cp_questions_34799_v1.jsonl"),
    "--output", os.path.join(script_dir, "doubao_seed_cp_34799_v1.jsonl"),
    "--workers", "500",
]

exec(open(os.path.join(script_dir, "gen_doubao_seed.py"), encoding="utf-8").read())
