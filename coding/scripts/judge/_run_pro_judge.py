"""启动 Pro judge pipeline (处理好 PATH 和工作目录)"""
import subprocess, sys, os
from pathlib import Path

# 找到 coding_judge 目录
base = list(Path('H:/').iterdir())[0]
mac = [x for x in base.iterdir() if 'Mac' in x.name][0]
cj = mac / 'data flow' / 'coding_judge'

# 确保 g++ 在 PATH 中
env = os.environ.copy()
try:
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
        machine_path = winreg.QueryValueEx(key, "Path")[0]
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
        user_path = winreg.QueryValueEx(key, "Path")[0]
    env['PATH'] = machine_path + ';' + user_path
except Exception:
    pass

print(f"Working dir: {cj}")
subprocess.run(
    [sys.executable, str(cj / 'run_judge_pipeline.py'),
     '--input', str(cj / 'nemotron_cp_unique_questions_34729_withimages_pro.json'),
     '--cases', str(cj / 'nemotron_cp_cases_34799_v1.jsonl'),
     '--output', str(cj / 'judge_results_pro.jsonl'),
     '--workers', '4',
     '--timeout', '10',
    ],
    env=env,
    cwd=str(cj),
)
