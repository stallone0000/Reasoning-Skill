# -*- coding: utf-8 -*-
"""
run_judge_linux.py

Linux 服务器大规模并行 judge 启动器。
- 同时并行运行 Pro 和 Flash 两个 judge pipeline
- 每个 pipeline 内部使用多线程 (ThreadPoolExecutor)
- 支持断点续传 (已有结果会自动跳过)
- 充分利用多核 CPU

使用方法:
  # 同时运行 Pro + Flash (默认各 32 workers, 共 64)
  python3 run_judge_linux.py

  # 只运行 Pro
  python3 run_judge_linux.py --only pro

  # 只运行 Flash
  python3 run_judge_linux.py --only flash

  # 自定义 worker 数
  python3 run_judge_linux.py --workers 48

  # 测试模式 (只 judge 前 10 条)
  python3 run_judge_linux.py --limit 10

  # 重新运行 (清除旧结果)
  python3 run_judge_linux.py --fresh
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# 脚本所在目录
SCRIPT_DIR = Path(__file__).resolve().parent

# 数据文件路径
CASES_FILE = SCRIPT_DIR / "nemotron_cp_cases_34799_v1.jsonl"
PRO_INPUT = SCRIPT_DIR / "nemotron_cp_unique_questions_34729_withimages_pro.json"
FLASH_INPUT = SCRIPT_DIR / "nemotron_cp_unique_questions_34729_withimages_flash.json"
PRO_OUTPUT = SCRIPT_DIR / "judge_results_pro.jsonl"
FLASH_OUTPUT = SCRIPT_DIR / "judge_results_flash.jsonl"


def check_files():
    """检查所有必要文件是否存在。"""
    missing = []
    for name, path in [
        ("Cases", CASES_FILE),
        ("Pro input", PRO_INPUT),
        ("Flash input", FLASH_INPUT),
    ]:
        if not path.exists():
            missing.append(f"  {name}: {path}")
    if missing:
        print("错误: 以下文件缺失:")
        for m in missing:
            print(m)
        sys.exit(1)


def count_done(path: Path) -> int:
    """统计已完成的 judge 数量。"""
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def run_judge(
    name: str,
    input_path: Path,
    output_path: Path,
    workers: int,
    timeout: float,
    mem_mb: int,
    limit: int | None,
    fresh: bool,
):
    """启动一个 judge pipeline 子进程。"""
    if fresh and output_path.exists():
        print(f"[{name}] 清除旧结果: {output_path}")
        output_path.unlink()

    done = count_done(output_path)
    if done > 0:
        print(f"[{name}] 断点续传: 已有 {done} 条结果, 将跳过已完成的")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_judge_multiproc.py"),
        "--input", str(input_path),
        "--cases", str(CASES_FILE),
        "--output", str(output_path),
        "--workers", str(workers),
        "--timeout", str(timeout),
        "--mem_mb", str(mem_mb),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print(f"[{name}] 启动 judge pipeline:")
    print(f"  输入: {input_path.name}")
    print(f"  输出: {output_path.name}")
    print(f"  Workers: {workers}")
    print(f"  Timeout: {timeout}s")
    print()

    # 创建日志文件
    log_path = SCRIPT_DIR / f"judge_{name.lower()}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(SCRIPT_DIR),
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        },
    )
    return proc, log_file, log_path


def tail_log(log_path: Path, last_pos: int, prefix: str) -> int:
    """打印日志文件的新增内容。"""
    if not log_path.exists():
        return last_pos
    size = log_path.stat().st_size
    if size <= last_pos:
        return last_pos
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(last_pos)
        new_content = f.read()
        for line in new_content.strip().split("\n"):
            if line.strip():
                print(f"  [{prefix}] {line}")
    return size


def main():
    ap = argparse.ArgumentParser(description="Linux 大规模并行 Judge 启动器")
    ap.add_argument(
        "--only", choices=["pro", "flash"],
        help="只运行指定的 pipeline (默认两个都运行)"
    )
    ap.add_argument(
        "--workers", type=int, default=96,
        help="每个 pipeline 的 worker 进程数 (默认: 96)"
    )
    ap.add_argument(
        "--timeout", type=float, default=10.0,
        help="每个 test case 的超时时间 (秒, 默认: 10)"
    )
    ap.add_argument(
        "--mem_mb", type=int, default=1024,
        help="每次运行的内存限制 (MB, 默认: 1024)"
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="限制 judge 数量 (用于测试)"
    )
    ap.add_argument(
        "--fresh", action="store_true",
        help="清除旧结果, 重新运行"
    )
    args = ap.parse_args()

    check_files()

    print("=" * 60)
    print("Nemotron CP Judge - Linux 多线程并行版")
    print("=" * 60)
    print(f"CPU 核心数: {os.cpu_count()}")
    print(f"每个 pipeline workers: {args.workers}")
    print(f"Timeout: {args.timeout}s")
    print()

    processes = []
    log_files = []
    log_paths = []
    names = []

    t0 = time.time()

    if args.only != "flash":
        proc, lf, lp = run_judge(
            "Pro", PRO_INPUT, PRO_OUTPUT,
            args.workers, args.timeout, args.mem_mb,
            args.limit, args.fresh,
        )
        processes.append(proc)
        log_files.append(lf)
        log_paths.append(lp)
        names.append("Pro")

    if args.only != "pro":
        proc, lf, lp = run_judge(
            "Flash", FLASH_INPUT, FLASH_OUTPUT,
            args.workers, args.timeout, args.mem_mb,
            args.limit, args.fresh,
        )
        processes.append(proc)
        log_files.append(lf)
        log_paths.append(lp)
        names.append("Flash")

    # 监控进度
    print("=" * 60)
    print("运行中... (日志文件: judge_pro.log, judge_flash.log)")
    print("=" * 60)

    log_positions = [0] * len(processes)

    try:
        while True:
            all_done = True
            for i, proc in enumerate(processes):
                if proc.poll() is None:
                    all_done = False

                # 打印新的日志输出
                log_positions[i] = tail_log(
                    log_paths[i], log_positions[i], names[i]
                )

            if all_done:
                # 最后一次读取剩余日志
                for i in range(len(processes)):
                    tail_log(log_paths[i], log_positions[i], names[i])
                break

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n中断! 正在终止子进程...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait(timeout=30)
        print("已终止。已完成的结果已保存，下次运行可继续。")

    finally:
        for lf in log_files:
            lf.close()

    # 最终统计
    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"全部完成! 总耗时: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print("=" * 60)

    for i, name in enumerate(names):
        rc = processes[i].returncode
        out_path = PRO_OUTPUT if name == "Pro" else FLASH_OUTPUT
        done_count = count_done(out_path)
        status = "成功" if rc == 0 else f"退出码 {rc}"
        print(f"  [{name}] {status}, 共 {done_count} 条结果")

    print()
    print("结果文件:")
    if "Pro" in names:
        print(f"  Pro:   {PRO_OUTPUT}")
    if "Flash" in names:
        print(f"  Flash: {FLASH_OUTPUT}")


if __name__ == "__main__":
    main()
