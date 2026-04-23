#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成空输出的记录
"""

import os
import json
import time
import argparse
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 复用 gen_qiniu_oss.py 中的函数
import sys
sys.path.insert(0, os.path.dirname(__file__))
from gen_qiniu_oss import (
    DEFAULT_MODEL,
    iter_jsonl,
    process_one,
    resolve_api_key,
    QihooChatClient,
)

CLIENT = None

def main():
    parser = argparse.ArgumentParser(description="重新生成空输出的记录")
    parser.add_argument("--input", default="qiniu_oss_cp_34799_v1_with_prompt.jsonl", help="输入 JSONL 文件路径")
    parser.add_argument("--output", default="qiniu_oss_cp_34799_v1_with_prompt.jsonl", help="输出 JSONL 文件路径（覆盖）")
    parser.add_argument("--empty_keys", default="empty_output_keys.txt", help="空输出 key 列表文件")
    parser.add_argument("--questions", default="nemotron_cp_questions_34799_v1.jsonl", help="题目文件")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--workers", type=int, default=200, help="并发线程数（降低以避免资源竞争）")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    parser.add_argument("--api_url", type=str, default=None, help="API URL")
    parser.add_argument("--api_host", type=str, default=None, help="Host 头")
    args = parser.parse_args()

    global CLIENT
    try:
        api_key = resolve_api_key(args.api_key, required=True)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return
    CLIENT = QihooChatClient(api_key=api_key, api_url=args.api_url, api_host=args.api_host)

    import gen_qiniu_oss as gen_qiniu_oss_module
    gen_qiniu_oss_module.CLIENT = CLIENT

    # 1) 加载空输出的 key
    empty_keys = set()
    if os.path.exists(args.empty_keys):
        with open(args.empty_keys, 'r') as f:
            for line in f:
                key = line.strip()
                if key:
                    empty_keys.add(key)
    else:
        # 从输入文件中找出空输出
        print(f"[INFO] 从 {args.input} 中查找空输出...")
        with open(args.input, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    output = rec.get('llm_output', '')
                    if not output or not output.strip():
                        empty_keys.add(rec.get('unique_key'))
                except:
                    pass
    
    print(f"[INFO] 找到 {len(empty_keys)} 个空输出记录需要重新生成")

    if not empty_keys:
        print("[INFO] 没有需要重新生成的记录")
        return

    # 2) 加载题目
    print(f"[INFO] 读取题目文件: {args.questions}")
    questions_map = {}
    for row in iter_jsonl(args.questions):
        uk = row.get("unique_key", "")
        if uk in empty_keys:
            questions_map[uk] = row.get("question", "")

    print(f"[INFO] 找到 {len(questions_map)} 个对应的题目")

    # 3) 加载已有输出（用于合并）
    existing_outputs = {}
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    uk = rec.get("unique_key")
                    output = rec.get("llm_output", "")
                    if uk and (uk not in empty_keys or (output and output.strip())):
                        # 保留非空输出或不在重试列表中的记录
                        existing_outputs[uk] = rec
                except:
                    pass

    print(f"[INFO] 已有 {len(existing_outputs)} 条有效记录")

    # 4) 准备待处理任务
    todo_items = []
    for uk, question in questions_map.items():
        if not question.strip():
            continue
        todo_items.append((uk, question))

    if not todo_items:
        print("[INFO] 没有需要处理的条目")
        return

    # 5) 线程安全写入
    write_lock = threading.Lock()
    temp_output = args.output + ".tmp"
    fout = open(temp_output, "w", encoding="utf-8")
    n_success = 0
    n_fail = 0

    # 先写入已有记录
    for uk, rec in existing_outputs.items():
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        fout.write(line)

    def write_result(unique_key, llm_output):
        nonlocal n_success, n_fail
        if llm_output is None or not llm_output.strip():
            n_fail += 1
            return
        
        row = {
            "unique_key": unique_key,
            "llm_output": llm_output,
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with write_lock:
            fout.write(line)
            n_success += 1

    # 6) 并发调用
    t0 = time.time()
    print(f"[INFO] 开始重新生成，模型={args.model}，并发={args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, uk, q, args.model): uk
            for uk, q in todo_items
        }

        with tqdm(total=len(futures), desc="Refilling", unit="item") as pbar:
            for fut in as_completed(futures):
                uk = futures[fut]
                try:
                    unique_key, llm_output = fut.result()
                    write_result(unique_key, llm_output)
                except Exception as e:
                    print(f"[EXCEPTION] unique_key={uk}, error={e}")
                    n_fail += 1

                pbar.update(1)

                if (pbar.n) % 200 == 0:
                    with write_lock:
                        fout.flush()

    fout.close()

    # 7) 替换原文件
    if os.path.exists(temp_output):
        os.replace(temp_output, args.output)
        print(f"[INFO] 已更新输出文件: {args.output}")

    elapsed = time.time() - t0
    total_done = n_success + n_fail
    print(f"\n{'='*60}")
    print(f"重新生成完成: {total_done} 条, 用时 {elapsed:.1f}s ({total_done/elapsed:.1f} items/s)")
    print(f"  成功: {n_success} 条")
    print(f"  失败: {n_fail} 条")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
