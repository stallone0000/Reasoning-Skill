# -*- coding: utf-8 -*-
"""
export_nemotron_cp_unique.py

稳定版要点：
- 自动禁用/移除 hf-xet、hf_transfer（它们会走 xethub/CAS，很多代理直接 tunnel 失败）
- snapshot_download 拉到本地，再本地扫 competitive_coding_*.jsonl 去重
- 回填题面 + cases
- function 题（TACO/APPS）导出 args_json / expected_json，方便后续 C++/Python function judge
"""

import os
import sys
import time
import random
import subprocess
import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

# ================== 代理：务必带 scheme（很多库不带 scheme 会出奇怪问题） ==================
PROXY = "http://proxy.so.qihoo.net:8025"
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY

# ================== HF 环境变量（必须在 import huggingface/datasets 前） ==================
os.environ["HF_HUB_DISABLE_XET"] = "1"          # 理论上禁用 hf-xet（但某些版本有 bug）
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"   # 禁用 hf_transfer
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "1800"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ================== 关键：如果装了 hf_xet / hf_transfer，就卸载并重启 ==================
def _pip_uninstall(pkg: str):
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        pass

def _maybe_purge_rust_downloaders_and_restart():
    # 避免无限循环：只做一次
    if os.environ.get("HF_PURGE_DONE") == "1":
        return

    has_hf_xet = importlib.util.find_spec("hf_xet") is not None
    has_hf_transfer = importlib.util.find_spec("hf_transfer") is not None

    if not (has_hf_xet or has_hf_transfer):
        return

    print("[preflight] Detected:", "hf_xet" if has_hf_xet else "", "hf_transfer" if has_hf_transfer else "")
    print("[preflight] Uninstalling hf-xet / hf_transfer to avoid xethub(CAS) + proxy tunnel failures...")

    # 常见 pip 包名：hf-xet、hf_transfer（也有人环境里是 hf_xet）
    if has_hf_xet:
        _pip_uninstall("hf-xet")
        _pip_uninstall("hf_xet")
    if has_hf_transfer:
        _pip_uninstall("hf_transfer")
        _pip_uninstall("hf-transfer")

    os.environ["HF_PURGE_DONE"] = "1"
    print("[preflight] Restarting python process...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

_maybe_purge_rust_downloaders_and_restart()

# ================== 正式 imports（在 purge 之后） ==================
import orjson
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from datasets import load_dataset, DownloadConfig

REPO = "nvidia/Nemotron-Competitive-Programming-v1"

OUT_Q = "nemotron_cp_questions_34799_v1.jsonl"
OUT_CASES = "nemotron_cp_cases_34799_v1.jsonl"

HF_SOURCES = {
    "taco": ("BAAI/TACO", True),
    "apps": ("codeparrot/apps", True),
    "code_contests": ("deepmind/code_contests", False),
    "open-r1/codeforces": ("open-r1/codeforces", False),
}

DL_CFG = DownloadConfig(
    resume_download=True,
    max_retries=30,
)

split_cache: Dict[Tuple[str, str], Any] = {}

def _sleep_backoff(attempt: int, base: float = 1.0, cap: float = 45.0):
    t = min(cap, base * (2 ** attempt)) + random.random()
    time.sleep(t)

def safe_snapshot_download_nemotron() -> Path:
    """
    下载 Nemotron competitive_coding JSONL 到本地，带重试 + 降并发兜底。
    """
    allow = ["data/competitive_coding_*.jsonl", "README.md"]

    # 第一轮：少量并发
    for attempt in range(6):
        try:
            local_dir = snapshot_download(
                repo_id=REPO,
                repo_type="dataset",
                allow_patterns=allow,
                max_workers=4,
            )
            return Path(local_dir)
        except Exception as e:
            print(f"[snapshot_download retry {attempt+1}/6] {type(e).__name__}: {e}")
            _sleep_backoff(attempt)

    # 兜底：单线程
    for attempt in range(6):
        try:
            local_dir = snapshot_download(
                repo_id=REPO,
                repo_type="dataset",
                allow_patterns=allow,
                max_workers=1,
            )
            return Path(local_dir)
        except Exception as e:
            print(f"[snapshot_download single-worker retry {attempt+1}/6] {type(e).__name__}: {e}")
            _sleep_backoff(attempt)

    raise RuntimeError("snapshot_download failed after retries (even with max_workers=1)")

def iter_jsonl(path: Path):
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except Exception:
                continue

def _loads_maybe_json(x):
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, (bytes, bytearray)):
        return orjson.loads(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        return orjson.loads(s)
    return None

def get_hf_split(ds_name: str, split: str):
    hub_id, trust_remote_code = HF_SOURCES[ds_name]
    k = (ds_name, split)
    if k in split_cache:
        return split_cache[k]
    ds = load_dataset(
        hub_id,
        split=split,
        trust_remote_code=trust_remote_code,
        download_config=DL_CFG,
        keep_in_memory=False,
    )
    split_cache[k] = ds
    return ds

def get_hf_row(ds_name: str, split: str, index: int):
    ds = get_hf_split(ds_name, split)
    return ds[int(index)]

def get_question(ds_name: str, split: str, index: int) -> Optional[str]:
    row = get_hf_row(ds_name, split, index)

    if ds_name == "code_contests":
        return row.get("description") or None

    if ds_name in ("taco", "apps"):
        return row.get("question") or None

    if ds_name == "open-r1/codeforces":
        if not row.get("description"):
            return None
        q = row["description"]
        if row.get("input_format"):
            q += "\n\nInput\n\n" + row["input_format"]
        if row.get("output_format"):
            q += "\n\nOutput\n\n" + row["output_format"]
        if row.get("examples"):
            q += "\n\nExamples"
            for ex in row["examples"]:
                if "input" in ex:
                    q += "\n\nInput\n\n" + ex["input"]
                if "output" in ex:
                    q += "\n\nOutput\n\n" + ex["output"]
        if row.get("note"):
            q += "\n\nNote\n\n" + row["note"]
        return q

    return None

def _zip_io(inputs, outputs):
    if inputs is None or outputs is None:
        return []
    n = min(len(inputs), len(outputs))
    return [{"input": inputs[i], "output": outputs[i]} for i in range(n)]

def _to_args_list(x):
    # function 题：单参数也标准化为 [x]
    return x if isinstance(x, list) else [x]

def get_cases_and_judge(ds_name: str, split: str, index: int):
    row = get_hf_row(ds_name, split, index)

    # ========== TACO / APPS ==========
    if ds_name in ("taco", "apps"):
        io = _loads_maybe_json(row.get("input_output"))
        if not io:
            return {"kind": "none", "tests": [], "fn_name": None}, {"mode": "unknown"}

        fn_name = io.get("fn_name")
        inputs = io.get("inputs")
        outputs = io.get("outputs")

        if fn_name:
            tests = []
            if inputs is not None and outputs is not None:
                n = min(len(inputs), len(outputs))
                for i in range(n):
                    args = _to_args_list(inputs[i])
                    exp = outputs[i]
                    tests.append({
                        "args_json": orjson.dumps(args).decode("utf-8"),
                        "expected_json": orjson.dumps(exp).decode("utf-8"),
                    })
            cases = {"kind": "function", "fn_name": fn_name, "tests": tests}
            judge = {"mode": "function", "dataset": ds_name}
            return cases, judge

        cases = {"kind": "stdio", "fn_name": None, "tests": _zip_io(inputs, outputs)}
        judge = {"mode": "stdio", "dataset": ds_name}
        return cases, judge

    # ========== CodeContests ==========
    if ds_name == "code_contests":
        def pack(d):
            if not d:
                return []
            return _zip_io(d.get("input", []), d.get("output", []))

        cases = {
            "kind": "code_contests",
            "public": pack(row.get("public_tests")),
            "private": pack(row.get("private_tests")),
            "generated": pack(row.get("generated_tests")),
        }

        tl = row.get("time_limit")
        timeout_s = None
        if isinstance(tl, dict):
            timeout_s = float(tl.get("seconds", 0)) + float(tl.get("nanos", 0)) / 1e9

        judge = {
            "mode": "stdio",
            "timeout_s": timeout_s,
            "memory_limit_bytes": row.get("memory_limit_bytes"),
        }
        return cases, judge

    # ========== open-r1/codeforces ==========
    if ds_name == "open-r1/codeforces":
        official = row.get("official_tests") or []
        cases = {
            "kind": "codeforces",
            "official": [{"input": t.get("input", ""), "output": t.get("output", "")} for t in official],
            "generated_count": row.get("generated_tests", 0),
        }
        judge = {
            "mode": "stdio" if row.get("input_mode") == "stdio" else "file",
            "input_mode": row.get("input_mode"),
            "time_limit_s": row.get("time_limit"),
            "memory_limit_mb": row.get("memory_limit"),
            "checker_py": row.get("generated_checker") or None,
            "problem_id": row.get("id") or None,
            "official_tests_complete": row.get("official_tests_complete"),
        }
        return cases, judge

    return {"kind": "none"}, {"mode": "unknown"}

def main():
    # 1) 下载 Nemotron competitive_coding 分片到本地
    local_repo = safe_snapshot_download_nemotron()
    data_dir = local_repo / "data"
    files = sorted(data_dir.glob("competitive_coding_*.jsonl"))
    print("local_repo:", local_repo)
    print("found files:", [p.name for p in files])
    if not files:
        raise RuntimeError("No competitive_coding_*.jsonl found under snapshot data/")

    # 2) 本地扫描构建 unique（按来源三元组去重）
    unique: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for fp in files:
        for ex in tqdm(iter_jsonl(fp), desc=f"scan {fp.name}"):
            ds_name = ex.get("dataset")
            ds_split = ex.get("split")
            ds_index = ex.get("index")
            if ds_name is None or ds_split is None or ds_index is None:
                continue
            key = (ds_name, ds_split, int(ds_index))
            if key not in unique:
                unique[key] = {
                    "question_id": ex.get("question_id"),
                    "source": ex.get("source"),
                    "difficulty": ex.get("difficulty"),
                }

    print("unique_by_origin_triplet =", len(unique))

    # 3) 导出 questions + cases
    items = list(unique.items())
    items.sort(key=lambda kv: f"{kv[0][0]}/{kv[0][1]}/{kv[0][2]}")

    missing_q = 0
    missing_cases = 0
    hard_fail = 0

    with open(OUT_Q, "wb") as fq, open(OUT_CASES, "wb") as fc:
        for (ds_name, ds_split, ds_index), meta in tqdm(items, desc="export"):
            unique_key = f"{ds_name}/{ds_split}/{ds_index}"

            q = ""
            cases, judge = {"kind": "none"}, {"mode": "unknown"}
            ok_row = False

            for attempt in range(6):
                try:
                    qv = get_question(ds_name, ds_split, ds_index)
                    q = "" if qv is None else qv
                    cases, judge = get_cases_and_judge(ds_name, ds_split, ds_index)
                    ok_row = True
                    break
                except Exception:
                    if attempt == 5:
                        hard_fail += 1
                    _sleep_backoff(attempt)

            if not ok_row:
                q = ""
                cases, judge = {"kind": "none"}, {"mode": "unknown"}

            if not q or q.strip() in {"", "-"}:
                missing_q += 1
                q = "" if q is None else q

            if cases.get("kind") == "none":
                missing_cases += 1

            fq.write(orjson.dumps({
                "unique_key": unique_key,
                "question_id": meta.get("question_id"),
                "question": q,
                "source": meta.get("source"),
                "difficulty": meta.get("difficulty"),
                "origin_dataset": ds_name,
                "origin_split": ds_split,
                "origin_index": ds_index,
            }))
            fq.write(b"\n")

            fc.write(orjson.dumps({
                "unique_key": unique_key,
                "origin_dataset": ds_name,
                "origin_split": ds_split,
                "origin_index": ds_index,
                "cases": cases,
                "judge": judge,
            }))
            fc.write(b"\n")

    print("wrote:", OUT_Q, OUT_CASES)
    print("missing_questions:", missing_q)
    print("missing_cases:", missing_cases)
    print("hard_fail_rows:", hard_fail)

if __name__ == "__main__":
    main()
