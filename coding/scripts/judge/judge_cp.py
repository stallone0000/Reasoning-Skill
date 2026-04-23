# -*- coding: utf-8 -*-
"""
judge_cp.py

Batch judge for Nemotron CP exports:
- cases jsonl: produced by your exporter (supports stdio/file/function; function uses args_json/expected_json)
- gens jsonl: model outputs, each line contains at least:
    {"unique_key": "...", "llm_output": "..."}  (llm_output may contain code fences)

Supports:
- language auto-detect: Python / C++
- stdio, fileio, function
- codeforces checker.py (if provided in judge.checker_py)

Usage examples at bottom.

Security:
- This is NOT a hardened sandbox. Use docker/nsjail/bwrap for untrusted code.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import orjson
    _HAS_ORJSON = True
except Exception:
    _HAS_ORJSON = False

# ----------------------------
# JSONL utils
# ----------------------------

def loads_json(line: bytes) -> dict:
    if _HAS_ORJSON:
        return orjson.loads(line)
    return json.loads(line)

def dumps_json(obj: dict) -> bytes:
    if _HAS_ORJSON:
        return orjson.dumps(obj)
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield loads_json(line)

def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "wb") as f:
        for r in rows:
            b = dumps_json(r)
            if not b.endswith(b"\n"):
                b += b"\n"
            f.write(b)

# ----------------------------
# Code extraction & language guess
# ----------------------------

FENCE_RE = re.compile(r"```([a-zA-Z0-9_+\-]*)\n(.*?)\n```", re.DOTALL)
# 支持 <code_block> 标签（Gemini Pro 使用的格式）
CODE_BLOCK_RE = re.compile(r"<code_block>(.*?)</code_block>", re.DOTALL | re.IGNORECASE)

LANG_ALIASES = {
    "py": "python",
    "python": "python",
    "python3": "python",
    "cpp": "cpp",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "cplusplus": "cpp",
}

# 不支持的语言列表（这些语言会被明确排除）
UNSUPPORTED_LANGS = {"rust", "rs", "java", "javascript", "js", "go", "php", "ruby", "scala", "kotlin", "swift"}

def extract_code(llm_output: str) -> Tuple[Optional[str], str]:
    """
    Return (lang_hint, code).
    Prefer the largest fenced block; prefer blocks whose language is python/cpp.
    Supports both markdown ``` format and <code_block> tags.
    If no fences, return whole text as code candidate (with tags stripped).
    Explicitly excludes unsupported languages (e.g., Rust, Java).
    """
    text = llm_output or ""
    blocks: List[Tuple[Optional[str], str]] = []
    
    # 1. 提取 markdown 代码块 (```)
    for lang, code in FENCE_RE.findall(text):
        lang = (lang or "").strip().lower()
        # 如果是不支持的语言，跳过
        if lang in UNSUPPORTED_LANGS:
            continue
        lang = LANG_ALIASES.get(lang, lang) if lang else None
        c = (code or "").strip("\n")
        if c.strip():
            blocks.append((lang if lang in ("python", "cpp") else None, c))
    
    # 2. 提取 <code_block> 标签中的代码
    for code_match in CODE_BLOCK_RE.finditer(text):
        code = code_match.group(1).strip()
        if code.strip():
            # 先检查是否包含不支持语言的语法（如 Rust）
            if re.search(r'\buse\s+std::', code, re.IGNORECASE) or \
               re.search(r'\bfn\s+\w+\s*\(', code, re.IGNORECASE) or \
               re.search(r'for\s+_\s+in\s+\d+\.\.', code, re.IGNORECASE):
                # 可能是 Rust 代码，跳过
                continue
            # 尝试从代码内容推断语言
            lang_hint = None
            if "#include" in code or "using namespace std" in code or re.search(r"\bint\s+main\s*\(", code):
                lang_hint = "cpp"
            elif code.startswith("#!/usr/bin/env python") or re.search(r"^\s*def\s+\w+\(", code, re.M):
                lang_hint = "python"
            blocks.append((lang_hint, code))
    
    if blocks:
        def score(b):
            lang, code = b
            s = len(code)
            if lang in ("python", "cpp"):
                s += 10_000_000
            return s
        return max(blocks, key=score)

    # 如果没有找到代码块，仅清理 code_block 标签后返回。
    # 不要清理通用 <...>，否则会误删 C++ include/template（如 <bits/stdc++.h>）。
    cleaned = text.strip()
    # 移除 <code_block> 标签但保留内容
    cleaned = CODE_BLOCK_RE.sub(r'\1', cleaned)
    return None, cleaned

def guess_language(code: str, hint: Optional[str]) -> str:
    """
    Guess the programming language from code.
    Returns "python", "cpp", or "unsupported" for languages we don't support.
    """
    if hint in ("python", "cpp"):
        return hint

    s = code.lstrip()

    # 首先检查是否是不支持的语言（如 Rust）
    # Rust 的强信号
    rust_signals = [
        re.search(r'\buse\s+std::', s, re.IGNORECASE),  # use std::io
        re.search(r'\bfn\s+\w+\s*\([^)]*\)\s*->', s, re.IGNORECASE),  # fn main() ->
        re.search(r'for\s+_\s+in\s+\d+\.\.', s, re.IGNORECASE),  # for _ in 0..n
        re.search(r'let\s+mut\s+\w+', s, re.IGNORECASE),  # let mut x
        re.search(r'Vec<\w+>', s, re.IGNORECASE),  # Vec<i32>
        re.search(r'\.unwrap\(\)', s, re.IGNORECASE),  # .unwrap()
    ]
    if any(rust_signals):
        return "unsupported"  # 标记为不支持的语言

    # Strong signals for supported languages
    if "#include" in s or "using namespace std" in s or re.search(r"\bint\s+main\s*\(", s):
        return "cpp"
    if s.startswith("#!/usr/bin/env python") or re.search(r"^\s*def\s+\w+\(", code, re.M):
        return "python"

    # Scoring
    py = 0
    cp = 0
    py += 3 if "import " in code else 0
    py += 2 if "print(" in code else 0
    py += 2 if re.search(r"^\s*if __name__\s*==\s*['\"]__main__['\"]\s*:", code, re.M) else 0

    cp += 3 if "#include" in code else 0
    cp += 2 if "std::" in code else 0
    cp += 2 if "bits/stdc++.h" in code else 0
    cp += 2 if re.search(r"^\s*cout\s*<<", code, re.M) else 0

    return "cpp" if cp >= py else "python"

# ----------------------------
# Output comparison
# ----------------------------

def io_to_text(x: Any) -> str:
    """Best-effort conversion of test IO payloads to text."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, list):
        # Many datasets store stdio as list-of-lines.
        if all(not isinstance(v, (list, dict)) for v in x):
            return "\n".join(io_to_text(v) for v in x)
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    if isinstance(x, dict):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    if isinstance(x, (int, float, bool)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)


def norm_text(s: Any) -> str:
    s = io_to_text(s).replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(line.rstrip() for line in lines)


_NUM_RE = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


def compare_output(got: str, exp: str, float_tol: float = 1e-6) -> bool:
    """
    Compare two outputs. First try exact match (after norm_text).
    If that fails, try token-by-token comparison with float tolerance
    for numeric tokens (handles trailing zeros, small precision diffs).
    """
    g = norm_text(got)
    e = norm_text(exp)
    if g == e:
        return True

    # Token-by-token comparison with float tolerance
    g_tokens = g.split()
    e_tokens = e.split()
    if len(g_tokens) != len(e_tokens):
        return False

    for gt, et in zip(g_tokens, e_tokens):
        if gt == et:
            continue
        # Both look like numbers?
        if _NUM_RE.match(gt) and _NUM_RE.match(et):
            try:
                gv = float(gt)
                ev = float(et)
                if ev == 0:
                    if abs(gv) > float_tol:
                        return False
                else:
                    # relative + absolute tolerance
                    if abs(gv - ev) > float_tol and abs(gv - ev) / max(abs(ev), 1e-15) > float_tol:
                        return False
            except ValueError:
                return False
        else:
            return False
    return True

def json_equal(a: Any, b: Any, float_tol: float = 1e-6) -> bool:
    # Semantic equality with numeric tolerance
    if a is b:
        return True
    if type(a) != type(b):
        # allow int/float compare
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) <= float_tol
        return False
    if isinstance(a, (int, float)):
        return abs(float(a) - float(b)) <= float_tol
    if isinstance(a, str):
        return a == b
    if isinstance(a, bool):
        return a == b
    if a is None:
        return b is None
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(json_equal(x, y, float_tol) for x, y in zip(a, b))
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(json_equal(a[k], b[k], float_tol) for k in a.keys())
    return a == b

def compare_function_outputs(got_json: str, exp_json: str, float_tol: float = 1e-6) -> bool:
    try:
        g = json.loads(got_json)
        e = json.loads(exp_json)
    except Exception:
        return False
    return json_equal(g, e, float_tol=float_tol)

# ----------------------------
# Resource limits (Linux)
# ----------------------------

def _preexec_limits(cpu_s: int, mem_mb: int):
    def fn():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            mem = int(mem_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except Exception:
            pass
    return fn

def run_process(
    cmd: List[str],
    stdin_str: Any,
    cwd: str,
    timeout_s: float,
    cpu_s: int,
    mem_mb: int,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str, float]:
    env = os.environ.copy()
    # Reduce accidental thread bombs
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    if extra_env:
        env.update(extra_env)

    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            input=io_to_text(stdin_str).encode("utf-8", errors="replace"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            timeout=timeout_s,
            preexec_fn=_preexec_limits(cpu_s, mem_mb) if os.name == "posix" else None,
        )
    except PermissionError as e:
        # Windows Defender / antivirus may block compiled executables in temp dirs
        dt = time.time() - t0
        return -1, "", f"PermissionError: {e}", dt
    except OSError as e:
        dt = time.time() - t0
        return -1, "", f"OSError: {e}", dt
    dt = time.time() - t0
    return p.returncode, p.stdout.decode("utf-8", errors="replace"), p.stderr.decode("utf-8", errors="replace"), dt

# ----------------------------
# C++ compile helpers
# ----------------------------

def compile_cpp(code: str, workdir: str) -> Tuple[bool, str, str]:
    """
    Return (ok, exe_path, compile_stderr).
    """
    src = os.path.join(workdir, "main.cpp")
    exe_name = "main.exe" if os.name == "nt" else "main.out"
    exe = os.path.join(workdir, exe_name)
    with open(src, "w", encoding="utf-8") as f:
        f.write(code)

    cmd = ["g++", "-O2", "-std=gnu++17", "-pipe", src, "-o", exe]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    except FileNotFoundError:
        return False, exe, "g++ not found in PATH. Install MinGW/GCC to compile C++ code."
    except subprocess.TimeoutExpired:
        return False, exe, "Compilation timed out (>60s)."
    err = p.stderr.decode("utf-8", errors="replace")
    ok = (p.returncode == 0 and os.path.exists(exe))
    return ok, exe, err

def write_python(code: str, workdir: str) -> str:
    path = os.path.join(workdir, "main.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path

# ----------------------------
# Python function runner
# ----------------------------

PY_FUNC_RUNNER = r"""
import json, sys, types

# read full stdin as JSON
raw = sys.stdin.read()
obj = json.loads(raw)  # either [args...] or {"args":[...], "fn_name":"..."}
if isinstance(obj, dict):
    args = obj.get("args", [])
    fn_name = obj.get("fn_name")
else:
    args = obj
    fn_name = None

# load user code from main_user.py
code = open("main_user.py", "r", encoding="utf-8").read()
g = {"__name__": "__main__"}
exec(compile(code, "main_user.py", "exec"), g, g)

def call(fn_name, args):
    # prefer class Solution().fn_name
    if "Solution" in g:
        try:
            sol = g["Solution"]()
            if fn_name and hasattr(sol, fn_name):
                return getattr(sol, fn_name)(*args)
        except Exception:
            pass
    # then top-level function
    if fn_name and fn_name in g and callable(g[fn_name]):
        return g[fn_name](*args)

    # fallback: if only one callable is present, try it (risky but useful)
    cands = []
    for k,v in g.items():
        if k.startswith("_"): 
            continue
        if callable(v) and k not in ("print",):
            cands.append((k,v))
    if len(cands) == 1:
        return cands[0][1](*args)

    raise RuntimeError("Cannot find target function")

res = call(fn_name, args)
sys.stdout.write(json.dumps(res, ensure_ascii=False))
"""

# ----------------------------
# C++ function harness (JSON I/O) with signature parsing
# ----------------------------

CPP_SIG_RE = re.compile(
    r"""(?P<ret>[\w:\s<>,*&]+?)\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^)]*)\)""",
    re.DOTALL,
)

def normalize_cpp_type(t: str) -> str:
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("std::", "")
    t = t.replace("const ", "")
    t = t.replace("&", "").replace("*", "")
    return t.strip()

def split_cpp_args(arg_str: str) -> List[str]:
    s = (arg_str or "").strip()
    if not s:
        return []
    out, cur, depth = [], [], 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur).strip())
    return out

SUPPORTED_TYPES = {
    "int", "long long", "double", "bool", "string",
    "vector<int>", "vector<long long>", "vector<double>", "vector<string>",
    "vector<vector<int>>", "vector<vector<long long>>", "vector<vector<double>>", "vector<vector<string>>",
}

def parse_solution_signature(user_cpp: str, fn_name: str) -> Optional[Tuple[str, List[str]]]:
    # try to focus on class Solution block
    m = re.search(r"class\s+Solution\s*\{.*?\};", user_cpp, re.DOTALL)
    region = m.group(0) if m else user_cpp

    for sm in CPP_SIG_RE.finditer(region):
        name = sm.group("name")
        if name != fn_name:
            continue
        ret = normalize_cpp_type(sm.group("ret"))
        args_raw = split_cpp_args(sm.group("args"))
        arg_types: List[str] = []
        for a in args_raw:
            a = a.split("=")[0].strip()
            # remove trailing variable name (heuristic)
            toks = a.split()
            if len(toks) >= 2:
                last = toks[-1]
                if re.match(r"^[A-Za-z_]\w*$", last) and "<" not in last and ">" not in last:
                    a = " ".join(toks[:-1])
            arg_types.append(normalize_cpp_type(a))
        return ret, arg_types
    return None

CPP_JSON_UTIL = r"""
#include <bits/stdc++.h>
using namespace std;

struct J {
  enum T {NUL, BOOL, NUM, STR, ARR, OBJ} t=NUL;
  bool b=false; double n=0; string s;
  vector<J> a;
  map<string,J> o;
};

static inline void skip_ws(const string& x, size_t& i){ while(i<x.size() && isspace((unsigned char)x[i])) i++; }

static J parse_value(const string& x, size_t& i);

static string parse_string(const string& x, size_t& i){
  string r; i++;
  while(i<x.size()){
    char c=x[i++];
    if(c=='"') break;
    if(c=='\\'){
      char e=x[i++];
      if(e=='"') r.push_back('"');
      else if(e=='\\') r.push_back('\\');
      else if(e=='/') r.push_back('/');
      else if(e=='b') r.push_back('\b');
      else if(e=='f') r.push_back('\f');
      else if(e=='n') r.push_back('\n');
      else if(e=='r') r.push_back('\r');
      else if(e=='t') r.push_back('\t');
      else r.push_back(e);
    } else r.push_back(c);
  }
  return r;
}

static J parse_number(const string& x, size_t& i){
  size_t j=i;
  while(j<x.size() && (isdigit((unsigned char)x[j])||x[j]=='-'||x[j]=='+'||x[j]=='.'||x[j]=='e'||x[j]=='E')) j++;
  double v = strtod(x.c_str()+i, nullptr);
  i=j;
  J r; r.t=J::NUM; r.n=v; return r;
}

static J parse_array(const string& x, size_t& i){
  J r; r.t=J::ARR; i++; skip_ws(x,i);
  if(i<x.size() && x[i]==']'){ i++; return r; }
  while(i<x.size()){
    J v = parse_value(x,i);
    r.a.push_back(std::move(v));
    skip_ws(x,i);
    if(i<x.size() && x[i]==','){ i++; skip_ws(x,i); continue; }
    if(i<x.size() && x[i]==']'){ i++; break; }
    break;
  }
  return r;
}

static J parse_object(const string& x, size_t& i){
  J r; r.t=J::OBJ; i++; skip_ws(x,i);
  if(i<x.size() && x[i]=='}'){ i++; return r; }
  while(i<x.size()){
    skip_ws(x,i);
    string k = parse_string(x,i);
    skip_ws(x,i);
    if(i<x.size() && x[i]==':') i++;
    skip_ws(x,i);
    J v = parse_value(x,i);
    r.o.emplace(std::move(k), std::move(v));
    skip_ws(x,i);
    if(i<x.size() && x[i]==','){ i++; skip_ws(x,i); continue; }
    if(i<x.size() && x[i]=='}'){ i++; break; }
    break;
  }
  return r;
}

static J parse_value(const string& x, size_t& i){
  skip_ws(x,i);
  if(i>=x.size()) return J{};
  char c=x[i];
  if(c=='n'){ i+=4; return J{}; }
  if(c=='t'){ J r; r.t=J::BOOL; r.b=true; i+=4; return r; }
  if(c=='f'){ J r; r.t=J::BOOL; r.b=false; i+=5; return r; }
  if(c=='"'){ J r; r.t=J::STR; r.s=parse_string(x,i); return r; }
  if(c=='[') return parse_array(x,i);
  if(c=='{') return parse_object(x,i);
  return parse_number(x,i);
}

static J parse_json(const string& x){
  size_t i=0; return parse_value(x,i);
}

static void dump_json(const J& v, string& out);

static void dump_str(const string& s, string& out){
  out.push_back('"');
  for(char c: s){
    if(c=='"'||c=='\\'){ out.push_back('\\'); out.push_back(c); }
    else if(c=='\n'){ out += "\\n"; }
    else if(c=='\r'){ out += "\\r"; }
    else if(c=='\t'){ out += "\\t"; }
    else out.push_back(c);
  }
  out.push_back('"');
}

static void dump_json(const J& v, string& out){
  switch(v.t){
    case J::NUL: out += "null"; break;
    case J::BOOL: out += (v.b?"true":"false"); break;
    case J::NUM: {
      ostringstream oss; oss.setf(std::ios::fixed); oss<<setprecision(10)<<v.n;
      string s=oss.str();
      while(s.size() && s.find('.')!=string::npos && s.back()=='0') s.pop_back();
      if(s.size() && s.back()=='.') s.pop_back();
      if(s.empty()) s="0";
      out += s;
      break;
    }
    case J::STR: dump_str(v.s,out); break;
    case J::ARR:
      out.push_back('[');
      for(size_t i=0;i<v.a.size();i++){
        if(i) out.push_back(',');
        dump_json(v.a[i],out);
      }
      out.push_back(']');
      break;
    case J::OBJ:
      out.push_back('{');
      {
        bool first=true;
        for(auto &kv: v.o){
          if(!first) out.push_back(',');
          first=false;
          dump_str(kv.first,out);
          out.push_back(':');
          dump_json(kv.second,out);
        }
      }
      out.push_back('}');
      break;
  }
}

template<class T> T fromJ(const J& v);
template<class T> J toJ(const T& x);

template<> int fromJ<int>(const J& v){ return (int) llround(v.n); }
template<> long long fromJ<long long>(const J& v){ return (long long) llround(v.n); }
template<> double fromJ<double>(const J& v){ return v.n; }
template<> bool fromJ<bool>(const J& v){ return v.b; }
template<> string fromJ<string>(const J& v){ return v.s; }

template<> J toJ<int>(const int& x){ J r; r.t=J::NUM; r.n=x; return r; }
template<> J toJ<long long>(const long long& x){ J r; r.t=J::NUM; r.n=(double)x; return r; }
template<> J toJ<double>(const double& x){ J r; r.t=J::NUM; r.n=x; return r; }
template<> J toJ<bool>(const bool& x){ J r; r.t=J::BOOL; r.b=x; return r; }
template<> J toJ<string>(const string& x){ J r; r.t=J::STR; r.s=x; return r; }

template<class T>
vector<T> fromJ_vec(const J& v){
  vector<T> r; r.reserve(v.a.size());
  for(auto &e: v.a) r.push_back(fromJ<T>(e));
  return r;
}
template<class T>
J toJ_vec(const vector<T>& x){
  J r; r.t=J::ARR; r.a.reserve(x.size());
  for(auto &e: x) r.a.push_back(toJ<T>(e));
  return r;
}

template<> vector<int> fromJ<vector<int>>(const J& v){ return fromJ_vec<int>(v); }
template<> vector<long long> fromJ<vector<long long>>(const J& v){ return fromJ_vec<long long>(v); }
template<> vector<double> fromJ<vector<double>>(const J& v){ return fromJ_vec<double>(v); }
template<> vector<string> fromJ<vector<string>>(const J& v){ return fromJ_vec<string>(v); }

template<> J toJ<vector<int>>(const vector<int>& x){ return toJ_vec<int>(x); }
template<> J toJ<vector<long long>>(const vector<long long>& x){ return toJ_vec<long long>(x); }
template<> J toJ<vector<double>>(const vector<double>& x){ return toJ_vec<double>(x); }
template<> J toJ<vector<string>>(const vector<string>& x){ return toJ_vec<string>(x); }

template<> vector<vector<int>> fromJ<vector<vector<int>>>(const J& v){
  vector<vector<int>> r; r.reserve(v.a.size());
  for(auto &e: v.a) r.push_back(fromJ<vector<int>>(e));
  return r;
}
template<> vector<vector<long long>> fromJ<vector<vector<long long>>>(const J& v){
  vector<vector<long long>> r; r.reserve(v.a.size());
  for(auto &e: v.a) r.push_back(fromJ<vector<long long>>(e));
  return r;
}
template<> vector<vector<double>> fromJ<vector<vector<double>>>(const J& v){
  vector<vector<double>> r; r.reserve(v.a.size());
  for(auto &e: v.a) r.push_back(fromJ<vector<double>>(e));
  return r;
}
template<> vector<vector<string>> fromJ<vector<vector<string>>>(const J& v){
  vector<vector<string>> r; r.reserve(v.a.size());
  for(auto &e: v.a) r.push_back(fromJ<vector<string>>(e));
  return r;
}

template<> J toJ<vector<vector<int>>>(const vector<vector<int>>& x){
  J r; r.t=J::ARR; r.a.reserve(x.size());
  for(auto &e: x) r.a.push_back(toJ<vector<int>>(e));
  return r;
}
template<> J toJ<vector<vector<long long>>>(const vector<vector<long long>>& x){
  J r; r.t=J::ARR; r.a.reserve(x.size());
  for(auto &e: x) r.a.push_back(toJ<vector<long long>>(e));
  return r;
}
template<> J toJ<vector<vector<double>>>(const vector<vector<double>>& x){
  J r; r.t=J::ARR; r.a.reserve(x.size());
  for(auto &e: x) r.a.push_back(toJ<vector<double>>(e));
  return r;
}
template<> J toJ<vector<vector<string>>>(const vector<vector<string>>& x){
  J r; r.t=J::ARR; r.a.reserve(x.size());
  for(auto &e: x) r.a.push_back(toJ<vector<string>>(e));
  return r;
}
"""

def build_cpp_function_harness(user_cpp: str, fn_name: str, ret: str, arg_types: List[str]) -> str:
    ret_n = normalize_cpp_type(ret)
    if ret_n not in SUPPORTED_TYPES:
        raise ValueError(f"unsupported return type: {ret_n}")
    arg_n = [normalize_cpp_type(t) for t in arg_types]
    for t in arg_n:
        if t not in SUPPORTED_TYPES:
            raise ValueError(f"unsupported arg type: {t}")

    reads = []
    call_args = []
    for i, t in enumerate(arg_n):
        reads.append(f"  {t} a{i} = fromJ<{t}>(args.a[{i}]);")
        call_args.append(f"a{i}")

    main = f"""
int main() {{
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  string input, line;
  while (getline(cin, line)) {{
    if(!input.empty()) input.push_back('\\n');
    input += line;
  }}
  J root = parse_json(input);

  // protocol:
  // 1) args array: [arg1, arg2, ...]
  // 2) object: {{ "fn_name": "...", "args": [...] }}
  J args;
  if(root.t == J::ARR) args = root;
  else if(root.t == J::OBJ && root.o.count("args")) args = root.o["args"];
  else return 2;

  if(args.t != J::ARR) return 2;

  Solution sol;
{"".join(reads)}
  {ret_n} ans = sol.{fn_name}({", ".join(call_args)});
  J outv = toJ<{ret_n}>(ans);
  string out_s;
  dump_json(outv, out_s);
  cout << out_s;
  return 0;
}}
"""
    return CPP_JSON_UTIL + "\n\n" + user_cpp + "\n\n" + main

# ----------------------------
# Checker runner (Codeforces)
# ----------------------------

def run_checker(checker_py: str, cwd: str, inp: Any, correct: Any, got: Any, timeout_s: float) -> Tuple[bool, str, str]:
    with open(os.path.join(cwd, "checker.py"), "w", encoding="utf-8") as f:
        f.write(checker_py)
    with open(os.path.join(cwd, "input.txt"), "w", encoding="utf-8") as f:
        f.write(io_to_text(inp))
    with open(os.path.join(cwd, "correct_output.txt"), "w", encoding="utf-8") as f:
        f.write(io_to_text(correct))
    with open(os.path.join(cwd, "solution_output.txt"), "w", encoding="utf-8") as f:
        f.write(io_to_text(got))

    # Checker can be much slower than one test run; avoid converting this into JUDGE_ERROR.
    checker_timeout = max(5.0, float(timeout_s) * 3.0)
    try:
        p = subprocess.run(
            [sys.executable, "checker.py", "input.txt", "correct_output.txt", "solution_output.txt"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=checker_timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "", f"checker timeout after {checker_timeout:.1f}s"
    except Exception as e:
        return False, "", f"checker runtime error: {e}"
    out = p.stdout.decode("utf-8", errors="replace").strip()
    err = p.stderr.decode("utf-8", errors="replace")

    # common pattern: checker prints score; treat >0 as pass (customize if you want)
    ok = False
    try:
        score = float(out.split()[0])
        ok = score > 0.0
    except Exception:
        ok = False
    return ok, out, err

# ----------------------------
# Test selection
# ----------------------------

def pick_stdio_tests(cases: dict, include_private: bool) -> List[dict]:
    kind = (cases or {}).get("kind", "none")
    if kind == "stdio":
        return cases.get("tests", []) or []
    if kind == "code_contests":
        tests = (cases.get("public", []) or []) + (cases.get("generated", []) or [])
        if include_private:
            tests += (cases.get("private", []) or [])
        return tests
    if kind == "codeforces":
        return cases.get("official", []) or []
    return []


def should_force_function_mode(mode: str, cases: dict) -> bool:
    """Some datasets are mislabeled as stdio/file while tests are structured function args."""
    if mode not in ("stdio", "file"):
        return False
    tests = pick_stdio_tests(cases, include_private=False)
    if not tests:
        return False
    sample = tests[0] if isinstance(tests[0], dict) else {}
    fn_name = cases.get("fn_name")
    if "args_json" in sample or "expected_json" in sample:
        return True
    if not fn_name:
        return False
    inp = sample.get("input")
    out = sample.get("output")
    return isinstance(inp, (list, dict)) and isinstance(out, (list, dict))


def to_function_tests(tests: List[dict]) -> List[dict]:
    out: List[dict] = []
    for t in tests:
        args = t.get("args_json")
        exp = t.get("expected_json")
        if args is None:
            args = json.dumps(t.get("input", []), ensure_ascii=False)
        if exp is None:
            exp = json.dumps(t.get("output", None), ensure_ascii=False)
        out.append({"args_json": args, "expected_json": exp})
    return out

# ----------------------------
# Main judging per item
# ----------------------------

@dataclass
class JudgeConfig:
    float_tol: float = 1e-6
    include_private: bool = False
    default_timeout_s: float = 2.0
    default_mem_mb: int = 1024
    max_consecutive_fail: int = 10  # 0 = unlimited; stop early after N consecutive non-AC cases


def _make_result(
    unique_key: str,
    status: str,
    lang: Optional[str],
    case_results: List[dict],
    total_tests: int,
    **extra,
) -> dict:
    """Build a standardised judge result dict with per-case detail."""
    passed = sum(1 for c in case_results if c.get("status") == "AC")
    return {
        "unique_key": unique_key,
        "status": status,
        "lang": lang,
        "passed": passed,
        "total_tests": total_tests,
        "case_results": case_results,
        **extra,
    }


def _should_stop(case_results: List[dict], max_consec: int) -> bool:
    """Return True if last *max_consec* results are all non-AC."""
    if max_consec <= 0:
        return False
    if len(case_results) < max_consec:
        return False
    return all(c.get("status") != "AC" for c in case_results[-max_consec:])


def judge_one(
    problem: dict,
    llm_output: str,
    cfg: JudgeConfig,
) -> dict:
    unique_key = problem.get("unique_key", "")
    cases = problem.get("cases", {}) or {}
    jcfg = problem.get("judge", {}) or {}

    hint, code = extract_code(llm_output)
    lang = guess_language(code, hint)
    
    # 如果不支持的语言，直接返回
    if lang == "unsupported":
        return _make_result(
            unique_key, 
            "UNSUPPORTED_LANG", 
            None, 
            [], 
            0, 
            detail="Detected unsupported language (e.g., Rust, Java). Only Python and C++ are supported."
        )

    mode = jcfg.get("mode", "stdio")  # "stdio" | "file" | "function"
    timeout_s = float(jcfg.get("timeout_s") or jcfg.get("time_limit_s") or cfg.default_timeout_s)
    mem_mb = int(jcfg.get("memory_limit_mb") or cfg.default_mem_mb)
    cpu_s = max(1, min(30, int(timeout_s)))  # simple cap
    checker_py = jcfg.get("checker_py")
    max_consec = cfg.max_consecutive_fail

    if should_force_function_mode(mode, cases):
        mode = "function"

    # ----------------------------------------------------------------
    # function mode
    # ----------------------------------------------------------------
    if mode == "function":
        fn_name = cases.get("fn_name")
        if not fn_name:
            return _make_result(unique_key, "UNSUPPORTED_CANNOT_PARSE_SIGNATURE", lang, [], 0, detail="missing fn_name for function mode")
        raw_tests = cases.get("tests", []) or []
        tests = to_function_tests(raw_tests)
        if not fn_name or not tests:
            return _make_result(unique_key, "NO_TESTS", lang, [], 0)

        case_results: List[dict] = []

        with tempfile.TemporaryDirectory() as td:
            # --- Python function ---
            if lang == "python":
                with open(os.path.join(td, "main_user.py"), "w", encoding="utf-8") as f:
                    f.write(code)
                with open(os.path.join(td, "runner.py"), "w", encoding="utf-8") as f:
                    f.write(PY_FUNC_RUNNER)

                for i, t in enumerate(tests):
                    if _should_stop(case_results, max_consec):
                        for j in range(i, len(tests)):
                            case_results.append({"case": j, "status": "SKIPPED"})
                        break

                    args_json = t.get("args_json", "[]")
                    exp_json = t.get("expected_json", "null")
                    inp = json.dumps({"fn_name": fn_name, "args": json.loads(args_json)}, ensure_ascii=False)

                    try:
                        rc, out, err, dt = run_process(
                            [sys.executable, "-I", "-S", "runner.py"],
                            inp, cwd=td, timeout_s=timeout_s, cpu_s=cpu_s, mem_mb=mem_mb,
                            extra_env={"PYTHONNOUSERSITE": "1"},
                        )
                    except subprocess.TimeoutExpired:
                        case_results.append({"case": i, "status": "TLE"})
                        continue

                    if rc != 0:
                        case_results.append({"case": i, "status": "RE", "stderr": err[:500]})
                        continue

                    if compare_function_outputs(out.strip(), exp_json.strip(), float_tol=cfg.float_tol):
                        case_results.append({"case": i, "status": "AC"})
                    else:
                        case_results.append({"case": i, "status": "WA", "got": out[:500], "exp": exp_json[:500]})

                overall = "AC" if all(c["status"] == "AC" for c in case_results) else next(c["status"] for c in case_results if c["status"] != "AC")
                return _make_result(unique_key, overall, lang, case_results, len(tests))

            # --- C++ function ---
            if lang == "cpp":
                sig = parse_solution_signature(code, fn_name)
                if not sig:
                    return _make_result(unique_key, "UNSUPPORTED_CANNOT_PARSE_SIGNATURE", "cpp", [], len(tests))
                ret, arg_types = sig
                try:
                    harness = build_cpp_function_harness(code, fn_name, ret, arg_types)
                except Exception as e:
                    return _make_result(unique_key, "UNSUPPORTED_TYPE", "cpp", [], len(tests), detail=str(e))

                ok, exe, ce = compile_cpp(harness, td)
                if not ok:
                    return _make_result(unique_key, "CE", "cpp", [], len(tests), compile_error=ce[:4000])

                for i, t in enumerate(tests):
                    if _should_stop(case_results, max_consec):
                        for j in range(i, len(tests)):
                            case_results.append({"case": j, "status": "SKIPPED"})
                        break

                    args_json = t.get("args_json", "[]")
                    exp_json = t.get("expected_json", "null")
                    try:
                        rc, out, err, dt = run_process(
                            [exe], args_json, cwd=td, timeout_s=timeout_s, cpu_s=cpu_s, mem_mb=mem_mb,
                        )
                    except subprocess.TimeoutExpired:
                        case_results.append({"case": i, "status": "TLE"})
                        continue

                    if rc != 0:
                        case_results.append({"case": i, "status": "RE", "stderr": err[:500]})
                        continue

                    if compare_function_outputs(out.strip(), exp_json.strip(), float_tol=cfg.float_tol):
                        case_results.append({"case": i, "status": "AC"})
                    else:
                        case_results.append({"case": i, "status": "WA", "got": out[:500], "exp": exp_json[:500]})

                overall = "AC" if all(c["status"] == "AC" for c in case_results) else next(c["status"] for c in case_results if c["status"] != "AC")
                return _make_result(unique_key, overall, "cpp", case_results, len(tests))

            return _make_result(unique_key, "UNSUPPORTED_LANG", lang, [], len(tests))

    # ----------------------------------------------------------------
    # stdio / file mode
    # ----------------------------------------------------------------
    tests = pick_stdio_tests(cases, include_private=cfg.include_private)
    if not tests:
        return _make_result(unique_key, "NO_TESTS", lang, [], 0)

    case_results: List[dict] = []

    with tempfile.TemporaryDirectory() as td:
        # prepare executable
        if lang == "python":
            py = write_python(code, td)
            cmd = [sys.executable, "-I", "-S", py]
        else:
            ok, exe, ce = compile_cpp(code, td)
            if not ok:
                return _make_result(unique_key, "CE", "cpp", [], len(tests), compile_error=ce[:4000])
            cmd = [exe]

        for i, t in enumerate(tests):
            if _should_stop(case_results, max_consec):
                for j in range(i, len(tests)):
                    case_results.append({"case": j, "status": "SKIPPED"})
                break

            inp = t.get("input", "")
            exp = t.get("output", "")
            cr: dict = {"case": i}  # current case result

            try:
                if mode == "file":
                    with open(os.path.join(td, "input.txt"), "w", encoding="utf-8") as f:
                        f.write(io_to_text(inp))
                    try:
                        rc, out, err, dt = run_process(
                            cmd, "", cwd=td, timeout_s=timeout_s, cpu_s=cpu_s, mem_mb=mem_mb,
                            extra_env={"PYTHONNOUSERSITE": "1"} if lang == "python" else None,
                        )
                    except subprocess.TimeoutExpired:
                        cr["status"] = "TLE"
                        case_results.append(cr)
                        continue
                    if rc != 0:
                        cr.update({"status": "RE", "stderr": err[:500]})
                        case_results.append(cr)
                        continue
                    out_path = os.path.join(td, "output.txt")
                    got = ""
                    if os.path.exists(out_path):
                        got = open(out_path, "r", encoding="utf-8", errors="replace").read()

                    if checker_py:
                        okc, chk_out, chk_err = run_checker(checker_py, td, inp, exp, got, timeout_s)
                        if okc:
                            cr["status"] = "AC"
                        else:
                            cr.update({"status": "WA_CHECKER", "checker_out": chk_out[:500]})
                    else:
                        if compare_output(got, exp, float_tol=cfg.float_tol):
                            cr["status"] = "AC"
                        else:
                            cr.update({"status": "WA", "got": got[:500], "exp": exp[:500]})
                else:
                    # stdio
                    try:
                        rc, got, err, dt = run_process(
                            cmd, inp, cwd=td, timeout_s=timeout_s, cpu_s=cpu_s, mem_mb=mem_mb,
                            extra_env={"PYTHONNOUSERSITE": "1"} if lang == "python" else None,
                        )
                    except subprocess.TimeoutExpired:
                        cr["status"] = "TLE"
                        case_results.append(cr)
                        continue
                    if rc != 0:
                        cr.update({"status": "RE", "stderr": err[:500]})
                        case_results.append(cr)
                        continue

                    if checker_py:
                        okc, chk_out, chk_err = run_checker(checker_py, td, inp, exp, got, timeout_s)
                        if okc:
                            cr["status"] = "AC"
                        else:
                            cr.update({"status": "WA_CHECKER", "checker_out": chk_out[:500]})
                    else:
                        if compare_output(got, exp, float_tol=cfg.float_tol):
                            cr["status"] = "AC"
                        else:
                            cr.update({"status": "WA", "got": got[:500], "exp": exp[:500]})

                case_results.append(cr)

            finally:
                for fn in ("input.txt", "output.txt", "checker.py", "correct_output.txt", "solution_output.txt"):
                    p = os.path.join(td, fn)
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

        overall = "AC" if case_results and all(c["status"] == "AC" for c in case_results) else (
            next((c["status"] for c in case_results if c["status"] != "AC"), "NO_TESTS")
        )
        return _make_result(unique_key, overall, lang, case_results, len(tests))

# ----------------------------
# CLI
# ----------------------------

def load_cases_index(cases_path: str) -> Dict[str, dict]:
    idx: Dict[str, dict] = {}
    for row in iter_jsonl(cases_path):
        k = row.get("unique_key")
        if k:
            idx[k] = row
    return idx

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="nemotron_cp_cases_34799.jsonl")
    ap.add_argument("--gens", required=True, help="model outputs jsonl with unique_key + llm_output")
    ap.add_argument("--out", required=True, help="results jsonl")
    ap.add_argument("--llm_field", default="llm_output", help="field name in gens jsonl")
    ap.add_argument("--include_private", action="store_true", help="include code_contests private tests")
    ap.add_argument("--float_tol", type=float, default=1e-6, help="float tolerance for function outputs")
    ap.add_argument("--default_timeout", type=float, default=2.0)
    ap.add_argument("--default_mem_mb", type=int, default=1024)
    args = ap.parse_args()

    cfg = JudgeConfig(
        float_tol=args.float_tol,
        include_private=args.include_private,
        default_timeout_s=args.default_timeout,
        default_mem_mb=args.default_mem_mb,
    )

    cases_idx = load_cases_index(args.cases)

    def results_iter():
        for gen in iter_jsonl(args.gens):
            k = gen.get("unique_key")
            out = gen.get(args.llm_field, "")
            if not k:
                continue
            prob = cases_idx.get(k)
            if not prob:
                yield {"unique_key": k, "status": "NO_CASES", "lang": None}
                continue
            try:
                r = judge_one(prob, out, cfg)
            except Exception as e:
                r = {"unique_key": k, "status": "JUDGE_ERROR", "error": str(e)[:2000]}
            yield r

    write_jsonl(args.out, results_iter())
    print("wrote:", args.out)

if __name__ == "__main__":
    main()
