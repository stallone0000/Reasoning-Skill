# Coding Reproduction Code

This folder contains the sanitized code needed to reproduce the paper's competitive-programming experiments on Nemotron-Competitive-Programming-v1.

It includes:

- `scripts/judge/`: local compile-and-run judge for Python/C++ outputs.
- `baselines/`: Direct, NoWait, CoD, and TALE baseline runners and statistics scripts.
- `experience_rag/`: hybrid retrieval and TRS inference scripts, including the v5 full-diagnostic branch used by the paper's coding pipeline.
- `experience_rag/exp_v8_failure_multiversion/`: failure-card quality tooling used for later diagnostic iterations.
- `rm_runtime.py`: shared 360/Qihoo-compatible API client helpers. API keys are read from environment variables only.

Large data files are intentionally not duplicated inside the GitHub release folder. The camera-ready staging copy is under:

```text
camera_ready_assets/coding_reproduction/
```

The Hugging Face dataset staging folder contains the normalized coding skill cards:

```text
camera_ready_assets/hf_skill_data/data/coding_nemotron_competitive_programming.jsonl.gz
```

For public release, replace private provider endpoints with your own OpenAI-compatible endpoint settings through environment variables, and run the judge only in an isolated environment such as Docker, nsjail, or a dedicated sandbox.
