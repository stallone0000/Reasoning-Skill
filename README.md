# Thinking with Reasoning Skills (TRS)

This folder contains a lightweight, sanitized implementation of the main TRS pipeline used in the paper:

1. Generate source CoT / reasoning traces for source problems.
2. Distill the traces into reusable skill cards.
3. Retrieve relevant skills for a new query.
4. Inject retrieved skills into the prompt and run TRS inference.

The code intentionally excludes datasets and private experiment logs. API keys are never stored in this repository; configure them through environment variables.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`, then export it into the shell before online runs:

```bash
set -a
source .env
set +a
```

Alternatively, export variables directly:

```bash
export TRS_API_KEY=""
export TRS_API_BASE_URL="https://api.openai.com/v1/chat/completions"
export TRS_MODEL="your-inference-model"
export TRS_SOURCE_MODEL="your-source-cot-model"
export TRS_DISTILL_MODEL="your-summarizer-model"
```

Any OpenAI-compatible chat-completions endpoint can be used by setting `TRS_API_BASE_URL` to the full chat-completions URL.

## Quick Start

The `scripts/` directory contains the runnable release of the pipeline,
including mock mode, benchmark export, and representation-library construction
utilities. The examples below run fully offline with `--mock`; remove `--mock`
and set the model/API variables above for online runs.

Generate source reasoning traces:

```bash
python scripts/generate_source_cot.py \
  data/sample_problems.jsonl \
  prompts/source_cot_prompt.txt \
  outputs/source_traces.jsonl \
  --mock
```

Distill skill cards:

```bash
python scripts/distill_skill_cards.py \
  outputs/source_traces.jsonl \
  prompts/skill_distillation_prompt.txt \
  outputs/skill_cards.jsonl \
  --mock
```

Run TRS inference:

```bash
python scripts/retrieve_and_infer.py \
  data/sample_benchmark.jsonl \
  outputs/skill_cards.jsonl \
  prompts/trs_prompt.txt \
  outputs/trs_predictions.jsonl \
  --mock \
  --top-k 1
```

Run direct and TRS benchmark conditions together:

```bash
python scripts/benchmark_runner.py \
  --input-file data/sample_benchmark.jsonl \
  --library-file outputs/skill_cards.jsonl \
  --results-root outputs/benchmark \
  --modes direct,trs \
  --mock
```

Export summary tables:

```bash
python scripts/export_benchmark_summary.py \
  --results-root outputs/benchmark \
  --output-dir outputs/tables
```

## File Layout

- `prompts/source_cot_prompt.txt`: prompt for generating original CoT traces.
- `prompts/skill_distillation_prompt.txt`: prompt for converting traces into skill cards.
- `prompts/direct_prompt.txt`: direct baseline prompt.
- `prompts/trs_prompt.txt`: skill-injected TRS prompt.
- `scripts/common.py`: OpenAI-compatible API client, prompt rendering, answer extraction, and BM25 retrieval.
- `scripts/generate_source_cot.py`: source-trace generation.
- `scripts/distill_skill_cards.py`: skill-card distillation and XML parsing.
- `scripts/summarize_reasoning.py`: free-form summary baseline.
- `scripts/build_representation_library.py`: structured/free-summary/raw-example/raw-CoT retrieval-library construction.
- `scripts/retrieve_and_infer.py`: lexical skill retrieval and TRS inference.
- `scripts/benchmark_runner.py`: paired Direct/TRS benchmark runner.
- `scripts/export_benchmark_summary.py`: condition and comparison summary export.
- `coding/`: sanitized competitive-programming reproduction code copied from the original Nemotron CP experiments, including local judge utilities, baseline runners, v5 hybrid retrieval code, and failure-card quality tools. Large data and indices are staged separately under `camera_ready_assets/coding_reproduction/`.
- `data/`: tiny mock data files used by the expanded scripts.

## Notes

- The paper uses larger production scripts and experiment runners; this release keeps only the core logic needed to reproduce the method.
- The default retriever is a small lexical scorer for portability. For large-scale experiments, replace it with BM25 / hybrid retrieval as described in the paper.
- Dataset and skill-card releases are prepared separately for Hugging Face.
- Coding experiments use a local compile-and-run judge and optional BGE-M3/FAISS hybrid retrieval. See `coding/README.md` and the `coding_reproduction` staging folder for the full data/index artifacts.
- Never commit `.env` files or provider API keys. Use `.env.example` only as a template.
