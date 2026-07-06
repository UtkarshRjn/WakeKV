# dynamic-head-kv (private)

Research repo for **WakeKV** (working title): reversible, decode-time
head-priority management for the LLM KV cache — demote cooling attention
heads by offloading their KV to CPU, detect heads waking up, promote them
back. Targeting a workshop paper (NeurIPS 2026 workshops primary, ICLR
2027 workshops backup).

- [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) — claims, phases, decision gates,
  risks, compute budget.
- [`notes/`](notes/) — literature deep-reads and ideation trail.
- [`wakekv/`](wakekv/) + [`scripts/`](scripts/) — implementation (see below).

## Phase 0 — churn measurement harness

Measures whether head importance shifts during decoding on OUR target
models/regimes (gate G0 in the plan). Runs on one 24 GB+ CUDA GPU.

```bash
pip install -r requirements.txt
pytest tests/            # metrics unit tests (CPU-only)

# NIAH churn (2602.11162 replication on our models)
python scripts/run_phase0.py --model Qwen/Qwen2.5-7B-Instruct \
    --task niah --context-tokens 5000 --depths 0.25 0.5 0.75 --seeds 0 1 2

# Long-CoT question-lookback churn (reasoning model)
python scripts/run_phase0.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --task cot --max-new-tokens 2048

# Multi-turn recall churn
python scripts/run_phase0.py --model Qwen/Qwen2.5-7B-Instruct --task multiturn

# Analyze + G0 gate report
python scripts/analyze_phase0.py runs/Qwen__Qwen2.5-7B-Instruct/niah
```

Notes: models load with `attn_implementation="eager"` (needed to read
attention weights); each step's attention is reduced to top-k immediately,
so logs stay ~100–200 MB per run.

**Private:** contains unpublished research strategy. Do not make public
before the preprint is out.
