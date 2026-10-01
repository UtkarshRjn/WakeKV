# WakeKV

**WakeKV: Reactive, Reversible KV Residency for Heads That Change Their Minds**

Accepted to the NeurIPS 2026 Workshop on ML for Systems (ML for Systems 2026).

Most KV-cache compression methods classify each attention head once, offline
or at prefill, and keep that decision fixed for the rest of generation.
WakeKV does not. Cooling heads are demoted to a recoverable CPU reservoir
instead of being frozen in place or permanently evicted, and a demoted page
is fetched back the moment it is needed again. Across 1.5B–8B models, that
reactive rule lowers miss rate against frozen classification and destructive
eviction at matched memory or budget, and a FlexiCache/vLLM deployment on
Mistral-7B improves throughput while holding LongBench quality.

**Paper:** [`paper/wakekv_mlforsys.pdf`](paper/wakekv_mlforsys.pdf)

```bibtex
@inproceedings{ranjan2026wakekv,
  title     = {WakeKV: Reactive, Reversible {KV} Residency for Heads That Change Their Minds},
  author    = {Ranjan, Utkarsh},
  booktitle = {NeurIPS 2026 Workshop on ML for Systems},
  year      = {2026}
}
```

- [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) — claims, phases, decision gates,
  risks, compute budget.
- [`notes/`](notes/) — literature deep-reads and ideation trail.
- [`notes/phase01_results.md`](notes/phase01_results.md) — **Phase 0–1
  results** (numeric record): G0 PASS (heads churn during decoding), G1
  cheap signals cannot predict wake-ups well enough to prefetch. Plain-
  language version: [`docs/phase01_explainer.html`](docs/phase01_explainer.html).
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

### Gotchas

- **Use `--dtype float32` for bf16-native models on pre-Ampere GPUs.**
  `--dtype auto` falls back to fp16 when the GPU lacks bf16 (e.g. GTX 1080 Ti,
  RTX 2080 Ti — anything before Ampere). bf16-native reasoning models such as
  `DeepSeek-R1-Distill-*` **overflow to NaN in fp16**: attention weights go NaN
  from the first layers on, generation collapses to a single repeated token
  (`!!!!…`, argmax of NaN logits), and downstream `needle_score`/`copy_paste`
  become all-NaN / all-False. The symptom in the analysis is an empty binary
  channel (adjacent Jaccard 1.0, 0 active heads) and zero Phase-1 wake events.
  On these GPUs run the R1-Distill CoT task with `--dtype float32` (a 1.5B model
  fits in fp32 on 11 GB); on Ampere+ `auto`→bf16 is fine.

- **CoT needle score is question-lookback, normalized to [0,1].** For `--task
  cot` the "needle" span is the problem statement at the *start* of the prompt.
  Because that span overlaps the local window while the context is still short,
  the sink/local exclusions in the needle-mass denominator (`instrument.py`)
  explicitly skip needle positions — otherwise the denominator collapses and the
  ratio blows past 1 (to ~1e5) in early steps. The exclusion is a no-op for
  NIAH, where the needle is disjoint from sink/local.

## Phase 1 — wake-up signal study

Consumes Phase-0 logs; no extra GPU time. Evaluates candidate early-warning
signals (online RCO, drift trigger, entropy trend, needle-mass delta) for
precision/recall at lead times 1–32 steps, against the PCIe transfer bar
(gate G1 in the plan):

```bash
python scripts/analyze_phase1.py runs/Qwen__Qwen2.5-7B-Instruct/niah \
    --pcie-gbps 21 --decode-step-ms 30   # override with measured values
```

Output: `signal_study.md` per task dir — the G1 gate report.
