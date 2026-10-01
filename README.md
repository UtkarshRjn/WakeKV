# WakeKV

**WakeKV: Reactive, Reversible KV Residency for Heads That Change Their Minds**

Accepted to the NeurIPS 2026 Workshop on ML for Systems.

Most KV-cache compression methods classify each attention head once, offline
or at prefill, and keep that decision for the rest of generation. WakeKV does
not. Each head keeps an LRU working set on GPU, capped at a budget. A page
that falls out of that budget is copied to a CPU reservoir, and fetched back
the moment a later step needs it. A stale choice costs one recoverable stall
instead of a permanent loss of that head's history.

Across Qwen2.5-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B, and
DeepSeek-R1-Distill-Llama-8B, on needle retrieval, long chain-of-thought, and
multi-turn recall, most heads change what they read at least once during
generation. Cheap online signals do not predict those changes well enough to
prefetch. At matched memory or budget, reacting to demand lowers miss rate
against a frozen classifier, against destructive eviction on the same LRU
schedule, and against SnapKV, uniform R-KV, and ReasonAlloc. A
FlexiCache/vLLM deployment on Mistral-7B (A30) improves throughput over stock
FlexiCache while holding LongBench quality.

```bibtex
@inproceedings{ranjan2026wakekv,
  title     = {WakeKV: Reactive, Reversible {KV} Residency for Heads That Change Their Minds},
  author    = {Ranjan, Utkarsh},
  booktitle = {NeurIPS 2026 Workshop on ML for Systems},
  year      = {2026}
}
```

## What is in this repo

| Path | Role |
|---|---|
| `wakekv/` | Churn metrics, wake-up signals, the residency simulator, and the FlexiCache/vLLM shim |
| `scripts/run_phase0.py` | Log top-k attention during decoding |
| `scripts/analyze_phase0.py` | Head-churn report from those logs |
| `scripts/analyze_phase1.py` | Wake-up prediction study (no extra GPU time) |
| `scripts/simulate_residency.py` | Miss rate for reactive, frozen, evict, SnapKV, R-KV, and ReasonAlloc |
| `scripts/run_wakekv.py` | Run a FlexiCache/vLLM command with the WakeKV shim installed |
| `scripts/wakekv_rerank_sweep.sh` | Throughput sweep on a FlexiCache checkout |
| `scripts/wakekv_longbench_check.sh` | LongBench quality check beside that sweep |
| `tests/` | CPU unit tests |

Logs and plots stay local. `runs/` and `figures/` are gitignored.

## Setup

```bash
pip install -r requirements.txt
pytest tests/
```

Measurement runs need a CUDA GPU. The 1.5B and 3B logs in the paper were
taken on an RTX 2080 Ti (11 GB); the 8B logs and the Mistral-7B system run
were taken on an A30 (24 GB). Models load with eager attention so weights can
be read. Each step is reduced to top-k immediately.

On pre-Ampere GPUs, pass `--dtype float32` for bf16-native models such as
`DeepSeek-R1-Distill-*`. `--dtype auto` falls back to fp16, and those models
overflow to NaN in fp16.

## Head churn

```bash
python scripts/run_phase0.py --model Qwen/Qwen2.5-3B-Instruct \
    --task niah --context-tokens 5000 --depths 0.25 0.5 0.75 --seeds 0 1 2

python scripts/run_phase0.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --task cot --max-new-tokens 2048 --dtype float32

python scripts/run_phase0.py --model Qwen/Qwen2.5-3B-Instruct --task multiturn

python scripts/analyze_phase0.py runs/Qwen__Qwen2.5-3B-Instruct/niah
```

## Wake-up prediction

Reads the Phase-0 logs. Override the PCIe and step-time defaults with
measured values.

```bash
python scripts/analyze_phase1.py runs/Qwen__Qwen2.5-3B-Instruct/niah \
    --pcie-gbps 21 --decode-step-ms 30
```

## Residency simulator

Replays logged attention. No GPU.

```bash
python scripts/simulate_residency.py runs/Qwen__Qwen2.5-3B-Instruct/niah
bash scripts/wakekv_simulate_residency_all.sh
```

Per-task reports land in `runs/<model>/<task>/residency.md`. The combined
report is `runs/residency_all.md`.

```bash
python scripts/make_figures.py heatmap runs/Qwen__Qwen2.5-3B-Instruct/niah
python scripts/make_figures.py pareto \
    runs/Qwen__Qwen2.5-3B-Instruct/niah \
    runs/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/cot
```

## FlexiCache / vLLM

The system result is a shim over [FlexiCache](https://github.com/NazmulTakbir/FlexiCache)
(Apache 2.0). Point `FLEXI_ROOT` at that checkout and `WAKEKV_ROOT` at this
one. Reactive mode classifies no heads as stable and reranks on the interval
you set. `R=1` is the WakeKV design point.

```bash
python scripts/run_wakekv.py --mode reactive --rerank-interval 1 -- \
    python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --enable-flexicache --num-unstable-heads 64 --topK-budget 64 \
    --unstable-heads-profile-task gov_report

bash scripts/wakekv_rerank_sweep.sh
bash scripts/wakekv_longbench_check.sh
```

The target command has to start with `python` or `python3`. The shim is
installed inside that process; replacing the process with `exec` would drop
the patch and silently run stock FlexiCache.

## Future work

The paper leaves these open:

- The measurement covers 1.5B–8B models, and not every model is run in every
  regime.
- The simulator counts misses, not PCIe stall time, and it compares against
  FlexiCache-style policies rather than that system's exact code.
- The real-system result is one model and one regime (Mistral-7B) on one A30.
- The next measurement is reversible versus destructive eviction on a real
  system, scored by stall time.
