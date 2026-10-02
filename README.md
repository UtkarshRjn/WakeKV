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
| `scripts/fetch_logs.sh` | Download the published attention logs into `runs/` |
| `scripts/reproduce_paper.sh` | The paper's five model/regime combinations |
| `scripts/attention/` | Log top-k attention, then churn, wake-up, clustering, and residency reports |
| `scripts/system/` | FlexiCache/vLLM shim, throughput sweep, and LongBench quality |
| `tests/` | CPU unit tests |

Logs and plots stay local. `runs/` and `figures/` are gitignored.

## Setup

```bash
pip install -e .
pytest
```

`requirements.txt` lists the same dependencies. Apache-2.0; see `LICENSE`.

Measurement runs need a CUDA GPU. The 1.5B and 3B logs in the paper were
taken on an RTX 2080 Ti (11 GB); the 8B logs and the Mistral-7B system run
were taken on an A30 (24 GB). Models load with eager attention so weights can
be read. Each step is reduced to top-k immediately.

On pre-Ampere GPUs, pass `--dtype float32` for bf16-native models such as
`DeepSeek-R1-Distill-*`. `--dtype auto` falls back to fp16, and those models
overflow to NaN in fp16.

## Attention logs

`runs/` is gitignored, so a clone does not include them. The paper's
attention logs are the GitHub release
[attention-logs-v1](https://github.com/UtkarshRjn/WakeKV/releases/tag/attention-logs-v1):
20 `log.npz` files, each with a `meta.json` beside it. Derived reports are
not in the archive; the analyze stage writes them.

```bash
bash scripts/fetch_logs.sh
```

The script checks the archive sha256 and unpacks into `runs/`. It does
nothing when that tree is already present, unless you pass `--force`.
The layout is:

| Directory | Runs |
|---|---|
| `runs/Qwen__Qwen2.5-3B-Instruct/niah/` | depths 0.25, 0.50, 0.75 × seeds 0, 1, 2 |
| `runs/Qwen__Qwen2.5-3B-Instruct/multiturn/recall/` | one multi-turn recall |
| `runs/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/cot/` | `math500-0` … `math500-4` |
| `runs/deepseek-ai__DeepSeek-R1-Distill-Llama-8B/cot/` | `math500-0` … `math500-4` |

`log.npz` holds top-k indices and weights, context length, generated token
ids, needle score, and the copy-paste mask. No GPU is required to read them.

```bash
bash scripts/reproduce_paper.sh analyze
```

That writes `summary.md`, `signal_study.md`, `clustering.md`, and
`residency.md` next to each task. The Mistral-7B/NIAH attention logs
could not be recovered, so they are not in the release. The analyze
stage skips that combination unless
`runs/mistralai__Mistral-7B-Instruct-v0.2/niah/` contains a `log.npz`.
Regenerating it uses needle depth 0.5, the harness default; the depth
used for the paper was not recorded. The Mistral-7B FlexiCache system
run is separate from these logs.

## Head churn

```bash
python scripts/attention/log_attention.py --model Qwen/Qwen2.5-3B-Instruct \
    --task niah --context-tokens 5000 --depths 0.25 0.5 0.75 --seeds 0 1 2

python scripts/attention/log_attention.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --task cot --max-new-tokens 2048 --dtype float32

python scripts/attention/log_attention.py --model Qwen/Qwen2.5-3B-Instruct --task multiturn

python scripts/attention/analyze_churn.py runs/Qwen__Qwen2.5-3B-Instruct/niah
```

## Wake-up prediction

Reads the attention logs. Override the PCIe and step-time defaults with
measured values.

```bash
python scripts/attention/analyze_wakeups.py runs/Qwen__Qwen2.5-3B-Instruct/niah \
    --pcie-gbps 21 --decode-step-ms 30
bash scripts/attention/analyze_wakeups_all.sh
```

The combined prediction report is `runs/signal_study_all.md`.

## Residency simulator

Replays logged attention. No GPU.

```bash
python scripts/attention/simulate_residency.py runs/Qwen__Qwen2.5-3B-Instruct/niah
bash scripts/attention/wakekv_simulate_residency_all.sh
```

Per-task reports land in `runs/<model>/<task>/residency.md`. The combined
report is `runs/residency_all.md`.

SnapKV, uniform R-KV, and ReasonAlloc in this simulator are
reimplementations, not the authors' code. SnapKV's observation window is
the log's first step. R-KV and ReasonAlloc score importance only, because
the logs do not contain key vectors, so the redundancy term is omitted.
ReasonAlloc uses an equal per-layer budget instead of that paper's
Reasoning-Wave split. On Mistral-7B/NIAH the R-KV and ReasonAlloc interval
is 8 steps, not the default 128, and those two columns are left out of the
paper's win counts.

The five combinations in the paper, and that Mistral exception, are in
`scripts/reproduce_paper.sh`:

```bash
bash scripts/reproduce_paper.sh log       # GPU
bash scripts/reproduce_paper.sh analyze   # CPU, after logs exist
```

```bash
python scripts/attention/make_figures.py heatmap runs/Qwen__Qwen2.5-3B-Instruct/niah
python scripts/attention/make_figures.py pareto \
    runs/Qwen__Qwen2.5-3B-Instruct/niah \
    runs/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/cot
```

## FlexiCache / vLLM

The system result is a shim over [FlexiCache](https://github.com/NazmulTakbir/FlexiCache)
(Apache 2.0). Set `FLEXI_ROOT` to that checkout before the shell scripts;
they have no default path. `WAKEKV_ROOT` defaults to this repository. The
FlexiCache commit used for the A30 numbers was not recorded here, so record
the commit you build. The shim clears `unstable_heads`, which puts every
head on FlexiCache's sparse top-B path, and sets the rerank interval.
`R=1` is the WakeKV design point.

```bash
export FLEXI_ROOT=/path/to/FlexiCache
bash scripts/reproduce_paper.sh system
python scripts/system/wakekv_sweep_table.py "$FLEXI_ROOT/benchmarks/FlexiCache/Throughput/wakekv_sweep"

python scripts/system/run_wakekv.py --mode reactive --rerank-interval 1 -- \
    python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --enable-flexicache --num-unstable-heads 64 --topK-budget 64 \
    --unstable-heads-profile-task gov_report
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
