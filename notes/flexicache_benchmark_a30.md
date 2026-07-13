# FlexiCache vs vLLM — throughput & accuracy reproduction (Mistral-7B, A30)

Reproduction of FlexiCache's end-to-end throughput claim on our hardware, using the
authors' own benchmark harness (`benchmarks/FlexiCache/Throughput/`). Relevant to this
project because our Phase-0/1 analysis references the FlexiCache temporal-stability
metric (unstable-head fraction, TS<0.5); this checks whether their system-level payoff
actually shows up when driven end-to-end.

## Setup

| | |
|---|---|
| Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| GPU | 1× NVIDIA A30, 24 GB (Ampere **sm_80**) |
| FlexiCache | vLLM 0.8.2 fork, built from source, Triton V1 attn backend |
| Config | `--enable-flexicache --num-unstable-heads 64 --rerank-frequency 16 --topK-budget 64 --unstable-heads-profile-task gov_report` |
| Baseline | same engine, FlexiCache off (`topK=0`) = stock vLLM full-KV |
| Host KV pool | 16 GB (paper default 180 GB; trimmed to fit a 32 GB container) |

**This is not the paper's testbed.** The paper used an H100 94 GB with a 180 GB host
KV pool and 30k-token / 500-prompt inputs. The A30 has 24 GB, so we scaled the run to a
point where **both** the dense baseline and FlexiCache fit in memory:

| Param | Paper (H100) | Here (A30) |
|---|---|---|
| input length | 30,000 | 8,000 |
| num prompts | 500 | 24 |
| output lengths | 8-point sweep | 100, 500, 1000 |
| max-num-seqs | 64 | 16 |
| gpu-mem-util | 0.95 | 0.90 |

## Results

Generation throughput (`output_tokens_per_second`), 24 prompts × 8k-token input:

| output len | vLLM baseline | FlexiCache | **speedup** | vLLM wall | FC wall |
|---:|---:|---:|---:|---:|---:|
| 100 | 57.8 tok/s | 53.8 tok/s | **0.93×** | 41.5 s | 44.6 s |
| 500 | 132.5 tok/s | 174.4 tok/s | **1.32×** | 90.6 s | 68.8 s |
| 1000 | 154.7 tok/s | 257.8 tok/s | **1.67×** | 155.1 s | 93.1 s |

## Read

- **Speedup grows with output length** — FlexiCache's central claim reproduces. Short
  generations (output=100) are marginally *slower* (0.93×): the rerank/offload overhead
  isn't amortized when there's little decoding. By output=1000 FlexiCache is **1.67×
  faster** and clears the same 24 prompts in 93 s vs 155 s (**40 % less wall time**).
- **Mechanism**: longer decode → more steps reusing the stable-head top-K resident on
  GPU while offloaded KV stays on host → less GPU KV pressure, larger effective batch,
  less per-step attention compute. Crossover is between output 100 and 500.
- Paper reports **1.38×–1.55×** offline throughput on an H100; we see **1.32×–1.67×** on
  an A30 at 8k context. Absolute numbers aren't comparable (different GPU, much smaller
  scale) but the **relative trend holds** — which is what the authors' guide says to check.

## Accuracy retention (LongBench)

Second reproduction: does FlexiCache's top-K sparse attention preserve output quality?
Dense (stock vLLM, full KV) vs FlexiCache (topK=64), Mistral-7B, on a 4-task LongBench
subset (2 QA + multi-hop + summarization). Scaled for the A30: 8k truncation, batch 4,
**30 samples/task** (paper uses full ~200/task).

| Task | Metric | Dense | FlexiCache | Δ |
|---|---|---:|---:|---:|
| Qasper | F1 | 35.03 | 38.24 | +3.21 |
| 2WikiMQA | F1 | 15.33 | 17.28 | +1.95 |
| TriviaQA | F1 | 87.06 | 87.06 | 0.00 |
| MultiNews | Rouge-L | 26.11 | 25.83 | −0.28 |
| **Avg ratio (FC/Dense)** | | — | **1.05** | |

**Read:** FlexiCache preserves accuracy — TriviaQA identical, MultiNews within 0.3, the
two QA tasks nominally higher. The 1.05 ratio is **not** evidence FlexiCache is more
accurate; at 30 samples/task the QA bumps are noise. Conclusion: **no quality loss from
the 64-page sparse-attention budget**, matching the paper's ~99% retention. Full-sample
LongBench would tighten the ratio toward 1.00.

Reproduce: `bash /home/utranjan/flexicache_longbench_a30.sh`
(local runner edits: `run_benchmark.py` +`--limit`/`trust_remote_code`,
`config/model2maxlen.json` Mistral 8k cap — none pushed upstream).

## Caveats

- Small N (24 prompts, 3 output lengths) → some run-to-run noise.
- `cpu_kv_cache_size` trimmed 180 → 16 GB; `sm_80` build; single A30 vs H100. Ranks the
  two policies on *this* box; does not predict the paper's absolute throughput.
- Environment adaptations (CUDA 12.4 toolkit, `MAX_JOBS=4` for a 32 GB build cgroup,
  `TORCH_CUDA_ARCH_LIST=8.0`) are local only — no upstream code changes.

## Reproduce

```bash
# built FlexiCache conda env, Mistral access on HF required
bash /home/utranjan/flexicache_throughput_a30.sh   # idempotent; skips completed runs
```

Raw JSON: `FlexiCache/benchmarks/FlexiCache/Throughput/Results_A30/`
