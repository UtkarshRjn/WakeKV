# M2b-0 Runbook — Reproduce FlexiCache on wolverine

*The first Phase-2b milestone. Time-box: 2 working days. **No code changes
to FlexiCache** at this stage — the whole point is to prove their code
runs on our hardware and produces their published quality numbers before
we start touching it.*

**Machine:** wolverine, A30 24GB HBM2, PCIe Gen4 x16, CUDA 13.0, driver 580.
**Goal:** LongBench average score on at least one of their tested models,
within ±0.5 of their Table 4.

---

## Success bar (concrete)

From FlexiCache MLSys 2026 Table 4, budget 1024:

| Model | LongBench avg (their number) | Our target |
|---|---|---|
| Llama-3.1-8B-Instruct | ~49.4 | 48.9 – 49.9 |
| Mistral-7B-Instruct | ~49.0 | 48.5 – 49.5 |

Either passing = success. Multiple models is bonus, not required.

**Success bar does NOT include throughput.** FlexiCache reports 1.4× on
H100/PCIe-5.0; on A30/PCIe-4.0 with narrower HBM and PCIe, expect
noticeably slower. Absolute throughput will diverge from their paper for
hardware reasons and that's fine — the ratio (their sparse vs their
dense) should still be in the same shape.

## Failure modes to anticipate (with mitigations)

| Symptom | Likely cause | First fix |
|---|---|---|
| OOM at prefill | vLLM's memory profiler is too aggressive | `--gpu-memory-utilization 0.85` |
| Their build errors on CUDA 13 | Their setup pinned CUDA 12.x | Match their pinned Torch/CUDA (see their `requirements.txt`); PyTorch 2.5 + CUDA 12.4 is likely their target |
| Modified Triton kernel fails to compile | Triton version mismatch | Use whatever Triton their pyproject pins |
| LongBench score wildly off | Wrong tokenizer / chat template | Print the actual prompt at inference; compare to their expected prefix |
| Score off by ~5 pts | Their profile file is model-specific | Confirm using the profile they shipped for this exact model |

If we hit any of these past ~4 hours of poking, we skip ahead: run Full-KV
LongBench on their harness (no FlexiCache-specific pieces) as our own
baseline anchor and move to M2b-1. Losing M2b-0 is not fatal — losing
2 days on it *is*.

---

## Step 0 — Environment (30 min)

```bash
# Assumes conda or venv already available; use whichever the box has.
python -m venv ~/flexi-env
source ~/flexi-env/bin/activate
pip install --upgrade pip

# Check we can see the GPU from Python.
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
# expect: NVIDIA A30

# Measure achieved PCIe bandwidth. We'll use this number in Phase 2b analyses.
python - <<'PY'
import torch, time
size = 256 * 1024 * 1024  # 256 MB, a realistic reservoir-transfer chunk
cpu = torch.empty(size, dtype=torch.uint8, pin_memory=True)
gpu = torch.empty(size, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20):
    gpu.copy_(cpu, non_blocking=True)
torch.cuda.synchronize()
gb_s = 20 * size / (time.perf_counter() - t) / 1e9
print(f"H2D achieved: {gb_s:.1f} GB/s")
PY
# expect: something in the 20-28 GB/s range at Gen4 x16
```

**Record the measured H2D number** in your run log — this becomes the
`--pcie-gbps` argument for all Phase 2b analyses. Do not use the 32 GB/s
theoretical peak; use what your box actually achieves.

## Step 1 — Clone and build FlexiCache (2 hours)

```bash
cd ~ && git clone https://github.com/NazmulTakbir/FlexiCache
cd FlexiCache

# Read their README first, not the setup script — the setup script may
# assume a specific CUDA/Torch pin. Note the pinned versions before
# blindly installing.
head -80 README.md

# If they pin CUDA 12.4 and torch 2.5 (typical for MLSys 2026 papers), do:
#   pip install torch==2.5.* --index-url https://download.pytorch.org/whl/cu124
# then install their remaining requirements.

pip install -r requirements.txt
# custom kernels / triton build:
pip install -e .   # or whatever their install command is

# Quick sanity: import their package and instantiate a tiny model.
python -c "import flexicache; print(flexicache.__version__)"
```

Two things to watch for and NOT ignore:
1. **CUDA 13 vs their CUDA 12.x pin.** If their code assumes 12.x, install
   the pinned Torch wheel from PyTorch's index; CUDA 13 back-compat should
   let it run. If it crashes, install a matching CUDA userspace via conda
   (`conda install -c conda-forge cudatoolkit=12.4`) — don't force-upgrade.
2. **Triton version.** Their custom Flash-Decoding may pin a specific
   Triton. Symptom is a compile-time error on first attention call.

## Step 2 — Download the target model (30 min)

```bash
# Their Llama-3.1-8B-Instruct is HF-gated; use HF login first.
huggingface-cli login   # paste your HF token
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ~/models/Llama-3.1-8B-Instruct
```

If Llama-3.1 gating is blocking, use **Mistral-7B-Instruct-v0.2** instead —
their paper reports it too, no gating.

## Step 3 — Run their profiling pass (1 hour)

FlexiCache's classification comes from a one-time profile per model. Their
paper says GovReport takes ~2h on Llama-3.1-8B on H100; expect ~4h on A30
(half the throughput, roughly).

```bash
# Command shape (check their scripts/README for the exact form):
python scripts/profile.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --task govreport \
    --out profiles/llama-3.1-8b-govreport.pkl
```

**Skip this and use their shipped profile if they publish one.** Zero-cost
head start. Check `profiles/` in the repo before profiling from scratch.

## Step 4 — Run LongBench with the profile (4 hours)

```bash
# Command shape (verify against their harness):
python scripts/eval_longbench.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --profile profiles/llama-3.1-8b-govreport.pkl \
    --budget 1024 \
    --unstable-frac 0.25 \
    --rerank-interval 16 \
    --out results/longbench-flexicache-a30.json
```

Their default flags in the paper: budget 1024, unstable 25%, rerank every
16 steps. Do **not** deviate on this run — the point is to match Table 4.

## Step 5 — Compare (30 min)

Compute LongBench average across their 16 tasks (or whatever subset their
harness produces), then compare to their published number:

- **Within ±0.5 → M2b-0 PASS.** Record the numbers, note the achieved
  PCIe bandwidth from Step 0, note the throughput they *did* achieve on
  A30 (for the paper — this is the reproduction receipt), advance to M2b-1.
- **Off by 0.5–2.0 → investigate.** Most likely: wrong profile source
  task, wrong budget, wrong tokenizer/chat template. Fix and rerun once.
- **Off by >2.0 or crashing → escalate.** Either their code has a bug on
  our hardware or we are running it wrong. Consider opening an issue on
  their repo with the exact discrepancy; move to M2b-1 in parallel using
  Full-KV as the baseline anchor while waiting.

---

## What lands in the repo when M2b-0 completes

- `notes/phase2b_results.md` (new file) — first section: **M2b-0
  reproduction**. Include: measured PCIe bandwidth, model + task list,
  achieved LongBench average vs their published number, achieved
  throughput on A30 (tokens/sec) vs their H100 number for context, any
  version mismatches noted, and a green/yellow/red verdict.
- `paper/wakekv.tex` Setup section: update the follow-up-hardware note
  with the measured A30 PCIe bandwidth from Step 0.

M2b-1 begins after this doc has real numbers in it.

---

## Time budget honesty

The 2-day time-box is tight if their build breaks on our CUDA 13. If we
hit day 1.5 without a successful LongBench run, that's the trigger to
stop debugging environment and fall through to the fallback plan
(§ scoping doc §2 last paragraph): run FullKV on their harness as our
baseline anchor, note the FlexiCache-repro failure as a limitation, and
proceed to M2b-1 (reactive controller graft). We don't need FlexiCache
to run for our controller to run — we only need it to run to *compare
against them*, which we can defer to a second attempt after M2b-1 is
working.
