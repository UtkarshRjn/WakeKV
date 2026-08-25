# M2b-0 Runbook — Reproduce FlexiCache on wolverine

*The first Phase-2b milestone. Time-box: 2 working days. **No code changes
to FlexiCache** at this stage — the whole point is to prove their code
runs on our hardware and produces their published quality numbers before
we start touching it.*

**Machine:** wolverine, **H100 81.5GB HBM3** (not the A30 this doc originally
assumed — corrected after the first real attempt), 503GB system RAM,
measured H2D PCIe **55.7 GB/s** (see Step 0). CUDA driver/version: TBD,
confirm via `nvidia-smi` on next attempt.
**Goal:** LongBench average score on at least one of their tested models,
within ±0.5 of their Table 4.

**Known blocker (as of the first real attempt on wolverine): disk space.**
Environment setup and dependency resolution (dry-run) succeeded — torch
2.6.0+cu124 resolves cleanly — but the actual installs are blocked: root
`/` had only 3.6GB free and `/mnt/conda` had 0 bytes, against torch's
~6-7GB installed footprint plus vLLM 0.8.2's from-source build (tens of
GB of CUDA object files). **Do not force this** — filling a shared box's
root partition to zero risks other users' jobs and will fail partway
anyway. Find a larger mount (`df -h`) before retrying Step 1.

---

## Success bar (concrete)

From FlexiCache MLSys 2026 Table 4, budget 1024:

| Model | LongBench avg (their number) | Our target |
|---|---|---|
| Llama-3.1-8B-Instruct | ~49.4 | 48.9 – 49.9 |
| Mistral-7B-Instruct | ~49.0 | 48.5 – 49.5 |

Either passing = success. Multiple models is bonus, not required.

**Success bar does NOT include throughput.** FlexiCache reports 1.4× on
H100/PCIe-5.0 — since wolverine is also an H100, our throughput should
land close to their own testbed's shape, not the "expect noticeably
slower" caveat this doc originally had for an assumed A30. Two hardware
deltas still worth watching: their card is the 94GB NVL variant vs our
81.5GB (may bind on batch size / host KV pool sooner), and their
`config.json` defaults a 180GB host KV reservoir against our 503GB total
RAM but currently only ~480GB available.

## Failure modes to anticipate (with mitigations)

| Symptom | Likely cause | First fix |
|---|---|---|
| Install fails partway through, disk fills up | Shared box's root partition too small for torch (~6-7GB) + vLLM source build (tens of GB) | Check `df -h` for a larger mount FIRST; don't start `pip install -e .` on a box already near capacity — see "Known blocker" above |
| OOM at prefill | vLLM's memory profiler is too aggressive | `--gpu-memory-utilization 0.85` |
| Their build errors on CUDA 13 | Their setup pinned CUDA 12.x | Match their pinned Torch/CUDA (see their `requirements/` dir); confirmed pin is torch 2.6.0+cu124 (see Step 1) |
| Modified Triton kernel fails to compile | Triton version mismatch | Confirmed: Triton 3.2.0 resolves cleanly against their pin (dry-run verified) |
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
# expect: NVIDIA H100

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
# MEASURED on wolverine: 55.7 GB/s (H100/PCIe-5.0 -- ~2.6x the A30/Gen4
# 21 GB/s this doc originally assumed). Use --pcie-gbps 55.7 for any
# transfer_steps_needed() call on this host -- the 21.0 default in
# wakekv/signals.py is HeteroCache's measured Gen4 number, not this box's.
```

**Record the measured H2D number** in your run log — this becomes the
`--pcie-gbps` argument for all Phase 2b analyses on this host. Do not use
a theoretical peak or another host's measured number; use what wolverine
actually achieves (55.7 GB/s, confirmed above).

## Step 1 — Clone and build FlexiCache (2 hours)

```bash
cd ~ && git clone https://github.com/NazmulTakbir/FlexiCache
cd FlexiCache

# Read their README first, not the setup script — the setup script may
# assume a specific CUDA/Torch pin. Note the pinned versions before
# blindly installing.
head -80 README.md

# CONFIRMED pin (dry-run resolved cleanly on wolverine): torch 2.6.0+cu124,
# Python 3.12, Triton 3.2.0, transformers 4.50.0, datasets 3.6.0, CUDA
# runtime 12.8. TORCH_CUDA_ARCH_LIST="9.0;9.0a" in their config is already
# correct for this H100 (compute cap 9.0) -- no override needed.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# pip install -e . FIRST -- this builds vLLM 0.8.2 from source (tens of GB,
# ~30 min at MAX_JOBS=20). Needs real free disk -- see "Known blocker" above.
pip install -e .

# THEN the extra deps -- there is no root requirements.txt, only a
# requirements/ directory; the FlexiCache-specific extras are here and
# install AFTER -e ., not before:
pip install -r requirements/flexicache_requirements.txt

# Quick sanity: there is no top-level `flexicache` package -- it lives at
# vllm/v1/flexicache/. This is the working check (matches
# docs/phase2b_m1b_smoke_test.md):
python -c "from vllm.v1.flexicache.config import FlexiCacheConfig; print('ok')"
```

One thing to watch for and NOT ignore:
- **Disk space.** `pip install -e .` alone needs tens of GB for the vLLM
  build. Confirm free space on a real mount (not just `~`, which may be on
  a small root partition) before starting -- see "Known blocker" above.

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
paper says GovReport takes ~2h on Llama-3.1-8B on H100 — wolverine is also
an H100, so expect close to their number, not the ~4h an A30 would need.

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
    --out results/longbench-flexicache-wolverine.json
```

Their default flags in the paper: budget 1024, unstable 25%, rerank every
16 steps. Do **not** deviate on this run — the point is to match Table 4.

## Step 5 — Compare (30 min)

Compute LongBench average across their 16 tasks (or whatever subset their
harness produces), then compare to their published number:

- **Within ±0.5 → M2b-0 PASS.** Record the numbers, note the achieved
  PCIe bandwidth from Step 0, note the throughput achieved on wolverine
  vs. their published H100 number (for the paper — this is the
  reproduction receipt), advance to M2b-1.
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
  throughput on wolverine (tokens/sec) vs their published H100 number for
  context, any version mismatches noted, and a green/yellow/red verdict.
- `paper/wakekv.tex` Setup section: update the follow-up-hardware note
  with the measured wolverine PCIe bandwidth from Step 0 (55.7 GB/s).

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
