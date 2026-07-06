# The Big Four — Deep-Read Briefing (2026-07-06)

*Companion to `kv_cache_ideation_notes.txt`. Four parallel close-reads of the
papers that define the boundary of idea v1 ("dynamic head priority during
decoding + reversible demotion via CPU offload"). Read this instead of the
four papers first; go to the originals with this map in hand.*

**KEEP PRIVATE — contains unpublished research strategy. Do not commit to the
public repo.**

---

## 0. TL;DR

The gap holds after close reading, and it's sharper than the abstract-level
sweep suggested:

- **FlexiCache** (MLSys 2026) has the *machinery* (per-head offload + recall,
  open-source on vLLM) but frozen head classes — and its own tables show the
  premise is shaky (23–35% page churn per rerank even for "stable" heads).
- **HeteroCache** (ACL 2026 Main) has *reversibility* (CPU reservoir) but
  frozen roles/budgets, a structural pivot-as-proxy flaw, and reservoir-less
  "anchor" heads whose mistakes are permanently unrecoverable.
- **ReasonAlloc** (Jun 2026 preprint) has *decode-time dynamism* but
  destructive demotion, no budget-trajectory evidence, math-only eval, and no
  memory-layout story.
- **Retrieval Heads are Dynamic** (ACL 2026) proves the premise (adjacent-step
  Jaccard 0.28–0.51) and *names your project as its future work* — strong
  positioning, but expect concurrent work. Its probe is synchronous only; a
  k-step-ahead predictor is unbuilt.

Framing sentence (unchanged, now evidence-backed): *prior decode-time
reallocation is evict-only (ReasonAlloc); prior reversible head offloading is
static-role (FlexiCache, HeteroCache); we make head priority both dynamic and
reversible.*

---

## 1. Retrieval Heads are Dynamic (arXiv:2602.11162, ACL 2026) — the premise

**What it shows.** Per-decode-step retrieval scoring (binary: head's max
attention hits the needle AND the attended token equals the generated one)
across 5 models (Llama-3.1-8B, Llama-3.2-3B, Qwen3-8B, Llama-2-13B,
Phi-4-mini) on NIAH + HotpotQA.

**Numbers to cite:**
- Adjacent-step Jaccard of active head set: **0.28 (Llama-3.1-8B) to 0.51
  (Phi-4-mini)** — heads churn between consecutive tokens.
- Llama-3.1-8B activates **238 distinct heads** over one generation (~12x the
  static top-20 set); only ~13 active at any step.
- Activation entropy 3.0–4.9 vs ln(20)=3.0 baseline — responsibility spread
  far beyond any static subset.
- **Static substitution fails:** masking per-step dynamic heads → accuracy
  collapses to 0.0 by k=20 masked heads; masking equal numbers of static or
  random heads → only partial degradation. The model tries to compensate via
  static heads and fails. (= your Figure-1 ammunition, already peer-reviewed.)

**The predictive signal (your early-warning candidate):**
- CCA between hidden state at step n and head scores at n+k: top correlation
  **0.966 (k=0), 0.931 (k=1), 0.915 (k=2)**, decaying through k≈10.
- Trained MLP probe (3 hidden layers, ~70–90M params, input = last hidden
  state): F1 0.80–0.86 on per-head activation — but **synchronous (k=0)
  only**. A true k-step-ahead probe (what prefetch needs) was never built.
  Probe latency never measured (though trivially cheap vs a forward pass).

**Gaps you can fill / caveats to respect:**
- Short contexts only (~5K haystack); no 32K–128K where offload matters.
- No phase analysis of churn (random vs correlated with copying/reasoning
  transitions) — characterizing churn timing + per-head predictability
  horizon is itself a novel measurement contribution.
- Binary-threshold artifact not controlled: near-argmax heads flip in/out;
  no continuous-score robustness check of the churn statistic. Don't
  over-lean on the exact Jaccard numbers; replicate with a soft score.
- In their own RAG application, dynamic heads LOSE to static on Llama-3.2-3B
  (0.499 vs 0.539) — small models may behave differently. Check before
  betting the demo on a 3B model.
- **Verbatim future-work sentence — your project, named by them:** "Future
  inference systems could leverage the predictive mechanism (e.g., via
  lightweight probes) to dynamically retain or fetch KV pairs only for the
  specific heads active at the current step." Cite it; move fast.

## 2. FlexiCache (arXiv:2511.00868v2, MLSys 2026) — the machinery

**What it does.** Offline profiling (8 LongBench/L-Eval summarization tasks,
~2h per model) computes per-head temporal stability = random-corrected
overlap (RCO) of Quest-style top-K page sets across decode steps. Bottom 25%
of heads = "unstable": full KV on GPU, reranked every step. Top 75% =
"stable": only top-K pages on GPU, full KV mirrored in pinned host memory,
reranked every 16 steps, only newly-promoted pages fetched back.
vLLM + H100 + PCIe 5.0. Code: github.com/NazmulTakbir/FlexiCache (Apache 2.0).

**Numbers that matter:**
- Accuracy retention ~0.99–1.00 on LongBench/L-Eval at budget 1024–2048.
- Throughput 1.38–1.55x (growing to 2.08x at 1500 output tokens); memory
  savings →75% asymptotically.
- **Premise cracks:** promoted delta per rerank = **23–35% of the top-K set
  even for stable heads**; cross-task unstable-set overlap mean 0.76–0.83,
  **worst pair 0.56**; RCO decays with offset; stability window W=16 is
  circularly defined to equal the rerank interval.
- Ablations: no-rerank = 0.89 avg; all-stable = 0.96 avg — but hiding a
  **21% single-task collapse** (CUAD 37.20→29.38) inside that average.

**Attack surface (all confirmed untested):**
1. Reasoning / long generation: accuracy evals average **<400 output
   tokens**; no reasoning models anywhere.
2. Multi-turn: never tested; a new turn = new query distribution against
   mostly-offloaded stable heads. Measure promoted-delta spikes at turn
   boundaries (their own Table-4 metric, per-step).
3. Cross-genre transfer: profile transfer only shown summarization→
   summarization. GovReport-profile → math/code is unmeasured.
4. **No runtime misclassification detection:** a stable head gone unstable
   silently serves stale pages up to 15 steps. No per-request incident
   analysis exists.

**Honest warning:** in their tested regime the classification-fix headroom is
small (0.96→0.99 average). Your case must be made in the untested regimes
above, or on efficiency (smaller promoted deltas / fewer request pauses) —
not on LongBench averages.

**Reuse (biggest practical win of the whole lit check):**
- GPU-resident **MinMax score cache decoupled from KV pages** — lets you
  score offloaded pages for recall. This IS the reversibility mechanism.
- Per-(layer,head) block tables + dirty tracking + fused recycle kernel;
  UVA-based async transfer kernels (low-priority stream, chunked).
- The online stability signal you need (overlap of consecutive top-K sets)
  is **already computed at rerank time** — a runtime classifier is nearly
  free. Their "which heads rerank this step" hook is where your dynamic
  controller plugs in. Strongly consider building ON their codebase.

## 3. HeteroCache (arXiv:2601.13684v2, ACL 2026 Main) — the reservoir

**What it does.** Offline calibration (50 Wikipedia samples @10K tokens,
**100 decode steps observed**) computes per-head stability (top-k overlap vs
prefill) and intra-layer similarity, yielding a 2x2 role taxonomy:
- **volatile** (~7%): full KV on GPU (unstable, unique)
- **pivot** (~13%): full KV on GPU, cluster representative + drift monitor
- **anchor** (~4%): statically compressed, **no CPU reservoir**
- **satellite** (~70-77%): compressed on GPU, **full KV in CPU reservoir**
Budgets l_i ∝ 1/stability, frozen. At runtime, pivots monitor drift
(windowed-median overlap vs prefill baseline < 0.5); on trigger, satellites'
GPU slices are overwritten from CPU **at indices chosen by the pivot's
attention**. Roles and budgets never change during decoding (confirmed from
Algorithm 1 + Fig. 2 caption).

**Numbers that matter:**
- LongBench @50% budget: 49.42 vs full 49.77 (beats Quest/ShadowKV/OmniKV/
  CAKE). Up to 3x decode speedup at 224K; 40x vs OmniKV.
- **Drift trigger fires on only 0.823% of decode steps** → hanging a role
  re-evaluation off the same trigger is nearly free.
- Feasibility envelope for your system: CPU RAM 17.95 GB and PCIe 21 GB/s at
  128K context; P99 decode latency ≤0.132s at 224K; works on RTX 4090 +
  PCIe Gen3. (= proof your idea fits a single-GPU budget.)
- Transferability self-defense: re-clustering on real 128K docs keeps only
  ">80%" of satellite assignments — i.e., **up to ~20% of roles are
  input-dependent** by their own measurement.

**Attack surface:**
1. **Anchors** (~4–10% of heads): statically compressed, never monitored,
   never refreshed, no reservoir — any mistake is permanent. This is the
   SnapKV irreversible-discard failure reintroduced at head level, and the
   cleanest single gap for "reversible for everyone + dynamic roles."
2. **Pivot-as-proxy flaw:** satellite refresh indices come from the PIVOT's
   attention, not the satellite's own. A satellite that decouples from its
   cluster mid-run gets wrong tokens fetched and no one notices. Similarity
   was a MEDIAN over 100 calibration steps — tail divergence is unhandled.
3. Calibration = 50 Wikipedia articles, 100 decode steps. Deployment on
   code/math/multilingual/long-CoT is far outside calibration.
4. No multi-turn eval; longest analyzed generation 1K tokens; reasoning
   model tested only on LongBench v2 (never on long CoT).
5. Effectively batch-size-1 throughout; paged/batched serving untested.

**Reuse:** overlap-coefficient metric pipeline (temporal + spatial, cheap,
training-free — computable ONLINE per window); greedy star clustering as
initializer for online re-clustering; drift trigger machinery (windowed
median + baseline reset); CPU reservoir + async index-gather update path;
their memory-budget-ratio evaluation protocol (fairer than fixed token
budgets — reviewers will expect it).

## 4. ReasonAlloc (arXiv:2606.11164v1, preprint) — the dynamism

**What it does.** Two-level allocation for reasoning models: (a) offline
layer-wise "Reasoning Wave" preallocation (per-architecture, from ≤8 probe
prompts; non-monotonic wave refuting PyramidKV's decay); (b) online
head-wise budgets recomputed **every Δ=128 decode steps**: pool all heads'
scores in a layer (score = R-KV's 0.1·importance + 0.9·redundancy), take the
layer-budget-th largest as threshold τ, count each head's tokens ≥ τ, then
smooth/clip/floor (min 25% of average — their concession that starvation is
dangerous). Eviction is delegated to R-KV. No code release.

**Numbers that matter:**
- Strong at tight budgets: MATH-500 @128: 61.4 vs R-KV 51.8, SnapKV 33.8;
  AIME @256: 20.0% vs 10.4%/1.3%. Gains shrink to parity at 2048+.
- Overhead of online routing: ~0% (218.8 vs 217.4 tok/s).
- Throughput up to 5.5x vs FullKV — but entirely from the bounded cache;
  identical to uniform R-KV at same budget.

**Holes (= your evaluation plan):**
1. **No budget-trajectory evidence.** Zero figures/statistics of head
   budgets over time. Their central "real-time utility shifts" claim is
   validated only by end-to-end accuracy. Plot what they didn't.
2. **Irreversibility never mentioned** — no "recover", "offload", "CPU",
   "once evicted" anywhere. No shrink→grow ablation, no evicted-then-needed
   measurement, no recoverable-cache oracle. The budget floor is the only
   (implicit) acknowledgment.
3. **Math-only eval** (MATH-500, AIME 2024) despite profiling on
   code/LongBench. No long-input, no multi-turn.
4. **No memory-layout story:** per-head heterogeneous budgets vs paged
   attention never discussed; "freed" memory may not be reclaimable
   (tension #6).
5. Hyperparameters (α, Δ, μ, γ, ρ) essentially unablated. Related work
   misses Ada-KV/HeadKV/CAKE/DuoAttention — their "first" claim is soft.

**Reuse:** the utility score + layer-pooled KthLargest thresholding (adopt
verbatim for apples-to-apples comparison — then swap evict→offload); Δ=128
cadence with proven ~0% overhead; the offline-layer/online-head split
(spend dynamism only at head granularity). Note: with reversible demotion
the μ-floor can be lowered aggressively (starvation becomes recoverable) —
a measurable advantage.

---

## 5. Synthesis — design constraints for the system

**Signal options for promotion/demotion (pick 1–2, ablate):**

| Signal | Source | Cost | Status |
|---|---|---|---|
| Consecutive top-K overlap (online RCO) | FlexiCache | ~free at rerank | exists offline; online use = novel |
| Drift trigger (windowed-median overlap) | HeteroCache | fires 0.8% of steps | exists for content; role use = novel |
| Hidden-state MLP probe, k-step-ahead | 2602.11162 | tiny MLP/step | k=0 probe exists; k>0 = novel + gives prefetch lead time |
| Utility score (importance+redundancy) | ReasonAlloc/R-KV | computed anyway | exists for budgets; reuse verbatim |

**Where to demonstrate (regimes where ALL frozen-role systems are untested):**
1. Long-CoT reasoning models (R1-distill-8B; ≥4K-token generations) — also
   where ReasonAlloc lives, enabling direct comparison.
2. Multi-turn (SCBench subset) — measure role churn + promoted-delta spikes
   at turn boundaries.
3. Cross-genre profile transfer (profile on summarization, deploy on
   math/code) — attacks FlexiCache's 0.56-floor directly.

**Experiments the competitors' omissions hand you:**
- Head-budget/role trajectory plots over a generation (nobody has one).
- Shrink→grow event analysis: quality recovered by CPU reinstatement vs
  ReasonAlloc-style permanent loss vs HeteroCache anchor mistakes.
- Per-request "stable head went unstable" incident analysis (FlexiCache's
  own logging harness can produce it).
- Floor ablation: reversibility lets μ→0; eviction methods can't follow.
- Churn phase analysis (when do heads wake up — correlated with reasoning
  transitions?) + per-head predictability horizon. Novel even as pure
  measurement on top of 2602.11162.

**System path of least resistance:** fork FlexiCache (vLLM, Apache 2.0),
replace the frozen 25% classification with an online controller hung off
the existing rerank hook; keep their MinMax cache, block tables, transfer
kernels. HeteroCache's reservoir semantics + trigger debouncing inform the
controller; ReasonAlloc's scoring makes budget comparisons fair.

**Risks:**
- Concurrent work: 2602.11162 names this as future work (v2 May 2026);
  HeteroCache revised Apr 2026; field produced 3 near-misses in 8 months.
  Target: arXiv preprint within ~2 months, workshop deadline after.
- Small-model caveat: dynamic heads underperformed static on Llama-3.2-3B
  in 2602.11162's application. Validate the premise at your model scale
  early (cheap: their methodology on one model, one afternoon of GPU).
- Headroom risk: in standard LongBench regimes the ceiling over frozen
  classification is ~3 points. Regime choice is not optional — it IS the
  paper.
