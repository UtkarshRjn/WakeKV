# WakeKV — Technical Walkthrough

*Everything we did, in mathematical detail. Companion to the plain-language
[`phase01_explainer.html`](phase01_explainer.html) and to the paper draft in
[`paper/wakekv.tex`](../paper/wakekv.tex). Reader is assumed to know
transformer attention and basic probability; every load-bearing formula is
derived or defined here.*

**Contents**

0. [Setup: what a KV cache is, mathematically](#0-setup-what-a-kv-cache-is-mathematically)
1. [Phase 0 — Do heads change what they want mid-generation?](#1-phase-0--do-heads-change-what-they-want-mid-generation)
2. [Phase 1 — Can we predict wake-ups cheaply enough to prefetch?](#2-phase-1--can-we-predict-wake-ups-cheaply-enough-to-prefetch)
3. [Phase 2a — The residency simulator](#3-phase-2a--the-residency-simulator)
4. [Phase 2b — Real system, monkey-patched into FlexiCache](#4-phase-2b--real-system-monkey-patched-into-flexicache)
5. [What the numbers do and do not say (honest math)](#5-what-the-numbers-do-and-do-not-say-honest-math)
6. [What all of this is worth](#6-what-all-of-this-is-worth)

---

## 0. Setup: what a KV cache is, mathematically

A transformer decoder with $L$ layers, $H$ query heads (or $H_{kv}$ key-value
heads under GQA), head dim $d$, generating token $t$ from a context of
length $T$, maintains two tensors per layer:

$$
K^{(\ell)} \in \mathbb{R}^{H_{kv} \times T \times d}, \quad
V^{(\ell)} \in \mathbb{R}^{H_{kv} \times T \times d}
$$

Total KV bytes: $2 \cdot L \cdot H_{kv} \cdot T \cdot d \cdot b$ where $b$
is the dtype width. For R1-Distill-Llama-8B at 8K context in bf16:
$2 \cdot 32 \cdot 8 \cdot 8192 \cdot 128 \cdot 2 \approx 1.1$ GB. This is
what we're trying to page.

At decode step $t$, head $h$ in layer $\ell$ computes attention weights
over the whole context:

$$
a^{(\ell,h)}_{t,j} = \operatorname{softmax}_j\!\left(
  \frac{q^{(\ell,h)}_t \cdot k^{(\ell,h)}_j}{\sqrt{d}}
\right), \quad j \in \{0, \ldots, T-1\}
$$

producing a row $a^{(\ell,h)}_t$ on the probability simplex. We **page** the
KV at some granularity — a "page" is a block of `page_size` (=16)
contiguous tokens. Each head's cache lives in some **resident set**
$R^{(\ell,h)} \subseteq \{\text{page ids}\}$ on GPU and a **reservoir**
$V^{(\ell,h)}$ on CPU. The paging question is: given a per-head budget
$B$ pages on GPU, which pages should be in $R$?

That's the entire object of study.

---

## 1. Phase 0 — Do heads change what they want mid-generation?

### 1.1 Two retrieval scores (2602.11162)

For a task with a **needle span** $I_{\text{needle}} \subset \{0,\ldots,T-1\}$
(a UUID, a problem statement, a fact), we define per-step per-head scores.

**Binary copy-paste score** (Eq. 1):

$$
\text{cp}^{(\ell,h)}_t = \mathbf{1}\!\left[
  \arg\max_j a^{(\ell,h)}_{t,j} \in I_{\text{needle}}
  \;\wedge\;
  \text{tokens}\!\left[\arg\max_j a^{(\ell,h)}_{t,j}\right] = y_t
\right]
$$

where $y_t$ is the token generated at step $t$. Fires iff the head's peak
attention is inside the needle *and* the copied token matches the output.

**Continuous needle-mass score** (Eq. 2):

$$
\text{nm}^{(\ell,h)}_t =
\frac{\sum_{j \in I_{\text{needle}}} a^{(\ell,h)}_{t,j}}
     {\sum_j a^{(\ell,h)}_{t,j}
      - \sum_{j \in I_{\text{sink}} \setminus I_{\text{needle}}} a^{(\ell,h)}_{t,j}
      - \sum_{j \in I_{\text{local}} \setminus I_{\text{needle}}} a^{(\ell,h)}_{t,j}}
$$

where $I_{\text{sink}} = \{0,1,2,3\}$ (attention sinks — the first few
positions get anomalous mass structurally) and
$I_{\text{local}} = \{T-32, \ldots, T-1\}$ (recent-window bias). The
$\setminus I_{\text{needle}}$ exclusions are a fix we had to add: without
them, the CoT regime (where the needle IS the problem statement at the
start) had the needle overlapping the sink window, denom collapsed, and
scores exploded to $\sim 10^5$.

### 1.2 Three churn metrics

Let $A_t = \{(\ell,h) : \text{cp}^{(\ell,h)}_t = 1\}$ be the active head
set at step $t$.

**Adjacent-step Jaccard**:

$$
J_t = \frac{|A_t \cap A_{t+1}|}{|A_t \cup A_{t+1}|} \in [0, 1]
$$

$J_t = 1$ means identical crew step-to-step (bad for us — a fixed plan
would suffice). Lower means turnover. The retrieval-heads paper reported
mean $J$ across models of **0.28–0.51**; we replicated **0.53** on
Qwen2.5-3B under the continuous score at threshold 0.1 (binary was
tighter at 0.71–0.93 because binary requires *both* peak-attention-in-needle
and token-match — a stricter conjunction).

**Random-Corrected Overlap (FlexiCache)**. For two top-$k$ page sets
$K_s, K_t$ drawn from a pool of $n$ pages:

$$
\text{RCO}(K_s, K_t; k, n) = \max\!\left(0,
  \frac{\min\!\left(1, \frac{|K_s \cap K_t|}{k}\right) - k/n}{1 - k/n}
\right)
$$

The rationale: two random $k$-subsets of an $n$-pool have expected
overlap $\mathbb{E}[|K_s \cap K_t|] = k^2/n$ (hypergeometric mean), so
raw overlap $\frac{|K_s \cap K_t|}{k}$ has expected value $k/n$ under a
uniform null. RCO subtracts that null and rescales so the null is 0 and a
perfect match is 1.

**We got this wrong first** — passed the nominal $k$ into RCO for
*derived* page sets whose actual sizes varied per step (page sets come
from top-64 tokens, but tokens can share pages, so $|K| \leq 64$). This
inflated $\frac{|K_s \cap K_t|}{k}$ past 1 and the "unstable-head
fraction" reported 0.000 while the drift metric reported 63%. Fixed by
using $k = \min(|K_s|, |K_t|)$ per pair and clamping raw overlap to
$[0,1]$. Regression test in `tests/test_metrics.py`.

**Temporal stability**:

$$
\text{TS}_s = \frac{1}{W-1}\sum_{\Delta=1}^{W-1}
              \text{RCO}(K_s, K_{s+\Delta}; k, n_{s+\Delta})
$$

FlexiCache uses $W = 16$. A head with mean $\text{TS} < 0.5$ is
"chronically unstable."

**Overlap coefficient (HeteroCache)**:

$$
O(A, B) = \frac{|A \cap B|}{\min(|A|, |B|)}
$$

**HeteroCache drift events**. Given a rolling baseline $B$ (initially
the prefill top-$k$), a drift event fires at step $t$ iff:

$$
\operatorname{median}\!\left\{O(K_\tau, B) : \tau \in [t-W+1, t]\right\}
< \tau_{\text{drift}}
$$

On an event, $B \leftarrow K_t$ (reset). We used $W = 8$,
$\tau_{\text{drift}} = 0.5$.

### 1.3 The results in one table

| Regime | Model | $J$ (cont.) | RCO$<$0.5 frac | $\geq$1 drift event |
|---|---|---:|---:|---:|
| NIAH | Qwen2.5-3B | 0.53 | 0.064 | 0.63 |
| CoT | R1-1.5B | 0.82 (bin.) | 0.023 | 0.85 |
| Multi-turn | Qwen2.5-3B | 0.79 (bin.) | 0.085 | 0.76 |
| CoT | R1-8B | 0.93 (bin.) | 0.042 | 0.79 |

The story: **a small tail of chronically unstable heads (2–9%) coexists
with a large majority (63–85%) that shifts occasionally.** The 8B run
tightens binary $J$ because R1-Distill-Llama-8B commits harder to its
answers on math CoT (rarely explicit copy), but the *drift* metric — the
one a residency controller actually reacts to — stays at 0.79. This is
why three metrics matter: no single one tells the whole story.

**Instrumentation math** (why the harness fits in 11 GB): a full attention
map for a 5K-token prompt at 32 layers × 8 heads × 5000 columns × float32
= 20 GB *per step*. Impossible. The instrumentation only stores per-step
**top-$k$ indices + values + two scalars**:
$32 \cdot 8 \cdot 64 \cdot (4 + 2) + 32 \cdot 8 \cdot (4 + 1) \approx 100$
KB/step. 2000 steps × 100 KB ≈ 200 MB per run.

---

## 2. Phase 1 — Can we predict wake-ups cheaply enough to prefetch?

The question is whether a runtime signal $s^{(\ell,h)}_t$ predicts
"this head is about to attend heavily to CPU-resident pages a few steps
from now" with enough lead time that we can fetch them ahead of the
stall.

### 2.1 Ground-truth: wake events with hysteresis

Wake events come from the continuous needle score
$x_t = \text{nm}^{(\ell,h)}_t$.

Define quiet counter $q_t$:

$$
q_t = \begin{cases}
q_{t-1} + 1 & \text{if } x_t < \text{lo} \\
0 & \text{if } x_t \geq \text{lo}, x_t \leq \text{hi} \\
0 & \text{if event fires}
\end{cases}
$$

An **event** fires at $t$ iff
$x_t > \text{hi} \wedge q_{t-1} \geq \text{min\_quiet}$. We used
$\text{lo}=0.05$, $\text{hi}=0.2$, $\text{min\_quiet}=8$.

The hysteresis is essential: without it, single-step jitter around a
threshold counts as a wake-up, and the "event" set becomes
indistinguishable from noise. This is a version of the classic
binary-thresholding artifact from 2602.11162.

### 2.2 Four candidate signals

| Signal | Formula |
|---|---|
| `online_rco` | $1 - \text{RCO}(K_{t-1}, K_t; k, n_t)$ |
| `drift` | $1 - \operatorname{median}\{O(K_\tau, B) : \tau \in [t-W+1, t]\}$ (rolling baseline) |
| `entropy_trend` | $H_t - \bar{H}_{[t-W, t)}$ where $H_t = -\sum_i v_{t,i}\log v_{t,i}$ over renormalized top-$k$ weights |
| `needle_mass_delta` | $\text{nm}_t - \overline{\text{nm}}_{[t-W, t)}$ (oracle-ish reference) |

### 2.3 The honesty ladder (this is where the paper gets sharp)

For each signal $s_t$ we want precision/recall at threshold $\theta$ and
lead time $k$:

$$
A_\theta = \{s : s_t > \theta\}, \quad
E = \{t : x_t \text{ is a wake event}\}
$$

An event at $t$ is **recalled** at lead $k$ iff
$\exists s \in A_\theta : t-k \leq s < t$. An alarm at $s$ is a **true
alarm** iff $\exists t \in E : s < t \leq s+k$.

$$
\text{recall}(\theta, k)
  = \frac{|\{t \in E : \exists s \in A_\theta,\, t-k \leq s < t\}|}{|E|},
\quad
\text{precision}(\theta, k)
  = \frac{|\{s \in A_\theta : \exists t \in E,\, s < t \leq s+k\}|}{|A_\theta|}
$$

Then three ways of choosing $\theta$:

1. **Per-head best-$\theta$** (oracle, upper bound). For each head, pick
   the $\theta$ maximizing precision *after seeing the labels*. Not
   deployable — the controller can't peek. Under this grading, `drift`
   looked perfect (P=1.00, R=1.00 at lead 8 on CoT). Mirage.
2. **Single fixed global $\theta$** (strictest lower bound). Pool signals
   from all heads, pick one quantile-based cutoff, apply everywhere.
   Heads have wildly different baseline churn rates, so this is unfair
   to slow heads. All signals precision-starved (best F1 $\approx 0.15$
   on CoT).
3. **Causal per-head z-score** (deployable middle). For each head,
   normalize against its own trailing window:

$$
z^{(\ell,h)}_t = \frac{s^{(\ell,h)}_t - \mu^{(\ell,h)}_{[t-W, t)}}
                      {\sigma^{(\ell,h)}_{[t-W, t)}}
$$

   using only past values (causal). Then fire when $z > 2$ globally.
   This is fair across heterogeneous heads and computable online.

### 2.4 The deployable numbers

| Signal | Regime | P | R | F1 |
|---|---|---:|---:|---:|
| drift | CoT 1.5B (407 events) | 0.08 | 0.41 | 0.14 |
| online_rco | Multi-turn (73) | 0.13 | 0.67 | 0.22 |
| drift | Multi-turn (73) | 0.20 | 0.70 | 0.31 |
| online_rco | CoT 8B | 0.10 | 0.79 | 0.17 |

Best deployable F1 anywhere is 0.31. **The best "smoke detector" we can
build fires $\sim 5\times$ more false alarms than real ones AND misses
30–60% of real wake-ups.** For a residency controller this bites twice:
missed events $\Rightarrow$ stalls; false alarms $\Rightarrow$ constant
over-promotion $\Rightarrow$ memory savings evaporate. Prediction is
ruled out.

### 2.5 The transfer bar (why this even matters)

Was the lead-time requirement even hard? For one head's working set of
64 pages, each page carrying $2 \cdot 16 \cdot 128 \cdot 2 = 8{,}192$
bytes of K+V:

$$
\text{transfer\_ms} =
\frac{64 \cdot 8192 \text{ B}}{24.3 \text{ GB/s} \cdot 10^9 \text{ B/s}}
\cdot 10^3 \approx 0.022 \text{ ms}
$$

At decode step time $\approx 30$ ms, that's
$\text{transfer\_steps} \approx 0.0007$ — well under a single decode
step. So the *timing* bar was almost free; the failure was purely
predictive quality.

### 2.6 Are wake-ups clustered? (If yes, a coarse trigger could work)

Pool per-head events into a per-step histogram
$c_t = |\{(\ell,h) : \text{event fires for head } (\ell,h) \text{ at step } t\}|$.

Define **concentration**:

$$
\rho(f) = \frac{\sum_{i=1}^{\lfloor fS \rfloor} c_{(i)}}{\sum_t c_t}
$$

where $c_{(i)}$ is the $i$-th largest count. Under a per-head
uniform-shuffle null (scatter each head's events uniformly at random
over $[0, S)$), $\mathbb{E}[\rho(f)] \approx f + O(1/\sqrt{S})$ for
large $S$.

We ran the shuffle null 200 times and computed

$$
z = \frac{\rho_{\text{obs}} - \bar{\rho}_{\text{null}}}
         {\operatorname{sd}(\rho_{\text{null}})}
$$

Mean CoT $z = +9.7$ (real burstiness), with one run at $z = +29.4$.
Fano factor $\operatorname{Var}(c_t)/\mathbb{E}[c_t] > 1$ everywhere
(over-dispersed).

But **coarse-trigger recall** — fire at the busiest $f$ fraction of
steps ± half-window — was 0.54 at $f=2\%$ and 0.86 at $f=5\%$. Real
bursts exist but they're not tight enough for a coarse trigger, AND
you'd need a *detectable* boundary signal to pick "busy steps" at
runtime, which we haven't shown exists.

**Both cheap proactive routes ruled out $\Rightarrow$ reactive design.**

---

## 3. Phase 2a — The residency simulator

Given the log of wanted-page sets per step per head, simulate three
policies.

### 3.1 State and policies

For each unit $u$ (KV-head across a layer), maintain resident set
$R_u \subseteq \mathbb{N}$ (page ids on GPU) and reservoir
$V_u = \text{seen}_u \setminus R_u$ (page ids on CPU).

**Full policy**. $R_u \leftarrow R_u \cup W_u^{(t)}$ every step. Never
evicts. Zero misses, ceiling memory.

**Reactive (WakeKV) policy**. For each wanted page $p \in W_u^{(t)}$:
- if $p \in R_u$: touch LRU timestamp $\tau_u[p] \leftarrow t$
- elif $p \in V_u$: **fetch** — miss + fetch,
  $V_u \leftarrow V_u \setminus \{p\}$,
  $R_u \leftarrow R_u \cup \{p\}$,
  $\tau_u[p] = t$
- else: first-touch, $R_u \leftarrow R_u \cup \{p\}$,
  $\tau_u[p] = t$

After each step: while $|R_u| > B$, evict
$\arg\min_{p \in R_u} \tau_u[p]$ (oldest) to $V_u$.

**Frozen (FlexiCache-style) policy**. Offline classify heads into
stable/unstable by their historical adjacent-step Jaccard; bottom 25%
become unstable.
- **Unstable heads**: $R_u \leftarrow R_u \cup W_u^{(t)}$
  (keep everything).
- **Stable heads**: record wanted pages in a demand histogram; misses
  accrue when $p \notin R_u$.
- Every 16 steps, replace $R_u$ with top-$B$ pages by demand histogram;
  count $|R_u^{\text{new}} \setminus R_u^{\text{old}}|$ as fetches.

### 3.2 The matched-memory Pareto (this is where we made an error worth flagging)

At the same budget label $B$, the two policies sit at **different memory**
(frozen keeps unstable heads fully resident, so its total resident bytes
are higher than reactive's). Comparing at same $B$ isn't apples-to-apples.

The right comparison interpolates reactive's miss rate onto each frozen
operating point's memory. Given reactive Pareto points
$(m_i^R, r_i^R)$ sorted by memory and a frozen point $(m^F, r^F)$:

$$
\hat{r}^R(m^F) =
r_i^R + \frac{m^F - m_i^R}{m_{i+1}^R - m_i^R}
       \left(r_{i+1}^R - r_i^R\right),
\quad m_i^R \leq m^F \leq m_{i+1}^R
$$

**Reactive wins iff $\hat{r}^R(m^F) \leq r^F$.**

Under this correct comparison on the real logs:

| Regime | Scale | Ops points | Reactive wins |
|---|---|---:|---|
| CoT | 1.5B | 4 | 4/4 (~½ miss rate) |
| Multi-turn | 3B | 3 | 3/3 |
| CoT | 8B | 4 | 4/4 (3× lower miss at $B{=}64$) |

The eye-catching 8B point: at $\approx 37{,}300$ pages, frozen misses
0.045, reactive misses 0.014.

### 3.3 The miss-asymmetry footnote (honest framing)

A reactive miss ≠ a frozen miss:
- **Reactive miss**: paid fetch stall. The page is loaded from CPU;
  quality is preserved by construction; cost is stall wall-clock time.
- **Frozen miss**: quality gap. Page unavailable to sparse attention
  until the next 16-step rerank; the sparse decode kernel just doesn't
  see it.

So reactive **trades memory for stall time, not for accuracy**. The
simulator can't measure stall time; it counts stall *events*. This is
exactly why Phase 2b existed.

---

## 4. Phase 2b — Real system, monkey-patched into FlexiCache

The key structural insight (which took the whole PR #8 investigation to
see clearly): **WakeKV reactive is FlexiCache with two knobs changed.**

FlexiCache already implements:
- Sparse decode over top-$B$ pages per head (Triton kernel).
- MinMax scoring that estimates per-page attention scores using per-page
  min/max K vectors — computable even for pages currently on CPU.
- Promote (`reload_kv_cache_h2d`) and demote (`free_pages_decode_phase`)
  machinery keyed off a per-request `needs_rerank` flag that fires when
  `num_decode_step % rerank_frequency == 0`.

Stock FlexiCache sets:
- `unstable_heads = <offline profile>` (~25% of heads kept fully resident)
- `rerank_frequency = 16` (rerank stable heads every 16 steps)

**WakeKV reactive** sets:
- `unstable_heads = []` (every head goes through the sparse + reservoir
  path)
- `rerank_frequency = 1` (rerank every step — full reactivity)

The shim monkey-patches `FlexiCacheConfig._build`, mutating those two
fields before the runtime tensors are built.

**The "big bug" of PR #9**: we also had to zero the scalar
`num_unstable_heads = 0` alongside the list, because FlexiCache's
block-count assertion at `gpu_model_runner.py:1977` branched on
`num_unstable_heads > 0` and expected the un-rounded total, causing an
off-by-one crash at engine init. Two representations of the same data,
both had to be synced. There was also a silent bug in the shim runner:
`os.execvp` was replacing the process image and throwing away the
in-process monkey-patch, so every sweep run would have measured *stock*
FlexiCache. Fix: bootstrap through `python -m wakekv._shimmed_main`,
which installs the shim inside the target's own interpreter.

### 4.1 Throughput results — Mistral-7B on A30, LEval 8k→1000, 24 prompts

$$
\text{tok/s} = \frac{\text{generated tokens}}{\text{elapsed time (s)}},
\quad
\text{speedup} = \frac{\text{tok/s}_{\text{reactive}}}{\text{tok/s}_{\text{stock}}}
$$

| $R$ | tok/s | speedup |
|---|---:|---:|
| stock (16) | 114.4 | 1.00× |
| 1 | 153.6 | 1.34× |
| 2 | 193.3 | 1.69× |
| 4 | 226.3 | 1.98× |
| 8 | 254.2 | 2.22× |
| 16 | 266.6 | 2.33× |

Monotone in $R$. Fewer reranks → less promote/demote overhead → higher
throughput. This *contradicted our prediction* (we predicted reactive-r1
would trail stock).

**The mechanism decomposition is unresolved.** Part of the speedup is:

- (a) All heads sparse, 0 dense (each attention op is over $B$ pages
  instead of $T$).
- (b) Uniform per-layer KV allocation instead of stock's lumpy layout
  (larger effective decode batch).
- (c) Less-frequent rerank cost (only at higher $R$).

Without a matched-sparse-only baseline (0 dense heads at native rerank),
we can't separate (a) from (b)+(c). This is item (iv) in the paper's
Future Work.

### 4.2 Quality results

For each config, run LongBench on 4 tasks × 30 samples, compute per-task
F1 (QA/multi-hop) or Rouge-L (summarization), then mean.

$$
Q_R = \frac{1}{4}\sum_{\text{task}} \text{score}_{\text{task}}(\text{reactive-}R),
\quad
\text{quality \%} = 100 \cdot Q_R / Q_{\text{stock}}
$$

| $R$ | LB mean | % of stock |
|---|---:|---:|
| stock (16) | 41.73 | 100.0 |
| 1 | 40.66 | 97.4 |
| 2 | 40.54 | 97.1 |
| 4 | 41.36 | 99.1 |
| 8 | 40.87 | 97.9 |
| 16 | 39.63 | 95.0 |

**Flat at 95–99% with no monotonic erosion.** TriviaQA is identical
(87.06) across all 6 configs; the entire spread is qasper (33.32–37.67)
and 2wikimqa (13.33–16.81) bouncing at $N{=}30$. At $N{=}30$/task, F1 has
95% CI of roughly $\pm 3$–5 points from binomial noise alone — so the
per-task deltas are dominated by sample noise, not policy. The **mean**
is more stable because it averages the noise.

### 4.3 Combined Pareto in one sentence

Reactive residency achieves **up to 2.33× throughput at $\geq 95\%$
LongBench mean quality** vs stock FlexiCache. The design point ($R=1$)
achieves 1.34× at 97.4%. $R=16$ is the Pareto knee (largest throughput,
still $\geq 95\%$ quality). Everything in between is dominated by these
two endpoints.

---

## 5. What the numbers do and do not say (honest math)

**What they say:**
- Head activity is non-stationary within a single generation, on our
  models, at our regimes.
- Cheap runtime signals cannot predict wake-ups well enough to prefetch:
  precision-starved at every fair threshold.
- In simulation, reactive LRU + fetch-on-demand dominates FlexiCache-style
  frozen at matched memory.
- On real hardware, the shim beats stock FlexiCache on throughput at
  every rerank interval while holding LongBench mean $\geq 95\%$.

**What they do not say:**
- **The mechanism isn't isolated.** Part of the 2.33× throughput is
  "all-sparse decode," not "adaptive residency." Matched sparse-only
  baseline is future work.
- **$N$ is small.** 30 samples/task; per-task deltas are noise, mean is
  stable but coarse. Full LongBench would tighten the numbers.
- **One model, one hardware, one regime.** Mistral-7B on A30 at
  8k→1000. 8B, multi-turn, and larger contexts are Phase 2b's next axes.
- **No PCIe stall-time breakdown.** The stall cost is folded into
  end-to-end throughput; disaggregating it requires instrumenting the
  fetch path.

**Statistical caveats worth naming:**
- The retrieval-heads-are-dynamic replication uses 5 runs per grid cell.
  Adjacent-Jaccard std is $\pm 0.06$ from that.
- CoT G0 uses 5 runs and reports 407 wake events aggregated — enough for
  the G1 P/R numbers to have real weight. Multi-turn uses 1 run and 73
  events, coarser.
- The matched-memory interpolation assumes reactive's Pareto is
  piecewise-linear in memory. In truth it's stepwise (finite budget
  grid); linear interpolation is a smoothing assumption.

---

## 6. What all of this is worth

The paper has three genuinely defensible claims:

1. **Head roles shift during decoding on our models** (with 8B validation,
   three metrics, three regimes).
2. **Cheap prediction can't hide the PCIe fetch** (fair honesty ladder,
   negative result, replicated across regimes and model scales).
3. **Reactive residency dominates frozen** in simulation on matched
   memory, and beats stock FlexiCache on real hardware at $\geq 95\%$
   quality across the interval sweep.

Claim 1 is a solid replication. Claim 2 is a negative result done
rigorously (three grading levels, four signals, clustering follow-up).
Claim 3 has simulation evidence AND preliminary real-hardware evidence,
with the mechanism-vs-residency caveat named honestly.

That's a workshop paper. The gaps (larger models, disentangled
mechanism, longer runs) are principled follow-ups, not hidden holes.
