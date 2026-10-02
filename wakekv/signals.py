"""Wake-up signals and lead-time evaluation.

Operates on logged attention (numpy only). A wake-up is a sustained
transition of a head's continuous needle score from inactive to active,
with hysteresis so threshold jitter is not counted as churn.

The paper's question is whether a cheap online signal predicts that
transition early enough to prefetch the head's KV from CPU before it is
needed (see ``transfer_steps_needed``). In the reported experiments, none
of them do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wakekv.metrics import overlap_coefficient, rco


# ---------------------------------------------------------------------------
# Wake-up event extraction (ground truth for signal evaluation)
# ---------------------------------------------------------------------------

def wake_events(
    score: np.ndarray, lo: float = 0.05, hi: float = 0.2, min_quiet: int = 8
) -> list[int]:
    """Steps where a head transitions quiet -> active with hysteresis.

    ``score``: [steps] continuous needle score for one head.
    An event fires at step t if score[t] > hi and the head spent the
    previous ``min_quiet`` steps below ``lo``.
    """
    events = []
    quiet = 0
    for t, s in enumerate(score):
        if s > hi and quiet >= min_quiet:
            events.append(t)
            quiet = 0
        elif s < lo:
            quiet += 1
        else:
            quiet = 0
    return events


# ---------------------------------------------------------------------------
# Candidate signals — each returns [steps] per head, higher = "waking up"
# ---------------------------------------------------------------------------

def signal_online_rco(topk_sets: list[set], pool_sizes: list[int], k: int | None) -> np.ndarray:
    """1 - RCO between consecutive top-k sets (FlexiCache's statistic, online)."""
    out = np.zeros(len(topk_sets))
    for t in range(1, len(topk_sets)):
        out[t] = 1.0 - rco(topk_sets[t - 1], topk_sets[t], k, pool_sizes[t])
    return out


def signal_drift(topk_sets: list[set], window: int = 8) -> np.ndarray:
    """1 - windowed-median overlap vs a rolling baseline (HeteroCache-style)."""
    out = np.zeros(len(topk_sets))
    base = topk_sets[0]
    hist: list[float] = []
    for t, s in enumerate(topk_sets):
        hist.append(overlap_coefficient(s, base))
        out[t] = 1.0 - float(np.median(hist[-window:]))
    return out


def signal_entropy_trend(topk_val: np.ndarray, window: int = 8) -> np.ndarray:
    """Rising attention entropy (renormalized over the stored top-k weights).

    ``topk_val``: [steps, k] attention weights for one head. Entropy is
    approximate, because the log stores only the top-k weights.
    """
    v = topk_val.astype(np.float64)
    v = v / np.clip(v.sum(-1, keepdims=True), 1e-9, None)
    ent = -(v * np.log(np.clip(v, 1e-12, None))).sum(-1)
    trend = np.zeros_like(ent)
    for t in range(len(ent)):
        left = ent[max(0, t - window) : t]
        trend[t] = ent[t] - left.mean() if len(left) else 0.0
    return trend


def signal_needle_mass_delta(score: np.ndarray, window: int = 4) -> np.ndarray:
    """Short-horizon rise in needle mass itself (oracle-ish upper bound —
    uses the same quantity that defines events, at shorter horizon)."""
    out = np.zeros_like(score)
    for t in range(len(score)):
        left = score[max(0, t - window) : t]
        out[t] = score[t] - left.mean() if len(left) else 0.0
    return out


# ---------------------------------------------------------------------------
# Token/wake-event correlation — DATA-DRIVEN, not a hand-picked word list.
#
# A fixed "discourse marker" vocabulary (wait, so, therefore, ...) would be
# an arbitrary, uncited guess about which words matter, and tuning it by eye
# against results would be circular. Instead: scan every token that actually
# occurred, test each one's empirical lift in predicting a wake burst via a
# two-proportion z-test, correct for testing thousands of vocab entries at
# once (Bonferroni), and evaluate the selected tokens ONLY on data disjoint
# from what selected them (see analyze_phase1.py's discovery/held-out
# split) — using the same data for both is the textbook multiple-comparisons
# trap: at a 30k-token vocabulary, plenty of tokens will look "significant"
# by chance alone at an uncorrected threshold.
# ---------------------------------------------------------------------------

def event_horizon_mask(events: list[int], length: int, lead: int) -> np.ndarray:
    """True at step t if a wake event occurs in (t, t+lead] — "a wake is
    about to happen." The binary outcome variable for the token/wake
    correlation test; matches the "true alarm" window definition used
    throughout ``evaluate_signal``/``evaluate_fixed_threshold``."""
    mask = np.zeros(length, dtype=bool)
    for e in events:
        lo = max(0, e - lead)
        mask[lo:e] = True
    return mask


@dataclass
class TokenWakeStat:
    token_id: int
    count: int
    hit_rate: float
    base_rate: float
    lift: float
    z: float


def token_wake_stats(
    token_ids: np.ndarray, wake_soon: np.ndarray, min_count: int = 5
) -> list[TokenWakeStat]:
    """Two-proportion z-test, per distinct token id that occurred, of
    P(wake soon | this token) vs. the run's base rate P(wake soon).

    ``token_ids``: [steps] generated token id at each step.
    ``wake_soon``: [steps] boolean, e.g. from ``event_horizon_mask``.
    Tokens occurring fewer than ``min_count`` times are skipped — a z-test
    on a handful of samples is noise, not evidence. Caller must still guard
    against multiple comparisons (``bonferroni_z_bar``) and must not reuse
    this SAME data to both select and evaluate a token (see module note).
    """
    token_ids = np.asarray(token_ids)
    wake_soon = np.asarray(wake_soon, dtype=bool)
    n0 = len(wake_soon)
    p0 = float(wake_soon.mean()) if n0 else 0.0
    out = []
    for tok in np.unique(token_ids):
        idx = token_ids == tok
        n1 = int(idx.sum())
        if n1 < min_count:
            continue
        p1 = float(wake_soon[idx].mean())
        pooled = (p1 * n1 + p0 * n0) / (n1 + n0)
        if 0.0 < pooled < 1.0:
            se = float(np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0)))
        else:
            se = 0.0
        z = (p1 - p0) / se if se > 1e-12 else 0.0
        if p0 > 1e-12:
            lift = p1 / p0
        else:
            lift = float("inf") if p1 > 0 else float("nan")
        out.append(TokenWakeStat(int(tok), n1, p1, p0, lift, float(z)))
    return out


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (quantile function). Peter Acklam's
    public-domain rational approximation (~1e-9 relative error) — avoids
    adding scipy as a dependency for one function."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return ((((( a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               ((((( b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = np.sqrt(-2 * np.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def bonferroni_z_bar(n_tests: int, alpha: float = 0.05) -> float:
    """One-sided z-threshold so that, after testing ``n_tests`` distinct
    tokens, the family-wise false-positive rate stays at ``alpha`` (Bonferroni:
    per-test alpha' = alpha / n_tests). Bigger vocabularies demand a higher
    bar — this is what makes the discovery step honest at 30k+ token scale."""
    n_tests = max(1, n_tests)
    return _norm_ppf(1.0 - alpha / n_tests)


def signal_discovered_tokens(token_ids: np.ndarray, marker_token_ids: set) -> np.ndarray:
    """Binary per-step signal: 1 if the generated token at this step is in
    ``marker_token_ids`` — a set chosen by ``token_wake_stats`` +
    ``bonferroni_z_bar`` on DIFFERENT (earlier) data than this array. This is
    the data-driven replacement for a fixed marker-word list: what counts as
    a "marker" is discovered per model/run rather than guessed in advance."""
    ids = np.asarray(token_ids)
    if not marker_token_ids:
        return np.zeros(len(ids), dtype=np.float64)
    return np.isin(ids, list(marker_token_ids)).astype(np.float64)


def ensemble_vote(zscored_signals: list[np.ndarray], threshold: float = 2.0) -> np.ndarray:
    """Per-step count of how many (already causal-z-scored) input signals
    exceed ``threshold`` at that step.

    Grade with ``evaluate_fixed_threshold`` at, e.g., threshold=1.5 to mean
    "fires when at least 2 of the signals agree" — page-overlap change,
    windowed-median drift, entropy trend, and needle-mass delta each fail
    alone to predict wake-ups,
    but they need not share the same false positives; requiring agreement
    trades recall for precision only if their errors are actually
    decorrelated, which this tests.
    """
    if not zscored_signals:
        raise ValueError("ensemble_vote requires at least one signal")
    stacked = np.stack(zscored_signals, axis=0)
    return (stacked > threshold).sum(axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# Lead-time evaluation
# ---------------------------------------------------------------------------

@dataclass
class LeadTimeResult:
    lead: int
    threshold: float
    precision: float
    recall: float
    n_events: int
    n_alarms: int


def evaluate_signal(
    signal: np.ndarray,
    events: list[int],
    leads: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    thresholds: np.ndarray | None = None,
) -> list[LeadTimeResult]:
    """Precision/recall of "signal crossed θ within [t-lead, t)" per event.

    An event at t is RECALLED at lead k if the signal exceeded θ at any
    step in [t-k, t). An alarm at step s is a TRUE alarm if any event
    occurs in (s, s+lead]. Sweeps thresholds over signal quantiles unless
    given explicitly.
    """
    if thresholds is None:
        qs = np.quantile(signal, [0.80, 0.90, 0.95, 0.99])
        thresholds = np.unique(qs)
    results = []
    for lead in leads:
        for th in thresholds:
            alarms = np.flatnonzero(signal > th)
            if len(events):
                recalled = sum(
                    1 for t in events if np.any((alarms >= t - lead) & (alarms < t))
                )
                recall = recalled / len(events)
            else:
                recall = float("nan")
            if len(alarms):
                true_alarm = sum(
                    1 for s in alarms if any(s < t <= s + lead for t in events)
                )
                precision = true_alarm / len(alarms)
            else:
                precision = float("nan")
            results.append(
                LeadTimeResult(lead, float(th), precision, recall, len(events), len(alarms))
            )
    return results


def evaluate_fixed_threshold(
    per_head: list[tuple[np.ndarray, list[int]]],
    threshold: float,
    leads: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
) -> dict[int, LeadTimeResult]:
    """Pooled precision/recall at ONE global threshold across all heads.

    The honest counterpart to picking the best per-head threshold: the
    runtime controller fires on a single fixed cutoff, so grade the signal
    the same way. ``per_head`` is a list of (signal_array, events) pairs;
    events and alarms are pooled across every head before computing P/R,
    so a signal can't look good by cherry-picking a cutoff per head.
    """
    out: dict[int, LeadTimeResult] = {}
    for lead in leads:
        tot_ev = rec = tot_al = true_al = 0
        for sig, events in per_head:
            alarms = np.flatnonzero(sig > threshold)
            tot_ev += len(events)
            rec += sum(
                1 for t in events if np.any((alarms >= t - lead) & (alarms < t))
            )
            tot_al += len(alarms)
            true_al += sum(
                1 for s in alarms if any(s < t <= s + lead for t in events)
            )
        precision = true_al / tot_al if tot_al else float("nan")
        recall = rec / tot_ev if tot_ev else float("nan")
        out[lead] = LeadTimeResult(lead, threshold, precision, recall, tot_ev, tot_al)
    return out


def causal_zscore(signal: np.ndarray, window: int = 32, warmup: int = 4) -> np.ndarray:
    """Per-head causal z-score: (x_t - trailing mean) / trailing std.

    Uses only past values (deployable online), and normalizes away each
    head's own baseline churn rate. A fixed threshold on the z-score is
    therefore fair across heterogeneous heads — the deployable middle
    ground between the per-head-best oracle (peeks at labels) and a single
    global raw threshold (too strict, since heads churn at different rates).
    """
    out = np.zeros(len(signal), dtype=np.float64)
    for t in range(len(signal)):
        past = signal[max(0, t - window) : t]
        if len(past) >= warmup:
            mu = float(past.mean())
            sd = float(past.std())
            dev = signal[t] - mu
            if sd > 1e-9:
                out[t] = dev / sd
            elif abs(dev) > 1e-9:
                # Dead-flat baseline then a jump: the clearest wake-up there
                # is. Don't let the zero-variance guard silence it — saturate
                # well above any usable z-threshold instead of returning 0.
                out[t] = 10.0 if dev > 0 else -10.0
    return out


def pooled_threshold(per_head: list[tuple[np.ndarray, list[int]]], quantile: float) -> float:
    """A single global threshold: the given quantile of all signal values
    pooled across heads. Used to grade a signal as the controller would."""
    allvals = np.concatenate([sig for sig, _ in per_head]) if per_head else np.zeros(1)
    return float(np.quantile(allvals, quantile))


def transfer_steps_needed(
    pages_to_fetch: int,
    page_kv_bytes: int,
    pcie_gbps: float = 21.0,
    decode_step_ms: float = 30.0,
) -> float:
    """Decode steps a CPU-to-GPU prefetch needs.

    A signal is early enough to prefetch only if its lead covers this.

    Defaults: 21 GB/s effective PCIe (HeteroCache's measured Gen4 number)
    and 30 ms/step (~33 tok/s single-request 8B decode). Override with
    measured values from your hardware.
    """
    bytes_total = pages_to_fetch * page_kv_bytes
    ms = bytes_total / (pcie_gbps * 1e9) * 1e3
    return ms / decode_step_ms


def page_kv_bytes(
    page_tokens: int = 16,
    n_kv_heads_per_fetch: int = 1,
    head_dim: int = 128,
    dtype_bytes: int = 2,
) -> int:
    """Bytes of one KV page for one KV head (K and V)."""
    return 2 * page_tokens * n_kv_heads_per_fetch * head_dim * dtype_bytes
