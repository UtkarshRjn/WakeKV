"""Phase 1: candidate wake-up signals + lead-time evaluation (gate G1).

All functions operate offline on Phase-0 logs (numpy only). A "wake-up
event" for a head is a sustained transition of its continuous needle score
from inactive to active (with hysteresis, to avoid counting threshold
jitter as churn — the binary-artifact caveat in 2602.11162).

A signal passes G1 if it predicts wake-ups with enough LEAD TIME that a
CPU->GPU prefetch of the head's working set completes before the head is
needed (see ``transfer_steps_needed``).
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
    approximate (top-k only) — flagged as such in reports.
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
    """Decode steps a CPU->GPU promotion needs; the lead-time bar for G1.

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
