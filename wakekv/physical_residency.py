"""GQA-aware physical KV residency accounting.

Independent query heads overcount storage under grouped-query attention.
For each physical KV head ``g``, aggregate demand across its query group:

    D_{t,g} = union_{h in G(g)} D_{t,h}

Physical KV bytes:

    M_KV = 2 * P * d_head * b * sum_{l,g} |S_{l,g}|

where P is tokens/page, b is bytes/scalar, and the leading 2 accounts for
keys and values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wakekv.reactive_controller import ReactiveController
from wakekv.residency import SimStats, Stream, simulate_frozen, simulate_full


def query_to_kv_map(n_q_heads: int, n_kv_heads: int) -> list[int]:
    """Map each query head index to its KV group index within a layer."""
    if n_kv_heads <= 0 or n_q_heads <= 0:
        raise ValueError("head counts must be positive")
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(f"n_q_heads={n_q_heads} not divisible by n_kv_heads={n_kv_heads}")
    group = n_q_heads // n_kv_heads
    return [h // group for h in range(n_q_heads)]


def aggregate_stream_to_physical(
    stream: Stream,
    n_layers: int,
    n_q_heads: int,
    n_kv_heads: int,
) -> tuple[Stream, int]:
    """Collapse per-query-head demand into per-KV-head unions.

    Input stream units are layer-major query heads (L * H_q).
    Output stream units are layer-major KV heads (L * H_kv).
    """
    q2kv = query_to_kv_map(n_q_heads, n_kv_heads)
    n_phys = n_layers * n_kv_heads
    out: Stream = []
    for row in stream:
        if len(row) != n_layers * n_q_heads:
            raise ValueError(
                f"stream row has {len(row)} units, expected {n_layers * n_q_heads}"
            )
        phys: list[set[int]] = [set() for _ in range(n_phys)]
        for layer in range(n_layers):
            for qh in range(n_q_heads):
                unit = layer * n_q_heads + qh
                g = layer * n_kv_heads + q2kv[qh]
                phys[g] |= set(row[unit])
        out.append([frozenset(s) for s in phys])
    return out, n_phys


def kv_bytes(
    n_resident_pages: int,
    page_tokens: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """M_KV for a snapshot: 2 * P * d * b * |S|."""
    return 2 * page_tokens * head_dim * dtype_bytes * n_resident_pages


@dataclass
class PhysicalSimStats:
    policy: str
    budget: int | None
    logical: SimStats
    n_physical_units: int
    page_tokens: int
    head_dim: int
    dtype_bytes: int = 2
    compulsory_admissions: int = 0
    reloads: int = 0
    peak_pages_before_trim: int = 0
    peak_reservoir_pages: int = 0
    mean_resident_pages: float = 0.0
    peak_resident_pages: int = 0

    def peak_kv_bytes(self) -> int:
        return kv_bytes(
            self.peak_resident_pages, self.page_tokens, self.head_dim, self.dtype_bytes
        )

    def mean_kv_bytes(self) -> float:
        return float(
            kv_bytes(
                int(round(self.mean_resident_pages)),
                self.page_tokens,
                self.head_dim,
                self.dtype_bytes,
            )
        )

    def as_row(self) -> dict:
        base = self.logical.as_row()
        base.update({
            "n_physical_units": self.n_physical_units,
            "compulsory_admissions": self.compulsory_admissions,
            "reloads": self.reloads,
            "peak_pages_before_trim": self.peak_pages_before_trim,
            "peak_reservoir_pages": self.peak_reservoir_pages,
            "peak_kv_bytes": self.peak_kv_bytes(),
            "mean_kv_bytes": self.mean_kv_bytes(),
            "page_tokens": self.page_tokens,
            "head_dim": self.head_dim,
            "dtype_bytes": self.dtype_bytes,
        })
        return base


def _account_reactive(
    stream: Stream,
    n_units: int,
    budget: int,
    demotion: str,
    page_tokens: int,
    head_dim: int,
    dtype_bytes: int,
) -> PhysicalSimStats:
    ctrl = ReactiveController(n_units, budget, demotion=demotion)
    compulsory = reloads = 0
    peak_before = 0
    peak_res = 0
    seen: list[set[int]] = [set() for _ in range(n_units)]
    for t, row in enumerate(stream):
        ctrl.begin_step(t)
        before = 0
        for u, wanted in enumerate(row):
            for p in wanted:
                if p not in seen[u]:
                    compulsory += 1
                    seen[u].add(p)
                elif p not in ctrl.resident_pages(u):
                    reloads += 1
            ctrl.on_unit_lru(u, set(wanted))
            before += len(ctrl.resident_pages(u)) + max(0, len(wanted) - budget)
        peak_before = max(peak_before, before)
        peak_res = max(peak_res, sum(len(ctrl.reservoir_pages(u)) for u in range(n_units)))
        ctrl.end_step()
    s = ctrl.stats
    logical = SimStats(
        policy="reactive" if demotion == "offload" else "evict",
        budget=budget,
        steps=s.steps,
        total_accesses=s.total_wanted,
        misses=s.total_misses,
        fetches=s.total_fetches,
        stall_steps=s.stall_steps,
        resident_page_steps=s.resident_page_steps,
        peak_resident_pages=s.peak_resident_pages,
    )
    return PhysicalSimStats(
        policy=logical.policy,
        budget=budget,
        logical=logical,
        n_physical_units=n_units,
        page_tokens=page_tokens,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
        compulsory_admissions=compulsory,
        reloads=reloads,
        peak_pages_before_trim=peak_before,
        peak_reservoir_pages=peak_res,
        mean_resident_pages=logical.mean_resident_pages,
        peak_resident_pages=logical.peak_resident_pages,
    )


def simulate_physical(
    stream: Stream,
    n_layers: int,
    n_q_heads: int,
    n_kv_heads: int,
    policy: str,
    budget: int | None,
    page_tokens: int = 16,
    head_dim: int = 128,
    dtype_bytes: int = 2,
    unstable_frac: float = 0.25,
    unstable: set[int] | None = None,
) -> PhysicalSimStats:
    """Run a policy on the GQA-aggregated physical stream."""
    phys, n_phys = aggregate_stream_to_physical(stream, n_layers, n_q_heads, n_kv_heads)
    if policy == "full":
        logical = simulate_full(phys, n_phys)
        return PhysicalSimStats(
            policy="full",
            budget=None,
            logical=logical,
            n_physical_units=n_phys,
            page_tokens=page_tokens,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
            mean_resident_pages=logical.mean_resident_pages,
            peak_resident_pages=logical.peak_resident_pages,
        )
    if policy == "reactive":
        return _account_reactive(
            phys, n_phys, int(budget), "offload", page_tokens, head_dim, dtype_bytes
        )
    if policy == "evict":
        return _account_reactive(
            phys, n_phys, int(budget), "evict", page_tokens, head_dim, dtype_bytes
        )
    if policy == "frozen":
        if unstable is not None:
            # Remap query-head unstable labels to KV groups: a KV head is
            # unstable if any of its query heads is.
            q2kv = query_to_kv_map(n_q_heads, n_kv_heads)
            phys_unstable: set[int] = set()
            for layer in range(n_layers):
                for qh in range(n_q_heads):
                    unit = layer * n_q_heads + qh
                    if unit in unstable:
                        phys_unstable.add(layer * n_kv_heads + q2kv[qh])
            unstable = phys_unstable
        logical = simulate_frozen(
            phys, n_phys, int(budget), unstable_frac=unstable_frac, unstable=unstable
        )
        return PhysicalSimStats(
            policy="frozen",
            budget=budget,
            logical=logical,
            n_physical_units=n_phys,
            page_tokens=page_tokens,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
            mean_resident_pages=logical.mean_resident_pages,
            peak_resident_pages=logical.peak_resident_pages,
        )
    raise ValueError(f"unknown policy {policy!r}")


def matched_memory_wins(
    rows: list[PhysicalSimStats],
    baseline: str = "frozen",
    challenger: str = "reactive",
) -> dict:
    """Compare challenger vs baseline at interpolated matched peak KV bytes."""
    base = [r for r in rows if r.policy == baseline]
    chal = [r for r in rows if r.policy == challenger]
    wins = ties = losses = 0
    details = []
    for b in base:
        # nearest challenger by peak bytes
        if not chal:
            break
        nearest = min(chal, key=lambda c: abs(c.peak_kv_bytes() - b.peak_kv_bytes()))
        if nearest.logical.miss_rate < b.logical.miss_rate:
            wins += 1
            outcome = "win"
        elif nearest.logical.miss_rate > b.logical.miss_rate:
            losses += 1
            outcome = "loss"
        else:
            ties += 1
            outcome = "tie"
        details.append({
            "baseline_budget": b.budget,
            "challenger_budget": nearest.budget,
            "baseline_bytes": b.peak_kv_bytes(),
            "challenger_bytes": nearest.peak_kv_bytes(),
            "baseline_miss": b.logical.miss_rate,
            "challenger_miss": nearest.logical.miss_rate,
            "outcome": outcome,
        })
    return {"wins": wins, "ties": ties, "losses": losses, "details": details}
