"""Shared adaptive per-head budgets under a global page cap.

Instead of B_h = B for every head,

    sum_h B_h(t) <= B_total

Allocation uses recent reuse, drift, reload rate, attention mass, and an
optional detector score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wakekv.reactive_controller import ReactiveController


@dataclass
class HeadSignal:
    reuse: float = 0.0          # fraction of wants already resident
    drift: float = 0.0          # 1 - Jaccard(prev, cur)
    reloads: float = 0.0        # fetches this window
    attention_mass: float = 0.0
    detector_score: float = 0.0


@dataclass
class AdaptiveAllocator:
    n_units: int
    total_budget: int
    min_per_head: int = 1
    ema: float = 0.8
    signals: list[HeadSignal] = field(init=False)
    budgets: list[int] = field(init=False)

    def __post_init__(self) -> None:
        if self.total_budget < self.n_units * self.min_per_head:
            raise ValueError("total_budget too small for min_per_head")
        self.signals = [HeadSignal() for _ in range(self.n_units)]
        equal = self.total_budget // self.n_units
        self.budgets = [max(self.min_per_head, equal) for _ in range(self.n_units)]
        self._normalize()

    def _normalize(self) -> None:
        self.budgets = [max(self.min_per_head, int(b)) for b in self.budgets]
        total = sum(self.budgets)
        if total == self.total_budget:
            return
        # fix remainder on highest-signal heads
        order = sorted(
            range(self.n_units),
            key=lambda u: self._priority(u),
            reverse=(total < self.total_budget),
        )
        i = 0
        while total != self.total_budget and i < self.n_units * 4:
            u = order[i % self.n_units]
            if total < self.total_budget:
                self.budgets[u] += 1
                total += 1
            elif self.budgets[u] > self.min_per_head:
                self.budgets[u] -= 1
                total -= 1
            i += 1

    def _priority(self, u: int) -> float:
        s = self.signals[u]
        return (
            0.25 * s.drift
            + 0.25 * s.reloads
            + 0.25 * s.attention_mass
            + 0.15 * s.detector_score
            + 0.10 * (1.0 - s.reuse)
        )

    def observe(
        self,
        unit: int,
        reuse: float,
        drift: float,
        reloads: float,
        attention_mass: float = 0.0,
        detector_score: float = 0.0,
    ) -> None:
        s = self.signals[unit]
        a = self.ema
        s.reuse = a * s.reuse + (1 - a) * reuse
        s.drift = a * s.drift + (1 - a) * drift
        s.reloads = a * s.reloads + (1 - a) * reloads
        s.attention_mass = a * s.attention_mass + (1 - a) * attention_mass
        s.detector_score = a * s.detector_score + (1 - a) * detector_score

    def reallocate(self) -> list[int]:
        scores = np.array([max(1e-6, self._priority(u)) for u in range(self.n_units)])
        weights = scores / scores.sum()
        raw = weights * self.total_budget
        self.budgets = [max(self.min_per_head, int(round(x))) for x in raw]
        self._normalize()
        return list(self.budgets)


class AdaptiveReactiveController:
    """ReactiveController with per-unit budgets from AdaptiveAllocator."""

    def __init__(self, n_units: int, total_budget: int, demotion: str = "offload") -> None:
        self.allocator = AdaptiveAllocator(n_units, total_budget)
        # underlying controller uses max budget; we trim per-unit manually
        self.ctrl = ReactiveController(n_units, total_budget, demotion=demotion)
        self.n_units = n_units
        self._prev: list[set[int]] = [set() for _ in range(n_units)]

    def begin_step(self, step: int) -> None:
        self.ctrl.begin_step(step)

    def on_unit(self, unit: int, wanted: set[int], attention_mass: float = 0.0) -> None:
        resident = self.ctrl.resident_pages(unit)
        reuse = len(wanted & resident) / len(wanted) if wanted else 1.0
        prev = self._prev[unit]
        drift = 1.0 - (len(wanted & prev) / len(wanted | prev) if (wanted or prev) else 1.0)
        before_fetches = self.ctrl.stats.total_fetches
        self.ctrl.budget = self.allocator.budgets[unit]
        self.ctrl.on_unit_lru(unit, wanted)
        # hard trim to per-head budget (controller may have used global)
        res = self.ctrl._resident[unit]
        b = self.allocator.budgets[unit]
        if len(res) > b:
            excess = len(res) - b
            oldest = sorted(res.keys(), key=res.__getitem__)[:excess]
            for p in oldest:
                del res[p]
                if self.ctrl.demotion == "evict":
                    self.ctrl._destroyed[unit].add(p)
                else:
                    self.ctrl._reservoir[unit].add(p)
        reloads = float(self.ctrl.stats.total_fetches - before_fetches)
        self.allocator.observe(unit, reuse, drift, reloads, attention_mass)
        self._prev[unit] = set(wanted)

    def end_step(self) -> None:
        self.ctrl.end_step()
        if self.ctrl.stats.steps % 8 == 0:
            self.allocator.reallocate()
