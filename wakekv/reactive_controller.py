"""Reactive residency controller.

Per-(layer, head) bookkeeping of a GPU-resident working set and a CPU
reservoir. WakeKV's rule: each head keeps an LRU set capped at budget B;
a page past that budget is copied to the reservoir and fetched back when
a later step wants it.

``on_unit_lru`` is what the simulator calls. The caller passes the pages
attended at this decode step. ``on_unit_explicit`` diffs a caller-chosen
resident set, such as FlexiCache's top-B by MinMax score. The
FlexiCache/vLLM deployment does not call either one. It clears
``unstable_heads`` and sets ``rerank_frequency``, and FlexiCache's own
promote and demote code moves the pages. ``on_unit_explicit`` supports
reversible offload only.

``rerank_interval`` leaves residency unchanged between reranks, matching
FlexiCache's periodic refresh. Interval 1 is WakeKV's design point.
Interval 16 is FlexiCache's native cadence.

``demotion="offload"`` (default) is WakeKV. ``demotion="evict"`` is the
destructive foil on the same LRU schedule: a demoted page is destroyed,
and every later want of it is a miss. Re-admitting it for free would make
the miss rate identical to offload and hide reversibility. The hardware
shim does not implement that foil. Comparing reversible and destructive
eviction on a real system, scored by stall time, is left open.

No GPU, no vLLM dependency. Pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepEvent:
    """What happened to one unit in one step."""

    fetches: set[int] = field(default_factory=set)
    evictions: set[int] = field(default_factory=set)
    n_misses: int = 0  # wanted-but-not-resident (LRU mode only)


@dataclass
class ControllerStats:
    steps: int = 0
    total_wanted: int = 0
    total_misses: int = 0
    total_fetches: int = 0
    total_evictions: int = 0
    resident_page_steps: int = 0  # sum over steps of total resident pages
    peak_resident_pages: int = 0
    rerank_steps: int = 0  # steps on which the rerank actually ran
    stall_steps: int = 0  # steps with ≥1 fetch (proxy for stall wall-clock)

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "total_wanted": self.total_wanted,
            "total_misses": self.total_misses,
            "miss_rate": (
                self.total_misses / self.total_wanted
                if self.total_wanted
                else float("nan")
            ),
            "total_fetches": self.total_fetches,
            "total_evictions": self.total_evictions,
            "mean_resident_pages": (
                self.resident_page_steps / self.steps if self.steps else 0.0
            ),
            "peak_resident_pages": self.peak_resident_pages,
            "rerank_steps": self.rerank_steps,
            "stall_steps": self.stall_steps,
        }


class ReactiveController:
    """Per-unit LRU bookkeeping of GPU-resident vs. CPU-reservoir pages.

    Usage:
        ctrl = ReactiveController(n_units=576, budget=64, rerank_interval=1)
        for step, wants_per_unit in enumerate(stream):
            ctrl.begin_step(step)
            for u, wanted in enumerate(wants_per_unit):
                ctrl.on_unit_lru(u, wanted)
            ctrl.end_step()
        print(ctrl.stats.as_dict())
    """

    def __init__(
        self,
        n_units: int,
        budget: int,
        rerank_interval: int = 1,
        demotion: str = "offload",
    ) -> None:
        if n_units < 0:
            raise ValueError(f"n_units must be >= 0, got {n_units}")
        if budget < 0:
            raise ValueError(f"budget must be >= 0, got {budget}")
        if rerank_interval < 1:
            raise ValueError(f"rerank_interval must be >= 1, got {rerank_interval}")
        if demotion not in ("offload", "evict"):
            raise ValueError(f"demotion must be 'offload' or 'evict', got {demotion!r}")
        self.n_units = n_units
        self.budget = budget
        self.rerank_interval = rerank_interval
        self.demotion = demotion
        # page -> last-touched step (dict order = insertion; last-touched via update)
        self._resident: list[dict[int, int]] = [{} for _ in range(n_units)]
        self._reservoir: list[set[int]] = [set() for _ in range(n_units)]
        # "evict" mode only: pages demoted here never return (on_unit_lru).
        self._destroyed: list[set[int]] = [set() for _ in range(n_units)]
        self.stats = ControllerStats()
        self._current_step: int | None = None
        self._step_had_fetch = False

    # ------------------------------------------------------------------ step
    def begin_step(self, step: int) -> None:
        """Call once per decode step before any on_unit_* calls."""
        if self._current_step is not None:
            raise RuntimeError(
                f"begin_step({step}) called without end_step for {self._current_step}"
            )
        self._current_step = step
        self._step_had_fetch = False

    def end_step(self) -> None:
        """Call once per decode step after all on_unit_* calls: tallies
        memory footprint and advances the step counter."""
        if self._current_step is None:
            raise RuntimeError("end_step called without a matching begin_step")
        total = sum(len(r) for r in self._resident)
        self.stats.resident_page_steps += total
        if total > self.stats.peak_resident_pages:
            self.stats.peak_resident_pages = total
        self.stats.steps += 1
        if self._step_had_fetch:
            self.stats.stall_steps += 1
        self._current_step = None

    # ---------------------------------------------------------- LRU entry
    def on_unit_lru(self, unit: int, wanted_pages: set[int]) -> StepEvent:
        """Simulator entry point. Caller passes the pages this unit
        actually wanted this decode step. The controller: (a) fetches any
        wanted page currently on CPU (counted as a miss + a fetch), or, in
        ``demotion="evict"`` mode, permanently misses any wanted page that
        was previously destroyed (b) touches its LRU timestamp for the
        pages it saw, and (c) at rerank cadence evicts oldest-first —
        to the reservoir (``"offload"``) or permanently (``"evict"``) —
        until len(resident) <= budget.
        """
        step = self._require_step()
        ev = StepEvent()
        resident = self._resident[unit]
        reservoir = self._reservoir[unit]
        destroyed = self._destroyed[unit]

        for p in wanted_pages:
            self.stats.total_wanted += 1
            if p in resident:
                resident[p] = step
            elif p in reservoir:
                # Miss: wanted page was on CPU (offload mode). Fetch it now.
                ev.fetches.add(p)
                ev.n_misses += 1
                reservoir.discard(p)
                resident[p] = step
            elif p in destroyed:
                # Evict mode: gone for good. Miss every time, no fetch, no
                # re-admission -- see wakekv.residency.simulate_evict, same
                # semantics, needed so the miss RATE actually diverges from
                # offload's (a naive "recompute instantly" re-admit would
                # make evict and offload's miss rate identical).
                ev.n_misses += 1
            else:
                # First-touch: brand-new page, treat as normal cache growth.
                resident[p] = step

        if step % self.rerank_interval == 0 and len(resident) > self.budget:
            self.stats.rerank_steps += 1
            excess = len(resident) - self.budget
            oldest = sorted(resident.keys(), key=resident.__getitem__)[:excess]
            for p in oldest:
                del resident[p]
                ev.evictions.add(p)
                if self.demotion == "evict":
                    destroyed.add(p)
                else:
                    reservoir.add(p)

        self._commit(ev)
        return ev

    # ------------------------------------------------------ explicit entry
    def on_unit_explicit(self, unit: int, desired_set: set[int]) -> StepEvent:
        """Shim entry point. Caller passes the exact set that should be
        resident after this step (e.g. top-B by MinMax score). At rerank
        cadence, the controller diffs desired vs. current and reports the
        implied fetches and evictions; on non-rerank steps residency is
        left untouched (matches FlexiCache's every-N-steps refresh).
        """
        step = self._require_step()
        # Off-cadence: FlexiCache-style, don't touch residency.
        if step % self.rerank_interval != 0:
            return StepEvent()

        if len(desired_set) > self.budget:
            raise ValueError(
                f"desired_set has {len(desired_set)} pages > budget {self.budget}"
            )

        ev = StepEvent()
        resident = self._resident[unit]
        reservoir = self._reservoir[unit]

        # Evictions: currently resident but no longer desired.
        for p in list(resident.keys()):
            if p not in desired_set:
                del resident[p]
                reservoir.add(p)
                ev.evictions.add(p)
        # Fetches: newly desired, not currently resident. First-touch pages
        # (never seen) are counted as fetches too — the caller decided this
        # page needs to be resident, so the KV must be brought in.
        for p in desired_set:
            if p not in resident:
                reservoir.discard(p)
                ev.fetches.add(p)
            resident[p] = step

        self.stats.rerank_steps += 1
        self._commit(ev)
        return ev

    # ------------------------------------------------------------ helpers
    def resident_pages(self, unit: int) -> set[int]:
        """Snapshot of a unit's currently-resident page set (for tests)."""
        return set(self._resident[unit].keys())

    def reservoir_pages(self, unit: int) -> set[int]:
        """Snapshot of a unit's CPU reservoir (for tests)."""
        return set(self._reservoir[unit])

    def destroyed_pages(self, unit: int) -> set[int]:
        """Snapshot of a unit's permanently-destroyed pages (evict mode
        only; always empty in offload mode)."""
        return set(self._destroyed[unit])

    def total_resident_pages(self) -> int:
        """Sum of resident pages across all units (right-now, not integrated)."""
        return sum(len(r) for r in self._resident)

    def _require_step(self) -> int:
        if self._current_step is None:
            raise RuntimeError("call begin_step() before on_unit_*()")
        return self._current_step

    def _commit(self, ev: StepEvent) -> None:
        self.stats.total_fetches += len(ev.fetches)
        self.stats.total_evictions += len(ev.evictions)
        self.stats.total_misses += ev.n_misses
        if ev.fetches:
            self._step_had_fetch = True
