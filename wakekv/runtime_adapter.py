"""Bridge WakeKV residency to FlexiCache transfer primitives.

At runtime there is no oracle top-64: a ``DemandDetector`` scores
CPU-resident pages from the current query and key summaries, then the
``TransferManager`` moves candidates under the GPU pool cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from wakekv.demand_detector import detect_quest, detect_previous_demand
from wakekv.kv_pool import KvPool
from wakekv.reactive_controller import ReactiveController
from wakekv.transfer_manager import TransferManager


@dataclass
class RuntimeConfig:
    n_units: int
    budget: int
    pool_capacity: int
    page_bytes: int
    fetch_candidates: int = 16
    detector: str = "quest"  # quest | previous_demand
    demotion: str = "offload"


@dataclass
class WakeKVRuntime:
    cfg: RuntimeConfig
    controller: ReactiveController = field(init=False)
    pool: KvPool = field(init=False)
    transfers: TransferManager = field(init=False)
    kmin: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    kmax: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    last_wanted: dict[int, set[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.controller = ReactiveController(
            self.cfg.n_units, self.cfg.budget, demotion=self.cfg.demotion
        )
        self.pool = KvPool(self.cfg.pool_capacity)
        self.transfers = TransferManager(self.pool, self.cfg.page_bytes)

    def bind_flexicache(self, reload_h2d: Callable[..., Any], free_d2h: Callable[..., Any]) -> None:
        self.transfers.bind_flexicache(reload_h2d, free_d2h)

    def update_extrema(self, unit: int, page: int, kmin: np.ndarray, kmax: np.ndarray) -> None:
        self.kmin.setdefault(unit, {})[page] = kmin
        self.kmax.setdefault(unit, {})[page] = kmax

    def step(
        self,
        step: int,
        queries: dict[int, np.ndarray],
        oracle_wanted: dict[int, set[int]] | None = None,
    ) -> dict:
        """One decode step. ``oracle_wanted`` is for tests only; production
        leaves it None and uses the detector over CPU pages."""
        self.controller.begin_step(step)
        fetched_all: list[tuple[int, int]] = []
        for unit in range(self.cfg.n_units):
            cpu = self.controller.reservoir_pages(unit)
            gpu = self.controller.resident_pages(unit)
            if oracle_wanted is not None:
                wanted = oracle_wanted.get(unit, set())
            else:
                wanted = self._detect(unit, queries.get(unit), cpu)
            # fetch candidates that are on CPU
            need = [p for p in wanted if p in cpu][: self.cfg.fetch_candidates]
            moved = self.transfers.fetch_batch([(unit, p) for p in need])
            fetched_all.extend(moved)
            # controller bookkeeping uses the pages we intend to attend
            attend = gpu | {p for _, p in moved} | (wanted - cpu)
            self.controller.on_unit_lru(unit, attend if attend else wanted)
            self.last_wanted[unit] = wanted
        self.controller.end_step()
        # attention finished: unprotect fetched pages
        self.transfers.end_attention(fetched_all)
        return {
            "fetched": len(fetched_all),
            "resident": self.pool.occupancy,
            "miss_rate": self.controller.stats.as_dict()["miss_rate"],
            "transfer_stats": self.transfers.stats.__dict__.copy(),
        }

    def _detect(self, unit: int, q: np.ndarray | None, cpu: set[int]) -> set[int]:
        if not cpu:
            return set()
        if self.cfg.detector == "previous_demand" or q is None:
            return detect_previous_demand(
                self.last_wanted.get(unit, set()), cpu, self.cfg.fetch_candidates
            ).selected
        return detect_quest(
            q,
            self.kmin.get(unit, {}),
            self.kmax.get(unit, {}),
            cpu,
            self.cfg.fetch_candidates,
        ).selected
