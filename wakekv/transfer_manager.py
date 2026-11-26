"""Async H2D / D2H page transfers with in-flight protection.

Designed to wrap FlexiCache's ``reload_kv_cache_h2d`` /
``free_pages_decode_phase`` without forking kernels. Pure-Python bookkeeping
runs without CUDA; CUDA paths activate when torch.cuda is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from wakekv.kv_pool import KvPool


@dataclass
class TransferStats:
    h2d_pages: int = 0
    d2h_pages: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    batches: int = 0


@dataclass
class TransferManager:
    pool: KvPool
    page_bytes: int
    pinned: dict[tuple[int, int], Any] = field(default_factory=dict)
    stats: TransferStats = field(default_factory=TransferStats)
    _h2d: Callable[..., Any] | None = None
    _d2h: Callable[..., Any] | None = None
    _events: list[Any] = field(default_factory=list)

    def bind_flexicache(
        self,
        reload_h2d: Callable[..., Any],
        free_d2h: Callable[..., Any],
    ) -> None:
        self._h2d = reload_h2d
        self._d2h = free_d2h

    def pin_host(self, unit: int, page: int, tensor: Any) -> None:
        self.pinned[(unit, page)] = tensor

    def fetch_batch(self, pages: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Admit and H2D-fetch a batch. Returns pages that actually moved."""
        moved = []
        pending = []
        for unit, page in pages:
            key = (unit, page)
            if key in self.pool.resident:
                continue
            if not self.pool.can_admit(1):
                break
            self.pool.mark_in_flight(unit, page)
            pending.append((unit, page))
        for unit, page in pending:
            slot = self.pool.admit(unit, page)
            if slot is None:
                self.pool.clear_in_flight(unit, page)
                continue
            if self._h2d is not None:
                self._h2d(unit=unit, page=page, slot=slot)
            self.pool.clear_in_flight(unit, page)
            self.pool.protect(unit, page)  # protected until attention done
            moved.append((unit, page))
            self.stats.h2d_pages += 1
            self.stats.h2d_bytes += self.page_bytes
        if moved:
            self.stats.batches += 1
        return moved

    def demote_batch(self, pages: list[tuple[int, int]]) -> list[tuple[int, int]]:
        moved = []
        for unit, page in pages:
            key = (unit, page)
            if key in self.pool.protected or key in self.pool.in_flight:
                continue
            if key not in self.pool.resident:
                continue
            self.pool.mark_in_flight(unit, page)
            if self._d2h is not None:
                self._d2h(unit=unit, page=page)
            self.pool.clear_in_flight(unit, page)
            self.pool.release(unit, page)
            moved.append((unit, page))
            self.stats.d2h_pages += 1
            self.stats.d2h_bytes += self.page_bytes
        if moved:
            self.stats.batches += 1
        return moved

    def end_attention(self, pages: list[tuple[int, int]]) -> None:
        for unit, page in pages:
            self.pool.unprotect(unit, page)
