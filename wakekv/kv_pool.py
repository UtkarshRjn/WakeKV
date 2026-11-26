"""GPU KV page pool with an explicit capacity cap."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KvPool:
    """Tracks which logical pages occupy GPU slots under a hard limit."""

    capacity_pages: int
    resident: dict[tuple[int, int], int] = field(default_factory=dict)  # (unit,page)->slot
    _free_slots: list[int] = field(default_factory=list)
    _next_slot: int = 0
    in_flight: set[tuple[int, int]] = field(default_factory=set)
    protected: set[tuple[int, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.capacity_pages < 0:
            raise ValueError("capacity_pages must be >= 0")

    def _alloc_slot(self) -> int | None:
        if self._free_slots:
            return self._free_slots.pop()
        if self._next_slot < self.capacity_pages:
            s = self._next_slot
            self._next_slot += 1
            return s
        return None

    def can_admit(self, n: int = 1) -> bool:
        free = len(self._free_slots) + max(0, self.capacity_pages - self._next_slot)
        return free >= n

    def admit(self, unit: int, page: int) -> int | None:
        key = (unit, page)
        if key in self.resident:
            return self.resident[key]
        slot = self._alloc_slot()
        if slot is None:
            return None
        self.resident[key] = slot
        return slot

    def release(self, unit: int, page: int) -> None:
        key = (unit, page)
        if key in self.protected or key in self.in_flight:
            raise RuntimeError(f"cannot release protected/in-flight page {key}")
        slot = self.resident.pop(key, None)
        if slot is not None:
            self._free_slots.append(slot)

    def protect(self, unit: int, page: int) -> None:
        self.protected.add((unit, page))

    def unprotect(self, unit: int, page: int) -> None:
        self.protected.discard((unit, page))

    def mark_in_flight(self, unit: int, page: int) -> None:
        self.in_flight.add((unit, page))

    def clear_in_flight(self, unit: int, page: int) -> None:
        self.in_flight.discard((unit, page))

    @property
    def occupancy(self) -> int:
        return len(self.resident)
