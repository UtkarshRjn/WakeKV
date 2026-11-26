"""Lightweight profiling counters for offload / pressure sweeps."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProfileSample:
    label: str
    elapsed_s: float
    output_tokens: int
    peak_gpu_bytes: int = 0
    peak_cpu_bytes: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    correct: bool | None = None

    @property
    def tok_per_s(self) -> float:
        return self.output_tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def tpot_ms(self) -> float:
        if self.output_tokens <= 0:
            return float("nan")
        return 1000.0 * self.elapsed_s / self.output_tokens


@dataclass
class Profiler:
    samples: list[ProfileSample] = field(default_factory=list)
    _t0: float | None = None
    _label: str = ""

    def begin(self, label: str) -> None:
        self._label = label
        self._t0 = time.perf_counter()

    def end(
        self,
        output_tokens: int,
        peak_gpu_bytes: int = 0,
        peak_cpu_bytes: int = 0,
        h2d_bytes: int = 0,
        d2h_bytes: int = 0,
        correct: bool | None = None,
    ) -> ProfileSample:
        if self._t0 is None:
            raise RuntimeError("Profiler.end without begin")
        sample = ProfileSample(
            label=self._label,
            elapsed_s=time.perf_counter() - self._t0,
            output_tokens=output_tokens,
            peak_gpu_bytes=peak_gpu_bytes,
            peak_cpu_bytes=peak_cpu_bytes,
            h2d_bytes=h2d_bytes,
            d2h_bytes=d2h_bytes,
            correct=correct,
        )
        self.samples.append(sample)
        self._t0 = None
        return sample

    def summary(self) -> dict:
        if not self.samples:
            return {}
        tpots = [s.tpot_ms for s in self.samples if s.output_tokens > 0]
        tpots_sorted = sorted(tpots)

        def pct(p: float) -> float:
            if not tpots_sorted:
                return float("nan")
            idx = min(len(tpots_sorted) - 1, int(round(p * (len(tpots_sorted) - 1))))
            return tpots_sorted[idx]

        return {
            "n": len(self.samples),
            "mean_tok_per_s": sum(s.tok_per_s for s in self.samples) / len(self.samples),
            "tpot_p50_ms": pct(0.5),
            "tpot_p95_ms": pct(0.95),
            "peak_gpu_bytes": max(s.peak_gpu_bytes for s in self.samples),
            "peak_cpu_bytes": max(s.peak_cpu_bytes for s in self.samples),
            "total_h2d_bytes": sum(s.h2d_bytes for s in self.samples),
            "total_d2h_bytes": sum(s.d2h_bytes for s in self.samples),
            "accuracy": (
                sum(1 for s in self.samples if s.correct) / len(self.samples)
                if any(s.correct is not None for s in self.samples)
                else None
            ),
        }
