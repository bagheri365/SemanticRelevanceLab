"""Latency measurement helpers."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Latency distribution in milliseconds."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "count": float(self.count),
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }


def percentile(values: list[float], quantile: float) -> float:
    """Return a nearest-rank percentile from non-empty values."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def summarize_latencies(latencies_ms: list[float]) -> LatencySummary:
    """Summarize measured latencies."""
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty")
    return LatencySummary(
        count=len(latencies_ms),
        mean_ms=sum(latencies_ms) / len(latencies_ms),
        p50_ms=percentile(latencies_ms, 0.50),
        p95_ms=percentile(latencies_ms, 0.95),
        p99_ms=percentile(latencies_ms, 0.99),
    )


def benchmark(
    function: Callable[[], T],
    *,
    iterations: int = 10,
    warmup: int = 1,
) -> tuple[LatencySummary, T]:
    """Benchmark a zero-argument callable and return summary plus last result."""
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    for _ in range(warmup):
        function()

    latencies_ms: list[float] = []
    result: T | None = None
    for _ in range(iterations):
        start = time.perf_counter()
        result = function()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    assert result is not None
    return summarize_latencies(latencies_ms), result
