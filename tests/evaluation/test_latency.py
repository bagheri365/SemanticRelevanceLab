import pytest

from semantic_relevance.evaluation import (
    benchmark,
    percentile,
    summarize_latencies,
)


def test_percentile_uses_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.50) == 3.0
    assert percentile(values, 0.95) == 5.0


def test_percentile_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        percentile([1.0], 1.1)


def test_summarize_latencies() -> None:
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])
    assert summary.count == 4
    assert summary.mean_ms == pytest.approx(2.5)
    assert summary.p50_ms == 2.0
    assert summary.p95_ms == 4.0
    assert summary.p99_ms == 4.0


def test_benchmark_returns_last_result_and_summary() -> None:
    calls = 0

    def work() -> int:
        nonlocal calls
        calls += 1
        return calls

    summary, result = benchmark(work, iterations=3, warmup=2)
    assert calls == 5
    assert result == 5
    assert summary.count == 3
    assert summary.mean_ms >= 0.0


def test_benchmark_validates_counts() -> None:
    with pytest.raises(ValueError, match="iterations"):
        benchmark(lambda: None, iterations=0)
    with pytest.raises(ValueError, match="warmup"):
        benchmark(lambda: None, warmup=-1)
