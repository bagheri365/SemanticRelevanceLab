import pytest

from semantic_relevance.experiments.promotion_analysis import (
    analyze_promotions,
    describe,
    movement_group,
)


def _row(
    *,
    query_id: str = "1",
    relevance: int,
    bm25_rank: int,
    semantic_rank: int,
    rare: float,
    title: float,
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query": "example query",
        "document_id": f"d-{relevance}-{bm25_rank}-{semantic_rank}",
        "title": "example",
        "relevance": relevance,
        "semantic_delta_ndcg@10": 0.1 if relevance else -0.1,
        "query_token_coverage": 0.5,
        "title_token_coverage": title,
        "rare_query_term_coverage": rare,
        "bm25_score": 2.0,
        "semantic_score": 3.0,
        "bm25_rank": bm25_rank,
        "semantic_rank": semantic_rank,
        "rank_disagreement": bm25_rank - semantic_rank,
        "absolute_rank_disagreement": abs(bm25_rank - semantic_rank),
    }


def test_movement_group() -> None:
    assert movement_group(
        _row(relevance=2, bm25_rank=10, semantic_rank=2, rare=1, title=1)
    ) == "promoted_relevant"
    assert movement_group(
        _row(relevance=0, bm25_rank=2, semantic_rank=10, rare=0, title=0)
    ) == "demoted_non_relevant"


def test_describe_reports_distribution() -> None:
    result = describe([1.0, 2.0, 3.0, 4.0])
    assert result["mean"] == pytest.approx(2.5)
    assert result["median"] == pytest.approx(2.5)
    assert result["p25"] == pytest.approx(1.75)
    assert result["p75"] == pytest.approx(3.25)


def test_analysis_compares_good_and_bad_promotions() -> None:
    rows = [
        _row(relevance=2, bm25_rank=20, semantic_rank=2, rare=0.9, title=0.8),
        _row(relevance=0, bm25_rank=30, semantic_rank=3, rare=0.2, title=0.1),
        _row(relevance=1, bm25_rank=2, semantic_rank=10, rare=0.8, title=0.7),
        _row(relevance=0, bm25_rank=3, semantic_rank=12, rare=0.1, title=0.0),
    ]
    result = analyze_promotions(
        {"experiment": "relevance_features", "rows": rows}
    )
    assert result["groups"]["promoted_relevant"]["count"] == 1
    assert result["groups"]["promoted_non_relevant"]["count"] == 1
    query = result["per_query"]["1"]
    assert query["promotion_precision"] == pytest.approx(0.5)
    assert query["rare_coverage_gap"] == pytest.approx(0.7)
    assert query["title_coverage_gap"] == pytest.approx(0.7)
