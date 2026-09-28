import pytest

from semantic_relevance.experiments.graded_relevance_analysis import (
    analyze_graded_relevance,
)


def _row(doc: str, relevance: int, bm25: int, semantic: int) -> dict[str, object]:
    return {
        "query_id": "1",
        "query": "example",
        "document_id": doc,
        "title": doc,
        "relevance": relevance,
        "semantic_delta_ndcg@10": -0.2,
        "bm25_rank": bm25,
        "semantic_rank": semantic,
    }


def test_graded_analysis_counts_top_k_grades() -> None:
    rows = [
        _row("high", 2, 1, 3),
        _row("partial", 1, 2, 1),
        _row("none", 0, 3, 2),
    ]
    result = analyze_graded_relevance(
        {"experiment": "relevance_features", "rows": rows},
        top_k=2,
    )
    query = result["queries"][0]
    assert query["bm25_top_k_grades"] == {"0": 0, "1": 1, "2": 1}
    assert query["semantic_top_k_grades"] == {"0": 1, "1": 1, "2": 0}
    assert query["top_k_grade_delta"]["2"] == -1
    assert query["top_k_grade_delta"]["0"] == 1


def test_graded_analysis_tracks_promotions_by_grade() -> None:
    rows = [
        _row("high", 2, 5, 1),
        _row("partial", 1, 3, 2),
        _row("none", 0, 1, 5),
    ]
    result = analyze_graded_relevance(
        {"experiment": "relevance_features", "rows": rows}
    )
    promoted = result["queries"][0]["promoted_by_grade"]
    assert promoted == {"0": 0, "1": 1, "2": 1}


def test_graded_analysis_reports_mean_rank_change() -> None:
    rows = [
        _row("high-a", 2, 10, 2),
        _row("high-b", 2, 6, 4),
        _row("none", 0, 1, 11),
    ]
    result = analyze_graded_relevance(
        {"experiment": "relevance_features", "rows": rows}
    )
    high = result["queries"][0]["grade_rank_changes"]["2"]
    assert high["bm25_mean_rank"] == pytest.approx(8.0)
    assert high["semantic_mean_rank"] == pytest.approx(3.0)
    assert high["mean_rank_change"] == pytest.approx(5.0)


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="top_k"):
        analyze_graded_relevance({"rows": []}, top_k=0)
