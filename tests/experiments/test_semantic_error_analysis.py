import pytest

from semantic_relevance.experiments.semantic_error_analysis import (
    build_error_analysis,
)


def _query(query_id, delta):
    return {
        "query_id": query_id,
        "query": f"query {query_id}",
        "delta_ndcg@10": delta,
        "delta_mrr": 0.0,
        "bm25": {"ndcg@10": 0.2},
        "semantic": {"ndcg@10": 0.2 + delta},
        "candidates": [
            {
                "document_id": "d1",
                "relevance": 2,
                "bm25_rank": 3,
                "semantic_rank": 1,
                "rank_movement": 2,
                "bm25_score": 5.0,
                "semantic_score": 8.0,
                "title": "Example",
                "snippet": "example text",
            }
        ],
    }


def test_build_error_analysis_selects_extremes() -> None:
    payload = {
        "model": "fake",
        "candidate_k": 100,
        "queries": [_query("bad", -0.3), _query("mid", 0.0), _query("good", 0.5)],
    }
    result = build_error_analysis(payload, query_count=1, result_count=1)
    assert result["regressions"][0]["query_id"] == "bad"
    assert result["improvements"][0]["query_id"] == "good"
    assert result["improvements"][0]["results"][0]["rank_movement"] == 2


def test_build_error_analysis_validates_limits() -> None:
    payload = {"model": "fake", "candidate_k": 100, "queries": []}
    with pytest.raises(ValueError, match="greater than zero"):
        build_error_analysis(payload, query_count=0)
