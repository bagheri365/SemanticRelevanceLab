import pytest

from semantic_relevance.experiments.learned_ranking_diagnostics import (
    build_ranking_diagnostics,
)


def test_diagnostics_measure_score_ties_and_rank_movement() -> None:
    payload = {
        "queries": [
            {"query_id": "q1", "query": "test", "delta_vs_semantic": -0.2}
        ],
        "candidates": [
            {
                "query_id": "q1", "document_id": "d1", "relevance": 2,
                "semantic_rank": 1, "learned_rank": 3, "learned_score": 0.5,
            },
            {
                "query_id": "q1", "document_id": "d2", "relevance": 0,
                "semantic_rank": 2, "learned_rank": 1, "learned_score": 1.0,
            },
            {
                "query_id": "q1", "document_id": "d3", "relevance": 1,
                "semantic_rank": 3, "learned_rank": 2, "learned_score": 1.0,
            },
        ],
    }
    result = build_ranking_diagnostics(payload, top_n=2)
    row = result["queries"][0]
    assert row["unique_learned_scores"] == 2
    assert row["largest_tie_group"] == 2
    assert row["worst_relevant_demotions"][0]["document_id"] == "d1"
    assert result["summary"]["regressions_vs_semantic"] == 1


def test_diagnostics_reject_nonpositive_top_n() -> None:
    with pytest.raises(ValueError, match="top_n"):
        build_ranking_diagnostics({"queries": [], "candidates": []}, top_n=0)
