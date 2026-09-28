from semantic_relevance.experiments.learned_relevance_failure_analysis import (
    analyze_learned_failures,
)


def test_failure_analysis_selects_semantic_regressions() -> None:
    learned = {
        "candidate_k": 2,
        "queries": [
            {
                "query_id": "q1",
                "query": "one",
                "delta_vs_semantic": -0.2,
                "delta_vs_bm25": 0.1,
            },
            {
                "query_id": "q2",
                "query": "two",
                "delta_vs_semantic": 0.1,
                "delta_vs_bm25": 0.2,
            },
        ],
    }
    features = {
        "rows": [
            {
                "query_id": "q1",
                "document_id": "d1",
                "relevance": 2,
                "semantic_rank": 1,
                "query_token_coverage": 1.0,
                "title_token_coverage": 0.5,
                "rare_query_term_coverage": 1.0,
                "rank_disagreement": 2.0,
            },
            {
                "query_id": "q1",
                "document_id": "d2",
                "relevance": 0,
                "semantic_rank": 2,
                "query_token_coverage": 0.2,
                "title_token_coverage": 0.0,
                "rare_query_term_coverage": 0.0,
                "rank_disagreement": 4.0,
            },
        ],
    }
    result = analyze_learned_failures(learned, features, top_n=2)
    assert result["summary"]["regressions_vs_semantic"] == 1
    assert result["regressions"][0]["query_id"] == "q1"
    assert result["evaluation_scope"]["candidate_universe"] == 2


def test_failure_analysis_rejects_nonpositive_top_n() -> None:
    try:
        analyze_learned_failures({"queries": []}, {"rows": []}, top_n=0)
    except ValueError as exc:
        assert "top_n" in str(exc)
    else:
        raise AssertionError("expected ValueError")
