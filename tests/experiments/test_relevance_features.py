from semantic_relevance.experiments.relevance_features import summarize_features


def test_summarize_features_separates_relevance_groups() -> None:
    base = {
        "title_token_coverage": 0.0,
        "rare_query_term_coverage": 0.0,
        "exact_query_phrase": 0,
        "bm25_score": 1.0,
        "semantic_score": 1.0,
        "bm25_rank": 2,
        "semantic_rank": 1,
        "rank_disagreement": 1,
        "absolute_rank_disagreement": 1,
    }
    rows = [
        {
            **base,
            "relevance": 2,
            "semantic_delta_ndcg@10": 0.2,
            "query_token_coverage": 1.0,
        },
        {
            **base,
            "relevance": 0,
            "semantic_delta_ndcg@10": -0.2,
            "query_token_coverage": 0.25,
        },
    ]
    summary = summarize_features(rows)
    assert summary["relevant"]["count"] == 1
    assert summary["non_relevant"]["count"] == 1
    assert (
        summary["relevant"]["feature_means"]["query_token_coverage"]
        > summary["non_relevant"]["feature_means"]["query_token_coverage"]
    )
