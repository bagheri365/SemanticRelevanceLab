from semantic_relevance.evaluation.analysis import (
    query_metrics,
    relevant_document_count,
)


def test_query_metrics_reports_query_level_scores() -> None:
    judgments = {"d1": 2, "d2": 1, "d3": 0}
    metrics = query_metrics(["d3", "d1", "d2"], judgments)

    assert 0.0 < metrics["ndcg@10"] < 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["recall@100"] == 1.0


def test_relevant_document_count_ignores_zero_relevance() -> None:
    assert relevant_document_count({"d1": 2, "d2": 1, "d3": 0}) == 2
