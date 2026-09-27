import math

import pytest

from semantic_relevance.evaluation import (
    dcg_at_k,
    evaluate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_dcg_uses_graded_relevance() -> None:
    expected = 7.0 + 3.0 / math.log2(3) + 1.0 / math.log2(4)
    assert dcg_at_k([3, 2, 1], 3) == pytest.approx(expected)


def test_ndcg_is_one_for_ideal_ranking() -> None:
    judgments = {"d1": 3, "d2": 2, "d3": 1}
    assert ndcg_at_k(["d1", "d2", "d3"], judgments, 3) == pytest.approx(1.0)


def test_ndcg_penalizes_bad_ordering() -> None:
    judgments = {"d1": 3, "d2": 2, "d3": 1}
    assert ndcg_at_k(["d3", "d2", "d1"], judgments, 3) < 1.0


def test_ndcg_handles_unjudged_documents_as_zero() -> None:
    judgments = {"d1": 3}
    score = ndcg_at_k(["unknown", "d1"], judgments, 2)
    assert 0.0 < score < 1.0


def test_reciprocal_rank_uses_first_relevant_document() -> None:
    judgments = {"d1": 0, "d2": 2}
    assert reciprocal_rank(["d1", "d2"], judgments) == pytest.approx(0.5)


def test_recall_at_k() -> None:
    judgments = {"d1": 2, "d2": 1, "d3": 0}
    assert recall_at_k(["d3", "d1"], judgments, 2) == pytest.approx(0.5)


def test_metrics_return_zero_without_relevant_documents() -> None:
    judgments = {"d1": 0}
    assert ndcg_at_k(["d1"], judgments, 10) == 0.0
    assert recall_at_k(["d1"], judgments, 10) == 0.0
    assert reciprocal_rank(["d1"], judgments) == 0.0


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        dcg_at_k([1], 0)


def test_evaluate_averages_over_judged_queries() -> None:
    qrels = {"q1": {"d1": 2, "d2": 0}, "q2": {"d3": 1}}
    run = {"q1": ["d1", "d2"], "q2": ["missing", "d3"]}
    metrics = evaluate(run, qrels)
    assert 0.0 < metrics["ndcg@10"] <= 1.0
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["recall@10"] == pytest.approx(1.0)
    assert metrics["recall@100"] == pytest.approx(1.0)
