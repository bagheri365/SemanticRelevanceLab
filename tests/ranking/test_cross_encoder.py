import pytest

from semantic_relevance.ranking import rerank


class FakeScorer:
    def score_pairs(self, pairs):
        return [
            10.0 if "relevant concept" in document else 1.0
            for _, document in pairs
        ]


class BrokenScorer:
    def score_pairs(self, pairs):
        return [1.0]


def test_rerank_orders_candidates_by_semantic_score() -> None:
    candidates = [
        ("d1", "lexical overlap"),
        ("d2", "contains relevant concept"),
    ]
    assert rerank("query", candidates, FakeScorer()) == [
        ("d2", 10.0),
        ("d1", 1.0),
    ]


def test_rerank_handles_empty_candidates() -> None:
    assert rerank("query", [], FakeScorer()) == []


def test_rerank_validates_score_count() -> None:
    with pytest.raises(ValueError, match="different number"):
        rerank(
            "query",
            [("d1", "one"), ("d2", "two")],
            BrokenScorer(),
        )
