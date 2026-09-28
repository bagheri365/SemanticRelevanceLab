import pytest

from semantic_relevance.learning.dataset import RelevanceExample
from semantic_relevance.learning.pairwise import PairwiseLinearRanker


def _example(query: str, document: str, value: float, relevance: int) -> RelevanceExample:
    return RelevanceExample(query, document, (value, value * 0.5), relevance)


def test_pairwise_ranker_learns_preference_direction() -> None:
    examples = [
        _example("q1", "low1", -2.0, 0),
        _example("q1", "high1", 2.0, 2),
        _example("q2", "low2", -1.0, 0),
        _example("q2", "high2", 1.0, 2),
    ]
    model = PairwiseLinearRanker(
        epochs=20,
        learning_rate=0.1,
        max_pairs_per_query=100,
    ).fit(examples)
    assert model.score((2.0, 1.0)) > model.score((-2.0, -1.0))
    assert model.training_stats is not None
    assert model.training_stats.queries == 2


def test_pairwise_ranker_requires_differing_labels() -> None:
    examples = [
        _example("q1", "a", 0.0, 1),
        _example("q1", "b", 1.0, 1),
    ]
    with pytest.raises(ValueError, match="no differing relevance pairs"):
        PairwiseLinearRanker().fit(examples)


def test_pairwise_ranker_validates_parameters() -> None:
    with pytest.raises(ValueError, match="epochs"):
        PairwiseLinearRanker(epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        PairwiseLinearRanker(learning_rate=0.0)
