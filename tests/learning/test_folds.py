import pytest

from semantic_relevance.learning.folds import (
    QueryFold,
    make_query_folds,
    validate_query_folds,
)


def test_query_folds_are_disjoint_and_exhaustive() -> None:
    query_ids = [str(i) for i in range(50)]
    folds = make_query_folds(query_ids, n_splits=5, seed=42)
    validate_query_folds(folds, query_ids)
    assert [len(fold.test_query_ids) for fold in folds] == [10] * 5
    assert [len(fold.train_query_ids) for fold in folds] == [40] * 5


def test_query_folds_are_deterministic() -> None:
    ids = [str(i) for i in range(12)]
    assert make_query_folds(ids, seed=7) == make_query_folds(ids, seed=7)
    assert make_query_folds(ids, seed=7) != make_query_folds(ids, seed=8)


def test_validator_detects_leakage() -> None:
    folds = [QueryFold(0, ("1", "2"), ("2",))]
    with pytest.raises(ValueError, match="leakage"):
        validate_query_folds(folds, ["1", "2"])


def test_too_many_folds_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        make_query_folds(["1", "2"], n_splits=3)
