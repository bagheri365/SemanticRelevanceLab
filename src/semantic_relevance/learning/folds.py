"""Query-level cross-validation utilities for learned relevance experiments."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable


@dataclass(frozen=True, slots=True)
class QueryFold:
    """One query-disjoint train/test fold."""

    fold: int
    train_query_ids: tuple[str, ...]
    test_query_ids: tuple[str, ...]


def make_query_folds(
    query_ids: Iterable[str],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[QueryFold]:
    """Create deterministic, balanced folds with no query leakage."""
    unique = sorted(set(query_ids))
    if len(unique) < 2:
        raise ValueError("at least two unique queries are required")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(unique):
        raise ValueError("n_splits cannot exceed the number of unique queries")

    shuffled = unique[:]
    Random(seed).shuffle(shuffled)

    test_buckets: list[list[str]] = [[] for _ in range(n_splits)]
    for index, query_id in enumerate(shuffled):
        test_buckets[index % n_splits].append(query_id)

    folds: list[QueryFold] = []
    all_queries = set(unique)
    for index, bucket in enumerate(test_buckets):
        test_ids = tuple(sorted(bucket))
        train_ids = tuple(sorted(all_queries - set(test_ids)))
        folds.append(
            QueryFold(
                fold=index,
                train_query_ids=train_ids,
                test_query_ids=test_ids,
            )
        )
    return folds


def validate_query_folds(
    folds: Iterable[QueryFold],
    expected_query_ids: Iterable[str],
) -> None:
    """Raise if folds leak queries or fail to hold out every query once."""
    fold_list = list(folds)
    expected = set(expected_query_ids)
    held_out: list[str] = []

    for fold in fold_list:
        train = set(fold.train_query_ids)
        test = set(fold.test_query_ids)
        if train & test:
            raise ValueError(f"query leakage detected in fold {fold.fold}")
        if train | test != expected:
            raise ValueError(f"fold {fold.fold} does not cover all queries")
        held_out.extend(fold.test_query_ids)

    if set(held_out) != expected or len(held_out) != len(expected):
        raise ValueError("each query must appear in exactly one test fold")
