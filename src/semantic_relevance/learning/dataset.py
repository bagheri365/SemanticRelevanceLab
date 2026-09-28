"""Dataset construction for learned relevance experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

FEATURE_NAMES = (
    "bm25_score",
    "bm25_rank",
    "semantic_score",
    "semantic_rank",
    "query_token_coverage",
    "title_token_coverage",
    "rare_query_term_coverage",
    "rank_disagreement",
)


@dataclass(frozen=True, slots=True)
class RelevanceExample:
    """One candidate represented as numeric features plus a graded label."""

    query_id: str
    document_id: str
    features: tuple[float, ...]
    relevance: int


def build_relevance_examples(
    rows: Iterable[Mapping[str, object]],
) -> list[RelevanceExample]:
    """Convert feature-analysis rows into model-ready examples."""
    examples: list[RelevanceExample] = []
    for row in rows:
        relevance = int(row["relevance"])
        if relevance not in (0, 1, 2):
            raise ValueError(f"unsupported relevance grade: {relevance}")
        examples.append(
            RelevanceExample(
                query_id=str(row["query_id"]),
                document_id=str(row["document_id"]),
                features=tuple(float(row[name]) for name in FEATURE_NAMES),
                relevance=relevance,
            )
        )
    return examples


def split_examples_by_query(
    examples: Iterable[RelevanceExample],
    *,
    train_query_ids: Iterable[str],
    test_query_ids: Iterable[str],
) -> tuple[list[RelevanceExample], list[RelevanceExample]]:
    """Split examples using query IDs and reject overlapping partitions."""
    train_ids = set(train_query_ids)
    test_ids = set(test_query_ids)
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"query leakage detected: {sorted(overlap)}")

    train: list[RelevanceExample] = []
    test: list[RelevanceExample] = []
    for example in examples:
        if example.query_id in train_ids:
            train.append(example)
        elif example.query_id in test_ids:
            test.append(example)
    return train, test
