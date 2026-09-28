import pytest

from semantic_relevance.learning.dataset import (
    FEATURE_NAMES,
    build_relevance_examples,
    split_examples_by_query,
)


def _row(query_id: str, relevance: int) -> dict[str, object]:
    row = {
        "query_id": query_id,
        "document_id": f"d{query_id}",
        "relevance": relevance,
    }
    row.update({name: index + 0.5 for index, name in enumerate(FEATURE_NAMES)})
    return row


def test_build_examples_preserves_features_and_grade() -> None:
    example = build_relevance_examples([_row("q1", 2)])[0]
    assert example.query_id == "q1"
    assert example.relevance == 2
    assert len(example.features) == len(FEATURE_NAMES)


def test_split_examples_is_query_disjoint() -> None:
    examples = build_relevance_examples([_row("q1", 2), _row("q2", 0)])
    train, test = split_examples_by_query(
        examples,
        train_query_ids=["q1"],
        test_query_ids=["q2"],
    )
    assert {row.query_id for row in train} == {"q1"}
    assert {row.query_id for row in test} == {"q2"}


def test_split_rejects_query_leakage() -> None:
    examples = build_relevance_examples([_row("q1", 1)])
    with pytest.raises(ValueError, match="leakage"):
        split_examples_by_query(
            examples,
            train_query_ids=["q1"],
            test_query_ids=["q1"],
        )


def test_invalid_relevance_grade_is_rejected() -> None:
    with pytest.raises(ValueError, match="grade"):
        build_relevance_examples([_row("q1", 3)])
