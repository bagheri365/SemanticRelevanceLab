from semantic_relevance.experiments.learned_relevance_cv import build_cv_manifest
from semantic_relevance.learning.dataset import FEATURE_NAMES


def _row(query_id: str, document_id: str, relevance: int) -> dict[str, object]:
    row = {
        "query_id": query_id,
        "document_id": document_id,
        "relevance": relevance,
    }
    row.update({name: 1.0 for name in FEATURE_NAMES})
    return row


def test_manifest_holds_out_each_query_once() -> None:
    rows = []
    for query in range(10):
        rows.append(_row(str(query), f"d{query}-a", 0))
        rows.append(_row(str(query), f"d{query}-b", 2))

    result = build_cv_manifest(
        {"experiment": "relevance_features", "rows": rows},
        n_splits=5,
        seed=42,
    )

    held_out = [
        query_id
        for fold in result["folds"]
        for query_id in fold["test_query_ids"]
    ]
    assert len(held_out) == 10
    assert len(set(held_out)) == 10
    assert result["examples"] == 20
    assert result["label_counts"] == {"0": 10, "1": 0, "2": 10}
