from semantic_relevance.experiments.learned_relevance_oof import (
    run_learned_relevance,
)
from semantic_relevance.learning.dataset import FEATURE_NAMES


def _row(
    query_id: str,
    document_id: str,
    relevance: int,
    bm25_rank: int,
    semantic_rank: int,
) -> dict[str, object]:
    row = {
        "query_id": query_id,
        "query": f"query {query_id}",
        "document_id": document_id,
        "relevance": relevance,
        "bm25_rank": bm25_rank,
        "semantic_rank": semantic_rank,
    }
    for index, name in enumerate(FEATURE_NAMES):
        row.setdefault(name, float(index))
    row["semantic_score"] = float(relevance)
    row["query_token_coverage"] = float(relevance)
    return row


def test_oof_experiment_predicts_every_query_without_leakage() -> None:
    rows = []
    for q in range(10):
        query_id = str(q)
        rows.extend(
            [
                _row(query_id, f"{q}-good", 2, 1, 1),
                _row(query_id, f"{q}-bad", 0, 2, 2),
            ]
        )

    result = run_learned_relevance(
        {"experiment": "relevance_features", "rows": rows},
        n_splits=5,
        max_depth=2,
        min_samples_leaf=1,
    )

    assert len(result["queries"]) == 10
    held_out = [
        query_id
        for fold in result["cross_validation"]["folds"]
        for query_id in fold["test_query_ids"]
    ]
    assert len(held_out) == len(set(held_out)) == 10
    assert result["learned"]["ndcg@10"] > 0.99
