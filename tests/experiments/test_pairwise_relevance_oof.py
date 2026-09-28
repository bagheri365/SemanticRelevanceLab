from semantic_relevance.experiments.pairwise_relevance_oof import (
    run_pairwise_relevance,
)
from semantic_relevance.learning.dataset import FEATURE_NAMES


def _row(
    query_id: str,
    document_id: str,
    relevance: int,
    rank: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "query_id": query_id,
        "query": f"query {query_id}",
        "document_id": document_id,
        "relevance": relevance,
    }
    for index, name in enumerate(FEATURE_NAMES):
        if name == "bm25_rank" or name == "semantic_rank":
            row[name] = rank
        else:
            row[name] = float(relevance * 2 + index / 100)
    return row


def test_pairwise_oof_covers_queries_without_leakage() -> None:
    rows = []
    for query_index in range(4):
        query_id = f"q{query_index}"
        rows.extend(
            [
                _row(query_id, f"{query_id}-high", 2, 1),
                _row(query_id, f"{query_id}-mid", 1, 2),
                _row(query_id, f"{query_id}-low", 0, 3),
            ]
        )
    result = run_pairwise_relevance(
        {"rows": rows},
        n_splits=2,
        epochs=5,
        learning_rate=0.05,
        max_pairs_per_query=100,
    )
    assert result["pairwise"]["ndcg@10"] > 0.9
    assert len(result["queries"]) == 4
    assert result["score_resolution"]["mean_unique_scores"] >= 2
    held_out = [
        query_id
        for fold in result["cross_validation"]["folds"]
        for query_id in fold["test_query_ids"]
    ]
    assert sorted(held_out) == ["q0", "q1", "q2", "q3"]
