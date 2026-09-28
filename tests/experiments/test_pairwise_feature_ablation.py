from semantic_relevance.experiments.pairwise_feature_ablation import (
    FEATURE_SETS,
    run_pairwise_feature_ablation,
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
        if name in ("bm25_rank", "semantic_rank"):
            row[name] = rank
        elif name == "semantic_score":
            row[name] = float(relevance * 3)
        elif name == "bm25_score":
            row[name] = float(relevance * 2)
        else:
            row[name] = float(relevance + index / 100)
    return row


def test_feature_sets_include_expected_ablations() -> None:
    assert FEATURE_SETS["semantic_only"] == ("semantic_score",)
    assert "query_token_coverage" not in FEATURE_SETS[
        "minus_query_token_coverage"
    ]
    assert FEATURE_SETS["all"] == FEATURE_NAMES


def test_pairwise_feature_ablation_reports_stability() -> None:
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

    result = run_pairwise_feature_ablation(
        {"rows": rows},
        n_splits=2,
        epochs=4,
        max_pairs_per_query=100,
    )

    assert len(result["ablations"]) == len(FEATURE_SETS)
    all_features = next(
        row for row in result["ablations"] if row["name"] == "all"
    )
    assert all_features["metrics"]["ndcg@10"] > 0.9
    assert set(all_features["weight_stability"]) == set(FEATURE_NAMES)
    assert all(
        len(stats["fold_weights"]) == 2
        for stats in all_features["weight_stability"].values()
    )
