"""Evaluate pairwise learned relevance out-of-fold by query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from semantic_relevance.evaluation import evaluate
from semantic_relevance.learning.dataset import (
    FEATURE_NAMES,
    build_relevance_examples,
    split_examples_by_query,
)
from semantic_relevance.learning.folds import make_query_folds, validate_query_folds
from semantic_relevance.learning.pairwise import PairwiseLinearRanker


def run_pairwise_relevance(
    payload: dict[str, object],
    *,
    n_splits: int = 5,
    seed: int = 42,
    epochs: int = 30,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    max_pairs_per_query: int = 2000,
) -> dict[str, object]:
    """Train pairwise models on training queries and rank held-out queries."""
    rows = list(payload["rows"])
    examples = build_relevance_examples(rows)
    query_ids = sorted({example.query_id for example in examples})
    folds = make_query_folds(query_ids, n_splits=n_splits, seed=seed)
    validate_query_folds(folds, query_ids)

    rows_by_query: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_query.setdefault(str(row["query_id"]), []).append(row)

    qrels = {
        query_id: {
            str(row["document_id"]): int(row["relevance"])
            for row in rows_by_query[query_id]
        }
        for query_id in query_ids
    }
    bm25_run = {
        query_id: [
            str(row["document_id"])
            for row in sorted(
                rows_by_query[query_id],
                key=lambda row: int(row["bm25_rank"]),
            )
        ]
        for query_id in query_ids
    }
    semantic_run = {
        query_id: [
            str(row["document_id"])
            for row in sorted(
                rows_by_query[query_id],
                key=lambda row: int(row["semantic_rank"]),
            )
        ]
        for query_id in query_ids
    }

    pairwise_run: dict[str, list[str]] = {}
    pairwise_scores: dict[tuple[str, str], float] = {}
    fold_rows: list[dict[str, object]] = []
    weight_rows: list[list[float]] = []

    for fold in folds:
        train, test = split_examples_by_query(
            examples,
            train_query_ids=fold.train_query_ids,
            test_query_ids=fold.test_query_ids,
        )
        model = PairwiseLinearRanker(
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            max_pairs_per_query=max_pairs_per_query,
            seed=seed + fold.fold,
        ).fit(train)
        scores = model.predict([example.features for example in test])
        weight_rows.append(list(model.weights or []))

        scored_by_query: dict[str, list[tuple[str, float]]] = {}
        for example, score in zip(test, scores, strict=True):
            pairwise_scores[(example.query_id, example.document_id)] = float(score)
            scored_by_query.setdefault(example.query_id, []).append(
                (example.document_id, float(score))
            )
        for query_id, scored in scored_by_query.items():
            pairwise_run[query_id] = [
                document_id
                for document_id, _ in sorted(
                    scored,
                    key=lambda item: (-item[1], item[0]),
                )
            ]

        fold_qrels = {query_id: qrels[query_id] for query_id in fold.test_query_ids}
        fold_rows.append(
            {
                "fold": fold.fold,
                "train_queries": len(fold.train_query_ids),
                "test_queries": len(fold.test_query_ids),
                "test_query_ids": list(fold.test_query_ids),
                "training_pairs": model.training_stats.pairs,
                "ndcg@10": evaluate(
                    {qid: pairwise_run[qid] for qid in fold.test_query_ids},
                    fold_qrels,
                )["ndcg@10"],
            }
        )

    bm25 = evaluate(bm25_run, qrels)
    semantic = evaluate(semantic_run, qrels)
    pairwise = evaluate(pairwise_run, qrels)

    per_query: list[dict[str, object]] = []
    unique_scores: list[int] = []
    for query_id in query_ids:
        judgments = qrels[query_id]
        bm25_metric = evaluate({query_id: bm25_run[query_id]}, {query_id: judgments})
        semantic_metric = evaluate(
            {query_id: semantic_run[query_id]}, {query_id: judgments}
        )
        pairwise_metric = evaluate(
            {query_id: pairwise_run[query_id]}, {query_id: judgments}
        )
        scores = [
            pairwise_scores[(query_id, document_id)]
            for document_id in pairwise_run[query_id]
        ]
        unique = len(set(scores))
        unique_scores.append(unique)
        per_query.append(
            {
                "query_id": query_id,
                "query": rows_by_query[query_id][0]["query"],
                "bm25_ndcg@10": bm25_metric["ndcg@10"],
                "semantic_ndcg@10": semantic_metric["ndcg@10"],
                "pairwise_ndcg@10": pairwise_metric["ndcg@10"],
                "delta_vs_bm25": pairwise_metric["ndcg@10"] - bm25_metric["ndcg@10"],
                "delta_vs_semantic": (
                    pairwise_metric["ndcg@10"] - semantic_metric["ndcg@10"]
                ),
                "unique_pairwise_scores": unique,
            }
        )

    average_weights = [
        mean(weights[index] for weights in weight_rows)
        for index in range(len(FEATURE_NAMES))
    ]
    return {
        "experiment": "pairwise_relevance_oof",
        "features": list(FEATURE_NAMES),
        "model": {
            "type": "pairwise_linear_logistic",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "max_pairs_per_query": max_pairs_per_query,
        },
        "cross_validation": {
            "n_splits": n_splits,
            "seed": seed,
            "folds": fold_rows,
            "mean_fold_ndcg@10": mean(row["ndcg@10"] for row in fold_rows),
        },
        "bm25": bm25,
        "semantic": semantic,
        "pairwise": pairwise,
        "delta_vs_bm25": {key: pairwise[key] - bm25[key] for key in pairwise},
        "delta_vs_semantic": {
            key: pairwise[key] - semantic[key] for key in pairwise
        },
        "robustness": {
            "regressions_vs_bm25": sum(
                float(row["delta_vs_bm25"]) < 0 for row in per_query
            ),
            "regressions_vs_semantic": sum(
                float(row["delta_vs_semantic"]) < 0 for row in per_query
            ),
            "worst_vs_bm25": min(float(row["delta_vs_bm25"]) for row in per_query),
            "worst_vs_semantic": min(
                float(row["delta_vs_semantic"]) for row in per_query
            ),
        },
        "score_resolution": {
            "mean_unique_scores": mean(unique_scores),
            "min_unique_scores": min(unique_scores),
            "max_unique_scores": max(unique_scores),
        },
        "mean_feature_weights": dict(zip(FEATURE_NAMES, average_weights, strict=True)),
        "queries": sorted(
            per_query,
            key=lambda row: float(row["delta_vs_semantic"]),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/results/pairwise_relevance_oof.json",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-pairs-per-query", type=int, default=2000)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = run_pairwise_relevance(
        payload,
        n_splits=args.folds,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        max_pairs_per_query=args.max_pairs_per_query,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"saved {output}")
    print("BM25:    ", result["bm25"])
    print("semantic:", result["semantic"])
    print("pairwise:", result["pairwise"])
    print("delta vs BM25:", result["delta_vs_bm25"])
    print("delta vs semantic:", result["delta_vs_semantic"])
    print("robustness:", result["robustness"])
    print("score resolution:", result["score_resolution"])
    print("worst regressions:")
    for row in result["queries"][:5]:
        print(
            f'{row["query_id"]}: {float(row["delta_vs_semantic"]):+.4f} '
            f'- {row["query"]}'
        )


if __name__ == "__main__":
    main()
