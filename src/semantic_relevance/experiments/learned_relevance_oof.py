"""Evaluate a small learned relevance model out-of-fold by query."""

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
from semantic_relevance.learning.tree import RegressionTree


def run_learned_relevance(
    payload: dict[str, object],
    *,
    n_splits: int = 5,
    seed: int = 42,
    max_depth: int = 4,
    min_samples_leaf: int = 20,
) -> dict[str, object]:
    """Train only on training queries and rank every query out-of-fold."""
    rows = list(payload["rows"])
    examples = build_relevance_examples(rows)
    query_ids = sorted({example.query_id for example in examples})
    folds = make_query_folds(query_ids, n_splits=n_splits, seed=seed)
    validate_query_folds(folds, query_ids)

    rows_by_key = {
        (str(row["query_id"]), str(row["document_id"])): row for row in rows
    }
    qrels: dict[str, dict[str, int]] = {}
    bm25_run: dict[str, list[str]] = {}
    semantic_run: dict[str, list[str]] = {}
    learned_run: dict[str, list[str]] = {}
    fold_rows: list[dict[str, object]] = []

    for query_id in query_ids:
        query_rows = [row for row in rows if str(row["query_id"]) == query_id]
        qrels[query_id] = {
            str(row["document_id"]): int(row["relevance"]) for row in query_rows
        }
        bm25_run[query_id] = [
            str(row["document_id"])
            for row in sorted(query_rows, key=lambda row: int(row["bm25_rank"]))
        ]
        semantic_run[query_id] = [
            str(row["document_id"])
            for row in sorted(query_rows, key=lambda row: int(row["semantic_rank"]))
        ]

    for fold in folds:
        train, test = split_examples_by_query(
            examples,
            train_query_ids=fold.train_query_ids,
            test_query_ids=fold.test_query_ids,
        )
        model = RegressionTree(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        ).fit(
            [example.features for example in train],
            [example.relevance for example in train],
        )
        scores = model.predict([example.features for example in test])

        scored_by_query: dict[str, list[tuple[str, float]]] = {}
        for example, score in zip(test, scores, strict=True):
            scored_by_query.setdefault(example.query_id, []).append(
                (example.document_id, score)
            )
        for query_id, scored in scored_by_query.items():
            learned_run[query_id] = [
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
                "ndcg@10": evaluate(
                    {qid: learned_run[qid] for qid in fold.test_query_ids},
                    fold_qrels,
                )["ndcg@10"],
            }
        )

    if set(learned_run) != set(query_ids):
        raise ValueError("out-of-fold predictions do not cover every query")

    bm25 = evaluate(bm25_run, qrels)
    semantic = evaluate(semantic_run, qrels)
    learned = evaluate(learned_run, qrels)

    per_query = []
    for query_id in query_ids:
        judgments = qrels[query_id]
        bm25_metric = evaluate({query_id: bm25_run[query_id]}, {query_id: judgments})
        semantic_metric = evaluate(
            {query_id: semantic_run[query_id]}, {query_id: judgments}
        )
        learned_metric = evaluate(
            {query_id: learned_run[query_id]}, {query_id: judgments}
        )
        row = rows_by_key[(query_id, learned_run[query_id][0])]
        per_query.append(
            {
                "query_id": query_id,
                "query": row["query"],
                "bm25_ndcg@10": bm25_metric["ndcg@10"],
                "semantic_ndcg@10": semantic_metric["ndcg@10"],
                "learned_ndcg@10": learned_metric["ndcg@10"],
                "delta_vs_bm25": (
                    learned_metric["ndcg@10"] - bm25_metric["ndcg@10"]
                ),
                "delta_vs_semantic": (
                    learned_metric["ndcg@10"] - semantic_metric["ndcg@10"]
                ),
            }
        )

    return {
        "experiment": "learned_relevance_oof",
        "features": list(FEATURE_NAMES),
        "model": {
            "type": "regression_tree",
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        },
        "cross_validation": {
            "n_splits": n_splits,
            "seed": seed,
            "folds": fold_rows,
            "mean_fold_ndcg@10": mean(row["ndcg@10"] for row in fold_rows),
        },
        "bm25": bm25,
        "semantic": semantic,
        "learned": learned,
        "delta_vs_bm25": {
            key: learned[key] - bm25[key] for key in learned
        },
        "delta_vs_semantic": {
            key: learned[key] - semantic[key] for key in learned
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
        "queries": sorted(
            per_query,
            key=lambda row: float(row["delta_vs_semantic"]),
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/results/learned_relevance_oof.json",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = run_learned_relevance(
        payload,
        n_splits=args.folds,
        seed=args.seed,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"saved {output}")
    print("BM25:   ", result["bm25"])
    print("semantic:", result["semantic"])
    print("learned: ", result["learned"])
    print("delta vs BM25:", result["delta_vs_bm25"])
    print("delta vs semantic:", result["delta_vs_semantic"])
    print("robustness:", result["robustness"])


if __name__ == "__main__":
    main()
