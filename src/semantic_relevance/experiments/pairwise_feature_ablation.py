"""Ablate pairwise relevance features and inspect fold weight stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from semantic_relevance.evaluation import evaluate
from semantic_relevance.learning.dataset import FEATURE_NAMES, RelevanceExample
from semantic_relevance.learning.folds import make_query_folds, validate_query_folds
from semantic_relevance.learning.pairwise import PairwiseLinearRanker

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "semantic_only": ("semantic_score",),
    "bm25_only": ("bm25_score",),
    "scores": ("bm25_score", "semantic_score"),
    "scores_lexical": (
        "bm25_score",
        "semantic_score",
        "query_token_coverage",
        "title_token_coverage",
        "rare_query_term_coverage",
    ),
    "scores_ranks": (
        "bm25_score",
        "semantic_score",
        "bm25_rank",
        "semantic_rank",
        "rank_disagreement",
    ),
    "all": FEATURE_NAMES,
    "minus_query_token_coverage": tuple(
        name for name in FEATURE_NAMES if name != "query_token_coverage"
    ),
    "minus_rare_query_term_coverage": tuple(
        name for name in FEATURE_NAMES if name != "rare_query_term_coverage"
    ),
    "minus_rank_disagreement": tuple(
        name for name in FEATURE_NAMES if name != "rank_disagreement"
    ),
}


def _examples(
    rows: list[dict[str, object]],
    feature_names: tuple[str, ...],
) -> list[RelevanceExample]:
    return [
        RelevanceExample(
            query_id=str(row["query_id"]),
            document_id=str(row["document_id"]),
            features=tuple(float(row[name]) for name in feature_names),
            relevance=int(row["relevance"]),
        )
        for row in rows
    ]


def _split(
    examples: list[RelevanceExample],
    train_ids: tuple[str, ...],
    test_ids: tuple[str, ...],
) -> tuple[list[RelevanceExample], list[RelevanceExample]]:
    train_set = set(train_ids)
    test_set = set(test_ids)
    return (
        [example for example in examples if example.query_id in train_set],
        [example for example in examples if example.query_id in test_set],
    )


def _weight_stability(
    feature_names: tuple[str, ...],
    fold_weights: list[list[float]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for index, name in enumerate(feature_names):
        values = [weights[index] for weights in fold_weights]
        positive = sum(value > 0 for value in values)
        negative = sum(value < 0 for value in values)
        result[name] = {
            "mean": mean(values),
            "std": pstdev(values),
            "min": min(values),
            "max": max(values),
            "positive_folds": positive,
            "negative_folds": negative,
            "sign_consistent": positive == len(values) or negative == len(values),
            "fold_weights": values,
        }
    return result


def run_pairwise_feature_ablation(
    payload: dict[str, object],
    *,
    n_splits: int = 5,
    seed: int = 42,
    epochs: int = 30,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    max_pairs_per_query: int = 2000,
) -> dict[str, object]:
    """Evaluate fixed feature subsets using the same query-level folds."""
    rows = list(payload["rows"])
    query_ids = sorted({str(row["query_id"]) for row in rows})
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
    semantic = evaluate(semantic_run, qrels)

    experiments: list[dict[str, object]] = []
    for name, feature_names in FEATURE_SETS.items():
        examples = _examples(rows, feature_names)
        run: dict[str, list[str]] = {}
        fold_weights: list[list[float]] = []
        fold_rows: list[dict[str, object]] = []

        for fold in folds:
            train, test = _split(
                examples,
                fold.train_query_ids,
                fold.test_query_ids,
            )
            model = PairwiseLinearRanker(
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
                max_pairs_per_query=max_pairs_per_query,
                seed=seed + fold.fold,
            ).fit(train)
            fold_weights.append(list(model.weights or []))
            scores = model.predict([example.features for example in test])

            scored_by_query: dict[str, list[tuple[str, float]]] = {}
            for example, score in zip(test, scores, strict=True):
                scored_by_query.setdefault(example.query_id, []).append(
                    (example.document_id, float(score))
                )
            for query_id, scored in scored_by_query.items():
                run[query_id] = [
                    document_id
                    for document_id, _ in sorted(
                        scored,
                        key=lambda item: (-item[1], item[0]),
                    )
                ]

            fold_qrels = {
                query_id: qrels[query_id] for query_id in fold.test_query_ids
            }
            fold_rows.append(
                {
                    "fold": fold.fold,
                    "test_query_ids": list(fold.test_query_ids),
                    "ndcg@10": evaluate(
                        {qid: run[qid] for qid in fold.test_query_ids},
                        fold_qrels,
                    )["ndcg@10"],
                }
            )

        metrics = evaluate(run, qrels)
        per_query = []
        for query_id in query_ids:
            judgments = qrels[query_id]
            model_metric = evaluate(
                {query_id: run[query_id]},
                {query_id: judgments},
            )["ndcg@10"]
            semantic_metric = evaluate(
                {query_id: semantic_run[query_id]},
                {query_id: judgments},
            )["ndcg@10"]
            per_query.append(
                {
                    "query_id": query_id,
                    "delta_vs_semantic": model_metric - semantic_metric,
                }
            )

        experiments.append(
            {
                "name": name,
                "features": list(feature_names),
                "metrics": metrics,
                "delta_vs_semantic": {
                    key: metrics[key] - semantic[key] for key in metrics
                },
                "robustness_vs_semantic": {
                    "regressions": sum(
                        float(row["delta_vs_semantic"]) < 0 for row in per_query
                    ),
                    "worst_regression": min(
                        float(row["delta_vs_semantic"]) for row in per_query
                    ),
                },
                "folds": fold_rows,
                "weight_stability": _weight_stability(
                    feature_names,
                    fold_weights,
                ),
            }
        )

    return {
        "experiment": "pairwise_feature_ablation",
        "semantic": semantic,
        "cross_validation": {"n_splits": n_splits, "seed": seed},
        "ablations": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/results/pairwise_feature_ablation.json",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-pairs-per-query", type=int, default=2000)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = run_pairwise_feature_ablation(
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
    print("name                         ndcg@10   mrr      regress  worst")
    for row in result["ablations"]:
        print(
            f'{row["name"]:<28} '
            f'{row["metrics"]["ndcg@10"]:.4f}   '
            f'{row["metrics"]["mrr"]:.4f}   '
            f'{row["robustness_vs_semantic"]["regressions"]:>3}      '
            f'{row["robustness_vs_semantic"]["worst_regression"]:+.4f}'
        )
    print("\nall-feature weight stability:")
    all_features = next(
        row for row in result["ablations"] if row["name"] == "all"
    )
    for feature, stats in sorted(
        all_features["weight_stability"].items(),
        key=lambda item: abs(float(item[1]["mean"])),
        reverse=True,
    ):
        print(
            f'{float(stats["mean"]):+.4f} ± {float(stats["std"]):.4f} '
            f'{feature} sign_consistent={stats["sign_consistent"]}'
        )


if __name__ == "__main__":
    main()
