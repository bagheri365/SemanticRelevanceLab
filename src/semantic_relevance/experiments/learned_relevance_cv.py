"""Build and validate query-disjoint folds for Experiment 5."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from semantic_relevance.learning.dataset import build_relevance_examples
from semantic_relevance.learning.folds import make_query_folds, validate_query_folds


def build_cv_manifest(
    payload: dict[str, object],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Build a serializable cross-validation manifest from feature rows."""
    examples = build_relevance_examples(payload["rows"])
    query_ids = sorted({example.query_id for example in examples})
    folds = make_query_folds(query_ids, n_splits=n_splits, seed=seed)
    validate_query_folds(folds, query_ids)

    rows_by_query = Counter(example.query_id for example in examples)
    labels = Counter(example.relevance for example in examples)

    return {
        "experiment": "learned_relevance_cv_manifest",
        "source_experiment": payload.get("experiment"),
        "seed": seed,
        "n_splits": n_splits,
        "queries": len(query_ids),
        "examples": len(examples),
        "label_counts": {str(label): labels[label] for label in (0, 1, 2)},
        "folds": [
            {
                "fold": fold.fold,
                "train_query_ids": list(fold.train_query_ids),
                "test_query_ids": list(fold.test_query_ids),
                "train_queries": len(fold.train_query_ids),
                "test_queries": len(fold.test_query_ids),
                "train_examples": sum(
                    rows_by_query[query_id] for query_id in fold.train_query_ids
                ),
                "test_examples": sum(
                    rows_by_query[query_id] for query_id in fold.test_query_ids
                ),
            }
            for fold in folds
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/results/learned_relevance_cv_manifest.json",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    manifest = build_cv_manifest(
        payload,
        n_splits=args.folds,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"saved {output}")
    print(
        f'queries={manifest["queries"]} examples={manifest["examples"]} '
        f'labels={manifest["label_counts"]}'
    )
    for fold in manifest["folds"]:
        print(
            f'fold {fold["fold"]}: '
            f'train_queries={fold["train_queries"]} '
            f'test_queries={fold["test_queries"]} '
            f'train_examples={fold["train_examples"]} '
            f'test_examples={fold["test_examples"]}'
        )


if __name__ == "__main__":
    main()
