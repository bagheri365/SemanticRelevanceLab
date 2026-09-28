"""Diagnose where an out-of-fold learned ranker loses to semantic reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _candidate_map(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {
        (str(row["query_id"]), str(row["document_id"])): row
        for row in rows
    }


def analyze_learned_failures(
    learned: dict[str, object],
    features: dict[str, object],
    *,
    top_n: int = 10,
) -> dict[str, object]:
    """Summarize OOF regressions and feature patterns behind them."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")

    rows = list(features["rows"])
    by_key = _candidate_map(rows)
    regressions = [
        row for row in learned["queries"]
        if float(row["delta_vs_semantic"]) < 0
    ]

    details: list[dict[str, object]] = []
    for query in regressions:
        query_id = str(query["query_id"])
        query_rows = [
            row for row in rows if str(row["query_id"]) == query_id
        ]
        semantic_top = sorted(
            query_rows, key=lambda row: int(row["semantic_rank"])
        )[:top_n]

        # The OOF artifact currently stores metrics, not learned document ranks.
        # Approximate the diagnostic population with semantic top-N candidates and
        # expose the features most associated with relevance there.
        relevant = [row for row in semantic_top if int(row["relevance"]) > 0]
        nonrelevant = [row for row in semantic_top if int(row["relevance"]) == 0]

        def avg(group: list[dict[str, object]], name: str) -> float | None:
            if not group:
                return None
            return mean(float(row[name]) for row in group)

        details.append(
            {
                "query_id": query_id,
                "query": query["query"],
                "delta_vs_semantic": query["delta_vs_semantic"],
                "delta_vs_bm25": query["delta_vs_bm25"],
                "semantic_top10_relevant": len(relevant),
                "semantic_top10_nonrelevant": len(nonrelevant),
                "feature_contrast": {
                    name: {
                        "relevant": avg(relevant, name),
                        "non_relevant": avg(nonrelevant, name),
                    }
                    for name in (
                        "query_token_coverage",
                        "title_token_coverage",
                        "rare_query_term_coverage",
                        "rank_disagreement",
                    )
                },
            }
        )

    details.sort(key=lambda row: float(row["delta_vs_semantic"]))
    return {
        "experiment": "learned_relevance_failure_analysis",
        "evaluation_scope": {
            "candidate_universe": int(learned.get("candidate_k", 100)),
            "note": (
                "Metrics in learned_relevance_oof rank only the fixed candidate "
                "set. Recall@100=1.0 therefore means all candidate-set documents "
                "were returned, not full-corpus recall."
            ),
        },
        "summary": {
            "queries": len(learned["queries"]),
            "regressions_vs_semantic": len(regressions),
            "regression_rate": len(regressions) / len(learned["queries"]),
            "mean_delta_vs_semantic_on_regressions": (
                mean(float(row["delta_vs_semantic"]) for row in regressions)
                if regressions else 0.0
            ),
        },
        "regressions": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/results/learned_relevance_failure_analysis.json",
    )
    args = parser.parse_args()

    learned = json.loads(Path(args.learned).read_text(encoding="utf-8"))
    features = json.loads(Path(args.features).read_text(encoding="utf-8"))
    result = analyze_learned_failures(learned, features)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"saved {output}")
    print(
        "regressions vs semantic:",
        result["summary"]["regressions_vs_semantic"],
        f'/{result["summary"]["queries"]}',
    )
    print(
        "mean regression:",
        f'{result["summary"]["mean_delta_vs_semantic_on_regressions"]:+.4f}',
    )
    print("worst regressions:")
    for row in result["regressions"][:5]:
        print(
            f'{row["query_id"]}: {float(row["delta_vs_semantic"]):+.4f} '
            f'- {row["query"]}'
        )


if __name__ == "__main__":
    main()
