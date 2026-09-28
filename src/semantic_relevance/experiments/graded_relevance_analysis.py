"""Analyze semantic reranking through graded relevance judgments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

GRADES = (0, 1, 2)


def _grade_counts(rows: list[dict[str, object]], rank_field: str, k: int) -> dict[str, int]:
    top = sorted(rows, key=lambda row: int(row[rank_field]))[:k]
    counts = Counter(int(row["relevance"]) for row in top)
    return {str(grade): counts[grade] for grade in GRADES}


def _mean_rank(rows: list[dict[str, object]], rank_field: str, grade: int) -> float:
    ranks = [
        int(row[rank_field])
        for row in rows
        if int(row["relevance"]) == grade
    ]
    return mean(ranks) if ranks else 0.0


def analyze_graded_relevance(
    payload: dict[str, object],
    *,
    top_k: int = 10,
) -> dict[str, object]:
    """Compare BM25 and semantic ranking behavior by relevance grade."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    rows = list(payload["rows"])
    query_ids = sorted({str(row["query_id"]) for row in rows}, key=int)
    queries: list[dict[str, object]] = []

    for query_id in query_ids:
        query_rows = [row for row in rows if str(row["query_id"]) == query_id]
        bm25_counts = _grade_counts(query_rows, "bm25_rank", top_k)
        semantic_counts = _grade_counts(query_rows, "semantic_rank", top_k)

        grade_rank_changes = {}
        for grade in GRADES:
            before = _mean_rank(query_rows, "bm25_rank", grade)
            after = _mean_rank(query_rows, "semantic_rank", grade)
            grade_rank_changes[str(grade)] = {
                "bm25_mean_rank": before,
                "semantic_mean_rank": after,
                "mean_rank_change": before - after,
            }

        promoted = [
            row for row in query_rows
            if int(row["semantic_rank"]) < int(row["bm25_rank"])
        ]
        promotion_grades = Counter(int(row["relevance"]) for row in promoted)

        queries.append(
            {
                "query_id": query_id,
                "query": query_rows[0]["query"],
                "semantic_delta_ndcg@10": float(
                    query_rows[0]["semantic_delta_ndcg@10"]
                ),
                "bm25_top_k_grades": bm25_counts,
                "semantic_top_k_grades": semantic_counts,
                "top_k_grade_delta": {
                    str(grade): semantic_counts[str(grade)] - bm25_counts[str(grade)]
                    for grade in GRADES
                },
                "promoted_by_grade": {
                    str(grade): promotion_grades[grade] for grade in GRADES
                },
                "grade_rank_changes": grade_rank_changes,
            }
        )

    regressions = sorted(
        (row for row in queries if float(row["semantic_delta_ndcg@10"]) < 0),
        key=lambda row: float(row["semantic_delta_ndcg@10"]),
    )

    return {
        "experiment": "graded_relevance_analysis",
        "source_experiment": payload.get("experiment"),
        "top_k": top_k,
        "queries": queries,
        "regressions": regressions,
        "aggregate": {
            "bm25_top_k_grades": {
                str(grade): sum(
                    int(row["bm25_top_k_grades"][str(grade)]) for row in queries
                )
                for grade in GRADES
            },
            "semantic_top_k_grades": {
                str(grade): sum(
                    int(row["semantic_top_k_grades"][str(grade)]) for row in queries
                )
                for grade in GRADES
            },
        },
    }


def save_result(payload: dict[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/results/graded_relevance_analysis.json",
    )
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = analyze_graded_relevance(payload, top_k=args.top_k)
    output = save_result(result, args.output)
    print(f"saved {output}")

    aggregate = result["aggregate"]
    print(f"aggregate top-{args.top_k} relevance grades:")
    print("  BM25:    ", aggregate["bm25_top_k_grades"])
    print("  semantic:", aggregate["semantic_top_k_grades"])

    print("\nsemantic regressions:")
    for row in result["regressions"]:
        print(
            f'{row["query_id"]}: delta={row["semantic_delta_ndcg@10"]:+.4f} '
            f'BM25={row["bm25_top_k_grades"]} '
            f'semantic={row["semantic_top_k_grades"]} '
            f'- {row["query"]}'
        )


if __name__ == "__main__":
    main()
