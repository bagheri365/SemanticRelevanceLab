"""Inspect score ties and document movements in OOF learned rankings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def build_ranking_diagnostics(
    payload: dict[str, object],
    *,
    top_n: int = 10,
) -> dict[str, object]:
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")

    candidates = list(payload["candidates"])
    query_metrics = {str(row["query_id"]): row for row in payload["queries"]}
    query_ids = sorted({str(row["query_id"]) for row in candidates})
    rows: list[dict[str, object]] = []

    for query_id in query_ids:
        group = [row for row in candidates if str(row["query_id"]) == query_id]
        counts = Counter(float(row["learned_score"]) for row in group)
        relevant = [row for row in group if int(row["relevance"]) > 0]
        movements = [
            {
                "document_id": row["document_id"],
                "relevance": int(row["relevance"]),
                "semantic_rank": int(row["semantic_rank"]),
                "learned_rank": int(row["learned_rank"]),
                "movement": int(row["semantic_rank"]) - int(row["learned_rank"]),
                "learned_score": float(row["learned_score"]),
            }
            for row in relevant
        ]
        movements.sort(key=lambda row: int(row["movement"]))

        semantic_top = sorted(group, key=lambda row: int(row["semantic_rank"]))[:top_n]
        learned_top = sorted(group, key=lambda row: int(row["learned_rank"]))[:top_n]

        def grades(items: list[dict[str, object]]) -> dict[str, int]:
            result = {"0": 0, "1": 0, "2": 0}
            for item in items:
                result[str(int(item["relevance"]))] += 1
            return result

        metric = query_metrics[query_id]
        rows.append(
            {
                "query_id": query_id,
                "query": metric["query"],
                "delta_vs_semantic": metric["delta_vs_semantic"],
                "unique_learned_scores": len(counts),
                "largest_tie_group": max(counts.values()),
                "tie_fraction": max(counts.values()) / len(group),
                "semantic_top_grades": grades(semantic_top),
                "learned_top_grades": grades(learned_top),
                "worst_relevant_demotions": movements[:5],
                "best_relevant_promotions": movements[-5:][::-1],
            }
        )

    regressions = [row for row in rows if float(row["delta_vs_semantic"]) < 0]
    return {
        "experiment": "learned_ranking_diagnostics",
        "summary": {
            "queries": len(rows),
            "mean_unique_learned_scores": mean(
                int(row["unique_learned_scores"]) for row in rows
            ),
            "mean_largest_tie_group": mean(
                int(row["largest_tie_group"]) for row in rows
            ),
            "mean_tie_fraction": mean(float(row["tie_fraction"]) for row in rows),
            "regressions_vs_semantic": len(regressions),
            "regression_mean_unique_scores": (
                mean(int(row["unique_learned_scores"]) for row in regressions)
                if regressions else 0.0
            ),
            "regression_mean_tie_fraction": (
                mean(float(row["tie_fraction"]) for row in regressions)
                if regressions else 0.0
            ),
        },
        "queries": sorted(rows, key=lambda row: float(row["delta_vs_semantic"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/results/learned_ranking_diagnostics.json",
    )
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = build_ranking_diagnostics(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"saved {output}")
    for key, value in result["summary"].items():
        print(f"{key}: {value}")
    print("worst regressions:")
    for row in result["queries"][:5]:
        print(
            f'{row["query_id"]}: delta={float(row["delta_vs_semantic"]):+.4f} '
            f'unique={row["unique_learned_scores"]} '
            f'largest_tie={row["largest_tie_group"]} - {row["query"]}'
        )


if __name__ == "__main__":
    main()
