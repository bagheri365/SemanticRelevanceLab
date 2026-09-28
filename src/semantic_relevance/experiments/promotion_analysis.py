"""Analyze which semantic promotions help or hurt relevance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

FEATURES = (
    "query_token_coverage",
    "title_token_coverage",
    "rare_query_term_coverage",
    "bm25_score",
    "semantic_score",
    "absolute_rank_disagreement",
)

KNOWN_FAILURE_QUERY_IDS = ("24", "19", "36", "34", "23")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values: list[float]) -> dict[str, float]:
    """Return compact distribution statistics."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "p25": _quantile(values, 0.25),
        "p75": _quantile(values, 0.75),
    }


def movement_group(row: dict[str, object]) -> str:
    """Classify a candidate by semantic movement and relevance."""
    bm25_rank = int(row["bm25_rank"])
    semantic_rank = int(row["semantic_rank"])
    relevant = int(row["relevance"]) > 0
    if semantic_rank < bm25_rank:
        movement = "promoted"
    elif semantic_rank > bm25_rank:
        movement = "demoted"
    else:
        movement = "unchanged"
    return f"{movement}_{'relevant' if relevant else 'non_relevant'}"


def analyze_promotions(payload: dict[str, object]) -> dict[str, object]:
    """Compare good and bad semantic movements and known failure queries."""
    rows = list(payload["rows"])
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(movement_group(row), []).append(row)

    groups: dict[str, object] = {}
    for name, group in sorted(grouped.items()):
        groups[name] = {
            "count": len(group),
            "features": {
                feature: describe([float(row[feature]) for row in group])
                for feature in FEATURES
            },
            "promotion_magnitude": describe(
                [
                    float(int(row["bm25_rank"]) - int(row["semantic_rank"]))
                    for row in group
                ]
            ),
        }

    per_query: dict[str, object] = {}
    query_ids = sorted({str(row["query_id"]) for row in rows}, key=int)
    for query_id in query_ids:
        query_rows = [row for row in rows if str(row["query_id"]) == query_id]
        good = [
            row for row in query_rows
            if movement_group(row) == "promoted_relevant"
        ]
        bad = [
            row for row in query_rows
            if movement_group(row) == "promoted_non_relevant"
        ]
        per_query[query_id] = {
            "query": query_rows[0]["query"] if query_rows else "",
            "semantic_delta_ndcg@10": (
                float(query_rows[0]["semantic_delta_ndcg@10"])
                if query_rows else 0.0
            ),
            "promoted_relevant": len(good),
            "promoted_non_relevant": len(bad),
            "promotion_precision": (
                len(good) / (len(good) + len(bad))
                if good or bad else 0.0
            ),
            "rare_coverage_gap": (
                mean(float(row["rare_query_term_coverage"]) for row in good)
                - mean(float(row["rare_query_term_coverage"]) for row in bad)
                if good and bad else 0.0
            ),
            "title_coverage_gap": (
                mean(float(row["title_token_coverage"]) for row in good)
                - mean(float(row["title_token_coverage"]) for row in bad)
                if good and bad else 0.0
            ),
        }

    failures: dict[str, object] = {}
    for query_id in KNOWN_FAILURE_QUERY_IDS:
        query_rows = [row for row in rows if str(row["query_id"]) == query_id]
        promoted = sorted(
            (
                row for row in query_rows
                if int(row["semantic_rank"]) < int(row["bm25_rank"])
            ),
            key=lambda row: (
                int(row["semantic_rank"]) - int(row["bm25_rank"]),
                int(row["semantic_rank"]),
            ),
        )
        failures[query_id] = {
            "query": query_rows[0]["query"] if query_rows else "",
            "semantic_delta_ndcg@10": (
                float(query_rows[0]["semantic_delta_ndcg@10"])
                if query_rows else 0.0
            ),
            "largest_promotions": [
                {
                    "document_id": row["document_id"],
                    "title": row["title"],
                    "relevance": row["relevance"],
                    "bm25_rank": row["bm25_rank"],
                    "semantic_rank": row["semantic_rank"],
                    "promotion": int(row["bm25_rank"]) - int(row["semantic_rank"]),
                    "query_token_coverage": row["query_token_coverage"],
                    "title_token_coverage": row["title_token_coverage"],
                    "rare_query_term_coverage": row["rare_query_term_coverage"],
                }
                for row in promoted[:10]
            ],
        }

    return {
        "experiment": "semantic_promotion_analysis",
        "source_experiment": payload.get("experiment"),
        "groups": groups,
        "per_query": per_query,
        "known_failures": failures,
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
        default="artifacts/results/promotion_analysis.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = analyze_promotions(payload)
    output = save_result(result, args.output)
    print(f"saved {output}")

    groups = result["groups"]
    for name in ("promoted_relevant", "promoted_non_relevant",
                 "demoted_relevant", "demoted_non_relevant"):
        group = groups.get(name)
        if not group:
            continue
        print(
            f'{name}: n={group["count"]} '
            f'rare={group["features"]["rare_query_term_coverage"]["mean"]:.4f} '
            f'title={group["features"]["title_token_coverage"]["mean"]:.4f} '
            f'movement={group["promotion_magnitude"]["mean"]:+.2f}'
        )

    print("\nknown semantic regressions:")
    for query_id, row in result["known_failures"].items():
        promotions = row["largest_promotions"]
        bad = sum(int(item["relevance"]) == 0 for item in promotions)
        print(
            f'{query_id}: delta={row["semantic_delta_ndcg@10"]:+.4f} '
            f'nonrelevant_in_top_promotions={bad}/{len(promotions)} '
            f'- {row["query"]}'
        )


if __name__ == "__main__":
    main()
