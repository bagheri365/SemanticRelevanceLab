"""Render semantic reranking wins and regressions for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_error_analysis(
    payload: dict[str, object],
    *,
    query_count: int = 5,
    result_count: int = 10,
) -> dict[str, object]:
    """Select largest NDCG wins/regressions and their top semantic results."""
    if query_count <= 0 or result_count <= 0:
        raise ValueError("query_count and result_count must be greater than zero")

    queries = list(payload["queries"])
    regressions = queries[:query_count]
    improvements = list(reversed(queries[-query_count:]))

    def summarize(row: dict[str, object]) -> dict[str, object]:
        candidates = list(row["candidates"])
        candidates.sort(key=lambda item: int(item["semantic_rank"]))
        return {
            "query_id": row["query_id"],
            "query": row["query"],
            "delta_ndcg@10": row["delta_ndcg@10"],
            "delta_mrr": row["delta_mrr"],
            "bm25": row["bm25"],
            "semantic": row["semantic"],
            "results": candidates[:result_count],
        }

    return {
        "experiment": "semantic_error_analysis",
        "model": payload["model"],
        "candidate_k": payload["candidate_k"],
        "improvements": [summarize(row) for row in improvements],
        "regressions": [summarize(row) for row in regressions],
    }


def save_analysis(payload: dict[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _print_group(title: str, rows: list[dict[str, object]]) -> None:
    print(f"\n{title}")
    for row in rows:
        print(
            f'\n{row["query_id"]}: ΔNDCG={row["delta_ndcg@10"]:+.4f} '
            f'| {row["query"]}'
        )
        for result in row["results"]:
            print(
                f'  semantic #{result["semantic_rank"]:>2} '
                f'(BM25 #{result["bm25_rank"]:>2}, '
                f'move {result["rank_movement"]:+d}) '
                f'rel={result["relevance"]} '
                f'ce={result["semantic_score"]:.3f} '
                f'bm25={result["bm25_score"]:.3f} '
                f'| {result["title"]}'
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/results/semantic_error_analysis.json",
    )
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--results", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    analysis = build_error_analysis(
        payload,
        query_count=args.queries,
        result_count=args.results,
    )
    output = save_analysis(analysis, args.output)
    print(f"saved {output}")
    _print_group("BIGGEST IMPROVEMENTS", analysis["improvements"])
    _print_group("BIGGEST REGRESSIONS", analysis["regressions"])


if __name__ == "__main__":
    main()
