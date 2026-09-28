"""Evaluate lexical/semantic score fusion from a saved reranking artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from semantic_relevance.evaluation.analysis import query_metrics


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1], preserving ties for constant inputs."""
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        return {key: 0.0 for key in scores}
    scale = high - low
    return {key: (value - low) / scale for key, value in scores.items()}


def fuse_query(
    candidates: list[dict[str, object]],
    *,
    alpha: float,
) -> list[str]:
    """Fuse normalized BM25 and semantic scores for one candidate set."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    bm25 = {
        str(row["document_id"]): float(row["bm25_score"])
        for row in candidates
    }
    semantic = {
        str(row["document_id"]): float(row["semantic_score"])
        for row in candidates
    }
    bm25_norm = minmax_normalize(bm25)
    semantic_norm = minmax_normalize(semantic)

    fused = {
        document_id: (1.0 - alpha) * bm25_norm[document_id]
        + alpha * semantic_norm[document_id]
        for document_id in bm25
    }
    return sorted(fused, key=lambda document_id: (-fused[document_id], document_id))


def _aggregate(query_metrics_rows: list[dict[str, float]]) -> dict[str, float]:
    names = ("ndcg@10", "recall@10", "recall@100")
    return {
        name: mean(row[name] for row in query_metrics_rows)
        for name in names
    }


def run_score_fusion(
    payload: dict[str, object],
    *,
    alphas: list[float],
) -> dict[str, object]:
    """Sweep fusion weights and report quality plus per-query robustness."""
    if not alphas:
        raise ValueError("at least one alpha is required")
    if any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError("alpha must be between 0 and 1")

    rows = list(payload["queries"])
    experiments: list[dict[str, object]] = []

    for alpha in alphas:
        metric_rows: list[dict[str, float]] = []
        deltas_vs_bm25: list[float] = []
        deltas_vs_semantic: list[float] = []

        for row in rows:
            candidates = list(row["candidates"])
            judgments = {
                str(candidate["document_id"]): int(candidate["relevance"])
                for candidate in candidates
                if int(candidate["relevance"]) > 0
            }
            fused_ids = fuse_query(candidates, alpha=alpha)
            fused_metrics = query_metrics(fused_ids, judgments)
            metric_rows.append(fused_metrics)

            bm25_ndcg = float(row["bm25"]["ndcg@10"])
            semantic_ndcg = float(row["semantic"]["ndcg@10"])
            deltas_vs_bm25.append(fused_metrics["ndcg@10"] - bm25_ndcg)
            deltas_vs_semantic.append(fused_metrics["ndcg@10"] - semantic_ndcg)

        aggregate = _aggregate(metric_rows)
        experiments.append(
            {
                "alpha": alpha,
                "metrics": aggregate,
                "robustness": {
                    "vs_bm25": _robustness(deltas_vs_bm25),
                    "vs_semantic": _robustness(deltas_vs_semantic),
                },
            }
        )

    return {
        "experiment": "score_fusion",
        "source_experiment": payload.get("experiment"),
        "model": payload.get("model"),
        "candidate_k": payload.get("candidate_k"),
        "normalization": "per-query min-max",
        "formula": "(1-alpha)*bm25 + alpha*semantic",
        "alphas": experiments,
    }


def _robustness(deltas: list[float]) -> dict[str, float | int]:
    epsilon = 1e-12
    return {
        "improved_queries": sum(delta > epsilon for delta in deltas),
        "regressed_queries": sum(delta < -epsilon for delta in deltas),
        "unchanged_queries": sum(abs(delta) <= epsilon for delta in deltas),
        "mean_delta_ndcg@10": mean(deltas),
        "worst_delta_ndcg@10": min(deltas),
        "best_delta_ndcg@10": max(deltas),
    }


def save_result(payload: dict[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _parse_alphas(value: str) -> list[float]:
    try:
        alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alphas must be comma-separated numbers") from exc
    if not alphas or any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise argparse.ArgumentTypeError("alphas must be between 0 and 1")
    return alphas


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/results/score_fusion.json",
    )
    parser.add_argument(
        "--alphas",
        type=_parse_alphas,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="comma-separated semantic weights",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = run_score_fusion(payload, alphas=args.alphas)
    output = save_result(result, args.output)
    print(f"saved {output}")
    print("alpha  ndcg@10  recall@10  regress-vs-bm25  worst-vs-bm25")
    for row in result["alphas"]:
        metrics = row["metrics"]
        robustness = row["robustness"]["vs_bm25"]
        print(
            f'{row["alpha"]:>4.2f}   {metrics["ndcg@10"]:.4f}    '
            f'{metrics["recall@10"]:.4f}       '
            f'{robustness["regressed_queries"]:>2}             '
            f'{robustness["worst_delta_ndcg@10"]:+.4f}'
        )


if __name__ == "__main__":
    main()
