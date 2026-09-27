"""Run the BM25 baseline on a local TREC-COVID dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_relevance.data import load_trec_covid
from semantic_relevance.evaluation import (
    ExperimentResult,
    benchmark,
    evaluate,
    runtime_metadata,
)
from semantic_relevance.retrieval import BM25Index


def run_bm25_experiment(
    data_dir: str | Path,
    *,
    top_k: int = 100,
    k1: float = 1.2,
    b: float = 0.75,
) -> ExperimentResult:
    """Load TREC-COVID, run BM25, and return metrics plus query latency."""
    dataset = load_trec_covid(data_dir)
    index = BM25Index(dataset.corpus.values(), k1=k1, b=b)

    run: dict[str, list[str]] = {}
    per_query_ms: list[float] = []

    for query_id, query in dataset.queries.items():
        summary, results = benchmark(
            lambda query_text=query.text: index.search(
                query_text,
                top_k=top_k,
            ),
            iterations=1,
            warmup=0,
        )
        per_query_ms.append(summary.mean_ms)
        run[query_id] = [document_id for document_id, _ in results]

    metrics = evaluate(run, dataset.qrels)

    from semantic_relevance.evaluation import summarize_latencies

    latency = summarize_latencies(per_query_ms)
    metadata = runtime_metadata()
    metadata.update(
        {
            "dataset": "trec-covid",
            "retriever": "bm25",
            "top_k": str(top_k),
            "k1": str(k1),
            "b": str(b),
            "documents": str(len(dataset.corpus)),
            "queries": str(len(dataset.queries)),
        }
    )

    return ExperimentResult(
        experiment="bm25_baseline",
        metrics=metrics,
        latency=latency.to_dict(),
        metadata=metadata,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="BEIR TREC-COVID directory")
    parser.add_argument(
        "--output",
        default="artifacts/results/bm25.json",
        help="Path for experiment JSON",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_bm25_experiment(
        args.data,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b,
    )
    output = result.save_json(args.output)
    print(f"saved {output}")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
