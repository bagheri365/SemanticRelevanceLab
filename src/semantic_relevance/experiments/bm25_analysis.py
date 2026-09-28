"""Generate query-level diagnostics for the BM25 TREC-COVID baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from semantic_relevance.data import load_trec_covid
from semantic_relevance.evaluation.analysis import (
    query_metrics,
    relevant_document_count,
)
from semantic_relevance.retrieval import BM25Index


def run_bm25_analysis(
    data_dir: str | Path,
    *,
    top_k: int = 100,
    inspect_k: int = 10,
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, object]:
    """Return per-query BM25 metrics and top-result diagnostics."""
    if inspect_k <= 0:
        raise ValueError("inspect_k must be greater than zero")
    if inspect_k > top_k:
        raise ValueError("inspect_k must not exceed top_k")

    dataset = load_trec_covid(data_dir)
    index = BM25Index(dataset.corpus.values(), k1=k1, b=b)
    queries: list[dict[str, object]] = []

    for query_id, query in dataset.queries.items():
        start = perf_counter()
        results = index.search(query.text, top_k=top_k)
        latency_ms = (perf_counter() - start) * 1000.0
        ranked_ids = [document_id for document_id, _ in results]
        judgments = dataset.qrels.get(query_id, {})

        top_results = []
        for rank, (document_id, score) in enumerate(
            results[:inspect_k], start=1
        ):
            document = dataset.corpus[document_id]
            top_results.append(
                {
                    "rank": rank,
                    "document_id": document_id,
                    "score": score,
                    "relevance": judgments.get(document_id, 0),
                    "title": document.title,
                }
            )

        queries.append(
            {
                "query_id": query_id,
                "query": query.text,
                "relevant_documents": relevant_document_count(judgments),
                "latency_ms": latency_ms,
                "metrics": query_metrics(ranked_ids, judgments),
                "top_results": top_results,
            }
        )

    queries.sort(
        key=lambda item: (
            item["metrics"]["ndcg@10"],  # type: ignore[index]
            item["metrics"]["recall@100"],  # type: ignore[index]
            item["query_id"],
        )
    )

    return {
        "experiment": "bm25_error_analysis",
        "dataset": "trec-covid",
        "parameters": {
            "top_k": top_k,
            "inspect_k": inspect_k,
            "k1": k1,
            "b": b,
        },
        "queries": queries,
    }


def save_analysis(payload: dict[str, object], path: str | Path) -> Path:
    """Serialize analysis output as readable JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/results/bm25_analysis.json",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--inspect-k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_bm25_analysis(
        args.data,
        top_k=args.top_k,
        inspect_k=args.inspect_k,
        k1=args.k1,
        b=args.b,
    )
    output = save_analysis(payload, args.output)
    print(f"saved {output}")

    queries = payload["queries"]
    print("worst queries by NDCG@10:")
    for item in queries[:5]:  # type: ignore[index]
        metrics = item["metrics"]  # type: ignore[index]
        print(
            f'  {item["query_id"]}: ndcg@10={metrics["ndcg@10"]:.4f} '
            f'recall@100={metrics["recall@100"]:.4f} '
            f'- {item["query"]}'
        )


if __name__ == "__main__":
    main()
