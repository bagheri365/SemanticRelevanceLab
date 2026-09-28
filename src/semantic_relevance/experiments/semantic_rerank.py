"""Compare BM25 with cross-encoder reranking on TREC-COVID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from semantic_relevance.data import load_trec_covid
from semantic_relevance.evaluation import evaluate
from semantic_relevance.evaluation.analysis import query_metrics
from semantic_relevance.ranking import CrossEncoderScorer, PairScorer, rerank
from semantic_relevance.retrieval import BM25Index

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def _reciprocal_rank(ids: list[str], judgments: dict[str, int]) -> float:
    for rank, document_id in enumerate(ids, start=1):
        if judgments.get(document_id, 0) >= 1:
            return 1.0 / rank
    return 0.0


def run_semantic_rerank_experiment(
    data_dir: str | Path,
    *,
    scorer: PairScorer,
    model_name: str,
    candidate_k: int = 100,
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, object]:
    """Rerank fixed BM25 candidates and compare aggregate/query metrics."""
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero")

    dataset = load_trec_covid(data_dir)
    index = BM25Index(dataset.corpus.values(), k1=k1, b=b)

    bm25_run: dict[str, list[str]] = {}
    semantic_run: dict[str, list[str]] = {}
    query_rows: list[dict[str, object]] = []
    rerank_latencies: list[float] = []

    for query_id, query in dataset.queries.items():
        bm25_results = index.search(query.text, top_k=candidate_k)
        bm25_ids = [document_id for document_id, _ in bm25_results]
        candidates = [
            (document_id, dataset.corpus[document_id].searchable_text)
            for document_id in bm25_ids
        ]

        start = perf_counter()
        semantic_results = rerank(query.text, candidates, scorer)
        rerank_latencies.append((perf_counter() - start) * 1000.0)
        semantic_ids = [document_id for document_id, _ in semantic_results]

        bm25_run[query_id] = bm25_ids
        semantic_run[query_id] = semantic_ids
        judgments = dataset.qrels.get(query_id, {})
        before = query_metrics(bm25_ids, judgments)
        after = query_metrics(semantic_ids, judgments)
        query_rows.append(
            {
                "query_id": query_id,
                "query": query.text,
                "bm25": before,
                "semantic": after,
                "delta_ndcg@10": after["ndcg@10"] - before["ndcg@10"],
                "delta_mrr": (
                    _reciprocal_rank(semantic_ids, judgments)
                    - _reciprocal_rank(bm25_ids, judgments)
                ),
            }
        )

    bm25_metrics = evaluate(bm25_run, dataset.qrels)
    semantic_metrics = evaluate(semantic_run, dataset.qrels)
    query_rows.sort(key=lambda row: float(row["delta_ndcg@10"]))

    return {
        "experiment": "semantic_rerank",
        "dataset": "trec-covid",
        "model": model_name,
        "candidate_k": candidate_k,
        "bm25": bm25_metrics,
        "semantic": semantic_metrics,
        "delta": {
            name: semantic_metrics[name] - bm25_metrics[name]
            for name in bm25_metrics
        },
        "rerank_latency_ms": {
            "mean": sum(rerank_latencies) / len(rerank_latencies),
            "min": min(rerank_latencies),
            "max": max(rerank_latencies),
        },
        "queries": query_rows,
    }


def save_result(payload: dict[str, object], path: str | Path) -> Path:
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
        default="artifacts/results/semantic_rerank.json",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--candidate-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scorer = CrossEncoderScorer(args.model)
    payload = run_semantic_rerank_experiment(
        args.data,
        scorer=scorer,
        model_name=args.model,
        candidate_k=args.candidate_k,
    )
    output = save_result(payload, args.output)
    print(f"saved {output}")
    print("BM25:", payload["bm25"])
    print("semantic:", payload["semantic"])
    print("delta:", payload["delta"])


if __name__ == "__main__":
    main()
