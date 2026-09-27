"""Ranking metrics used throughout SemanticRelevanceLab."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Sequence[str]]
Metric = Callable[[Sequence[str], Mapping[str, int]], float]


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero")


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    """Compute discounted cumulative gain at rank k."""
    _validate_k(k)
    return sum(
        (2**relevance - 1) / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances[:k])
    )


def ndcg_at_k(
    ranked_doc_ids: Sequence[str],
    judgments: Mapping[str, int],
    k: int = 10,
) -> float:
    """Compute NDCG@k for one query."""
    _validate_k(k)
    observed = [judgments.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    ideal = sorted(judgments.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(observed, k) / ideal_dcg


def reciprocal_rank(
    ranked_doc_ids: Sequence[str],
    judgments: Mapping[str, int],
    relevance_threshold: int = 1,
) -> float:
    """Return reciprocal rank of the first relevant result."""
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if judgments.get(doc_id, 0) >= relevance_threshold:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    ranked_doc_ids: Sequence[str],
    judgments: Mapping[str, int],
    k: int,
    relevance_threshold: int = 1,
) -> float:
    """Compute recall@k for one query."""
    _validate_k(k)
    relevant = {
        doc_id
        for doc_id, relevance in judgments.items()
        if relevance >= relevance_threshold
    }
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def mean_metric(run: Run, qrels: Qrels, metric: Metric) -> float:
    """Average a query-level metric over judged queries."""
    if not qrels:
        return 0.0
    scores = [
        metric(run.get(query_id, ()), judgments)
        for query_id, judgments in qrels.items()
    ]
    return sum(scores) / len(scores)


def evaluate(run: Run, qrels: Qrels) -> dict[str, float]:
    """Return the core ranking metrics for an experiment run."""
    return {
        "ndcg@10": mean_metric(
            run, qrels, lambda docs, rels: ndcg_at_k(docs, rels, 10)
        ),
        "mrr": mean_metric(run, qrels, reciprocal_rank),
        "recall@10": mean_metric(
            run, qrels, lambda docs, rels: recall_at_k(docs, rels, 10)
        ),
        "recall@100": mean_metric(
            run, qrels, lambda docs, rels: recall_at_k(docs, rels, 100)
        ),
    }
