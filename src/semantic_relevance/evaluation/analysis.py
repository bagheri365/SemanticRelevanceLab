"""Query-level ranking diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .metrics import ndcg_at_k, recall_at_k


def query_metrics(
    ranked_doc_ids: Sequence[str],
    judgments: Mapping[str, int],
) -> dict[str, float]:
    """Compute metrics useful for per-query error analysis."""
    return {
        "ndcg@10": ndcg_at_k(ranked_doc_ids, judgments, 10),
        "recall@10": recall_at_k(ranked_doc_ids, judgments, 10),
        "recall@100": recall_at_k(ranked_doc_ids, judgments, 100),
    }


def relevant_document_count(
    judgments: Mapping[str, int],
    *,
    relevance_threshold: int = 1,
) -> int:
    """Count documents considered relevant for a query."""
    return sum(
        relevance >= relevance_threshold for relevance in judgments.values()
    )
