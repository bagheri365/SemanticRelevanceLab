"""Evaluation utilities for ranking experiments."""

from .experiment import ExperimentResult, runtime_metadata, set_seed
from .latency import LatencySummary, benchmark, percentile, summarize_latencies
from .metrics import dcg_at_k, evaluate, ndcg_at_k, recall_at_k, reciprocal_rank
from .types import RankedResult, RelevanceJudgment, build_qrels

__all__ = [
    "ExperimentResult",
    "LatencySummary",
    "RankedResult",
    "RelevanceJudgment",
    "benchmark",
    "build_qrels",
    "dcg_at_k",
    "evaluate",
    "ndcg_at_k",
    "percentile",
    "recall_at_k",
    "reciprocal_rank",
    "runtime_metadata",
    "set_seed",
    "summarize_latencies",
]
