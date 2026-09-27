"""Core data structures for ranking experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    """A graded relevance judgment for one query-document pair."""

    query_id: str
    document_id: str
    relevance: int

    def __post_init__(self) -> None:
        if self.relevance < 0:
            raise ValueError("relevance must be non-negative")


@dataclass(frozen=True, slots=True)
class RankedResult:
    """A scored document returned for a query."""

    query_id: str
    document_id: str
    score: float


def build_qrels(
    judgments: list[RelevanceJudgment],
) -> dict[str, dict[str, int]]:
    """Convert judgment records to query -> document -> relevance."""
    qrels: dict[str, dict[str, int]] = {}
    for judgment in judgments:
        qrels.setdefault(judgment.query_id, {})[judgment.document_id] = (
            judgment.relevance
        )
    return qrels
