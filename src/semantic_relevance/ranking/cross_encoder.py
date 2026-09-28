"""Cross-encoder semantic relevance scoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class PairScorer(Protocol):
    """Interface used by reranking experiments."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Score query-document text pairs."""


class CrossEncoderScorer:
    """Thin adapter around sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self.model.predict(list(pairs), show_progress_bar=False)
        return [float(score) for score in scores]


def rerank(
    query: str,
    candidates: Sequence[tuple[str, str]],
    scorer: PairScorer,
) -> list[tuple[str, float]]:
    """Rerank (document_id, text) candidates by semantic score."""
    if not candidates:
        return []
    scores = scorer.score_pairs([(query, text) for _, text in candidates])
    if len(scores) != len(candidates):
        raise ValueError("scorer returned a different number of scores")
    ranked = [
        (document_id, score)
        for (document_id, _), score in zip(candidates, scores, strict=True)
    ]
    return sorted(ranked, key=lambda item: (-item[1], item[0]))
