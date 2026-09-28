"""Small dependency-free pairwise linear ranker."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from random import Random
from typing import Sequence

from semantic_relevance.learning.dataset import RelevanceExample


@dataclass(frozen=True, slots=True)
class PairwiseTrainingStats:
    """Summary of generated preference pairs."""

    pairs: int
    queries: int


class PairwiseLinearRanker:
    """Learn a linear scoring function from within-query relevance preferences."""

    def __init__(
        self,
        *,
        epochs: int = 30,
        learning_rate: float = 0.05,
        l2: float = 1e-4,
        max_pairs_per_query: int = 2000,
        seed: int = 42,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if max_pairs_per_query < 1:
            raise ValueError("max_pairs_per_query must be at least 1")
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.max_pairs_per_query = max_pairs_per_query
        self.seed = seed
        self.weights: list[float] | None = None
        self.means: list[float] | None = None
        self.scales: list[float] | None = None
        self.training_stats: PairwiseTrainingStats | None = None

    def fit(self, examples: Sequence[RelevanceExample]) -> "PairwiseLinearRanker":
        if not examples:
            raise ValueError("examples must be non-empty")
        width = len(examples[0].features)
        if width == 0 or any(len(example.features) != width for example in examples):
            raise ValueError("feature rows must have a consistent non-zero width")

        columns = list(zip(*(example.features for example in examples), strict=True))
        self.means = [sum(column) / len(column) for column in columns]
        self.scales = []
        for column, center in zip(columns, self.means, strict=True):
            variance = sum((value - center) ** 2 for value in column) / len(column)
            self.scales.append(sqrt(variance) or 1.0)

        by_query: dict[str, list[RelevanceExample]] = {}
        for example in examples:
            by_query.setdefault(example.query_id, []).append(example)

        pairs: list[tuple[tuple[float, ...], float]] = []
        rng = Random(self.seed)
        for query_id in sorted(by_query):
            group = by_query[query_id]
            query_pairs: list[tuple[tuple[float, ...], float]] = []
            for left_index, left in enumerate(group):
                for right in group[left_index + 1 :]:
                    if left.relevance == right.relevance:
                        continue
                    preferred, other = (
                        (left, right)
                        if left.relevance > right.relevance
                        else (right, left)
                    )
                    preferred_features = self._normalize(preferred.features)
                    other_features = self._normalize(other.features)
                    difference = tuple(
                        a - b
                        for a, b in zip(
                            preferred_features, other_features, strict=True
                        )
                    )
                    query_pairs.append((difference, 1.0))
                    query_pairs.append(
                        (tuple(-value for value in difference), -1.0)
                    )
            if len(query_pairs) > self.max_pairs_per_query:
                rng.shuffle(query_pairs)
                query_pairs = query_pairs[: self.max_pairs_per_query]
            pairs.extend(query_pairs)

        if not pairs:
            raise ValueError("training data contains no differing relevance pairs")

        weights = [0.0] * width
        order = list(range(len(pairs)))
        for _ in range(self.epochs):
            rng.shuffle(order)
            for index in order:
                difference, label = pairs[index]
                margin = label * sum(
                    weight * value
                    for weight, value in zip(weights, difference, strict=True)
                )
                # Stable derivative of logistic pairwise loss.
                if margin >= 0:
                    probability = exp(-margin) / (1.0 + exp(-margin))
                else:
                    probability = 1.0 / (1.0 + exp(margin))
                for feature_index, value in enumerate(difference):
                    gradient = -label * value * probability
                    gradient += self.l2 * weights[feature_index]
                    weights[feature_index] -= self.learning_rate * gradient

        self.weights = weights
        self.training_stats = PairwiseTrainingStats(
            pairs=len(pairs),
            queries=len(by_query),
        )
        return self

    def score(self, features: Sequence[float]) -> float:
        if self.weights is None:
            raise ValueError("model must be fitted before scoring")
        normalized = self._normalize(features)
        return sum(
            weight * value
            for weight, value in zip(self.weights, normalized, strict=True)
        )

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        return [self.score(row) for row in features]

    def _normalize(self, features: Sequence[float]) -> tuple[float, ...]:
        if self.means is None or self.scales is None:
            raise ValueError("model must be fitted before normalization")
        if len(features) != len(self.means):
            raise ValueError("feature width does not match fitted model")
        return tuple(
            (float(value) - center) / scale
            for value, center, scale in zip(
                features, self.means, self.scales, strict=True
            )
        )
