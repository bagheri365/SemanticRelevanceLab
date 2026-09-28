"""Small dependency-free regression tree for relevance scoring."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from statistics import mean
from typing import Sequence


@dataclass(slots=True)
class _Node:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None


class RegressionTree:
    """A small CART-style regression tree for graded relevance."""

    def __init__(
        self,
        *,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        max_thresholds: int = 16,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.root: _Node | None = None

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[float],
    ) -> "RegressionTree":
        if not features or len(features) != len(labels):
            raise ValueError("features and labels must be non-empty and aligned")
        width = len(features[0])
        if width == 0 or any(len(row) != width for row in features):
            raise ValueError("feature rows must have a consistent non-zero width")
        rows = [tuple(map(float, row)) for row in features]
        targets = [float(label) for label in labels]
        self.root = self._build(rows, targets, depth=0)
        return self

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        if self.root is None:
            raise ValueError("model must be fitted before prediction")
        return [self._predict_one(tuple(map(float, row)), self.root) for row in features]

    def _predict_one(self, row: tuple[float, ...], node: _Node) -> float:
        while node.feature is not None:
            if row[node.feature] <= float(node.threshold):
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
        return node.value

    def _build(
        self,
        rows: list[tuple[float, ...]],
        labels: list[float],
        *,
        depth: int,
    ) -> _Node:
        node = _Node(value=mean(labels))
        if depth >= self.max_depth or len(rows) < 2 * self.min_samples_leaf:
            return node
        if max(labels) == min(labels):
            return node

        feature, threshold = self._best_split(rows, labels)
        if feature is None:
            return node

        left_rows, left_labels, right_rows, right_labels = [], [], [], []
        for row, label in zip(rows, labels, strict=True):
            target_rows, target_labels = (
                (left_rows, left_labels)
                if row[feature] <= threshold
                else (right_rows, right_labels)
            )
            target_rows.append(row)
            target_labels.append(label)

        node.feature = feature
        node.threshold = threshold
        node.left = self._build(left_rows, left_labels, depth=depth + 1)
        node.right = self._build(right_rows, right_labels, depth=depth + 1)
        return node

    def _best_split(
        self,
        rows: list[tuple[float, ...]],
        labels: list[float],
    ) -> tuple[int | None, float]:
        best_loss = inf
        best_feature: int | None = None
        best_threshold = 0.0

        for feature in range(len(rows[0])):
            values = sorted({row[feature] for row in rows})
            if len(values) < 2:
                continue
            candidates = self._thresholds(values)
            for threshold in candidates:
                left = [
                    label for row, label in zip(rows, labels, strict=True)
                    if row[feature] <= threshold
                ]
                right = [
                    label for row, label in zip(rows, labels, strict=True)
                    if row[feature] > threshold
                ]
                if (
                    len(left) < self.min_samples_leaf
                    or len(right) < self.min_samples_leaf
                ):
                    continue
                loss = self._squared_error(left) + self._squared_error(right)
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _thresholds(self, values: list[float]) -> list[float]:
        mids = [(a + b) / 2.0 for a, b in zip(values, values[1:])]
        if len(mids) <= self.max_thresholds:
            return mids
        return [
            mids[round(i * (len(mids) - 1) / (self.max_thresholds - 1))]
            for i in range(self.max_thresholds)
        ]

    @staticmethod
    def _squared_error(values: list[float]) -> float:
        center = mean(values)
        return sum((value - center) ** 2 for value in values)
