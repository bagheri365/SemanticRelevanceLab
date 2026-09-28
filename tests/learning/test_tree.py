import pytest

from semantic_relevance.learning.tree import RegressionTree


def test_tree_learns_simple_relevance_boundary() -> None:
    features = [(0.0,), (0.1,), (0.9,), (1.0,)]
    labels = [0, 0, 2, 2]
    model = RegressionTree(max_depth=2, min_samples_leaf=1).fit(features, labels)
    predictions = model.predict([(0.05,), (0.95,)])
    assert predictions[0] < predictions[1]


def test_tree_requires_fit_before_predict() -> None:
    with pytest.raises(ValueError, match="fitted"):
        RegressionTree().predict([(1.0,)])


def test_tree_rejects_unaligned_training_data() -> None:
    with pytest.raises(ValueError, match="aligned"):
        RegressionTree().fit([(1.0,)], [])
