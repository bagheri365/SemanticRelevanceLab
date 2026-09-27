import pytest

from semantic_relevance.evaluation import RelevanceJudgment, build_qrels


def test_build_qrels_groups_judgments_by_query() -> None:
    judgments = [
        RelevanceJudgment("q1", "d1", 2),
        RelevanceJudgment("q1", "d2", 0),
        RelevanceJudgment("q2", "d3", 1),
    ]
    assert build_qrels(judgments) == {
        "q1": {"d1": 2, "d2": 0},
        "q2": {"d3": 1},
    }


def test_relevance_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RelevanceJudgment("q1", "d1", -1)


def test_duplicate_judgment_uses_latest_value() -> None:
    judgments = [
        RelevanceJudgment("q1", "d1", 1),
        RelevanceJudgment("q1", "d1", 3),
    ]
    assert build_qrels(judgments)["q1"]["d1"] == 3
