import pytest

from semantic_relevance.features.relevance import (
    extract_relevance_features,
    query_token_idf,
    tokenize,
)


def test_tokenize_normalizes_text() -> None:
    assert tokenize("SARS-CoV-2, Spike!") == ["sars", "cov", "2", "spike"]


def test_extract_relevance_features_tracks_coverage_and_disagreement() -> None:
    result = extract_relevance_features(
        query="covid diabetes complications",
        title="Diabetes and COVID complications",
        text="Patients with diabetes experienced complications.",
        bm25_score=12.0,
        semantic_score=7.5,
        bm25_rank=20,
        semantic_rank=3,
        idf={"covid": 1.0, "diabetes": 3.0, "complications": 2.0},
    )
    assert result["query_token_coverage"] == pytest.approx(1.0)
    assert result["title_token_coverage"] == pytest.approx(1.0)
    assert result["rare_query_term_coverage"] == pytest.approx(1.0)
    assert result["rank_disagreement"] == 17
    assert result["absolute_rank_disagreement"] == 17


def test_rare_term_coverage_weights_missing_terms() -> None:
    result = extract_relevance_features(
        query="covid diabetes",
        title="COVID complications",
        text="general coronavirus material",
        bm25_score=1.0,
        semantic_score=1.0,
        bm25_rank=1,
        semantic_rank=1,
        idf={"covid": 1.0, "diabetes": 4.0},
    )
    assert result["rare_query_term_coverage"] == pytest.approx(0.2)


def test_query_token_idf_makes_rare_terms_heavier() -> None:
    idf = query_token_idf(
        ["covid diabetes", "covid treatment", "covid symptoms"],
        query_tokens=["covid", "diabetes"],
    )
    assert idf["diabetes"] > idf["covid"]
