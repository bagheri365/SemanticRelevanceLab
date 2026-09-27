import pytest

from semantic_relevance.data import Document
from semantic_relevance.retrieval import BM25Index, tokenize


def test_tokenize_lowercases_and_splits_words() -> None:
    assert tokenize("Vaccine RESPONSE, COVID-19!") == [
        "vaccine",
        "response",
        "covid",
        "19",
    ]


def test_bm25_ranks_matching_document_first() -> None:
    index = BM25Index(
        [
            Document("d1", "Virus transmission", "coronavirus spread droplets"),
            Document("d2", "Antibody study", "vaccine immune response"),
            Document("d3", "Cooking", "pasta tomato sauce"),
        ]
    )
    results = index.search("coronavirus transmission", top_k=3)
    assert results[0][0] == "d1"
    assert results[0][1] > 0.0


def test_bm25_returns_only_documents_with_matching_terms() -> None:
    index = BM25Index(
        [
            Document("d1", "", "alpha beta"),
            Document("d2", "", "gamma delta"),
        ]
    )
    assert [doc_id for doc_id, _ in index.search("alpha", top_k=10)] == ["d1"]


def test_bm25_respects_top_k() -> None:
    index = BM25Index(
        [
            Document("d1", "", "virus"),
            Document("d2", "", "virus virus"),
            Document("d3", "", "virus virus virus"),
        ]
    )
    assert len(index.search("virus", top_k=2)) == 2


def test_bm25_rejects_invalid_parameters() -> None:
    documents = [Document("d1", "", "text")]
    with pytest.raises(ValueError, match="k1"):
        BM25Index(documents, k1=0)
    with pytest.raises(ValueError, match="b"):
        BM25Index(documents, b=1.1)


def test_bm25_rejects_duplicate_document_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        BM25Index(
            [
                Document("d1", "", "first"),
                Document("d1", "", "second"),
            ]
        )


def test_bm25_rejects_nonpositive_top_k() -> None:
    index = BM25Index([Document("d1", "", "text")])
    with pytest.raises(ValueError, match="top_k"):
        index.search("text", top_k=0)


def test_idf_is_higher_for_rarer_term() -> None:
    index = BM25Index(
        [
            Document("d1", "", "common rare"),
            Document("d2", "", "common"),
            Document("d3", "", "common"),
        ]
    )
    assert index.inverse_document_frequency("rare") > (
        index.inverse_document_frequency("common")
    )
