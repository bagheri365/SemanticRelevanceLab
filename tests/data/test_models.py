from semantic_relevance.data import Document, Query


def test_document_searchable_text_combines_title_and_body() -> None:
    document = Document("d1", "Vaccine response", "Antibody study")
    assert document.searchable_text == "Vaccine response Antibody study"


def test_document_searchable_text_handles_empty_title() -> None:
    document = Document("d1", "", "Antibody study")
    assert document.searchable_text == "Antibody study"


def test_query_holds_identifier_and_text() -> None:
    query = Query("q1", "coronavirus transmission")
    assert query.query_id == "q1"
    assert query.text == "coronavirus transmission"
