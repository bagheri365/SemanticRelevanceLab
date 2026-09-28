import pytest

from semantic_relevance.experiments.semantic_rerank import _snippet


def test_snippet_compacts_whitespace() -> None:
    assert _snippet("one\n\n two   three") == "one two three"


def test_snippet_truncates_long_text() -> None:
    value = _snippet("abcdefghij", limit=6)
    assert value == "abcde…"
    assert len(value) == 6
