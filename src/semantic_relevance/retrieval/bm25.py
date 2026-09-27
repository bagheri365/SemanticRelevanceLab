"""A small BM25 implementation for controlled retrieval experiments."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from semantic_relevance.data import Document

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text using a simple deterministic rule."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory inverted index with BM25 scoring."""

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b
        self.documents: dict[str, Document] = {}
        self.doc_lengths: dict[str, int] = {}
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)

        for document in documents:
            if document.document_id in self.documents:
                raise ValueError(f"duplicate document id: {document.document_id}")
            self.documents[document.document_id] = document
            tokens = tokenize(document.searchable_text)
            self.doc_lengths[document.document_id] = len(tokens)
            for term, frequency in Counter(tokens).items():
                self.postings[term][document.document_id] = frequency

        self.document_count = len(self.documents)
        self.average_doc_length = (
            sum(self.doc_lengths.values()) / self.document_count
            if self.document_count
            else 0.0
        )

    def inverse_document_frequency(self, term: str) -> float:
        """Return Robertson/Sparck Jones BM25 IDF with a positive offset."""
        document_frequency = len(self.postings.get(term, {}))
        return math.log(
            1.0
            + (
                self.document_count - document_frequency + 0.5
            )
            / (document_frequency + 0.5)
        )

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """Return document ids and BM25 scores in descending score order."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not self.documents:
            return []

        scores: dict[str, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.inverse_document_frequency(term)
            for document_id, term_frequency in postings.items():
                doc_length = self.doc_lengths[document_id]
                length_ratio = (
                    doc_length / self.average_doc_length
                    if self.average_doc_length
                    else 0.0
                )
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                scores[document_id] += idf * (
                    term_frequency * (self.k1 + 1.0) / denominator
                )

        return sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
