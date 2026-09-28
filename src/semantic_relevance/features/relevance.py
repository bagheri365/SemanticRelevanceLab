"""Interpretable query-document relevance features."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization for diagnostic features."""
    return TOKEN_RE.findall(text.lower())


def _coverage(query_tokens: list[str], document_tokens: set[str]) -> float:
    unique_query = set(query_tokens)
    if not unique_query:
        return 0.0
    return len(unique_query & document_tokens) / len(unique_query)


def query_token_idf(
    documents: Iterable[str],
    *,
    query_tokens: Iterable[str],
) -> dict[str, float]:
    """Compute smoothed IDF for only the query tokens we need."""
    wanted = set(query_tokens)
    if not wanted:
        return {}

    document_frequency: Counter[str] = Counter()
    document_count = 0
    for text in documents:
        document_count += 1
        present = set(tokenize(text)) & wanted
        document_frequency.update(present)

    return {
        token: math.log((document_count + 1) / (document_frequency[token] + 1)) + 1.0
        for token in wanted
    }


def extract_relevance_features(
    *,
    query: str,
    title: str,
    text: str,
    bm25_score: float,
    semantic_score: float,
    bm25_rank: int,
    semantic_rank: int,
    idf: Mapping[str, float] | None = None,
) -> dict[str, float | int]:
    """Extract interpretable lexical and rank-disagreement features."""
    query_tokens = tokenize(query)
    title_tokens = set(tokenize(title))
    body_tokens = set(tokenize(text))
    document_tokens = title_tokens | body_tokens

    phrase = " ".join(query_tokens)
    normalized_title = " ".join(tokenize(title))
    normalized_text = " ".join(tokenize(text))
    exact_phrase = bool(phrase) and (
        phrase in normalized_title or phrase in normalized_text
    )

    rare_coverage = 0.0
    if idf and query_tokens:
        unique_query = set(query_tokens)
        total_weight = sum(idf.get(token, 0.0) for token in unique_query)
        if total_weight:
            matched_weight = sum(
                idf.get(token, 0.0)
                for token in unique_query
                if token in document_tokens
            )
            rare_coverage = matched_weight / total_weight

    return {
        "query_token_coverage": _coverage(query_tokens, document_tokens),
        "title_token_coverage": _coverage(query_tokens, title_tokens),
        "rare_query_term_coverage": rare_coverage,
        "exact_query_phrase": int(exact_phrase),
        "bm25_score": float(bm25_score),
        "semantic_score": float(semantic_score),
        "bm25_rank": int(bm25_rank),
        "semantic_rank": int(semantic_rank),
        "rank_disagreement": int(bm25_rank) - int(semantic_rank),
        "absolute_rank_disagreement": abs(int(bm25_rank) - int(semantic_rank)),
    }
