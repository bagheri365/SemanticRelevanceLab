"""Load TREC-COVID data stored in BEIR's JSONL/TSV layout."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .models import Document, Query


@dataclass(frozen=True, slots=True)
class TrecCovidDataset:
    """In-memory representation of a TREC-COVID split."""

    corpus: dict[str, Document]
    queries: dict[str, Query]
    qrels: dict[str, dict[str, int]]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}"
                ) from exc
    return rows


def _load_corpus(path: Path) -> dict[str, Document]:
    corpus: dict[str, Document] = {}
    for row in _read_jsonl(path):
        document_id = str(row.get("_id", "")).strip()
        if not document_id:
            raise ValueError(f"document without _id in {path}")
        corpus[document_id] = Document(
            document_id=document_id,
            title=str(row.get("title", "") or ""),
            text=str(row.get("text", "") or ""),
        )
    return corpus


def _load_queries(path: Path) -> dict[str, Query]:
    queries: dict[str, Query] = {}
    for row in _read_jsonl(path):
        query_id = str(row.get("_id", "")).strip()
        text = str(row.get("text", "") or "").strip()
        if not query_id:
            raise ValueError(f"query without _id in {path}")
        if not text:
            raise ValueError(f"query {query_id!r} has empty text")
        queries[query_id] = Query(query_id=query_id, text=text)
    return queries


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            query_id = row["query-id"].strip()
            document_id = row["corpus-id"].strip()
            raw_relevance = int(row["score"])
            # The BEIR TREC-COVID export contains -1 entries. Treat them as
            # non-relevant so the rest of our evaluation code keeps a
            # non-negative relevance scale.
            relevance = max(raw_relevance, 0)
            qrels.setdefault(query_id, {})[document_id] = relevance
    return qrels


def load_trec_covid(
    root: str | Path,
    *,
    split: str = "test",
) -> TrecCovidDataset:
    """Load a local BEIR-format TREC-COVID dataset directory."""
    root = Path(root)
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels" / f"{split}.tsv"

    missing = [
        path for path in (corpus_path, queries_path, qrels_path) if not path.exists()
    ]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing TREC-COVID dataset files: {names}")

    corpus = _load_corpus(corpus_path)
    queries = _load_queries(queries_path)
    qrels = _load_qrels(qrels_path)

    unknown_queries = set(qrels) - set(queries)
    if unknown_queries:
        raise ValueError(f"qrels reference unknown queries: {sorted(unknown_queries)}")

    judged_documents = {
        document_id for judgments in qrels.values() for document_id in judgments
    }
    unknown_documents = judged_documents - set(corpus)
    if unknown_documents:
        raise ValueError(
            f"qrels reference unknown documents: {sorted(unknown_documents)}"
        )

    return TrecCovidDataset(corpus=corpus, queries=queries, qrels=qrels)
