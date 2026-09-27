import json

import pytest

from semantic_relevance.data import load_trec_covid


def _write_fixture(root) -> None:
    (root / "qrels").mkdir(parents=True)
    corpus = [
        {"_id": "d1", "title": "Virus", "text": "coronavirus transmission"},
        {"_id": "d2", "title": "", "text": "vaccine antibody response"},
    ]
    queries = [
        {"_id": "q1", "text": "how does coronavirus spread"},
        {"_id": "q2", "text": "vaccine response"},
    ]
    with (root / "corpus.jsonl").open("w") as handle:
        for row in corpus:
            handle.write(json.dumps(row) + "\n")
    with (root / "queries.jsonl").open("w") as handle:
        for row in queries:
            handle.write(json.dumps(row) + "\n")
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t2\n"
        "q1\td2\t0\n"
        "q2\td2\t1\n"
    )


def test_load_trec_covid_reads_beir_layout(tmp_path) -> None:
    _write_fixture(tmp_path)
    dataset = load_trec_covid(tmp_path)
    assert dataset.corpus["d1"].title == "Virus"
    assert dataset.queries["q1"].text == "how does coronavirus spread"
    assert dataset.qrels["q1"] == {"d1": 2, "d2": 0}


def test_load_trec_covid_supports_named_split(tmp_path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "qrels" / "dev.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq2\td2\t2\n"
    )
    dataset = load_trec_covid(tmp_path, split="dev")
    assert dataset.qrels == {"q2": {"d2": 2}}


def test_load_trec_covid_reports_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing TREC-COVID"):
        load_trec_covid(tmp_path)


def test_load_trec_covid_rejects_unknown_query(tmp_path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nunknown\td1\t1\n"
    )
    with pytest.raises(ValueError, match="unknown queries"):
        load_trec_covid(tmp_path)


def test_load_trec_covid_rejects_unknown_document(tmp_path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\tmissing\t1\n"
    )
    with pytest.raises(ValueError, match="unknown documents"):
        load_trec_covid(tmp_path)


def test_load_trec_covid_rejects_negative_relevance(tmp_path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t-1\n"
    )
    with pytest.raises(ValueError, match="non-negative"):
        load_trec_covid(tmp_path)
