import json

import pytest

from semantic_relevance.experiments.bm25_analysis import (
    run_bm25_analysis,
    save_analysis,
)


def _write_fixture(root) -> None:
    (root / "qrels").mkdir(parents=True)
    corpus = [
        {"_id": "d1", "title": "Virus", "text": "coronavirus transmission"},
        {"_id": "d2", "title": "Vaccine", "text": "antibody immune response"},
        {"_id": "d3", "title": "Other", "text": "unrelated document"},
    ]
    queries = [
        {"_id": "q1", "text": "coronavirus transmission"},
        {"_id": "q2", "text": "vaccine antibody"},
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
        "q2\td2\t2\n"
    )


def test_analysis_contains_query_metrics_and_ranked_results(tmp_path) -> None:
    _write_fixture(tmp_path)
    payload = run_bm25_analysis(tmp_path, top_k=3, inspect_k=2)

    assert payload["experiment"] == "bm25_error_analysis"
    assert len(payload["queries"]) == 2

    query = next(item for item in payload["queries"] if item["query_id"] == "q1")
    assert query["metrics"]["ndcg@10"] == 1.0
    assert query["relevant_documents"] == 1
    assert query["top_results"][0]["document_id"] == "d1"
    assert query["top_results"][0]["relevance"] == 2
    assert query["latency_ms"] >= 0.0


def test_analysis_sorts_worst_ndcg_first(tmp_path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "unmatched words"}) + "\n"
        + json.dumps({"_id": "q2", "text": "vaccine antibody"}) + "\n"
    )
    payload = run_bm25_analysis(tmp_path, top_k=3, inspect_k=2)
    assert payload["queries"][0]["query_id"] == "q1"


def test_analysis_validates_inspect_k(tmp_path) -> None:
    _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="inspect_k"):
        run_bm25_analysis(tmp_path, top_k=3, inspect_k=0)
    with pytest.raises(ValueError, match="must not exceed"):
        run_bm25_analysis(tmp_path, top_k=3, inspect_k=4)


def test_save_analysis_writes_json(tmp_path) -> None:
    output = save_analysis(
        {"experiment": "test", "queries": []},
        tmp_path / "nested" / "analysis.json",
    )
    assert json.loads(output.read_text())["experiment"] == "test"
