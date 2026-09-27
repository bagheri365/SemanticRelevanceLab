import json

from semantic_relevance.experiments.bm25 import run_bm25_experiment


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


def test_run_bm25_experiment_produces_metrics_and_metadata(tmp_path) -> None:
    _write_fixture(tmp_path)
    result = run_bm25_experiment(tmp_path, top_k=3)

    assert result.experiment == "bm25_baseline"
    assert result.metrics["ndcg@10"] == 1.0
    assert result.metrics["mrr"] == 1.0
    assert result.metrics["recall@10"] == 1.0
    assert result.latency["count"] == 2.0
    assert result.metadata["dataset"] == "trec-covid"
    assert result.metadata["documents"] == "3"
    assert result.metadata["queries"] == "2"
