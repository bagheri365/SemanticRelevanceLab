import json

import pytest

from semantic_relevance.experiments.semantic_rerank import (
    run_semantic_rerank_experiment,
)


class FakeScorer:
    def score_pairs(self, pairs):
        return [
            10.0 if "target concept" in document else 1.0
            for _, document in pairs
        ]


def _write_fixture(root) -> None:
    (root / "qrels").mkdir(parents=True)
    corpus = [
        {"_id": "d1", "title": "Query words", "text": "query words distractor"},
        {"_id": "d2", "title": "Answer", "text": "target concept query"},
    ]
    queries = [{"_id": "q1", "text": "query words"}]
    with (root / "corpus.jsonl").open("w") as handle:
        for row in corpus:
            handle.write(json.dumps(row) + "\n")
    with (root / "queries.jsonl").open("w") as handle:
        for row in queries:
            handle.write(json.dumps(row) + "\n")
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t0\n"
        "q1\td2\t2\n"
    )


def test_semantic_reranking_improves_fixed_candidates(tmp_path) -> None:
    _write_fixture(tmp_path)
    payload = run_semantic_rerank_experiment(
        tmp_path,
        scorer=FakeScorer(),
        model_name="fake",
        candidate_k=2,
    )

    assert payload["semantic"]["ndcg@10"] > payload["bm25"]["ndcg@10"]
    assert payload["semantic"]["recall@100"] == payload["bm25"]["recall@100"]
    assert payload["delta"]["recall@100"] == 0.0
    assert payload["queries"][0]["delta_ndcg@10"] > 0.0


def test_semantic_reranking_validates_candidate_k(tmp_path) -> None:
    _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="candidate_k"):
        run_semantic_rerank_experiment(
            tmp_path,
            scorer=FakeScorer(),
            model_name="fake",
            candidate_k=0,
        )
