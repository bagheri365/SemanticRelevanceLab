import pytest

from semantic_relevance.experiments.score_fusion import (
    fuse_query,
    minmax_normalize,
    run_score_fusion,
)


def _candidate(document_id, relevance, bm25_score, semantic_score):
    return {
        "document_id": document_id,
        "relevance": relevance,
        "bm25_score": bm25_score,
        "semantic_score": semantic_score,
    }


def test_minmax_normalize() -> None:
    assert minmax_normalize({"a": 2.0, "b": 4.0}) == {"a": 0.0, "b": 1.0}


def test_minmax_normalize_constant_scores() -> None:
    assert minmax_normalize({"a": 3.0, "b": 3.0}) == {"a": 0.0, "b": 0.0}


def test_fuse_query_endpoints_match_each_signal() -> None:
    candidates = [
        _candidate("lexical", 0, 10.0, 1.0),
        _candidate("semantic", 2, 1.0, 10.0),
    ]
    assert fuse_query(candidates, alpha=0.0)[0] == "lexical"
    assert fuse_query(candidates, alpha=1.0)[0] == "semantic"


def test_fuse_query_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        fuse_query([], alpha=1.1)


def test_run_score_fusion_reports_robustness() -> None:
    payload = {
        "experiment": "semantic_rerank",
        "model": "fake",
        "candidate_k": 2,
        "queries": [
            {
                "query_id": "q1",
                "bm25": {"ndcg@10": 0.6309297535714575},
                "semantic": {"ndcg@10": 1.0},
                "candidates": [
                    _candidate("d1", 0, 10.0, 1.0),
                    _candidate("d2", 2, 1.0, 10.0),
                ],
            }
        ],
    }
    result = run_score_fusion(payload, alphas=[0.0, 1.0], qrels={"q1": {"d2": 2}})
    lexical, semantic = result["alphas"]
    assert lexical["robustness"]["vs_bm25"]["unchanged_queries"] == 1
    assert semantic["robustness"]["vs_bm25"]["improved_queries"] == 1
    assert semantic["metrics"]["ndcg@10"] > lexical["metrics"]["ndcg@10"]


def test_run_score_fusion_requires_alphas() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_score_fusion({"queries": []}, alphas=[], qrels={})


def test_endpoints_use_full_qrels_not_candidate_only_qrels() -> None:
    payload = {
        "experiment": "semantic_rerank",
        "model": "fake",
        "candidate_k": 2,
        "queries": [
            {
                "query_id": "q1",
                "bm25": {"ndcg@10": 0.38685280723454163},
                "semantic": {"ndcg@10": 0.6131471927654584},
                "candidates": [
                    _candidate("d1", 0, 10.0, 1.0),
                    _candidate("d2", 2, 1.0, 10.0),
                ],
            }
        ],
    }
    # d3 is relevant but absent from the candidate set. It must still affect
    # NDCG/recall denominators, just as it does in the source experiment.
    qrels = {"q1": {"d2": 2, "d3": 2}}
    result = run_score_fusion(payload, alphas=[0.0, 1.0], qrels=qrels)
    lexical, semantic = result["alphas"]

    assert lexical["metrics"]["recall@10"] == pytest.approx(0.5)
    assert semantic["metrics"]["recall@10"] == pytest.approx(0.5)
    assert lexical["metrics"]["ndcg@10"] < semantic["metrics"]["ndcg@10"]
