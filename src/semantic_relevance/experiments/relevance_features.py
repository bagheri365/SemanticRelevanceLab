"""Extract interpretable relevance features from semantic reranking artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from semantic_relevance.data import load_trec_covid
from semantic_relevance.features.relevance import (
    extract_relevance_features,
    query_token_idf,
    tokenize,
)


FEATURE_NAMES = (
    "query_token_coverage",
    "title_token_coverage",
    "rare_query_term_coverage",
    "exact_query_phrase",
    "bm25_score",
    "semantic_score",
    "bm25_rank",
    "semantic_rank",
    "rank_disagreement",
    "absolute_rank_disagreement",
)


def build_feature_dataset(
    payload: dict[str, object],
    *,
    data_dir: str | Path,
) -> dict[str, object]:
    """Join reranker diagnostics with text and full relevance judgments."""
    dataset = load_trec_covid(data_dir)
    rows: list[dict[str, object]] = []

    corpus_texts = [
        document.searchable_text
        for document in dataset.corpus.values()
    ]

    for query_row in payload["queries"]:
        query_id = str(query_row["query_id"])
        query = str(query_row["query"])
        q_tokens = tokenize(query)
        idf = query_token_idf(corpus_texts, query_tokens=q_tokens)
        judgments = dataset.qrels.get(query_id, {})

        for candidate in query_row["candidates"]:
            document_id = str(candidate["document_id"])
            document = dataset.corpus[document_id]
            feature_values = extract_relevance_features(
                query=query,
                title=document.title,
                text=document.text,
                bm25_score=float(candidate["bm25_score"]),
                semantic_score=float(candidate["semantic_score"]),
                bm25_rank=int(candidate["bm25_rank"]),
                semantic_rank=int(candidate["semantic_rank"]),
                idf=idf,
            )
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "document_id": document_id,
                    "title": document.title,
                    "relevance": int(judgments.get(document_id, 0)),
                    "semantic_delta_ndcg@10": float(
                        query_row["delta_ndcg@10"]
                    ),
                    **feature_values,
                }
            )

    return {
        "experiment": "relevance_features",
        "source_experiment": payload.get("experiment"),
        "dataset": "trec-covid",
        "feature_names": list(FEATURE_NAMES),
        "rows": rows,
        "summary": summarize_features(rows),
    }


def summarize_features(rows: list[dict[str, object]]) -> dict[str, object]:
    """Compare feature means for relevant and non-relevant candidates."""
    groups = {
        "relevant": [row for row in rows if int(row["relevance"]) > 0],
        "non_relevant": [row for row in rows if int(row["relevance"]) == 0],
        "semantic_regression": [
            row
            for row in rows
            if float(row["semantic_delta_ndcg@10"]) < 0
        ],
        "semantic_improvement": [
            row
            for row in rows
            if float(row["semantic_delta_ndcg@10"]) > 0
        ],
    }

    summary: dict[str, object] = {}
    for name, group in groups.items():
        summary[name] = {
            "count": len(group),
            "feature_means": {
                feature: (
                    mean(float(row[feature]) for row in group)
                    if group
                    else 0.0
                )
                for feature in FEATURE_NAMES
            },
        }
    return summary


def save_result(payload: dict[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/results/relevance_features.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = build_feature_dataset(payload, data_dir=args.data)
    output = save_result(result, args.output)
    print(f"saved {output}")
    print(f'candidate rows: {len(result["rows"])}')

    summary = result["summary"]
    for feature in (
        "query_token_coverage",
        "title_token_coverage",
        "rare_query_term_coverage",
        "absolute_rank_disagreement",
    ):
        relevant = summary["relevant"]["feature_means"][feature]
        non_relevant = summary["non_relevant"]["feature_means"][feature]
        print(
            f"{feature}: relevant={relevant:.4f} "
            f"non_relevant={non_relevant:.4f}"
        )


if __name__ == "__main__":
    main()
