# SemanticRelevanceLab — Research Progress

## Project Goal

Study semantic relevance modeling for multi-stage search systems through controlled retrieval, reranking, error-analysis, and score-fusion experiments.

The project focuses on a central search question:

> How can lexical and semantic relevance signals be combined to improve ranking quality while understanding and limiting query-specific failures?

## Dataset

Experiments use the **TREC-COVID** dataset distributed through BEIR:

- 171,332 documents
- 50 queries
- graded relevance judgments
- corpus, queries, and qrels stored locally under `data/trec-covid/`

The same qrels are used across experiments so ranking results remain comparable.

## Evaluation

Primary metrics:

- NDCG@10 — ranking quality with graded relevance near the top of the result list
- MRR — position of the first relevant result
- Recall@10 — relevant-document coverage in the top 10
- Recall@100 — candidate-set recall

Query-level analysis is used alongside aggregate metrics because average quality can hide severe regressions on individual queries.

---

## Experiment 1 — BM25 Baseline

### Research question

How strong is a lexical retrieval baseline, and where does lexical matching fail to model relevance?

### Setup

BM25 retrieves the top 100 documents for every query.

Parameters:

- `k1 = 1.2`
- `b = 0.75`
- `top_k = 100`

### Results

| Metric | BM25 |
| --- | ---: |
| NDCG@10 | 0.5552 |
| MRR | 0.7797 |
| Recall@10 | 0.0157 |
| Recall@100 | 0.0969 |

Mean measured query latency was approximately 228 ms in the baseline run.

### Error analysis

Several of the weakest BM25 queries exposed cases where lexical overlap did not correspond well to the user's information need.

Examples included:

- SARS-CoV-2 subtypes
- coronavirus triage guidelines
- coronavirus versus seasonal flu
- origin of COVID-19
- quarantine best practices

For the subtype query, BM25 ranked broadly coronavirus-related documents above documents that directly discussed SARS-CoV-2 subtypes.

### Learning

BM25 can provide a useful candidate set, but lexical term matching alone does not reliably model whether a document satisfies the semantic relationship expressed by a query.

This motivated keeping BM25 as candidate generation while changing only the relevance-scoring stage.

---

## Experiment 2 — Cross-Encoder Semantic Reranking

### Research question

Given a fixed BM25 candidate set, can semantic relevance scoring improve top-rank quality?

### Setup

Pipeline:

```text
query
  ↓
BM25 top 100 candidates
  ↓
cross-encoder query-document scoring
  ↓
semantic reranking
```

Model:

`cross-encoder/ms-marco-MiniLM-L6-v2`

The candidate set is held constant. This isolates ranking quality from retrieval coverage.

### Results

| Metric | BM25 | Semantic | Delta |
| --- | ---: | ---: | ---: |
| NDCG@10 | 0.5552 | 0.6750 | +0.1198 |
| MRR | 0.7797 | 0.8657 | +0.0859 |
| Recall@10 | 0.0157 | 0.0181 | +0.0024 |
| Recall@100 | 0.0969 | 0.0969 | 0 |

NDCG@10 improved by approximately **21.6% relative** to BM25.

Recall@100 remained unchanged, confirming that the improvement came from reranking the same candidate set rather than retrieving additional documents.

Mean measured reranking latency was approximately 686 ms per query in this experiment.

### Largest observed improvements

Examples included:

- weather effects on coronavirus: `+0.6187` NDCG@10
- initial COVID-19 symptoms: `+0.5595`
- SARS-CoV-2 phylogenetic analysis: `+0.4658`
- SARS-CoV-2 subtypes: `+0.3996`
- immunity and cross-protection: `+0.3803`

The subtype query demonstrated the intended behavior particularly clearly: documents buried deep in the BM25 candidate list because of lexical scoring could be promoted when they directly satisfied the semantic intent.

### Largest observed regressions

Examples included:

- COVID-19 complications associated with diabetes: `-0.3475`
- hand sanitizer needed to destroy COVID-19: `-0.2214`
- SARS-CoV-2 spike protein structure: `-0.1729`
- longer-term complications after recovery: `-0.1425`
- COVID-19 complications associated with hypertension: `-0.1316`

### Semantic failure modes

Error analysis suggested several distinct relevance problems.

#### 1. Constraint underweighting

A document can match the broad topic while failing an important query constraint.

Example:

```text
COVID-19 + complications + diabetes
```

The reranker sometimes strongly rewarded `COVID-19 + complications` without sufficiently enforcing `diabetes`.

#### 2. Topical similarity versus answer relevance

Documents can be semantically close to the topic without actually satisfying the requested information need.

The hand-sanitizer query exposed this distinction.

#### 3. Entity specificity

High semantic similarity does not guarantee that the correct entity is being discussed.

The spike-protein query showed confusion between SARS-related material and SARS-CoV-2-specific material.

#### 4. Aspect and temporal mismatch

A document discussing recovery or post-COVID concepts is not necessarily relevant to a query specifically asking about long-term complications after recovery.

### Learning

Semantic similarity is not equivalent to relevance.

A relevance model must account for constraints, entities, aspects, and other query-document relationships rather than only broad semantic relatedness.

---

## Experiment 3 — Lexical + Semantic Score Fusion

### Research question

Can lexical evidence reduce severe semantic-ranking regressions while preserving most of the semantic model's aggregate quality gain?

### Setup

For each BM25 candidate, both BM25 and cross-encoder scores are available.

Scores are normalized per query using min-max normalization and combined as:

```text
fused_score =
    (1 - alpha) * normalized_bm25
    + alpha * normalized_semantic
```

The semantic weight `alpha` is swept across:

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

All variants use the same candidate set and the authoritative full TREC-COVID qrels.

### Evaluation integrity lesson

An initial implementation reconstructed qrels from only the top-100 candidates. This incorrectly changed recall denominators and the ideal ranking used by NDCG.

The problem was detected through endpoint sanity checks:

- `alpha = 0` must reproduce the original BM25 metrics.
- `alpha = 1` must reproduce the original semantic-reranker metrics.

The experiment was corrected to evaluate every fused ranking against the complete dataset qrels.

This is an important experimental lesson:

> Evaluation infrastructure is part of the research. Sanity checks against known endpoints can catch metric bugs before incorrect conclusions are drawn.

### Corrected results

| Semantic weight α | NDCG@10 | Recall@10 | Queries regressed vs BM25 | Worst NDCG regression |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.5552 | 0.0157 | 0 | +0.0000 |
| 0.25 | 0.5899 | 0.0164 | 7 | -0.0636 |
| 0.50 | 0.6237 | 0.0170 | 9 | -0.0442 |
| 0.75 | 0.6714 | **0.0185** | 10 | -0.1677 |
| 1.00 | **0.6750** | 0.0181 | 12 | -0.3475 |

### Interpretation

Pure semantic reranking achieves the highest mean NDCG@10, but it also produces the largest observed per-query regression.

At `alpha = 0.75`:

- NDCG@10 is 0.6714 versus 0.6750 for pure semantic ranking.
- the number of regressions versus BM25 falls from 12 to 10.
- the worst regression improves from `-0.3475` to `-0.1677`.
- Recall@10 reaches 0.0185, slightly above the pure semantic result.

At `alpha = 0.50`:

- NDCG@10 remains substantially above BM25 at 0.6237.
- the worst observed regression is only `-0.0442`.

These results reveal a **quality-versus-robustness tradeoff** rather than a single universally best scoring configuration.

Lexical evidence appears capable of acting as a useful complementary signal when the semantic model overgeneralizes.

---

## Research Progression

The project has evolved through a sequence of controlled questions:

```text
BM25 baseline
    ↓
lexical failure analysis
    ↓
semantic cross-encoder reranking
    ↓
+21.6% relative NDCG@10
    ↓
semantic failure analysis
    ↓
constraint / entity / aspect failures
    ↓
lexical + semantic score fusion
    ↓
quality versus robustness tradeoff
```

The key shift is from asking:

> Which ranking model has the highest aggregate metric?

to asking:

> Which signals explain relevance, when do they fail, and how can multiple signals produce a ranking system that is both effective and robust?

---

## Current Findings

1. **Candidate generation and relevance ranking are separable problems.**  
   Holding BM25 top-100 candidates fixed allowed the project to isolate semantic ranking improvements.

2. **Semantic reranking substantially improves average top-rank quality.**  
   NDCG@10 increased from 0.5552 to 0.6750.

3. **Aggregate gains hide query-specific failures.**  
   Pure semantic ranking produced regressions as large as `-0.3475` NDCG@10.

4. **Semantic similarity does not guarantee relevance.**  
   Constraint satisfaction, entity specificity, and query aspect matter.

5. **Lexical and semantic signals are complementary.**  
   Score fusion can trade a small amount of average semantic quality for substantially better worst-query behavior.

6. **Evaluation correctness must itself be tested.**  
   Endpoint checks exposed an invalid candidate-only-qrels evaluation before conclusions were drawn from it.

---

## Next Study — Feature-Based Relevance Diagnostics

### Research question

> Can simple, interpretable query-document features identify semantic relevance failures and explain when lexical evidence should influence a semantic ranker?

### Candidate features

Initial features to investigate:

- query token coverage
- rare-query-term coverage
- title token coverage
- exact phrase overlap
- BM25 score
- BM25 rank
- semantic score
- semantic rank
- lexical/semantic rank disagreement
- lexical/semantic score disagreement

### Hypotheses

**H1 — Constraint coverage**

Semantic regressions are more likely when a document has a high cross-encoder score but weak coverage of important query terms.

**H2 — Signal disagreement**

Large disagreement between lexical and semantic rankings identifies queries/documents where relevance decisions are less reliable.

**H3 — Title evidence**

Relevant documents may exhibit stronger coverage of discriminative query terms in their titles than semantically similar but irrelevant documents.

### Initial experiment

Before training another ranker:

1. extract interpretable features for every query-document candidate pair;
2. join features with relevance judgments;
3. compare feature distributions for relevant and non-relevant candidates;
4. compare semantic wins and semantic regressions;
5. identify features associated with large lexical/semantic disagreement;
6. inspect whether those features explain known failure cases.

The immediate goal is **diagnosis, not model training**.

If useful signals emerge, a later experiment can test a small learning-to-rank model or calibrated relevance classifier without requiring large-scale model training.

---

## Engineering Principles Used So Far

- fixed candidate sets for controlled ranking comparisons
- query-level metrics in addition to aggregate metrics
- reproducible JSON experiment artifacts
- explicit latency measurement
- automated tests for evaluation infrastructure
- endpoint sanity checks
- failure analysis before introducing additional modeling complexity
- separation of retrieval, ranking, evaluation, and experiment code

---

## Next Milestone

Build feature extraction and feature-analysis infrastructure for the fixed BM25 candidate set.

The milestone should answer:

> What observable query-document signals distinguish semantic-ranking successes from semantic-ranking failures?

Only after answering that question should the project move toward learned feature combination or learning-to-rank.

---

## Experiment 4 — Interpretable Relevance Diagnostics

### Research question

Can simple, interpretable query-document features help explain when semantic reranking succeeds or fails?

### Feature extraction

The fixed BM25 top-100 candidate set was converted into 5,000 query-document rows.

Features included:

- query token coverage
- title token coverage
- IDF-weighted rare-query-term coverage
- exact query phrase match
- BM25 score and rank
- semantic score and rank
- lexical/semantic rank disagreement

### Relevant versus non-relevant candidates

| Feature | Relevant | Non-relevant | Difference |
| --- | ---: | ---: | ---: |
| Query token coverage | 0.7088 | 0.6672 | +0.0416 |
| Title token coverage | 0.3207 | 0.2486 | +0.0721 |
| Rare-query-term coverage | 0.6459 | 0.5883 | +0.0576 |
| Absolute rank disagreement | 26.8094 | 27.8756 | -1.0662 |

Relevant candidates show stronger lexical evidence, particularly in titles and in coverage of rarer query terms.

Absolute lexical/semantic rank disagreement alone does not separate relevance well. This suggests that disagreement is not inherently a failure: semantic reranking must disagree with BM25 to improve results. The important question is what evidence accompanies the disagreement.

### Promotion analysis

Candidates were divided into semantic ranking movements:

- promoted and relevant
- promoted and non-relevant
- demoted and relevant
- demoted and non-relevant

Observed group means:

| Group | Count | Rare-term coverage | Title coverage | Mean rank movement |
| --- | ---: | ---: | ---: | ---: |
| Promoted relevant | 1,189 | 0.6259 | 0.3303 | +29.72 |
| Promoted non-relevant | 1,250 | 0.5760 | 0.2638 | +26.56 |
| Demoted relevant | 925 | 0.6697 | 0.3076 | -23.99 |
| Demoted non-relevant | 1,564 | 0.5978 | 0.2360 | -29.64 |

Useful semantic promotions retain stronger rare-term and title evidence than harmful promotions.

However, demoted relevant documents have even stronger rare-term coverage than promoted relevant documents. Therefore no single lexical feature provides a sufficient relevance rule. The behavior depends on interactions between lexical evidence, semantic evidence, and ranking movement.

### Known semantic regressions

Among the ten largest semantic promotions for previously identified regression queries:

| Query | NDCG@10 delta | Non-relevant among 10 largest promotions |
| --- | ---: | ---: |
| COVID-19 complications associated with diabetes | -0.3475 | 5 / 10 |
| Hand sanitizer needed to destroy COVID-19 | -0.2214 | 8 / 10 |
| SARS-CoV-2 spike protein structure | -0.1729 | 1 / 10 |
| Long-term complications after recovery | -0.1425 | 8 / 10 |
| COVID-19 complications associated with hypertension | -0.1316 | 3 / 10 |

These failures are not homogeneous.

For sanitizer and long-term complications, semantic reranking makes many clearly harmful promotions. The spike-protein query is different: almost all of the largest promotions are judged relevant, yet NDCG still falls substantially.

This motivated analysis of graded rather than binary relevance.

---

## Experiment 4B — Graded Relevance Analysis

### Research question

Are semantic regressions caused only by irrelevant promotions, or can the model also fail to distinguish highly relevant from partially relevant documents?

TREC-COVID relevance grades were kept separate:

```text
0 = non-relevant
1 = partially relevant
2 = highly relevant
```

### Aggregate top-10 composition

Across 50 queries, there are 500 top-10 positions.

| Grade | BM25 | Semantic | Change |
| --- | ---: | ---: | ---: |
| 0 — non-relevant | 187 | 123 | -64 |
| 1 — partially relevant | 75 | 73 | -2 |
| 2 — highly relevant | 238 | 304 | +66 |

This provides a more concrete explanation for the semantic model's aggregate NDCG improvement.

Semantic reranking replaces many non-relevant top-10 results with highly relevant grade-2 results. The improvement is therefore not merely an increase in binary relevance; it substantially improves the grade composition of the result set.

### Regression: diabetes complications

```text
BM25:     grade 0 = 1, grade 1 = 0, grade 2 = 9
Semantic: grade 0 = 2, grade 1 = 2, grade 2 = 6
```

BM25 already produces an unusually strong result set. Semantic reranking replaces several highly relevant results with weaker or irrelevant documents.

### Regression: hand sanitizer

```text
BM25:     grade 0 = 3, grade 1 = 4, grade 2 = 3
Semantic: grade 0 = 7, grade 1 = 2, grade 2 = 1
```

This is a broad relevance failure: the semantic ranker substantially increases irrelevant results while reducing highly relevant results.

### Regression: SARS-CoV-2 spike protein structure

```text
BM25:     grade 0 = 0, grade 1 = 1, grade 2 = 9
Semantic: grade 0 = 1, grade 1 = 2, grade 2 = 7
```

Promotion analysis had shown only one non-relevant document among the ten largest semantic promotions. Graded analysis explains the apparent contradiction.

The semantic ranker is not primarily promoting unrelated material. Instead, it degrades a very strong BM25 ranking by replacing some grade-2 documents with grade-1 and grade-0 documents.

This is a fine-grained relevance discrimination failure rather than simply a topical matching failure.

### Regression: hypertension complications

```text
BM25:     grade 0 = 3, grade 1 = 2, grade 2 = 5
Semantic: grade 0 = 2, grade 1 = 2, grade 2 = 6
```

Top-10 grade composition improves, yet NDCG@10 decreases by 0.1316.

This demonstrates that grade counts alone are insufficient. NDCG is position-sensitive: moving highly relevant documents lower in the ranking can reduce quality even when the total number of grade-2 documents increases.

### Learning

The diagnostic experiments reveal multiple kinds of semantic-ranking failure:

1. promoting irrelevant documents;
2. failing important lexical/query constraints;
3. confusing partial relevance with high relevance;
4. degrading already-strong lexical rankings;
5. ordering highly relevant documents poorly within the top ranks.

The evidence also shows why neither lexical overlap nor semantic score should be treated as a sufficient relevance signal in isolation.

---

## Updated Research Progression

```text
BM25 baseline
    ↓
lexical failure analysis
    ↓
cross-encoder semantic reranking
    ↓
+21.6% relative NDCG@10
    ↓
semantic error analysis
    ↓
lexical + semantic score fusion
    ↓
quality / robustness frontier
    ↓
interpretable relevance features
    ↓
good vs harmful semantic promotions
    ↓
graded relevance analysis
    ↓
multiple distinct ranking failure modes
    ↓
next: learned relevance combination
```

---

## Next Study — Small Learned Relevance Model

### Research question

> Can a small model learn when to trust lexical evidence, semantic evidence, and constraint-related features in order to predict graded relevance on unseen queries?

### Inputs

The initial feature set will use signals already produced by the project:

- BM25 score
- BM25 rank
- semantic score
- semantic rank
- query token coverage
- title token coverage
- rare-query-term coverage
- exact query phrase match
- lexical/semantic rank disagreement

### Target

Use the original graded relevance judgments:

```text
0 = non-relevant
1 = partially relevant
2 = highly relevant
```

### Evaluation design

Candidate rows must **not** be randomly split.

Documents from the same query share the same information need, so randomly distributing query-document rows between train and test sets would leak query-specific information and produce an unrealistically easy evaluation.

Instead, evaluation should split by query.

Proposed protocol:

```text
50 queries
    ↓
5-fold cross-validation by query
    ↓
~40 training queries / ~10 held-out queries per fold
    ↓
predict relevance for held-out candidates
    ↓
reconstruct held-out rankings
    ↓
evaluate NDCG@10
```

Every query must appear in exactly one held-out fold.

### Baselines

The learned ranker should be compared against:

- BM25
- pure semantic reranking
- lexical/semantic score fusion

Evaluation should include:

- mean NDCG@10
- Recall@10 where appropriate
- number of regressions versus BM25
- worst per-query regression
- query-level wins and losses

### Modeling principle

The first learned model should remain deliberately small.

The objective is not to outperform the cross-encoder through model scale. It is to test whether interactions among existing relevance signals can improve ranking decisions.

An example interaction suggested by the diagnostic work is:

```text
high semantic score
+ large semantic promotion
+ weak rare-term/title evidence
        ↓
potentially risky promotion
```

A small tree-based model can represent interactions like this without manually encoding them as fixed rules.

### Success criterion

The experiment is interesting even if the learned model does not achieve the highest aggregate NDCG.

The central question is whether learned signal combination can improve the quality/robustness tradeoff observed during manual score fusion while generalizing to queries excluded from training.
