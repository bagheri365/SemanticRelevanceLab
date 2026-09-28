# SemanticRelevanceLab

**How should lexical, semantic, and learned relevance signals be
combined for robust search ranking?**

SemanticRelevanceLab is a controlled research project for studying
semantic relevance modeling, neural reranking, and learning-to-rank in
multi-stage search.

The project rule is simple:

> Improve ranking quality, measure query-level failures, and preserve
> the evidence --- including regressions and negative results.

## At a Glance

-   **Research question:** how much does semantic relevance improve over
    lexical retrieval, where do the gains come from, and how should
    relevance signals be combined?
-   **Benchmark:** TREC-COVID via BEIR, with 50 queries and 171,332
    documents
-   **BM25 baseline:** **0.5552 NDCG@10**, **0.7797 MRR**
-   **Cross-encoder reranking:** **0.6750 NDCG@10**, **0.8657 MRR**
-   **Pairwise learned ranker:** **0.6979 NDCG@10**, **0.8907 MRR**
-   **Best pairwise ablation:** **0.7024 NDCG@10**, **0.9170 MRR**
-   **Main pattern:** semantic reranking supplies the largest quality
    gain; learned signal combinations can improve ranking further
-   **Remaining problem:** aggregate improvements still hide substantial
    query-level regressions

Detailed experiment history and diagnostics:
[`docs/research_progress.md`](docs/research_progress.md)

## Why This Project Exists

Search systems rarely rank documents with one relevance signal.

A practical ranking stack may combine:

-   lexical retrieval,
-   semantic similarity,
-   cross-encoder scores,
-   handcrafted relevance features,
-   learned ranking models,
-   multi-stage candidate selection.

When aggregate quality improves, it is easy to miss where the gain came
from or which queries became worse.

SemanticRelevanceLab instead asks:

> Which relevance signals help, how should they be combined, and what
> failure modes remain hidden by aggregate metrics?

The project therefore treats ranking quality and ranking robustness as
separate objects of study.

## Key Findings

### Semantic Reranking Produces a Large Gain

A cross-encoder reranker was applied to the top 100 BM25 candidates.

  System                NDCG@10           MRR     Recall@10
  --------------- ------------- ------------- -------------
  BM25                   0.5552        0.7797        0.0157
  Cross-encoder      **0.6750**    **0.8657**    **0.0181**
  Delta             **+0.1198**   **+0.0859**   **+0.0024**

Recall@100 is unchanged because semantic reranking reorders a fixed BM25
candidate set rather than retrieving new documents.

Some of the largest semantic gains occurred on queries about weather
effects, initial COVID-19 symptoms, SARS-CoV-2 phylogeny, viral
subtypes, and immunity.

### Aggregate Gains Hide Query-Level Regressions

Semantic reranking improved the aggregate result, but not every query.

The strongest observed regressions included queries about:

-   COVID-19 complications associated with diabetes,
-   hand sanitizer requirements,
-   SARS-CoV-2 spike protein structure,
-   longer-term complications after recovery,
-   complications associated with hypertension.

This motivated query-level diagnostics rather than relying only on mean
NDCG.

### Graded Relevance Explains More Than Binary Relevance

Across the top 10 results for all 50 queries:

  Relevance grade     BM25   Semantic
  ----------------- ------ ----------
  0                    187    **123**
  1                     75         73
  2                    238    **304**

The semantic reranker substantially reduces non-relevant documents and
increases highly relevant documents in the top 10 overall.

However, several regression queries show the opposite local pattern:
highly relevant BM25 results are displaced by partially relevant or
non-relevant documents.

### Interpretable Features Reveal Useful Relevance Signals

The candidate-level feature study covers 5,000 BM25 candidates.

Relevant documents show stronger lexical evidence on average:

  Feature                        Relevant   Non-relevant
  ---------------------------- ---------- --------------
  Query-token coverage             0.7088         0.6672
  Title-token coverage             0.3207         0.2486
  Rare-query-term coverage         0.6459         0.5883
  Absolute rank disagreement        26.81          27.88

The promotion analysis also shows that semantic promotion is not
inherently good or bad. Promoted relevant documents tend to have
stronger rare-term and title evidence than promoted non-relevant
documents.

### Pointwise Learning Improves BM25 but Loses Ranking Resolution

A query-level out-of-fold regression tree improved over BM25 but
remained below the semantic reranker:

  System                            NDCG@10          MRR
  ---------------------------- ------------ ------------
  BM25 candidate ranking             0.5684       0.7797
  Semantic candidate ranking     **0.6981**   **0.8657**
  Pointwise regression tree          0.6302       0.8212

The diagnostic exposed severe score quantization:

-   mean unique learned scores among 100 candidates: **7.38**
-   mean largest tie group: **42.9**
-   mean largest tie fraction: **0.429**

This result motivated changing the learning objective rather than simply
increasing tree complexity.

### Pairwise Learning Fixes the Score-Resolution Problem

The pairwise model directly learns within-query document preferences.

  System                         NDCG@10          MRR
  ---------------------------- --------- ------------
  BM25 candidate ranking          0.5684       0.7797
  Semantic candidate ranking      0.6981       0.8657
  Pairwise ranker                 0.6979   **0.8907**

The pairwise model produced **100 unique scores for 100 candidates on
average**, eliminating the coarse score ties observed in the pointwise
tree.

Its learned feature weights also remained directionally stable across
folds for most major signals.

### Simpler Pairwise Feature Sets Can Be Stronger

Feature ablation showed that adding every available feature is not
automatically beneficial.

  Feature set                            NDCG@10          MRR   Regressions vs semantic
  --------------------------------- ------------ ------------ -------------------------
  Semantic only                           0.6981       0.8657                         0
  BM25 only                               0.5684       0.7797                        36
  Scores                                  0.7020       0.9033                        22
  Scores + lexical                        0.6951       0.8737                        23
  Scores + ranks                          0.6978       0.8917                        16
  All features                            0.6979       0.8907                        24
  All minus query-token coverage          0.6954       0.8817                        23
  All minus rare-term coverage            0.6978       0.8745                        27
  **All minus rank disagreement**     **0.7024**   **0.9170**                        21

The best aggregate result so far comes from removing rank disagreement
from the full pairwise feature set.

The broader lesson is that more relevance features do not necessarily
produce a better ranker.

## Benchmark

The initial benchmark is **TREC-COVID** through the BEIR
information-retrieval benchmark.

It provides:

-   a scientific-document corpus,
-   natural-language queries,
-   graded relevance judgments,
-   enough relevant documents per query to study both ranking quality
    and recall.

The working dataset contains:

  Item                                         Count
  ---------------------------------------- ---------
  Documents                                  171,332
  Queries                                         50
  BM25 candidates per query                      100
  Candidate rows used in feature studies       5,000

Two evaluation contexts appear in the project and should not be
confused.

The original retrieval experiment evaluates against the full qrels, so
Recall@100 measures how much relevant material BM25 retrieves from the
corpus.

Later learned-ranking experiments operate only on the fixed 100-document
BM25 candidate sets. In that candidate-set evaluation, Recall@100 is
necessarily 1.0 because the evaluation universe is the candidate set
itself.

## Research Evolution

> retrieve → rerank → diagnose → combine → learn → ablate

  ------------------------------------------------------------------------------
  Stage             Question          Result               Disposition
  ----------------- ----------------- -------------------- ---------------------
  BM25 baseline     How strong is     0.5552 NDCG@10       Retain as
                    lexical                                corpus-level baseline
                    retrieval?                             

  Semantic          Does a            0.6750 NDCG@10       Retain
  reranking         cross-encoder                          
                    improve BM25                           
                    candidates?                            

  Score fusion      Can lexical and   Smooth               Retain as diagnostic
                    semantic scores   quality/robustness   
                    be blended        tradeoff             
                    safely?                                

  Relevance         Which             Lexical coverage     Retain
  features          interpretable     signals are          
                    signals separate  informative          
                    relevant                               
                    candidates?                            

  Promotion         What does         Both useful and      Retain
  analysis          semantic          harmful promotions   
                    reranking promote occur                
                    and demote?                            

  Graded analysis   Are regressions   Several regressions  Retain
                    about relevance   replace grade-2      
                    degree?           results with weaker  
                                      results              

  Pointwise OOF     Can a small       0.6302 NDCG@10;      Retain as
                    learned model     severe score ties    negative/diagnostic
                    combine the                            result
                    signals?                               

  Pairwise OOF      Does learning     0.6979 NDCG@10;      Retain
                    relative          0.8907 MRR           
                    preference                             
                    improve ranking?                       

  Pairwise ablation Which feature     Best: 0.7024         Current best
                    groups actually   NDCG@10; 0.9170 MRR  aggregate result
                    help?                                  
  ------------------------------------------------------------------------------

## Negative Results and Failure Analysis

SemanticRelevanceLab preserves experiments that fail to beat the
strongest baseline.

The pointwise regression-tree experiment is the clearest example.

It improved over BM25, but:

-   regressed against semantic ranking on **28 of 50 queries**,
-   had a worst semantic regression of **-0.4492 NDCG@10**,
-   produced only about seven distinct scores for 100 candidates on
    average.

Those results were not discarded. They led directly to the
ranking-resolution diagnostic and then to the pairwise objective.

The pairwise model solved the score-resolution problem but did not
eliminate robustness failures. It remained almost tied with the semantic
reranker on NDCG@10 and still produced substantial per-query
regressions.

Feature ablation then showed another negative result: the full feature
set was not the best model. Removing rank disagreement improved both
NDCG@10 and MRR.

## Experimental Protocol

The learned-ranking experiments use query-level out-of-fold evaluation.

``` text
split query IDs into folds
        ↓
train only on training queries
        ↓
score candidates for held-out queries
        ↓
reconstruct held-out rankings
        ↓
evaluate ranking metrics
```

Every query appears in exactly one held-out fold.

This prevents candidates from the same query from appearing in both the
training and evaluation partitions.

The principal metrics are:

-   NDCG@10,
-   MRR,
-   Recall@10,
-   query-level regressions,
-   worst per-query regression,
-   score resolution.

Aggregate metrics are always interpreted alongside per-query behavior.

## Reproducibility

Experiments write structured JSON artifacts under `artifacts/results/`.

Important outputs include:

``` text
artifacts/results/
├── bm25.json
├── bm25_analysis.json
├── semantic_rerank.json
├── score_fusion.json
├── relevance_features.json
├── promotion_analysis.json
├── graded_relevance_analysis.json
├── learned_relevance_cv_manifest.json
├── learned_relevance_oof.json
├── learned_relevance_failure_analysis.json
├── learned_ranking_diagnostics.json
├── pairwise_relevance_oof.json
└── pairwise_feature_ablation.json
```

The repository also keeps the experiment logic and tests separate from
generated artifacts so that results can be reproduced without committing
the local dataset or model downloads.

## Repository Structure

``` text
SemanticRelevanceLab/
├── docs/
│   └── research_progress.md
├── src/semantic_relevance/
│   ├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── features/
│   ├── learning/
│   ├── ranking/
│   └── retrieval/
├── tests/
├── artifacts/              # local generated experiment outputs
├── data/                   # local benchmark data
├── pyproject.toml
└── README.md
```

## Running Locally

Create a Python 3.12 environment and install the project:

``` bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the test suite:

``` bash
pytest
```

Run the BM25 baseline:

``` bash
python -m semantic_relevance.experiments.bm25 \
  --data data/trec-covid \
  --output artifacts/results/bm25.json
```

Run semantic reranking:

``` bash
python -m semantic_relevance.experiments.semantic_rerank \
  --data data/trec-covid \
  --output artifacts/results/semantic_rerank.json
```

Run pairwise out-of-fold ranking:

``` bash
python -m semantic_relevance.experiments.pairwise_relevance_oof \
  --input artifacts/results/relevance_features.json \
  --output artifacts/results/pairwise_relevance_oof.json
```

Run pairwise feature ablation:

``` bash
python -m semantic_relevance.experiments.pairwise_feature_ablation \
  --input artifacts/results/relevance_features.json \
  --output artifacts/results/pairwise_feature_ablation.json
```

## Experimental Discipline

The intended discipline is:

-   establish a lexical baseline before adding learned components,
-   hold candidate sets fixed when comparing rerankers,
-   split learned experiments by query rather than candidate,
-   preserve graded relevance,
-   report aggregate and per-query metrics together,
-   inspect regressions rather than only wins,
-   preserve negative experiments,
-   prefer interpretable experiments before adding model complexity,
-   change one major modeling assumption at a time.

A higher aggregate NDCG is not treated as evidence that every query
improved.

## Limitations

-   The current study uses one benchmark: TREC-COVID.
-   The corpus and query domain are biomedical.
-   Semantic experiments currently use one primary cross-encoder
    configuration.
-   Reranking is limited by the BM25 candidate set.
-   The benchmark has only 50 queries, so query-level conclusions have
    limited statistical power.
-   Learned models use a deliberately small feature space.
-   Candidate-set metrics are not directly interchangeable with the
    original full-qrels retrieval metrics.
-   Aggregate gains still coexist with meaningful query-level
    regressions.
-   External-domain replication has not yet been performed.

The current results are evidence about ranking behavior in this
controlled benchmark, not a universal ranking of retrieval or
learning-to-rank methods.

## What the Experiments Suggest

The evidence so far supports a division of labor:

-   **BM25** provides inexpensive lexical candidate generation,
-   **cross-encoder scoring** supplies the largest semantic quality
    improvement,
-   **interpretable lexical features** provide useful complementary
    evidence,
-   **pairwise learning** is better suited to fine-grained ranking than
    the tested pointwise tree,
-   **feature selection matters** because redundant or noisy ranking
    signals can reduce quality.

A working design hypothesis from SemanticRelevanceLab is:

> Use lexical retrieval for candidate generation, semantic models for
> strong relevance judgments, and learned ranking only when its
> additional signals are validated out-of-fold and at the query level.

## Future Research

-   investigate the remaining pairwise regressions,
-   test regularization and alternative pair sampling,
-   evaluate stronger pairwise or listwise objectives,
-   study query-dependent gating between semantic and learned ranking,
-   test whether robustness can improve without sacrificing aggregate
    NDCG,
-   expand semantic and lexical feature families carefully,
-   evaluate inference-cost versus relevance-quality tradeoffs,
-   replicate the study on additional BEIR datasets,
-   test multi-stage ranking cascades,
-   investigate hard-negative mining.

## Milestones

  Milestone                                 Status
  ----------------------------------------- ----------
  Evaluation harness                        Complete
  BM25 baseline                             Complete
  Query-level BM25 error analysis           Complete
  Cross-encoder semantic reranking          Complete
  Semantic failure analysis                 Complete
  Lexical/semantic score fusion             Complete
  Interpretable relevance features          Complete
  Promotion and graded-relevance analysis   Complete
  Query-level CV infrastructure             Complete
  Pointwise learned relevance               Complete
  Ranking-resolution diagnostics            Complete
  Pairwise learning-to-rank                 Complete
  Pairwise feature ablation                 Complete
  Multi-stage cascade experiments           Next
  Hard-negative experiments                 Planned
  Inference-efficiency experiments          Planned

## Current Conclusion

SemanticRelevanceLab asks:

> How should lexical, semantic, and learned relevance signals be
> combined for robust search ranking?

The current evidence says:

> **Semantic relevance is the strongest individual improvement, but
> combining signals well is a ranking problem of its own.**

BM25 remains a useful candidate generator. Cross-encoder reranking
produces a large relevance gain. Pairwise learning can match or slightly
exceed semantic ranking in aggregate while improving MRR, but learned
combinations still introduce query-level regressions.

The most useful result so far is therefore not simply a higher NDCG
number. It is a clearer experimental picture of where semantic ranking
helps, where learned combinations fail, and why ranking robustness must
be measured alongside average quality.
