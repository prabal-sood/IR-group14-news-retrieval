# News Retrieval: BM25 vs. Neural Search under Vocabulary Mismatch

CS F469/U469 Information Retrieval — Group 14, BITS Pilani

## Team
- Prabal Sood (2023A7PS0680P)
- Diveel Agrawal (2023A7PS0676P)
- G. Siddartha Reddy (2023A7PS0598P)
- Pranjali Gupta (2023A7PS0563P)

## Project
Comparing a classic lexical retrieval engine (BM25) against a neural dense-retrieval
engine (sentence-transformer embeddings + cosine similarity) on the MIND news dataset,
to diagnose exactly where and why each engine fails under vocabulary mismatch.

## Structure
- `dataset/` — MINDsmall download, parsing, and stratified sampling (10,000 articles,
  18 categories) into `final_news_corpus.csv`
- `engine_a/` — BM25 lexical retrieval engine (rank_bm25, negation-preserving
  preprocessing, k1/b tuning)
- `engine_b/` — Neural dense retrieval engine (MiniLM embeddings, cosine similarity)
- `eval/` — Combined preliminary evaluation notebook, relevance judgments, and results
  comparing both engines

## Status
Dataset prepared, both engines built and tested on the real corpus. Preliminary
evaluation complete on 8 queries with group relevance judging (Precision@5, Recall@5,
F1, MAP, NDCG@5 computed for both engines). Intermediate status update submitted
25 Oct 2026.

Remaining for final submission: full 20-query typed suite, two-annotator graded
(0/1/2) relevance judging, per-category breakdown, and failure-mode analysis.
Due mid-November 2026.
