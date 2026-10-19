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
- `dataset/` — MINDsmall download, parsing, and subsample preparation
- `engine_a/` — BM25 lexical retrieval engine
- `engine_b/` — Neural dense retrieval engine
- `eval/` — Evaluation scripts and results (NDCG@5, MAP, etc.)
