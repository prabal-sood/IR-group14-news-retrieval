# Engine A — BM25 (Lexical Retrieval)

Classic lexical retrieval using Okapi BM25 over the shared MIND news corpus.
Part of Group 14's BM25 vs. Neural comparative study (CS F469/U469).

## Files
- `engine_a_bm25.ipynb` — full pipeline: load corpus, preprocess, build index,
  tune parameters, run test queries

## Input
Expects `final_news_corpus.csv` (from `dataset/`) in the same working directory —
10,000 articles with `news_id`, `category`, `subcategory`, `title`, `abstract`,
`document_text`.

## Pipeline
1. **Preprocessing** — lowercase, regex tokenisation, stopword removal, Porter
   stemming. Negation words ("not", "no", "nor") are explicitly excluded from the
   stopword list, since standard lists strip them by default — without this, BM25
   would never see the negation cue in a query.
2. **Indexing** — inverted index built with `rank_bm25` (`BM25Okapi`).
3. **Ranking** — k1 = 1.5, b = 0.75, tuned on this corpus via a parameter sweep
   rather than left at library defaults.
4. **Testing** — 8 vocabulary-mismatch test queries (shared with Engine B for direct
   comparison), plus a dedicated negation sanity check.

## Key design decisions
- **No chunking needed** — documents are title + abstract only (MINDsmall doesn't
  ship full article bodies), typically under 100 words, so no length handling
  required.
- **No from-scratch BM25 scorer or TF-IDF baseline** — considered in the original
  proposal, dropped as out of scope for this comparative study; `rank_bm25` alone
  is used.

## Status
Built and tested on the real 10,000-document corpus. Results feed into the combined
evaluation in `eval/`.
