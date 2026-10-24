# Engine B — Neural Dense Retrieval System

**Course:** Information Retrieval  
**Project:** Comparative Study: BM25 vs. Neural Dense Retrieval  
**Model:** `sentence-transformers/all-MiniLM-L6-v2` (zero-shot; no fine-tuning)

## Overview

Engine B is a neural dense retrieval implementation for searching news articles. It encodes documents and queries into dense vectors with Sentence Transformers and ranks documents by cosine similarity. The implementation supports individual and batched queries, compressed index caching, and a command-line demonstration.

## Input corpus

The `load_mind_corpus(csv_path)` function reads `final_news_corpus.csv`, produced by `IR14.ipynb`. The CSV must contain:

- `news_id`
- `title`
- `abstract`
- `document_text`

The loader also retains `category` and `subcategory` in its returned document records when those fields are present. Missing abstracts and document text are filled with empty strings. Text is cleaned with Unicode NFKC normalization, HTML tag removal, and whitespace normalization. `news_id` is renamed to `id` for engine use. The source comments note that 526 articles lack abstracts and are handled with title-only embeddings.

## Requirements

Install the Python dependencies used by the module:

```bash
pip install numpy pandas sentence-transformers
```

Sentence Transformers also installs its own underlying model dependencies.

## How to run

Place the Python implementation and `final_news_corpus.csv` in the working directory. The module's `__main__` block uses:

```python
CORPUS_CSV = "final_news_corpus.csv"
CACHE_DIR = "./cache"
TOP_K = 5
```

Run the module using its actual filename, for example:

```bash
python engine_b.py
```

On the first run, the engine builds and saves the index. Later runs restore it from the cache unless rebuilding is requested.

## Retrieval workflow

1. **Load documents:** Read and clean the MIND news corpus.
2. **Initialize model:** Load `all-MiniLM-L6-v2` with configurable `cache_dir`, `batch_size` (default 128), and device.
3. **Build index:** Encode `document_text` (or combine title and abstract when no text field is specified).
4. **Normalize vectors:** Produce unit-L2-normalized document embeddings, stored as `float32`.
5. **Cache index:** Save document IDs and the embedding matrix in a compressed `.npz` file in `./cache`.
6. **Search:** Encode the query, compute similarities, and return the highest-scoring document IDs.

For unit-normalized vectors, cosine similarity equals their dot product:

```text
cosine_similarity(query, document) = query_vector · document_vector
```

Single-query scores are calculated with `doc_matrix @ query_vector`. Batched scores are calculated with `query_matrix @ doc_matrix.T`. Top-K selection uses `numpy.argpartition`, followed by sorting the selected candidates by score.

## Main API

```python
from engine_b import NeuralRetrievalEngine, load_mind_corpus

documents = load_mind_corpus("final_news_corpus.csv")
engine = NeuralRetrievalEngine(
    model_name="all-MiniLM-L6-v2",
    cache_dir="./cache",
    batch_size=128,
)
engine.build_index(documents, text_field="document_text")

results = engine.search("severe storm causes property destruction", top_k=5)
for document_id, cosine_score in results:
    print(document_id, cosine_score)

batch_results = engine.search_batch([
    "professional athlete switches teams",
    "retirement savings and personal wealth",
], top_k=5)
```

Additional methods and properties include `corpus_stats()`, `clear_cache()`, `is_ready`, `corpus_size`, `embedding_dim`, and `index_size_mb`. Use `build_index(..., overwrite=True)` to rebuild an existing index.

## Demonstration queries

The module includes eight vocabulary-mismatch queries with expected categories:

| Query | Expected category |
|---|---|
| professional athlete switches teams | sports |
| government spending and economic downturn | news |
| retirement savings and personal wealth | finance |
| severe storm causes property destruction | weather |
| restaurant dining and culinary trends | foodanddrink |
| road trip and outdoor adventure destinations | travel |
| car model release with advanced technology | autos |
| patient treatment and hospital procedures | health |

The demonstration runs each query with `top_k=5`, prints document IDs, cosine scores, article snippets, and category-match indicators, and then performs a batched search over the same eight queries.

## Index persistence and checks

The compressed cache contains `doc_ids` and `doc_matrix`. A new engine instance automatically loads this cache when it exists. The demonstration checks that the restored engine is ready and that its corpus size is **10,000 documents**.

`corpus_stats()` reports the model, corpus size, embedding dimension, index memory, cache path, readiness, and available index-build metadata such as encoding time and throughput.

## Evaluation scope

The provided Python file demonstrates search results, expected-category indicators, batched-query execution time, and index statistics. It **does not calculate** Precision@K, Recall@K, F1, MAP, NDCG, or a measured BM25 comparison. Those results should not be inferred from this implementation alone.

## Notes

- The model is used zero-shot, without fine-tuning.
- Retrieval scores are cosine similarities, not relevance labels.
- The cache filename is derived from the model name and ends in `_index.npz`.
- The `__main__` demonstration expects a 10,000-document corpus for its cache-persistence assertion.
