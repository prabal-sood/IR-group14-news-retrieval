# Engine B - Neural Dense Retrieval

## Overview
Engine B is a semantic news retrieval engine built
using Sentence Transformers.

## Model
- Model: all-MiniLM-L6-v2
- Embedding dimensions: 384
- Retrieval method: Cosine similarity
- Dataset: MIND news dataset
- Indexed documents: 10,000

## Features
- Semantic document retrieval
- Top-K ranked search results
- Cosine similarity scoring
- Long-document chunk embeddings
- Query latency measurement
- Batch retrieval
- Saved embeddings and document IDs
- Standalone Python retrieval module

## Performance
- Average query latency: 10.55 ms
- Median query latency: 10.31 ms
- Minimum query latency: 9.37 ms

Measured on 10 test queries in Google Colab.
Performance may vary by environment.

## Files
- engine_b.py: Standalone retrieval module
- document_ids.npy: Document ID mapping
- long_document_chunks.npz: Long-document embeddings
- engine_b_batch_results.csv: Batch retrieval results
- engine_b_final_latency.csv: Query latency benchmark
- engine_b_final_summary.csv: Performance summary

## Usage
Load the standalone engine_b.py module and initialize
it with the saved embeddings and document metadata.

Call the engine's search method with a text query
and the desired top_k value.

## Dependencies
- numpy
- pandas
- sentence-transformers
- torch
- scikit-learn
