
import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class NeuralRetrievalEngine:

    def __init__(self, data_path, embedding_dir,
                 model_name="all-MiniLM-L6-v2"):

        self.df = pd.read_csv(data_path).fillna("")
        self.model = SentenceTransformer(model_name)

        self.embeddings = np.load(
            os.path.join(embedding_dir, "document_embeddings.npy")
        )

        self.document_ids = np.load(
            os.path.join(embedding_dir, "document_ids.npy"),
            allow_pickle=True
        ).astype(str)

        assert len(self.df) == len(self.embeddings)
        assert np.array_equal(
            self.df["news_id"].astype(str).to_numpy(),
            self.document_ids
        ), "Document ID alignment mismatch"


        # Load saved long-document chunks
        chunk_path = os.path.join(
            embedding_dir, "long_document_chunks.npz"
        )

        self.long_chunks = {}

        if os.path.exists(chunk_path):
            with np.load(chunk_path) as saved:
                for idx in saved.files:
                    vectors = saved[idx].astype(np.float32)
                    norms = np.linalg.norm(
                        vectors, axis=1, keepdims=True
                    )
                    self.long_chunks[int(idx)] = (
                        vectors / np.maximum(norms, 1e-12)
                    )

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        self.embeddings = self.embeddings / np.maximum(norms, 1e-12)

    def search(self, query, top_k=10):

        start = time.perf_counter()

        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        scores = self.embeddings @ query_vector


        # Score long documents using their best matching chunk
        for idx, chunk_vectors in self.long_chunks.items():
            scores[idx] = np.max(chunk_vectors @ query_vector)

        top_k = min(top_k, len(scores))
        indices = np.argsort(-scores)[:top_k]

        results = self.df.iloc[indices][
            ["news_id", "title", "category"]
        ].copy()

        results["similarity_score"] = scores[indices]
        results["rank"] = range(1, top_k + 1)

        latency_ms = (time.perf_counter() - start) * 1000

        return results.reset_index(drop=True), latency_ms
