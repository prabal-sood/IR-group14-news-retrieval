"""
Engine B — Neural Dense Retrieval System
Course   : Information Retrieval | Comparative Study: BM25 vs. Neural Dense Retrieval
Model    : sentence-transformers/all-MiniLM-L6-v2 (zero-shot, no fine-tuning)
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

def load_mind_corpus(csv_path: str) -> List[Dict[str, str]]:
    """
    Load final_news_corpus.csv (IR14.ipynb output) and return a list of dicts
    with keys: id, category, subcategory, title, abstract, document_text.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Corpus CSV not found at '{csv_path}'. Run IR14.ipynb first."
        )

    df = pd.read_csv(csv_path, dtype=str)

    required_cols = {"news_id", "title", "abstract", "document_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    # 526 articles have no abstract — fill with empty string
    null_count = df["abstract"].isna().sum()
    df["abstract"]      = df["abstract"].fillna("")
    df["document_text"] = df["document_text"].fillna("")

    if null_count > 0:
        logger.warning(
            "%d articles (%.2f%%) have no abstract — title-only embeddings.",
            null_count, null_count / len(df) * 100,
        )

    # Defensive clean: NFKC normalisation + strip residual HTML
    _html = re.compile(r"<[^>]+>")

    def _clean(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = _html.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()

    for col in ("title", "abstract", "document_text"):
        df[col] = df[col].apply(_clean)

    # Rename news_id → id for engine compatibility
    df = df.rename(columns={"news_id": "id"})

    logger.info(
        "Corpus loaded — %d docs | %d categories | %d title-only",
        len(df), df["category"].nunique(), null_count,
    )
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# NeuralRetrievalEngine
# ---------------------------------------------------------------------------

class NeuralRetrievalEngine:
    """
    Encodes documents into L2-normalised dense vectors and ranks them against
    a query using cosine similarity computed as a BLAS dot-product.

    When every vector has unit L2-norm (‖v‖ = 1):
        cosine_sim(u, v) = u · v
    Scoring all N docs for one query:
        scores = doc_matrix @ query_vec          shape (N,)   — one BLAS call
    Scoring Q queries at once (batch):
        score_matrix = query_matrix @ doc_matrix.T  shape (Q, N) — one BLAS call
    Top-K extraction uses np.argpartition (O(N)) + argsort on K items (O(k log k)),
    avoiding the O(N log N) cost of a full sort.
    """

    _CACHE_SUFFIX = "_index.npz"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "./cache",
        batch_size: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir  = Path(cache_dir)
        self.batch_size = batch_size

        _safe = model_name.replace("/", "_").replace("-", "_")
        self.cache_path: Path = self.cache_dir / f"{_safe}{self._CACHE_SUFFIX}"

        self.doc_ids   : Optional[np.ndarray] = None  # shape (N,)
        self.doc_matrix: Optional[np.ndarray] = None  # shape (N, D) float32
        self._build_meta: Dict = {}

        logger.info("Loading model: '%s'", model_name)
        self.model = SentenceTransformer(model_name, device=device)

        if self.cache_path.is_file():
            logger.info("Cache found — restoring index.")
            self._load_index()
        else:
            logger.info("No cache found. Call build_index() to create one.")

    # ------------------------------------------------------------------
    # Index Construction
    # ------------------------------------------------------------------

    def build_index(
        self,
        documents: List[Dict[str, str]],
        text_field: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Encode corpus, L2-normalise every vector, and persist as .npz cache.

        Args:
            documents  : List of dicts with 'id' and either a text_field key
                         or both 'title' and 'abstract'.
            text_field : Pre-computed text column to encode (e.g. 'document_text').
                         If None, encodes f"{title} {abstract}".
            overwrite  : Force rebuild even if a cache exists.
        """
        if self.cache_path.is_file() and not overwrite:
            logger.info("Cache exists. Pass overwrite=True to rebuild.")
            self._load_index()
            return

        n_docs = len(documents)
        logger.info("Building index for %d documents ...", n_docs)

        required = {"id", text_field} if text_field else {"id", "title", "abstract"}
        for i, doc in enumerate(documents):
            missing = required - doc.keys()
            if missing:
                raise ValueError(f"Document[{i}] missing keys: {missing}")

        article_ids = [str(doc["id"]) for doc in documents]

        if text_field:
            texts = [str(doc[text_field]) for doc in documents]
        else:
            texts = [f"{doc['title']} {doc['abstract']}" for doc in documents]

        # Warn if a significant portion of docs are title-only (≤12 words)
        title_only = sum(1 for t in texts if len(t.split()) <= 12)
        if title_only:
            logger.warning("%d docs are title-only (≤12 words).", title_only)

        t0 = time.perf_counter()
        # normalize_embeddings=True ensures ‖v‖₂ = 1 for every row,
        # enabling cosine similarity = dot product at query time.
        embeddings: np.ndarray = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        elapsed = time.perf_counter() - t0

        self.doc_ids    = np.array(article_ids, dtype=str)
        self.doc_matrix = embeddings.astype(np.float32)  # float32 saves memory vs float64

        self._build_meta = {
            "n_docs"       : n_docs,
            "embedding_dim": self.doc_matrix.shape[1],
            "title_only"   : title_only,
            "encode_time_s": round(elapsed, 2),
            "throughput_dps": round(n_docs / elapsed),
        }

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            doc_ids   =self.doc_ids,
            doc_matrix=self.doc_matrix,
        )
        logger.info(
            "Index cached → '%s' | shape: %s | %.1fs",
            self.cache_path, self.doc_matrix.shape, elapsed,
        )

    def _load_index(self) -> None:
        """Restore doc_ids and doc_matrix from the .npz cache file."""
        if not self.cache_path.is_file():
            raise FileNotFoundError(f"Cache not found: '{self.cache_path}'.")

        data = np.load(self.cache_path, allow_pickle=False)
        self.doc_ids    = data["doc_ids"]
        self.doc_matrix = data["doc_matrix"]

        logger.info(
            "Index loaded — %d docs | dim: %d | dtype: %s",
            len(self.doc_ids), self.doc_matrix.shape[1], self.doc_matrix.dtype,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Encode a single query and return the top-K most similar documents.

        Returns:
            List of (doc_id, cosine_score) tuples, sorted descending.
        """
        self._check_ready()
        if not query.strip():
            raise ValueError("Query must not be empty.")

        actual_k = self._clamp_k(top_k)

        # Encode query to unit-norm vector (D,)
        q_vec: np.ndarray = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # (N, D) @ (D,) → (N,)  — cosine similarity for all docs in one call
        scores: np.ndarray = self.doc_matrix @ q_vec

        return self._top_k_results(scores, actual_k)

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        Encode all queries in one forward pass and rank all simultaneously.
        Preferred method for evaluation — far more efficient than looping search().

        Returns:
            One result list per query, each sorted descending by cosine score.
        """
        self._check_ready()
        if not queries:
            return []

        actual_k = self._clamp_k(top_k)

        # Encode all Q queries at once: (Q, D), unit-norm rows
        Q_mat: np.ndarray = self.model.encode(
            queries,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        # (Q, D) @ (D, N) → (Q, N)  — all query-document scores in one BLAS call
        score_matrix: np.ndarray = Q_mat @ self.doc_matrix.T

        return [self._top_k_results(row, actual_k) for row in score_matrix]

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _check_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError("Index not loaded. Call build_index() first.")

    def _clamp_k(self, top_k: int) -> int:
        n = len(self.doc_ids)
        if top_k > n:
            logger.warning("top_k=%d clamped to corpus size=%d.", top_k, n)
            return n
        return top_k

    def _top_k_results(
        self, scores: np.ndarray, actual_k: int
    ) -> List[Tuple[str, float]]:
        # argpartition: O(N) — finds K largest without full sort
        # argsort on K candidates: O(k log k)
        top_idx = np.argpartition(scores, -actual_k)[-actual_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(str(self.doc_ids[i]), float(scores[i])) for i in top_idx]

    # ------------------------------------------------------------------
    # Properties & Utilities
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self.doc_matrix is not None and self.doc_ids is not None

    @property
    def corpus_size(self) -> int:
        return len(self.doc_ids) if self.doc_ids is not None else 0

    @property
    def embedding_dim(self) -> int:
        return self.doc_matrix.shape[1] if self.doc_matrix is not None else 0

    @property
    def index_size_mb(self) -> float:
        if self.doc_matrix is None:
            return 0.0
        return round(self.doc_matrix.nbytes / (1024 ** 2), 2)

    def corpus_stats(self) -> Dict:
        """Return index metadata useful for the report's methodology section."""
        return {
            "model"        : self.model_name,
            "corpus_size"  : self.corpus_size,
            "embedding_dim": self.embedding_dim,
            "index_size_mb": self.index_size_mb,
            "cache_path"   : str(self.cache_path),
            "index_ready"  : self.is_ready,
            **self._build_meta,
        }

    def clear_cache(self) -> None:
        """Delete the .npz cache file and reset the in-memory index."""
        if self.cache_path.is_file():
            self.cache_path.unlink()
            logger.info("Cache deleted: %s", self.cache_path)
        self.doc_ids     = None
        self.doc_matrix  = None
        self._build_meta = {}

    def __repr__(self) -> str:
        status = (
            f"{self.corpus_size} docs | dim={self.embedding_dim} | {self.index_size_mb} MB"
            if self.is_ready else "index not loaded"
        )
        return f"NeuralRetrievalEngine(model='{self.model_name}', {status})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    CORPUS_CSV = "final_news_corpus.csv"
    CACHE_DIR  = "./cache"
    TOP_K      = 5

    SEP  = "=" * 68
    SEP2 = "-" * 68

    print(f"\n{SEP}")
    print("  ENGINE B — Neural Dense Retrieval")
    print(f"{SEP}\n")

    # 1. Load corpus
    documents  = load_mind_corpus(CORPUS_CSV)
    doc_lookup = {doc["id"]: doc for doc in documents}

    # 2. Initialise engine
    engine = NeuralRetrievalEngine(
        model_name="all-MiniLM-L6-v2",
        cache_dir=CACHE_DIR,
        batch_size=128,
    )

    # 3. Build index (first run ~25s CPU; subsequent runs restore from cache)
    engine.build_index(documents, text_field="document_text", overwrite=False)

    # 4. Corpus statistics
    print(f"\n{SEP2}")
    print("  CORPUS & INDEX STATISTICS")
    print(SEP2)
    for k, v in engine.corpus_stats().items():
        print(f"  {k:<22} {v}")

    # 5. Vocabulary-mismatch queries
    # Queries deliberately avoid exact corpus terms to stress-test semantic retrieval
    single_queries = [
        ("professional athlete switches teams",          "sports"),
        ("government spending and economic downturn",    "news"),
        ("retirement savings and personal wealth",       "finance"),
        ("severe storm causes property destruction",     "weather"),
        ("restaurant dining and culinary trends",        "foodanddrink"),
        ("road trip and outdoor adventure destinations", "travel"),
        ("car model release with advanced technology",   "autos"),
        ("patient treatment and hospital procedures",    "health"),
    ]

    print(f"\n{SEP2}")
    print(f"  SINGLE-QUERY SEARCH  (top-{TOP_K}, vocabulary mismatch tests)")
    print(SEP2)

    for query, expected_cat in single_queries:
        print(f'\n  Query    : "{query}"')
        print(f"  Expected : {expected_cat}")
        print(f"  {'Rank':<6}{'doc_id':<12}{'cosine':>8}  document_text (first 80 chars)")
        print(f"  {'----':<6}{'------':<12}{'------':>8}  " + "-" * 40)

        for rank, (doc_id, score) in enumerate(engine.search(query, top_k=TOP_K), 1):
            doc  = doc_lookup.get(doc_id, {})
            cat  = doc.get("category", "?")
            text = doc.get("document_text", "")[:80]
            flag = " ✓" if cat == expected_cat else ""
            print(f"  [{rank}]   {doc_id:<12}{score:>8.4f}  [{cat}]{flag}  {text}")

    # 6. Batch search
    print(f"\n{SEP2}")
    print("  BATCH SEARCH  (all queries encoded in one forward pass)")
    print(SEP2)

    batch_queries = [q for q, _ in single_queries]
    t0 = time.perf_counter()
    batch_results = engine.search_batch(batch_queries, top_k=TOP_K)
    elapsed = time.perf_counter() - t0

    print(f"\n  {len(batch_queries)} queries ranked in {elapsed:.3f}s")
    print(f"\n  {'Query':<46} {'Top-1 ID':<12} {'Score':>7}  Category")
    print(f"  {'-'*46} {'-'*12} {'-'*7}  --------")
    for query, results in zip(batch_queries, batch_results):
        if results:
            doc_id, score = results[0]
            cat = doc_lookup.get(doc_id, {}).get("category", "?")
            print(f"  {query[:45]:<46} {doc_id:<12} {score:>7.4f}  {cat}")

    # 7. Cache persistence check
    print(f"\n{SEP2}")
    print("  CACHE PERSISTENCE CHECK")
    print(SEP2)
    engine2 = NeuralRetrievalEngine(model_name="all-MiniLM-L6-v2", cache_dir=CACHE_DIR)
    assert engine2.is_ready,              "Index failed to restore from cache."
    assert engine2.corpus_size == 10_000, "Corpus size mismatch after reload."
    print(f"\n  {engine2}")
    print(f"\n{SEP}")
    print("  Engine B ready for integration with BM25 comparative study.")
    print(f"{SEP}\n")
