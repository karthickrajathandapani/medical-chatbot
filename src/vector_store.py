"""
vector_store.py
================
# PURPOSE (one file, one purpose):
#   Own ALL communication with Pinecone — creating the index, upserting
#   hybrid (dense + sparse) vectors, and running hybrid queries.
#   No other file should import the `pinecone` package directly.

# WHY PINECONE (requirement #4):
#   Fully-managed, low-latency, serverless vector DB with native support
#   for combining dense + sparse vectors in ONE query (hybrid search)
#   using the "dotproduct" metric — exactly what requirement #6 needs.

# WHY HYBRID SEARCH (requirement #6 — "best retrieval search method"):
#   alpha-weighted combination of dense (semantic) + sparse (BM25/exact
#   keyword) scores consistently outperforms either method alone on
#   domain-specific corpora like medicine, per Pinecone's own hybrid
#   search benchmarks — dense alone misses exact terms, sparse alone
#   misses paraphrased questions.
"""

import time
from typing import Dict, List

from pinecone import Pinecone, ServerlessSpec

from config.config import settings
from src.chunking import Chunk


class VectorStore:
    def __init__(self):
        self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._index = None  # lazy-connected in connect()

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------
    def create_index_if_missing(self) -> None:
        existing = [idx["name"] for idx in self._pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in existing:
            self._pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.EMBEDDING_DIM,
                metric=settings.PINECONE_METRIC,  # "dotproduct" -> required for hybrid
                spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
            )
        self._wait_until_ready()
        self.connect()

    def _wait_until_ready(self, timeout_seconds: int = 90) -> None:
        """
        A freshly-created serverless index isn't immediately queryable —
        its host needs a few seconds to come online. Without this wait,
        the very first upsert/query right after create_index() can fail
        with a confusing connection error ("Remote end closed connection",
        "Failed to connect; did you specify the correct index name?").
        We poll Pinecone's own `describe_index` status until it reports
        ready, instead of guessing with a fixed sleep.
        """
        print("      Waiting for index to finish spinning up...")
        start = time.time()
        while time.time() - start < timeout_seconds:
            desc = self._pc.describe_index(settings.PINECONE_INDEX_NAME)
            if desc.get("status", {}).get("ready"):
                print(f"      Index ready after {round(time.time() - start, 1)}s.")
                return
            time.sleep(2)
        raise TimeoutError(
            f"Pinecone index '{settings.PINECONE_INDEX_NAME}' did not become ready "
            f"within {timeout_seconds}s. Check its status at https://app.pinecone.io"
        )

    def connect(self) -> None:
        self._index = self._pc.Index(settings.PINECONE_INDEX_NAME)

    def _require_connected(self):
        if self._index is None:
            self.connect()

    # ------------------------------------------------------------------
    # Ingestion (upsert)
    # ------------------------------------------------------------------
    def upsert_hybrid(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict],
        batch_size: int = 100,
    ) -> int:
        """
        Push chunks + their dense & sparse vectors to Pinecone in batches.
        Returns the number of vectors upserted.
        """
        self._require_connected()
        records = []
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
            records.append(
                {
                    "id": chunk.chunk_id,
                    "values": dense,
                    "sparse_values": sparse,
                    "metadata": {
                        "text": chunk.text,
                        "source_file": chunk.source_file,
                        "page_number": chunk.page_number,
                    },
                }
            )

        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self._index.upsert(vectors=batch)
            total += len(batch)
        return total

    # ------------------------------------------------------------------
    # Retrieval (hybrid query)
    # ------------------------------------------------------------------
    @staticmethod
    def _scale_hybrid(dense: List[float], sparse: Dict, alpha: float):
        """
        Apply Pinecone's recommended convex-combination scaling so alpha
        actually controls the dense/sparse balance at query time:
            alpha=1.0 -> pure dense, alpha=0.0 -> pure sparse
        """
        sparse_scaled = {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        }
        dense_scaled = [v * alpha for v in dense]
        return dense_scaled, sparse_scaled

    def hybrid_query(
        self,
        dense_vector: List[float],
        sparse_vector: Dict,
        top_k: int = settings.TOP_K_RETRIEVE,
        alpha: float = settings.HYBRID_ALPHA,
    ) -> List[Dict]:
        """
        Run a hybrid (dense + sparse) similarity search.

        Returns a list of {text, source_file, page_number, score} dicts,
        ordered by Pinecone's blended relevance score (highest first).
        """
        self._require_connected()
        dense_scaled, sparse_scaled = self._scale_hybrid(dense_vector, sparse_vector, alpha)

        result = self._index.query(
            vector=dense_scaled,
            sparse_vector=sparse_scaled,
            top_k=top_k,
            include_metadata=True,
        )

        matches = []
        for match in result["matches"]:
            meta = match["metadata"]
            matches.append(
                {
                    "text": meta["text"],
                    "source_file": meta["source_file"],
                    "page_number": meta["page_number"],
                    "score": match["score"],
                }
            )
        return matches


# Singleton
vector_store = VectorStore()
