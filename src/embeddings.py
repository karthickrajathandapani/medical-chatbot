"""
embeddings.py
=============
# PURPOSE (one file, one purpose):
#   Turn text into dense vectors (the "semantic meaning" half of hybrid
#   search). This is the ONLY file that loads/calls the embedding model.

# WHY "pritamdeka/S-PubMedBert-MS-MARCO" IS THE BEST EMBEDDING MODEL HERE:
#   - It's a Sentence-Transformers model (requirement #3), so it drops
#     straight into the .encode() API used everywhere below.
#   - Base model is PubMedBERT, pretrained on 14M+ PubMed abstracts —
#     it already "speaks" medical vocabulary (drug names, symptoms,
#     anatomy) far better than a general-purpose model like MiniLM.
#   - Fine-tuned with an MS-MARCO retrieval objective, so its embeddings
#     are optimized specifically for the "does this chunk answer this
#     question" task — not just generic sentence similarity.
#   A generic model (all-MiniLM-L6-v2) is faster/smaller but noticeably
#   weaker at distinguishing similar-sounding medical terms.
"""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from config.config import settings


class EmbeddingModel:
    """Thin, cached wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        # Loaded once per process — loading a transformer is the
        # slowest part of startup, so we never reload it per-request.
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of chunks for ingestion (no cache — always fresh)."""
        vectors = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # cosine-ready + needed for dotproduct hybrid math
        )
        return vectors.tolist()

    @lru_cache(maxsize=256)
    def embed_query_cached(self, text: str) -> tuple:
        """
        Embed a single user query, with an LRU cache.
        # SPEED (requirement #14): repeated/similar questions in a chat
        # session (e.g. follow-ups that repeat a term) skip re-embedding.
        Returns a tuple (hashable, so it's cache-friendly) instead of list.
        """
        vector = self._model.encode(text, normalize_embeddings=True)
        return tuple(vector.tolist())

    def embed_query(self, text: str) -> List[float]:
        """Public helper returning a plain list, for callers that need one."""
        return list(self.embed_query_cached(text))


# Singleton — import this everywhere instead of constructing a new model.
embedding_model = EmbeddingModel()
