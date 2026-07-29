"""
reranker.py
===========
# PURPOSE (one file, one purpose):
#   Take the top-K candidates from hybrid search and re-order them with
#   a cross-encoder that reads the (question, chunk) pair TOGETHER —
#   giving a much more accurate relevance score than the bi-encoder
#   (embedding) similarity used for the initial retrieval pass.

# WHY A RERANKER AT ALL (requirement #7):
#   Bi-encoders (used in embeddings.py) embed the question and the chunk
#   SEPARATELY, then compare vectors — fast, but approximate. A
#   cross-encoder feeds "[question] [SEP] [chunk]" through one model in
#   a single pass, so it can directly judge "does this chunk actually
#   answer this question", which retrieval-only pipelines often miss on
#   nuanced medical phrasing (e.g. "signs" vs "causes" vs "risk factors").

# WHY cross-encoder/ms-marco-MiniLM-L-6-v2:
#   It's small (~22M params) and runs in single-digit milliseconds per
#   pair on CPU, so reranking 20 candidates costs <100ms — protecting
#   requirement #14 (speed) while still meaningfully improving precision
#   over hybrid search alone.
"""

from typing import Dict, List

from sentence_transformers import CrossEncoder

from config.config import settings


class Reranker:
    def __init__(self, model_name: str = settings.RERANKER_MODEL):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = settings.TOP_K_RERANK) -> List[Dict]:
        """
        Args:
            query: the user's question.
            candidates: list of dicts from vector_store.hybrid_query()
                        (each must have a "text" key).
            top_k: how many of the best candidates to keep after rerank.

        Returns:
            The top_k candidates, sorted best-first, each with an added
            "rerank_score" key.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]


# Singleton
reranker = Reranker()
