"""
sparse_encoder.py
==================
# PURPOSE (one file, one purpose):
#   Produce SPARSE (keyword-weighted) vectors — the other half of
#   "hybrid retrieval" (requirement #6). Dense embeddings (embeddings.py)
#   capture MEANING; sparse vectors capture EXACT TERMS.

# WHY THIS MATTERS FOR A MEDICAL CHATBOT:
#   Dense-only search can conflate similar-sounding conditions ("Type 1"
#   vs "Type 2" diabetes) because they're semantically close. BM25
#   sparse vectors weight the exact token "Type 2" highly, so an exact
#   drug name, dosage figure, or ICD-style code is never "smoothed away"
#   by semantic similarity. Blending both (see vector_store.py) is the
#   best-practice "hybrid search" pattern Pinecone itself recommends.

# WHY BM25Encoder (pinecone-text) SPECIFICALLY:
#   It's Pinecone's own sparse encoder, purpose-built to output vectors
#   in the exact sparse format Pinecone's hybrid upsert/query API expects
#   — no manual format-wrangling required.
"""

import json
from pathlib import Path
from typing import Dict, List

from pinecone_text.sparse import BM25Encoder

_FITTED_PARAMS_PATH = Path("data/bm25_params.json")


class SparseEncoder:
    def __init__(self):
        self._encoder = BM25Encoder()
        self._is_fitted = False

    def fit(self, corpus: List[str]) -> None:
        """
        Fit BM25 term statistics (document frequencies, avg length) over
        the WHOLE corpus once, during ingestion. Must run before encoding.
        """
        self._encoder.fit(corpus)
        self._is_fitted = True
        _FITTED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._encoder.dump(str(_FITTED_PARAMS_PATH))

    def load(self) -> bool:
        """Load previously-fitted BM25 params, if they exist on disk."""
        if _FITTED_PARAMS_PATH.exists():
            self._encoder = BM25Encoder().load(str(_FITTED_PARAMS_PATH))
            self._is_fitted = True
            return True
        return False

    def encode_documents(self, texts: List[str]) -> List[Dict]:
        """Encode chunks for ingestion. Requires fit() to have been run."""
        self._require_fitted()
        return self._encoder.encode_documents(texts)

    def encode_query(self, text: str) -> Dict:
        """Encode a single user query at search time."""
        self._require_fitted()
        return self._encoder.encode_queries([text])[0]

    def _require_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "BM25 encoder isn't fitted yet. Run ingest.py first, "
                "or call sparse_encoder.load()."
            )


# Singleton — ingest.py fits it, app.py loads the fitted version.
sparse_encoder = SparseEncoder()
