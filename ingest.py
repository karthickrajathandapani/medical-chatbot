"""
ingest.py
=========
# PURPOSE (one file, one purpose):
#   Run this ONCE (and again any time you add/change source PDFs) to:
#     1. Load every PDF in data/            (src/pdf_loader.py)
#     2. Split pages into chunks            (src/chunking.py)
#     3. Fit BM25 + build dense embeddings  (src/sparse_encoder.py, src/embeddings.py)
#     4. Create the Pinecone index if needed and upsert everything (src/vector_store.py)

Usage:
    python ingest.py
"""

import hashlib
import pickle
import sys
import time
from pathlib import Path

from config.config import settings, validate_keys
from src.chunking import chunk_pages
from src.embeddings import embedding_model
from src.pdf_loader import load_all_pdfs
from src.sparse_encoder import sparse_encoder
from src.vector_store import vector_store

# Where the "slow" intermediate results (chunks + both kinds of vectors)
# get cached. If steps 1-4 already ran once and only the Pinecone upsert
# (step 5) failed, re-running this script reuses the cache instead of
# spending another ~18 minutes re-embedding everything from scratch.
_CACHE_PATH = Path("data/ingest_cache.pkl")


def _cache_key(texts, model_name: str) -> str:
    """Fingerprint of the exact inputs that produced the cached vectors."""
    joined = "".join(texts) + model_name
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _load_cached_embeddings(key: str):
    if not _CACHE_PATH.exists():
        return None
    try:
        with open(_CACHE_PATH, "rb") as f:
            cached_key, dense_vectors = pickle.load(f)
        return dense_vectors if cached_key == key else None
    except Exception:
        return None  # corrupt/old cache -> ignore and recompute


def _save_cached_embeddings(key: str, dense_vectors) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump((key, dense_vectors), f)


def main():
    print("=" * 60)
    print("  MEDICAL CHATBOT — INGESTION PIPELINE")
    print("=" * 60)

    validate_keys()

    # 1. LOAD -----------------------------------------------------------
    print(f"\n[1/5] Loading PDFs from '{settings.PDF_DIR}/' ...")
    t0 = time.perf_counter()
    pages = load_all_pdfs(settings.PDF_DIR)
    print(f"      -> {len(pages)} pages loaded in {time.perf_counter() - t0:.1f}s")
    if not pages:
        print("      No PDFs found! Put your .pdf file(s) in the data/ folder.")
        sys.exit(1)

    # 2. CHUNK ------------------------------------------------------------
    print(f"\n[2/5] Chunking (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP}) ...")
    t0 = time.perf_counter()
    chunks = chunk_pages(pages, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    print(f"      -> {len(chunks)} chunks created in {time.perf_counter() - t0:.1f}s")

    texts = [c.text for c in chunks]
    key = _cache_key(texts, settings.EMBEDDING_MODEL)

    # 3. SPARSE (BM25) — always (re)run: it's fast (~15-25s for this book)
    # and MUST write a fresh data/bm25_params.json, which app.py needs at
    # query time. Only the slow step (embedding) gets cached below.
    print("\n[3/5] Fitting BM25 sparse encoder over full corpus ...")
    t0 = time.perf_counter()
    sparse_encoder.fit(texts)
    sparse_vectors = sparse_encoder.encode_documents(texts)
    print(f"      -> done in {time.perf_counter() - t0:.1f}s")

    # 4. DENSE (Sentence-Transformer embeddings) ---------------------------
    cached_dense = _load_cached_embeddings(key)
    if cached_dense:
        print(f"\n[4/5] Found cached embeddings from a previous run — skipping "
              f"re-embedding (this is what saves you from waiting ~18+ minutes again).")
        dense_vectors = cached_dense
    else:
        print(f"\n[4/5] Embedding {len(texts)} chunks with '{settings.EMBEDDING_MODEL}' ...")
        t0 = time.perf_counter()
        dense_vectors = embedding_model.embed_documents(texts)
        print(f"      -> done in {time.perf_counter() - t0:.1f}s")
        _save_cached_embeddings(key, dense_vectors)

    # 5. UPSERT TO PINECONE -------------------------------------------------
    print(f"\n[5/5] Upserting hybrid vectors into Pinecone index '{settings.PINECONE_INDEX_NAME}' ...")
    t0 = time.perf_counter()
    vector_store.create_index_if_missing()
    count = vector_store.upsert_hybrid(chunks, dense_vectors, sparse_vectors)
    print(f"      -> upserted {count} vectors in {time.perf_counter() - t0:.1f}s")

    print("\n✅ Ingestion complete! You can now run: python app.py")


if __name__ == "__main__":
    main()
