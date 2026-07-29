"""
rag_pipeline.py
===============
# PURPOSE (one file, one purpose):
#   The single orchestrator that wires every component into one
#   request/response flow:
#     query -> embed (dense+sparse) -> hybrid retrieve -> rerank
#           -> memory-aware prompt -> stream LLM answer -> save memory
#   app.py should only ever call INTO this file, never into src/* directly.
#   This keeps the "RAG logic" in exactly one place.
"""

import time
from typing import Dict, Generator, List, Tuple

from config.config import settings
from src.embeddings import embedding_model
from src.llm import stream_answer
from src.memory import conversation_memory
from src.reranker import reranker
from src.sparse_encoder import sparse_encoder
from src.vector_store import vector_store


def retrieve_context(question: str) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Run the retrieval half of the pipeline: hybrid search -> rerank.

    Returns:
        (reranked_chunks, timing_metrics)
    """
    timings = {}

    t0 = time.perf_counter()
    dense_vec = embedding_model.embed_query(question)
    sparse_vec = sparse_encoder.encode_query(question)
    timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t1 = time.perf_counter()
    candidates = vector_store.hybrid_query(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=settings.TOP_K_RETRIEVE,
        alpha=settings.HYBRID_ALPHA,
    )
    timings["retrieve_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    t2 = time.perf_counter()
    top_chunks = reranker.rerank(question, candidates, top_k=settings.TOP_K_RERANK)
    timings["rerank_ms"] = round((time.perf_counter() - t2) * 1000, 1)

    timings["top_score"] = round(top_chunks[0]["rerank_score"], 3) if top_chunks else 0.0
    timings["chunks_considered"] = len(candidates)
    timings["chunks_used"] = len(top_chunks)

    return top_chunks, timings


def answer_stream(session_id: str, question: str) -> Generator[Dict, None, None]:
    """
    Full pipeline, streamed for the frontend.

    Yields dicts of one of these shapes, in order:
      {"type": "metrics", "data": {...}}      # once, before the answer
      {"type": "token",   "data": "..."}      # many, as text streams in
      {"type": "sources", "data": [...]}      # once, after the answer
      {"type": "done"}                        # once, at the very end
    """
    top_chunks, timings = retrieve_context(question)
    yield {"type": "metrics", "data": timings}

    history = conversation_memory.as_openai_messages(session_id)

    full_answer = ""
    t_llm = time.perf_counter()
    for token in stream_answer(question, top_chunks, history):
        full_answer += token
        yield {"type": "token", "data": token}
    llm_ms = round((time.perf_counter() - t_llm) * 1000, 1)

    conversation_memory.add_turn(session_id, question, full_answer)

    sources = [
        {"source_file": c["source_file"], "page_number": c["page_number"], "score": round(c["rerank_score"], 3)}
        for c in top_chunks
    ]
    yield {"type": "sources", "data": sources}
    yield {"type": "metrics", "data": {**timings, "generation_ms": llm_ms}}
    yield {"type": "done"}
