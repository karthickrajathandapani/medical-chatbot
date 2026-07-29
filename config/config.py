"""
config.py
=========
# PURPOSE (one file, one purpose):
#   Single source of truth for every setting the app needs — API keys,
#   model names, chunking numbers, retrieval weights, etc.
#   Nothing else in the codebase should call os.getenv() directly.
#   This keeps all secrets in ONE place (requirement #10: separate API keys).

Reads values from a `.env` file in the project root using python-dotenv,
then falls back to sane defaults if a value is missing.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load the .env file once, as early as possible.
load_dotenv()


def _env(key: str, default: str = "") -> str:
    """Small helper so every read is stripped + defaulted the same way."""
    return os.getenv(key, default).strip()


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------------
    # 🔑 API KEYS  (kept ONLY in .env — never hard-code these)
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    PINECONE_API_KEY: str = field(default_factory=lambda: _env("PINECONE_API_KEY"))

    # ------------------------------------------------------------------
    # 📄 PDF / INGESTION SETTINGS
    # ------------------------------------------------------------------
    PDF_DIR: str = field(default_factory=lambda: _env("PDF_DIR", "data"))

    # Chunking (recursive, structure-aware — see src/chunking.py for why)
    CHUNK_SIZE: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "500")))
    CHUNK_OVERLAP: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "80")))

    # ------------------------------------------------------------------
    # 🧠 EMBEDDING MODEL (Sentence-Transformers, biomedical-tuned)
    # ------------------------------------------------------------------
    # PubMedBERT fine-tuned for retrieval (MS-MARCO objective) on biomedical
    # text — outperforms generic all-MiniLM on clinical / medical corpora.
    EMBEDDING_MODEL: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")
    )
    EMBEDDING_DIM: int = field(default_factory=lambda: int(_env("EMBEDDING_DIM", "768")))

    # ------------------------------------------------------------------
    # 🌲 PINECONE (vector database)
    # ------------------------------------------------------------------
    PINECONE_INDEX_NAME: str = field(
        default_factory=lambda: _env("PINECONE_INDEX_NAME", "medical-chatbot-hybrid")
    )
    PINECONE_CLOUD: str = field(default_factory=lambda: _env("PINECONE_CLOUD", "aws"))
    PINECONE_REGION: str = field(default_factory=lambda: _env("PINECONE_REGION", "us-east-1"))
    # Hybrid search REQUIRES the "dotproduct" metric in Pinecone.
    PINECONE_METRIC: str = "dotproduct"

    # ------------------------------------------------------------------
    # 🔀 HYBRID RETRIEVAL WEIGHTS
    # ------------------------------------------------------------------
    # alpha = 1.0 -> pure dense (semantic) search
    # alpha = 0.0 -> pure sparse (keyword / BM25) search
    # 0.5–0.75 blends both — great for medical text where exact drug/
    # disease names (sparse) matter as much as semantic meaning (dense).
    HYBRID_ALPHA: float = field(default_factory=lambda: float(_env("HYBRID_ALPHA", "0.6")))
    TOP_K_RETRIEVE: int = field(default_factory=lambda: int(_env("TOP_K_RETRIEVE", "20")))
    TOP_K_RERANK: int = field(default_factory=lambda: int(_env("TOP_K_RERANK", "4")))

    # ------------------------------------------------------------------
    # 🏆 RERANKER (cross-encoder)
    # ------------------------------------------------------------------
    # ms-marco-MiniLM is small + fast (~5ms/pair on CPU) which matters for
    # requirement #14 (speed). BAAI/bge-reranker-base is a slower, higher
    # quality alternative — swap the name below if latency isn't a concern.
    RERANKER_MODEL: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # ------------------------------------------------------------------
    # 🤖 OPENAI LLM
    # ------------------------------------------------------------------
    # gpt-4o-mini is the speed/quality sweet spot for RAG answer synthesis.
    OPENAI_MODEL: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    OPENAI_TEMPERATURE: float = field(
        default_factory=lambda: float(_env("OPENAI_TEMPERATURE", "0.2"))
    )
    OPENAI_MAX_TOKENS: int = field(default_factory=lambda: int(_env("OPENAI_MAX_TOKENS", "600")))

    # ------------------------------------------------------------------
    # 💬 MEMORY
    # ------------------------------------------------------------------
    MEMORY_MAX_TURNS: int = field(default_factory=lambda: int(_env("MEMORY_MAX_TURNS", "6")))

    # ------------------------------------------------------------------
    # 🖥️ FLASK SERVER
    # ------------------------------------------------------------------
    FLASK_HOST: str = field(default_factory=lambda: _env("FLASK_HOST", "0.0.0.0"))
    # Most free hosts (Render, Railway, etc.) assign your app a port
    # dynamically via a $PORT environment variable at runtime — we prefer
    # that if present, and fall back to FLASK_PORT/7860 for local runs or
    # Hugging Face Spaces (which expects a fixed port instead).
    FLASK_PORT: int = field(
        default_factory=lambda: int(_env("PORT") or _env("FLASK_PORT", "7860"))
    )
    FLASK_DEBUG: bool = field(default_factory=lambda: _env("FLASK_DEBUG", "False") == "True")


# A single, importable instance — `from config.config import settings`
settings = Settings()


def validate_keys() -> None:
    """Fail fast (with a clear message) if required secrets are missing."""
    missing = []
    if not settings.OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not settings.PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"Copy .env.example to .env and fill these in."
        )
