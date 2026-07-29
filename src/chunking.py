"""
chunking.py
===========
# PURPOSE (one file, one purpose):
#   Turn page-level text into small, semantically coherent chunks that
#   are the right size for embedding + retrieval.

# WHY "RECURSIVE CHARACTER" CHUNKING IS THE BEST STRATEGY HERE:
#   Medical encyclopedia entries are structured: Definition -> Causes ->
#   Symptoms -> Treatment, with paragraphs and sentences as natural
#   boundaries. A recursive splitter tries the "biggest" separator first
#   (paragraph breaks) and only falls back to smaller ones (sentences,
#   then words) when a chunk is still too big. This means:
#     - Chunks stay on paragraph/sentence boundaries whenever possible
#       (unlike naive fixed-length slicing, which can cut a sentence
#       describing a drug dosage in half).
#     - Overlap keeps context continuous across chunk edges, so a
#       "Symptoms" chunk still carries a bit of the preceding heading.
#   This is the same strategy LangChain's RecursiveCharacterTextSplitter
#   uses in production RAG systems, and is a strong, fast default
#   compared to heavier semantic-similarity chunking (which requires an
#   extra embedding pass just to decide where to cut).
"""

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.pdf_loader import PageDocument


@dataclass
class Chunk:
    """A retrieval-ready unit of text with citation metadata attached."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int


# Separators ordered from "most preferred cut point" to "least preferred".
# Medical-book specific additions: section headers often end with a colon
# or are followed by two newlines (e.g. "TREATMENT\n\n").
MEDICAL_SEPARATORS = [
    "\n\n",   # paragraph / section breaks
    "\n",     # line breaks
    ". ",     # sentence boundaries
    "; ",     # clause boundaries (common in symptom lists)
    ", ",     # last resort before hard word-split
    " ",
    "",
]


def chunk_pages(pages: List[PageDocument], chunk_size: int = 500, chunk_overlap: int = 80) -> List[Chunk]:
    """
    Split page documents into overlapping chunks.

    Args:
        pages: output of pdf_loader.load_pdf()
        chunk_size: target chunk size in characters (~500 chars ≈ 100-120
                    tokens — small enough for precise retrieval, large
                    enough to keep a full clinical thought intact).
        chunk_overlap: characters shared between consecutive chunks, so
                       context isn't lost at a cut boundary.

    Returns:
        List[Chunk]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=MEDICAL_SEPARATORS,
        length_function=len,
    )

    chunks: List[Chunk] = []
    for page in pages:
        pieces = splitter.split_text(page.text)
        for idx, piece in enumerate(pieces):
            chunk_id = f"{page.source_file}-p{page.page_number}-c{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    source_file=page.source_file,
                    page_number=page.page_number,
                )
            )
    return chunks
