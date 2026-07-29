"""
pdf_loader.py
=============
# PURPOSE (one file, one purpose):
#   Load a PDF from disk and return clean, page-tagged text.
#   This is the ONLY file that touches raw PDF bytes.

# WHY PyMuPDF (fitz) IS "THE BEST" LOADER HERE:
#   - Speed: PyMuPDF is written in C (MuPDF) and is 5-10x faster than
#     pure-Python loaders like PyPDF2/pypdf on large files (our medical
#     book is 16MB+ / hundreds of pages).
#   - Fidelity: it preserves reading order and handles multi-column
#     medical-textbook layouts far better than pdfplumber or PyPDF2,
#     which often scramble column text.
#   - Extras: gives per-page metadata (page number) for free, which we
#     use later for citations like "[Source: Medical_book.pdf, p. 42]".
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


@dataclass
class PageDocument:
    """A single page of extracted, cleaned text + its citation metadata."""
    page_number: int          # 1-indexed, human-friendly
    source_file: str          # original filename, for citations
    text: str                 # cleaned page text


def _clean_text(raw_text: str) -> str:
    """
    Normalize whitespace/noise that PDFs commonly introduce:
    - Collapse repeated newlines/spaces from column breaks.
    - Drop page headers/footers that are just page numbers.
    """
    text = re.sub(r"[ \t]+", " ", raw_text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lone-number lines (typical page-number footers)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def load_pdf(pdf_path: str) -> List[PageDocument]:
    """
    Open a PDF and return one PageDocument per non-empty page.

    Args:
        pdf_path: path to a .pdf file.

    Returns:
        List[PageDocument] — ready to be handed to the chunker.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    pages: List[PageDocument] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            raw = page.get_text("text")  # reading-order text extraction
            cleaned = _clean_text(raw)
            if cleaned:  # skip blank / image-only pages
                pages.append(
                    PageDocument(page_number=i, source_file=path.name, text=cleaned)
                )

    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path}. Is it a scanned/image PDF?")

    return pages


def load_all_pdfs(pdf_dir: str) -> List[PageDocument]:
    """Load every .pdf file found in a directory."""
    pdf_dir_path = Path(pdf_dir)
    all_pages: List[PageDocument] = []
    for pdf_file in sorted(pdf_dir_path.glob("*.pdf")):
        all_pages.extend(load_pdf(str(pdf_file)))
    return all_pages
