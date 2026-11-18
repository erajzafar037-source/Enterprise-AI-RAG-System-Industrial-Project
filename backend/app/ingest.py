"""PDF text extraction and bounded, overlap-aware chunking."""

from pathlib import Path

from pypdf import PdfReader

from .config import settings
from .retriever import Chunk


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split normalized text into chunks without silently accepting bad bounds.

    Chunk boundaries prefer whitespace so words are not cut in the middle.
    Overlap is measured in characters and is capped by the supplied settings.
    """
    chunk_size = settings.chunk_size if size is None else size
    chunk_overlap = settings.chunk_overlap if overlap is None else overlap

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("size must be a positive integer")
    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise ValueError("overlap must be a non-negative integer")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than size")
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            # Avoid tiny chunks while preferring a clean word boundary.
            boundary = normalized.rfind(" ", start + max(1, chunk_size // 2), end)
            if boundary > start:
                end = boundary

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_length:
            break

        next_start = end - chunk_overlap
        # Always make forward progress, even for pathological boundary layouts.
        start = next_start if next_start > start else end

    return chunks


def extract_pdf(path: str | Path) -> list[Chunk]:
    """Extract page-aware chunks from a PDF file."""
    reader = PdfReader(str(path))
    output: list[Chunk] = []
    source = Path(path).name

    for page_no, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        output.extend(
            Chunk(piece, source, page_no)
            for piece in chunk_text(page_text)
        )

    return output
