"""Chunking strategies.

M1 ships exactly one, on purpose: **fixed-size character windows with overlap**.
It is the naive baseline the ablation in M4 has to beat, so it must stay dumb
and stable. New strategies get added alongside it, never in place of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ingest.loading import Document


@dataclass
class Chunk:
    doc_id: str
    title: str
    url: str
    reference_date: str
    chunk_index: int
    page_number: int
    text: str


def fixed_size_chunks(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Slide a fixed window over each page, keeping page provenance intact.

    Chunking per page (rather than over the concatenated document) means every
    chunk can cite a page number, which the gold set needs for span checking.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    index = 0
    stride = chunk_size - overlap

    for page in doc.pages:
        text = page.text
        start = 0
        while start < len(text):
            piece = text[start : start + chunk_size].strip()
            if len(piece) >= 80:  # drop slivers left at page ends
                chunks.append(
                    Chunk(
                        doc_id=doc.doc_id,
                        title=doc.title,
                        url=doc.url,
                        reference_date=doc.reference_date,
                        chunk_index=index,
                        page_number=page.page_number,
                        text=piece,
                    )
                )
                index += 1
            if start + chunk_size >= len(text):
                break
            start += stride

    return chunks
