"""Reading the whole indexed corpus back out of Qdrant.

Needed because complete qrels require judging every chunk, not just retrieved
ones. Scrolling 636 chunks costs well under a second; if this corpus ever grows
past what fits in memory, the right move is to restrict the scroll to the
documents the gold set actually names, not to approximate the denominator.

The scroll itself lives in `retrieval.store` because BM25 needs the very same
records to build its index from. Sharing one reader is what guarantees the dense
and sparse arms of the hybrid see byte-identical text.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from eval.qrels import ChunkRef
from retrieval.store import Retrieved, scroll_all


def to_chunk_refs(records: list[Retrieved]) -> list[ChunkRef]:
    return [
        ChunkRef(
            doc_id=record.doc_id,
            chunk_index=record.chunk_index,
            page_number=record.page_number,
            text=record.text,
        )
        for record in records
    ]


def load_chunks(client: QdrantClient, collection: str | None = None) -> list[ChunkRef]:
    """Every chunk in the collection, as `ChunkRef`s."""
    return to_chunk_refs(scroll_all(client, collection))
