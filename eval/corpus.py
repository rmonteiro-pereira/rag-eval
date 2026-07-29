"""Reading the whole indexed corpus back out of Qdrant.

Needed because complete qrels require judging every chunk, not just retrieved
ones. Scrolling 636 chunks costs well under a second; if this corpus ever grows
past what fits in memory, the right move is to restrict the scroll to the
documents the gold set actually names, not to approximate the denominator.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from eval.qrels import ChunkRef
from rag.config import settings

SCROLL_BATCH = 256


def load_chunks(client: QdrantClient, collection: str | None = None) -> list[ChunkRef]:
    """Every chunk in the collection, as `ChunkRef`s."""
    collection = collection or settings.qdrant_collection
    chunks: list[ChunkRef] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in batch:
            payload = point.payload or {}
            chunks.append(
                ChunkRef(
                    doc_id=payload["doc_id"],
                    chunk_index=payload["chunk_index"],
                    page_number=payload["page_number"],
                    text=payload["text"],
                )
            )
        if offset is None:
            break
    return chunks
