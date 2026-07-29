"""Qdrant vector store access."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ingest.chunking import Chunk
from rag.config import settings

SCROLL_BATCH = 256


@dataclass
class Retrieved:
    score: float
    doc_id: str
    title: str
    url: str
    reference_date: str
    chunk_index: int
    page_number: int
    text: str
    #: Per-stage scores kept for the ablation report — `{"dense": .., "bm25": ..,
    #: "rrf": .., "rerank": ..}`. Purely diagnostic; `score` stays the number the
    #: ranking was actually made on, whichever stage produced it last.
    signals: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable chunk identity — mirrors `eval.qrels.chunk_key`."""
        return f"{self.doc_id}#{self.chunk_index}"

    def citation(self) -> str:
        return f"{self.title} (p. {self.page_number}, chunk {self.chunk_index})"


def get_client(url: str | None = None) -> QdrantClient:
    return QdrantClient(url=url or settings.qdrant_url, timeout=120)


def recreate_collection(client: QdrantClient, dimension: int, name: str | None = None) -> str:
    name = name or settings.qdrant_collection
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=dimension,
            distance=qmodels.Distance.COSINE,
        ),
    )
    # Metadata filters (by document, by date) are what M4/M5 will lean on.
    client.create_payload_index(
        collection_name=name,
        field_name="doc_id",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )
    return name


def upsert_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    vectors: list[list[float]],
    start_id: int = 0,
    name: str | None = None,
) -> int:
    name = name or settings.qdrant_collection
    points = [
        qmodels.PointStruct(
            id=start_id + i,
            vector=vector,
            payload={
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "url": chunk.url,
                "reference_date": chunk.reference_date,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "text": chunk.text,
            },
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
    ]
    client.upsert(collection_name=name, points=points, wait=True)
    return len(points)


def _from_payload(
    payload: dict,
    score: float,
    signals: dict[str, float] | None = None,
) -> Retrieved:
    return Retrieved(
        score=score,
        doc_id=payload["doc_id"],
        title=payload["title"],
        url=payload["url"],
        reference_date=payload.get("reference_date", ""),
        chunk_index=payload["chunk_index"],
        page_number=payload["page_number"],
        text=payload["text"],
        signals=dict(signals or {}),
    )


def doc_id_filter(doc_ids: Sequence[str] | None) -> qmodels.Filter | None:
    """A Qdrant payload filter restricting search to a set of documents.

    This is the server-side half of metadata filtering (M4) and, in M5, of
    access control: the same mechanism that says "only the June 2026 meeting"
    says "only the documents this user may see". Enforcing it in the query
    rather than by discarding results afterwards is the point — a post-filter
    silently shortens the result list and leaks the existence of what it hid.
    """
    if not doc_ids:
        return None
    return qmodels.Filter(
        must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchAny(any=list(doc_ids)))]
    )


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int | None = None,
    name: str | None = None,
    query_filter: qmodels.Filter | None = None,
) -> list[Retrieved]:
    name = name or settings.qdrant_collection
    hits = client.query_points(
        collection_name=name,
        query=query_vector,
        limit=top_k or settings.top_k,
        with_payload=True,
        query_filter=query_filter,
    ).points
    return [_from_payload(hit.payload, hit.score, {"dense": hit.score}) for hit in hits]


def scroll_all(client: QdrantClient, name: str | None = None) -> list[Retrieved]:
    """Every chunk in the collection, payload only.

    Needed twice over: complete qrels have to judge every chunk (not just the
    retrieved ones), and BM25 has to index the same text the dense side sees.
    Reading both from the store is what guarantees they are the same text.
    """
    name = name or settings.qdrant_collection
    records: list[Retrieved] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=name,
            limit=SCROLL_BATCH,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        records.extend(_from_payload(point.payload or {}, score=0.0) for point in batch)
        if offset is None:
            break
    return records


def collection_size(client: QdrantClient, name: str | None = None) -> int:
    name = name or settings.qdrant_collection
    return client.count(collection_name=name, exact=True).count
