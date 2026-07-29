"""Qdrant vector store access."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ingest.chunking import Chunk
from rag.config import settings


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

    def citation(self) -> str:
        return f"{self.title} (p. {self.page_number}, chunk {self.chunk_index})"


def get_client(url: str | None = None) -> QdrantClient:
    return QdrantClient(url=url or settings.qdrant_url, timeout=120)


def recreate_collection(client: QdrantClient, dimension: int, name: str | None = None) -> str:
    name = name or settings.qdrant_collection
    client.recreate_collection(
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


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int | None = None,
    name: str | None = None,
) -> list[Retrieved]:
    name = name or settings.qdrant_collection
    hits = client.query_points(
        collection_name=name,
        query=query_vector,
        limit=top_k or settings.top_k,
        with_payload=True,
    ).points
    return [
        Retrieved(
            score=hit.score,
            doc_id=hit.payload["doc_id"],
            title=hit.payload["title"],
            url=hit.payload["url"],
            reference_date=hit.payload.get("reference_date", ""),
            chunk_index=hit.payload["chunk_index"],
            page_number=hit.payload["page_number"],
            text=hit.payload["text"],
        )
        for hit in hits
    ]


def collection_size(client: QdrantClient, name: str | None = None) -> int:
    name = name or settings.qdrant_collection
    return client.count(collection_name=name, exact=True).count
