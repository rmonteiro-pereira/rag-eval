"""Dense retriever — the M1 baseline.

Deliberately the simplest thing that works: embed the query with the same
encoder used at ingest time, take top-k by cosine similarity. No BM25, no
reranking, no query rewriting. Those land in M4 and each has to earn its place
with a measured delta.
"""

from __future__ import annotations

from ingest.embedding import Embedder
from rag.config import settings
from retrieval.store import Retrieved, get_client, search


class DenseRetriever:
    def __init__(self, embedder: Embedder | None = None, top_k: int | None = None) -> None:
        self.embedder = embedder or Embedder()
        self.top_k = top_k or settings.top_k
        self.client = get_client()

    def retrieve(self, question: str, top_k: int | None = None) -> list[Retrieved]:
        vector = self.embedder.embed_query(question)
        return search(self.client, vector, top_k=top_k or self.top_k)
