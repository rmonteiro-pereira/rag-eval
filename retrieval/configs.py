"""Named retrieval configurations — the arms of the M4 ablation.

Each arm is one `RetrievalConfig`, and `build_retriever` assembles it out of the
same four stages:

    metadata filter  ->  dense / BM25  ->  RRF fusion  ->  cross-encoder rerank

Turning a stage off is a flag, not a different code path. That is the whole
point of the ablation: `hybrid` and `hybrid+rerank` must differ in exactly one
thing, or the delta between them measures the refactor rather than the reranker.

The arms:

* **`dense`** — the M1 baseline. bge-m3, cosine, top-k. Nothing else.
* **`bm25`** — lexical only. Not a serious candidate for production; it is here
  because a hybrid whose sparse arm is never measured alone is a hybrid nobody
  can reason about. On this corpus it turned out to be the second-best arm,
  which is precisely the kind of thing a missing control hides.
* **`hybrid`** — RRF over dense + BM25.
* **`hybrid+rerank`** — hybrid down to 30 candidates, cross-encoder on top.
* **`dense+metadata`**, **`hybrid+metadata`** — the filter with everything else
  held fixed, so its contribution can be read off directly instead of inferred
  from the top of the stack.
* **`hybrid+rerank+metadata`** — the full stack.

Stage ordering matters and is not arbitrary: the metadata filter runs *first*,
as a payload filter inside the vector search, so it shrinks the candidate pool
the other stages work on rather than trimming their output afterwards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient

from ingest.embedding import Embedder
from rag.config import settings
from retrieval.fusion import RRF_K, reciprocal_rank_fusion
from retrieval.metadata import DocumentMeta, document_meta, parse_hint, resolve
from retrieval.rerank import Reranker
from retrieval.sparse import BM25Index
from retrieval.store import Retrieved, doc_id_filter, get_client, scroll_all, search


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    description: str
    dense: bool = True
    sparse: bool = False
    metadata_filter: bool = False
    rerank: bool = False
    #: Depth of the first stage when a second stage follows. Ignored when
    #: nothing reranks, in which case the first stage returns `top_k` directly.
    candidate_k: int = 30
    rrf_k: int = RRF_K

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "dense": self.dense,
            "sparse": self.sparse,
            "metadata_filter": self.metadata_filter,
            "rerank": self.rerank,
            "candidate_k": self.candidate_k if self.rerank else None,
            "rrf_k": self.rrf_k if (self.dense and self.sparse) else None,
        }


DENSE = RetrievalConfig(
    name="dense",
    description="M1 baseline: bge-m3 cosine top-k, nothing else.",
    dense=True,
)
BM25 = RetrievalConfig(
    name="bm25",
    description="Lexical only: in-repo BM25 (k1=1.5, b=0.75) over the same chunks.",
    dense=False,
    sparse=True,
)
HYBRID = RetrievalConfig(
    name="hybrid",
    description="Reciprocal rank fusion of dense and BM25 (k=60).",
    dense=True,
    sparse=True,
)
HYBRID_RERANK = RetrievalConfig(
    name="hybrid+rerank",
    description="Hybrid to 30 candidates, reordered by the bge-reranker cross-encoder.",
    dense=True,
    sparse=True,
    rerank=True,
)
DENSE_META = RetrievalConfig(
    name="dense+metadata",
    description="Baseline plus the meeting filter, and nothing else — isolates the filter.",
    dense=True,
    metadata_filter=True,
)
HYBRID_META = RetrievalConfig(
    name="hybrid+metadata",
    description="Hybrid plus the meeting filter, no reranker.",
    dense=True,
    sparse=True,
    metadata_filter=True,
)
HYBRID_RERANK_META = RetrievalConfig(
    name="hybrid+rerank+metadata",
    description=(
        "Full stack: resolve the meeting named in the question to a doc_id set, "
        "apply it as a Qdrant payload filter, then hybrid, then rerank."
    ),
    dense=True,
    sparse=True,
    metadata_filter=True,
    rerank=True,
)

#: Not a cumulative ladder. A ladder can only ever say "the whole stack beats the
#: baseline", which is the least interesting thing an ablation can say — and on
#: this corpus it would have been actively misleading, because two of the middle
#: rungs are *worse* than a rung below them.
#:
#: Instead the arms are chosen so that meaningful pairs of them differ in exactly
#: one component, and `eval.ablation.CONTRASTS` names those pairs. That is what
#: lets the writeup attribute a number to a component rather than to a position
#: in a list.
ABLATION_CONFIGS: tuple[RetrievalConfig, ...] = (
    DENSE,
    BM25,
    HYBRID,
    HYBRID_RERANK,
    DENSE_META,
    HYBRID_META,
    HYBRID_RERANK_META,
)

CONFIGS_BY_NAME: dict[str, RetrievalConfig] = {cfg.name: cfg for cfg in ABLATION_CONFIGS}

#: The arm the rest of the system uses by default once M4 has been measured.
DEFAULT_CONFIG_NAME = "hybrid+rerank+metadata"


class CachedEmbedder:
    """Memoises query vectors.

    The ablation asks the same 49 questions once per arm. bge-m3 on CPU is the
    single most expensive thing in the run and its answer for a given string
    never changes, so encoding it five times is pure waste.
    """

    def __init__(self, inner: Embedder | None = None) -> None:
        self.inner = inner or Embedder()
        self._cache: dict[str, list[float]] = {}

    def embed_query(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self.inner.embed_query(text)
        return self._cache[text]


@dataclass
class RetrievalContext:
    """Everything the arms share, built once per process.

    Sharing the corpus scroll, the BM25 index, the embedder cache and the
    reranker across arms is not just a speed-up: it means every arm is measured
    against the identical corpus snapshot.
    """

    client: QdrantClient
    corpus: list[Retrieved]
    documents: list[DocumentMeta]
    bm25: BM25Index
    embedder: CachedEmbedder
    _reranker: Reranker | None = None

    @classmethod
    def build(cls, client: QdrantClient | None = None) -> RetrievalContext:
        client = client or get_client()
        corpus = scroll_all(client)
        if not corpus:
            raise RuntimeError(
                f"collection {settings.qdrant_collection!r} is empty — "
                "run `uv run python -m ingest.pipeline` first"
            )
        return cls(
            client=client,
            corpus=corpus,
            documents=document_meta(corpus),
            bm25=BM25Index.build(corpus),
            embedder=CachedEmbedder(),
        )

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker


class ConfigurableRetriever:
    """One ablation arm, executed."""

    def __init__(
        self,
        config: RetrievalConfig,
        context: RetrievalContext,
        top_k: int | None = None,
    ) -> None:
        self.config = config
        self.context = context
        self.top_k = top_k or settings.top_k

    @property
    def name(self) -> str:
        return self.config.name

    def resolve_documents(self, question: str) -> set[str]:
        """Document ids the question's meeting hint points at; empty when none.

        Empty also covers "the hint named something this corpus does not have"
        — see `retrieval.metadata`. Retrieval then runs unfiltered rather than
        over nothing.
        """
        if not self.config.metadata_filter:
            return set()
        return resolve(parse_hint(question), self.context.documents)

    def retrieve(self, question: str, top_k: int | None = None) -> list[Retrieved]:
        cfg = self.config
        top_k = top_k or self.top_k
        depth = max(cfg.candidate_k, top_k) if cfg.rerank else top_k

        allowed = self.resolve_documents(question)
        allowed_ids: Sequence[str] | None = sorted(allowed) if allowed else None

        rankings: list[list[Retrieved]] = []
        if cfg.dense:
            vector = self.context.embedder.embed_query(question)
            rankings.append(
                search(
                    self.context.client,
                    vector,
                    top_k=depth,
                    query_filter=doc_id_filter(allowed_ids),
                )
            )
        if cfg.sparse:
            rankings.append(self.context.bm25.search(question, depth, allowed_doc_ids=allowed_ids))

        if not rankings:
            raise ValueError(f"config {cfg.name!r} has no retrieval arm enabled")

        candidates = (
            reciprocal_rank_fusion(rankings, k=cfg.rrf_k, top_k=depth)
            if len(rankings) > 1
            else rankings[0][:depth]
        )

        if cfg.rerank:
            candidates = self.context.reranker.rerank(question, candidates, top_k=top_k)

        return candidates[:top_k]


def build_retriever(
    config: RetrievalConfig | str,
    context: RetrievalContext | None = None,
    top_k: int | None = None,
) -> ConfigurableRetriever:
    if isinstance(config, str):
        try:
            config = CONFIGS_BY_NAME[config]
        except KeyError as exc:
            known = ", ".join(CONFIGS_BY_NAME)
            raise ValueError(f"unknown retrieval config {config!r}; known: {known}") from exc
    return ConfigurableRetriever(config, context or RetrievalContext.build(), top_k=top_k)
