"""Cross-encoder reranking with a local bge-reranker.

Dense retrieval and BM25 both score a query against a document *independently* —
the query is embedded once, the document was embedded at ingest time, and the
two never meet. A cross-encoder puts them in the same forward pass, so it can
attend from `junho de 2026` in the question to `279ª Reunião` in the chunk
header. On a corpus of thirty near-identical monetary-policy decisions that
difference is the whole ballgame.

The price is that it cannot be precomputed: every (query, chunk) pair is a model
call. So it runs strictly as a second stage over the first stage's top
`candidate_k`, never over the corpus.

`BAAI/bge-reranker-base` rather than `-v2-m3`: it is XLM-RoBERTa based (so
genuinely multilingual, which Portuguese needs), a third of the size, and scores
31 candidates in ~2 s on CPU. Model choice is measured in the ablation like
everything else; if the base model turns out to be the ceiling, swapping it is
one setting.

Downloaded once from HuggingFace into `~/.cache/huggingface`. No API key, no
network at query time, no per-call cost — the same rule as the embedder.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from rag.config import settings
from retrieval.store import Retrieved


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, device: str, max_length: int):
    # Lazy, like the embedder: importing sentence_transformers drags in torch.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device, max_length=max_length)


class Reranker:
    """Reorders candidates by cross-encoder relevance."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.reranker_device
        self.max_length = max_length or settings.reranker_max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name, self.device, self.max_length)
        return self._model

    def rerank(self, question: str, candidates: Sequence[Retrieved], top_k: int) -> list[Retrieved]:
        """Score every candidate against the question and return the best `top_k`.

        The reranker score *replaces* `score`, because it is now what the
        ranking means. The first-stage scores stay in `signals`, so a report can
        still show that a chunk arrived at rank 20 from BM25 and left at rank 1.
        """
        if not candidates:
            return []

        scores = self.model.predict([(question, hit.text) for hit in candidates])
        for hit, score in zip(candidates, scores, strict=True):
            hit.signals["rerank"] = float(score)
            hit.score = float(score)

        return sorted(candidates, key=lambda hit: -hit.score)[:top_k]
