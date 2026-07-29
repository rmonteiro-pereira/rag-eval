"""Metric implementations.

`retrieval` is pure math over rankings and qrels — no Qdrant, no embeddings, no
network — so it can be unit-tested against hand-computed fixtures. Generation
and end-to-end metrics land in M3 alongside the LLM backend.
"""

from eval.metrics.retrieval import (
    RetrievalScores,
    dcg,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_ranking,
)

__all__ = [
    "RetrievalScores",
    "dcg",
    "hit_rate_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "score_ranking",
]
