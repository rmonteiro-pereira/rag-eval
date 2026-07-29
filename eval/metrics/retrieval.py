"""Retrieval metrics — pure math, no infrastructure.

Everything here operates on two plain values:

* ``ranking`` — the retrieved chunk keys, best first.
* ``qrels`` — a mapping ``chunk_key -> gain``, the *complete* set of relevant
  chunks for that query. Complete matters: recall is only recall if the
  denominator is the whole relevant set, which is why the runner materialises
  qrels by scanning every chunk in the collection rather than by looking at
  what happened to be retrieved.

Gains are graded (see ``eval/qrels.py``): 2 for a chunk that actually contains
the gold span, 1 for a chunk on the right page of the right document. Recall,
hit rate and MRR binarise at ``relevance_threshold``; only nDCG uses the grades.

nDCG uses the traditional linear-gain formulation (Jarvelin & Kekalainen 2002),
``sum(gain_i / log2(i + 1))`` with ``i`` the 1-indexed rank — not the
exponential ``2^gain - 1`` variant. With only two grades the two orderings
rarely disagree, and the linear form is the one the unit-test fixtures are
hand-computed against.

No metric here silently invents a value: scoring a query with an empty relevant
set raises rather than returning 0.0, because "we retrieved none of the zero
relevant chunks" is not a score, it is a category error.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def _relevant_keys(qrels: Mapping[str, int], relevance_threshold: int) -> set[str]:
    return {key for key, gain in qrels.items() if gain >= relevance_threshold}


def _require_relevant(qrels: Mapping[str, int], relevance_threshold: int) -> set[str]:
    relevant = _relevant_keys(qrels, relevance_threshold)
    if not relevant:
        raise ValueError(
            "cannot score a query whose relevant set is empty; "
            "abstention rows must be excluded before scoring"
        )
    return relevant


def recall_at_k(
    ranking: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
    relevance_threshold: int = 1,
) -> float:
    """Fraction of *all* relevant chunks that appear in the top ``k``.

    Bounded above by ``k / len(relevant)``, so recall@1 is mechanically small
    when a gold span spills across several chunks. That is the honest reading
    of set recall; ``hit_rate_at_k`` is the "did we get anything useful" view.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = _require_relevant(qrels, relevance_threshold)
    found = sum(1 for key in ranking[:k] if key in relevant)
    return found / len(relevant)


def hit_rate_at_k(
    ranking: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
    relevance_threshold: int = 1,
) -> float:
    """1.0 if at least one relevant chunk is in the top ``k``, else 0.0."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = _require_relevant(qrels, relevance_threshold)
    return 1.0 if any(key in relevant for key in ranking[:k]) else 0.0


def first_relevant_rank(
    ranking: Sequence[str],
    qrels: Mapping[str, int],
    relevance_threshold: int = 1,
) -> int | None:
    """1-indexed rank of the first relevant chunk, or ``None`` if there is none."""
    relevant = _require_relevant(qrels, relevance_threshold)
    for rank, key in enumerate(ranking, start=1):
        if key in relevant:
            return rank
    return None


def reciprocal_rank(
    ranking: Sequence[str],
    qrels: Mapping[str, int],
    relevance_threshold: int = 1,
) -> float:
    """1 / rank of the first relevant chunk; 0.0 when none was retrieved."""
    rank = first_relevant_rank(ranking, qrels, relevance_threshold)
    return 0.0 if rank is None else 1.0 / rank


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain of an already-ordered gain sequence."""
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Normalised DCG at ``k`` against the ideal ordering of the full qrels."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not any(gain > 0 for gain in qrels.values()):
        raise ValueError("cannot score a query whose relevant set is empty")

    actual = dcg([max(qrels.get(key, 0), 0) for key in ranking[:k]])
    ideal = dcg(sorted((g for g in qrels.values() if g > 0), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


@dataclass(frozen=True)
class RetrievalScores:
    """Per-query retrieval scores."""

    recall: dict[int, float]
    hit_rate: dict[int, float]
    ndcg: dict[int, float]
    mrr: float
    first_relevant_rank: int | None
    n_relevant_total: int
    n_retrieved: int

    def to_json(self) -> dict[str, float | int | None]:
        payload: dict[str, float | int | None] = {}
        for k, value in sorted(self.recall.items()):
            payload[f"recall@{k}"] = value
        for k, value in sorted(self.hit_rate.items()):
            payload[f"hit_rate@{k}"] = value
        for k, value in sorted(self.ndcg.items()):
            payload[f"ndcg@{k}"] = value
        payload["mrr"] = self.mrr
        payload["first_relevant_rank"] = self.first_relevant_rank
        payload["n_relevant_total"] = self.n_relevant_total
        payload["n_retrieved"] = self.n_retrieved
        return payload


def score_ranking(
    ranking: Sequence[str],
    qrels: Mapping[str, int],
    ks: Sequence[int] = DEFAULT_KS,
    relevance_threshold: int = 1,
) -> RetrievalScores:
    """Score one ranking against one complete qrels mapping."""
    relevant = _require_relevant(qrels, relevance_threshold)
    return RetrievalScores(
        recall={k: recall_at_k(ranking, qrels, k, relevance_threshold) for k in ks},
        hit_rate={k: hit_rate_at_k(ranking, qrels, k, relevance_threshold) for k in ks},
        ndcg={k: ndcg_at_k(ranking, qrels, k) for k in ks},
        mrr=reciprocal_rank(ranking, qrels, relevance_threshold),
        first_relevant_rank=first_relevant_rank(ranking, qrels, relevance_threshold),
        n_relevant_total=len(relevant),
        n_retrieved=len(ranking),
    )


def aggregate(scores: Sequence[RetrievalScores]) -> dict[str, float | int]:
    """Macro-average over queries — every query weighs the same.

    Macro rather than micro on purpose: the gold set deliberately over-samples
    hard capabilities, and micro-averaging would let the easy single-hop rows
    (which have more relevant chunks each) quietly dominate the headline number.
    """
    if not scores:
        raise ValueError("cannot aggregate an empty score list")

    n = len(scores)
    ks = sorted(scores[0].recall)
    payload: dict[str, float | int] = {}
    for k in ks:
        payload[f"recall@{k}"] = sum(s.recall[k] for s in scores) / n
    for k in ks:
        payload[f"hit_rate@{k}"] = sum(s.hit_rate[k] for s in scores) / n
    for k in ks:
        payload[f"ndcg@{k}"] = sum(s.ndcg[k] for s in scores) / n
    payload["mrr"] = sum(s.mrr for s in scores) / n
    payload["n_queries"] = n
    return payload
