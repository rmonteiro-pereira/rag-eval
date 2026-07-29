"""Reciprocal Rank Fusion — how the dense and sparse arms are combined.

    RRF(d) = sum over arms of  1 / (k + rank_arm(d))          [Cormack et al. 2009]

Chosen over a weighted sum of the raw scores for one concrete reason: BM25
scores are unbounded sums of IDF terms and cosine similarities live in [-1, 1],
so any weighted combination of the two needs a normalisation whose parameters
would have to be tuned on the very gold set the ablation is meant to measure.
RRF only reads *ranks*, so it has one constant and it is not fitted to anything.

`k = 60` is the value from the original paper. It is left alone deliberately:
tuning it against a 49-row draft gold set would be fitting noise, and the
ablation's job is to show what each component contributes, not to squeeze the
last point out of the fusion constant.
"""

from __future__ import annotations

from collections.abc import Sequence

from retrieval.store import Retrieved

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Retrieved]],
    k: int = RRF_K,
    top_k: int | None = None,
) -> list[Retrieved]:
    """Fuse several ranked lists into one, best first.

    The returned records carry the fused score in `score` and keep each arm's
    own score under `signals`, so a report can say *why* something ranked where
    it did rather than only that it did.
    """
    fused: dict[str, Retrieved] = {}
    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = hit.key
            contribution = 1.0 / (k + rank)
            scores[key] = scores.get(key, 0.0) + contribution
            if key not in fused:
                # Copy so fusing does not mutate the caller's lists.
                fused[key] = Retrieved(
                    score=0.0,
                    doc_id=hit.doc_id,
                    title=hit.title,
                    url=hit.url,
                    reference_date=hit.reference_date,
                    chunk_index=hit.chunk_index,
                    page_number=hit.page_number,
                    text=hit.text,
                    signals=dict(hit.signals),
                )
            else:
                fused[key].signals.update(hit.signals)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    result: list[Retrieved] = []
    for key, score in ordered[: top_k or len(ordered)]:
        record = fused[key]
        record.score = score
        record.signals["rrf"] = score
        result.append(record)
    return result
