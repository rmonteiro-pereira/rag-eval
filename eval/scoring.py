"""Scoring a retriever against the gold set — the part `run_eval` and the
ablation share.

Extracted from `run_eval` when M4 arrived, for the reason that makes an ablation
mean anything: every arm has to be scored by *the same code*. If the ablation
had its own copy of the scoring loop, a delta between two arms would be
confounded with a delta between two implementations of recall.

Nothing here talks to Qdrant except through the retriever it is handed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from eval.gold import GoldRow
from eval.metrics.retrieval import DEFAULT_KS, RetrievalScores, aggregate, score_ranking
from eval.qrels import ChunkRef, RowQrels, build_row_qrels, chunk_key
from retrieval.store import Retrieved


@dataclass
class ScoredQuery:
    """One answerable gold row, retrieved and scored."""

    row: GoldRow
    hits: list[Retrieved]
    qrels: RowQrels
    scores: RetrievalScores

    @property
    def rank1_doc_id(self) -> str | None:
        return self.hits[0].doc_id if self.hits else None

    @property
    def rank1_doc_correct(self) -> bool:
        """Did the top hit come from the document the gold row names?

        Coarser than the span-level metrics on purpose. It is the question the
        wrong-meeting defect is actually about: not "was the paragraph right"
        but "was it the right meeting's copy of that paragraph".
        """
        return self.rank1_doc_id == self.row.source_doc_id

    def doc_in_top(self, k: int) -> bool:
        return any(hit.doc_id == self.row.source_doc_id for hit in self.hits[:k])

    def to_json(self, ks: Sequence[int]) -> dict:
        return {
            "id": self.row.id,
            "question": self.row.question,
            "capability": self.row.capability,
            "difficulty": self.row.difficulty,
            "answer_type": self.row.answer_type,
            "source_doc_id": self.row.source_doc_id,
            "source_page": self.row.source_page,
            "span_matched": self.qrels.span_matched,
            "rank1_doc_id": self.rank1_doc_id,
            "rank1_doc_correct": self.rank1_doc_correct,
            "metrics": self.scores.to_json(),
            "retrieved": [
                {
                    "rank": rank,
                    "doc_id": hit.doc_id,
                    "page": hit.page_number,
                    "chunk_index": hit.chunk_index,
                    "score": round(hit.score, 6),
                    "signals": {k: round(v, 6) for k, v in sorted(hit.signals.items())},
                    "gain": self.qrels.qrels.get(chunk_key(hit.doc_id, hit.chunk_index), 0),
                }
                for rank, hit in enumerate(self.hits, start=1)
            ],
        }


@dataclass
class SuiteResult:
    """Everything one arm produced over one gold set."""

    scored: list[ScoredQuery] = field(default_factory=list)
    abstention: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def aggregate(self) -> dict[str, float | int]:
        return aggregate([sq.scores for sq in self.scored])

    def breakdown(self, attribute: str) -> dict[str, dict[str, float | int]]:
        """The same metrics per capability / per difficulty.

        Cheap, and it is where the interesting failures show up: a healthy
        headline number can hide a capability that never retrieves anything.
        """
        buckets: dict[str, list[RetrievalScores]] = {}
        for sq in self.scored:
            buckets.setdefault(getattr(sq.row, attribute), []).append(sq.scores)
        return {name: aggregate(group) for name, group in sorted(buckets.items())}

    def unmatched_span_ids(self) -> list[str]:
        return [sq.row.id for sq in self.scored if not sq.qrels.span_matched]


def score_rows(
    retriever,
    rows: Sequence[GoldRow],
    chunks: Sequence[ChunkRef],
    ks: Sequence[int] = DEFAULT_KS,
    top_k: int | None = None,
) -> SuiteResult:
    """Retrieve and score every answerable row.

    Abstention rows carry no span, so they have no relevant chunk and are
    excluded from retrieval metrics rather than scored as zeros — "we retrieved
    none of the zero relevant chunks" is a category error, not a bad score. They
    are counted, and the abstention metric itself belongs to generation (M5).
    """
    top_k = top_k or max(ks)
    result = SuiteResult()

    for row in rows:
        if row.is_abstention:
            result.abstention.append(
                {"id": row.id, "question": row.question, "capability": row.capability}
            )
            continue

        assert row.source_doc_id is not None  # guaranteed by the gold loader
        row_qrels = build_row_qrels(
            source_doc_id=row.source_doc_id,
            source_page=row.source_page,
            source_span=row.source_span,
            chunks=chunks,
        )
        if not row_qrels.qrels:
            result.skipped.append(
                {
                    "id": row.id,
                    "reason": "no chunk in the collection matches this row's document/page",
                    "source_doc_id": row.source_doc_id,
                    "source_page": row.source_page,
                }
            )
            continue

        hits = retriever.retrieve(row.question, top_k=top_k)
        ranking = [chunk_key(hit.doc_id, hit.chunk_index) for hit in hits]
        result.scored.append(
            ScoredQuery(
                row=row,
                hits=hits,
                qrels=row_qrels,
                scores=score_ranking(ranking, row_qrels.qrels, ks=ks),
            )
        )

    return result
