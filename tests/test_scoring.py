"""`eval/scoring.py` — the function every published retrieval number comes from.

Written to close the largest gap mutation testing found: `score_rows` had **60
mutants and not one covering test**. It is exercised end to end by `run_eval` and
`ablation`, which is why the committed reports are trustworthy — but it means a
change to this arithmetic would be caught only by a full re-run against a live
Qdrant, never by `pytest`.

The three behaviours that matter here are not "does recall work" — that lives in
`eval/metrics/retrieval.py` and is tested there. They are the decisions
`score_rows` makes *about which rows count*:

1. Abstention rows are **excluded**, not scored as zeros. Scoring them zero would
   silently drag every aggregate down by the share of negatives in the gold set,
   and the number would still look like a retrieval metric.
2. A row whose document/page is missing from the collection is **skipped with a
   reason**, not scored zero. A zero there says "the retriever failed"; the truth
   is "the question was unanswerable from what was indexed".
3. Every arm is scored by *this* code, so a delta between arms is a delta between
   retrievers. That is the whole basis of the ablation.

No Qdrant, no embeddings: the retriever is a stub that returns what the test
tells it to.
"""

from __future__ import annotations

import pytest

from eval.gold import GoldRow
from eval.qrels import ChunkRef, chunk_key
from eval.scoring import ScoredQuery, SuiteResult, score_rows
from retrieval.store import Retrieved

SPAN = "O Copom decidiu reduzir a taxa basica de juros para 13,25% a.a."

CHUNKS = [
    ChunkRef(doc_id="ata-a", chunk_index=0, page_number=1, text="abertura da reuniao"),
    ChunkRef(doc_id="ata-a", chunk_index=1, page_number=6, text=f"contexto. {SPAN} e entende"),
    ChunkRef(doc_id="ata-a", chunk_index=2, page_number=6, text="outro trecho da mesma pagina"),
    ChunkRef(doc_id="ata-b", chunk_index=0, page_number=6, text=f"contexto. {SPAN} e entende"),
]


def row(
    row_id: str = "gold-001",
    *,
    answer_type: str = "extractive",
    doc: str | None = "ata-a",
    page: int | None = 6,
    span: str | None = SPAN,
    capability: str = "single-hop lookup",
    difficulty: str = "easy",
) -> GoldRow:
    return GoldRow(
        id=row_id,
        status="draft",
        question="para que nivel a taxa foi levada?",
        answer="13,25%",
        answer_type=answer_type,
        source_doc_id=doc,
        source_title="Ata A",
        source_page=page,
        source_span=span,
        difficulty=difficulty,
        capability=capability,
    )


def hit(doc_id: str, chunk_index: int, score: float = 1.0) -> Retrieved:
    ref = next(c for c in CHUNKS if c.doc_id == doc_id and c.chunk_index == chunk_index)
    return Retrieved(
        score=score,
        doc_id=doc_id,
        title=f"Ata {doc_id[-1].upper()}",
        url="about:blank",
        reference_date="2023-03-22",
        chunk_index=chunk_index,
        page_number=ref.page_number,
        text=ref.text,
    )


class StubRetriever:
    """Returns a fixed ranking, and records how it was called."""

    def __init__(self, ranking: list[Retrieved]) -> None:
        self.ranking = ranking
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, top_k: int) -> list[Retrieved]:
        self.calls.append((question, top_k))
        return self.ranking[:top_k]


# --------------------------------------------------------------------------
# Which rows are scored, and which are not
# --------------------------------------------------------------------------


def test_abstention_rows_are_excluded_rather_than_scored_zero():
    """The distinction the whole gold set depends on.

    Seven of the 56 gold rows are negatives. Scoring them as zeros would pull
    every aggregate down by ~1/8 and the result would still look like a
    retrieval metric, which is the dangerous part.
    """
    retriever = StubRetriever([hit("ata-a", 1)])
    rows = [row("gold-001"), row("neg-1", answer_type="abstention")]
    result = score_rows(retriever, rows, CHUNKS)

    assert [sq.row.id for sq in result.scored] == ["gold-001"]
    assert [a["id"] for a in result.abstention] == ["neg-1"]
    assert result.aggregate()["n_queries"] == 1
    # And the retriever was never asked about the abstention row.
    assert len(retriever.calls) == 1


def test_a_row_whose_document_is_not_indexed_is_skipped_with_a_reason():
    """Not scored zero: "unanswerable from what was indexed" is not "retriever failed"."""
    retriever = StubRetriever([hit("ata-a", 1)])
    result = score_rows(retriever, [row("gold-404", doc="ata-missing")], CHUNKS)

    assert result.scored == []
    assert len(result.skipped) == 1
    assert result.skipped[0]["id"] == "gold-404"
    assert result.skipped[0]["source_doc_id"] == "ata-missing"
    assert "no chunk" in result.skipped[0]["reason"]
    # A skipped row must not consume a retrieval call either.
    assert retriever.calls == []


def test_the_three_row_classes_are_counted_separately_and_do_not_overlap():
    retriever = StubRetriever([hit("ata-a", 1)])
    rows = [row("ok"), row("neg", answer_type="abstention"), row("gone", doc="ata-missing")]
    result = score_rows(retriever, rows, CHUNKS)

    assert (len(result.scored), len(result.abstention), len(result.skipped)) == (1, 1, 1)
    ids = (
        {sq.row.id for sq in result.scored}
        | {a["id"] for a in result.abstention}
        | {s["id"] for s in result.skipped}
    )
    assert ids == {"ok", "neg", "gone"}


# --------------------------------------------------------------------------
# What gets handed to the retriever
# --------------------------------------------------------------------------


def test_top_k_defaults_to_the_largest_cutoff_being_measured():
    """Asking for fewer than `max(ks)` would silently zero the deepest metric."""
    retriever = StubRetriever([hit("ata-a", 1)])
    score_rows(retriever, [row()], CHUNKS, ks=(1, 3, 5, 10))
    assert retriever.calls[0][1] == 10


def test_an_explicit_top_k_overrides_the_cutoffs():
    retriever = StubRetriever([hit("ata-a", 1)])
    score_rows(retriever, [row()], CHUNKS, ks=(1, 3), top_k=7)
    assert retriever.calls[0][1] == 7


def test_the_question_is_passed_through_unmodified():
    """A retriever must see what the gold row asks, not a normalised variant."""
    retriever = StubRetriever([hit("ata-a", 1)])
    score_rows(retriever, [row()], CHUNKS)
    assert retriever.calls[0][0] == "para que nivel a taxa foi levada?"


# --------------------------------------------------------------------------
# The wrong-meeting probe — the defect this project is about
# --------------------------------------------------------------------------


def test_rank1_doc_correct_is_true_only_for_the_document_the_row_names():
    """Right paragraph, wrong ata, is the failure the headline result is about."""
    right = score_rows(StubRetriever([hit("ata-a", 1)]), [row()], CHUNKS).scored[0]
    wrong = score_rows(StubRetriever([hit("ata-b", 0)]), [row()], CHUNKS).scored[0]

    assert right.rank1_doc_correct is True
    assert wrong.rank1_doc_correct is False
    # `ata-b` holds a verbatim copy of the span, so the span-level metrics do not
    # separate these two. That is exactly why the probe exists.
    assert wrong.rank1_doc_id == "ata-b"


def test_rank1_doc_correct_is_false_when_nothing_was_retrieved():
    scored = score_rows(StubRetriever([]), [row()], CHUNKS).scored[0]
    assert scored.rank1_doc_id is None
    assert scored.rank1_doc_correct is False


def test_doc_in_top_k_looks_only_at_the_first_k_hits():
    ranking = [hit("ata-b", 0), hit("ata-b", 0), hit("ata-a", 1)]
    scored = score_rows(StubRetriever(ranking), [row()], CHUNKS).scored[0]
    assert scored.doc_in_top(1) is False
    assert scored.doc_in_top(2) is False
    assert scored.doc_in_top(3) is True


# --------------------------------------------------------------------------
# SuiteResult
# --------------------------------------------------------------------------


def test_breakdown_buckets_by_the_named_attribute():
    retriever = StubRetriever([hit("ata-a", 1)])
    rows = [
        row("a", capability="single-hop lookup"),
        row("b", capability="reverse lookup"),
        row("c", capability="single-hop lookup"),
    ]
    result = score_rows(retriever, rows, CHUNKS)
    by_capability = result.breakdown("capability")

    assert set(by_capability) == {"single-hop lookup", "reverse lookup"}
    assert by_capability["single-hop lookup"]["n_queries"] == 2
    assert by_capability["reverse lookup"]["n_queries"] == 1


def test_breakdown_covers_every_scored_row_exactly_once():
    """A bucket that drops rows makes a per-capability table quietly wrong."""
    retriever = StubRetriever([hit("ata-a", 1)])
    rows = [row("a", difficulty="easy"), row("b", difficulty="hard"), row("c", difficulty="hard")]
    result = score_rows(retriever, rows, CHUNKS)
    total = sum(b["n_queries"] for b in result.breakdown("difficulty").values())
    assert total == len(result.scored) == 3


def test_unmatched_span_ids_names_rows_whose_span_was_not_found():
    """Span match is the difference between gain 2 and gain 1 — worth surfacing."""
    retriever = StubRetriever([hit("ata-a", 1)])
    rows = [row("found", span=SPAN), row("lost", span="uma frase que nao existe em lugar nenhum")]
    result = score_rows(retriever, rows, CHUNKS)
    assert result.unmatched_span_ids() == ["lost"]


def test_aggregating_an_empty_suite_raises_rather_than_returning_zeros():
    """This is the correct behaviour, and the first version of this test assumed
    the opposite.

    `--min-status validated` returns zero rows today by design, so the empty case
    is real rather than hypothetical. But a metric dict full of 0.0 is
    indistinguishable from a retriever that found nothing, and would be published
    as if it were a measurement. Raising forces the caller to decide.

    `run_eval` does decide, before it ever reaches here: it prints why the set is
    empty and exits 3. The two behaviours are a pair — the guard upstream is only
    safe because the arithmetic downstream refuses to invent a number.
    """
    result = SuiteResult()
    with pytest.raises(ValueError, match="empty"):
        result.aggregate()

    # The non-arithmetic accessors are still safe on an empty suite.
    assert result.unmatched_span_ids() == []
    assert result.breakdown("capability") == {}


def test_run_eval_exits_3_on_an_empty_gold_set_rather_than_aggregating():
    """The upstream half of the pair above."""
    from eval.run_eval import main

    assert main(["--min-status", "validated", "--quiet"]) == 3


# --------------------------------------------------------------------------
# The serialised row — what every committed report is made of
# --------------------------------------------------------------------------


def test_to_json_attaches_the_graded_gain_to_each_retrieved_chunk():
    """The audit trail: a reader must be able to see WHY a hit scored.

    Gain 2 is the chunk holding the span, gain 1 another chunk of the same
    document and page, 0 everything else. Without the gain on the row, the
    per-query record cannot be re-derived by hand.
    """
    ranking = [hit("ata-a", 1), hit("ata-a", 2), hit("ata-b", 0)]
    scored = score_rows(StubRetriever(ranking), [row()], CHUNKS).scored[0]
    payload = scored.to_json(ks=(1, 3, 5, 10))

    gains = {r["doc_id"] + "#" + str(r["chunk_index"]): r["gain"] for r in payload["retrieved"]}
    assert gains[chunk_key("ata-a", 1)] == 2
    assert gains[chunk_key("ata-a", 2)] == 1
    assert gains[chunk_key("ata-b", 0)] == 0

    assert payload["id"] == "gold-001"
    assert payload["rank1_doc_correct"] is True
    assert payload["span_matched"] is True
    assert [r["rank"] for r in payload["retrieved"]] == [1, 2, 3]


def test_to_json_reports_the_metrics_at_the_cutoffs_it_was_given():
    scored = score_rows(StubRetriever([hit("ata-a", 1)]), [row()], CHUNKS, ks=(1, 5)).scored[0]
    metrics = scored.to_json(ks=(1, 5))["metrics"]
    assert "recall@1" in metrics and "recall@5" in metrics
    assert "recall@3" not in metrics


@pytest.mark.parametrize("bad_ranking", [[], [hit("ata-b", 0)]])
def test_a_row_is_still_scored_when_retrieval_finds_nothing_relevant(bad_ranking):
    """Zero is a legitimate score here — the row WAS answerable and we missed it.

    This is the case that must not be confused with the two exclusions above.
    """
    result = score_rows(StubRetriever(bad_ranking), [row()], CHUNKS)
    assert len(result.scored) == 1
    assert result.skipped == []
    assert result.aggregate()["recall@10"] == 0.0


def test_scored_query_is_constructible_for_a_direct_metric_check():
    """Guards the dataclass contract the ablation and run_eval both rely on."""
    result = score_rows(StubRetriever([hit("ata-a", 1)]), [row()], CHUNKS)
    sq = result.scored[0]
    assert isinstance(sq, ScoredQuery)
    assert sq.qrels.qrels[chunk_key("ata-a", 1)] == 2
    assert sq.scores.to_json()["mrr"] == pytest.approx(1.0)
