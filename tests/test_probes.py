"""Probe grouping and rank-1 document scoring.

The probes are how the wrong-meeting defect is claimed fixed, so the thing worth
pinning is that group membership is decided by a rule and not by a hand-written
list of ids — otherwise "we fixed the probes" would be a statement about which
queries someone chose to put in them.
"""

from __future__ import annotations

import pytest

from eval.gold import GoldRow
from eval.metrics.retrieval import score_ranking
from eval.probes import (
    MEETING_DISAMBIGUATION,
    REVERSE_LOOKUP,
    assign_groups,
    classify,
    hint_diagnostics,
    run_probes,
)
from eval.qrels import RowQrels
from eval.scoring import ScoredQuery
from retrieval.metadata import DocumentMeta
from retrieval.store import Retrieved

DOCS = [
    DocumentMeta("ata-jul-2024", "264ª Reunião - 30-31 julho, 2024", "2024-07-31"),
    DocumentMeta("ata-jun-2026", "279ª Reunião - 16-17 junho, 2026", "2026-06-17"),
]


def hit(doc_id: str) -> Retrieved:
    return Retrieved(
        score=1.0,
        doc_id=doc_id,
        title=doc_id,
        url="http://example.invalid",
        reference_date="2024-07-31",
        chunk_index=0,
        page_number=6,
        text="O Copom decidiu manter a taxa basica de juros.",
    )


def scored(row_id: str, question: str, expected: str, returned: list[str]) -> ScoredQuery:
    row = GoldRow(
        id=row_id,
        status="draft",
        question=question,
        answer="",
        answer_type="extractive",
        source_doc_id=expected,
        source_title=None,
        source_page=6,
        source_span="span",
        difficulty="easy",
        capability="single-hop lookup",
    )
    hits = [hit(doc_id) for doc_id in returned]
    qrels = RowQrels(qrels={f"{expected}#0": 2}, span_parts=1, span_parts_matched=1)
    ranking = [f"{h.doc_id}#{h.chunk_index}" for h in hits]
    return ScoredQuery(
        row=row,
        hits=hits,
        qrels=qrels,
        scores=score_ranking(ranking, qrels.qrels, ks=(1, 3, 5)),
    )


@pytest.mark.parametrize(
    ("question", "expected_group"),
    [
        ("Na ata de julho de 2024, quais eram as projecoes?", MEETING_DISAMBIGUATION),
        ("Quem votou na 279a reuniao?", MEETING_DISAMBIGUATION),
        ("Em qual reuniao a Selic foi reduzida para 12,75% a.a.?", REVERSE_LOOKUP),
        ("Quais eram as expectativas do Focus para 2026 e 2027?", REVERSE_LOOKUP),
    ],
)
def test_grouping_follows_the_rule(question, expected_group):
    assert classify(question, DOCS) == expected_group


def test_a_hint_the_corpus_cannot_satisfy_joins_neither_group():
    """The 280th meeting does not exist. Scoring it as a disambiguation
    failure would blame the retriever for a question the corpus cannot answer;
    scoring it as reverse lookup would credit the retriever with content
    reasoning it never did."""
    assert classify("Qual foi a decisao do Copom na 280a reuniao?", DOCS) is None


def test_groups_partition_and_do_not_overlap():
    queries = [
        scored("a", "Na ata de julho de 2024?", "ata-jul-2024", ["ata-jul-2024"]),
        scored("b", "Em qual reuniao a Selic caiu para 12,75%?", "ata-jun-2026", ["ata-jul-2024"]),
        scored("c", "Decisao da 280a reuniao?", "ata-jun-2026", ["ata-jun-2026"]),
    ]
    groups = assign_groups(queries, DOCS)
    ids = [sq.row.id for members in groups.values() for sq in members]
    assert sorted(ids) == ["a", "b"]  # "c" belongs to neither
    assert len(set(ids)) == len(ids)


def test_rank1_accuracy_counts_the_top_hit_only():
    queries = [
        scored("a", "Na ata de julho de 2024?", "ata-jul-2024", ["ata-jul-2024", "ata-jun-2026"]),
        # right document present, but at rank 2 — a miss for the probe.
        scored("b", "Na ata de junho de 2026?", "ata-jun-2026", ["ata-jul-2024", "ata-jun-2026"]),
    ]
    group = run_probes(queries, DOCS)[MEETING_DISAMBIGUATION]
    assert group.n == 2
    assert group.rank1_doc_accuracy == 0.5
    assert group.doc_in_top3 == 1.0
    assert [f["id"] for f in group.failures] == ["b"]


def test_hint_diagnostics_flags_a_confidently_wrong_filter():
    """The failure that matters most: a hint that resolves, and resolves wrong.

    It removes the right document before any later stage can rescue it, so it
    has to be counted separately from a hint that simply does not fire.
    """
    queries = [
        scored("a", "Na ata de julho de 2024?", "ata-jun-2026", ["ata-jul-2024"]),
        scored("b", "Na ata de junho de 2026?", "ata-jun-2026", ["ata-jun-2026"]),
    ]
    diagnostics = hint_diagnostics(queries, DOCS)
    assert diagnostics["hint_present"] == 2
    assert diagnostics["hint_resolved"] == 2
    assert diagnostics["hint_resolved_and_correct"] == 1
    assert diagnostics["precision_when_resolved"] == 0.5
    assert [w["id"] for w in diagnostics["resolved_but_wrong"]] == ["a"]


def test_hint_diagnostics_ignores_questions_without_a_hint():
    queries = [scored("a", "Em qual reuniao a Selic caiu?", "ata-jun-2026", ["ata-jun-2026"])]
    diagnostics = hint_diagnostics(queries, DOCS)
    assert diagnostics["hint_present"] == 0
    assert diagnostics["precision_when_resolved"] is None
