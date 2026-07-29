"""The wrong-meeting probes.

The headline retrieval metrics answer "did we retrieve the right chunk". They do
not answer the question this corpus actually fails on, which is coarser and
nastier:

    Thirty documents contain a paragraph reading
    "O Copom decidiu <verb> a taxa basica de juros para <rate>% a.a., e entende
    que essa decisao e compativel com a estrategia de convergencia da
    inflacao..."
    — did we return the one the user asked about?

So the probes score a single boolean per query: **is the rank-1 hit from the
document the gold row names**. No graded gain, no cutoffs. Rank-1 because that
is the passage a generator leads with and the one a user reads.

Two groups, split by a mechanical rule rather than by hand-picking, so the split
survives the gold set growing:

* **`meeting_disambiguation`** — the question names its meeting
  (`retrieval.metadata.parse_hint` resolves to at least one document in the
  corpus). Thirty near-identical candidates, and the question told us which one.
  Failing here is the defect in its purest form: the information needed was in
  the query and was thrown away.

* **`reverse_lookup`** — the question names no meeting and asks the corpus to
  identify one from its content ("Em qual reuniao o Copom reduziu a Selic para
  13,25% a.a.?"). Metadata filtering is structurally unable to help; the only
  thing that can is a retriever that actually reads the numbers. This is the
  control group, and it is what stops "we fixed it with a regex" from being the
  whole story.

A row can fall in neither group (it names a meeting the corpus does not have) or
— by construction, never — in both: the groups are defined by whether a hint
resolves, so they partition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from eval.scoring import ScoredQuery
from retrieval.metadata import DocumentMeta, parse_hint, resolve

MEETING_DISAMBIGUATION = "meeting_disambiguation"
REVERSE_LOOKUP = "reverse_lookup"

PROBE_DESCRIPTIONS = {
    MEETING_DISAMBIGUATION: (
        "the question names a meeting that exists in the corpus; rank-1 must be that meeting"
    ),
    REVERSE_LOOKUP: (
        "the question names no meeting and must be answered from content alone; "
        "metadata filtering cannot help here"
    ),
}


@dataclass(frozen=True)
class ProbeGroup:
    name: str
    description: str
    ids: tuple[str, ...]
    n: int
    rank1_doc_accuracy: float
    doc_in_top3: float
    doc_in_top5: float
    failures: tuple[dict, ...]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "n": self.n,
            "rank1_doc_accuracy": self.rank1_doc_accuracy,
            "doc_in_top3": self.doc_in_top3,
            "doc_in_top5": self.doc_in_top5,
            "ids": list(self.ids),
            "failures": [dict(f) for f in self.failures],
        }


def classify(question: str, documents: Sequence[DocumentMeta]) -> str | None:
    """Which probe group a question belongs to, or None if neither."""
    hint = parse_hint(question)
    if not hint:
        return REVERSE_LOOKUP
    return MEETING_DISAMBIGUATION if resolve(hint, documents) else None


def assign_groups(
    scored: Sequence[ScoredQuery],
    documents: Sequence[DocumentMeta],
) -> dict[str, list[ScoredQuery]]:
    groups: dict[str, list[ScoredQuery]] = {MEETING_DISAMBIGUATION: [], REVERSE_LOOKUP: []}
    for sq in scored:
        group = classify(sq.row.question, documents)
        if group is not None:
            groups[group].append(sq)
    return groups


def _summarise(name: str, members: Sequence[ScoredQuery]) -> ProbeGroup:
    n = len(members)
    if n == 0:
        return ProbeGroup(
            name=name,
            description=PROBE_DESCRIPTIONS[name],
            ids=(),
            n=0,
            rank1_doc_accuracy=0.0,
            doc_in_top3=0.0,
            doc_in_top5=0.0,
            failures=(),
        )
    failures = tuple(
        {
            "id": sq.row.id,
            "question": sq.row.question,
            "expected_doc_id": sq.row.source_doc_id,
            "rank1_doc_id": sq.rank1_doc_id,
            "expected_doc_in_top5": sq.doc_in_top(5),
        }
        for sq in members
        if not sq.rank1_doc_correct
    )
    return ProbeGroup(
        name=name,
        description=PROBE_DESCRIPTIONS[name],
        ids=tuple(sq.row.id for sq in members),
        n=n,
        rank1_doc_accuracy=sum(sq.rank1_doc_correct for sq in members) / n,
        doc_in_top3=sum(sq.doc_in_top(3) for sq in members) / n,
        doc_in_top5=sum(sq.doc_in_top(5) for sq in members) / n,
        failures=failures,
    )


def run_probes(
    scored: Sequence[ScoredQuery],
    documents: Sequence[DocumentMeta],
) -> dict[str, ProbeGroup]:
    groups = assign_groups(scored, documents)
    return {name: _summarise(name, members) for name, members in groups.items()}


def hint_diagnostics(
    scored: Sequence[ScoredQuery],
    documents: Sequence[DocumentMeta],
) -> dict:
    """How often the meeting hint is present, resolvable, and *correct*.

    Reported separately from the probe accuracies because a filter that is
    confidently wrong is far worse than one that abstains: it removes the right
    document from the candidate pool before any other stage can rescue it.
    `resolved_but_wrong` is the number to watch.
    """
    present = resolved = correct = 0
    wrong: list[dict] = []
    for sq in scored:
        hint = parse_hint(sq.row.question)
        if not hint:
            continue
        present += 1
        doc_ids = resolve(hint, documents)
        if not doc_ids:
            continue
        resolved += 1
        if sq.row.source_doc_id in doc_ids:
            correct += 1
        else:
            wrong.append(
                {
                    "id": sq.row.id,
                    "question": sq.row.question,
                    "hint": hint.describe(),
                    "resolved_to": sorted(doc_ids),
                    "expected_doc_id": sq.row.source_doc_id,
                }
            )
    return {
        "n_scored": len(scored),
        "hint_present": present,
        "hint_resolved": resolved,
        "hint_resolved_and_correct": correct,
        "precision_when_resolved": (correct / resolved) if resolved else None,
        "resolved_but_wrong": wrong,
        "note": (
            "hint_present counts questions naming a month+year or a meeting ordinal; "
            "hint_resolved counts those that matched a document in this corpus. A hint "
            "that resolves to nothing (e.g. the 280th meeting) disables the filter "
            "rather than emptying the candidate pool."
        ),
    }
