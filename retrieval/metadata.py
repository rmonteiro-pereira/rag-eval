"""Resolving a question to the Copom meeting it is asking about.

**This is the module that exists because of the wrong-meeting defect.**

Thirty Copom minutes, each with a section headed *Decisão de política
monetária*, each phrasing it within a few words of the others:

    O Copom decidiu reduzir a taxa basica de juros para 14,25% a.a., e entende
    que essa decisao e compativel com a estrategia de convergencia da inflacao...

Dense retrieval on that corpus does exactly what it is asked to: it finds the
paragraph that is closest in meaning. It has no reason to prefer June 2026's
copy over March 2025's, because semantically there is barely a difference. The
baseline's recall@1 of 0.05 is not a bug in the embedder — it is the embedder
answering a question the user did not ask.

But the user *did* say which meeting. Forty of the forty-nine answerable gold
questions name it outright — "na ata de julho de 2024", "na 279a reuniao". That
is a hard constraint sitting in plain text, thrown away by an encoder that maps
`julho de 2024` and `maio de 2024` to nearly the same point. Recovering it with
a regex and turning it into a payload filter is unglamorous and it is the single
largest measured win in the ablation.

## What counts as a hint

* **Meeting ordinal** — `279a reuniao`, `279ª Reunião`. Strongest signal: it
  identifies exactly one document. Wins outright when present.
* **Month + year** — `de julho de 2024`. The year must sit within a short
  distance *after* the month name, which is what stops
  `expectativas de inflacao para 2026 e 2027` from being read as a date: those
  years are forecast horizons, they follow no month, and half the gold set
  contains one. A bare year is never a hint.

Several months may resolve together — `reuniao de outubro/novembro de 2023`
names one meeting that straddles two months, so both are resolved and the union
is taken.

## What happens when the hint resolves to nothing

Nothing. `resolve` returns an empty set and the caller retrieves unfiltered. A
question about the *280th* meeting (which does not exist in a corpus ending at
the 279th) must not be answered by filtering the corpus down to zero documents
and reporting an empty result as if it were retrieval quality. That an
unresolvable hint is a strong abstention signal is true and is used in the
guardrail suite, not here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from retrieval.text import normalise

MONTHS_PT: dict[str, int] = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}

_MONTH = re.compile(r"\b(" + "|".join(MONTHS_PT) + r")\b")
_YEAR = re.compile(r"\b(19|20)(\d{2})\b")

#: `279a reuniao`, `279ª reunião`, `279 reuniao`. The ordinal marker is folded
#: to `a` by `normalise`, so only the ASCII form needs matching here.
_ORDINAL = re.compile(r"\b(\d{2,4})\s*a?\s*[-\s]*reuni")

#: How far after a month name the year may sit and still belong to it.
#: `julho de 2024` needs 4; the slack covers `julho/agosto de 2024`.
_YEAR_WINDOW = 24


@dataclass(frozen=True)
class MeetingHint:
    """What a question said about *which* meeting it is asking about."""

    ordinals: frozenset[int] = frozenset()
    year_months: frozenset[tuple[int, int]] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.ordinals or self.year_months)

    def describe(self) -> str:
        parts = []
        if self.ordinals:
            parts.append("reuniao " + "/".join(str(n) for n in sorted(self.ordinals)))
        if self.year_months:
            parts.append(
                " ".join(f"{y:04d}-{m:02d}" for y, m in sorted(self.year_months)),
            )
        return "; ".join(parts) or "none"


def parse_hint(question: str) -> MeetingHint:
    """Extract meeting ordinals and month+year pairs from a question."""
    text = normalise(question)

    ordinals = {int(match.group(1)) for match in _ORDINAL.finditer(text)}

    year_months: set[tuple[int, int]] = set()
    for match in _MONTH.finditer(text):
        month = MONTHS_PT[match.group(1)]
        window = text[match.end() : match.end() + _YEAR_WINDOW]
        year_match = _YEAR.search(window)
        if year_match:
            year_months.add((int(year_match.group(0)), month))

    return MeetingHint(ordinals=frozenset(ordinals), year_months=frozenset(year_months))


@dataclass(frozen=True)
class DocumentMeta:
    """The metadata the resolver needs about one indexed document."""

    doc_id: str
    title: str
    reference_date: str  # YYYY-MM-DD

    @property
    def ordinal(self) -> int | None:
        """Meeting number parsed from the title (`279ª Reunião - ...`)."""
        match = _ORDINAL.search(normalise(self.title))
        return int(match.group(1)) if match else None

    @property
    def year_month(self) -> tuple[int, int] | None:
        try:
            year, month, _ = self.reference_date.split("-", 2)
            return int(year), int(month)
        except (ValueError, AttributeError):
            return None


def document_meta(records: Iterable) -> list[DocumentMeta]:
    """Collapse a chunk-level corpus down to one entry per document."""
    seen: dict[str, DocumentMeta] = {}
    for record in records:
        if record.doc_id not in seen:
            seen[record.doc_id] = DocumentMeta(
                doc_id=record.doc_id,
                title=record.title,
                reference_date=record.reference_date,
            )
    return list(seen.values())


def resolve(hint: MeetingHint, documents: Iterable[DocumentMeta]) -> set[str]:
    """Document ids the hint points at; empty set when it points at nothing.

    An ordinal that matches wins outright — it is exact, and pairing it with a
    month would only ever widen a set that is already right.
    """
    documents = list(documents)
    if not hint:
        return set()

    by_ordinal = {doc.doc_id for doc in documents if doc.ordinal in hint.ordinals}
    if by_ordinal:
        return by_ordinal

    return {doc.doc_id for doc in documents if doc.year_month in hint.year_months}


def resolve_question(question: str, documents: Iterable[DocumentMeta]) -> set[str]:
    """`parse_hint` + `resolve`, the form the retriever actually calls."""
    return resolve(parse_hint(question), documents)
