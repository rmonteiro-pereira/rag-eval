"""Meeting-hint parsing and resolution.

The parser is the fix for the wrong-meeting defect, so its false positives
matter more than its recall: a hint that resolves to the wrong document removes
the right one from the candidate pool before any later stage can rescue it.
Most of these tests are therefore about what it must *not* claim to know.
"""

from __future__ import annotations

import pytest

from retrieval.metadata import DocumentMeta, parse_hint, resolve, resolve_question

DOCS = [
    DocumentMeta("2024-05-08-262", "262ª Reunião - 7-8 maio, 2024", "2024-05-08"),
    DocumentMeta("2024-07-31-264", "264ª Reunião - 30-31 julho, 2024", "2024-07-31"),
    DocumentMeta("2023-11-01-258", "258ª Reunião - 31 outubro-1 novembro, 2023", "2023-11-01"),
    DocumentMeta("2026-06-17-279", "279ª Reunião - 16-17 junho, 2026", "2026-06-17"),
]


def test_parses_month_and_year():
    hint = parse_hint("Na ata de julho de 2024, quais eram as projecoes?")
    assert hint.year_months == frozenset({(2024, 7)})
    assert not hint.ordinals


def test_parses_meeting_ordinal():
    hint = parse_hint("Quem votou pela decisao da 279a reuniao do Copom?")
    assert hint.ordinals == frozenset({279})


def test_parses_accented_ordinal_and_month():
    hint = parse_hint("Qual a decisao da 279ª Reunião de junho de 2026?")
    assert hint.ordinals == frozenset({279})
    assert hint.year_months == frozenset({(2026, 6)})


def test_forecast_horizons_are_not_dates():
    """The false positive that would wreck half the gold set.

    `expectativas de inflacao para 2026 e 2027` names two years and no meeting.
    Reading either as a meeting date would filter the corpus to the wrong ata.
    """
    hint = parse_hint("Quais eram as expectativas de inflacao do Focus para 2026 e 2027?")
    assert not hint
    assert hint.year_months == frozenset()


def test_month_far_from_year_is_not_paired():
    hint = parse_hint(
        "Em maio o Copom discutiu diversos temas de natureza estrutural e regulatoria, "
        "bem como o cenario externo, antes de projetar o horizonte de 2027."
    )
    assert hint.year_months == frozenset()


def test_reverse_lookup_questions_carry_no_hint():
    assert not parse_hint("Em qual reuniao o Copom reduziu a taxa Selic para 13,25% a.a.?")
    assert not parse_hint("Em qual ata o cenario de referencia partia de R$6,00/US$?")


def test_bare_year_is_never_a_hint():
    assert not parse_hint("Qual era a taxa Selic definida pelo Copom em 2019?")


def test_resolves_month_year_to_one_document():
    assert resolve_question("Na ata de julho de 2024, ...", DOCS) == {"2024-07-31-264"}


def test_resolves_ordinal_and_ignores_the_month():
    """An ordinal is exact; widening it with a month could only ever hurt."""
    assert resolve_question("Decisao da 279a reuniao, de junho de 2026?", DOCS) == {
        "2026-06-17-279"
    }


def test_straddling_months_resolve_to_the_union():
    """`outubro/novembro de 2023` is one meeting recorded under November."""
    assert resolve_question(
        "Qual foi a decisao na reuniao de outubro/novembro de 2023?", DOCS
    ) == {"2023-11-01-258"}


def test_unresolvable_hint_returns_empty_not_garbage():
    """The 280th meeting does not exist; the caller must retrieve unfiltered."""
    hint = parse_hint("Qual foi a decisao do Copom na 280a reuniao?")
    assert hint.ordinals == frozenset({280})
    assert resolve(hint, DOCS) == set()


def test_out_of_corpus_month_returns_empty():
    assert resolve_question("Qual a decisao da reuniao de janeiro de 2019?", DOCS) == set()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Qual foi a decisao na reuniao de marco de 2025?", frozenset({(2025, 3)})),
        ("Qual foi a decisao na reunião de março de 2025?", frozenset({(2025, 3)})),
    ],
)
def test_accent_folding_makes_ascii_queries_match(question, expected):
    assert parse_hint(question).year_months == expected


def test_document_meta_parses_its_own_ordinal():
    assert DOCS[3].ordinal == 279
    assert DOCS[3].year_month == (2026, 6)
