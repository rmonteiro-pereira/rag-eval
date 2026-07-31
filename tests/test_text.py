"""Tokenizer tests.

The decimal weld is the one piece of cleverness in `retrieval.text`, and
cleverness that touches numbers in a corpus about interest rates has to be
pinned from both sides: it must join what `pypdf` split, and it must leave
year lists alone.
"""

from __future__ import annotations

from retrieval.text import normalise, strip_accents, tokenize


def test_strip_accents_folds_portuguese():
    assert strip_accents("março") == "marco"
    assert strip_accents("reunião") == "reuniao"
    assert strip_accents("política monetária") == "politica monetaria"


def test_normalise_welds_pdf_split_decimals():
    # pypdf renders `0,25` as `0, 25` and `14,25` as `14 , 25`.
    assert "0,25" in normalise("reducao de 0, 25 ponto percentual")
    assert "14,25" in normalise("para 14 , 25% a.a.")


def test_normalise_leaves_year_lists_alone():
    """The failure mode that would silently destroy half the gold set."""
    out = normalise("expectativas para 2022, 2023 e 2024")
    assert "2022" in out and "2023" in out and "2024" in out
    assert "2022,2023" not in out


def test_tokenize_keeps_rates_as_single_tokens():
    tokens = tokenize("Em qual reuniao o Copom reduziu a Selic para 13,25% a.a.?")
    assert "13,25" in tokens
    assert "13" not in tokens and "25" not in tokens


def test_tokenize_normalises_decimal_point_to_comma():
    assert tokenize("taxa de 13.25%") == tokenize("taxa de 13,25%")


def test_tokenize_drops_stopwords_but_keeps_content():
    tokens = tokenize("Qual foi a decisao do Copom sobre a taxa Selic?")
    assert "qual" not in tokens and "foi" not in tokens
    assert "decisao" in tokens and "copom" in tokens and "selic" in tokens


def test_tokenize_can_keep_stopwords():
    assert "qual" in tokenize("Qual decisao?", drop_stopwords=False)


def test_normalise_returns_exactly_the_repaired_text():
    """Kills `_SPLIT_DECIMAL.sub(r"\1\2\3", ...)` -> `r"XX\1\2\3XX"`.

    Every other tokenizer test asserts membership — `"13,25" in tokens` — which
    stays true when the substitution injects garbage around the match, because
    the tokenizer then simply yields the garbage as separate tokens. Asserting
    the exact string is what makes the repair itself testable.
    """
    assert normalise("taxa de 13. 25%") == "taxa de 13.25%"
    assert normalise("Reuniao de MARCO") == "reuniao de marco"
    assert normalise("em 2022, 2023 e 2024") == "em 2022, 2023 e 2024"
