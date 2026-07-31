"""Deterministic generation metrics.

These are the metrics the report leans on precisely because they do not involve
a model, so they have to be right. The hallucination detector in particular is
the one number in the generation suite that can be trusted without a human, and
it is only worth that if it neither misses a fabricated rate nor cries wolf over
a number the user typed.
"""

from __future__ import annotations

import pytest

from eval.metrics.generation import (
    GenerationScores,
    abstained,
    aggregate_generation,
    lexical_groundedness,
    numbers_in,
    numeric_recall,
    unsupported_numbers,
)

CONTEXT = (
    "[1] 279a Reuniao - pagina 6\n"
    "O Copom decidiu reduzir a taxa basica de juros para 14,25% a.a., e entende que "
    "essa decisao e compativel com a estrategia de convergencia da inflacao."
)


def test_numbers_survive_as_whole_rates():
    assert "14,25" in numbers_in("para 14,25% a.a.")
    assert "14" not in numbers_in("para 14,25% a.a.")


def test_decimal_point_and_comma_are_the_same_number():
    assert numbers_in("13.25") == numbers_in("13,25")


def test_pdf_split_decimals_are_welded_before_comparison():
    assert "0,25" in numbers_in("reducao de 0, 25 ponto percentual")


def test_numeric_recall_is_full_when_every_rate_is_repeated():
    assert numeric_recall("Selic em 14,25% a.a.", "O Copom fixou a Selic em 14,25% ao ano.") == 1.0


def test_numeric_recall_is_partial_when_one_of_two_is_missing():
    score = numeric_recall("elevacao de 1,00 p.p. para 12,25%", "elevou para 12,25% a.a.")
    assert score == 0.5


def test_numeric_recall_is_none_when_the_reference_states_no_number():
    """An abstention row, or a list of names. Zero would report a failure that
    was never possible."""
    assert numeric_recall("Nao encontrei essa informacao.", "Nao encontrei.") is None


def test_a_fabricated_rate_is_caught():
    answer = "O Copom reduziu a Selic para 13,75% a.a."
    assert unsupported_numbers(answer, CONTEXT) == ["13,75"]


def test_a_quoted_rate_is_not_flagged():
    answer = "O Copom decidiu reduzir a taxa para 14,25% a.a."
    assert unsupported_numbers(answer, CONTEXT) == []


def test_echoing_a_number_from_the_question_is_not_hallucination():
    """Repeating the year the user asked about is quoting them, not inventing."""
    question = "Qual foi a decisao da reuniao de junho de 2026?"
    answer = "Em 2026 o Copom reduziu a taxa para 14,25% a.a."
    assert unsupported_numbers(answer, CONTEXT, question) == []
    # Without the question in the allowed set, the same echo reads as invented —
    # which is why the question is part of the allowed set.
    assert unsupported_numbers(answer, CONTEXT) == ["2026"]


def test_verbatim_quote_is_fully_grounded():
    quote = "O Copom decidiu reduzir a taxa basica de juros para 14,25% a.a."
    assert lexical_groundedness(quote, CONTEXT) == 1.0


def test_invented_content_lowers_groundedness():
    invented = "O Copom citou o desemprego, a balanca comercial e o petroleo Brent."
    assert lexical_groundedness(invented, CONTEXT) < 0.3


def test_groundedness_is_none_for_an_empty_answer():
    assert lexical_groundedness("", CONTEXT) is None


@pytest.mark.parametrize(
    "text",
    [
        "Nao encontrei essa informacao nos documentos recuperados.",
        "Não encontrei essa informação nos documentos recuperados.",
        "Os trechos nao contem essa informacao.",
        "Nao e possivel responder com base nos trechos.",
    ],
)
def test_refusals_are_detected_across_paraphrases(text):
    assert abstained(text)


def test_an_actual_answer_is_not_read_as_a_refusal():
    assert not abstained("O Copom decidiu reduzir a taxa para 14,25% a.a.")


def scores(**kwargs) -> GenerationScores:
    base = dict(
        numeric_recall=1.0,
        unsupported_numbers=(),
        lexical_groundedness=1.0,
        abstained=False,
        cited_gold_doc=True,
        n_citations=5,
    )
    base.update(kwargs)
    return GenerationScores(**base)


def test_abstention_is_scored_in_the_opposite_direction_from_answerable_rows():
    """The metric that would be meaningless if the two were averaged together.

    Refusing a negative is correct; refusing an answerable row is a failure.
    One mean over both would rise for two opposite reasons.
    """
    aggregate = aggregate_generation(
        [scores(abstained=True), scores(abstained=True)],
        is_abstention=[False, True],
    )
    assert aggregate["false_refusal_rate"] == 1.0
    assert aggregate["abstention_correctness"] == 1.0
    assert aggregate["n_answerable"] == 1 and aggregate["n_negative"] == 1


def test_aggregate_reports_none_rather_than_zero_for_an_empty_slice():
    aggregate = aggregate_generation([scores()], is_abstention=[False])
    assert aggregate["abstention_correctness"] is None


def test_hallucination_rate_counts_answerable_rows_only():
    aggregate = aggregate_generation(
        [scores(unsupported_numbers=("13,75",)), scores()],
        is_abstention=[False, False],
    )
    assert aggregate["hallucinated_number_rate"] == 0.5


def test_aggregate_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        aggregate_generation([scores()], is_abstention=[False, True])


def test_citation_markers_are_not_read_as_hallucinated_numbers():
    """`[2, 19]` is a pointer to passages 2 and 19, not the decimal 2.19.

    This is a regression test for a false positive that actually fired: the
    tokenizer welds `2, 19` into `2,19` (see `_SPLIT_DECIMAL` in
    `retrieval/text.py`), that value appears in no passage, and the row was
    scored as a hallucinated number. One row was enough to move
    `hallucinated_number_rate` from 0.000 to 0.020.
    """
    answer = (
        "A projecao era de 3,7% [2, Tabela 1 e 17]. O Comite julgou como mais "
        "adequadas trajetorias menos discrepantes [2, 19]."
    )
    context = "A projecao para o quarto trimestre de 2027 era de 3,7% no cenario de referencia."
    assert unsupported_numbers(answer, context) == []


def test_stripping_citations_does_not_hide_a_real_hallucination():
    """The fix must not become a way for invented numbers to pass."""
    context = "A projecao para 2027 era de 3,7% no cenario de referencia."
    assert unsupported_numbers("A Selic foi reduzida para 9,99% a.a. [1]", context) == ["9,99"]


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("A Selic foi para [9,99%] a.a.", ["9,99"]),
        ("A taxa ficou em [14,25] pontos.", ["14,25"]),
        ("O IPCA acumulado foi de [7,5] por cento.", ["7,5"]),
    ],
)
def test_a_fabricated_number_inside_brackets_is_still_caught(answer, expected):
    r"""Brackets must not be an escape hatch for the hallucination metric.

    A permissive `\[.*\]` would strip `[9,99%]` — a rate no passage contains —
    and report a clean run. `_CITATION` therefore requires the bracket to OPEN
    with a passage index that is not glued to a decimal mark, so a bracketed
    claim is never mistaken for a pointer.
    """
    context = "A projecao para 2027 era de 3,7% no cenario de referencia."
    assert unsupported_numbers(answer, context) == expected


def test_citation_shapes_the_generator_actually_emits_are_all_stripped():
    """`generation/prompt.py` asks for `[n]`; models embellish it in these ways."""
    context = "A projecao era de 3,7%."
    for answer in (
        "A projecao era de 3,7% [2].",
        "A projecao era de 3,7% [2, 19].",
        "A projecao era de 3,7% [1, Tabela 1 e 17].",
        "A projecao era de 3,7% [ 3 ; 4 ].",
    ):
        assert unsupported_numbers(answer, context) == [], answer


def test_the_summary_survives_an_arm_with_no_rows():
    """`--min-status validated` returns zero rows today, by design.

    That is a documented state, not a hypothetical, so the reporting path has to
    survive it. It previously raised `IndexError` computing a median over an
    empty list — the one code path the repo's own headline caveat guarantees
    someone will hit.
    """
    from eval.run_generation import _render_summary

    empty = dict.fromkeys(
        (
            "numeric_recall",
            "lexical_groundedness",
            "hallucinated_number_rate",
            "citation_correctness",
            "abstention_correctness",
            "false_refusal_rate",
        )
    )
    report = {
        "setup": {"retriever": "hybrid+rerank+metadata", "judge_model": "llama3.1"},
        "gold": {"n_rows": 0, "n_negative": 0},
        "arms": [
            {
                "arm": "extractive",
                "backend": "extractive",
                "deterministic": empty,
                "judge": None,
                "latency": {"median_ms": None},
                "per_row": [],
            }
        ],
        "calibration": {
            "n_items": 0,
            "human_labels_filled": 0,
            "sheet_path": "eval/datasets/judge_calibration_sheet.jsonl",
            "agreement": {"criteria": {}},
            "note": "no human labels yet",
        },
        "caveat": "draft",
    }
    out = _render_summary(report)
    assert "n/a" in out
