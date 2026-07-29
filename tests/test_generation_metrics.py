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
