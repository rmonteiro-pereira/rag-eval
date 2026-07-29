"""Judge output parsing and aggregation.

A local 3B/8B model asked for JSON returns JSON *sometimes*. The parser's job is
to recover a verdict when one is recoverable and to record a failure when it is
not — never to invent a score. A judge that silently returns 0 on malfunction
makes the system look broken; one that silently returns 2 makes it look perfect.
Both are worse than a recorded null, and that is what these tests pin.
"""

from __future__ import annotations

from generation.judge import Judge, Judgement, aggregate_judgements, parse_judgement

CLEAN = '{"faithfulness": 2, "faithfulness_reason": "tudo consta nos trechos", "answer_relevance": 1, "answer_relevance_reason": "responde em parte"}'  # noqa: E501


def test_parses_a_clean_verdict():
    verdict = parse_judgement(CLEAN, "llama3.1")
    assert verdict.faithfulness == 2
    assert verdict.answer_relevance == 1
    assert verdict.faithfulness_reason == "tudo consta nos trechos"
    assert verdict.ok


def test_parses_a_verdict_wrapped_in_prose_and_fences():
    raw = f"Claro! Aqui esta minha avaliacao:\n```json\n{CLEAN}\n```\nEspero ter ajudado."
    assert parse_judgement(raw, "llama3.1").faithfulness == 2


def test_takes_the_last_object_when_the_model_restates_the_schema_first():
    """Small models echo the template, then fill it in. The second one is real."""
    template = '{"faithfulness": 0, "faithfulness_reason": "<uma frase>", "answer_relevance": 0, "answer_relevance_reason": "<uma frase>"}'  # noqa: E501
    verdict = parse_judgement(f"O formato e {template}\n\nMinha resposta:\n{CLEAN}", "llama3.1")
    assert verdict.faithfulness == 2
    assert verdict.answer_relevance == 1


def test_a_score_outside_the_rubric_is_dropped_not_clamped():
    """`5` is not a 2. Clamping would silently invent a verdict the judge never
    gave, which is exactly the failure this parser exists to avoid."""
    verdict = parse_judgement('{"faithfulness": 5, "answer_relevance": 2}', "llama3.1")
    assert verdict.faithfulness is None
    assert verdict.answer_relevance == 2
    assert not verdict.ok


def test_unparsable_output_records_a_failure_rather_than_a_score():
    verdict = parse_judgement("Acho que a resposta esta boa, nota alta.", "llama3.1")
    assert verdict.faithfulness is None
    assert verdict.answer_relevance is None
    assert verdict.parse_error
    assert not verdict.ok


def test_empty_output_does_not_crash():
    assert parse_judgement("", "llama3.1").parse_error


class _ExplodingLLM:
    name = "broken"
    backend = "ollama"

    def complete(self, system, prompt):  # noqa: ARG002
        raise TimeoutError("model took too long")


def test_a_dead_judge_does_not_kill_the_run():
    verdict = Judge(llm=_ExplodingLLM()).judge("q", "ctx", "a")
    assert not verdict.ok
    assert "TimeoutError" in verdict.parse_error


def verdict(faithfulness, relevance) -> Judgement:
    return Judgement(
        faithfulness=faithfulness,
        faithfulness_reason="",
        answer_relevance=relevance,
        answer_relevance_reason="",
        judge_model="llama3.1",
    )


def test_aggregate_excludes_failures_from_the_means_but_counts_them():
    failed = Judgement(None, "", None, "", "llama3.1", parse_error="boom")
    aggregate = aggregate_judgements([verdict(2, 2), verdict(0, 0), failed])
    assert aggregate["n"] == 3
    assert aggregate["n_parsed"] == 2
    assert aggregate["parse_failure_rate"] == 1 / 3
    assert aggregate["faithfulness_mean"] == 1.0
    assert aggregate["faithfulness_at_2"] == 0.5


def test_aggregate_of_all_failures_reports_none_not_zero():
    failed = Judgement(None, "", None, "", "llama3.1", parse_error="boom")
    aggregate = aggregate_judgements([failed])
    assert aggregate["faithfulness_mean"] is None
    assert aggregate["parse_failure_rate"] == 1.0
