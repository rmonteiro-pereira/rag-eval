"""SQL validation, the HITL gate, and the agent loop's control flow.

The SQL tool executes model-written statements, so its validator is the most
security-relevant code in the repo and is tested from the hostile direction
first: what must it refuse. The gate is tested for the property that actually
matters — that a refusal *stops execution* rather than merely being recorded.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from agent.hitl import (
    POLICY_AUTO,
    POLICY_DENY,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    ConfirmationGate,
    assess,
)
from agent.loop import Agent, parse_action
from agent.tools import DEFAULT_ROW_LIMIT, SqlQueryTool, SqlRejected, normalise_sql

# --------------------------------------------------------------------------
# SQL validation — what it must refuse
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE mart_fx",
        "DELETE FROM mart_fx",
        "UPDATE mart_fx SET rate = 0",
        "INSERT INTO mart_fx VALUES (1)",
        "CREATE TABLE evil AS SELECT 1",
        "ATTACH 'other.db' AS other",
        "COPY mart_fx TO 'out.csv'",
        "PRAGMA database_list",
    ],
)
def test_mutating_statements_are_refused(sql):
    with pytest.raises(SqlRejected):
        normalise_sql(sql)


def test_a_second_statement_is_refused_before_anything_else_is_checked():
    """`; DROP` must not survive because the first statement looked fine."""
    with pytest.raises(SqlRejected, match="multiple statements"):
        normalise_sql("SELECT 1; DROP TABLE mart_fx")


def test_an_empty_statement_is_refused():
    with pytest.raises(SqlRejected, match="empty"):
        normalise_sql("   ")


def test_a_column_named_like_a_forbidden_verb_does_not_trip_the_rule():
    """Whole-word matching: `updated_at` is not `UPDATE`."""
    out = normalise_sql("SELECT updated_at, created_at FROM mart_fx WHERE rate > 5")
    assert "updated_at" in out


def test_select_and_with_are_both_accepted():
    assert normalise_sql("SELECT 1 LIMIT 1").startswith("SELECT")
    assert normalise_sql("WITH t AS (SELECT 1) SELECT * FROM t LIMIT 1").startswith("WITH")


def test_a_missing_limit_is_added():
    assert f"LIMIT {DEFAULT_ROW_LIMIT}" in normalise_sql("SELECT * FROM mart_macro_dashboard")


def test_an_existing_limit_is_left_alone():
    assert normalise_sql("SELECT 1 LIMIT 3").count("LIMIT") == 1


def test_a_trailing_semicolon_is_tolerated():
    assert normalise_sql("SELECT 1 LIMIT 1;") == "SELECT 1 LIMIT 1"


# --------------------------------------------------------------------------
# Risk assessment
# --------------------------------------------------------------------------


def test_a_bounded_read_of_a_small_mart_is_low_risk():
    sql = "SELECT month, selic_target FROM mart_macro_dashboard WHERE month >= '2026-01-01'"
    assert assess(sql).level == RISK_LOW


def test_an_unbounded_scan_of_a_large_mart_is_high_risk():
    """1.6M rows is a context-window incident, not an answer."""
    assessment = assess("SELECT * FROM mart_futures_curve")
    assert assessment.level == RISK_HIGH
    assert "mart_futures_curve" in assessment.tables


def test_a_filtered_read_of_a_large_mart_is_medium():
    sql = "SELECT * FROM mart_yield_curve WHERE date >= '2026-01-01'"
    assert assess(sql).level == RISK_MEDIUM


def test_many_joins_are_high_risk():
    sql = (
        "SELECT * FROM mart_macro_dashboard a "
        "JOIN mart_real_interest b ON a.month = b.month "
        "JOIN mart_inflation_panel c ON a.month = c.month WHERE a.month > '2020-01-01'"
    )
    assert assess(sql).level == RISK_HIGH


def test_sql_the_validator_rejects_is_automatically_high_risk():
    """Risk is assessed on the statement that would actually run, so anything
    the validator refuses cannot be classified as anything but high."""
    assessment = assess("DROP TABLE mart_fx")
    assert assessment.level == RISK_HIGH
    assert "rejected by the SQL validator" in assessment.reasons[0]


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_the_auto_policy_still_refuses_high_risk_with_nobody_watching():
    """The property that stops the demo gate from being a screenshot."""
    gate = ConfirmationGate(policy=POLICY_AUTO)
    decision = gate.confirm("SELECT * FROM mart_futures_curve")
    assert not decision.approved
    assert decision.risk.level == RISK_HIGH
    assert "needs a human" in decision.reason


def test_the_auto_policy_approves_ordinary_reads():
    gate = ConfirmationGate(policy=POLICY_AUTO)
    sql = "SELECT month FROM mart_macro_dashboard WHERE month >= '2026-01-01'"
    assert gate.confirm(sql).approved


def test_the_deny_policy_refuses_everything():
    gate = ConfirmationGate(policy=POLICY_DENY)
    assert not gate.confirm("SELECT 1 LIMIT 1").approved


def test_every_decision_is_recorded_including_the_automatic_ones():
    """A gate whose approvals are invisible cannot be audited."""
    gate = ConfirmationGate(policy=POLICY_AUTO)
    gate.confirm("SELECT month FROM mart_macro_dashboard WHERE month > '2020-01-01'")
    gate.confirm("SELECT * FROM mart_futures_curve")
    summary = gate.summary()
    assert summary["n_presented"] == 2
    assert summary["n_approved"] == 1
    assert summary["n_refused"] == 1
    assert all(d.sql for d in gate.decisions)


def test_the_gate_sees_the_normalised_sql_not_the_raw_string():
    """Approving one string and executing another is a confused deputy."""
    decision = ConfirmationGate(policy=POLICY_AUTO).confirm("SELECT * FROM mart_futures_curve")
    assert decision.risk.tables == ("mart_futures_curve",)


def test_a_refused_statement_never_reaches_the_database():
    """The property that makes the gate a control and not a log line."""

    class ExplodingSql(SqlQueryTool):
        def run(self, sql):  # noqa: ARG002
            raise AssertionError("the gate let a refused statement through")

    agent = Agent(
        sql_tool=ExplodingSql(),
        rag_tool=None,
        gate=ConfirmationGate(policy=POLICY_DENY),
        llm=_StubLLM([]),
    )
    step = agent._run_sql({"sql": "SELECT 1 LIMIT 1"}, "q", 1, 0.0)
    assert step.error == "blocked_by_gate"
    assert step.gate is not None and not step.gate.approved


# --------------------------------------------------------------------------
# Action parsing
# --------------------------------------------------------------------------


def test_parses_a_clean_tool_call():
    tool, args, error = parse_action('{"tool": "sql_query", "args": {"sql": "SELECT 1"}}')
    assert (tool, args["sql"], error) == ("sql_query", "SELECT 1", "")


def test_parses_a_tool_call_wrapped_in_prose_and_fences():
    raw = 'Vou consultar:\n```json\n{"tool": "rag_search", "args": {"question": "q"}}\n```'
    tool, args, error = parse_action(raw)
    assert tool == "rag_search" and args["question"] == "q" and not error


def test_a_missing_tool_key_is_an_error_not_a_guess():
    _, _, error = parse_action('{"args": {"sql": "SELECT 1"}}')
    assert error


def test_prose_with_no_json_is_an_error():
    tool, _, error = parse_action("Acho que devo consultar o mart de cambio.")
    assert not tool and error


def test_non_dict_args_are_wrapped_rather_than_dropped():
    _, args, error = parse_action('{"tool": "sql_query", "args": "SELECT 1"}')
    assert not error and args == {"value": "SELECT 1"}


# --------------------------------------------------------------------------
# Loop control flow
# --------------------------------------------------------------------------


class _StubLLM:
    name = "stub"
    backend = "stub"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete(self, system, prompt):  # noqa: ARG002
        from generation.llm import LLMResponse

        self.calls += 1
        exhausted = '{"tool": "final", "args": {"answer": "fim"}}'
        text = self.responses.pop(0) if self.responses else exhausted
        return LLMResponse(text=text, model=self.name, backend=self.backend)


class _FakeSql(SqlQueryTool):
    database: Path = Path("nonexistent.duckdb")

    def schema_prompt(self):
        return "- mart_macro_dashboard(month DATE, selic_target DOUBLE)"

    @property
    def available(self):
        return True


def test_the_loop_stops_at_the_step_limit_and_still_answers():
    """An agent that loops forever is worse than one that admits it ran out."""
    llm = _StubLLM(['{"tool": "rag_search", "args": {"question": "q"}}'] * 10)

    class _Rag:
        def run(self, question):  # noqa: ARG002
            return {"decision": "answered", "answer": "a", "sources": [], "excerpt": "e"}

    agent = Agent(_FakeSql(), _Rag(), ConfirmationGate(POLICY_AUTO), llm=llm, max_steps=2)
    run = agent.run("pergunta")
    assert len(run.steps) == 2
    assert run.stopped_reason == "step_limit"
    assert run.answer  # the wrap-up call still produced something


def test_a_parse_failure_is_a_recorded_step_and_the_loop_continues():
    llm = _StubLLM(["nao e json", '{"tool": "final", "args": {"answer": "ok"}}'])
    agent = Agent(_FakeSql(), None, ConfirmationGate(POLICY_AUTO), llm=llm, max_steps=3)
    run = agent.run("pergunta")
    assert run.steps[0].tool == "parse_error"
    assert run.answer == "ok"
    assert run.stopped_reason == "final"


def test_an_unknown_tool_is_reported_back_to_the_model():
    llm = _StubLLM(['{"tool": "rm_rf", "args": {}}', '{"tool": "final", "args": {"answer": "ok"}}'])
    agent = Agent(_FakeSql(), None, ConfirmationGate(POLICY_AUTO), llm=llm, max_steps=3)
    run = agent.run("pergunta")
    assert run.steps[0].error == "unknown_tool"
    assert run.answer == "ok"


# --------------------------------------------------------------------------
# Integration: the real marts
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_the_real_marts_are_readable_and_read_only():
    tool = SqlQueryTool()
    if not tool.available:
        pytest.skip(f"gold marts not present at {tool.database}")

    schema = tool.schema()
    assert len(schema) >= 6, "the export should carry at least 6 marts"
    assert all(name.startswith("mart_") or name == "_export_manifest" for name in schema)

    result = tool.run(
        "SELECT month, selic_target FROM mart_macro_dashboard "
        "WHERE month >= '2026-01-01' ORDER BY month"
    )
    assert result.ok and result.row_count > 0

    # A write must fail at the engine, not only at the validator. Asserting
    # `duckdb.Error` rather than bare `Exception` matters here: a typo in the
    # statement would also raise, and would pass a blind assertion while proving
    # nothing about read-only enforcement.
    with tool.connect() as con:
        with pytest.raises(duckdb.Error, match="(?i)read.?only"):
            con.execute("CREATE TABLE should_not_exist (x INT)")
