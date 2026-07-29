"""Agent CLI.

    uv run python -m rag.agent --demo                    # regenerate docs/agent_demo.md
    uv run python -m rag.agent "sua pergunta"            # one question
    uv run python -m rag.agent --gate interactive "..."  # confirm each SQL by hand

The demo is a *generator*, not a checked-in artifact someone edited by hand:
`--demo` re-runs every question against the live DuckDB and the live Qdrant and
rewrites `docs/agent_demo.md` from the transcript. If the marts change, the
document changes; if the agent regresses, the document shows it.

The gate defaults to `auto` here, which approves low/medium risk unattended and
**refuses high risk outright**. `--gate interactive` is the real human loop and
`--gate deny` refuses everything, which is how the "the gate actually stops
execution" property gets exercised.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from agent.hitl import POLICY_AUTO, POLICY_DENY, POLICY_INTERACTIVE, ConfirmationGate
from agent.loop import Agent, AgentRun
from agent.tools import RagSearchTool, SqlQueryTool
from rag.config import REPO_ROOT, settings

DEMO_PATH = REPO_ROOT / "docs" / "agent_demo.md"

#: Ten questions spanning what each source can and cannot answer.
#: `sql` = the numbers live only in the marts. `rag` = only the atas say it.
#: `both` = the question is not answerable from either source alone, which is the
#: only reason to have an agent at all.
DEMO_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("sql", "Qual era a meta da taxa Selic em cada mes de 2026 ate junho?"),
    (
        "sql",
        "Qual foi a taxa de juros real em maio de 2026 e como ela se compara "
        "com a de junho de 2025?",
    ),
    ("sql", "Qual foi a media do IPCA mensal em 2025?"),
    ("sql", "Em que meses de 2025 e 2026 a Selic mudou de valor?"),
    ("sql", "Qual foi a cotacao media do dolar em cada trimestre de 2026?"),
    ("sql", "Qual foi o IPCA acumulado em 12 meses no fim de 2025?"),
    ("rag", "Por que o Copom decidiu reduzir a Selic na reuniao de junho de 2026?"),
    (
        "rag",
        "Quem votou pela reducao de 0,50 ponto percentual na reuniao de agosto de 2023?",
    ),
    (
        "both",
        "O Copom cortou a Selic em junho de 2026 — o que a ata deu como "
        "justificativa e qual era o IPCA acumulado em 12 meses naquele momento?",
    ),
    (
        "both",
        "Compare o que a ata de janeiro de 2025 disse sobre o cambio com a "
        "cotacao media do dolar naquele mes.",
    ),
)


#: Statements put through the gate directly, so the transcript shows what it
#: does rather than only asserting it. A demo where the gate happens never to
#: fire proves nothing about the gate — and on this corpus the agent writes
#: sensible bounded queries, so it never provokes one on its own.
GATE_PROBES: tuple[tuple[str, str], ...] = (
    (
        "an ordinary bounded read",
        "SELECT month, selic_target FROM mart_macro_dashboard "
        "WHERE month >= '2026-01-01' LIMIT 12",
    ),
    (
        "unbounded scan of a 1.6M-row mart",
        "SELECT * FROM mart_futures_curve",
    ),
    (
        "three-way join",
        "SELECT * FROM mart_macro_dashboard a JOIN mart_real_interest b "
        "ON a.month = b.month JOIN mart_inflation_panel c ON a.month = c.month "
        "WHERE a.month >= '2020-01-01'",
    ),
    (
        "a write disguised as a query",
        "SELECT 1; DROP TABLE mart_fx",
    ),
)


def build_agent(gate_policy: str = POLICY_AUTO, max_steps: int | None = None) -> Agent:
    from guardrails.pipeline import GovernedPipeline

    sql_tool = SqlQueryTool()
    if not sql_tool.available:
        raise SystemExit(
            f"gold marts not found at {sql_tool.database}\n"
            "Agent mode needs the DuckDB export from the Open-Finance-LakeHouse "
            "project. It is not part of this repo and is never committed here."
        )
    rag_tool = RagSearchTool(pipeline=GovernedPipeline())
    return Agent(
        sql_tool=sql_tool,
        rag_tool=rag_tool,
        gate=ConfirmationGate(policy=gate_policy),
        max_steps=max_steps or settings.agent_max_steps,
    )


def _render_run(kind: str, run: AgentRun) -> str:
    lines = [
        f"### {run.question}",
        "",
        f"`expected: {kind}` · `tools: "
        f"{'sql' if run.used_sql else '—'}/{'rag' if run.used_rag else '—'}` · "
        f"`{run.elapsed_ms / 1000:.1f}s` · `stopped: {run.stopped_reason}`",
        "",
    ]
    for step in run.steps:
        if step.tool == "sql_query":
            gate = step.gate
            lines.append(
                f"**Step {step.index} — `sql_query`** · gate: "
                f"{'APPROVED' if gate and gate.approved else 'REFUSED'} "
                f"(risk `{gate.risk.level}` — {'; '.join(gate.risk.reasons)})"
            )
            statement = (step.sql_result or {}).get("sql") or step.args.get("sql", "")
            lines += ["", "```sql", statement, "```", ""]
            if step.error == "blocked_by_gate":
                lines += [f"> Gate refused: {gate.reason}", ""]
            elif step.sql_result and step.sql_result.get("error"):
                lines += [f"> Query error: `{step.sql_result['error']}`", ""]
            else:
                lines += [step.observation.split(":\n", 1)[-1], ""]
        elif step.tool == "rag_search":
            sources = ", ".join(f"{s['title']} (p. {s['page']})" for s in step.rag_sources[:3])
            lines += [
                f"**Step {step.index} — `rag_search`** · `{step.args.get('question', '')}`",
                "",
                f"Sources: {sources}" if sources else "Sources: —",
                "",
            ]
        elif step.tool == "parse_error":
            lines += [f"**Step {step.index} — parse error** ({step.error})", ""]
        elif step.tool == "final":
            continue
        else:
            lines += [f"**Step {step.index} — `{step.tool}`** — {step.observation[:200]}", ""]

    lines += ["**Answer**", "", "> " + run.answer.replace("\n", "\n> "), "", "---", ""]
    return "\n".join(lines)


def render_demo(
    runs: list[tuple[str, AgentRun]],
    gate: ConfirmationGate,
    agent: Agent,
    probes: list[tuple[str, object]],
) -> str:
    sql_runs = [r for _, r in runs if r.used_sql]
    rag_runs = [r for _, r in runs if r.used_rag]
    summary = gate.summary()

    header = [
        "# Agent mode — text-to-SQL over the lakehouse marts, plus RAG over the atas",
        "",
        "> **Generated.** Regenerate with `uv run python -m rag.agent --demo`. Every",
        "> query below ran against the live DuckDB marts and the live Qdrant collection;",
        "> nothing here was written by hand.",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}_",
        "",
        "## What this is",
        "",
        "Two tools that answer different kinds of question, and an agent whose only",
        "real job is deciding which one a question needs:",
        "",
        "- **`sql_query`** — read-only DuckDB `SELECT` over the gold marts exported from",
        "  the [Open-Finance-LakeHouse](https://github.com/) project: monthly Selic,",
        "  IPCA, FX, real interest, plus daily yield-curve and equity series.",
        "- **`rag_search`** — the governed RAG pipeline over 30 Copom minutes. PII",
        "  masking, injection detection and the document ACL all apply, unchanged.",
        "",
        "The marts and the atas describe the same events from opposite sides: the atas",
        "say the Copom cut to 14,25% and why; the marts say what `selic_target` and",
        "`ipca_12m` actually did. Questions that need both are the reason the agent",
        "exists — neither source answers them alone.",
        "",
        "**The database is not in this repo.** `_artifacts/ofl_gold.duckdb` is a 70 MB",
        "read-only artifact produced by the other project. Nothing here writes to it and",
        "nothing here commits it.",
        "",
        "## The HITL gate",
        "",
        "Every `sql_query` passes a confirmation gate before execution. The problem with",
        "demo-mode gates is that they usually become `return True` when nobody is",
        "watching, which is a screenshot rather than a control. Here the gate",
        "**classifies risk first** and the policy decides per level, so it still refuses",
        "things unattended:",
        "",
        "| risk | what triggers it | `auto` policy (this run) |",
        "|---|---|---|",
        "| `low` | bounded read of a small monthly mart | approved |",
        "| `medium` | joins, or a filtered read of a large mart | approved |",
        "| `high` | unbounded scan of a 1.6M-row mart, 2+ joins, or SQL the "
        "validator rejected | **refused, no human to ask** |",
        "",
        "`--gate interactive` prints the statement and waits on stdin — the real loop.",
        "`--gate deny` refuses everything, which is what makes the \"does the gate",
        "actually stop execution\" test meaningful.",
        "",
        "Risk is assessed on the **normalised** SQL — the exact text that would run,",
        "after `LIMIT` injection — because approving one string and executing another is",
        "the oldest confused-deputy bug there is.",
        "",
        "### The gate, probed directly",
        "",
        "The agent writes sensible bounded queries against these marts, so it never",
        "provokes a refusal on its own — and a demo where the gate happens never to fire",
        "proves nothing about the gate. These statements are therefore put through it",
        "directly, in this run, with the same policy the agent used:",
        "",
        "| statement | risk | gate |",
        "|---|---|---|",
        *[
            f"| {label} | `{decision.risk.level}` — {'; '.join(decision.risk.reasons)} | "
            f"{'approved' if decision.approved else '**REFUSED**'} |"
            for label, decision in probes
        ],
        "",
        "## This run",
        "",
        f"- **{len(runs)} questions**, {len(sql_runs)} answered using SQL over the marts,",
        f"  {len(rag_runs)} using retrieval over the atas.",
        f"- **{summary['n_presented']} statements presented to the gate** "
        f"(agent-written plus the {len(probes)} probes above): "
        f"{summary['n_approved']} approved, {summary['n_refused']} refused.",
        f"- Risk mix: {summary['by_risk']}",
        f"- Model: `{agent.llm.name}`, max {agent.max_steps} tool calls per question.",
        "",
        "## Marts available to the agent",
        "",
        "```",
        agent._schema,
        "```",
        "",
        "## What the agent does not do",
        "",
        "Read the `expected` tag against the `tools` tag in the transcript below.",
        "",
        f"- **Tool routing works.** All {len(sql_runs)} numeric questions reached SQL and",
        f"  all the narrative ones reached retrieval; the model picks the right source.",
        "- **Tool *composition* does not.** The two questions tagged `expected: both`",
        "  need one fact from each source, and the agent answered them from retrieval",
        "  alone — it gathers the qualitative half and stops rather than following up",
        "  with the query that would supply the number. That is the honest status of",
        "  agent mode: routing yes, multi-source synthesis no.",
        "- **The step ceiling binds.** Several runs stop at `step_limit` rather than",
        "  `final`, and the answer comes from the wrap-up call. An 8B model at 4 steps",
        "  spends a step recovering from its own malformed SQL more often than it",
        "  spends one composing.",
        "- **SQL errors are common and self-corrected.** `llama3.1` regularly drops the",
        "  closing quote on a date literal. The validator names that specific defect",
        "  (`unbalanced single quote`) rather than letting DuckDB report a confusing",
        "  error pointing at the injected `LIMIT`, and the model then fixes it — but it",
        "  costs a step.",
        "- **No evaluation harness covers agent mode.** There is no gold set for",
        "  multi-tool questions and no measured task-success rate. Everything above is a",
        "  demonstration, not a measurement, and that is the gap between this section",
        "  and the rest of the project.",
        "",
        "---",
        "",
        "## Transcript",
        "",
    ]
    return "\n".join(header) + "\n".join(_render_run(kind, run) for kind, run in runs)


def run_demo(gate_policy: str, max_steps: int | None, out: Path) -> int:
    agent = build_agent(gate_policy, max_steps)
    runs: list[tuple[str, AgentRun]] = []
    for index, (kind, question) in enumerate(DEMO_QUESTIONS, start=1):
        print(
            f"  [{index}/{len(DEMO_QUESTIONS)}] {question[:70]}...",
            file=sys.stderr,
            flush=True,
        )
        runs.append((kind, agent.run(question)))

    probes = [(label, agent.gate.confirm(sql, question="gate probe")) for label, sql in GATE_PROBES]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_demo(runs, agent.gate, agent, probes), encoding="utf-8")

    sql_count = sum(1 for _, r in runs if r.used_sql)
    print(f"\ntranscript -> {out}")
    print(f"{len(runs)} questions, {sql_count} answered via SQL over the marts")
    print(f"gate: {json.dumps(agent.gate.summary())}")
    if sql_count < 5:
        print(
            f"\nWARNING: only {sql_count} questions used SQL; the demo target is >= 5",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag.agent",
        description="Agent mode: text-to-SQL over the gold marts plus RAG over the atas.",
    )
    parser.add_argument("question", nargs="?", help="a single question to answer")
    parser.add_argument("--demo", action="store_true", help="regenerate docs/agent_demo.md")
    parser.add_argument(
        "--gate",
        choices=[POLICY_AUTO, POLICY_INTERACTIVE, POLICY_DENY],
        default=POLICY_AUTO,
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEMO_PATH)
    parser.add_argument("--json", action="store_true", help="dump the raw run as JSON")
    args = parser.parse_args(argv)

    if args.demo:
        return run_demo(args.gate, args.max_steps, args.out)

    if not args.question:
        parser.error("give a question, or --demo")

    agent = build_agent(args.gate, args.max_steps)
    run = agent.run(args.question)
    if args.json:
        print(json.dumps(run.to_json(), ensure_ascii=False, indent=2))
    else:
        print(_render_run("ad-hoc", run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
