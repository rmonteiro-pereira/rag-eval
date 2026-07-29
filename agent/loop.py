"""An explicit tool-use loop. No framework.

LangGraph was the other option and was not taken. The loop is about forty lines —
prompt, parse a tool call, run it, append the observation, repeat — and writing
it out means the whole control flow, including the HITL gate and the step
ceiling, is visible in one file instead of distributed across a graph definition
and a library's execution semantics. For a project whose entire argument is
"measure what your system actually does", a legible loop is worth more than a
framework's abstractions.

The guardrails are not re-implemented here either: `rag_search` goes through
`GovernedPipeline`, so PII masking, injection detection and the ACL apply to the
agent exactly as they do to the plain query path. The agent adds one control the
plain path does not have, because it is the only one with a destructive verb
anywhere near it: the SQL confirmation gate.

## Why JSON prompting rather than native tool calling

Ollama exposes native tool calling for some models. This uses plain JSON in the
completion instead, parsed defensively, for the same reason `generation/judge.py`
does: it works identically across every local model, and a parse failure becomes
a recorded step rather than a silent behaviour change when the model is swapped.
Small models emit prose around their JSON; the parser takes the last balanced
object and records a failure when there is none.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from agent.hitl import ConfirmationGate, GateDecision
from agent.tools import RagSearchTool, SqlQueryTool
from generation.llm import LLM, OllamaLLM
from rag.config import settings

MAX_STEPS = 4

SYSTEM_PROMPT = """Voce e um analista de politica monetaria brasileira. Voce responde \
usando DUAS ferramentas e nunca por conhecimento proprio.

FERRAMENTAS:
1. sql_query   - SQL SELECT (DuckDB) sobre marts do lakehouse. Use para NUMEROS, \
SERIES TEMPORAIS, medias, comparacoes entre periodos.
2. rag_search  - busca nas atas do Copom (texto). Use para DECISOES, JUSTIFICATIVAS, \
o que o comite DISSE ou como avaliou o cenario.

ESQUEMA DOS MARTS:
{schema}

REGRAS:
- Responda SEMPRE com um unico objeto JSON, sem texto antes ou depois.
- Para chamar uma ferramenta: {{"tool": "sql_query", "args": {{"sql": "SELECT ..."}}}}
  ou {{"tool": "rag_search", "args": {{"question": "..."}}}}
- Para responder ao usuario: {{"tool": "final", "args": {{"answer": "..."}}}}
- Use no maximo {max_steps} chamadas de ferramenta. Depois disso responda com o que tem.
- SQL: apenas SELECT. Sempre filtre por periodo (WHERE month >= '...') e use LIMIT.
- Datas nos marts mensais sao o primeiro dia do mes (month = DATE '2026-06-01').
- Se uma ferramenta retornar erro, corrija a chamada em vez de repetir a mesma.
- A resposta final deve ser TEXTO CORRIDO em portugues, interpretando os dados. \
Nao devolva o resultado bruto da consulta como resposta.
- Na resposta final, cite de onde veio cada numero: o mart e a coluna, ou a ata."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Step:
    index: int
    tool: str
    args: dict
    observation: str
    elapsed_ms: float
    gate: GateDecision | None = None
    sql_result: dict | None = None
    rag_sources: list[dict] = field(default_factory=list)
    error: str = ""

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "tool": self.tool,
            "args": self.args,
            "observation": self.observation[:1200],
            "elapsed_ms": round(self.elapsed_ms, 1),
            "gate": self.gate.to_json() if self.gate else None,
            "sql_result": self.sql_result,
            "rag_sources": self.rag_sources,
            "error": self.error,
        }


@dataclass
class AgentRun:
    question: str
    answer: str
    steps: list[Step]
    elapsed_ms: float
    stopped_reason: str

    @property
    def used_sql(self) -> bool:
        return any(s.tool == "sql_query" and s.sql_result and not s.error for s in self.steps)

    @property
    def used_rag(self) -> bool:
        return any(s.tool == "rag_search" and not s.error for s in self.steps)

    @property
    def executed_sql(self) -> list[str]:
        return [
            s.sql_result["sql"]
            for s in self.steps
            if s.tool == "sql_query" and s.sql_result and not s.sql_result.get("error")
        ]

    def to_json(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "used_sql": self.used_sql,
            "used_rag": self.used_rag,
            "executed_sql": self.executed_sql,
            "stopped_reason": self.stopped_reason,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "steps": [s.to_json() for s in self.steps],
        }


def parse_action(raw: str) -> tuple[str, dict, str]:
    """`(tool, args, error)`. A parse failure is a recorded step, not an exception."""
    match = _JSON_OBJECT.search(raw or "")
    if not match:
        return "", {}, "no JSON object in the model output"

    text = match.group(0)
    # Small models sometimes emit two objects; walk back to the last balanced one.
    for end in range(len(text), 0, -1):
        if text[end - 1] != "}":
            continue
        try:
            payload = json.loads(text[:end])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        tool = str(payload.get("tool", "")).strip()
        args = payload.get("args") or {}
        if not isinstance(args, dict):
            args = {"value": args}
        if tool:
            return tool, args, ""
        return "", {}, "JSON object has no `tool` key"
    return "", {}, "no parsable JSON object in the model output"


class Agent:
    def __init__(
        self,
        sql_tool: SqlQueryTool,
        rag_tool: RagSearchTool,
        gate: ConfirmationGate,
        llm: LLM | None = None,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.sql = sql_tool
        self.rag = rag_tool
        self.gate = gate
        self.llm = llm or OllamaLLM(model=settings.agent_model)
        self.max_steps = max_steps
        self._schema = sql_tool.schema_prompt() if sql_tool.available else "(indisponivel)"

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(schema=self._schema, max_steps=self.max_steps)

    def _run_sql(self, args: dict, question: str, index: int, started: float) -> Step:
        sql = str(args.get("sql", ""))
        decision = self.gate.confirm(sql, question=question)
        if not decision.approved:
            return Step(
                index=index,
                tool="sql_query",
                args=args,
                observation=(
                    f"EXECUCAO BLOQUEADA PELO GATE HUMANO: {decision.reason}. "
                    "Reformule para uma consulta mais restrita (filtre por periodo, "
                    "use LIMIT) ou responda sem SQL."
                ),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                gate=decision,
                error="blocked_by_gate",
            )

        result = self.sql.run(sql)
        return Step(
            index=index,
            tool="sql_query",
            args=args,
            observation=(
                f"erro: {result.error}"
                if result.error
                else f"{result.row_count} linha(s):\n{result.to_markdown(8)}"
            ),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            gate=decision,
            sql_result=result.to_json(),
            error=result.error,
        )

    def _run_rag(self, args: dict, index: int, started: float) -> Step:
        question = str(args.get("question") or args.get("query") or "")
        payload = self.rag.run(question)
        sources = payload["sources"]
        rendered = "; ".join(f"{s['title']} p.{s['page']}" for s in sources[:3])
        return Step(
            index=index,
            tool="rag_search",
            args=args,
            observation=f"trechos de [{rendered}]:\n{payload['excerpt']}",
            elapsed_ms=(time.perf_counter() - started) * 1000,
            rag_sources=sources,
        )

    def run(self, question: str) -> AgentRun:
        started_run = time.perf_counter()
        steps: list[Step] = []
        transcript = f"PERGUNTA: {question}\n"
        answer = ""
        stopped = "step_limit"

        for index in range(1, self.max_steps + 1):
            started = time.perf_counter()
            try:
                response = self.llm.complete(self.system_prompt(), transcript + "\nJSON:")
            except Exception as exc:  # noqa: BLE001
                # A dead or overloaded model server must not take the whole demo
                # with it. The step is recorded as what it was — a backend
                # failure, not a model decision — and the loop moves on with
                # whatever observations it already has.
                steps.append(
                    Step(index, "llm_error", {}, f"{type(exc).__name__}: {exc}",
                         (time.perf_counter() - started) * 1000,
                         error="llm_call_failed")
                )
                stopped = "llm_error"
                break

            tool, args, error = parse_action(response.text)

            if error:
                steps.append(
                    Step(index, "parse_error", {}, response.text[:400],
                         (time.perf_counter() - started) * 1000, error=error)
                )
                transcript += (
                    f"\nOBSERVACAO: sua saida nao era JSON valido ({error}). "
                    "Responda apenas com o objeto JSON.\n"
                )
                continue

            if tool == "final":
                answer = str(args.get("answer", "")).strip()
                stopped = "final"
                steps.append(
                    Step(index, "final", args, answer,
                         (time.perf_counter() - started) * 1000)
                )
                break

            if tool == "sql_query":
                step = self._run_sql(args, question, index, started)
            elif tool == "rag_search":
                step = self._run_rag(args, index, started)
            else:
                step = Step(
                    index, tool, args,
                    f"ferramenta desconhecida: {tool!r}",
                    (time.perf_counter() - started) * 1000,
                    error="unknown_tool",
                )

            steps.append(step)
            transcript += (
                f"\nACAO: {json.dumps({'tool': tool, 'args': args}, ensure_ascii=False)}"
                f"\nOBSERVACAO: {step.observation}\n"
            )

        if not answer:
            # Out of steps. Ask once for a final answer from what was gathered,
            # rather than returning nothing — the observations are usually enough.
            try:
                response = self.llm.complete(
                    self.system_prompt(),
                    transcript + "\nVoce atingiu o limite de ferramentas. "
                    'Responda agora com {"tool": "final", "args": {"answer": "..."}}\nJSON:',
                )
                _, args, _ = parse_action(response.text)
                answer = str(args.get("answer", "")).strip() or response.text.strip()[:800]
            except Exception as exc:  # noqa: BLE001
                answer = (
                    "(sem resposta final: a chamada ao modelo falhou — "
                    f"{type(exc).__name__}). As observacoes coletadas estao nos passos acima."
                )
                stopped = "llm_error"

        return AgentRun(
            question=question,
            answer=answer,
            steps=steps,
            elapsed_ms=(time.perf_counter() - started_run) * 1000,
            stopped_reason=stopped,
        )
