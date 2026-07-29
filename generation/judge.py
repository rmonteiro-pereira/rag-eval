"""LLM-as-judge, with the rubric written down and the judge treated as suspect.

Two scores per answer, each on an explicit 0–2 scale:

* **faithfulness** — is every claim in the answer supported by the retrieved
  passages? This is about grounding, not truth: an answer that is factually
  correct but not present in the context scores 0, because a RAG system that
  answers from parametric memory is broken even when it is right.
* **answer_relevance** — does the answer address the question that was asked?
  Judged *independently* of faithfulness, because the two fail apart: a verbatim
  quote of the wrong paragraph is perfectly faithful and useless.

## Why this judge should not be believed yet

It is a 3B/8B model running on a laptop, grading answers written by a 3B/8B
model running on the same laptop. Nothing about that arrangement produces a
trustworthy number, and the literature's agreement figures for LLM judges are
from frontier models on English.

Three things are done about it, none of which is "trust it anyway":

1. **The judge is a different model from the generator by default.** Grading
   your own homework has a known direction of bias; `settings.judge_model`
   defaults to `llama3.1` and the report records which model judged which arm,
   flagging any row where they coincide.
2. **Deterministic metrics are reported next to it** (`eval/metrics/generation.py`)
   and the report shows where the two disagree. A judge that says "faithful"
   about an answer containing a number found nowhere in the context is caught by
   arithmetic, not by another model.
3. **`eval/datasets/judge_calibration_sheet.jsonl`** carries 30 judged items with
   an empty human-label column. Until a human fills it, the judge's agreement
   with human judgement is *unknown* — and unknown is what the report says. It
   does not say "good".

## Output contract

The model is asked for one JSON object. Local models at this size emit prose
around it, fenced blocks, and occasionally two objects, so parsing is defensive
and a parse failure is recorded as a failure rather than silently coerced to a
score. A judge that quietly returns 0 when it malfunctions makes a system look
bad; one that quietly returns 2 makes it look good. Both are worse than a
recorded `null`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from generation.llm import LLM, OllamaLLM
from rag.config import settings

JUDGE_RUBRIC_VERSION = "judge-rubric-v1"

JUDGE_SYSTEM = """Voce e um avaliador rigoroso de sistemas de perguntas e respostas \
sobre documentos. Voce NAO responde a pergunta: voce avalia a resposta dada.

Responda SEMPRE com um unico objeto JSON, sem texto antes ou depois, no formato:
{"faithfulness": <0|1|2>, "faithfulness_reason": "<uma frase>", \
"answer_relevance": <0|1|2>, "answer_relevance_reason": "<uma frase>"}

CRITERIO faithfulness (a resposta e sustentada pelos TRECHOS?):
  2 = toda afirmacao da resposta aparece nos trechos fornecidos
  1 = a maior parte aparece, mas ha alguma afirmacao sem suporte
  0 = ha afirmacao central que nao aparece nos trechos, ou a resposta contradiz os trechos
  Atencao: uma resposta pode estar factualmente correta e ainda assim receber 0, \
se o fato nao estiver nos trechos. Avalie suporte, nao verdade.
  Uma recusa explicita ("nao encontrei essa informacao") recebe 2 se os trechos \
realmente nao contem a resposta.

CRITERIO answer_relevance (a resposta responde a PERGUNTA?):
  2 = responde diretamente e por completo
  1 = responde parcialmente, ou responde com informacao excedente que confunde
  0 = nao responde a pergunta feita, ou responde a outra pergunta
  Uma recusa recebe 0 se os trechos continham a resposta, e 2 se nao continham."""

_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)

VALID_SCORES = (0, 1, 2)


@dataclass(frozen=True)
class Judgement:
    """One judged answer. `None` scores mean the judge failed, not scored zero."""

    faithfulness: int | None
    faithfulness_reason: str
    answer_relevance: int | None
    answer_relevance_reason: str
    judge_model: str
    raw: str = ""
    parse_error: str = ""

    @property
    def ok(self) -> bool:
        return self.faithfulness is not None and self.answer_relevance is not None

    def to_json(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "faithfulness_reason": self.faithfulness_reason,
            "answer_relevance": self.answer_relevance,
            "answer_relevance_reason": self.answer_relevance_reason,
            "judge_model": self.judge_model,
            "parse_error": self.parse_error,
        }


def build_judge_prompt(question: str, context: str, answer: str) -> str:
    return (
        "TRECHOS RECUPERADOS:\n\n"
        f"{context}\n\n"
        "---\n\n"
        f"PERGUNTA: {question}\n\n"
        f"RESPOSTA A AVALIAR:\n{answer}\n\n"
        "---\n\n"
        "Avalie a resposta acima. Responda apenas com o objeto JSON."
    )


def _coerce_score(value) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if score in VALID_SCORES else None


def parse_judgement(raw: str, judge_model: str) -> Judgement:
    """Pull the verdict out of whatever the model actually emitted.

    Takes the *last* JSON object in the response: small models often restate the
    schema before filling it in, and the filled-in one comes second.
    """
    matches = _JSON_OBJECT.findall(raw or "")
    for candidate in reversed(matches):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        faithfulness = _coerce_score(payload.get("faithfulness"))
        relevance = _coerce_score(payload.get("answer_relevance"))
        if faithfulness is None and relevance is None:
            continue
        return Judgement(
            faithfulness=faithfulness,
            faithfulness_reason=str(payload.get("faithfulness_reason", ""))[:400],
            answer_relevance=relevance,
            answer_relevance_reason=str(payload.get("answer_relevance_reason", ""))[:400],
            judge_model=judge_model,
            raw=raw[:1000],
        )

    return Judgement(
        faithfulness=None,
        faithfulness_reason="",
        answer_relevance=None,
        answer_relevance_reason="",
        judge_model=judge_model,
        raw=(raw or "")[:1000],
        parse_error="no parsable JSON verdict in the judge response",
    )


class Judge:
    def __init__(self, llm: LLM | None = None, model: str | None = None) -> None:
        self.llm = llm or OllamaLLM(model=model or settings.judge_model)
        self.model = getattr(self.llm, "name", "unknown")

    def judge(self, question: str, context: str, answer: str) -> Judgement:
        prompt = build_judge_prompt(question, context, answer)
        try:
            response = self.llm.complete(JUDGE_SYSTEM, prompt)
        except Exception as exc:  # noqa: BLE001 - a dead judge must not kill the run
            return Judgement(
                faithfulness=None,
                faithfulness_reason="",
                answer_relevance=None,
                answer_relevance_reason="",
                judge_model=self.model,
                parse_error=f"judge call failed: {type(exc).__name__}: {exc}",
            )
        return parse_judgement(response.text, self.model)


def aggregate_judgements(judgements: list[Judgement]) -> dict:
    """Mean scores over the judgements that parsed, plus the failure count."""
    ok = [j for j in judgements if j.ok]
    return {
        "n": len(judgements),
        "n_parsed": len(ok),
        "parse_failure_rate": (
            (len(judgements) - len(ok)) / len(judgements) if judgements else None
        ),
        "faithfulness_mean": (sum(j.faithfulness for j in ok) / len(ok)) if ok else None,
        "answer_relevance_mean": (sum(j.answer_relevance for j in ok) / len(ok)) if ok else None,
        "faithfulness_at_2": (
            sum(j.faithfulness == 2 for j in ok) / len(ok) if ok else None
        ),
        "answer_relevance_at_2": (
            sum(j.answer_relevance == 2 for j in ok) / len(ok) if ok else None
        ),
        "rubric_version": JUDGE_RUBRIC_VERSION,
    }
