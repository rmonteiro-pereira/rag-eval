"""Assemble a cited answer from retrieved passages."""

from __future__ import annotations

from dataclasses import dataclass, field

from generation.llm import LLM, LLMResponse
from generation.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_prompt
from retrieval.store import Retrieved

NO_EVIDENCE = "Nao encontrei documentos relevantes para essa pergunta."


@dataclass
class Citation:
    marker: int
    title: str
    page_number: int
    chunk_index: int
    url: str
    score: float

    def render(self) -> str:
        return (
            f"[{self.marker}] {self.title} — pagina {self.page_number}, "
            f"chunk {self.chunk_index} (score {self.score:.3f})\n     {self.url}"
        )


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    backend: str
    model: str
    prompt_version: str = PROMPT_VERSION
    usage: dict = field(default_factory=dict)


def _citations_from(passages: list[Retrieved]) -> list[Citation]:
    return [
        Citation(
            marker=i,
            title=p.title,
            page_number=p.page_number,
            chunk_index=p.chunk_index,
            url=p.url,
            score=p.score,
        )
        for i, p in enumerate(passages, start=1)
    ]


def _extractive_text(passages: list[Retrieved]) -> str:
    """Verbatim evidence, one block per passage. No paraphrase, no synthesis."""
    parts = [
        "[modo extractivo — sem LLM: os trechos abaixo sao citados literalmente]",
        "",
    ]
    for i, p in enumerate(passages, start=1):
        parts.append(f"[{i}] {p.title} — pagina {p.page_number}")
        parts.append(p.text.strip())
        parts.append("")
    return "\n".join(parts).strip()


def generate_answer(question: str, passages: list[Retrieved], llm: LLM) -> Answer:
    if not passages:
        return Answer(
            question=question,
            text=NO_EVIDENCE,
            citations=[],
            backend=llm.backend,
            model=llm.name,
        )

    citations = _citations_from(passages)

    if llm.backend == "extractive":
        return Answer(
            question=question,
            text=_extractive_text(passages),
            citations=citations,
            backend=llm.backend,
            model=llm.name,
        )

    prompt = build_prompt(question, passages)
    response: LLMResponse = llm.complete(SYSTEM_PROMPT, prompt)
    return Answer(
        question=question,
        text=response.text or NO_EVIDENCE,
        citations=citations,
        backend=response.backend,
        model=response.model,
        usage={
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "latency_ms": response.latency_ms,
        },
    )
