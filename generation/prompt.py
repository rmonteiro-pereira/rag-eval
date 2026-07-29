"""The M1 baseline prompt: stuff top-k passages, demand citations, allow abstention.

Kept in one place and versioned (`PROMPT_VERSION`) because every eval number is
only meaningful next to the prompt that produced it.
"""

from __future__ import annotations

from retrieval.store import Retrieved

PROMPT_VERSION = "m1-naive-stuff-v1"

SYSTEM_PROMPT = """Voce e um assistente especializado em politica monetaria e \
regulacao financeira brasileira. Responda SOMENTE com base nos trechos fornecidos.

Regras:
1. Cite a fonte de cada afirmacao usando o marcador [n] do trecho correspondente.
2. Se os trechos nao contiverem a resposta, diga exatamente: \
"Nao encontrei essa informacao nos documentos recuperados." Nao invente.
3. Responda em portugues, de forma direta e objetiva.
4. Nao use conhecimento externo aos trechos."""


def format_context(passages: list[Retrieved]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        blocks.append(f"[{i}] {p.title} — pagina {p.page_number}\n{p.text}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, passages: list[Retrieved]) -> str:
    return (
        "Trechos recuperados:\n\n"
        f"{format_context(passages)}\n\n"
        "---\n\n"
        f"Pergunta: {question}\n\n"
        "Resposta (com citacoes [n]):"
    )
