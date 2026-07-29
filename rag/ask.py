"""CLI entrypoint.

    uv run python -m rag.ask "qual a decisao do Copom sobre a Selic?"
    uv run python -m rag.ask --top-k 8 --mode extractive "..."
"""

from __future__ import annotations

import argparse
import json
import sys

from rag.config import settings
from rag.pipeline import RagPipeline


def _render(result, show_passages: bool) -> str:
    a = result.answer
    out = [
        "",
        f"Pergunta: {a.question}",
        "",
        "Resposta",
        "--------",
        a.text,
        "",
        "Fontes",
        "------",
    ]
    if a.citations:
        out.extend(c.render() for c in a.citations)
    else:
        out.append("(nenhuma)")

    if show_passages:
        out += ["", "Trechos recuperados", "-------------------"]
        for i, p in enumerate(result.passages, start=1):
            out.append(f"[{i}] score={p.score:.4f} {p.citation()}")
            out.append(p.text[:400] + ("..." if len(p.text) > 400 else ""))
            out.append("")

    out += [
        "",
        f"backend={a.backend} model={a.model} prompt={a.prompt_version} "
        f"latency={result.latency_ms:.0f}ms",
    ]
    if result.trace_id:
        out.append(f"trace={settings.langfuse_host} (id {result.trace_id})")
    else:
        out.append("trace=disabled")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag.ask",
        description="Ask the naive RAG baseline a question about the BACEN corpus.",
    )
    parser.add_argument("question", help="pergunta em portugues")
    parser.add_argument("-k", "--top-k", type=int, default=settings.top_k)
    parser.add_argument(
        "-m",
        "--mode",
        choices=["auto", "ollama", "extractive"],
        default=settings.llm_mode,
        help="LLM backend (auto falls back to extractive when Ollama is absent)",
    )
    parser.add_argument("--no-trace", action="store_true", help="disable Langfuse tracing")
    parser.add_argument("--show-passages", action="store_true")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    from generation.llm import build_llm
    from rag.tracing import Tracer

    pipeline = RagPipeline(
        llm=build_llm(args.mode),
        tracer=Tracer(enabled=not args.no_trace),
    )
    result = pipeline.ask(args.question, top_k=args.top_k)

    if args.json:
        print(
            json.dumps(
                {
                    "question": result.answer.question,
                    "answer": result.answer.text,
                    "backend": result.answer.backend,
                    "model": result.answer.model,
                    "prompt_version": result.answer.prompt_version,
                    "latency_ms": round(result.latency_ms, 1),
                    "trace_id": result.trace_id,
                    "citations": [
                        {
                            "marker": c.marker,
                            "title": c.title,
                            "page": c.page_number,
                            "chunk_index": c.chunk_index,
                            "url": c.url,
                            "score": c.score,
                        }
                        for c in result.answer.citations
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(_render(result, args.show_passages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
