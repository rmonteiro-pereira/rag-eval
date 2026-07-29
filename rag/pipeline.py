"""The M1 naive RAG pipeline: retrieve -> stuff -> answer, fully traced.

One class, one method. Kept separate from the CLI so the eval harness (M3) can
call the exact same code path the user calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from generation.answer import Answer, generate_answer
from generation.llm import LLM, build_llm
from generation.prompt import PROMPT_VERSION
from rag.config import settings
from rag.tracing import Tracer, span
from retrieval.dense import DenseRetriever
from retrieval.store import Retrieved


@dataclass
class AskResult:
    answer: Answer
    passages: list[Retrieved]
    trace_id: str | None
    latency_ms: float


class RagPipeline:
    def __init__(
        self,
        retriever: DenseRetriever | None = None,
        llm: LLM | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.retriever = retriever or DenseRetriever()
        self.llm = llm or build_llm()
        self.tracer = tracer if tracer is not None else Tracer()

    def ask(self, question: str, top_k: int | None = None, user_id: str = "cli") -> AskResult:
        top_k = top_k or settings.top_k
        started = time.perf_counter()

        trace = self.tracer.trace(
            name="rag.ask",
            input={"question": question},
            user_id=user_id,
            metadata={
                "milestone": "M1-naive-baseline",
                "prompt_version": PROMPT_VERSION,
                "embedding_model": settings.embedding_model,
                "collection": settings.qdrant_collection,
                "top_k": top_k,
                "llm_backend": self.llm.backend,
                "llm_model": self.llm.name,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
            tags=["m1", "naive", self.llm.backend],
        )

        with span(trace, "retrieve", input={"question": question, "top_k": top_k}) as s:
            passages = self.retriever.retrieve(question, top_k=top_k)
            s.update(
                output={
                    "hits": [
                        {
                            "rank": i,
                            "score": p.score,
                            "title": p.title,
                            "page": p.page_number,
                            "chunk_index": p.chunk_index,
                        }
                        for i, p in enumerate(passages, start=1)
                    ]
                }
            )

        with span(
            trace,
            "generate",
            input={"question": question, "n_passages": len(passages)},
            metadata={"backend": self.llm.backend, "model": self.llm.name},
        ) as s:
            answer = generate_answer(question, passages, self.llm)
            s.update(output={"answer": answer.text, "usage": answer.usage})

        latency_ms = (time.perf_counter() - started) * 1000
        trace.update(
            output={
                "answer": answer.text,
                "citations": [c.render() for c in answer.citations],
            },
            metadata={"latency_ms": round(latency_ms, 1)},
        )
        self.tracer.flush()

        return AskResult(
            answer=answer,
            passages=passages,
            trace_id=getattr(trace, "id", None),
            latency_ms=latency_ms,
        )
