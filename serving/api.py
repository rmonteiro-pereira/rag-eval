"""FastAPI serving layer.

    uv run uvicorn serving.api:app --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"Qual foi a decisao do Copom em junho de 2026?"}'

Serves the **governed** pipeline, not the bare one. PII masking, injection
detection, the document ACL and the audit log all apply — the HTTP layer is the
one place where forgetting that would be least visible and most costly, so there
is no code path here that reaches retrieval without them.

`user` on the request selects a role from `governance/acl.py`. In a real
deployment that comes from an authenticated session, never from the request body;
the endpoint says so in its own docstring and in the response, because an ACL
whose subject is chosen by the caller is not an ACL. It is wired this way so the
demo can show both sides of the boundary from one browser tab.

The pipeline is built lazily on the first request and cached. Constructing it
loads bge-m3, the cross-encoder and a spaCy pipeline — about fifteen seconds — and
doing that at import time makes the process look hung on startup and breaks
`--reload`.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from governance.acl import ANALYST, SUPERVISOR, User
from rag.config import settings

ROLES: dict[str, User] = {"analyst": ANALYST, "supervisor": SUPERVISOR}

app = FastAPI(
    title="rag-eval",
    description=(
        "Governed RAG over BACEN Copom minutes. Every answer passes PII masking, "
        "injection detection, a document-level ACL and an audit log."
    ),
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    #: Demo-only. In a real deployment the subject comes from the session.
    user: str = Field(default="analyst")
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    doc_id: str
    title: str
    page: int
    chunk_index: int
    score: float
    url: str
    classification: str


class AskResponse(BaseModel):
    question: str
    answer: str
    decision: str
    sources: list[Source]
    guardrails: dict
    latency_ms: float
    warning: str = (
        "`user` is taken from the request body for demonstration only. In a real "
        "deployment the subject comes from an authenticated session — an ACL whose "
        "subject is chosen by the caller is not an ACL."
    )


@lru_cache(maxsize=1)
def get_pipeline():
    from guardrails.pipeline import GovernedPipeline

    return GovernedPipeline()


@app.get("/health")
def health() -> dict:
    """Cheap liveness check that does not build the pipeline."""
    return {"status": "ok", "collection": settings.qdrant_collection}


@app.get("/config")
def config() -> dict:
    return {
        "retrieval_config": settings.retrieval_config,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "llm_mode": settings.llm_mode,
        "roles": sorted(ROLES),
        "acl_restricted_count": settings.acl_restricted_count,
        "acl_note": (
            "The document classification is SYNTHETIC — these are public BACEN "
            "documents. The most recent meetings stand in for a publication embargo."
        ),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if request.user not in ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown user {request.user!r}; known: {', '.join(sorted(ROLES))}",
        )

    started = time.perf_counter()
    pipeline = get_pipeline()
    result = pipeline.ask(request.question, user=ROLES[request.user], top_k=request.top_k)

    return AskResponse(
        question=request.question,
        answer=result.answer.text,
        decision=result.decision,
        sources=[
            Source(
                doc_id=p.doc_id,
                title=p.title,
                page=p.page_number,
                chunk_index=p.chunk_index,
                score=round(float(p.score), 6),
                url=p.url,
                classification=pipeline.classifications.get(p.doc_id, "unknown"),
            )
            for p in result.passages
        ],
        guardrails={
            "pii_input": result.pii_input.to_json(),
            "pii_output": result.pii_output.to_json() if result.pii_output else None,
            "injection_input": result.injection_input.to_json(),
            "injection_context": result.injection_context.to_json(),
            "acl_leaks": result.acl_leaks,
            "audit_decision": result.event.decision,
        },
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
