"""Append-only audit log: who asked what, what was retrieved, what was decided.

One JSON object per line, one line per query. JSONL rather than a database
because the useful properties here are append-only, greppable, diffable and
trivially shippable to anything else — and because a log that needs a running
service to be readable tends not to be read.

## What is recorded, and what is deliberately not

Recorded: the user, the *masked* query, which guardrails fired, which documents
were retrieved and their classifications, the decision, and a hash of the raw
query.

**Not recorded: the raw query, the answer text, or any matched PII substring.**
That is the point of the design. An audit log that stores the unmasked query is a
second copy of exactly the data the masker exists to contain — and it is usually
the copy that gets exfiltrated, because logs are shipped, backed up and given
broader read access than the primary store.

The SHA-256 of the raw query is stored instead. It supports the questions an
audit actually needs to answer — "did this exact query happen before", "is this
the query in the incident report" — without the log itself being a PII store.
Correlating a hash back to its text requires already knowing the text.

Answer text is omitted for the same reason: the answer quotes the corpus, and in
a real deployment over internal documents the corpus is the sensitive thing. What
*is* recorded is which documents were used, which is what an audit needs — the
answer is reconstructible from the trace in Langfuse if someone with the right
access needs it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rag.config import REPO_ROOT

DEFAULT_AUDIT_PATH = REPO_ROOT / "data" / "audit" / "audit.jsonl"

_LOCK = threading.Lock()

#: Terminal outcomes. `answered` and `abstained` are both successful operation;
#: the `blocked_*` values are the guardrail having fired.
DECISION_ANSWERED = "answered"
DECISION_ABSTAINED = "abstained"
DECISION_BLOCKED_INJECTION = "blocked_injection"
DECISION_BLOCKED_PII = "blocked_pii"
DECISION_BLOCKED_ACL = "blocked_acl"


def query_fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


@dataclass
class AuditEvent:
    timestamp: str
    user_id: str
    clearances: list[str]
    query_masked: str
    query_sha256: str
    decision: str
    retrieved: list[dict] = field(default_factory=list)
    pii_input: dict | None = None
    pii_output: dict | None = None
    injection: dict | None = None
    acl: dict | None = None
    latency_ms: float | None = None
    notes: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def build_event(
    user,
    raw_query: str,
    masked_query: str,
    decision: str,
    hits: Sequence = (),
    classifications: dict[str, str] | None = None,
    pii_input=None,
    pii_output=None,
    injection=None,
    acl: dict | None = None,
    latency_ms: float | None = None,
    notes: str = "",
) -> AuditEvent:
    classifications = classifications or {}
    return AuditEvent(
        timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
        user_id=user.user_id,
        clearances=sorted(user.clearances),
        query_masked=masked_query,
        query_sha256=query_fingerprint(raw_query),
        decision=decision,
        retrieved=[
            {
                "doc_id": hit.doc_id,
                "chunk_index": hit.chunk_index,
                "page": hit.page_number,
                "classification": classifications.get(hit.doc_id, "unknown"),
                "score": round(float(hit.score), 6),
            }
            for hit in hits
        ],
        pii_input=pii_input.to_json() if pii_input is not None else None,
        pii_output=pii_output.to_json() if pii_output is not None else None,
        injection=injection.to_json() if injection is not None else None,
        acl=acl,
        latency_ms=round(latency_ms, 1) if latency_ms is not None else None,
        notes=notes,
    )


class AuditLog:
    """Append-only JSONL sink."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_AUDIT_PATH

    def append(self, event: AuditEvent) -> AuditEvent:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_json(), ensure_ascii=False) + "\n"
        # Lock plus append-mode open: `os.O_APPEND` writes are atomic for lines
        # below the pipe buffer, so concurrent writers cannot interleave a record.
        with _LOCK, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.read())
