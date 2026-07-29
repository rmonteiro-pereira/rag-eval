"""Loading and validating the gold set.

The file is JSONL with one leading `_comment` record (see
`eval/datasets/README.md`). Loading is strict: an unknown `status`, a missing
field or an answerable row without a span is a hard error, not a warning. A
gold set that silently degrades is worse than no gold set — the metrics keep
being produced and nobody notices the reference moved.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from rag.config import REPO_ROOT

DEFAULT_GOLD_PATH = REPO_ROOT / "eval" / "datasets" / "gold_seed.jsonl"

#: Statuses ordered by how much trust they carry. `--min-status validated`
#: keeps only rows at or above `validated`, which is what turns the human
#: validation pass into a one-command re-run.
STATUS_ORDER: dict[str, int] = {"draft": 0, "validated": 1}

#: Never scored, under any filter. Kept in the file as a record of what was
#: tried and rejected.
EXCLUDED_STATUSES: frozenset[str] = frozenset({"rejected"})

ANSWER_TYPES: frozenset[str] = frozenset({"extractive", "abstractive", "abstention"})

REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "status",
    "question",
    "answer",
    "answer_type",
    "source_doc_id",
    "source_page",
    "source_span",
    "difficulty",
    "capability",
)


@dataclass(frozen=True)
class GoldRow:
    id: str
    status: str
    question: str
    answer: str
    answer_type: str
    source_doc_id: str | None
    source_title: str | None
    source_page: int | None
    source_span: str | None
    difficulty: str
    capability: str
    notes: str = ""

    @property
    def is_abstention(self) -> bool:
        """True when the corpus cannot answer and refusal is the right behaviour.

        These rows carry no span, so they are excluded from retrieval metrics
        and counted separately; the abstention metric itself lands in M5.
        """
        return self.answer_type == "abstention"


class GoldSetError(ValueError):
    """Raised when the gold file violates its own schema."""


def _iter_records(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldSetError(f"{path}:{lineno}: not valid JSON — {exc}") from exc
            if "_comment" in record:
                continue
            yield lineno, record


def _parse_row(path: Path, lineno: int, record: dict) -> GoldRow:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise GoldSetError(f"{path}:{lineno}: missing field(s) {', '.join(missing)}")

    status = record["status"]
    if status not in STATUS_ORDER and status not in EXCLUDED_STATUSES:
        raise GoldSetError(f"{path}:{lineno}: unknown status {status!r}")

    answer_type = record["answer_type"]
    if answer_type not in ANSWER_TYPES:
        raise GoldSetError(f"{path}:{lineno}: unknown answer_type {answer_type!r}")

    row = GoldRow(
        id=record["id"],
        status=status,
        question=record["question"],
        answer=record["answer"],
        answer_type=answer_type,
        source_doc_id=record["source_doc_id"],
        source_title=record.get("source_title"),
        source_page=record["source_page"],
        source_span=record["source_span"],
        difficulty=record["difficulty"],
        capability=record["capability"],
        notes=record.get("notes", ""),
    )

    if row.is_abstention:
        if row.source_doc_id is not None:
            raise GoldSetError(f"{path}:{lineno}: {row.id} is abstention but names a source doc")
    else:
        if not row.source_doc_id:
            raise GoldSetError(f"{path}:{lineno}: {row.id} is answerable but has no source_doc_id")
        if not row.source_span:
            raise GoldSetError(f"{path}:{lineno}: {row.id} is answerable but has no source_span")
        if not isinstance(row.source_page, int):
            raise GoldSetError(f"{path}:{lineno}: {row.id} has a non-integer source_page")

    return row


def load_gold(
    path: Path | str | None = None,
    min_status: str = "draft",
) -> list[GoldRow]:
    """Load the gold set, keeping rows at or above ``min_status``.

    Raises on duplicate ids — ids are stable identifiers and reusing one
    silently rewrites history in every report that already cited it.
    """
    if min_status not in STATUS_ORDER:
        raise GoldSetError(
            f"unknown --min-status {min_status!r}; expected one of {sorted(STATUS_ORDER)}"
        )
    path = Path(path) if path is not None else DEFAULT_GOLD_PATH
    if not path.exists():
        raise GoldSetError(f"gold set not found: {path}")

    threshold = STATUS_ORDER[min_status]
    rows: list[GoldRow] = []
    seen: set[str] = set()

    for lineno, record in _iter_records(path):
        row = _parse_row(path, lineno, record)
        if row.id in seen:
            raise GoldSetError(f"{path}:{lineno}: duplicate id {row.id!r}")
        seen.add(row.id)
        if row.status in EXCLUDED_STATUSES:
            continue
        if STATUS_ORDER[row.status] >= threshold:
            rows.append(row)

    return rows


def status_counts(path: Path | str | None = None) -> dict[str, int]:
    """Count every row in the file by status, ignoring any filter."""
    path = Path(path) if path is not None else DEFAULT_GOLD_PATH
    counts: dict[str, int] = {}
    for lineno, record in _iter_records(path):
        row = _parse_row(path, lineno, record)
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts
