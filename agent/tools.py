"""The two tools, and the layered defence around the SQL one.

`sql_query` executes model-written SQL. That is the highest-risk thing anywhere
in this repo, so it gets defence in depth rather than one check:

1. **The connection is opened `read_only=True`.** DuckDB refuses writes at the
   engine level. Everything below is belt-and-braces on top of a connection that
   physically cannot mutate the file.
2. **One statement only.** A trailing `; DROP TABLE` is rejected before parsing
   goes any further, so no later check can be tricked by a second statement.
3. **`SELECT`/`WITH` only**, checked on the leading keyword, with a deny-list of
   mutating verbs checked as whole words anywhere in the text.
4. **A `LIMIT` is appended** when none is present. Not a security control — a
   sanity one. A model that writes `SELECT * FROM mart_futures_curve` against
   1.6M rows produces a context-window incident, not an answer.
5. **The human gate** (`agent/hitl.py`) sees the final SQL, after normalisation,
   and can refuse it.

The database is a **read-only artifact produced by another project** and is not
in this repo: `_artifacts/ofl_gold.duckdb`, 70 MB, eight gold marts exported from
the lakehouse's MinIO. Nothing here writes to it, and nothing here commits it.

The schema is read from the live file rather than hardcoded. A model prompted
with a schema that has drifted from the database writes SQL that fails, and the
failure looks like a model problem.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rag.config import settings

#: Verbs that must never appear. Whole-word matched, so a column named
#: `updated_at` does not trip the `UPDATE` rule.
_FORBIDDEN = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "attach",
    "detach",
    "copy",
    "export",
    "import",
    "install",
    "load",
    "pragma",
    "set",
    "call",
    "vacuum",
    "checkpoint",
    "truncate",
    "grant",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN) + r")\b", re.IGNORECASE)

_LIMIT_RE = re.compile(r"\blimit\s+\d+", re.IGNORECASE)
_LEADING = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

DEFAULT_ROW_LIMIT = 50
#: Marts only. `_export_manifest` is provenance, exposed on purpose so the agent
#: can report where the data came from.
VISIBLE_TABLES = re.compile(r"^(mart_|_export_manifest$)")


class SqlRejected(ValueError):
    """The statement never reached the database."""


@dataclass
class SqlResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    row_count: int
    truncated: bool
    elapsed_ms: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def to_markdown(self, max_rows: int = 12) -> str:
        if self.error:
            return f"_erro:_ `{self.error}`"
        if not self.rows:
            return "_(nenhuma linha)_"
        head = "| " + " | ".join(self.columns) + " |"
        rule = "|" + "|".join("---" for _ in self.columns) + "|"
        body = [
            "| " + " | ".join(_fmt(value) for value in row) + " |" for row in self.rows[:max_rows]
        ]
        extra = (
            [f"_... {len(self.rows) - max_rows} more rows_"] if len(self.rows) > max_rows else []
        )
        return "\n".join([head, rule, *body, *extra])

    def to_json(self) -> dict:
        return {
            "sql": self.sql,
            "columns": self.columns,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "error": self.error,
            "rows": [[_fmt(v) for v in row] for row in self.rows[:12]],
        }


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def normalise_sql(sql: str, row_limit: int = DEFAULT_ROW_LIMIT) -> str:
    """Validate and canonicalise. Raises `SqlRejected` rather than returning junk."""
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise SqlRejected("empty statement")
    if ";" in text:
        raise SqlRejected("multiple statements are not allowed")
    if not _LEADING.match(text):
        raise SqlRejected("only SELECT and WITH statements are allowed")
    if (match := _FORBIDDEN_RE.search(text)) is not None:
        raise SqlRejected(f"forbidden keyword: {match.group(0).upper()}")
    # Caught here rather than left to DuckDB, because of what happens next.
    # llama3.1 regularly drops the closing quote on a date literal
    # (`month <= '2025-12-01`), and appending `LIMIT 50` to that produces a
    # parser error pointing at the LIMIT — so the model reads the feedback as
    # "the LIMIT is wrong", removes it, and loops. Naming the actual defect is
    # what lets it fix the actual defect. Escaped quotes double up (`''`), so
    # parity holds for well-formed SQL.
    if text.count("'") % 2:
        raise SqlRejected("unbalanced single quote — a string literal is not closed")
    if text.count('"') % 2:
        raise SqlRejected("unbalanced double quote — an identifier is not closed")
    if not _LIMIT_RE.search(text):
        text = f"{text}\nLIMIT {row_limit}"
    return text


@dataclass
class SqlQueryTool:
    """Read-only SELECT over the lakehouse gold marts."""

    name: str = "sql_query"
    database: Path = field(default_factory=lambda: settings.gold_duckdb_path)
    row_limit: int = DEFAULT_ROW_LIMIT

    description: str = (
        "Executa uma consulta SQL SELECT (DuckDB) sobre os marts gold do lakehouse "
        "(series macroeconomicas mensais e diarias: Selic, IPCA, cambio, curva de "
        'juros). Use para NUMEROS e SERIES TEMPORAIS. Argumento: {"sql": "SELECT ..."}'
    )

    @property
    def available(self) -> bool:
        return self.database.exists()

    def connect(self):
        import duckdb

        if not self.available:
            raise FileNotFoundError(f"gold marts not found at {self.database}")
        return duckdb.connect(str(self.database), read_only=True)

    def schema(self) -> dict[str, list[tuple[str, str]]]:
        """`{table: [(column, type), ...]}` for the visible marts, read live."""
        with self.connect() as con:
            tables = [
                name
                for (name,) in con.execute("SHOW TABLES").fetchall()
                if VISIBLE_TABLES.match(name)
            ]
            return {
                table: [(row[0], row[1]) for row in con.execute(f'DESCRIBE "{table}"').fetchall()]
                for table in tables
            }

    def schema_prompt(self) -> str:
        """The schema block handed to the model."""
        lines = []
        for table, columns in sorted(self.schema().items()):
            rendered = ", ".join(f"{name} {dtype}" for name, dtype in columns)
            lines.append(f"- {table}({rendered})")
        return "\n".join(lines)

    def run(self, sql: str) -> SqlResult:
        try:
            statement = normalise_sql(sql, self.row_limit)
        except SqlRejected as exc:
            return SqlResult(
                sql=sql,
                columns=[],
                rows=[],
                row_count=0,
                truncated=False,
                elapsed_ms=0.0,
                error=f"rejeitado: {exc}",
            )

        started = time.perf_counter()
        try:
            with self.connect() as con:
                cursor = con.execute(statement)
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
        except Exception as exc:  # noqa: BLE001 - a bad query is data, not a crash
            return SqlResult(
                sql=statement,
                columns=[],
                rows=[],
                row_count=0,
                truncated=False,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

        elapsed = (time.perf_counter() - started) * 1000
        return SqlResult(
            sql=statement,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=len(rows) >= self.row_limit,
            elapsed_ms=elapsed,
        )


class _AskablePipeline(Protocol):
    """What this tool needs from a pipeline, and nothing more.

    Structural rather than a `GovernedPipeline` import: `guardrails.pipeline`
    imports from `retrieval`, which would make this a cycle, and stating the one
    method actually used documents the coupling better than the concrete class
    would. Any object with a compatible `ask` is a valid substitute — which is
    what the tests pass.
    """

    def ask(self, question: str, user: Any = ..., top_k: int | None = ...) -> Any: ...


@dataclass
class RagSearchTool:
    """Retrieval over the Copom minutes, through the governed pipeline."""

    pipeline: _AskablePipeline
    name: str = "rag_search"
    user: object = None
    top_k: int = 4

    description: str = (
        "Busca nas atas do Copom (texto em portugues, 2022-2026) e retorna trechos "
        "com citacao. Use para DECISOES, JUSTIFICATIVAS e o que o comite DISSE. "
        'Argumento: {"question": "..."}'
    )

    def run(self, question: str) -> dict:
        from governance.acl import SUPERVISOR

        result = self.pipeline.ask(question, user=self.user or SUPERVISOR, top_k=self.top_k)
        return {
            "decision": result.decision,
            "answer": result.answer.text,
            "sources": [
                {
                    "doc_id": p.doc_id,
                    "title": p.title,
                    "page": p.page_number,
                }
                for p in result.passages
            ],
            "excerpt": "\n\n".join(p.text[:400] for p in result.passages[:2]),
        }
