"""Langfuse tracing, wrapped so the pipeline never breaks when it is absent.

Tracing is observability, not business logic: if the Langfuse container is down
or the SDK is missing, `ask` must still answer. Every call therefore goes
through a no-op shim on failure.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from rag.config import settings

log = logging.getLogger(__name__)


class _NoopSpan:
    def update(self, **kwargs: Any) -> None:  # noqa: ARG002
        pass

    def end(self, **kwargs: Any) -> None:  # noqa: ARG002
        pass


class _NoopTrace:
    id = None

    def span(self, **kwargs: Any) -> _NoopSpan:  # noqa: ARG002
        return _NoopSpan()

    def generation(self, **kwargs: Any) -> _NoopSpan:  # noqa: ARG002
        return _NoopSpan()

    def update(self, **kwargs: Any) -> None:  # noqa: ARG002
        pass


class Tracer:
    """Minimal facade over the Langfuse client."""

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = settings.langfuse_enabled if enabled is None else enabled
        self._client = None
        if self.enabled:
            self._client = self._connect()
            self.enabled = self._client is not None

    def _connect(self):
        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            if not client.auth_check():
                log.warning("Langfuse auth check failed — continuing without tracing")
                return None
            return client
        except Exception as exc:  # pragma: no cover - depends on local infra
            log.warning("Langfuse unavailable (%s) — continuing without tracing", exc)
            return None

    def trace(self, name: str, **kwargs: Any):
        if not self._client:
            return _NoopTrace()
        try:
            return self._client.trace(name=name, **kwargs)
        except Exception as exc:  # pragma: no cover
            log.warning("Langfuse trace failed (%s)", exc)
            return _NoopTrace()

    def flush(self) -> None:
        if self._client:
            try:
                self._client.flush()
            except Exception as exc:  # pragma: no cover
                log.warning("Langfuse flush failed (%s)", exc)

    @property
    def host(self) -> str:
        return settings.langfuse_host


@contextmanager
def span(trace, name: str, **kwargs: Any):
    """Context manager over a Langfuse span; yields the span for `.update(...)`."""
    s = trace.span(name=name, **kwargs)
    try:
        yield s
    finally:
        try:
            s.end()
        except Exception:  # pragma: no cover
            pass
