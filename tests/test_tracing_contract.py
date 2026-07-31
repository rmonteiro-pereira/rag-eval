"""The tracing facade must call methods that exist on the installed SDK.

Written after triaging Dependabot PR #6 (langfuse 2.60.10 -> 4.14.1), which was
**green on CI** and would have silently destroyed observability.

The mechanism is worth stating, because it is this repo's recurring failure
family rather than a one-off. `rag/tracing.py` wraps every Langfuse call in
`except Exception` and degrades to a no-op, on the correct principle that
tracing is observability and must never break the pipeline. But v4 removed
`Langfuse.trace()` — the single entry point the facade uses — so on v4 that
defensive `except` turns a total API removal into a warning line and a silent
no-op. Every `rag.ask` keeps working. Every test keeps passing. CI stays green.
The traces just stop.

Nothing in this suite exercised Langfuse, so nothing could have noticed. This
test is the cheap thing that would have:

    assert the methods the facade calls exist on the SDK it is pinned to.

It needs no server, no network and no credentials — only the installed package.
"""

from __future__ import annotations

import pytest

#: (object, attribute) pairs `rag/tracing.py` depends on. Each is called behind
#: an `except Exception`, which is exactly why their absence is invisible.
CLIENT_METHODS = ("trace", "auth_check", "flush")
TRACE_METHODS = ("span", "generation", "update")


def test_the_installed_sdk_has_the_client_methods_the_facade_calls():
    """Kills the failure mode of PR #6: `client.trace()` gone in langfuse v4."""
    langfuse = pytest.importorskip("langfuse")
    client_cls = langfuse.Langfuse
    missing = [m for m in CLIENT_METHODS if not hasattr(client_cls, m)]
    assert not missing, (
        f"langfuse {getattr(langfuse, '__version__', '?')} has no "
        f"{missing} on Langfuse. `rag/tracing.py` calls these behind "
        "`except Exception`, so the pipeline will keep working and tracing will "
        "silently stop. Migrate the facade before moving the pin."
    )


def test_the_sdk_major_matches_the_self_hosted_server_pin():
    """SDK and server majors must agree, and the server pin is a real decision.

    `docker-compose.yml` pins `langfuse/langfuse:2` deliberately: v3+ additionally
    requires ClickHouse, Redis and MinIO, which is a different project from "runs
    on a laptop, free". An SDK from a later major speaks an ingestion API that
    server does not serve, so the two pins are one decision, not two.
    """
    import re
    from pathlib import Path

    langfuse = pytest.importorskip("langfuse")
    sdk_major = int(str(langfuse.__version__).split(".")[0])

    compose = Path(__file__).resolve().parents[1] / "docker-compose.yml"
    m = re.search(r"image:\s*langfuse/langfuse:(\d+)", compose.read_text(encoding="utf-8"))
    assert m, "could not find the langfuse server image pin in docker-compose.yml"
    server_major = int(m.group(1))

    assert sdk_major == server_major, (
        f"langfuse SDK major {sdk_major} vs server image major {server_major}. "
        "Bumping one without the other means the SDK talks to a server that does "
        "not serve its ingestion API — and `rag/tracing.py` will swallow the "
        "failure rather than report it."
    )
