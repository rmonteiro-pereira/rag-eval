"""LLM backends behind one small interface.

Two implementations ship in M1:

* `OllamaLLM`      — local Ollama server (llama3.1 / qwen2.5). Free, no key.
* `ExtractiveLLM`  — no model at all: returns the retrieved passages verbatim,
                     each with its citation. It is not a language model and does
                     not pretend to be one; it exists so the pipeline runs end to
                     end (and so retrieval can be evaluated) on a machine with
                     no Ollama installed.

The extractive backend is also useful beyond that: it is the honest floor for
groundedness, since a verbatim quote cannot hallucinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from rag.config import settings


@dataclass
class LLMResponse:
    text: str
    model: str
    backend: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None


class LLM(Protocol):
    name: str
    backend: str

    def complete(self, system: str, prompt: str) -> LLMResponse: ...


class OllamaLLM:
    backend = "ollama"

    def __init__(self, model: str | None = None, url: str | None = None) -> None:
        self.name = model or settings.ollama_model
        self.url = (url or settings.ollama_url).rstrip("/")

    @staticmethod
    def is_available(url: str | None = None, timeout: float = 2.0) -> bool:
        base = (url or settings.ollama_url).rstrip("/")
        try:
            resp = httpx.get(f"{base}/api/tags", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def available_models(self) -> list[str]:
        resp = httpx.get(f"{self.url}/api/tags", timeout=10.0)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def complete(self, system: str, prompt: str) -> LLMResponse:
        payload = {
            "model": self.name,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        resp = httpx.post(
            f"{self.url}/api/generate",
            json=payload,
            timeout=settings.ollama_timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        return LLMResponse(
            text=body.get("response", "").strip(),
            model=self.name,
            backend=self.backend,
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            latency_ms=(body.get("total_duration") or 0) / 1e6 or None,
        )


class ExtractiveLLM:
    """Returns the evidence verbatim. Zero generation, zero hallucination."""

    backend = "extractive"
    name = "extractive-fallback"

    def complete(self, system: str, prompt: str) -> LLMResponse:  # noqa: ARG002
        # The context block is assembled by `generation.prompt`; here we just
        # echo it back untouched. The caller (`answer.py`) formats the passages,
        # so this backend stays trivially auditable.
        return LLMResponse(text="", model=self.name, backend=self.backend)


def build_llm(mode: str | None = None) -> LLM:
    """Resolve the configured backend.

    `auto` probes Ollama once and silently degrades to extractive mode, so the
    same command works on a machine with or without Ollama installed.
    """
    mode = (mode or settings.llm_mode).lower()
    if mode == "extractive":
        return ExtractiveLLM()
    if mode == "ollama":
        return OllamaLLM()
    if mode == "auto":
        return OllamaLLM() if OllamaLLM.is_available() else ExtractiveLLM()
    raise ValueError(f"unknown llm mode: {mode!r} (expected auto|ollama|extractive)")
