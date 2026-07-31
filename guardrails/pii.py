"""PII detection and masking, on the way in and on the way out.

Both directions, because they defend against different things:

* **Input masking** stops a user's personal data from being embedded into a query
  vector, written into a trace in Langfuse, or persisted in the audit log. Once a
  CPF is in a trace it is in the trace.
* **Output masking** stops PII that was sitting in the *corpus* from reaching the
  user. The corpus is a third party neither the user nor the model controls, and
  in a real deployment over internal documents this is the leak that actually
  happens.

The masked query is what goes to retrieval. That has a cost worth stating: a
question that genuinely needs a name to be answerable becomes less answerable.
That is the intended trade in a governed system, and the adversarial suite
measures it rather than assuming it away.

Presidio does the general work (`PERSON`, `EMAIL_ADDRESS`, `IBAN_CODE`, ...) and
`guardrails/brazilian.py` supplies the identifiers it misses — measurably,
including the CPF.

The engine is expensive to construct (it loads a spaCy pipeline), so it is built
once and cached. `PiiScrubber` degrades to a regex-only mode if Presidio is not
installed, and says so in `backend` rather than silently masking nothing: a
scrubber that quietly stops scrubbing is worse than one that is absent.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from guardrails.brazilian import (
    BR_PHONE_ENTITY,
    CEP_ENTITY,
    CNPJ_ENTITY,
    CPF_ENTITY,
    brazilian_recognizers,
    is_valid_cnpj,
    is_valid_cpf,
)

#: Entities masked by default. `URL` is deliberately absent — the corpus is full
#: of bcb.gov.br links and masking them would destroy every citation.
DEFAULT_ENTITIES: tuple[str, ...] = (
    CPF_ENTITY,
    CNPJ_ENTITY,
    CEP_ENTITY,
    BR_PHONE_ENTITY,
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
)

#: Below this, a finding is reported but not masked. Presidio's generic
#: `PHONE_NUMBER` scores Brazilian numbers at 0.4 and `PERSON` fires on ordinary
#: capitalised words in Portuguese, so a floor is needed to keep the masker from
#: eating the text it is protecting.
DEFAULT_THRESHOLD = 0.5

#: Fallback patterns for when Presidio is unavailable. Validated where possible,
#: so the degraded mode is still precise about the identifiers that matter most.
_FALLBACK_PATTERNS: tuple[tuple[str, re.Pattern[str], Callable[[str], bool] | None], ...] = (
    (CPF_ENTITY, re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), is_valid_cpf),
    (CNPJ_ENTITY, re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"), is_valid_cnpj),
    ("EMAIL_ADDRESS", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), None),
    (BR_PHONE_ENTITY, re.compile(r"\(\d{2}\)\s?9?\d{4}-?\d{4}\b"), None),
    (CEP_ENTITY, re.compile(r"\b\d{5}-\d{3}\b"), None),
)


@dataclass(frozen=True)
class PiiFinding:
    entity_type: str
    start: int
    end: int
    score: float
    text: str

    def to_json(self) -> dict:
        # The matched text is *not* included. An audit log that records what was
        # redacted, verbatim, is a log that leaks exactly what the masker
        # existed to protect.
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "score": round(self.score, 4),
            "length": self.end - self.start,
        }


@dataclass(frozen=True)
class ScrubResult:
    text: str
    findings: tuple[PiiFinding, ...]
    backend: str

    @property
    def masked(self) -> bool:
        return bool(self.findings)

    @property
    def entity_types(self) -> list[str]:
        return sorted({finding.entity_type for finding in self.findings})

    def to_json(self) -> dict:
        return {
            "masked": self.masked,
            "backend": self.backend,
            "entity_types": self.entity_types,
            "n_findings": len(self.findings),
            "findings": [finding.to_json() for finding in self.findings],
        }


@lru_cache(maxsize=1)
def _analyzer():
    """Presidio with the Portuguese pipeline plus the Brazilian recognisers."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "pt", "model_name": "pt_core_news_sm"}],
        }
    )
    engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["pt"])
    for recognizer in brazilian_recognizers():
        engine.registry.add_recognizer(recognizer)
    return engine


def _mask_token(entity_type: str) -> str:
    return f"[{entity_type}]"


class PiiScrubber:
    def __init__(
        self,
        entities: tuple[str, ...] = DEFAULT_ENTITIES,
        threshold: float = DEFAULT_THRESHOLD,
        use_presidio: bool = True,
    ) -> None:
        self.entities = entities
        self.threshold = threshold
        self.backend = "regex-fallback"
        self._engine = None
        if use_presidio:
            try:
                self._engine = _analyzer()
                self.backend = "presidio+spacy-pt"
            except Exception:  # noqa: BLE001 - absence of presidio is a config state
                self._engine = None

    def scan(self, text: str) -> list[PiiFinding]:
        if not text:
            return []
        findings = self._scan_presidio(text) if self._engine else self._scan_fallback(text)
        return _drop_overlaps(sorted(findings, key=lambda f: (f.start, -f.score)))

    def _scan_presidio(self, text: str) -> list[PiiFinding]:
        # Only reachable via `scan()`, which checks `self._engine` first. Asserted
        # rather than assumed: silently scanning with no engine would return zero
        # findings, which reads exactly like "no PII present".
        assert self._engine is not None, "presidio backend selected without an engine"
        results = self._engine.analyze(text=text, language="pt", entities=list(self.entities))
        return [
            PiiFinding(
                entity_type=result.entity_type,
                start=result.start,
                end=result.end,
                score=result.score,
                text=text[result.start : result.end],
            )
            for result in results
            if result.score >= self.threshold
        ]

    def _scan_fallback(self, text: str) -> list[PiiFinding]:
        findings = []
        for entity_type, pattern, validator in _FALLBACK_PATTERNS:
            if entity_type not in self.entities:
                continue
            for match in pattern.finditer(text):
                if validator is not None and not validator(match.group(0)):
                    continue
                findings.append(
                    PiiFinding(
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        score=1.0,
                        text=match.group(0),
                    )
                )
        return findings

    def mask(self, text: str) -> ScrubResult:
        """Replace every finding with `[ENTITY_TYPE]`, right to left."""
        findings = self.scan(text)
        masked = text
        for finding in sorted(findings, key=lambda f: -f.start):
            token = _mask_token(finding.entity_type)
            masked = masked[: finding.start] + token + masked[finding.end :]
        return ScrubResult(text=masked, findings=tuple(findings), backend=self.backend)

    def leaks(self, text: str) -> list[PiiFinding]:
        """PII present in text that should not contain any — the output check."""
        return self.scan(text)


def _drop_overlaps(findings: list[PiiFinding]) -> list[PiiFinding]:
    """Keep the highest-scoring finding of any overlapping group.

    Presidio's `PERSON` and the CPF recogniser can both claim the same span when
    a name sits next to a number; masking both would corrupt the offsets.
    """
    kept: list[PiiFinding] = []
    for finding in findings:
        if any(finding.start < other.end and other.start < finding.end for other in kept):
            continue
        kept.append(finding)
    return kept


@lru_cache(maxsize=1)
def default_scrubber() -> PiiScrubber:
    return PiiScrubber()
