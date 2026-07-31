"""The governed query path: every guardrail, in the order they have to run.

    mask input PII
      -> detect direct injection
        -> retrieve under the ACL filter (server-side)
          -> detect indirect injection in what came back
            -> generate
              -> mask output PII
                -> audit

Order is not cosmetic:

* **PII masking precedes retrieval** because the query gets embedded, traced and
  logged. Masking after any of those has already lost.
* **The ACL is part of the vector query**, not a filter over results — see
  `governance/acl.py`. It cannot be reordered later without becoming a
  post-filter, which is a different and much weaker control.
* **Indirect-injection detection runs on the retrieved passages**, after
  retrieval and before generation. It is the only place it *can* run: the
  poisoned text does not exist until retrieval produces it.
* **Output masking is last**, because the corpus can contain PII the input never
  did, and that is the leak that actually happens in a real deployment.

Refusal is a decision, not an exception. Every path — answered, abstained,
blocked — produces an `AuditEvent`, because "the system declined" is exactly the
event an audit needs to contain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from generation.answer import NO_EVIDENCE, Answer, generate_answer
from generation.llm import LLM, build_llm
from governance.acl import (
    ANALYST,
    User,
    access_filter,
    apply_classifications,
    classify_documents,
    combine,
    leaked,
)
from governance.audit import (
    DECISION_ABSTAINED,
    DECISION_ANSWERED,
    DECISION_BLOCKED_ACL,
    DECISION_BLOCKED_INJECTION,
    AuditEvent,
    AuditLog,
    build_event,
)
from guardrails.injection import REFUSAL_MESSAGE, InjectionDetector, InjectionVerdict
from guardrails.pii import PiiScrubber, ScrubResult, default_scrubber
from rag.config import settings
from retrieval.configs import RetrievalContext, build_retriever
from retrieval.store import Retrieved, doc_id_filter, search


@dataclass
class GovernedResult:
    answer: Answer
    passages: list[Retrieved]
    decision: str
    event: AuditEvent
    pii_input: ScrubResult
    pii_output: ScrubResult | None
    injection_input: InjectionVerdict
    injection_context: InjectionVerdict
    acl_leaks: list[str]
    latency_ms: float

    @property
    def blocked(self) -> bool:
        return self.decision.startswith("blocked_")


class GovernedPipeline:
    """`RagPipeline` with the M5 controls around it.

    Kept separate rather than folded into `rag/pipeline.py` so the adversarial
    suite can measure the ungoverned path too. "The guardrail helps" is a claim
    that needs a control arm like any other.
    """

    def __init__(
        self,
        context: RetrievalContext | None = None,
        llm: LLM | None = None,
        scrubber: PiiScrubber | None = None,
        audit: AuditLog | None = None,
        restricted_count: int | None = None,
        block_on_injection: bool = True,
        mask_input: bool = True,
        mask_output: bool = True,
        enforce_acl: bool = True,
        apply_acl_payload: bool = True,
    ) -> None:
        self.context = context or RetrievalContext.build()
        self.retriever = build_retriever(settings.retrieval_config, self.context)
        self.llm = llm or build_llm()
        self.scrubber = scrubber or default_scrubber()
        self.detector = InjectionDetector()
        self.audit = audit or AuditLog()
        self.classifications = classify_documents(
            self.context.documents,
            restricted_count=(
                restricted_count if restricted_count is not None else settings.acl_restricted_count
            ),
        )
        if apply_acl_payload:
            # Idempotent, two `set_payload` calls, no re-embedding. Done at
            # construction so the in-process classification map and the payloads
            # the filter reads can never disagree — a drift between them is a
            # silent authorisation bug, which is the worst kind.
            apply_classifications(self.context.client, self.classifications)
        self.block_on_injection = block_on_injection
        self.mask_input = mask_input
        self.mask_output = mask_output
        self.enforce_acl = enforce_acl

    def _retrieve(self, question: str, user: User, top_k: int) -> list[Retrieved]:
        """Retrieve with the ACL and the meeting filter both applied, server-side."""
        meeting_ids = self.retriever.resolve_documents(question)
        filters = combine(
            access_filter(user) if self.enforce_acl else None,
            doc_id_filter(sorted(meeting_ids)) if meeting_ids else None,
        )
        depth = max(self.retriever.config.candidate_k, top_k)

        rankings: list[list[Retrieved]] = []
        if self.retriever.config.dense:
            vector = self.context.embedder.embed_query(question)
            rankings.append(search(self.context.client, vector, top_k=depth, query_filter=filters))
        if self.retriever.config.sparse:
            # BM25 is in-process, so the ACL is applied as an allow-list of the
            # documents this user may see — the same restriction the payload
            # filter expresses on the dense side.
            allowed = self._allowed_doc_ids(user, meeting_ids)
            rankings.append(self.context.bm25.search(question, depth, allowed_doc_ids=allowed))

        from retrieval.fusion import reciprocal_rank_fusion

        candidates = (
            reciprocal_rank_fusion(rankings, k=self.retriever.config.rrf_k, top_k=depth)
            if len(rankings) > 1
            else rankings[0][:depth]
        )
        if self.retriever.config.rerank:
            candidates = self.context.reranker.rerank(question, candidates, top_k=top_k)
        return candidates[:top_k]

    def _allowed_doc_ids(self, user: User, meeting_ids: set[str]) -> list[str] | None:
        if not self.enforce_acl:
            return sorted(meeting_ids) if meeting_ids else None
        allowed = {
            doc_id
            for doc_id, classification in self.classifications.items()
            if user.may_see(classification)
        }
        if meeting_ids:
            allowed &= meeting_ids
        return sorted(allowed)

    def ask(
        self,
        question: str,
        user: User = ANALYST,
        top_k: int | None = None,
        poisoned_passages: list[Retrieved] | None = None,
    ) -> GovernedResult:
        """Answer under the full control set.

        `poisoned_passages` splices attacker-controlled text into the retrieved
        context to exercise indirect injection. Nothing is written to Qdrant —
        the corpus on disk stays a faithful copy of what BACEN published.
        """
        top_k = top_k or settings.top_k
        started = time.perf_counter()

        pii_input = (
            self.scrubber.mask(question)
            if self.mask_input
            else ScrubResult(text=question, findings=(), backend="disabled")
        )
        query = pii_input.text

        injection_input = self.detector.inspect(question)
        if injection_input.detected and self.block_on_injection:
            return self._blocked(
                question,
                query,
                user,
                DECISION_BLOCKED_INJECTION,
                pii_input,
                injection_input,
                started,
            )

        passages = self._retrieve(query, user, top_k)
        if poisoned_passages:
            passages = (poisoned_passages + passages)[:top_k]

        acl_leaks = leaked(passages, self.classifications, user) if self.enforce_acl else []
        if acl_leaks:
            # Should be unreachable: the filter is inside the query. Reaching it
            # means the control failed, and failing closed is the only safe move.
            return self._blocked(
                question,
                query,
                user,
                DECISION_BLOCKED_ACL,
                pii_input,
                injection_input,
                started,
                acl_leaks=acl_leaks,
            )

        injection_context = self.detector.inspect_passages(passages)
        if injection_context.detected and self.block_on_injection:
            return self._blocked(
                question,
                query,
                user,
                DECISION_BLOCKED_INJECTION,
                pii_input,
                injection_input,
                started,
                passages=passages,
                injection_context=injection_context,
            )

        answer = generate_answer(query, passages, self.llm)

        pii_output = (
            self.scrubber.mask(answer.text)
            if self.mask_output
            else ScrubResult(text=answer.text, findings=(), backend="disabled")
        )
        answer.text = pii_output.text

        decision = (
            DECISION_ABSTAINED
            if (not passages or answer.text.startswith(NO_EVIDENCE[:20]))
            else DECISION_ANSWERED
        )
        latency_ms = (time.perf_counter() - started) * 1000

        event = self.audit.append(
            build_event(
                user=user,
                raw_query=question,
                masked_query=query,
                decision=decision,
                hits=passages,
                classifications=self.classifications,
                pii_input=pii_input,
                pii_output=pii_output,
                injection=injection_context,
                acl={"enforced": self.enforce_acl, "leaks": []},
                latency_ms=latency_ms,
            )
        )
        return GovernedResult(
            answer=answer,
            passages=passages,
            decision=decision,
            event=event,
            pii_input=pii_input,
            pii_output=pii_output,
            injection_input=injection_input,
            injection_context=injection_context,
            acl_leaks=[],
            latency_ms=latency_ms,
        )

    def _blocked(
        self,
        question: str,
        query: str,
        user: User,
        decision: str,
        pii_input: ScrubResult,
        injection_input: InjectionVerdict,
        started: float,
        passages: list[Retrieved] | None = None,
        injection_context: InjectionVerdict | None = None,
        acl_leaks: list[str] | None = None,
    ) -> GovernedResult:
        latency_ms = (time.perf_counter() - started) * 1000
        empty = InjectionVerdict(detected=False, signals=())
        answer = Answer(
            question=query,
            text=REFUSAL_MESSAGE,
            citations=[],
            backend=self.llm.backend,
            model=self.llm.name,
        )
        event = self.audit.append(
            build_event(
                user=user,
                raw_query=question,
                masked_query=query,
                decision=decision,
                hits=passages or [],
                classifications=self.classifications,
                pii_input=pii_input,
                injection=injection_context or injection_input,
                acl={"enforced": self.enforce_acl, "leaks": acl_leaks or []},
                latency_ms=latency_ms,
                notes="request refused by a guardrail before an answer was produced",
            )
        )
        return GovernedResult(
            answer=answer,
            passages=passages or [],
            decision=decision,
            event=event,
            pii_input=pii_input,
            pii_output=None,
            injection_input=injection_input,
            injection_context=injection_context or empty,
            acl_leaks=acl_leaks or [],
            latency_ms=latency_ms,
        )
