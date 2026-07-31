# 1. An evaluation harness before features

**Status:** Accepted · M1

## Context

A RAG demo takes an afternoon. The hard part is knowing whether it works, and by
how much, and which part of it is responsible. Most portfolio RAG projects ship
the pipeline and assert the quality.

## Decision

Build the measurement apparatus first and let it decide what gets built next. The
gold set, `run_eval`, and the committed baseline report landed before hybrid
retrieval, the reranker, generation, or serving. Every subsequent component had
to justify itself against a number that already existed.

## Alternative rejected

**Build the pipeline, add evaluation at the end.** Rejected because the "before"
number stops being available the moment you improve the system. The M1 baseline —
MRR 0.191, and the specific finding that dense retrieval returned the right
paragraph from the *wrong meeting* — could not have been recovered afterwards. It
is the entire premise of the M4 ablation, and it exists only because nothing had
been optimised yet.

The cost of this order is real and worth naming: the first committed result was
**bad**, and it stayed the headline for two milestones.

## Consequences

- `eval.run_eval` still defaults to `--config dense`, permanently, so the command
  that produced the baseline keeps producing it. See ADR 0003.
- Every claim in the README traces to a JSON report with a per-query audit trail.
- The project reads as an evaluation project that happens to contain a RAG
  system, which is the intended emphasis.

## Reverses if

Nothing plausible. If the corpus changed such that a fresh baseline could be
measured cheaply, the ordering would matter less — but the baseline is a
historical artifact, not a rerunnable one, so it cannot be recovered by spending
more compute later.
