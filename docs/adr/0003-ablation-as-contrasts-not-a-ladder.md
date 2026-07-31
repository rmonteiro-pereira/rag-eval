# 3. Ablation arms as controlled contrasts, not a cumulative ladder

**Status:** Accepted · M4

## Context

The conventional ablation table is a ladder: baseline, +A, +A+B, +A+B+C, each row
better than the last, ending at the system you shipped. It reads well and it
attributes nothing — if `+A+B` beats `+A`, that is not evidence about B unless
nothing else moved, and in a ladder something else always has.

## Decision

Define seven arms as **pairs differing in exactly one component**, and publish the
per-pair deltas (`CONTRASTS` in `eval/ablation.py`) rather than only the row
ordering. The reranker is measured *with* and *without* the metadata filter,
separately, because those are different questions.

## Alternative rejected

**The cumulative ladder.** Rejected because on this corpus it would have produced
a false narrative. Two arms are *worse* than arms below them:

- `hybrid+rerank` (MRR 0.342) is worse than `hybrid` (0.381).
- `hybrid` (rank-1 correct meeting 0.341) is worse than `bm25` alone (0.927).

A ladder ordered by final score would have shown a clean staircase to 0.741 and
buried both. The two findings that make this project worth reading — that the
cross-encoder does not pay for itself, and that RRF drags a strong arm down when
fused with a near-random one — are only visible as contrasts.

## Consequences

- Seven arms instead of four, so the ablation costs more compute. Acceptable: it
  is ~20 minutes on CPU.
- Adding an arm means adding its contrast pair, not appending a row. Documented
  in `CONTRIBUTING.md`.
- The default serving config (`hybrid+rerank+metadata`) is the best arm, but the
  eval baseline stays pinned to `dense` so the "before" half of every comparison
  cannot drift. Getting these two backwards would silently corrupt every
  before/after number in the repo.

## Reverses if

Components stop being separable — e.g. a single learned retriever replaces the
filter/sparse/rerank stack, at which point there is nothing to contrast and the
honest presentation is end-to-end.
