# 4. The meeting filter is a hard pre-filter, not a soft boost

**Status:** Accepted · M4

## Context

Thirty Copom minutes each contain a near-identical *"Decisão de política
monetária"* paragraph. Dense retrieval found the right paragraph and the wrong
meeting: on the 41 gold questions that name their meeting, rank-1 was correct
**4 times**.

`retrieval/metadata.py` reads a month+year or a meeting ordinal out of the
question and resolves it to a document id. The question is what to do with that
signal.

## Decision

Compile it to a **Qdrant payload filter applied inside the query**. Candidates
from other meetings are never scored, never returned, never reranked.

## Alternative rejected

**A score boost — retrieve normally, then add a bonus to hits from the resolved
meeting.** The safe-looking option: a wrong hint degrades ranking instead of
emptying the result set. Rejected on measured evidence.

The licence for the hard version is `hint_diagnostics` in `ablation.json`:
**41 hints present, 41 resolved, 41 correct — precision 1.000, zero false
positives.** A boost is the right design when the signal is noisy. This signal is
not noisy; it is a regex reading an explicit date out of a question that states
one. Paying ranking-quality for robustness against an error that never occurs is
paying for nothing.

The failure mode a boost protects against is handled directly instead: a hint
that resolves to **no** document in the corpus (asking about the 280th meeting)
**disables the filter** rather than emptying the candidate pool. Absent evidence
is not evidence of absence.

## Consequences

- +0.498 MRR on the bare baseline, at *negative* latency cost — filtering shrinks
  the candidate set.
- Rank-1 correct-meeting accuracy 0.098 → 1.000 on the 41 disambiguation probes.
- The filter cannot help **reverse lookup** (questions naming no meeting, which
  must be identified from content). Those 8 probes are structurally out of reach,
  and 4 still rank the wrong ata first. This is the honest remaining gap.
- A false positive would be *invisible*: the right document would be excluded
  before scoring. This is why the probe is committed and why the regression gate
  gates on it — see ADR 0007.

## Reverses if

`hint_diagnostics.precision_when_resolved` drops below 1.000 on a larger or
validated gold set. At that point the boost becomes the correct design, and the
number that decides it is already being measured every run.
