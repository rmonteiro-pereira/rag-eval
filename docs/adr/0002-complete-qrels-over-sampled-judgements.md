# 2. Complete qrels with graded gains, not sampled judgements

**Status:** Accepted · M2

## Context

To compute recall and nDCG you need to know, for each query, which chunks are
relevant. TREC-style evaluation samples: pool the top-k of several systems, judge
that pool, treat everything unjudged as irrelevant. That works at web scale, where
judging the whole collection is impossible.

This collection is **636 chunks**. Judging all of it is possible.

## Decision

Derive **complete** qrels mechanically from the gold span. For each gold row, the
chunk containing the cited span scores **gain 2**; other chunks from the same
document *and* page score **gain 1**; everything else is 0. Every chunk in the
collection therefore has a label, so recall has a true denominator and nDCG has a
true ideal ranking.

## Alternative rejected

**Pooled, sampled judgements.** Rejected for two reasons:

1. With 636 chunks, sampling would buy nothing and cost correctness. Unjudged-is-
   irrelevant systematically punishes whichever system retrieves *differently*
   from the ones that built the pool — precisely the comparison an ablation makes.
2. The pool would have to be rebuilt whenever an arm was added, so numbers from
   different milestones would not be comparable. Seven arms exist now.

**Binary relevance** was also rejected: page-level and span-level matches are not
the same thing, and collapsing them hides the failure mode this project is about
— retrieving the right *kind* of paragraph from the wrong meeting.

## Consequences

- `recall@1` is capped at `1/|relevant|` and looks broken. It is not. Documented
  in the README and printed by the harness; read `hit_rate@k` at low k.
- Adding a gold row means re-deriving qrels, which is mechanical and cheap.
- The labels are only as good as the gold set, which is `draft` — see ADR 0008.
  Complete qrels over unvalidated gold is *precise*, not *accurate*, and the
  reports say so.

## Reverses if

The corpus grows past roughly 10⁴ chunks, where labelling every chunk per query
stops being tractable and pooling becomes the honest choice.
