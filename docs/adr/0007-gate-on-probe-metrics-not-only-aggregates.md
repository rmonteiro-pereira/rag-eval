# 7. The regression gate gates probes as well as aggregates

**Status:** Accepted · M6

## Context

A regression gate compares a fresh evaluation report against a committed one and
fails the build on a drop beyond tolerance. The obvious thing to gate is the
headline aggregates: MRR, nDCG, hit_rate.

## Decision

Gate the aggregates **and** the probe metrics — `probe:meeting_disambiguation`
and `probe:reverse_lookup` — with their own tolerances.

## Alternative rejected

**Aggregates only.** Rejected on evidence from the degraded fixture. Simulate the
meeting resolver breaking and watch what each metric does:

| metric | baseline | degraded | Δ |
|---|--:|--:|--:|
| mrr | 0.7412 | 0.4076 | −0.334 |
| ndcg@10 | 0.6232 | 0.3428 | −0.280 |
| **probe:meeting_disambiguation** | **1.0000** | **0.1460** | **−0.854** |

The aggregates do fall. They fall **into a range that still looks like a working
retrieval system** — 0.41 MRR is a plausible number for a RAG demo, and on a
dashboard it reads as noise or a bad week. The probe collapses to near zero and
is impossible to misread.

Gate on the metric specific to your known defect, not only on the one that is
easy to average.

## Consequences

- `tests/test_regression_gate.py` asserts **both** directions — exit 0 on the
  committed report, exit 1 on the degraded fixture — plus a case where **only**
  the probe degrades and every aggregate is untouched. That third test is the
  point: a dashboard would show nothing wrong.
- `tests/fixtures/gate_degraded.json` carries a `_fixture_note` reading
  `DELIBERATELY DEGRADED`, so it can never be mistaken for a measurement.
- **Exit 2 is reserved for "could not compare"** — missing file, unknown arm,
  absent metric. A missing metric raises rather than being skipped, because a
  gate that passes because it could not find the numbers is worse than no gate.
- Tolerances are per-metric and hand-set. Both probes have one, and they differ by
  an order of magnitude for a reason:

  | metric | tolerance | why |
  |---|--:|---|
  | `recall@1`, `recall@5`, `ndcg@5`, `ndcg@10`, `mrr` | 0.02 | aggregate noise floor on n=49 |
  | `hit_rate@1`, `hit_rate@5` | 0.03 | coarser — one query is 1/49 ≈ 0.020 |
  | **`probe:meeting_disambiguation`** | **0.02** | it sits at **1.000**. There is no noise above a ceiling: any real drop means the meeting resolver broke, so the tolerance only absorbs a single query flipping (1/41 ≈ 0.024 — deliberately *just* under, so even one regression fires) |
  | **`probe:reverse_lookup`** | **0.13** | n=8, so one query is 0.125. A tighter bound would fire on noise, and a gate that cries wolf gets disabled |

  The asymmetry is the decision: a ceiling metric and a small-n metric need
  opposite treatment, and averaging them into one global tolerance would make one
  of them useless.

## Reverses if

The probes stop tracking a real defect — if reverse lookup were solved and
meeting disambiguation had been at 1.000 for a long time across many changes,
they would become ceiling metrics that only ever fire on catastrophes the
aggregates would also catch.
