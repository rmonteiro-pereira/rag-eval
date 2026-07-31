# 5. Keep the reranker even though it does not pay for itself

**Status:** Accepted, **with a stated expiry** · M4

## Context

The bge-reranker cross-encoder was added on the standard reasoning: rerankers
improve retrieval. Measured on this corpus, it mostly does not.

| contrast | ΔMRR | Δ rank-1 meeting | Δ p95 |
|---|--:|--:|--:|
| reranker, **without** the metadata filter | **−0.039** | **−0.146** | +2.5 s |
| reranker, **with** the metadata filter | +0.005 | 0.000 | +2.2 s |

Neither reranker contrast is distinguishable from zero: 95% CI [−0.164, +0.080]
(p = 0.54) without the filter and [−0.103, +0.115] (p = 0.93) with it, from
`eval/reports/significance.json`. So the honest reading is **"it does not help"**
rather than "it hurts"; this record said the latter before the intervals existed.
The mechanism below remains the plausible explanation for the negative point
estimate, not evidence for it.

Without the filter it reorders by semantic fit, and
semantic fit is exactly the signal that cannot tell two Copom meetings apart, so
it confidently promotes a beautifully-matching paragraph from the wrong ata.

## Decision

Keep it in the default config, and state in the README, the ablation and this
record that it does not pay for itself.

## Alternative rejected

**Cut it, and say the ablation proved rerankers are overrated.** Tempting — it is
the more striking claim, and the latency argument is strong. Rejected because it
would overstate what was measured. The reranker is the **only** component that
moves the reverse-lookup probe (0.375 → 0.500), and reverse lookup is the one
probe group the metadata filter structurally cannot help. Cutting it would close
the only avenue on the remaining headroom in order to make a cleaner story.

The second rejected option was **keeping it quietly**, letting the default config
imply it earned its place. That is the failure this repo exists to argue against.

## Consequences

- Default serving latency is ~2.5 s p95 instead of ~10 ms. For a research harness
  that is acceptable; for a product it would not be, and the README says so.
- The justification is explicitly a **research** one, not a serving one.
- Anyone under a latency budget should set
  `RETRIEVAL_CONFIG=hybrid+metadata` and lose 0.005 MRR.

## Reverses if

Either of these, and both are already measured every run:

1. **Reverse-lookup improves by other means** — a better tokenizer, numeric
   indexing, a query-rewriting step. The reranker's only remaining justification
   disappears and it should be cut.
2. **A latency arm is added and the p95 becomes a reported product metric.** See
   ADR 0009: the 2.2 s is a CPU-only number, and a GPU measurement could change
   the trade entirely without changing any accuracy figure.
