# Architecture decision records

One file per decision that would be expensive to reverse or that a reader would
otherwise assume was an accident.

Each record states **the alternative that was rejected** and **the condition that
would reverse the decision**. A record without those two is a description, not a
decision — anyone can see what the code does; what they cannot see is what it
almost did instead, and what would change someone's mind.

| # | Decision | Status |
|---|---|---|
| [0001](0001-evaluation-harness-before-features.md) | An evaluation harness before features | Accepted |
| [0002](0002-complete-qrels-over-sampled-judgements.md) | Complete qrels with graded gains, not sampled judgements | Accepted |
| [0003](0003-ablation-as-contrasts-not-a-ladder.md) | Ablation arms as controlled contrasts, not a cumulative ladder | Accepted |
| [0004](0004-metadata-filter-as-a-hard-prefilter.md) | The meeting filter is a hard pre-filter, not a soft boost | Accepted |
| [0005](0005-keep-the-reranker-despite-losing-on-latency.md) | Keep the reranker even though it does not pay for itself | Accepted, with a stated expiry |
| [0006](0006-acl-as-a-qdrant-payload-filter.md) | The ACL is a query pre-filter, never a post-filter | Accepted |
| [0007](0007-gate-on-probe-metrics-not-only-aggregates.md) | The regression gate gates probes as well as aggregates | Accepted |
| [0008](0008-gold-set-stays-draft-until-a-human-validates-it.md) | The gold set stays `draft` until a human validates it | Accepted |
| [0009](0009-cpu-torch-over-cuda-wheels.md) | Pin CPU torch, not CUDA wheels | Accepted, revisit if a latency arm is added |

## Format

Short. Context, the decision, the alternative rejected and why, the consequences
including the bad ones, and the condition that reverses it.
