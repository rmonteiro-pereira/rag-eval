# Metrics

**Retrieval metrics land here (M2). Generation metrics are still empty on
purpose** — they arrive with the LLM backend in M3.

`retrieval.py` is deliberately pure: it takes a ranking and a qrels mapping and
returns numbers. No Qdrant, no embeddings, no network. That is what makes
`tests/test_retrieval_metrics.py` able to assert hand-computed values instead of
smoke-testing whatever the code happens to produce.

## What is measured, and against what

Relevance is judged over **complete qrels**: `eval/run_eval.py` scrolls every
chunk out of the collection and asks, per gold row, which chunks are relevant.
Recall with a denominator of "whatever came back" is not recall, and at 636
chunks there is no reason to approximate it.

Two grades (`eval/qrels.py`):

| gain | meaning |
|---|---|
| 2 | the chunk contains the gold `source_span` — the generator needs this one |
| 1 | same document, same printed page, but not the span itself |
| 0 | everything else |

`recall`, `hit_rate` and `MRR` binarise at gain ≥ 1. Only `nDCG` uses the
grades, so it is the only metric that rewards ranking the span chunk above its
page neighbours.

Matching is done on a normalised form — accents stripped, case folded, all
non-alphanumerics removed. `pypdf` extraction of these PDFs emits `eleva r`,
`0, 25` and `202 6`; without normalisation those spans would silently fail to
match and every metric would move for reasons that have nothing to do with
retrieval.

| metric | question it answers |
|---|---|
| `recall@k` | what fraction of *all* relevant chunks made the top k? |
| `hit_rate@k` | did *any* relevant chunk make the top k? |
| `MRR` | how far down is the first relevant chunk? |
| `nDCG@k` | is the ranking ordered well, span chunks above page chunks? |

`recall@k` and `hit_rate@k` are reported separately because they disagree and
the disagreement is informative. A gold span usually spans 2–4 chunks, so
`recall@1` is capped at ~1/3 by construction; `hit_rate@1` is not. Reporting
only one of them would either flatter or slander the system.

Aggregation is **macro** — every query weighs the same. The gold set
deliberately over-samples hard capabilities, and micro-averaging would let the
easy single-hop rows quietly dominate the headline number.

## Not here yet

- **Generation** — faithfulness/groundedness, answer relevance, context
  precision/recall. Needs a working LLM backend (M3).
- **End-to-end** — task success, citation correctness.
- **LLM-as-judge** — rubric plus calibration against ~30 human labels, reporting
  agreement rather than trusting the judge.
- **Abstention correctness** — the 7 negatives in the gold set are counted and
  excluded from retrieval scoring today; scoring the refusal itself is a
  generation metric and lands in M5 with the guardrails.
