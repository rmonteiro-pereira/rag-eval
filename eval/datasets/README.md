This directory holds the two artefacts that gate the science of this project, and
both are waiting on the same person:

| file | what it is | human gate |
|---|---|---|
| `gold_seed.jsonl` | 56 Q/A pairs with source spans | flip `status` to `validated` |
| `judge_calibration_sheet.jsonl` | 30 judged answers | fill `human_faithfulness` / `human_answer_relevance` |

---

# Gold evaluation set

> **`gold_seed.jsonl` is a DRAFT — PENDING HUMAN VALIDATION.**
>
> It was drafted by an agent directly from the ingested BACEN Copom minutes. Every
> row is *plausible*, none is *validated*. Until a human has checked each question,
> each answer and each span against the source PDF, this file **must not be used to
> report any evaluation number**. A gold set an LLM wrote and an LLM grades measures
> nothing.

## Validation protocol (M2)

For each row, the reviewer must confirm:

1. **The span is real.** Open the source PDF at `source_page` and find `source_span`
   verbatim. PDF text extraction introduces spacing artifacts (`redu zir`, `202 6`)
   and drops accents; the span in this file is normalised, so match on substance.
2. **The answer follows from the span alone.** No outside knowledge, no inference
   the document does not license.
3. **The question is answerable and unambiguous.** If two documents could both
   answer it, either scope the question by date or drop it.
4. **`answer_type` is right.** `extractive` = the answer is a quote; `abstractive` =
   the answer synthesises; `abstention` = the corpus cannot answer and refusal is
   the correct behaviour.

Flip `"status": "draft"` to `"status": "validated"` only after all four hold. Rows
that fail get fixed or deleted — never silently kept.

## Schema

| field | meaning |
|---|---|
| `id` | stable identifier; never reused |
| `status` | `draft` \| `validated` \| `rejected` |
| `question` | the question, in Portuguese |
| `answer` | reference answer |
| `answer_type` | `extractive` \| `abstractive` \| `abstention` |
| `source_doc_id` | `doc_id` in `data/manifest.json`; `null` for abstention rows |
| `source_title` | human-readable document title |
| `source_page` | 1-indexed printed page holding the span |
| `source_span` | the text that answers the question (`\|` separates multi-hop spans) |
| `difficulty` | `easy` \| `medium` \| `hard` |
| `capability` | what the row is meant to probe |
| `notes` | reviewer guidance / known traps |

The first line of the `.jsonl` is a `_comment` record, not a test case. Loaders must
skip any object carrying a `_comment` key.

## Coverage (56 rows)

Rows 001–010 are the original M1 seed; 011–056 were added in M2. All 56 are
`draft`.

| capability family | rows | n |
|---|---|---|
| single-hop lookup (Selic decision, scoped by meeting date) | 001–003, 008, 011–022 | 16 |
| numeric extraction — Focus expectations | 004, 023–028 | 7 |
| numeric extraction — Copom's own projections | 006, 029–033 | 6 |
| reference-scenario assumptions (FX) | 034–037 | 4 |
| list extraction / votes | 005, 038–041 | 5 |
| reverse lookup (value → meeting) | 009, 042–048 | 8 |
| multi-hop within one document | 007, 049, 050 | 3 |
| out-of-scope abstention (negatives) | 010, 051–056 | 7 |

24 of the 30 ingested documents are cited by at least one row.

### How the M2 rows were derived

Every span was extracted from the **actually ingested text** (the chunks in
Qdrant, not the PDFs and not the model's memory) and then re-verified
programmatically before being written: each `source_span` must be locatable in a
chunk of its `source_doc_id`, on its `source_page`, under the normalisation
described in `../metrics/README.md`; each `source_doc_id` must exist in
`data/manifest.json`; each `source_title` must match the manifest verbatim. Two
rows failed the title check on the first pass and were corrected.

That makes the rows *grounded*. It does not make them *validated* — a span can
be real and the question built on it still be ambiguous, mis-scoped or
answerable from three other documents. That judgement is the human pass, and
the four checks above remain the protocol.

### Traps built in on purpose

The set is not a uniform sample; it is weighted towards the failure mode the M1
baseline is known to have (right topic, wrong meeting):

- **Level collisions.** 13,25%, 11,25% and 10,75% are each reached twice — once
  by a cut, once by a hike. Only the verb separates them (`gold-042`, `gold-045`,
  `gold-048`).
- **Near-duplicate paragraphs.** `gold-028` asks the `gold-004` question one
  meeting earlier; the sentence stem is identical and only the numbers differ.
- **Adjacent opposite answers.** `gold-038` and `gold-039` are the two halves of
  one 5–4 split vote, same page, same sentence stem.
- **PDF extraction noise.** `gold-014`, `gold-021`, `gold-033` and `gold-048`
  sit on spans the extractor broke (`deci di u`, `eleva r`, `11, 25%`).
- **In-corpus topic, out-of-corpus fact.** `gold-055` asks for an inflation
  target year the atas never state, on a topic they discuss constantly.

### Negatives

Seven rows are `answer_type: "abstention"`: the corpus cannot answer them and
refusal is the correct behaviour. They carry no span, so `eval/run_eval.py`
excludes them from retrieval metrics and counts them separately. Scoring the
refusal itself is a generation metric and arrives in M5.

M2's target was 50–100 rows; the file is at 56. Growing it further is cheap —
what is expensive, and what actually gates the science, is the human validation
pass.

---

# Judge calibration sheet (M3)

`judge_calibration_sheet.jsonl` — **30 judged answers, human column empty.**

The LLM judge in `generation/judge.py` scores every generated answer for
faithfulness and answer relevance on a 0–2 rubric. It is a local 3B/8B model
grading answers written by a local 3B/8B model. Nothing about that arrangement
produces a trustworthy number, and reporting its scores as *the* generation
result would be measuring one unvalidated model with another.

So `eval/reports/generation.json` reports **`agreement: null`** — meaning
*unknown*, not *good* — and this file is how that changes.

## Protocol

For each row, read `question`, `answer` and the `retrieved_doc_ids`, then fill:

| field | scale |
|---|---|
| `human_faithfulness` | `0` unsupported claim / contradicts the passages · `1` mostly supported · `2` every claim appears in the passages |
| `human_answer_relevance` | `0` does not answer the question asked · `1` partial or padded · `2` direct and complete |
| `human_notes` | free text, optional |

The rubric the judge was given is in `generation/judge.py` — use the same one, or
the disagreement measures two different questions rather than judge error.

Two rules:

- **Do not edit the `judge_*` fields.** They are the prediction under test.
- **Faithfulness is about support, not truth.** An answer that is factually
  correct but not present in the retrieved passages scores `0`, because a RAG
  system answering from parametric memory is broken even when it is right.

Then:

```bash
uv run python -m eval.calibration
```

which reports Cohen's kappa, raw agreement and the confusion matrix per criterion.

## Why these 30 rows

Not a random sample. A random sample over a system that mostly works would be
~28 easy agreements and two interesting rows. The selection is deterministic and
stratified, highest-information first:

1. **Judge/arithmetic conflicts** — the judge called an answer faithful while it
   asserts a number found nowhere in the context (or the reverse). These are rows
   where the judge is provably one thing or the other.
2. **Negatives** — where refusal is correct and the rubric is most awkward.
3. **Judge parse failures and low scores.**
4. **Spread across arms**, round-robin, so no single model owns the sheet.

## Kappa, not raw agreement

On a 3-point scale where most answers genuinely deserve a 2, a judge that has
learned to answer "2" and nothing else scores **100% raw agreement and kappa 0**.
Both are reported, and the gap between them is itself the finding.

## Re-running is safe

`eval.run_generation` rewrites this file every run, merging by `gold_id` + `arm`
so any human labels already entered survive. There is a test for it
(`tests/test_calibration.py::test_rewriting_the_sheet_preserves_human_labels`),
because silently wiping an afternoon of labelling is the one bug this file cannot
survive.
