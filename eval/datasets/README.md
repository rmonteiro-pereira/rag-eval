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

## Coverage of this seed (10 rows)

| capability | rows |
|---|---|
| single-hop lookup | 001, 002, 003, 008 |
| numeric / list extraction | 004, 005 |
| scenario-qualified lookup | 006 |
| multi-hop within one document | 007 |
| reverse lookup (value → meeting) | 009 |
| out-of-scope abstention | 010 |

M2 expands this to 50–100 rows. The distribution above is the target shape: mostly
answerable, with a deliberate minority of negatives so abstention is measurable.
