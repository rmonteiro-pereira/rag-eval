# 8. The gold set stays `draft` until a human validates it

**Status:** Accepted · M2, and permanently open

## Context

`eval/datasets/gold_seed.jsonl` holds 56 question/answer pairs with cited spans,
written by an agent from the ingested corpus. Every span was verified
programmatically: it must be locatable in a chunk of the document and page it
cites, the `source_doc_id` must exist in `data/manifest.json`, and the title must
match the manifest verbatim.

So the rows are **grounded**. The question is whether that makes them **valid**.

## Decision

Every row carries `"status": "draft"`. `--min-status validated` returns nothing
and exits 3. No agent may set `validated`; `tests/test_gold.py` fails if one ever
does. The same applies to the human columns in the judge calibration sheet.

## Alternative rejected

**Mark them `validated` once the programmatic checks pass**, and report clean
numbers. This is the option that makes the repo look finished, and it is the one
worth arguing about, because the checks are not weak — they verify the span
really exists where it says it does.

Rejected because grounded and valid are different properties, and only the second
one licenses the numbers. A span can be real and the question built on it still
be:

- **ambiguous** — "the projection for 2027" when the ata gives three scenarios;
- **mis-scoped** — answerable only with context the retriever was never given;
- **non-discriminating** — answerable identically from three other atas, which on
  *this* corpus is the default failure and precisely what the project measures.

None of those is detectable by checking that a string appears on a page. An
agent-written, agent-graded gold set measures the agent's consistency with itself.

## Consequences

- **Every number in this repository carries the draft caveat**, printed by the
  harness itself rather than added by the README. The relative movement between
  arms on a fixed question set is the defensible reading; absolute levels are not.
- The repo ships with a visible unfinished edge. That is the intended reading:
  the harness is done, the science is not.
- Validation is a **re-run, not a rewrite** — the same command with
  `--min-status validated` produces the number that counts, so the pipeline is
  already built and waiting.
- A second gate exists for the same reason: two local judges agree on
  faithfulness 44% of the time (κ = 0.109), so `agreement` is reported as
  **`null` — unknown, not good**.

## Reverses if

Nothing. This is the one decision in the repo that compute cannot close. It
reverses when a human reads 56 rows against 30 PDFs.
