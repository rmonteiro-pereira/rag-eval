# Contributing

## Setup

```bash
docker compose up -d && docker compose ps    # all three must report (healthy)
uv sync --extra dev
uv run python -m ingest.pipeline --download 30
```

`docs/REPRODUCE.md` is the same sequence run from a destroyed stack, with real
output, if you want to know what it should look like.

## Before you open a PR

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -q                             # 281 passed
uv run python -m eval.regression_gate \
    --baseline eval/reports/ablation.json --candidate eval/reports/ablation.json
```

CI runs the first four on every push. The gate runs there too, **in both
directions** — passing on the committed report and failing on the degraded
fixture — because a gate that has only ever passed is not evidence of anything.

## The rules that are not style preferences

Most of what follows exists because this repo's whole claim is that its numbers
mean something. Breaking one of these does not make the code worse; it makes the
measurements worthless.

### Never mark a gold row `validated`

`eval/datasets/gold_seed.jsonl` holds 56 rows, all `status: "draft"`. Flipping one
to `validated` is a **human** act: it asserts that a person read the question
against the source PDF and confirmed it is unambiguous, correctly scoped, and not
answerable from three other atas. An agent-written, agent-graded gold set measures
nothing. `tests/test_gold.py` fails if any row is ever `validated` without that
pass. The same applies to the human columns in
`eval/datasets/judge_calibration_sheet.jsonl`.

### Never edit a report to match a claim

If a number in `docs/` disagrees with `eval/reports/`, the report wins — re-run
the harness and update the prose. Reports carry their full per-query audit trail
for exactly this reason. A number you cannot drill into is a number the reader has
to take on faith.

If you change something that *feeds* a measurement — a fixture, a canary, a
prompt — **re-measure**. When the injection canary was changed for publication,
the adversarial suite was re-run rather than the report hand-edited, and the fact
that every rate came back identical is itself the evidence.

### Keep the baseline arm pinned

`eval.run_eval` defaults to `--config dense` on purpose. That command produced the
committed baseline and has to keep producing it; if it silently upgraded to the
best arm, the "before" half of every comparison in this repo would move. Serving
(`rag.ask`, `serving/api.py`) defaults to the measured winner. Do not unify them.

### An ablation arm changes exactly one thing

Arms are contrasts, not a ladder. If you add one, hold everything else fixed and
add it to `CONTRASTS` in `eval/ablation.py` so its delta is attributable. Two of
the current seven arms are *worse* than arms below them, which is the finding — a
cumulative ladder would have hidden it.

### Nothing over 5 MB, no weights, no corpus

`git add` explicit paths only, never `git add -A`. `.gitignore` allowlists exactly
four report files by name; everything else under `eval/reports/` is ignored.

## Style

- Conventional-commit subject lines, and a body that explains **why**. If the
  change is a trade-off, name what you gave up.
- Type hints on public interfaces; `mypy` runs in CI.
- Comments explain the non-obvious decision, not the syntax. The regexes in
  `retrieval/text.py` and the quote-balance check in `agent/tools.py` are the
  house style: each says what breaks without it.
- No bare `except:`, and no swallowed failure. If an LLM call can fail mid-run,
  record the failure as a step rather than losing the run — see `agent/loop.py`.
- Tests cover the failure path. `pytest.raises(Exception)` is not a test; assert
  the specific error, and match its message where the message is the point.

## Reviews

Run [CodeRabbit](https://coderabbit.ai) before opening a PR:

```bash
cr review --uncommitted
```

Fold in what it flags, or say in the PR why you disagree.
