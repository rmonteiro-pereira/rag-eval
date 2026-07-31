# Contributing

This repository is an **evaluation harness** first and a RAG pipeline second. The
most valuable contributions are ones that make a number more trustworthy, or show
that an existing number is wrong.

## Setup

Requires **Python 3.11 or 3.12** (`>=3.11,<3.13`) and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rmonteiro-pereira/rag-eval.git
cd rag-eval
uv sync --extra dev
```

For anything that touches retrieval you also need the stack and the corpus:

```bash
docker compose up -d && docker compose ps    # all three must report (healthy)
uv run python -m ingest.pipeline --download 30
```

Integration tests need the Qdrant container; they are marked `integration` and
skip themselves when it is not running, so a first-time contributor is not
blocked by it.

[`docs/REPRODUCE.md`](docs/REPRODUCE.md) is this same sequence run from a
destroyed stack, with real output, if you want to know what it should look like.

## Before you open a PR

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
uv run python -m eval.regression_gate \
    --baseline eval/reports/ablation.json --candidate eval/reports/ablation.json
```

Mutation testing is not in that list because it needs Linux — mutmut 3.x has no
native Windows support. **CI runs it on every push** (~30s), gates on the score
and fails if `docs/mutation-survivors.md` is stale. On Linux, run it yourself
before pushing changes to ranking or gate logic:

```bash
uv run mutmut run && uv run mutmut results --all true
```

CI runs the first four on every push. The gate runs there too, **in both
directions** — passing on the committed report and failing on the degraded
fixture — because a gate that has only ever passed is not evidence of anything.

Expect CI to report three fewer passes than you see locally: the `integration`
tests need a live Qdrant and are deselected there. No test count is written down
in this repository's docs, deliberately — a hard-coded count is wrong the first
time anyone adds a test. `uv run pytest -q` is the source of truth.

## The rules that make the results mean anything

Hard constraints, not style preferences. Breaking one of these does not make the
code worse; it makes the measurements worthless.

1. **Never promote a gold-set row to `validated`.** Validation is a human act: it
   asserts that a person read the question against the source PDF and confirmed it
   is unambiguous, correctly scoped, and not answerable from three other *atas*
   (an **ata** is the published minutes of one Copom meeting — the corpus is 30 of
   them, and their near-identical wording is the failure mode this repo measures).
   An
   agent-written, agent-graded gold set measures nothing. `--min-status validated`
   returning nothing today is intentional, not a bug to be fixed by relabelling,
   and `tests/test_gold.py` fails if a row is ever flipped. The same applies to
   the human columns in `eval/datasets/judge_calibration_sheet.jsonl`.
2. **Every published number must be reproducible from a committed artifact.** If a
   number in `docs/` disagrees with `eval/reports/`, the report wins — re-run the
   harness and update the prose, never the other way round. If you change anything
   that *feeds* a measurement (a fixture, a canary, a prompt), regenerate the
   report in the same PR and say which file it came from.
3. **Report a bad result rather than tuning until it looks good.** A measured
   regression is a finding; an unexplained improvement is a bug until it is
   explained.
4. **Ablation arms differ by one component at a time.** A cumulative ladder hides
   which change did the work — two of the current seven arms are *worse* than arms
   below them, which is the finding. Add the contrast pair to `CONTRASTS` in
   `eval/ablation.py`, not just a row to the table. See
   [ADR 0003](docs/adr/0003-ablation-as-contrasts-not-a-ladder.md).
5. **Guardrail changes need the ungoverned control arm re-run**, otherwise the
   attack-success numbers are not comparable.
6. **Keep the baseline arm pinned.** `eval.run_eval` defaults to `--config dense`
   on purpose: that command produced the committed baseline and has to keep
   producing it. Serving defaults to the measured winner. Do not unify them.

## Adding an evaluation arm

An arm is only useful if it is comparable. State the retrieval configuration, keep
the question set fixed, and record latency alongside quality — a win that costs
seconds of p95 is a different result from a win that is free.

## Decisions

Non-obvious choices live in [`docs/adr/`](docs/adr/), each with the alternative
that was rejected and the condition that would reverse it. If your change
contradicts one, update the ADR in the same PR rather than leaving the record
wrong — including the "reverses if" clause, which is the part that made the
decision falsifiable.

## Style

- Conventional Commits — `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `eval:` —
  with a body that explains **why** and states what you measured. If the change is
  a trade-off, name what you gave up.
- Type hints on public interfaces; `mypy` runs in CI.
- Comments explain the non-obvious decision, not the syntax. The regexes in
  `retrieval/text.py` and the quote-balance check in `agent/tools.py` are the
  house style: each says what breaks without it.
- No bare `except:`, and no swallowed failure. If an LLM call can fail mid-run,
  record the failure as a step rather than losing the run — see `agent/loop.py`.
- Tests cover the failure path. `pytest.raises(Exception)` is not a test; assert
  the specific error, and match its message where the message is the point.

## Dependency PRs

Dependabot opens these monthly, grouped and capped — see `.github/dependabot.yml`
for what is excluded and why. Two rules, both learned the hard way in a sibling
project:

- **"CLEAN" means no merge conflict. It does not mean it works.** A green tick
  and a mergeable state answer a different question from "does this break
  anything". Read the changelog on any major bump.
- **An action bump is a prompt to re-check that action's inputs**, not just its
  version. The sibling lane found `actions/upload-artifact` needed
  `include-hidden-files: true` — a setting that had *already* been wrong at v4,
  which the bump surfaced rather than caused.

Closing a bump with the reason recorded is a legitimate outcome. Never
auto-merge, and never merge on green alone: CI here does not exercise every
path, and the `eval-full` job has never executed at all.

## Pull requests

- Branch from `main`; never commit to it directly, never force-push, never
  rewrite history.
- Run [CodeRabbit](https://coderabbit.ai) first — `cr review --uncommitted` — and
  fold in what it flags, or say in the PR why you disagree.

## What not to commit

Corpus documents (only the manifest is versioned), model weights, vector-store
data, API keys, anything over 5 MB. `git add` explicit paths only, never
`git add -A`: `.gitignore` allowlists exactly four report files by name and
everything else under `eval/reports/` is ignored.
