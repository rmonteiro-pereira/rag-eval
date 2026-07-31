# Contributing

This repository is an **evaluation harness** first and a RAG pipeline second. The most
valuable contributions are ones that make a number more trustworthy, or show that an existing
number is wrong.

## Setup

Requires **Python 3.11 or 3.12** (`>=3.11,<3.13`) and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rmonteiro-pereira/rag-eval.git
cd rag-eval
uv sync --extra dev
```

Integration tests need the Qdrant container from `docker-compose.yml`; they are skipped when
it is not running, so a first-time contributor is not blocked by it.

## Tests and lint

```bash
uv run pytest -q
uv run ruff check .
```

## The rules that make the results mean anything

Please treat these as hard constraints — they are the point of the project:

1. **Never promote a gold-set row to `validated`.** Validation is a human act. `--min-status
   validated` returning nothing today is intentional, not a bug to be fixed by relabelling.
2. **Every published number must be reproducible from a committed artifact.** If you change a
   metric, regenerate the report in the same PR and say which file it came from.
3. **Report a bad result rather than tuning until it looks good.** A measured regression is a
   finding; an unexplained improvement is a bug until it is explained.
4. **Ablation arms differ by one component at a time.** A cumulative ladder hides which change
   did the work — that is why the existing ablation is laid out in pairs.
5. **Guardrail changes need the ungoverned control arm re-run**, otherwise the attack-success
   numbers are not comparable.

## Adding an evaluation arm

An arm is only useful if it is comparable. State the retrieval configuration, keep the
question set fixed, and record latency alongside quality — a win that costs seconds of p95 is
a different result from a win that is free.

## Pull requests

- Branch from `main`; never commit to it directly.
- [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `docs:`,
  `chore:`, `test:`, `eval:`.
- Explain **why**, and state what you measured.
- `pytest -q` and `ruff check .` green before opening.

## What not to commit

Corpus documents (only the manifest is versioned), model weights, vector-store data, API keys,
anything over 5 MB.
