# Mutation testing

380 passing tests says the suite runs. It does not say the suite would notice if
the code were wrong. Mutation testing answers that second question: change the
code in a small, plausible way, and see whether any test fails.

```bash
uv run mutmut run          # WSL/Linux only — mutmut 3.x has no native Windows support
uv run mutmut results --all true
uv run mutmut browse       # interactive
```

## The score

**73.4%** — 342 of 466 mutants with a covering test were killed.

```
566 mutants generated
    342  killed      a test failed, as it should have
    124  survived    the code changed and every test still passed
    100  no tests    no test in scope even imports the mutated line
```

Two denominators, because the choice flatters or damns and neither is dishonest
on its own:

| | |
|---|--:|
| killed / (killed + survived) — the conventional mutation score | **73.4%** |
| killed / all mutants, counting uncovered code as unkilled | **60.4%** |

The second is the one to worry about. 100 mutants live in code no in-scope test
imports at all, and that is a coverage fact rather than a test-quality one.

## Per module

| module | killed | survived | no test | score |
|---|--:|--:|--:|--:|
| `retrieval/text.py` | 35 | 0 | 0 | **100.0%** |
| `retrieval/metadata.py` | 37 | 0 | 10 | **100.0%** |
| `retrieval/fusion.py` | 57 | 2 | 0 | **96.6%** |
| `eval/regression_gate.py` | 119 | 67 | 30 | 64.0% |
| `eval/probes.py` | 94 | 55 | 0 | 63.1% |
| `eval/scoring.py` | 0 | 0 | 60 | **n/a — nothing covered** |

The shape is worth more than the headline. **The ranking arithmetic is at
96.6–100%; the reporting layer is at ~63%.** That is the right way round — the
tokenizer, the meeting resolver and the fusion are where a wrong answer looks
like a plausible number rather than a crash — but the reporting layer is not
where a reader should assume rigour.

## What it found, and what changed

The first run scored **71.2%**. Five tests were written against named survivors;
the score is now 73.4% and `retrieval/` went from 81–97% to 96.6–100%. Only two
mutants improved by luck rather than by a test; every other kill is a test that
did not exist before.

Four real defects in the test suite, in descending order of how much they matter:

1. **The RRF tie-break was not pinned.** `sorted(..., key=(-score, key))` mutated
   to `(-score, score)` and survived: the existing test accepted *either* of two
   tied documents. This repository's headline reproducibility claim is that the
   whole ablation regenerates to ±0.0000 from an empty index — which holds only
   if equal scores resolve identically every run. The suite was not checking the
   property the claim rests on.

   Worth recording: **the first test written for this also failed to kill it.**
   Fed the two arms in `a, b` order, Python's stable sort preserves insertion
   order and the broken tie-break returns the correct answer by accident. Only
   feeding `b` first — so insertion order and key order disagree — makes the
   tie-break observable. A test that passes for the wrong reason is exactly what
   mutation testing is for.

2. **`top_k` was never exercised.** `ordered[: top_k or len(ordered)]` mutated to
   `and` and survived, because no test passed a `top_k` at all — despite every
   caller in the repo passing one. With the mutation, `top_k=5` returns
   everything.

3. **Fusion could have dropped every document's title, URL and text.** Eight
   mutants replaced fields in the rebuilt `Retrieved` with `None` and survived:
   the tests asserted ordering and nothing else. Those fields are what the answer
   cites to the reader.

4. **`normalise` was only ever checked by membership.** Injecting literal garbage
   around the decimal repair (`\1\2\3` → `XX\1\2\3XX`) survived, because
   `"13,25" in tokens` stays true when the tokenizer splits the garbage into
   separate tokens. Asserting the exact returned string kills it.

## The survivors that remain

### Equivalent mutants — 2, and they are provably so

`retrieval/fusion.py` mutants 22 and 40 change the placeholder `score=0.0` in the
freshly constructed `Retrieved` to `None` and `1.0`. Both survive because
`record.score = score` at the end of the function overwrites it unconditionally
for every record that is returned. The value is not observable, so **no test can
kill these and none should be written to try.** Counting them as failures would
be measuring the tool, not the suite.

### `eval/scoring.py` — 60 mutants, zero covered

`score_rows` computes every retrieval metric this project publishes, and **no
test imports it directly.** It is exercised end to end, through
`eval/run_eval.py` and `eval/ablation.py`, which is why the committed numbers are
trustworthy — but it means a change to the metric arithmetic would be caught only
by a full re-run against a live Qdrant, not by `pytest`.

This is the single largest gap the exercise found and it is not fixed here.
Naming it is worth more than a rushed test that raises the number.

### `eval/regression_gate.py` — 67 survived, 30 uncovered

Two distinct causes, and they deserve different treatment:

- **`main()` — 56 survivors.** Argument parsing, exit-code plumbing and help
  text. Tests call `main([...])` and assert the exit code, so mutations to
  `--help` strings and argument metavars survive. Low value.
- **`render()` — 30 uncovered.** The human-readable table nobody asserts on.
- **`extract()` / `compare()` — 11 survivors, and these matter.**
  `entry.get("probes", {})` → `entry.get("probes", None)` survives because every
  fixture has a `probes` block. That is precisely the "could not compare" path
  the gate reserves **exit 2** for, and the fixtures never exercise it. A gate
  that claims to fail loudly on a malformed report should have a test with a
  malformed report.

### `eval/probes.py` — 55 survived

Almost all are report-payload mutations: JSON key names (`"expected_doc_id"` →
`"XXexpected_doc_idXX"`), and prose inside explanatory `note` strings. The metric
values are asserted; the **schema** is not. That is a real if unglamorous gap,
because `regression_gate.extract()` reads `probes[group]["rank1_doc_accuracy"]`
by name — a silent key rename would break the gate, and nothing round-trips a
probe report through the gate to catch it.

## Scope, and why it is what it is

Mutated: `retrieval/{text,sparse,fusion,metadata}.py`, `eval/{scoring,probes,regression_gate}.py`.

**Not mutated: `retrieval/{store,configs,rerank}.py`.** They need live Qdrant and
~2.5 GB of model weights, so mutants there would measure what the runner lacks
rather than what the tests miss. That is a stated exclusion, not a quiet one —
those modules are covered by the `integration`-marked tests and by the
clean-state reproduction in [`REPRODUCE.md`](REPRODUCE.md), neither of which runs
here.

Test selection is likewise scoped to the seven files that import the mutated
modules. Running all 380 tests against 566 mutants would cost hours to prove that
a test which never imports `fusion.py` cannot catch a mutation in it.

## The configuration is checked even though the run is not

A mutation setup can be present in a repository and absent from every run — a
config naming directories that do not exist, or a job pinned to `if: false`.
Both produce a green tick over nothing. Since the run itself is deliberately out
of CI (below), `tests/test_mutation_config.py` checks the *config* in the suite
that does run:

- every path in `source_paths`, `also_copy` and the test selection exists;
- none of those keys is empty;
- **every mutated module is imported by at least one selected test** — paths can
  all be valid and the score still meaningless if the selection is disjoint,
  because then every mutant is scored "no tests" over an empty denominator;
- `mutmut` is a declared dev dependency;
- this document states its exclusions and both denominators, so the score cannot
  quietly be narrowed into looking better.

Each guard was verified to fail on the defect it targets, not merely to pass:
pointing `source_paths` at a nonexistent file, pointing the test selection at an
unrelated file, and emptying `source_paths` each turn the suite red.

Import linkage is a floor, not a guarantee, and this repo shows the difference:
`eval/scoring.py` **is** imported by a selected test and still has 60 uncovered
mutants, because importing a module does not call `score_rows`. That is why the
table above carries a per-module `no test` column instead of one headline number.

## Why the run is not in CI

A full run is ~14 minutes on 10 cores, and mutmut 3.x does not run on Windows,
which is the development machine. It belongs in the same category as the full
ablation: run deliberately, report the number, do not put it on the PR path where
its cost would get it disabled. The number above is from
`mutmut run` at the commit that introduced this document, and it is stale the
moment the code moves — which is the honest status of every mutation score
anyone publishes.
