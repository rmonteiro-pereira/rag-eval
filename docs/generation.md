# M3 — Generation, and why the judge is not believed

Regenerate with:

```bash
uv run python -m eval.run_generation --out eval/reports/generation.json
uv run python -m eval.calibration --second-judge qwen2.5:3b     # judge reliability
```

Three backends answer all 56 gold questions (49 answerable + 7 negatives).
**Retrieval runs once per question and the passages are shared by every arm**, so
a difference between arms is a generation difference and nothing else. The
retriever is the M4 winner, so these are generation numbers *given good
retrieval*.

> Same caveat as everywhere else: the gold set is 56 `draft` rows, human-validated
> count zero. And the judge column below is worse than provisional — see
> [§ The judge](#the-judge-and-why-its-numbers-are-not-the-result).

---

## Deterministic results

Nothing in this table involves a model grading a model.

| arm | numeric recall | groundedness | hallucinated numbers | citation ok | abstention ok | false refusal | median ms |
|---|--:|--:|--:|--:|--:|--:|--:|
| `extractive` | **0.913** | **0.988** | **0.000** | 1.000 | **0.000** | 0.000 | 0 |
| `qwen2.5:3b` | 0.777 | 0.838 | **0.000** | 1.000 | **1.000** | 0.082 | 3010 |
| `llama3.1` | 0.837 | 0.887 | **0.000** | 1.000 | **1.000** | 0.041 | 3445 |

- **numeric recall** — of the numbers in the reference answer, how many appear in
  the generated one. On a corpus about policy rates this is close to a
  correctness oracle: `14,25` is either there or it is not.
- **hallucinated numbers** — numbers asserted by the answer that appear in neither
  the retrieved passages nor the question.
- **groundedness** — share of the answer's content words present in the context.
- **abstention ok / false refusal** — refusing is *correct* on the 7 negatives and
  a *failure* on the 49 answerable rows. Scored separately, because one mean over
  both would rise for two opposite reasons.

### 1. Zero hallucinated numbers, every arm, 168 answers

Not a single number was asserted that did not appear in the retrieved evidence or
the question. With a strong retriever and a prompt that demands citations, a 3B
model did not invent a policy rate once.

This is a narrow claim and it is worth stating narrowly: it catches *fabricated
numbers*, which is the failure mode with consequences in this domain. It does not
catch a fluent misreading of a number that *is* present — attributing June's rate
to March. That is what the retrieval probes and the human gate are for.

### 2. The safest-looking mode is the one that cannot say "I don't know"

`extractive` tops numeric recall (0.913) and groundedness (0.988) — unsurprising,
since it returns the passages verbatim and a verbatim quote cannot hallucinate. It
is the floor the generative arms have to justify themselves against, and on
grounding they do not beat it.

But its **abstention correctness is 0.000**. On all 7 out-of-scope questions —
*"Qual foi a decisao do Federal Reserve...?"*, *"Qual e a remuneracao mensal dos
membros do Copom?"* — it returned five passages of Copom minutes as though they
were an answer. It has no mechanism to refuse, because it has no mechanism to say
anything other than what it retrieved.

That is a governance finding, not a trivia point. The mode with the best
groundedness metrics is the one that **cannot abstain and cannot be made to**, and
abstention is what the M5 guardrail suite depends on. A metric dashboard that
ranked backends on groundedness alone would have picked exactly the wrong one.

### 3. Abstention is not free

Both generative arms abstain correctly on all 7 negatives — and both pay for it in
false refusals on answerable questions:

| arm | false refusals | which |
|---|--:|---|
| `qwen2.5:3b` | 4/49 (8.2%) | gold-015, gold-034, gold-037, gold-044 |
| `llama3.1` | 2/49 (4.1%) | gold-037, gold-050 |

`gold-037` defeats both. The trade is real and directional: the same instruction
that produces correct refusals produces incorrect ones, and a system tuned to
never refuse would score 1.000 on false refusal and 0.000 on abstention.

`llama3.1` beats `qwen2.5:3b` on every deterministic metric, at ~12% more latency.

---

## The judge, and why its numbers are not the result

| arm | judge faithfulness (0–2) | judge answer relevance | faith=2 rate | parse failures |
|---|--:|--:|--:|--:|
| `extractive` | 1.446 | 1.554 | 0.643 | 0.000 |
| `qwen2.5:3b` | 1.750 | 1.554 | 0.839 | 0.000 |
| `llama3.1` | **1.804** | **1.643** | **0.857** | 0.000 |

Read that table and the obvious conclusion is "llama3.1 writes the best answers".
Two things make that conclusion unsafe, and both are measured rather than
speculated about.

### The judge graded its own homework

The judge **is** `llama3.1`, and it wrote every answer in the `llama3.1` arm. All
56 rows of that arm are flagged `judge_is_generator: true`, and the report carries
`setup.judge_self_preference_warning`. The arm the judge wrote scored highest.
Self-preference and quality cannot be separated by this design; that is a known
bias direction for LLM judges, and this run walked straight into it.

### Two local judges barely agree with each other

`eval.calibration --second-judge qwen2.5:3b` re-scores the 30 calibration rows
with an independent judge. No human labels required:

| criterion | Cohen's κ | raw agreement | n |
|---|--:|--:|--:|
| **faithfulness** | **0.109** | **0.440** | 25 |
| answer relevance | 0.589 | 0.750 | 28 |

**On faithfulness, two local judges disagree with each other more than half the
time**, at a kappa of 0.109 — "slight" agreement on the Landis–Koch scale, barely
distinguishable from chance. Answer relevance fares better at 0.487 ("moderate"),
which fits: "does this address the question" is a shallower judgement than "is
every claim supported by this evidence".

Two raters that disagree with each other cannot both be right, so **the
faithfulness column above carries almost no information**. That is the finding.
The deterministic metrics are the ones to believe, and they are why this project
computes them first and treats the judge as the flexible, suspect instrument.

Judge-judge agreement is a *necessary* condition, not a sufficient one. Two models
with overlapping training data can agree and both be wrong, which is why model
consensus is not ground truth and the human column stays.

### What would make the judge believable

`eval/datasets/judge_calibration_sheet.jsonl` — **30 items, `human_faithfulness`
and `human_answer_relevance` empty.** Balanced 10/10/10 across
extractive/abstractive/abstention answer types and roughly evenly across the three
arms, stratified to over-select the rows that discriminate. Each row carries the
**retrieved passages verbatim**, because "is every claim supported by the
evidence" is not answerable from a list of document ids.

Until it is filled, `eval/reports/generation.json` reports
`agreement.criteria.*.judge_vs_human` as **unknown — not good**. Filling it and
running `uv run python -m eval.calibration` produces Cohen's kappa against human
judgement, which is the number that would license reporting the judge at all.

Protocol: `eval/datasets/README.md`.

---

## Honest limits

- **Draft gold set**, 56 rows, zero validated. Everything above inherits that.
- **The judge is uncalibrated against humans**, and against another model it is
  near-chance on faithfulness. Judge numbers are reported because hiding them
  would be worse, not because they are trusted.
- **One judge run, temperature 0, no self-consistency.** Sampling the judge k
  times and taking the majority is a known variance reduction and was not done.
- **n = 49 answerable rows, n = 7 negatives.** One negative is 14 percentage
  points of abstention correctness. The abstention numbers are directionally
  meaningful and no more.
- **Generation is measured only on top of the best retriever.** How these
  backends behave given the M1 baseline's retrieval — where the right passage is
  often absent — is not measured here, and abstention behaviour in particular
  would likely look very different.
- **No cost or token-economics arm.** Latency is CPU/GPU-local on one machine.
