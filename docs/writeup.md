# rag-eval — architecture, numbers, failure modes

A production-shaped RAG system over 30 BACEN Copom minutes, built around an
evaluation harness rather than a demo. Everything local, everything free: no paid
API, no key anywhere in the repo.

This is the artifact to read if you want to know what was measured, what the
measurements said, and where the numbers stop being trustworthy.

**Every number here is regenerable.** The command that produced it is next to it.

---

## 1. The one-paragraph version

The naive baseline scored **MRR 0.191** because dense retrieval kept returning
the right paragraph from the **wrong Copom meeting** — thirty documents each
contain a near-identical *"Decisão de política monetária"* paragraph. Seven
measured retrieval arms later it is **MRR 0.741, hit@5 0.959**, and rank-1
correct-meeting accuracy went from **0.098 to 1.000** on the 41 questions that
name their meeting. The component that did it was not the cross-encoder reranker
— which costs 2.2 s of p95 latency and does not pay for itself — but ~120 lines
of regex that read the date out of the question and turn it into a Qdrant payload
filter. On top of that: generation measured across three local backends with zero
hallucinated numbers, a guardrail suite with an **8.3%** residual injection
attack-success rate, a document ACL proven to leak zero restricted chunks, and a
CI gate that provably fails on a degraded fixture.

The two things it does *not* have are stated as loudly as the things it does: the
gold set is **56 draft rows that no human has validated**, and the LLM judge is
**near-chance** — two local judges agree on faithfulness 44% of the time.

---

## 2. Architecture

```
ingest/       PDFs -> pypdf (page-aware) -> 1200/200 char chunks -> bge-m3 -> Qdrant
retrieval/    metadata filter -> {dense | BM25} -> RRF -> bge-reranker
generation/   stuff top-k -> Ollama (qwen2.5:3b | llama3.1) | extractive -> citations
guardrails/   PII mask (in+out) · injection detection · the governed query path
governance/   document ACL as a Qdrant payload filter · append-only audit log
agent/        sql_query (DuckDB marts) + rag_search, behind an HITL gate
eval/         gold set · metrics · run_eval · ablation · run_generation · calibration
              · run_adversarial · regression_gate
serving/      FastAPI /ask + a minimal UI, over the governed pipeline
```

| layer | choice | why |
|---|---|---|
| Vector store | Qdrant | its payload filter is what both the meeting filter and the ACL compile down to |
| Embeddings | bge-m3 | multilingual, strong on Portuguese, CPU-viable |
| Sparse | BM25, written in-repo | forty lines of arithmetic; an ablation is more defensible when the thing it compares against is visible |
| Reranker | bge-reranker-base | XLM-R based (genuinely multilingual), a third the size of v2-m3 |
| LLM | Ollama: qwen2.5:3b, llama3.1 | free, local, no lock-in |
| PII | Presidio + custom BR recognisers | Presidio alone does not detect a CPF |
| Tracing | Langfuse v2 | self-hosted |
| Marts | DuckDB export from Open-Finance-LakeHouse | real data from the sibling project |

**Corpus.** 30 Copom minutes, Oct 2022 – Jun 2026, 194 text pages, **636 chunks**.
Only `data/manifest.json` (URL, title, date, SHA-256 per document) is committed;
the PDFs are reproducible from it, so no binaries enter git.

---

## 3. The defect this project is actually about

Every ata contains a paragraph within a few words of every other:

> O Copom decidiu **reduzir** a taxa basica de juros para **14,25%** a.a., e
> entende que essa decisao e compativel com a estrategia de convergencia da
> inflacao para o redor da meta...

Only the verb and the number differ. Dense retrieval does exactly what it is
asked — find the paragraph closest in meaning — and semantically all thirty are
the same paragraph. Asked about June 2026 the baseline would return March 2025's
copy without hesitation.

**The baseline got the right meeting at rank 1 on 4 of 41 questions that named
the meeting outright.** It was not confused about the topic. It was answering a
question nobody asked.

This is the failure mode that a headline `nDCG` hides, which is why
`eval/probes.py` exists: one boolean per query — *was the rank-1 hit from the
document the gold row names* — over two groups split by a mechanical rule, so the
split survives the gold set growing.

---

## 4. Retrieval: what each component actually bought

`uv run python -m eval.ablation` · full detail in [`ablation.md`](ablation.md)

| arm | recall@5 | hit@5 | nDCG@10 | MRR | p95 ms | rank-1 meeting |
|---|--:|--:|--:|--:|--:|--:|
| `dense` (M1 baseline) | 0.149 | 0.367 | 0.161 | 0.191 | 7 | 0.098 |
| `bm25` | 0.217 | 0.592 | 0.271 | 0.382 | 0.5 | 0.927 |
| `hybrid` | 0.181 | 0.510 | 0.261 | 0.381 | 7 | 0.341 |
| `hybrid+rerank` | 0.205 | 0.510 | 0.267 | 0.342 | 2553 | 0.195 |
| `dense+metadata` | 0.472 | 0.878 | 0.603 | 0.689 | 7 | **1.000** |
| `hybrid+metadata` | 0.466 | 0.898 | 0.619 | 0.736 | 7 | **1.000** |
| **`hybrid+rerank+metadata`** | 0.467 | **0.959** | **0.623** | **0.741** | 2217 | **1.000** |

Seven arms, **not a cumulative ladder** — deliberately, because two of the middle
rungs are *worse* than a rung below them and a ladder would have hidden that. The
arms are chosen so meaningful pairs differ in exactly one component:

| component | ΔMRR | Δhit@5 | Δ rank-1 meeting | Δ p95 ms |
|---|--:|--:|--:|--:|
| sparse (BM25 fused into dense) | +0.190 | +0.143 | +0.244 | ~0 |
| **metadata filter**, on `dense` | **+0.498** | **+0.510** | **+0.902** | ~0 |
| **metadata filter**, on `hybrid` | **+0.355** | **+0.388** | **+0.659** | ~0 |
| reranker, *without* the filter | **−0.039** | +0.000 | **−0.146** | +2546 |
| reranker, *with* the filter | +0.005 | +0.061 | +0.000 | +2210 |

**Three findings a leaderboard would have hidden:**

1. **The cheapest component won by an order of magnitude.** The meeting filter is
   a regex plus a payload filter. +0.498 MRR on the bare baseline, at *negative*
   latency cost (filtered search does less work), with **41/41 hint precision** —
   zero false positives, which is the number that licenses using it as a hard
   filter rather than a soft boost.

2. **The expensive, fashionable component did not pay for itself.** The
   cross-encoder is *actively harmful* without the filter (MRR −0.039, rank-1
   meeting −0.146): it reorders by semantic fit, and semantic fit is exactly the
   signal that cannot tell two Copom meetings apart. With the filter it is +0.005
   MRR for +2.2 s. **In a system with a latency budget it would be cut.** It
   survives in the default arm on one condition — it is the only thing that
   improves the probe group the filter structurally cannot help.

3. **Fusing a strong arm with a weak one drags the strong one down.** BM25 alone
   gets 0.927 rank-1 meeting accuracy; fused with dense (0.098) it gives 0.341.
   RRF weights arms equally, which is the wrong prior when one is near-random on
   the decisive axis.

A measurement bug found and fixed while writing the report: the shared
query-vector cache charged all bge-m3 encoding to whichever arm ran first, making
`dense` look 13× slower than `dense+metadata` for strictly less work. Encoding is
now warmed up front and reported once (75 ms median).

---

## 5. Generation

`uv run python -m eval.run_generation` · detail in [`generation.md`](generation.md)

Three backends over all 56 gold rows, **sharing one retrieval pass** so arm
differences are generation differences only.

| arm | numeric recall | groundedness | hallucinated numbers | abstention ok | false refusal |
|---|--:|--:|--:|--:|--:|
| `extractive` | **0.913** | **0.988** | **0.000** | **0.000** | 0.000 |
| `qwen2.5:3b` | 0.777 | 0.838 | **0.000** | 1.000 | 0.082 |
| `llama3.1` | 0.826 | 0.907 | **0.000** | 1.000 | 0.041 |

- **Zero hallucinated numbers across 168 answers.** No arm asserted a rate absent
  from the retrieved evidence or the question. Narrow claim, narrowly stated: it
  catches *fabricated* numbers, not a fluent misreading of a number that is
  present.
- **The mode with the best groundedness cannot say "I don't know".** `extractive`
  wins numeric recall and groundedness and scores **0.000 on abstention** — on
  all seven out-of-scope questions it returned five passages as though they were
  an answer. A dashboard ranking backends on groundedness alone picks exactly the
  wrong one.
- **Abstention is not free**: both generative arms refuse correctly on all seven
  negatives and pay with false refusals on answerable questions.

---

## 6. The LLM judge fails its own audit

This is the part a portfolio project usually skips.

| arm | judge faithfulness (0–2) | judge answer relevance |
|---|--:|--:|
| `extractive` | 1.446 | 1.554 |
| `qwen2.5:3b` | 1.750 | 1.554 |
| `llama3.1` | **1.804** | **1.643** |

Read that and the obvious conclusion is "llama3.1 writes the best answers". Two
measured facts make it unsafe:

**The judge graded its own homework.** The judge *is* `llama3.1`, and it wrote
every answer in the `llama3.1` arm — all 56 rows flagged `judge_is_generator`.
The arm the judge wrote scored highest. Self-preference and quality cannot be
separated by this design.

**Two local judges barely agree with each other.** `--second-judge qwen2.5:3b`
re-scores the calibration rows with an independent judge; no human needed:

| criterion | Cohen's κ | raw agreement | n |
|---|--:|--:|--:|
| **faithfulness** | **0.138** | **0.440** | 25 |
| answer relevance | 0.487 | 0.667 | 27 |

**On faithfulness two judges disagree more than half the time**, at a kappa
barely distinguishable from chance. Raters that disagree with each other cannot
both be right, so **the faithfulness column carries almost no information.** The
deterministic metrics are the ones to believe — which is why the harness computes
arithmetic first and treats the judge as the flexible, suspect instrument.

`eval/datasets/judge_calibration_sheet.jsonl` holds 30 items with the human
columns empty, balanced 10/10/10 across answer types, each carrying the retrieved
passages verbatim. Until it is filled, judge-vs-human agreement is reported as
**unknown — not good**.

---

## 7. Guardrails and governance

`uv run python -m eval.run_eval --suite adversarial` · detail in
[`governance.md`](governance.md)

| metric | governed | ungoverned |
|---|--:|--:|
| **injection attack success** (24 attacks) | **8.3%** | 16.7% |
| — direct surface | 11.1% | 16.7% |
| — **indirect (poisoned passage)** | **0.0%** | 16.7% |
| **PII output leak** (corpus-supplied) | **0.0%** | **100.0%** |
| PII false positives on clean queries | 0.0% | — |
| abstention / false refusal | 100% / 4.1% | — |
| **restricted chunks reaching an uncleared user** | **0** | — |

- **8.3%, not 0%.** Two attacks defeat the stack, both named in the report. The
  residual is a precision/recall property of a pattern detector — widening the
  rules to catch them would flag *"desconsidere o cenario alternativo"*, a
  legitimate monetary-policy question, and there is a test asserting it is not
  flagged.
- **Detection ≠ defence**, and the report keeps them apart. Detection was 58.3%;
  eight attacks were undetected *and failed anyway*, including base64 and
  character-stuffing evasions.
- **The indirect surface is where the guardrail earns its place** (0.0% vs
  16.7%). Injection carried by a retrieved document is the RAG-specific attack; a
  system that only inspects the user's question has no defence against it.
- **Presidio misses the CPF entirely** — measured, so `guardrails/brazilian.py`
  adds recognisers that validate **check digits**, not shape.
- **The ACL is a payload filter inside the query, not a post-filter.** A
  post-filter has already read the document, and a shortened result list leaks
  how many matched. Proven live: an analyst asking for 200 results gets zero
  restricted ones, a supervisor gets all five, and still zero when the query is
  aimed straight at a restricted document.
- **The audit log stores the masked query and a SHA-256 of the raw one** — never
  the raw query, the answer, or a matched PII substring.

---

## 8. Agent mode

`uv run python -m rag.agent --demo` · transcript in [`agent_demo.md`](agent_demo.md)

Two tools — `sql_query` (read-only DuckDB over 8 gold marts from the
Open-Finance-LakeHouse project) and `rag_search` (the governed pipeline) — behind
an explicit ~40-line loop rather than a framework. 10 questions, **6 answered via
SQL** with the statement shown, 24 statements gated (18 approved / 6 refused).

The HITL gate classifies risk *before* the policy decides, so `auto` still
refuses `high` outright with nobody to ask — and the demo **proves** it by
putting four representative statements through the gate directly, three of which
come back REFUSED.

Honest status: **tool routing works, tool composition does not.** The two
questions needing one fact from each source were answered from retrieval alone.
And there is **no eval harness for agent mode** — no gold set for multi-tool
questions, no measured task-success rate. It is a demonstration, not a
measurement, and that is the gap against the rest of the project.

---

## 9. CI gate and serving

**Gate** (`eval/regression_gate.py`) compares a fresh report to the committed one
with per-metric thresholds, and gates on the **probe** metrics as well as the
averages — the metric specific to the known defect, not only the one that is easy
to average. Breaking the meeting resolver collapses `meeting_disambiguation` from
1.000 to near zero and is impossible to miss; the averages would drop into a
range that still looks like "a retrieval system".

Proven both ways, asserted by `tests/test_regression_gate.py`:

```
$ uv run python -m eval.regression_gate --baseline eval/reports/ablation.json \
      --candidate eval/reports/ablation.json          # PASS — 9 metrics within tolerance → 0
$ uv run python -m eval.regression_gate --baseline tests/fixtures/gate_baseline.json \
      --candidate tests/fixtures/gate_degraded.json   # FAIL — 8 regressed → 1
```

Exit 2 is reserved for "could not compare" (missing file, unknown arm, absent
metric), because a gate that passes because it could not find the numbers is
worse than no gate.

`.github/workflows/eval.yml` is committed and **inert** — this repo is remote-less
by design, so it has never run. The hermetic `gate-selfcheck` job (no Qdrant, no
models, seconds) is the one meant to be required for merge; the full ablation is
`workflow_dispatch`-only, because a 20-minute CPU job with a model download on
every PR is how a gate gets disabled.

**Serving** (`serving/api.py`) exposes `/ask`, `/health`, `/config` and a minimal
UI over the **governed** pipeline — there is no code path to retrieval that skips
the guardrails. Smoke-tested end to end:

```
analyst   → "279a reuniao de junho de 2026"  → decision: abstained, 0 sources   (ACL)
supervisor → same question                    → 14,25%, 5 sources marked `restricted`
analyst   → CPF + "ignore as instrucoes"      → blocked_injection, CPF masked as BR_CPF
```

---

## 10. Honest limits

Ordered by how much they should change your reading.

1. **The gold set is 56 draft rows and nobody has validated one.** Every row was
   written by an agent from the ingested text and programmatically re-verified
   (span locatable in a chunk of its document on its page, doc id in the
   manifest, title matching verbatim) — so the rows are *grounded*, not
   *validated*. A span can be real and the question built on it still be
   ambiguous or answerable from three other atas. Relative movement between arms
   on a fixed question set is the defensible reading; **absolute levels are not.**
   `--min-status validated` turns the human pass into a one-command re-run.

2. **The LLM judge is near-chance on faithfulness** (κ=0.138 against a second
   judge, unknown against humans). Judge numbers are reported because hiding them
   would be worse, not because they are trusted.

3. **41 of 49 answerable questions name their meeting.** The gold set was written
   before the metadata filter existed — but by an agent reading these documents,
   so questions phrased the way the documents are indexed are over-represented
   relative to what a real user would type. **The measured +0.498 MRR from the
   filter is an upper bound** on what it would deliver against free-form
   questions.

4. **Reverse lookup is the unsolved half.** Questions naming no meeting
   (*"Em qual reuniao a Selic foi reduzida para 12,75%?"*) still rank the wrong
   ata first 4 times out of 8. All four have the right document in the top 5, so
   it is a ranking failure, not a retrieval one. The natural fix is a
   rate-to-meeting index built from the decision paragraphs — a metadata
   extraction problem, not a retrieval one.

5. **8.3% of injection attacks still succeed**, and 24 hand-written attacks is a
   small corpus targeting the failure modes its author thought of. One attack is
   4.2 percentage points. This is a demonstrated control, not a red-team result.

6. **The ACL classification is synthetic.** These are public BACEN documents; the
   five most recent meetings stand in for a publication embargo. Every report
   carries `synthetic: true`. It demonstrates the mechanism, not a real
   classification. The ACL is also document-level — no chunk or field redaction
   inside a permitted document.

7. **Single run, no seeds, no confidence intervals.** Retrieval is deterministic
   given a fixed collection, so re-running reproduces exactly — that is
   *repeatability*, not statistical significance. With n=49 and n=8, one query in
   the reverse-lookup probe is ±0.125.

8. **Chunking and the embedding model were never ablated.** Fixed at 1200/200 and
   bge-m3 across all seven arms. RRF is unweighted at k=60, the paper's constant;
   tuning it against 49 draft rows would be fitting noise.

9. **Latency is CPU/GPU-local on one machine with warm caches.** It ranks the
   arms; it does not predict production. There is no cost model and no
   token-economics arm.

10. **Agent mode has no evaluation at all** (see §8), and the corpus is one
    domain in one language. Nothing here has been run against a public QA
    benchmark for cross-corpus comparison.

---

## 11. Reproducing everything

```bash
docker compose up -d                       # Qdrant + Langfuse + Postgres
uv sync
uv run python -m ingest.pipeline --download 30

uv run python -m eval.run_eval --min-status draft --out eval/reports/baseline_dense.json
uv run python -m eval.ablation --out eval/reports/ablation.json
uv run python -m eval.run_generation --out eval/reports/generation.json
uv run python -m eval.calibration --second-judge qwen2.5:3b
uv run python -m eval.run_eval --suite adversarial
uv run python -m rag.agent --demo
uv run python -m eval.regression_gate --baseline eval/reports/ablation.json \
    --candidate eval/reports/ablation.json

uv run pytest -q && uv run ruff check .
uv run uvicorn serving.api:app --port 8000
```

**The two human gates neither an agent nor a model may close:**

- `eval/datasets/gold_seed.jsonl` — flip `status` to `validated` after checking
  each row against the source PDF. There is a test that fails if an agent ever
  sets it.
- `eval/datasets/judge_calibration_sheet.jsonl` — fill the human columns, then
  `uv run python -m eval.calibration` reports Cohen's kappa.

Until then, every number in this document is a harness that works, measuring a
system that has not been independently checked. That distinction is the project.
