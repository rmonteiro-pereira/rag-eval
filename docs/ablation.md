# M4 — Retrieval ablation

Regenerate with:

```bash
uv run python -m eval.ablation --out eval/reports/ablation.json
```

Every number below is read from `eval/reports/ablation.json`, produced by that
command against the live Qdrant collection (636 chunks, 30 BACEN Copom minutes,
250ª–279ª meetings) and the 49 answerable rows of the gold set. Seven arms, one
process, one corpus snapshot, one scoring function.

> **These are not validated results.** All 56 gold rows are `draft` — written by
> an agent, reviewed by nobody. The ablation measures *relative* movement
> between arms on a fixed question set, which is the part that survives the gold
> set being provisional. The absolute levels do not. See
> `eval/datasets/README.md`.

---

## The defect this was built to kill

Thirty meeting minutes, each containing one paragraph that reads within a few
words of every other:

> O Copom decidiu **reduzir** a taxa basica de juros para **14,25%** a.a., e
> entende que essa decisao e compativel com a estrategia de convergencia da
> inflacao para o redor da meta...

Only the verb and the number differ. Dense retrieval on this corpus does exactly
what it is asked: it finds the paragraph closest in meaning, and semantically all
thirty are the same paragraph. Asked about June 2026 it would return March 2025's
copy without hesitation.

**The baseline got the right meeting at rank 1 on 4 of 41 questions that named
the meeting outright** — 9.8%. It was not confused about the topic. It was
answering a question nobody asked.

---

## Arms

| arm | dense | BM25 | meeting filter | reranker |
|---|:--:|:--:|:--:|:--:|
| `dense` — the M1 baseline | ● | | | |
| `bm25` — isolated sparse control | | ● | | |
| `hybrid` — RRF, k=60 | ● | ● | | |
| `hybrid+rerank` | ● | ● | | ● |
| `dense+metadata` | ● | | ● | |
| `hybrid+metadata` | ● | ● | ● | |
| `hybrid+rerank+metadata` — full stack | ● | ● | ● | ● |

Not a cumulative ladder. A ladder can only conclude "the full stack beats the
baseline", and on this corpus it would have been actively misleading — two of
the middle rungs are *worse* than a rung below them. The arms are instead chosen
so that meaningful pairs differ in exactly one component.

## Results

Macro-averaged over 49 answerable rows. `r1-disamb` and `r1-rev` are the probes
(below): fraction of queries whose **rank-1 hit came from the right document**.

| arm | recall@1 | recall@5 | hit@1 | hit@5 | hit@10 | nDCG@10 | MRR | p95 ms | r1-disamb | r1-rev |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `dense` | 0.053 | 0.149 | 0.082 | 0.367 | 0.531 | 0.161 | 0.191 | 7 | 0.098 | 0.000 |
| `bm25` | 0.090 | 0.217 | 0.245 | 0.592 | 0.694 | 0.271 | 0.382 | 0.5 | 0.927 | 0.375 |
| `hybrid` | 0.101 | 0.181 | 0.265 | 0.510 | 0.694 | 0.261 | 0.381 | 7 | 0.341 | 0.375 |
| `hybrid+rerank` | 0.051 | 0.205 | 0.204 | 0.510 | 0.694 | 0.267 | 0.342 | 2553 | 0.195 | 0.500 |
| `dense+metadata` | 0.194 | 0.472 | 0.571 | 0.878 | 0.918 | 0.603 | 0.689 | 7 | **1.000** | 0.000 |
| `hybrid+metadata` | 0.191 | 0.466 | 0.592 | 0.898 | **1.000** | 0.619 | 0.736 | 7 | **1.000** | 0.375 |
| **`hybrid+rerank+metadata`** | 0.186 | 0.467 | 0.592 | **0.959** | 0.959 | **0.623** | **0.741** | 2217 | **1.000** | **0.500** |

Latencies exclude bge-m3 query encoding, which is identical for every arm using
dense retrieval and was measured once at **75 ms median / 84 ms p95** on CPU. Add
it back for end-to-end numbers. (The first version of this report did *not*
exclude it, and the shared query-vector cache charged the entire cost to
whichever arm ran first — making `dense` look 13× slower than `dense+metadata`
for strictly less work. The fix is in `eval/ablation.py`.)

**Full stack vs baseline:** MRR **0.191 → 0.741** (+0.550), hit@5 **0.367 →
0.959** (+0.592), nDCG@10 **0.161 → 0.623** (+0.462), and rank-1 meeting
accuracy **0.098 → 1.000**.

Note `recall@k` stays low in absolute terms even at the top. That is set recall
against complete qrels: a gold span typically spans several chunks, so recall@1
is bounded above by `1/|relevant|` and can never approach 1. `hit_rate@k` is the
"did we get something usable" view and is the one to read at low k.

---

## Controlled contrasts — what each component actually bought

Each row is a pair of arms differing in exactly one component.

Every delta below now carries a 95% CI and a paired randomisation p-value in
[`../eval/reports/significance.json`](../eval/reports/significance.json),
regenerable with `uv run python -m eval.significance` from the committed
per-query data. Three contrasts clear zero comfortably (p = 0.0001); the two
reranker contrasts do not (p = 0.54 and p = 0.93).

| component | ΔMRR | Δhit@5 | Δ r1-disamb | Δ p95 ms |
|---|--:|--:|--:|--:|
| sparse (BM25 fused into dense) | +0.190 | +0.143 | +0.244 | ~0 |
| **metadata filter**, on `dense` | **+0.498** | **+0.510** | **+0.902** | ~0 |
| **metadata filter**, on `hybrid` | **+0.355** | **+0.388** | **+0.659** | ~0 |
| **metadata filter**, on `hybrid+rerank` | **+0.399** | **+0.449** | **+0.805** | −336 |
| reranker, *without* the metadata filter | **−0.039** | +0.000 | **−0.146** | +2546 |
| reranker, *with* the metadata filter | +0.005 | +0.061 | +0.000 | +2210 |

### 1. The cheapest component won by an order of magnitude

The meeting filter is ~120 lines of regex and a Qdrant payload filter
(`retrieval/metadata.py`). It costs no measurable latency — it *saves* latency,
because filtered search and filtered reranking do less work. It moved MRR
**+0.498** on the bare baseline and took rank-1 meeting accuracy from 0.098 to
**1.000**.

It works because the information was never missing. Forty-one of the forty-nine
answerable questions say which meeting they mean — "na ata de julho de 2024",
"na 279a reuniao" — and the encoder maps `julho de 2024` and `maio de 2024` to
nearly the same point. The fix is not a better encoder. It is not throwing the
constraint away.

**Hint precision was 41/41.** Every question carrying a hint resolved to a
document set containing the gold document; zero false positives. That number is
the one that licenses using the hint as a hard filter rather than a soft boost —
a filter that is confidently wrong removes the right document before any later
stage can rescue it. It is reported per-run in `hint_diagnostics.resolved_but_wrong`
precisely so that regression is visible the moment it happens.

### 2. The expensive, fashionable component did not pay for itself

The bge-reranker cross-encoder costs **+2.2 seconds of p95 latency**, roughly
300× the entire rest of the pipeline. In exchange:

- Without the metadata filter its point estimates are negative — MRR −0.039,
  rank-1 meeting accuracy −0.146 — and **neither is distinguishable from zero**:
  95% CI [−0.164, +0.080] (p = 0.54) and [−0.342, +0.049] (p = 0.21) respectively.

  This document previously said "actively harmful", which the data does not
  support. The mechanism is still the plausible one — it reorders 30 candidates
  by semantic fit, and semantic fit is exactly the signal that cannot distinguish
  two meetings, so it can promote a beautifully-matching paragraph from the wrong
  ata. But a plausible mechanism plus a negative point estimate at n=49 is not
  evidence of harm. **The claim the numbers support is "it does not help".**
- With the filter it is roughly neutral on the headline: MRR +0.005, and it
  *loses* a query at hit@10 (1.000 → 0.959).

Its one real contribution is the reverse-lookup probe, 0.375 → 0.500. That is
four queries out of eight instead of three.

**If this were a production system with a latency budget, the reranker would be
cut.** It survives in the default config on one condition: it is the only
component that improves the probe group the metadata filter structurally cannot
help, and reverse lookup is where the remaining headroom is. That is a research
justification, not a serving one, and the writeup says so.

#### The 2.2 s is a CPU number, and that qualifier is load-bearing

Every latency in this document was measured with **`torch 2.13.0+cpu`** — a
CPU-only build, which is what `uv.lock` pins. A cross-encoder scoring 30
candidate pairs is exactly the workload a GPU eats, so the 2.2 s is close to a
worst case rather than an intrinsic property of reranking.

Two things follow, and only one of them is a code change:

- **The device is already configurable.** `settings.reranker_device` and
  `settings.embedding_device` (both defaulting to `cpu`) are passed straight to
  `CrossEncoder(device=...)` and `SentenceTransformer(device=...)`. On a machine
  with CUDA torch installed, `RERANKER_DEVICE=cuda` in `.env` is the whole
  change. Nothing here hardcodes CPU.
- **Shipping a CUDA torch is a trade, not an upgrade.** The CUDA wheels are
  platform-specific and roughly 2.5 GB, and pinning them would mean this
  repository no longer installs and reproduces on the CPU-only machine that
  `docs/REPRODUCE.md` demonstrates it installs and reproduces on. That is a
  worse default for a repository whose point is that its numbers come back.

So the honest status is: **the accuracy findings above are device-independent —
identical rankings, identical metrics — and only the latency column would move.**
The conclusion that the reranker buys +0.005 MRR for the money does not depend
on how fast the money is spent; what a GPU would change is how easy that trade
is to accept. Nothing in this repository has measured it, so nothing in this
repository claims it.

(The generation and judge numbers in `docs/generation.md` are *not* affected by
this: they run through Ollama, which uses the GPU on its own, independently of
torch. On the machine these were measured on, `ollama ps` reports `llama3.1`
fully resident in VRAM.)

### 3. Fusing a strong arm with a weak one drags the strong one down

`bm25` alone gets rank-1 meeting accuracy **0.927**. Fusing it with `dense`
(0.098) gives `hybrid` **0.341** — worse than the sparse arm on its own, on the
metric that matters most here, while MRR barely moves (0.382 → 0.381).

Reciprocal rank fusion weights its inputs equally. That is the right prior when
arms are comparably good and wrong when one is near-random on the decisive
dimension. BM25 wins on this corpus for an unsurprising reason once stated: the
questions contain rates (`13,25%`), the documents contain rates, and an exact
lexical match on a rare numeric token is worth more than any amount of semantic
similarity between paragraphs that are already near-duplicates. The tokenizer
keeping `13,25` as one token (`retrieval/text.py`) is load-bearing.

Weighted or learned fusion is the obvious next arm. It is not in this report
because tuning a fusion weight against a 49-row draft gold set would be fitting
noise and calling it a result.

---

## Probes — the wrong-meeting trap, scored directly

`eval/probes.py`. One boolean per query: **was the rank-1 hit from the document
the gold row names?** No graded gain, no cutoffs. Rank-1 because that is the
passage a generator leads with and a user reads.

Group membership is decided mechanically, not by hand, so the split survives the
gold set growing: a question belongs to `meeting_disambiguation` if its meeting
hint resolves to a document in the corpus, and to `reverse_lookup` if it carries
no hint at all.

| group | n | `dense` | `bm25` | `hybrid` | full stack |
|---|--:|--:|--:|--:|--:|
| `meeting_disambiguation` — the question names its meeting | 41 | 0.098 | 0.927 | 0.341 | **1.000** |
| `reverse_lookup` — the question must identify the meeting from content | 8 | 0.000 | 0.375 | 0.375 | **0.500** |

**`meeting_disambiguation` is solved: 41/41.** The defect is closed on the
population it was defined over.

**`reverse_lookup` is not, and it is where the honest remaining weakness lives.**
Four of eight still put the wrong meeting first:

| id | question | rank-1 returned | right doc in top-5? |
|---|---|---|:--:|
| gold-043 | *Em qual reuniao o Copom elevou a Selic em 1,00 p.p. para 12,25% a.a.?* | 258ª (nov 2023) | yes |
| gold-045 | *Em qual reuniao o Copom elevou a taxa Selic para 10,75% a.a.?* | 267ª (dez 2024) | yes |
| gold-046 | *Em qual reuniao a Selic foi reduzida para 12,75% a.a.?* | 258ª (nov 2023) | yes |
| gold-048 | *Em qual reuniao o Copom reduziu a Selic para 11,25% a.a.?* | 258ª (nov 2023) | yes |

All four have the correct document **in the top 5** (`doc_in_top5` = 1.000 for
this group at the full stack). So this is a *ranking* failure, not a retrieval
failure — the evidence is being fetched and then mis-ordered. Three of the four
land on the same wrong document, the 258ª, which is the ata whose text happens to
recite the most other meetings' rate levels.

The structural reason is plain: metadata filtering cannot fire when the question
names no meeting, so these eight queries run on hybrid+rerank alone — which is
the arm the contrasts above show to be the weak one. The natural next move is a
rate-to-meeting index built from the decision paragraphs, which is a metadata
extraction problem, not a retrieval one.

---

## What is now the default

`settings.retrieval_config = "hybrid+rerank+metadata"`. `eval.run_eval` still
defaults to `--config dense`, deliberately: that command produced the committed
M1/M2 baseline and has to keep producing it, or the "before" half of every
before/after number quietly moves.

## Honest limits of this ablation

- **Draft gold set.** 56 rows, agent-written, human-validated: zero. Relative
  movement between arms is the defensible reading; absolute levels are not.
- **41 of 49 answerable questions name their meeting.** The gold set was written
  before the metadata filter existed, but it was written by an agent reading
  these documents, and questions phrased the way the documents are indexed are
  over-represented relative to what a real user would type. The measured
  +0.498 MRR from the filter is an upper bound on what it would deliver against
  free-form questions.
- **Single run, no seeds, no confidence intervals.** Retrieval here is
  deterministic given a fixed collection, so re-running reproduces the numbers
  exactly; that is repeatability, not statistical significance. With n=49 and
  n=8, a one-query swing in the reverse-lookup probe is ±0.125.
- **One reranker, one embedder, one chunking strategy.** Chunk size and
  embedding model are held fixed across all seven arms and were never ablated.
- **Latency is CPU-only, on one machine, with warm caches.** It ranks the arms;
  it does not predict production.
