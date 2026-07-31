# 9. Pin CPU torch, not CUDA wheels

**Status:** Accepted, revisit if a latency arm is added · 2026-07-31

## Context

`uv.lock` pins `torch 2.13.0+cpu`, so bge-m3 embedding and the bge-reranker
cross-encoder run on CPU. That is where the reranker's **+2.2 s p95** comes from —
a cross-encoder scoring 30 candidate pairs is exactly the workload a GPU eats, so
that figure is close to a worst case.

The development machine has an RTX 4090. The question is whether the repository
should ship CUDA.

## Decision

Keep the CPU build pinned. Keep `settings.embedding_device` and
`settings.reranker_device` as settings (both defaulting to `cpu`), passed straight
to `SentenceTransformer(device=...)` and `CrossEncoder(device=...)`, so a GPU user
sets one environment variable.

## Alternative rejected

**Pin CUDA wheels and report GPU latencies.** Rejected on a trade rather than a
principle:

- CUDA wheels are ~2.5 GB and platform-specific. `docs/REPRODUCE.md` demonstrates
  this repo installing and reproducing from a destroyed stack on a CPU-only
  machine, and CI runs on CPU-only Ubuntu runners. Pinning CUDA breaks both, in
  exchange for a number no conclusion currently depends on.
- **The accuracy findings are device-independent.** Same candidates, same
  ordering, same metrics — only the latency column moves. That the reranker buys
  +0.005 MRR does not depend on how fast the money is spent.

Also rejected: **quietly measuring on GPU and reporting the faster number** while
the lockfile says CPU. The reported latency must come from the configuration a
reader would get.

## Consequences

- The 2.2 s stands as measured, and every place it appears now says *CPU* out
  loud — `docs/ablation.md`, README limit 9, ADR 0005.
- **No GPU latency is claimed anywhere**, because none was measured.
- Generation is unaffected and this is worth not confusing: the LLM arms run
  through **Ollama**, which uses the GPU independently of torch. On the machine
  the generation numbers were measured on, `ollama ps` reports `llama3.1` fully
  resident in VRAM. Those are already GPU numbers.

## Reverses if

Latency becomes a reported product metric, or a GPU-vs-CPU arm is added to the
ablation. At that point the right move is a **measured** arm with the device in
the arm name — not swapping the pin and restating the old table with faster
numbers.
