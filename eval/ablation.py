"""M4 ablation: score every retrieval arm on the same gold set, same corpus,
same scoring code, and report the deltas.

    uv run python -m eval.ablation --out eval/reports/ablation.json

Runs the arms in `retrieval.configs.ABLATION_CONFIGS` in order, each adding one
component to the last, and writes:

* per-arm aggregate retrieval metrics,
* `delta_vs_dense` and `delta_vs_previous` for every metric — the second is what
  actually attributes a gain to a component, since the first only says the whole
  stack is better than the baseline,
* median and p95 per-query latency, because a reranker that doubles quality and
  decuples latency is a trade-off to state, not a win to announce,
* a `probes` section scoring rank-1 document correctness on the wrong-meeting
  trap (see `eval.probes`).

One process, one corpus scroll, one BM25 index, one query-vector cache, one
reranker — shared across arms so the only thing that differs between them is the
component under test.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from eval.corpus import to_chunk_refs
from eval.gold import DEFAULT_GOLD_PATH, GoldRow, GoldSetError, load_gold, status_counts
from eval.metrics.retrieval import DEFAULT_KS
from eval.probes import hint_diagnostics, run_probes
from eval.run_eval import DRAFT_CAVEAT, _parse_ks
from eval.scoring import score_rows
from rag.config import REPO_ROOT, settings
from retrieval.configs import ABLATION_CONFIGS, RetrievalContext, build_retriever

SCHEMA_VERSION = 1

#: Metrics carried into the delta tables. Everything else stays in the per-arm
#: aggregate; these are the ones the writeup argues from.
HEADLINE_METRICS: tuple[str, ...] = (
    "recall@1",
    "recall@5",
    "hit_rate@1",
    "hit_rate@3",
    "hit_rate@5",
    "hit_rate@10",
    "ndcg@5",
    "ndcg@10",
    "mrr",
)

#: `(component, without, with)` — pairs of arms differing in exactly one thing.
#:
#: This is the part that makes the run an ablation rather than a leaderboard. A
#: leaderboard says the full stack wins; only a controlled pair can say *what*
#: won, and the same component measured against different backgrounds is allowed
#: to disagree with itself. On this corpus it does, loudly, and that disagreement
#: is the most useful thing in the report.
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("sparse (BM25 fused into dense)", "dense", "hybrid"),
    ("reranker, without the metadata filter", "hybrid", "hybrid+rerank"),
    ("reranker, with the metadata filter", "hybrid+metadata", "hybrid+rerank+metadata"),
    ("metadata filter, on the dense baseline", "dense", "dense+metadata"),
    ("metadata filter, on hybrid", "hybrid", "hybrid+metadata"),
    ("metadata filter, on hybrid+rerank", "hybrid+rerank", "hybrid+rerank+metadata"),
)


class TimedRetriever:
    """Wraps an arm to record per-query wall-clock latency."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.latencies_ms: list[float] = []

    def retrieve(self, question: str, top_k: int | None = None):
        started = time.perf_counter()
        hits = self.inner.retrieve(question, top_k=top_k)
        self.latencies_ms.append((time.perf_counter() - started) * 1000)
        return hits


def _latency_summary(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        return {"n": 0}
    ordered = sorted(samples)
    # p95 by nearest-rank; with ~49 samples anything fancier is false precision.
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))
    return {
        "n": len(ordered),
        "median_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(ordered[p95_index], 1),
        "mean_ms": round(statistics.fmean(ordered), 1),
    }


def _deltas(current: dict, reference: dict) -> dict[str, float]:
    return {
        metric: round(current[metric] - reference[metric], 6)
        for metric in HEADLINE_METRICS
        if metric in current and metric in reference
    }


def _build_contrasts(arms: list[dict]) -> list[dict]:
    """One entry per controlled A/B pair, including the probe deltas.

    Probe deltas are carried alongside the aggregate ones because rank-1
    document accuracy is the metric the wrong-meeting defect is actually about,
    and a component can move it hard while barely touching nDCG.
    """
    by_name = {arm["name"]: arm for arm in arms}
    contrasts: list[dict] = []
    for component, without_name, with_name in CONTRASTS:
        if without_name not in by_name or with_name not in by_name:
            continue
        without, with_ = by_name[without_name], by_name[with_name]
        contrasts.append(
            {
                "component": component,
                "without": without_name,
                "with": with_name,
                "delta": _deltas(with_["aggregate"], without["aggregate"]),
                "probe_delta": {
                    group: round(
                        with_["probes"][group]["rank1_doc_accuracy"]
                        - without["probes"][group]["rank1_doc_accuracy"],
                        6,
                    )
                    for group in with_["probes"]
                },
                "latency_delta_p95_ms": round(
                    with_["latency"].get("p95_ms", 0) - without["latency"].get("p95_ms", 0), 1
                ),
            }
        )
    return contrasts


def run_ablation(
    rows: list[GoldRow],
    gold_path: Path,
    min_status: str,
    ks: tuple[int, ...],
    configs=ABLATION_CONFIGS,
) -> dict:
    top_k = max(ks)
    context = RetrievalContext.build()
    chunks = to_chunk_refs(context.corpus)

    # Warm the query-vector cache before anything is timed.
    #
    # bge-m3 costs ~90 ms per question on CPU and is *identical* for every arm
    # that uses dense retrieval. Left inside the timed section it would be
    # charged entirely to whichever arm ran first and make every later arm look
    # artificially fast — which is exactly what the first version of this report
    # showed, with `dense` at 94 ms p95 and `dense+metadata` at 7 ms for strictly
    # more work. So it is paid up front, measured once, and reported separately.
    encoding_ms: list[float] = []
    for row in rows:
        if row.is_abstention:
            continue
        started = time.perf_counter()
        context.embedder.embed_query(row.question)
        encoding_ms.append((time.perf_counter() - started) * 1000)

    arms: list[dict] = []
    baseline_aggregate: dict | None = None
    first_scored = None

    for config in configs:
        print(f"  running arm {config.name} ...", file=sys.stderr, flush=True)
        retriever = TimedRetriever(build_retriever(config, context, top_k=top_k))
        result = score_rows(retriever, rows, chunks, ks=ks, top_k=top_k)
        if not result.scored:
            raise SystemExit(f"arm {config.name} scored no rows — nothing to compare")
        if first_scored is None:
            first_scored = result.scored

        aggregate = result.aggregate()
        if baseline_aggregate is None:
            baseline_aggregate = aggregate

        probes = run_probes(result.scored, context.documents)
        arms.append(
            {
                "name": config.name,
                "config": config.to_json(),
                "aggregate": aggregate,
                "delta_vs_dense": _deltas(aggregate, baseline_aggregate),
                "latency": _latency_summary(retriever.latencies_ms),
                "by_capability": result.breakdown("capability"),
                "by_difficulty": result.breakdown("difficulty"),
                "probes": {name: group.to_json() for name, group in probes.items()},
                "per_query": [sq.to_json(ks) for sq in result.scored],
            }
        )

    assert baseline_aggregate is not None and first_scored is not None

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus": {
            "n_chunks": len(chunks),
            "n_documents": len(context.documents),
            "collection": settings.qdrant_collection,
        },
        "setup": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": top_k,
            "ks": list(ks),
            "min_status": min_status,
            "gold_path": gold_path.relative_to(REPO_ROOT).as_posix(),
            "device": settings.embedding_device,
        },
        "gold": {
            "status_counts_in_file": status_counts(gold_path),
            "n_rows_after_status_filter": len(rows),
        },
        "baseline_arm": configs[0].name,
        "query_encoding_latency": {
            **_latency_summary(encoding_ms),
            "note": (
                "bge-m3 query encoding, measured once and excluded from every arm's "
                "latency so the arms are comparable. Add it back to any arm with "
                "dense=true to get end-to-end retrieval latency."
            ),
        },
        # Depends only on the questions and the corpus, never on the arm, so it
        # is computed once from the first arm's row set rather than repeated.
        "hint_diagnostics": hint_diagnostics(first_scored, context.documents),
        "probe_groups": {
            name: {"description": group.description, "n": group.n, "ids": list(group.ids)}
            for name, group in run_probes(first_scored, context.documents).items()
        },
        "contrasts": _build_contrasts(arms),
        "arms": arms,
        "caveat": DRAFT_CAVEAT,
    }


def _render_summary(report: dict) -> str:
    lines = [
        "",
        f"corpus  {report['corpus']['n_chunks']} chunks / "
        f"{report['corpus']['n_documents']} documents",
        f"gold    {report['arms'][0]['aggregate']['n_queries']} answerable rows scored per arm",
        "",
        f"{'arm':<24} {'recall@5':>9} {'hit@5':>7} {'ndcg@10':>8} {'mrr':>7} "
        f"{'p95 ms':>8} {'r1-disambig':>12} {'r1-reverse':>11}",
        "-" * 92,
    ]
    for arm in report["arms"]:
        agg = arm["aggregate"]
        probes = arm["probes"]
        lines.append(
            f"{arm['name']:<24} {agg['recall@5']:>9.3f} {agg['hit_rate@5']:>7.3f} "
            f"{agg['ndcg@10']:>8.3f} {agg['mrr']:>7.3f} "
            f"{arm['latency'].get('p95_ms', 0):>8.0f} "
            f"{probes['meeting_disambiguation']['rank1_doc_accuracy']:>12.3f} "
            f"{probes['reverse_lookup']['rank1_doc_accuracy']:>11.3f}"
        )

    lines += [
        "",
        "controlled contrasts — each pair differs in exactly one component",
        f"{'component':<40} {'mrr':>8} {'hit@5':>8} {'r1-disambig':>12} {'p95 ms':>9}",
        "-" * 82,
    ]
    for contrast in report["contrasts"]:
        lines.append(
            f"{contrast['component']:<40} "
            f"{contrast['delta']['mrr']:>+8.3f} "
            f"{contrast['delta']['hit_rate@5']:>+8.3f} "
            f"{contrast['probe_delta']['meeting_disambiguation']:>+12.3f} "
            f"{contrast['latency_delta_p95_ms']:>+9.0f}"
        )

    best = report["arms"][-1]
    lines += ["", f"delta {best['name']} vs {report['baseline_arm']}", "-" * 44]
    for metric, delta in best["delta_vs_dense"].items():
        lines.append(f"  {metric:<12} {delta:+.3f}")

    hints = report["hint_diagnostics"]
    encoding = report["query_encoding_latency"]
    lines += [
        "",
        f"meeting hint: present in {hints['hint_present']}/{hints['n_scored']} questions, "
        f"resolved for {hints['hint_resolved']}, "
        f"correct for {hints['hint_resolved_and_correct']}",
        f"latencies exclude bge-m3 query encoding, measured once at "
        f"{encoding['median_ms']:.0f} ms median / {encoding['p95_ms']:.0f} ms p95",
        "",
        report["caveat"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.ablation",
        description="Score every retrieval arm on the same gold set and report the deltas.",
    )
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--min-status", choices=["draft", "validated"], default="draft")
    parser.add_argument("--k", type=_parse_ks, default=DEFAULT_KS)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "eval" / "reports" / "ablation.json",
        help="write the JSON report here (default: eval/reports/ablation.json)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        rows = load_gold(args.gold, min_status=args.min_status)
    except GoldSetError as exc:
        print(f"gold set error: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"no gold rows at status >= {args.min_status!r}", file=sys.stderr)
        return 3

    report = run_ablation(
        rows=rows,
        gold_path=Path(args.gold) if args.gold else DEFAULT_GOLD_PATH,
        min_status=args.min_status,
        ks=tuple(args.k),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"report -> {args.out}")
        print(_render_summary(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
