"""Retrieval evaluation harness.

    uv run python -m eval.run_eval --min-status draft --out eval/reports/baseline_dense.json

Scores the current retriever against the gold set and writes a JSON report with
per-query and aggregate `recall@k`, `hit_rate@k`, `nDCG@k` and `MRR`.

Two design choices are worth stating plainly, because they are what makes the
numbers mean anything:

1. **Complete qrels.** Relevance is judged against every chunk in the
   collection (`eval/qrels.py`), so recall has a real denominator.
2. **`--min-status`.** The gold set is drafted by an agent and is *not*
   scientifically valid until a human validates each row. Today
   `--min-status draft` scores everything and the number is a smoke test of the
   harness, not a result. Once rows are flipped to `validated`, the same
   command with `--min-status validated` reports the number that counts. That
   is the whole reason the flag exists: the human pass is a re-run, not a
   rewrite.

Abstention rows carry no span, so they have no relevant chunk and are excluded
from retrieval metrics. They are counted in the report; the abstention metric
itself belongs to generation and lands in M5.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from eval.corpus import load_chunks
from eval.gold import DEFAULT_GOLD_PATH, GoldRow, GoldSetError, load_gold, status_counts
from eval.metrics.retrieval import DEFAULT_KS, RetrievalScores, aggregate, score_ranking
from eval.qrels import build_row_qrels, chunk_key
from rag.config import REPO_ROOT, settings

SCHEMA_VERSION = 1


def _parse_ks(raw: str) -> tuple[int, ...]:
    try:
        ks = tuple(sorted({int(part) for part in raw.split(",") if part.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--k must be a comma-separated int list: {exc}") from exc
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("--k must contain positive integers")
    return ks


def _breakdown(
    scored: list[tuple[GoldRow, RetrievalScores]],
    attribute: str,
) -> dict[str, dict[str, float | int]]:
    """Aggregate the same metrics per capability / per difficulty.

    Cheap, and it is where the interesting failures show up: a healthy headline
    number can hide a capability that never retrieves anything.
    """
    buckets: dict[str, list[RetrievalScores]] = {}
    for row, scores in scored:
        buckets.setdefault(getattr(row, attribute), []).append(scores)
    return {name: aggregate(group) for name, group in sorted(buckets.items())}


def build_report(
    rows: list[GoldRow],
    gold_path: Path,
    min_status: str,
    ks: tuple[int, ...],
    label: str,
) -> dict:
    from retrieval.dense import DenseRetriever
    from retrieval.store import get_client

    top_k = max(ks)
    client = get_client()
    chunks = load_chunks(client)
    if not chunks:
        raise SystemExit(
            f"collection {settings.qdrant_collection!r} is empty — "
            "run `uv run python -m ingest.pipeline` before evaluating"
        )

    retriever = DenseRetriever(top_k=top_k)

    per_query: list[dict] = []
    scored: list[tuple[GoldRow, RetrievalScores]] = []
    abstention: list[dict] = []
    skipped: list[dict] = []

    for row in rows:
        if row.is_abstention:
            abstention.append(
                {"id": row.id, "question": row.question, "capability": row.capability}
            )
            continue

        assert row.source_doc_id is not None  # guaranteed by the gold loader
        row_qrels = build_row_qrels(
            source_doc_id=row.source_doc_id,
            source_page=row.source_page,
            source_span=row.source_span,
            chunks=chunks,
        )
        if not row_qrels.qrels:
            skipped.append(
                {
                    "id": row.id,
                    "reason": "no chunk in the collection matches this row's document/page",
                    "source_doc_id": row.source_doc_id,
                    "source_page": row.source_page,
                }
            )
            continue

        hits = retriever.retrieve(row.question, top_k=top_k)
        ranking = [chunk_key(hit.doc_id, hit.chunk_index) for hit in hits]
        scores = score_ranking(ranking, row_qrels.qrels, ks=ks)
        scored.append((row, scores))

        per_query.append(
            {
                "id": row.id,
                "question": row.question,
                "capability": row.capability,
                "difficulty": row.difficulty,
                "answer_type": row.answer_type,
                "source_doc_id": row.source_doc_id,
                "source_page": row.source_page,
                "span_matched": row_qrels.span_matched,
                "metrics": scores.to_json(),
                "retrieved": [
                    {
                        "rank": rank,
                        "doc_id": hit.doc_id,
                        "page": hit.page_number,
                        "chunk_index": hit.chunk_index,
                        "score": round(hit.score, 6),
                        "gain": row_qrels.qrels.get(chunk_key(hit.doc_id, hit.chunk_index), 0),
                    }
                    for rank, hit in enumerate(hits, start=1)
                ],
            }
        )

    if not scored:
        raise SystemExit(
            "no answerable gold rows survived the filters — nothing to measure. "
            f"(--min-status {min_status}, {len(rows)} rows loaded)"
        )

    unmatched = [q["id"] for q in per_query if not q["span_matched"]]

    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "retriever": "dense",
            "collection": settings.qdrant_collection,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": top_k,
            "ks": list(ks),
            "min_status": min_status,
            "gold_path": gold_path.relative_to(REPO_ROOT).as_posix(),
            "relevance": (
                "graded over complete qrels: gain 2 = chunk contains the gold span, "
                "gain 1 = same document and page; recall/hit_rate/MRR binarise at gain >= 1"
            ),
        },
        "corpus": {
            "n_chunks": len(chunks),
            "n_documents": len({chunk.doc_id for chunk in chunks}),
        },
        "gold": {
            "status_counts_in_file": status_counts(gold_path),
            "n_rows_after_status_filter": len(rows),
            "n_scored": len(scored),
            "n_abstention_excluded": len(abstention),
            "n_skipped": len(skipped),
        },
        "aggregate": aggregate([scores for _, scores in scored]),
        "by_capability": _breakdown(scored, "capability"),
        "by_difficulty": _breakdown(scored, "difficulty"),
        "abstention_rows": abstention,
        "diagnostics": {
            "rows_with_unmatched_span": unmatched,
            "rows_skipped": skipped,
            "note": (
                "rows_with_unmatched_span are gold rows whose span could not be located "
                "verbatim in any chunk (chunk boundary or a typo in the gold file); they "
                "are still scored, at page-level relevance only"
            ),
        },
        "per_query": per_query,
        "caveat": (
            "DRAFT GOLD SET. Every row was written by an agent and none has been "
            "validated by a human, so these numbers measure the harness, not the "
            "system. Re-run with --min-status validated after the human pass."
        ),
    }


def _render_summary(report: dict) -> str:
    agg = report["aggregate"]
    ks = report["config"]["ks"]
    lines = [
        "",
        f"label            {report['label']}",
        f"retriever        {report['config']['retriever']} / {report['config']['embedding_model']}",
        f"corpus           {report['corpus']['n_chunks']} chunks, "
        f"{report['corpus']['n_documents']} documents",
        f"gold             {report['gold']['n_scored']} scored, "
        f"{report['gold']['n_abstention_excluded']} abstention excluded, "
        f"{report['gold']['n_skipped']} skipped "
        f"(--min-status {report['config']['min_status']})",
        "",
        "aggregate (macro-average over scored queries)",
        "--------------------------------------------",
    ]
    for name in ("recall", "hit_rate", "ndcg"):
        cells = "  ".join(f"@{k} {agg[f'{name}@{k}']:.3f}" for k in ks)
        lines.append(f"{name:<10} {cells}")
    lines.append(f"{'mrr':<10} {agg['mrr']:.3f}")

    worst = sorted(report["per_query"], key=lambda q: q["metrics"]["mrr"])[:5]
    lines += ["", "worst 5 queries by MRR", "----------------------"]
    for q in worst:
        lines.append(
            f"  {q['id']}  mrr={q['metrics']['mrr']:.3f}  "
            f"hit@10={q['metrics'][f'hit_rate@{ks[-1]}']:.0f}  {q['capability']}"
        )

    if report["diagnostics"]["rows_with_unmatched_span"]:
        lines += [
            "",
            "gold rows whose span matched no chunk verbatim: "
            + ", ".join(report["diagnostics"]["rows_with_unmatched_span"]),
        ]
    lines += ["", report["caveat"], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.run_eval",
        description="Score the retriever against the gold set.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="path to the gold JSONL (default: eval/datasets/gold_seed.jsonl)",
    )
    parser.add_argument(
        "--min-status",
        choices=["draft", "validated"],
        default="draft",
        help="lowest gold-row status to score; 'validated' is the number that counts",
    )
    parser.add_argument(
        "--k",
        type=_parse_ks,
        default=DEFAULT_KS,
        help="comma-separated cutoffs (default: 1,3,5,10)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--label", default="dense", help="name for this configuration")
    parser.add_argument("--quiet", action="store_true", help="suppress the console summary")
    args = parser.parse_args(argv)

    gold_path = args.gold if args.gold is not None else None
    try:
        rows = load_gold(gold_path, min_status=args.min_status)
    except GoldSetError as exc:
        print(f"gold set error: {exc}", file=sys.stderr)
        return 2

    if not rows:
        print(
            f"no gold rows at status >= {args.min_status!r}. "
            "The gold set is still entirely draft — this is expected until the human "
            "validation pass; run with --min-status draft to exercise the harness.",
            file=sys.stderr,
        )
        return 3

    report = build_report(
        rows=rows,
        gold_path=Path(gold_path) if gold_path else DEFAULT_GOLD_PATH,
        min_status=args.min_status,
        ks=tuple(args.k),
        label=args.label,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not args.quiet:
            print(f"report -> {args.out}")

    if not args.quiet:
        print(_render_summary(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
