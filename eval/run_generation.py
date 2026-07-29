"""M3 generation suite: answer every gold question with each backend, score it,
and write the calibration sheet the human pass needs.

    uv run python -m eval.run_generation --out eval/reports/generation.json

Three arms by default — `extractive`, `qwen2.5:3b`, `llama3.1` — over all 56 gold
rows including the seven negatives.

**Retrieval runs once per question and the passages are shared by every arm.**
That is the only way the comparison means anything: otherwise a difference
between two generators is confounded with a difference between two retrievals,
and this project's whole M4 finding is that retrieval differences on this corpus
are enormous.

The retriever used is the M4 winner (`hybrid+rerank+metadata`), so the numbers
here describe generation *given good retrieval*. Generation quality on top of the
naive baseline would be a different — and much worse — measurement, and it is not
the one a reader of this report wants.

`extractive` is included as the groundedness floor rather than as a competitor.
It returns the retrieved passages verbatim, so it cannot hallucinate, cannot be
irrelevant if retrieval was right, and cannot answer a question in a sentence. It
is what the generative arms have to justify themselves against.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from eval.calibration import (
    CALIBRATION_SHEET_PATH,
    agreement_report,
    build_calibration_sheet,
    load_sheet,
    write_sheet,
)
from eval.gold import DEFAULT_GOLD_PATH, GoldRow, GoldSetError, load_gold, status_counts
from eval.metrics.generation import aggregate_generation, score_generation
from eval.run_eval import DRAFT_CAVEAT
from generation.answer import generate_answer
from generation.judge import Judge, aggregate_judgements
from generation.llm import ExtractiveLLM, OllamaLLM
from generation.prompt import PROMPT_VERSION, format_context
from rag.config import REPO_ROOT, settings
from retrieval.configs import RetrievalContext, build_retriever

SCHEMA_VERSION = 1


def _build_llm(arm: str):
    return ExtractiveLLM() if arm == "extractive" else OllamaLLM(model=arm)


def _retrieve_once(rows: list[GoldRow], context: RetrievalContext, top_k: int) -> dict[str, list]:
    retriever = build_retriever(settings.retrieval_config, context, top_k=top_k)
    return {row.id: retriever.retrieve(row.question, top_k=top_k) for row in rows}


def run_generation(
    rows: list[GoldRow],
    gold_path: Path,
    min_status: str,
    arms: list[str],
    top_k: int,
    judge_model: str | None,
    calibration_size: int,
) -> tuple[dict, list[dict]]:
    context = RetrievalContext.build()

    print(f"  retrieving passages for {len(rows)} questions ...", file=sys.stderr, flush=True)
    passages_by_row = _retrieve_once(rows, context, top_k)

    judge = Judge(model=judge_model) if judge_model else None
    arm_reports: list[dict] = []
    all_items: list[dict] = []

    for arm in arms:
        print(f"  arm {arm} ...", file=sys.stderr, flush=True)
        llm = _build_llm(arm)
        scores, is_abstention, judgements, per_row = [], [], [], []

        for row in rows:
            passages = passages_by_row[row.id]
            context_text = format_context(passages)

            started = time.perf_counter()
            answer = generate_answer(row.question, passages, llm)
            latency_ms = (time.perf_counter() - started) * 1000

            score = score_generation(
                generated=answer.text,
                reference=row.answer,
                context=context_text,
                question=row.question,
                retrieved_doc_ids=[p.doc_id for p in passages],
                gold_doc_id=row.source_doc_id,
            )
            judgement = (
                judge.judge(row.question, context_text, answer.text) if judge is not None else None
            )

            scores.append(score)
            is_abstention.append(row.is_abstention)
            if judgement is not None:
                judgements.append(judgement)

            item = {
                "gold_id": row.id,
                "arm": arm,
                "question": row.question,
                # Kept verbatim: it is the evidence every downstream judgement is
                # made against, human or model, and a report you cannot re-judge
                # from is a report you have to take on faith.
                "context": context_text,
                "answer": answer.text,
                "reference_answer": row.answer,
                "answer_type": row.answer_type,
                "capability": row.capability,
                "gold_doc_id": row.source_doc_id,
                "retrieved_doc_ids": [p.doc_id for p in passages],
                "latency_ms": round(latency_ms, 1),
                "usage": answer.usage,
                "deterministic": score.to_json(),
                "judge": judgement.to_json() if judgement is not None else None,
                # A judge grading its own arm's output is a known bias direction.
                # Flagged per row rather than silently excluded, so the reader can
                # decide what to do with those rows.
                "judge_is_generator": bool(judge is not None and judge.model == arm),
            }
            per_row.append(item)
            all_items.append(item)

        arm_reports.append(
            {
                "arm": arm,
                "backend": llm.backend,
                "deterministic": aggregate_generation(scores, is_abstention),
                "judge": aggregate_judgements(judgements) if judgements else None,
                "latency": {
                    "median_ms": round(
                        sorted(item["latency_ms"] for item in per_row)[len(per_row) // 2], 1
                    ),
                },
                "per_row": per_row,
            }
        )

    sheet = build_calibration_sheet(all_items, size=calibration_size)
    # Merge with whatever is already on disk *before* measuring agreement, so a
    # partially-labelled sheet reports the agreement it has rather than zero.
    write_sheet(sheet)
    sheet = load_sheet()
    agreement = agreement_report(sheet)
    labelled = max(c["n_labelled"] for c in agreement["criteria"].values())

    self_judged = {
        arm["arm"]
        for arm in arm_reports
        if any(row["judge_is_generator"] for row in arm["per_row"])
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "setup": {
            "retriever": settings.retrieval_config,
            "prompt_version": PROMPT_VERSION,
            "top_k": top_k,
            "arms": arms,
            "judge_model": judge_model,
            "min_status": min_status,
            "gold_path": gold_path.relative_to(REPO_ROOT).as_posix(),
            "note": (
                "Retrieval runs once per question and is shared by every arm, so "
                "differences between arms are generation differences only. The "
                "retriever is the M4 winner, so these are generation numbers "
                "*given good retrieval*."
            ),
        },
        "gold": {
            "status_counts_in_file": status_counts(gold_path),
            "n_rows": len(rows),
            "n_negative": sum(row.is_abstention for row in rows),
        },
        "arms": arm_reports,
        "calibration": {
            "sheet_path": CALIBRATION_SHEET_PATH.relative_to(REPO_ROOT).as_posix(),
            "n_items": len(sheet),
            "human_labels_filled": labelled,
            "agreement": agreement,
            "note": (
                "Judge-vs-HUMAN agreement is UNKNOWN while human_labels_filled is 0 — "
                "unknown, not good. Judge-vs-JUDGE agreement needs no human and is "
                "under agreement.criteria.*.judge_vs_judge2; add it with "
                "`eval.calibration --second-judge <model>`. Where two local judges "
                "disagree with each other, neither can be trusted, and the "
                "deterministic metrics are the ones to believe."
            ),
        },
        "caveat": DRAFT_CAVEAT,
    }
    if self_judged:
        report["setup"]["judge_self_preference_warning"] = (
            f"The judge ({judge_model}) also wrote the answers for arm(s) "
            f"{sorted(self_judged)}, flagged per row as judge_is_generator. "
            "Self-preference cannot be separated from quality in that comparison."
        )
    return report, sheet


def _render_summary(report: dict) -> str:
    lines = [
        "",
        f"retriever {report['setup']['retriever']}  |  judge {report['setup']['judge_model']}",
        f"gold      {report['gold']['n_rows']} rows "
        f"({report['gold']['n_negative']} negatives)",
        "",
        f"{'arm':<16} {'num-recall':>11} {'grounded':>9} {'halluc-num':>11} "
        f"{'cite-ok':>8} {'abstain-ok':>11} {'false-ref':>10} {'med ms':>8}",
        "-" * 92,
    ]

    def cell(value, width=11, digits=3):
        return f"{'  n/a':>{width}}" if value is None else f"{value:>{width}.{digits}f}"

    for arm in report["arms"]:
        d = arm["deterministic"]
        lines.append(
            f"{arm['arm']:<16} {cell(d['numeric_recall'])} "
            f"{cell(d['lexical_groundedness'], 9)} {cell(d['hallucinated_number_rate'])} "
            f"{cell(d['citation_correctness'], 8)} {cell(d['abstention_correctness'])} "
            f"{cell(d['false_refusal_rate'], 10)} {arm['latency']['median_ms']:>8.0f}"
        )

    lines += [
        "",
        f"{'arm':<16} {'judge-faith':>12} {'judge-relev':>12} {'faith=2':>9} "
        f"{'relev=2':>9} {'parse-fail':>11}",
        "-" * 74,
    ]
    for arm in report["arms"]:
        j = arm["judge"]
        if not j:
            continue
        lines.append(
            f"{arm['arm']:<16} {cell(j['faithfulness_mean'], 12)} "
            f"{cell(j['answer_relevance_mean'], 12)} {cell(j['faithfulness_at_2'], 9)} "
            f"{cell(j['answer_relevance_at_2'], 9)} {cell(j['parse_failure_rate'], 11)}"
        )

    lines += ["", "judge reliability (needs no human labels)", "-" * 42]
    for name, criterion in report["calibration"]["agreement"]["criteria"].items():
        versus = criterion["judge_vs_judge2"]
        if versus:
            lines.append(
                f"  {name:<18} judge vs judge2: kappa {versus['kappa']:+.3f}, "
                f"raw {versus['raw_agreement']:.3f} (n={versus['n']})"
            )
        else:
            lines.append(f"  {name:<18} no second judge — run --second-judge <model>")

    warning = report["setup"].get("judge_self_preference_warning")
    lines += [
        "",
        f"calibration sheet: {report['calibration']['n_items']} items, "
        f"{report['calibration']['human_labels_filled']} labelled -> "
        f"{report['calibration']['sheet_path']}",
        *(["", f"WARNING: {warning}"] if warning else []),
        "",
        report["calibration"]["note"],
        "",
        report["caveat"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.run_generation",
        description="Score generation backends on the gold set and build the judge sheet.",
    )
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--min-status", choices=["draft", "validated"], default="draft")
    parser.add_argument(
        "--arms",
        default=settings.generation_arms,
        help="comma-separated backends (default: %(default)s)",
    )
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument(
        "--judge-model",
        default=settings.judge_model,
        help="Ollama model used as judge; pass '' to skip judging",
    )
    parser.add_argument("--calibration-size", type=int, default=30)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "eval" / "reports" / "generation.json",
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

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    for arm in arms:
        if arm != "extractive" and not OllamaLLM.is_available(arm):
            print(
                f"arm {arm!r} is not available in Ollama — `ollama pull {arm}` first, "
                "or drop it from --arms",
                file=sys.stderr,
            )
            return 4

    report, sheet = run_generation(
        rows=rows,
        gold_path=Path(args.gold) if args.gold else DEFAULT_GOLD_PATH,
        min_status=args.min_status,
        arms=arms,
        top_k=args.top_k,
        judge_model=args.judge_model or None,
        calibration_size=args.calibration_size,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"report -> {args.out}")
        print(_render_summary(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
