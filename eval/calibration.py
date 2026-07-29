"""The judge calibration sheet — and the agreement statistic that reads it back.

    uv run python -m eval.calibration          # report agreement, once labelled

`eval/datasets/judge_calibration_sheet.jsonl` is 30 judged answers with
`human_faithfulness` and `human_answer_relevance` left **null**. It is the second
human gate in this project, alongside gold-set validation, and it exists because
the alternative is reporting an LLM judge's scores as if they meant something.

Until it is filled, every judge number in `eval/reports/generation.json` is
labelled `agreement: null` — *unknown*, not *good*. Once filled, this module
turns it into Cohen's kappa plus a confusion matrix, and the honest sentence in
the writeup becomes a number instead of a promise.

## How the 30 are chosen

Not at random. A random sample of 168 answers over a system that mostly works
would be ~28 easy agreements and two interesting rows, and would establish
almost nothing about where the judge fails. The sample is stratified to
over-select disagreement and edge cases, in this priority order:

1. **Judge/arithmetic conflicts** — the judge called an answer faithful while it
   asserts a number found nowhere in the context, or called it unfaithful when
   every number checks out. These are where the judge is provably one thing or
   the other, and they are the highest-information rows a human can label.
2. **Negatives** — rows where refusal is correct. Abstention is the behaviour the
   guardrail suite depends on and the one the judge rubric handles most awkwardly.
3. **Judge parse failures and low scores** — the judge said something went wrong;
   a human should confirm something did.
4. **Spread across arms and capabilities** — so the result is not a statement
   about one model on one question type.

Negatives are **capped at a third of the sheet** (`NEGATIVE_SHARE`). Without the
cap the first version of this sheet came out 21/30 abstention rows: 7 negatives ×
3 arms outrank everything below them, and they filled the sheet before a single
answerable row got in. A calibration set that is 70% refusals calibrates the judge
on refusals, and the judge's job is mostly not that.

The selection is deterministic given a report, so re-running does not silently
reshuffle the sheet a human is halfway through labelling.

## The second judge

`--second-judge MODEL` re-scores the selected rows with a different local model
and stores the verdict alongside the first judge's. It costs one short run over
30 rows and it buys two things:

* **A number available today.** Judge-vs-judge kappa needs no human, so the
  report can say something concrete about judge reliability before the human gate
  clears. Two judges that disagree with each other cannot both be trusted, and
  that is worth knowing immediately.
* **It isolates self-preference.** In the first generation run the judge was
  `llama3.1`, and it graded `llama3.1`'s own answers on every row of that arm —
  rating them highest of the three. That result cannot distinguish "llama3.1
  writes better answers" from "llama3.1 likes its own answers", which is exactly
  the confound an LLM-judge setup is prone to. A second judge that did not write
  any of the answers is the control.

Judge-judge agreement is *not* a substitute for judge-human agreement. Two models
trained on overlapping data can agree with each other and both be wrong; that is
a well-known failure of using model consensus as ground truth. It is a cheap
necessary condition, not a sufficient one, and the human column stays.

## Kappa, not raw agreement

Cohen's kappa corrects for the agreement two raters would reach by chance. On a
3-point scale where most answers are genuinely a 2, raw agreement of 85% can
correspond to a kappa near zero — a judge that has learned to say "2" and nothing
else. Raw agreement is reported too, because the gap between the two is itself
the finding.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rag.config import REPO_ROOT

CALIBRATION_SHEET_PATH = REPO_ROOT / "eval" / "datasets" / "judge_calibration_sheet.jsonl"

SHEET_COMMENT = {
    "_comment": (
        "=== JUDGE CALIBRATION SHEET - HUMAN LABELS PENDING === "
        "Fill `human_faithfulness` and `human_answer_relevance` with 0, 1 or 2 using the "
        "rubric in generation/judge.py. Leave `judge_*` fields untouched - they are the "
        "prediction being tested. Then run `uv run python -m eval.calibration` for Cohen's "
        "kappa. Until then every judge number in eval/reports/generation.json is reported "
        "as agreement: null, meaning UNKNOWN."
    )
}

HUMAN_FIELDS = ("human_faithfulness", "human_answer_relevance")
JUDGE_FIELDS = ("judge_faithfulness", "judge_answer_relevance")
SCORES = (0, 1, 2)

#: Ceiling on the share of the sheet that may be abstention rows. See the module
#: docstring — without it, negatives crowd out the answerable path entirely.
NEGATIVE_SHARE = 1 / 3


def _priority(item: dict) -> tuple:
    """Sort key — lower sorts first, i.e. gets picked. See the module docstring.

    Deterministic: no randomness, ties broken by (arm, gold_id).
    """
    judge = item.get("judge") or {}
    deterministic = item["deterministic"]
    faithfulness = judge.get("faithfulness")

    parse_failed = judge.get("parse_error", "") != ""
    has_bad_numbers = deterministic["has_unsupported_numbers"]

    # 0 = the judge and the arithmetic contradict each other.
    conflict = (faithfulness == 2 and has_bad_numbers) or (
        faithfulness == 0 and not has_bad_numbers and deterministic["numeric_recall"] == 1.0
    )
    if conflict:
        rank = 0
    elif item["answer_type"] == "abstention":
        rank = 1
    elif parse_failed or (faithfulness is not None and faithfulness < 2):
        rank = 2
    else:
        rank = 3
    return (rank, item["arm"], item["gold_id"])


def _round_robin(items: Sequence[dict]) -> list[dict]:
    """Interleave by arm, preserving each arm's own order.

    Stops one arm owning the sheet just because it fails more often than the
    others — the result should describe the judge, not one generator.
    """
    by_arm: dict[str, list[dict]] = {}
    for item in items:
        by_arm.setdefault(item["arm"], []).append(item)

    interleaved: list[dict] = []
    while any(by_arm.values()):
        for arm in sorted(by_arm):
            if by_arm[arm]:
                interleaved.append(by_arm[arm].pop(0))
    return interleaved


def build_calibration_sheet(
    items: Sequence[dict],
    size: int = 30,
    negative_share: float = NEGATIVE_SHARE,
) -> list[dict]:
    """Pick `size` judged items, spread across arms, highest-information first."""
    by_rank: dict[int, list[dict]] = {}
    for item in sorted(items, key=_priority):
        by_rank.setdefault(_priority(item)[0], []).append(item)

    candidates: list[dict] = []
    for rank in sorted(by_rank):
        candidates.extend(_round_robin(by_rank[rank]))

    cap = max(1, round(size * negative_share))
    chosen: list[dict] = []
    deferred: list[dict] = []
    negatives = 0

    for item in candidates:
        if len(chosen) >= size:
            break
        if item["answer_type"] == "abstention":
            if negatives >= cap:
                deferred.append(item)
                continue
            negatives += 1
        chosen.append(item)

    # Only if there were not enough answerable rows to fill the sheet. Shipping a
    # short sheet would be worse than exceeding the cap.
    for item in deferred:
        if len(chosen) >= size:
            break
        chosen.append(item)

    return [_sheet_row(index, item) for index, item in enumerate(chosen[:size], start=1)]


def _sheet_row(index: int, item: dict) -> dict:
    judge = item.get("judge") or {}
    return {
        "item_id": f"cal-{index:03d}",
        "gold_id": item["gold_id"],
        "arm": item["arm"],
        "answer_type": item["answer_type"],
        "capability": item["capability"],
        "question": item["question"],
        # The retrieved passages, verbatim. Without them the human cannot judge
        # faithfulness at all — "is every claim supported by the evidence" is not
        # answerable from a list of document ids — and the second judge would be
        # scoring a different question from the first.
        "context": item["context"],
        "answer": item["answer"],
        "reference_answer": item["reference_answer"],
        "retrieved_doc_ids": item["retrieved_doc_ids"],
        # --- the prediction under test ---
        "judge_model": judge.get("judge_model"),
        "judge_faithfulness": judge.get("faithfulness"),
        "judge_faithfulness_reason": judge.get("faithfulness_reason", ""),
        "judge_answer_relevance": judge.get("answer_relevance"),
        "judge_answer_relevance_reason": judge.get("answer_relevance_reason", ""),
        "judge_parse_error": judge.get("parse_error", ""),
        # --- an independent second judge, filled by --second-judge ---
        "judge2_model": None,
        "judge2_faithfulness": None,
        "judge2_answer_relevance": None,
        # --- deterministic cross-check, for the human's benefit ---
        "unsupported_numbers": item["deterministic"]["unsupported_numbers"],
        "numeric_recall": item["deterministic"]["numeric_recall"],
        # --- THE HUMAN GATE: fill these in ---
        "human_faithfulness": None,
        "human_answer_relevance": None,
        "human_notes": "",
    }


def write_sheet(rows: list[dict], path: Path | None = None) -> Path:
    """Write the sheet, preserving any human labels already entered.

    Re-running the generation suite must never wipe work a human has done. Rows
    are matched by `gold_id` + `arm`, because `item_id` is positional and would
    drift as the selection changes.
    """
    path = path or CALIBRATION_SHEET_PATH
    existing = {}
    if path.exists():
        for row in load_sheet(path):
            existing[(row["gold_id"], row["arm"])] = row

    merged = []
    for row in rows:
        previous = existing.get((row["gold_id"], row["arm"]))
        if previous:
            for field in (*HUMAN_FIELDS, "human_notes"):
                if previous.get(field) not in (None, ""):
                    row[field] = previous[field]
        merged.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(SHEET_COMMENT, ensure_ascii=False) + "\n")
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_sheet(path: Path | None = None) -> list[dict]:
    path = path or CALIBRATION_SHEET_PATH
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "_comment" in record:
                continue
            rows.append(record)
    return rows


def cohens_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int] = SCORES) -> float:
    """Cohen's kappa for two raters over a fixed category set.

    Returns 0.0 when the raters agree exactly *and* only ever used one category:
    kappa is undefined there (expected agreement is 1), and 0.0 — "no better than
    chance" — is the honest reading of a rater that says the same thing every
    time.
    """
    if len(a) != len(b):
        raise ValueError("rating sequences must be the same length")
    if not a:
        raise ValueError("cannot compute kappa over zero items")

    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    expected = sum(
        (sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n) for c in categories
    )
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def confusion_matrix(
    rows: Sequence[int],
    columns: Sequence[int],
    categories: Sequence[int] = SCORES,
    row_label: str = "judge",
    column_label: str = "human",
) -> dict[str, dict[str, int]]:
    """Counts of (rater A score, rater B score). Axis labels are explicit because
    the same function compares judge-vs-human and judge-vs-judge, and a matrix
    labelled `human=` while holding a second model's scores is a lie."""
    matrix = {
        f"{row_label}={c}": {f"{column_label}={d}": 0 for d in categories} for c in categories
    }
    for a, b in zip(rows, columns, strict=True):
        matrix[f"{row_label}={a}"][f"{column_label}={b}"] += 1
    return matrix


def _pairwise(
    rows: Sequence[dict],
    field_a: str,
    field_b: str,
    label_a: str,
    label_b: str,
) -> dict | None:
    """Kappa, raw agreement and confusion for two rater columns, or None."""
    pairs = [
        (row[field_a], row[field_b])
        for row in rows
        if row.get(field_a) in SCORES and row.get(field_b) in SCORES
    ]
    if not pairs:
        return None
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    return {
        "n": len(pairs),
        "kappa": round(cohens_kappa(a, b), 4),
        "raw_agreement": round(sum(1 for x, y in pairs if x == y) / len(pairs), 4),
        "confusion": confusion_matrix(a, b, row_label=label_a, column_label=label_b),
    }


def agreement_report(rows: Sequence[dict]) -> dict:
    """Judge-vs-human and judge-vs-judge agreement, per criterion.

    Judge-vs-judge needs no human and is therefore reported immediately; it is a
    necessary condition for trusting the judge, not a sufficient one. Two models
    can agree with each other and both be wrong.
    """
    payload: dict = {"n_items": len(rows), "criteria": {}}
    for judge_field, human_field, second_field, name in (
        (JUDGE_FIELDS[0], HUMAN_FIELDS[0], "judge2_faithfulness", "faithfulness"),
        (JUDGE_FIELDS[1], HUMAN_FIELDS[1], "judge2_answer_relevance", "answer_relevance"),
    ):
        versus_human = _pairwise(rows, judge_field, human_field, "judge", "human")
        entry: dict = {
            "judge_vs_human": versus_human
            or {
                "n": 0,
                "kappa": None,
                "raw_agreement": None,
                "note": "no human labels yet — agreement is UNKNOWN, not good",
            },
            "judge_vs_judge2": _pairwise(rows, judge_field, second_field, "judge", "judge2"),
            "judge2_vs_human": _pairwise(rows, second_field, human_field, "judge2", "human"),
        }
        # Flat aliases so a caller does not have to know about the second judge.
        entry["n_labelled"] = entry["judge_vs_human"]["n"]
        entry["kappa"] = entry["judge_vs_human"]["kappa"]
        entry["raw_agreement"] = entry["judge_vs_human"]["raw_agreement"]
        if versus_human:
            entry["confusion"] = versus_human["confusion"]
        else:
            entry["note"] = entry["judge_vs_human"]["note"]
        payload["criteria"][name] = entry
    return payload


def add_second_judge(rows: list[dict], model: str) -> list[dict]:
    """Re-score every row with an independent judge, in place.

    Judges the identical (question, context, answer) triple the first judge saw —
    which is why the sheet carries `context` verbatim rather than a list of
    document ids. Anything less and the two judges would be answering different
    questions, and the agreement number would mean nothing.
    """
    from generation.judge import Judge

    judge = Judge(model=model)
    for row in rows:
        verdict = judge.judge(row["question"], row["context"], row["answer"])
        row["judge2_model"] = model
        row["judge2_faithfulness"] = verdict.faithfulness
        row["judge2_answer_relevance"] = verdict.answer_relevance
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.calibration",
        description="Report judge-human agreement over the calibration sheet.",
    )
    parser.add_argument("--sheet", type=Path, default=None)
    parser.add_argument(
        "--rebuild-from",
        type=Path,
        default=None,
        metavar="GENERATION_JSON",
        help=(
            "re-select the sheet from an existing generation report instead of "
            "re-running every model; human labels already entered are preserved"
        ),
    )
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument(
        "--second-judge",
        default=None,
        metavar="MODEL",
        help=(
            "re-score the sheet with an independent judge (e.g. qwen2.5:3b) and "
            "report judge-vs-judge agreement, which needs no human labels"
        ),
    )
    args = parser.parse_args(argv)

    path = args.sheet or CALIBRATION_SHEET_PATH

    if args.rebuild_from:
        report = json.loads(args.rebuild_from.read_text(encoding="utf-8"))
        items = [row for arm in report["arms"] for row in arm["per_row"]]
        write_sheet(build_calibration_sheet(items, size=args.size), path)
        print(f"rebuilt {args.size}-item sheet from {args.rebuild_from} -> {path}")

    if args.second_judge:
        if not path.exists():
            print(f"no sheet at {path} to re-judge", file=sys.stderr)
            return 2
        rows = add_second_judge(load_sheet(path), args.second_judge)
        write_sheet(rows, path)
        print(f"second judge {args.second_judge} scored {len(rows)} items -> {path}")
    if not path.exists():
        print(
            f"no calibration sheet at {path} — run `uv run python -m eval.run_generation` first",
            file=sys.stderr,
        )
        return 2

    rows = load_sheet(path)
    report = agreement_report(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    labelled = max(c["n_labelled"] for c in report["criteria"].values())
    if labelled == 0:
        print(
            f"\n{len(rows)} items, 0 labelled. Fill human_faithfulness and "
            "human_answer_relevance in the sheet, then re-run.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
