"""Calibration sheet construction and judge-human agreement.

Two things must hold for the sheet to be worth a human's afternoon: it has to
select the rows that discriminate rather than a random sample of easy
agreements, and re-running the generation suite must never destroy labels
already entered.

Kappa is tested with hand-computed values, including the case that matters most
for this project — a judge that always says "2" and therefore looks 100%
accurate while carrying no information.
"""

from __future__ import annotations

import json

import pytest

from eval.calibration import (
    agreement_report,
    build_calibration_sheet,
    cohens_kappa,
    load_sheet,
    write_sheet,
)


def item(gold_id, arm, *, faithfulness=2, bad_numbers=(), answer_type="extractive", recall=1.0):
    return {
        "gold_id": gold_id,
        "arm": arm,
        "question": "q",
        "context": "[1] ata — pagina 6\nO Copom decidiu manter a taxa em 13,75% a.a.",
        "answer": "a",
        "reference_answer": "r",
        "answer_type": answer_type,
        "capability": "single-hop lookup",
        "gold_doc_id": "doc",
        "retrieved_doc_ids": ["doc"],
        "deterministic": {
            "unsupported_numbers": list(bad_numbers),
            "has_unsupported_numbers": bool(bad_numbers),
            "numeric_recall": recall,
        },
        "judge": {
            "faithfulness": faithfulness,
            "faithfulness_reason": "",
            "answer_relevance": 2,
            "answer_relevance_reason": "",
            "judge_model": "llama3.1",
            "parse_error": "",
        },
    }


def test_judge_arithmetic_conflicts_are_selected_first():
    """The highest-information row a human can label: the judge called it
    faithful while it asserts a number found nowhere in the evidence."""
    items = [item(f"gold-{i:03d}", "qwen2.5:3b") for i in range(10)]
    items.append(item("gold-099", "qwen2.5:3b", faithfulness=2, bad_numbers=("13,75",)))
    sheet = build_calibration_sheet(items, size=3)
    assert sheet[0]["gold_id"] == "gold-099"


def test_negatives_outrank_ordinary_agreements():
    items = [item(f"gold-{i:03d}", "qwen2.5:3b") for i in range(10)]
    items.append(item("gold-051", "qwen2.5:3b", answer_type="abstention"))
    sheet = build_calibration_sheet(items, size=2)
    assert "gold-051" in {row["gold_id"] for row in sheet}


def test_negatives_cannot_crowd_out_the_answerable_path():
    """The first real sheet came out 21/30 abstention rows.

    7 negatives x 3 arms outrank every ordinary row, so they filled the sheet
    before a single answerable one got in — calibrating the judge on refusals,
    which is mostly not its job.
    """
    items = [item(f"neg-{i:03d}", "a", answer_type="abstention") for i in range(30)]
    items += [item(f"gold-{i:03d}", "a") for i in range(30)]
    sheet = build_calibration_sheet(items, size=30)
    negatives = sum(1 for row in sheet if row["answer_type"] == "abstention")
    assert negatives == 10
    assert len(sheet) == 30


def test_the_cap_relaxes_rather_than_shipping_a_short_sheet():
    """If there simply are not enough answerable rows, take the negatives."""
    items = [item(f"neg-{i:03d}", "a", answer_type="abstention") for i in range(30)]
    sheet = build_calibration_sheet(items, size=12)
    assert len(sheet) == 12


def test_no_single_arm_can_own_the_sheet():
    """Round-robin across arms, or the result describes one model."""
    items = [item(f"gold-{i:03d}", "qwen2.5:3b") for i in range(20)]
    items += [item(f"gold-{i:03d}", "llama3.1") for i in range(20)]
    sheet = build_calibration_sheet(items, size=10)
    counts = {arm: sum(1 for r in sheet if r["arm"] == arm) for arm in ("qwen2.5:3b", "llama3.1")}
    assert counts["qwen2.5:3b"] == counts["llama3.1"] == 5


def test_selection_is_deterministic():
    items = [item(f"gold-{i:03d}", "qwen2.5:3b") for i in range(20)]
    assert build_calibration_sheet(items, 8) == build_calibration_sheet(items, 8)


def test_human_columns_ship_empty():
    sheet = build_calibration_sheet([item("gold-001", "qwen2.5:3b")], size=1)
    assert sheet[0]["human_faithfulness"] is None
    assert sheet[0]["human_answer_relevance"] is None
    assert sheet[0]["judge_faithfulness"] == 2


def test_the_sheet_carries_the_evidence_the_human_has_to_judge_against():
    """Faithfulness is "is every claim supported by the passages". A human
    cannot answer that from a list of document ids, and neither can a second
    judge — so the passages travel with the row."""
    sheet = build_calibration_sheet([item("gold-001", "qwen2.5:3b")], size=1)
    assert "13,75" in sheet[0]["context"]


def test_rewriting_the_sheet_preserves_human_labels(tmp_path):
    """The failure that would silently destroy the human gate: re-running the
    generation suite must not wipe an afternoon of labelling."""
    path = tmp_path / "sheet.jsonl"
    original = build_calibration_sheet([item("gold-001", "qwen2.5:3b")], size=1)
    write_sheet(original, path)

    labelled = load_sheet(path)
    labelled[0]["human_faithfulness"] = 1
    labelled[0]["human_answer_relevance"] = 0
    labelled[0]["human_notes"] = "cita a ata errada"
    write_sheet(labelled, path)

    # A later run regenerates the row from scratch, with no human fields.
    regenerated = build_calibration_sheet([item("gold-001", "qwen2.5:3b")], size=1)
    write_sheet(regenerated, path)

    after = load_sheet(path)
    assert after[0]["human_faithfulness"] == 1
    assert after[0]["human_answer_relevance"] == 0
    assert after[0]["human_notes"] == "cita a ata errada"


def test_sheet_file_carries_the_protocol_comment(tmp_path):
    path = tmp_path / "sheet.jsonl"
    write_sheet(build_calibration_sheet([item("gold-001", "qwen2.5:3b")], size=1), path)
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "_comment" in first
    assert "HUMAN LABELS PENDING" in first["_comment"]
    assert load_sheet(path)[0]["gold_id"] == "gold-001"  # the comment is not a row


def test_kappa_is_one_for_perfect_agreement_with_spread():
    assert cohens_kappa([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]) == pytest.approx(1.0)


def test_kappa_is_zero_for_a_judge_that_always_says_two():
    """Raw agreement 100%, information content zero.

    This is the specific way an LLM judge fails on a corpus where most answers
    really are fine, and it is why the report shows kappa next to raw agreement
    rather than raw agreement alone.
    """
    assert cohens_kappa([2, 2, 2, 2], [2, 2, 2, 2]) == 0.0


def test_kappa_is_negative_when_the_judge_is_worse_than_chance():
    assert cohens_kappa([0, 0, 2, 2], [2, 2, 0, 0]) < 0


def test_kappa_matches_a_hand_computed_value():
    judge = [2, 2, 0, 0]
    human = [2, 0, 0, 0]
    # observed = 3/4; expected = (2/4)(1/4) + 0 + (2/4)(3/4) = 0.125 + 0.375 = 0.5
    assert cohens_kappa(judge, human) == pytest.approx((0.75 - 0.5) / 0.5)


def test_agreement_report_says_unknown_when_nothing_is_labelled():
    rows = build_calibration_sheet([item("gold-001", "qwen2.5:3b")], size=1)
    report = agreement_report(rows)
    faithfulness = report["criteria"]["faithfulness"]
    assert faithfulness["n_labelled"] == 0
    assert faithfulness["kappa"] is None
    assert "UNKNOWN" in faithfulness["note"]


def test_agreement_report_scores_the_labelled_subset():
    rows = build_calibration_sheet(
        [item("gold-001", "a"), item("gold-002", "a", faithfulness=0)], size=2
    )
    rows[0]["human_faithfulness"] = 2
    rows[0]["human_answer_relevance"] = 2
    rows[1]["human_faithfulness"] = 2
    rows[1]["human_answer_relevance"] = 2
    report = agreement_report(rows)
    assert report["criteria"]["faithfulness"]["n_labelled"] == 2
    assert report["criteria"]["faithfulness"]["raw_agreement"] == 0.5
    assert report["criteria"]["faithfulness"]["confusion"]["judge=0"]["human=2"] == 1


def test_judge_vs_judge_agreement_needs_no_human_labels():
    """The number available today: two judges that disagree with each other
    cannot both be trusted, and that is worth knowing before the human gate."""
    rows = build_calibration_sheet(
        [item("gold-001", "a"), item("gold-002", "a", faithfulness=0)], size=2
    )
    rows[0]["judge2_faithfulness"] = 2
    rows[1]["judge2_faithfulness"] = 2
    versus = agreement_report(rows)["criteria"]["faithfulness"]["judge_vs_judge2"]
    assert versus["n"] == 2
    assert versus["raw_agreement"] == 0.5
    assert versus["confusion"]["judge=0"]["judge2=2"] == 1
    # judge-vs-human stays unknown; the second judge is not a stand-in for a human
    assert agreement_report(rows)["criteria"]["faithfulness"]["kappa"] is None


def test_kappa_rejects_empty_input():
    with pytest.raises(ValueError, match="zero items"):
        cohens_kappa([], [])
