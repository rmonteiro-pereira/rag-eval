"""The CI gate.

A gate is only worth having if it fails, so both directions are asserted: exit 0
on the real committed report, exit 1 on a fixture where the meeting resolver has
been deliberately broken. A test that only checked the passing case would go
green forever the day someone inverted a comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.regression_gate import (
    AGGREGATE_THRESHOLDS,
    DEFAULT_ARM,
    GateError,
    compare,
    extract,
)
from eval.regression_gate import main as gate_main
from rag.config import REPO_ROOT

FIXTURES = Path(__file__).parent / "fixtures"
BASELINE = FIXTURES / "gate_baseline.json"
DEGRADED = FIXTURES / "gate_degraded.json"
REAL_REPORT = REPO_ROOT / "eval" / "reports" / "ablation.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- the two directions the brief requires ---------------------------------


def test_the_gate_passes_on_the_real_committed_report():
    """Exit 0 against the report the project actually ships."""
    assert REAL_REPORT.exists(), "eval/reports/ablation.json is a committed artifact"
    assert (
        gate_main(["--baseline", str(REAL_REPORT), "--candidate", str(REAL_REPORT), "--quiet"]) == 0
    )


def test_the_gate_fails_on_the_degraded_fixture():
    """Exit 1. The fixture simulates the meeting resolver breaking."""
    assert gate_main(["--baseline", str(BASELINE), "--candidate", str(DEGRADED), "--quiet"]) == 1


def test_the_gate_passes_when_the_candidate_equals_the_baseline():
    assert gate_main(["--baseline", str(BASELINE), "--candidate", str(BASELINE), "--quiet"]) == 0


# --- what it catches, specifically ------------------------------------------


def test_the_probe_collapse_alone_is_enough_to_fail():
    """The point of gating on probes rather than only on averages.

    Here the aggregate metrics are untouched — a dashboard would show nothing
    wrong — and only rank-1 meeting accuracy has collapsed. That is exactly what
    breaking the metadata filter looks like on a subset of queries, and the gate
    has to catch it.
    """
    baseline = load(BASELINE)
    candidate = json.loads(json.dumps(baseline))
    candidate["arms"][0]["probes"]["meeting_disambiguation"]["rank1_doc_accuracy"] = 0.10
    checks = compare(baseline, candidate)
    failed = [c for c in checks if not c.passed]
    assert [c.name for c in failed] == ["probe:meeting_disambiguation"]


def test_a_drop_inside_tolerance_passes():
    baseline = load(BASELINE)
    candidate = json.loads(json.dumps(baseline))
    candidate["arms"][0]["aggregate"]["mrr"] -= AGGREGATE_THRESHOLDS["mrr"] / 2
    assert all(c.passed for c in compare(baseline, candidate))


def test_a_drop_just_past_tolerance_fails():
    baseline = load(BASELINE)
    candidate = json.loads(json.dumps(baseline))
    candidate["arms"][0]["aggregate"]["mrr"] -= AGGREGATE_THRESHOLDS["mrr"] * 1.5
    failed = [c.name for c in compare(baseline, candidate) if not c.passed]
    assert failed == ["mrr"]


def test_an_improvement_is_never_a_regression():
    baseline = load(BASELINE)
    candidate = json.loads(json.dumps(baseline))
    candidate["arms"][0]["aggregate"]["mrr"] += 0.3
    assert all(c.passed for c in compare(baseline, candidate))


# --- refusing to pass vacuously ---------------------------------------------


def test_a_missing_report_exits_2_not_0():
    """A gate that passes because it could not find the numbers is worse than
    no gate. Exit 2 is 'could not compare', which CI must also treat as failure."""
    assert gate_main(["--baseline", "nope.json", "--candidate", str(BASELINE)]) == 2


def test_a_missing_metric_is_an_error_rather_than_a_skipped_check():
    baseline = load(BASELINE)
    candidate = json.loads(json.dumps(baseline))
    del candidate["arms"][0]["aggregate"]["mrr"]
    with pytest.raises(GateError, match="missing metric"):
        compare(baseline, candidate)


def test_a_missing_probe_group_is_an_error():
    baseline = load(BASELINE)
    candidate = json.loads(json.dumps(baseline))
    del candidate["arms"][0]["probes"]["meeting_disambiguation"]
    with pytest.raises(GateError, match="missing probe group"):
        compare(baseline, candidate)


def test_an_unknown_arm_is_an_error():
    with pytest.raises(GateError, match="not in report"):
        extract(load(BASELINE), arm="does-not-exist")


def test_a_report_of_the_wrong_shape_is_an_error():
    with pytest.raises(GateError, match="neither"):
        extract({"hello": "world"})


def test_no_comparable_metrics_refuses_to_pass():
    empty = {"arms": [{"name": DEFAULT_ARM, "aggregate": {}, "probes": {}}]}
    with pytest.raises(GateError, match="vacuously"):
        compare(empty, empty)


# --- schema tolerance --------------------------------------------------------


def test_a_plain_run_eval_report_is_accepted_without_probes():
    report = {"aggregate": {"mrr": 0.5, "recall@5": 0.4}}
    aggregate, probes = extract(report)
    assert aggregate["mrr"] == 0.5
    assert probes == {}


def test_the_degraded_fixture_says_what_it_is():
    """The fixture is committed; it has to be obvious it is not a real result."""
    note = load(DEGRADED)["_fixture_note"]
    assert "DELIBERATELY DEGRADED" in note
