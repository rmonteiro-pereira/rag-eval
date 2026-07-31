"""Bootstrap CIs and the paired randomisation test.

The statistics decide which of this repo's claims survive, so they get tested
against cases whose answer is known by construction rather than only against the
committed report.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.significance import (
    PROBE_METRIC,
    bootstrap_ci,
    build_report,
    paired_bootstrap_delta,
    per_query_values,
    randomisation_p,
)


def arm(name: str, values: list[float], probe: list[bool] | None = None) -> dict:
    probe = probe if probe is not None else [v > 0.5 for v in values]
    rows = [
        {"id": f"q{i}", "metrics": {"mrr": v}, "rank1_doc_correct": p}
        for i, (v, p) in enumerate(zip(values, probe, strict=True))
    ]
    return {
        "name": name,
        "per_query": rows,
        "probes": {"meeting_disambiguation": {"ids": [r["id"] for r in rows]}},
    }


# --------------------------------------------------------------------------
# The intervals
# --------------------------------------------------------------------------


def test_a_constant_sample_has_a_zero_width_interval():
    """No variation, nothing to be uncertain about."""
    lo, hi = bootstrap_ci(np.full(40, 0.5))
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_the_interval_brackets_the_mean():
    values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 8)
    lo, hi = bootstrap_ci(values)
    assert lo < values.mean() < hi


def test_a_smaller_sample_gives_a_wider_interval():
    """The whole reason this module exists at n=49."""
    rng = np.random.default_rng(0)
    big = rng.random(400)
    small = big[:20]
    wide = bootstrap_ci(small)
    narrow = bootstrap_ci(big)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_the_interval_is_deterministic_for_a_given_seed():
    """A published interval that moves between runs is not a published number.

    Only determinism is asserted, not that two different seeds differ: on a
    symmetric two-value sample the percentile bootstrap legitimately lands on the
    same endpoints for many seeds, and an earlier version of this test failed for
    that reason — the test was wrong, not the code.
    """
    values = np.array([0.1, 0.9] * 25)
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)
    assert paired_bootstrap_delta(values, values + 0.05, seed=3) == paired_bootstrap_delta(
        values, values + 0.05, seed=3
    )
    assert randomisation_p(values, values + 0.05, seed=3) == randomisation_p(
        values, values + 0.05, seed=3
    )


# --------------------------------------------------------------------------
# The paired delta — the part that decides the ablation's claims
# --------------------------------------------------------------------------


def test_an_identical_pair_has_a_zero_delta_and_p_of_one():
    a = np.array([0.3, 0.7, 0.1, 0.9] * 10)
    delta, lo, hi = paired_bootstrap_delta(a, a.copy())
    assert (delta, lo, hi) == (0.0, 0.0, 0.0)
    assert randomisation_p(a, a.copy()) == pytest.approx(1.0)


def test_a_uniform_improvement_is_detected_with_an_interval_excluding_zero():
    a = np.array([0.2] * 40)
    b = a + 0.3
    delta, lo, hi = paired_bootstrap_delta(a, b)
    assert delta == pytest.approx(0.3)
    assert lo > 0, "a constant improvement must not include zero"
    assert randomisation_p(a, b) < 0.01


def test_pure_noise_produces_an_interval_that_includes_zero():
    """The case that disproved 'the reranker is actively harmful'."""
    rng = np.random.default_rng(11)
    a = rng.random(49)
    b = rng.random(49)
    _delta, lo, hi = paired_bootstrap_delta(a, b)
    assert lo < 0 < hi
    assert randomisation_p(a, b) > 0.05


def test_pairing_uses_the_same_resampled_queries_for_both_arms():
    """Unpaired resampling would inflate the interval and hide real effects.

    Constructed so the per-query difference is constant while the per-query
    scores vary wildly: paired, the delta has no variance at all.
    """
    a = np.array([0.0, 1.0] * 25)
    b = a + 0.1
    _delta, lo, hi = paired_bootstrap_delta(a, b)
    assert hi - lo == pytest.approx(0.0, abs=1e-9)


def test_the_p_value_can_never_be_reported_as_exactly_zero():
    """(count + 1) / (n + 1): 10,000 permutations cannot justify p = 0."""
    a = np.zeros(40)
    b = np.ones(40)
    p = randomisation_p(a, b, n=100)
    assert p > 0
    assert p == pytest.approx(1 / 101)


def test_mismatched_arm_lengths_raise_rather_than_compare_different_questions():
    a, b = np.zeros(10), np.zeros(9)
    with pytest.raises(ValueError, match="different query counts"):
        paired_bootstrap_delta(a, b)
    with pytest.raises(ValueError, match="different query counts"):
        randomisation_p(a, b)


# --------------------------------------------------------------------------
# Reading the report
# --------------------------------------------------------------------------


def test_the_probe_metric_is_read_from_its_own_query_subset():
    """`meeting_disambiguation` covers 41 of 49 rows; using all 49 is a different
    quantity that happens to share a name. The first version of this module made
    exactly that mistake and reported -0.102 where the ablation says -0.146."""
    a = arm("x", [0.1] * 4, probe=[True, False, True, False])
    a["probes"]["meeting_disambiguation"]["ids"] = ["q0", "q1"]
    subset = per_query_values(a, PROBE_METRIC, {"q0", "q1"})
    every = per_query_values(a, PROBE_METRIC, None)
    assert len(subset) == 2 and len(every) == 4
    assert subset.mean() == pytest.approx(0.5)


def test_build_report_flags_a_contrast_naming_an_absent_arm():
    with pytest.raises(KeyError):
        build_report({"arms": [arm("dense", [0.1] * 5)]})
