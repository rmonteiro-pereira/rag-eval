"""Every number the docs publish, pinned to the report field it comes from.

This project's whole claim is that its figures mean something, which makes a
stale README a correctness bug rather than a typo. The failure mode is specific
and it has already happened twice here: a report gets regenerated, the numbers
move slightly, and the prose keeps quoting the old ones. Nothing catches it,
because nothing was comparing them.

So the mapping lives in a test. Regenerate a report and change a number, and
these fail with the field name and both values — the docs then get updated
because CI says so, not because someone remembered.

What this does NOT do is parse the README. Scraping prose for decimals is
brittle in the direction that matters (it goes green when the regex stops
matching). The table below is a second, independent copy of what the docs claim,
maintained by hand — deliberately, because a human writing the expected value
down is the check.

The counterpart guarantee — that the reports themselves reproduce — is
`docs/REPRODUCE.md`, where the whole ablation is regenerated from an empty index
and gated to +-0.0000.
"""

from __future__ import annotations

import json
import re

import pytest

from eval.metrics.generation import unsupported_numbers
from rag.config import REPO_ROOT

REPORTS = REPO_ROOT / "eval" / "reports"


def _load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ablation() -> dict:
    return _load("ablation.json")


@pytest.fixture(scope="module")
def adversarial() -> dict:
    return _load("adversarial.json")


@pytest.fixture(scope="module")
def generation() -> dict:
    return _load("generation.json")


def _arm(report: dict, name: str) -> dict:
    for arm in report["arms"]:
        if arm.get("name") == name or arm.get("arm") == name:
            return arm
    present = [a.get("name", a.get("arm")) for a in report["arms"]]
    raise AssertionError(f"no arm {name!r} in report; arms are {present}")


# --------------------------------------------------------------------------
# Retrieval — the headline table and the three findings
# --------------------------------------------------------------------------

RETRIEVAL = [
    # (arm, path-into-arm, published value)
    ("dense", ("aggregate", "mrr"), 0.191),
    ("dense", ("aggregate", "hit_rate@5"), 0.367),
    ("dense", ("aggregate", "ndcg@10"), 0.161),
    ("dense", ("aggregate", "recall@5"), 0.149),
    ("hybrid+rerank+metadata", ("aggregate", "mrr"), 0.741),
    ("hybrid+rerank+metadata", ("aggregate", "hit_rate@5"), 0.959),
    ("hybrid+rerank+metadata", ("aggregate", "ndcg@10"), 0.623),
    ("hybrid+rerank+metadata", ("aggregate", "recall@5"), 0.467),
    ("hybrid+rerank+metadata", ("delta_vs_dense", "mrr"), 0.550),
    ("hybrid+rerank+metadata", ("delta_vs_dense", "hit_rate@5"), 0.592),
    ("hybrid+rerank+metadata", ("delta_vs_dense", "ndcg@10"), 0.462),
    ("hybrid+rerank+metadata", ("delta_vs_dense", "recall@5"), 0.318),
    ("bm25", ("aggregate", "mrr"), 0.382),
    ("hybrid", ("aggregate", "mrr"), 0.381),
    ("dense+metadata", ("aggregate", "mrr"), 0.689),
    ("hybrid+metadata", ("aggregate", "mrr"), 0.736),
    ("hybrid+rerank", ("aggregate", "mrr"), 0.342),
]

PROBES = [
    ("dense", "meeting_disambiguation", 0.098),
    ("hybrid+rerank+metadata", "meeting_disambiguation", 1.000),
    ("bm25", "meeting_disambiguation", 0.927),
    ("hybrid", "meeting_disambiguation", 0.341),
    ("hybrid+rerank+metadata", "reverse_lookup", 0.500),
]


@pytest.mark.parametrize("arm_name, path, published", RETRIEVAL)
def test_retrieval_numbers_match_the_committed_ablation(ablation, arm_name, path, published):
    value = _arm(ablation, arm_name)
    for key in path:
        value = value[key]
    assert value == pytest.approx(published, abs=5e-4), f"{arm_name}.{'.'.join(path)}"


@pytest.mark.parametrize("arm_name, probe, published", PROBES)
def test_probe_numbers_match_the_committed_ablation(ablation, arm_name, probe, published):
    """The wrong-meeting defect is the project's headline; its probe is pinned."""
    value = _arm(ablation, probe_arm := arm_name)["probes"][probe]["rank1_doc_accuracy"]
    assert value == pytest.approx(published, abs=5e-4), f"{probe_arm}.probes.{probe}"


def test_the_meeting_filter_has_no_false_positives(ablation):
    """41/41 hint precision is what licenses a hard pre-filter over a soft boost.

    If this ever drops below 1.0, ADR 0004's decision reverses — the README, the
    ADR and the ablation all rest on this single number.
    """
    hints = ablation["hint_diagnostics"]
    assert hints["hint_present"] == 41
    assert hints["hint_resolved_and_correct"] == 41
    assert hints["precision_when_resolved"] == pytest.approx(1.0)
    assert hints["resolved_but_wrong"] == []


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------

GUARDRAILS = [
    ("injection_success_rate", 0.0833),
    ("injection_success_rate_ungoverned", 0.1667),
    ("pii_output_leak_rate", 0.0),
    ("pii_output_leak_rate_ungoverned", 1.0),
    ("pii_input_leak_rate", 0.0),
    ("pii_false_positive_rate", 0.0),
    ("abstention_correctness", 1.0),
    ("false_refusal_rate", 0.0408),
    ("acl_restricted_chunks_retrieved_by_uncleared_user", 0),
]


@pytest.mark.parametrize("field, published", GUARDRAILS)
def test_guardrail_numbers_match_the_committed_report(adversarial, field, published):
    assert adversarial["headline"][field] == pytest.approx(published, abs=5e-4), field


def test_the_indirect_injection_surface_is_where_the_guardrail_earns_its_place(adversarial):
    """0.0% governed vs 16.7% ungoverned — the RAG-specific attack."""
    gov = adversarial["injection"]["governed"]["attack_success_rate_indirect"]
    ungov = adversarial["injection"]["ungoverned"]["attack_success_rate_indirect"]
    assert gov == pytest.approx(0.0)
    assert ungov == pytest.approx(0.1667, abs=5e-4)


def test_the_two_attacks_that_succeed_are_still_the_two_that_are_named(adversarial):
    """`docs/governance.md` names inj-012 and inj-023 and explains each.

    If the set changes, the prose is describing attacks that no longer succeed —
    which reads as a cover-up whether or not it is one.
    """
    assert adversarial["injection"]["governed"]["succeeded"] == ["inj-012", "inj-023"]


# --------------------------------------------------------------------------
# Generation, including the corrected fabricated-answer rate
# --------------------------------------------------------------------------

GENERATION = [
    ("extractive", "numeric_recall", 0.913),
    ("extractive", "lexical_groundedness", 0.988),
    ("extractive", "abstention_correctness", 0.000),
    ("qwen2.5:3b", "numeric_recall", 0.777),
    ("qwen2.5:3b", "lexical_groundedness", 0.838),
    ("qwen2.5:3b", "false_refusal_rate", 0.082),
    ("llama3.1", "numeric_recall", 0.837),
    ("llama3.1", "lexical_groundedness", 0.887),
    ("llama3.1", "false_refusal_rate", 0.041),
]


@pytest.mark.parametrize("arm_name, field, published", GENERATION)
def test_generation_numbers_match_the_committed_report(generation, arm_name, field, published):
    assert _arm(generation, arm_name)["deterministic"][field] == pytest.approx(
        published, abs=5e-4
    ), f"{arm_name}.{field}"


def test_the_fabricated_answer_rate_is_zero_on_every_arm(generation):
    """ "Zero hallucinated numbers across all 168 answers" — the README's claim."""
    rates = {a["arm"]: a["deterministic"]["hallucinated_number_rate"] for a in generation["arms"]}
    assert rates == {"extractive": 0.0, "qwen2.5:3b": 0.0, "llama3.1": 0.0}
    assert sum(len(a["per_row"]) for a in generation["arms"]) == 168


def test_the_fabricated_answer_rate_re_derives_from_the_stored_evidence(generation):
    """Recompute it rather than trusting the aggregate the same run wrote.

    The report stores each answer with the exact context it was judged against,
    so the headline claim is checkable without a GPU, an LLM, or a re-run. This
    is the check that caught the citation-marker false positive: `[2, 19]` was
    being read as the decimal `2,19` and scored as a fabricated rate.
    """
    for arm in generation["arms"]:
        flags = [
            bool(unsupported_numbers(row["answer"], row["context"], row["question"]))
            for row in arm["per_row"]
        ]
        recomputed = sum(flags) / len(flags)
        assert recomputed == pytest.approx(arm["deterministic"]["hallucinated_number_rate"]), (
            f"{arm['arm']}: aggregate and per-row evidence disagree"
        )


def test_the_judge_agreement_lives_in_the_report_not_only_in_stdout(generation):
    """kappa 0.109 is quoted in the README; it has to be checkable from a file.

    It was not, until `eval.calibration` learned to fold the agreement block back
    into the report — the numbers existed only in a CLI's output while the docs
    quoted them.
    """
    criteria = generation["calibration"]["agreement"]["criteria"]
    faith = criteria["faithfulness"]["judge_vs_judge2"]
    assert faith is not None, "run `eval.calibration --second-judge qwen2.5:3b`"
    assert faith["kappa"] == pytest.approx(0.109, abs=5e-4)
    assert faith["raw_agreement"] == pytest.approx(0.44, abs=5e-4)


def test_both_human_gates_are_still_open(generation):
    """The two numbers that must NOT improve without a human.

    `agreement: null` against a human, and zero labels filled. An agent closing
    either of these silently is the most damaging change possible in this repo.
    """
    calibration = generation["calibration"]
    assert calibration["human_labels_filled"] == 0
    assert calibration["n_items"] == 30
    for criterion in calibration["agreement"]["criteria"].values():
        assert criterion["judge_vs_human"]["kappa"] is None


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def test_the_m1_baseline_still_says_what_the_readme_says_it_says():
    baseline = _load("baseline_dense.json")["aggregate"]
    assert baseline["mrr"] == pytest.approx(0.191, abs=5e-4)
    assert baseline["hit_rate@10"] == pytest.approx(0.531, abs=5e-4)
    assert baseline["hit_rate@5"] == pytest.approx(0.367, abs=5e-4)


# --------------------------------------------------------------------------
# Claims that are not metrics, and were therefore not covered above. A sibling
# project shipped a README citing "219 passing" to a file that contained no
# count at all; the real figure was 201 passed / 20 skipped. A test count is a
# published number like any other and gets the same treatment.
# --------------------------------------------------------------------------


def test_the_documented_test_count_is_the_real_one():
    """Collect the suite and compare, rather than trusting a number in prose.

    This is self-referential — the count includes this test — which is fine, and
    is why the docs quote the total. It caught a real drift: README and
    docs/mutation.md both said 380 after four tests had been added.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
    ).stdout
    collected = int(re.search(r"(\d+) tests? collected", out).group(1))

    for doc_name in ("README.md", "docs/mutation.md"):
        doc = (REPO_ROOT / doc_name).read_text(encoding="utf-8")
        for m in re.finditer(r"(\d{3}) (?:tests pass|passing tests|tests against)", doc):
            assert int(m.group(1)) == collected, (
                f"{doc_name} claims {m.group(1)} tests; pytest collects {collected}"
            )


def test_the_mutation_numbers_the_docs_quote_are_the_artifacts():
    """The mutation score and survivor count, pinned like every other figure.

    This drifted once already: the README kept saying "73.4%" and "all 124
    survivors" after a re-run had moved the committed artifact to 74.9% and
    132. Every artifact-side check passed, because none of them covered the
    numbers the mutation section publishes. Same treatment as the rest now —
    a hand-maintained expected value compared against the committed report.
    """
    mutation = _load("mutation.json")
    assert mutation["score"] == 74.9
    assert mutation["score_including_uncovered"] == 69.6
    assert mutation["killed"] == 394
    assert mutation["survived"] == 132
    assert mutation["no_tests"] == 40
    # internal consistency — the artifact cannot disagree with itself
    assert (
        mutation["killed"] + mutation["survived"] + mutation["no_tests"]
        == mutation["total_mutants"]
    )
    assert len(mutation["survivors"]) == mutation["survived"]
    conventional = round(100 * mutation["killed"] / (mutation["killed"] + mutation["survived"]), 1)
    assert mutation["score"] == conventional


def test_the_readme_mutation_claims_match_the_mutation_artifact():
    """The one deliberate exception to this file's no-prose-scraping rule.

    The drift above happened in README *prose*, which no artifact diff can
    catch — CI regenerates `mutation.json` and `mutation-survivors.md` and
    diffs them, but nothing reads the sentences that quote them. So these two
    claims are scraped, with the brittleness guarded in the direction that
    matters: if the regex stops matching, the test FAILS rather than going
    quietly green.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    survivor_claims = re.findall(r"[Aa]ll (\d+) survivors", readme)
    assert survivor_claims, "the README no longer states how many survivors are listed"
    mutation = _load("mutation.json")
    for claim in survivor_claims:
        assert int(claim) == mutation["survived"], (
            f"README says 'all {claim} survivors'; the artifact says {mutation['survived']}"
        )

    score_claims = re.findall(r"(\d{2}\.\d)% mutation score", readme)
    score_claims += re.findall(r"mutation score is \*\*(\d{2}\.\d)%\*\*", readme)
    assert score_claims, "the README no longer quotes the mutation score"
    for claim in score_claims:
        assert float(claim) == mutation["score"], (
            f"README quotes a {claim}% mutation score; the artifact says {mutation['score']}%"
        )


def test_the_survivor_inventory_totals_match_the_artifact():
    """`docs/mutation-survivors.md` header vs `eval/reports/mutation.json`.

    CI regenerates the inventory and diffs it, so on a machine that ran mutmut
    these cannot disagree — but that job only exists on Linux runners. This
    check runs everywhere pytest does, including Windows, where mutmut cannot.
    """
    doc = (REPO_ROOT / "docs" / "mutation-survivors.md").read_text(encoding="utf-8")
    header = re.search(
        r"\*\*(\d+) killed · (\d+) survived · (\d+) uncovered\*\* of (\d+) mutants", doc
    )
    assert header, "the survivors doc no longer states its totals"
    mutation = _load("mutation.json")
    assert [int(g) for g in header.groups()] == [
        mutation["killed"],
        mutation["survived"],
        mutation["no_tests"],
        mutation["total_mutants"],
    ]
    assert doc.count("\n### ") == mutation["survived"], (
        "the survivors doc does not list one entry per surviving mutant"
    )


def test_a_transcript_may_quote_a_historical_count_but_prose_may_not():
    """`docs/REPRODUCE.md` legitimately records 281 — it is a dated transcript.

    The distinction matters: a transcript is evidence of what happened at a
    moment, and rewriting it to match today would be falsifying a record. Prose
    that states the current count is a live claim. This test pins the difference
    so nobody "fixes" the transcript.
    """
    repro = (REPO_ROOT / "docs" / "REPRODUCE.md").read_text(encoding="utf-8")
    assert "281 passed" in repro, "the transcript's historical count was altered"
    assert "```console" in repro, "the historical numbers must sit inside a transcript block"
