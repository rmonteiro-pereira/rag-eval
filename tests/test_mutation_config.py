"""The mutation config has to name real things, and mutate covered ones.

A sibling project was caught shipping a mutation setup whose `paths_to_mutate`
listed five directories that did not exist, behind a CI job pinned to
`if: false`. Both halves are the same failure: a quality control that is present
in the repository and absent from any run. Nothing was wrong with the *idea* —
nothing executed it, and nothing said so.

These tests are the cheap version of not repeating that. They cost milliseconds
and they run in the same `pytest` invocation CI already runs, which is the point:
the mutation run DOES now run in CI (~30s, `mutation` job), but only on Linux —
mutmut 3.x has no native Windows support, which is the development machine here.
So the *configuration* is checked in the suite too, where it is caught locally.

The second test is the one that matters more. Paths can all exist and the score
still be meaningless, if the selected tests never import the mutated modules —
every mutant comes back "no tests", the run reports a small, tidy denominator,
and the number looks fine.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
import tomllib

import pytest

from rag.config import REPO_ROOT

CONFIG = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["mutmut"]

PATH_KEYS = ("source_paths", "also_copy", "pytest_add_cli_args_test_selection")


def _entries(key: str) -> list[str]:
    return list(CONFIG.get(key, []))


@pytest.mark.parametrize("key", PATH_KEYS)
def test_the_config_key_is_populated(key):
    """An empty `source_paths` mutates nothing and reports no failures."""
    assert _entries(key), f"[tool.mutmut].{key} is empty — the run would be a no-op"


@pytest.mark.parametrize(
    "key, entry",
    [(key, entry) for key in PATH_KEYS for entry in _entries(key)],
)
def test_every_path_in_the_mutation_config_exists(key, entry):
    assert (REPO_ROOT / entry).exists(), (
        f"[tool.mutmut].{key} names {entry!r}, which does not exist"
    )


def _imported_modules(test_file: pathlib.Path) -> set[str]:
    """Module names imported by a test file, `from x.y import z` included."""
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("source", _entries("source_paths"))
def test_every_mutated_module_is_imported_by_a_selected_test(source):
    """Paths existing is not enough — the selected tests must reach the code.

    If the selection is disjoint from the mutated modules, every mutant is scored
    "no tests", the score is computed over an empty denominator, and the run looks
    clean because nothing was measured. That is a disabled job wearing a green
    tick.

    This is a floor, not a guarantee, and the difference is visible in this very
    repo: `eval/scoring.py` IS imported by a selected test and still has 60
    uncovered mutants, because importing a module does not call `score_rows`.
    Import linkage catches the config being wrong; only the run itself reports
    what is actually exercised, which is why `docs/mutation.md` carries a
    per-module `no test` column rather than a single headline.
    """
    module = source.replace("/", ".").removesuffix(".py")
    selected = [REPO_ROOT / t for t in _entries("pytest_add_cli_args_test_selection")]
    importers = [t.name for t in selected if module in _imported_modules(t)]
    assert importers, (
        f"{source} is mutated but no selected test imports {module!r}; "
        f"every mutant there would be scored 'no tests'"
    )


def test_mutmut_is_a_declared_dev_dependency():
    """The config is inert if the tool is not installable from this repo alone."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    assert any(spec.startswith("mutmut") for spec in dev), dev


def test_the_documented_score_names_its_scope_and_its_exclusions():
    """`docs/mutation.md` must state what is NOT mutated, not only the score.

    A mutation score with an unstated scope is the flattering kind: the number
    goes up by narrowing the denominator, and the reader cannot tell.
    """
    doc = (REPO_ROOT / "docs" / "mutation.md").read_text(encoding="utf-8")
    assert "Not mutated" in doc, "the exclusion is not stated"
    for excluded in ("store", "configs", "rerank"):
        assert excluded in doc, f"{excluded}.py is excluded from mutation but unmentioned"


def test_the_documented_mutation_score_matches_the_committed_artifact():
    """OPEN the artifact and compare. Do not check that a string is present.

    This test previously asserted `"73.4%" in doc` — that the *characters* 73.4%
    appeared somewhere in the prose. It would have passed just as happily with a
    score of 61%, because it never opened anything. A sibling project shipped a
    README claiming a KS statistic of 0.041016 against an artifact that said
    0.390791 — a FAIL reported as a PASS — and the test meant to prevent it
    asserted only that the cited file existed, never opening it. This is that
    test, rewritten to open the file.
    """
    report = json.loads((REPO_ROOT / "eval" / "reports" / "mutation.json").read_text("utf-8"))
    covered = report["killed"] + report["survived"]
    # The artifact must be internally consistent before it is trusted as a source.
    assert report["killed"] + report["survived"] + report["no_tests"] == report["total_mutants"]
    assert report["score"] == pytest.approx(100 * report["killed"] / covered, abs=0.05)

    for doc_name in ("docs/mutation.md", "README.md"):
        doc = (REPO_ROOT / doc_name).read_text(encoding="utf-8")
        quoted = set(re.findall(r"(\d{2}\.\d)%", doc))
        for value, label in (
            (report["score"], "score"),
            (report["score_including_uncovered"], "score_including_uncovered"),
        ):
            assert f"{value:.1f}" in quoted, (
                f"{doc_name} does not quote the artifact's {label} ({value:.1f}%); "
                f"it quotes {sorted(quoted)}"
            )


#: Percentages the mutation docs may quote that the current artifact does not
#: produce. Every entry needs a reason: an unexplained number in a document about
#: measurement is the defect this list exists to stop.
#:
#: Anything neither listed here nor derivable from `mutation.json` fails
#: `test_no_stale_mutation_score_survives_in_the_docs`.
#: Keyed **per document**, deliberately. A superseded score is legitimate inside
#: the progression narrative in `docs/mutation.md` and is never legitimate in the
#: README, which states only what is current. A single global allow-list would
#: re-admit the exact defect this guards — the first version of this table was
#: global, and a stale 73.4% in the README passed it.
ALLOWED_NON_ARTIFACT_PERCENTAGES: dict[str, dict[float, str]] = {
    "README.md": {
        0.0: "abstention correctness / PII leak rates in the guardrail table",
        11.1: "injection attack success, direct surface — guardrail table",
        16.7: "injection attack success, ungoverned — guardrail table",
    },
    "docs/mutation.md": {
        0.0: "a 0.0% per-module figure in prose; not a headline score",
        71.2: (
            "HISTORICAL: the first mutation run, before any survivor-driven "
            "tests. Quoted only as the start of the progression."
        ),
        73.4: (
            "HISTORICAL: the score after the first five survivor-driven tests "
            "and before tests/test_scoring.py covered eval/scoring.py. Quoted "
            "only as the middle term of 71.2 -> 73.4 -> 74.9. This value went "
            "stale in the README once and was quoted outside the repository "
            "before anyone noticed, which is exactly why it is written here — "
            "and why the README is not allowed to mention it at all."
        ),
    },
}


def test_no_stale_mutation_score_survives_in_the_docs():
    """Two different values for the same metric, in one repository, must fail.

    The test above asserts the artifact's score IS quoted. That is a *presence*
    check, and presence is not consistency: with the README saying both 74.9%
    and 73.4%, `74.9 in {74.9, 73.4}` is true and it passes. It did exactly
    that — a reviewer following the README's own pointer to `docs/mutation.md`
    landed on a number contradicting the headline, and the stale figure was
    quoted outside the repository before it was caught.

    Same family as asserting a cited file exists without opening it, one level
    up: asserting the right answer appears, without asserting a wrong one does
    not.

    So every percentage in the mutation docs must be derivable from
    `mutation.json` — a headline score or a per-module score — or listed in
    `ALLOWED_NON_ARTIFACT_PERCENTAGES` with a written reason.
    """
    report = json.loads((REPO_ROOT / "eval" / "reports" / "mutation.json").read_text("utf-8"))

    derivable = {report["score"], report["score_including_uncovered"]}
    for counts in report["by_module"].values():
        covered = counts["killed"] + counts["survived"]
        if covered:
            derivable.add(round(100 * counts["killed"] / covered, 1))

    for doc_name, allowed in ALLOWED_NON_ARTIFACT_PERCENTAGES.items():
        doc = (REPO_ROOT / doc_name).read_text(encoding="utf-8")
        for raw in sorted({float(x) for x in re.findall(r"(\d{2}\.\d)%", doc)}):
            assert raw in derivable or raw in allowed, (
                f"{doc_name} quotes {raw}%, which the committed mutation artifact does "
                f"not produce and which is not allow-listed for this document. If it is "
                f"a stale score, delete it; if it is deliberately historical, add it to "
                f"ALLOWED_NON_ARTIFACT_PERCENTAGES['{doc_name}'] with the reason."
            )


def test_a_historical_mutation_score_is_labelled_as_one():
    """A past value may appear only where the prose says it is past.

    `73.4%` is legitimate inside "71.2 -> 73.4 -> the current 74.9" and is not
    legitimate as a bare claim. The allow-list says which values are historical;
    this asserts the sentence around them agrees.
    """
    doc = (REPO_ROOT / "docs" / "mutation.md").read_text(encoding="utf-8")
    markers = ("first run", "took it to", "current", "previously", "was ")
    for value, reason in ALLOWED_NON_ARTIFACT_PERCENTAGES["docs/mutation.md"].items():
        if not reason.startswith("HISTORICAL"):
            continue
        for line in doc.splitlines():
            if f"{value}%" not in line:
                continue
            start = doc.find(line)
            window = doc[max(0, start - 300) : start + len(line) + 300]
            assert any(m in window for m in markers), (
                f"{value}% appears in docs/mutation.md without marking it historical: {line!r}"
            )


def test_the_documented_survivor_count_matches_the_artifact():
    """The inventory, the artifact and the prose must agree on one number."""
    report = json.loads((REPO_ROOT / "eval" / "reports" / "mutation.json").read_text("utf-8"))
    inventory = (REPO_ROOT / "docs" / "mutation-survivors.md").read_text(encoding="utf-8")
    entries = len(re.findall(r"^### ", inventory, re.M))
    assert entries == report["survived"], (
        f"{entries} survivors documented, artifact says {report['survived']}"
    )
    assert len(report["survivors"]) == report["survived"]


def test_no_survivor_is_left_undecided():
    """192 of 192 survivors marked "undecided" is a list, not an assessment.

    Every entry must carry a decision: ACCEPTED with the reason, OWED with the
    work that would kill it, or EQUIVALENT with a line-level proof.
    """
    inventory = (REPO_ROOT / "docs" / "mutation-survivors.md").read_text(encoding="utf-8")
    assert "UNDECIDED" not in inventory
    headings = re.findall(r"^### `[^`]+` — (.+)$", inventory, re.M)
    assert headings, "no survivor headings found — has the format changed?"
    for decision in headings:
        assert decision.startswith(("ACCEPTED", "OWED", "EQUIVALENT")), decision


# --------------------------------------------------------------------------
# The score gate. Same discipline as `eval/regression_gate.py`: a gate that has
# only ever passed is not evidence, so both directions are asserted.
# --------------------------------------------------------------------------


def _fake_results(killed: int, survived: int, uncovered: int) -> list[tuple[str, str]]:
    return (
        [(f"m.x_f__mutmut_{i}", "killed") for i in range(killed)]
        + [(f"m.x_g__mutmut_{i}", "survived") for i in range(survived)]
        + [(f"m.x_h__mutmut_{i}", "no tests") for i in range(uncovered)]
    )


def test_the_score_gate_passes_above_its_floor(monkeypatch, capsys):
    from tools import mutation_survivors as tool

    monkeypatch.setattr(tool, "_results", lambda: _fake_results(342, 124, 100))
    assert tool.main(["--check-score", "70"]) == 0
    assert "73.4%" in capsys.readouterr().out


def test_the_score_gate_fails_below_its_floor(monkeypatch, capsys):
    from tools import mutation_survivors as tool

    monkeypatch.setattr(tool, "_results", lambda: _fake_results(342, 124, 100))
    assert tool.main(["--check-score", "99"]) == 1
    assert "MUTATION GATE FAILED" in capsys.readouterr().err


def test_a_run_that_covered_nothing_is_a_failure_not_a_pass(monkeypatch, capsys):
    """The Mall lane's defect, expressed as a number.

    A config pointed at the wrong paths produces mutants that no test covers.
    `killed/(killed+survived)` is then 0/0 — undefined, not 100%. A gate that
    divides by zero and shrugs, or that treats "nothing measured" as "nothing
    wrong", is the failure it exists to prevent.
    """
    from tools import mutation_survivors as tool

    monkeypatch.setattr(tool, "_results", lambda: _fake_results(0, 0, 566))
    assert tool.main(["--check-score", "0"]) == 1
    assert "no mutant had a covering test" in capsys.readouterr().err


def test_no_results_at_all_exits_2_rather_than_reporting_a_score(monkeypatch):
    """Distinct exit code for "could not measure", mirroring the regression gate."""
    from tools import mutation_survivors as tool

    monkeypatch.setattr(tool, "_results", list)
    assert tool.main(["--check-score", "70"]) == 2
