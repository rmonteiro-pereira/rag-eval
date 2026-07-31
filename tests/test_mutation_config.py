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
import pathlib
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
    # Both denominators, so neither can be quoted alone.
    assert "73.4%" in doc and "60.4%" in doc


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
