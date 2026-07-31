"""Generate the complete, per-mutant survivor inventory from a mutmut run.

    uv run mutmut run
    uv run python -m tools.mutation_survivors > docs/mutation-survivors.md

Categories are summarised in `docs/mutation.md`; this produces the exhaustive
list behind them, because "mostly harmless report-key mutations" is a claim a
reader cannot check without seeing all of them.

Classification is mechanical and deliberately conservative: a mutant is only
called EQUIVALENT when the reason is recorded in `EQUIVALENT` below with the
argument for it. Everything else is a gap, even where the gap is cheap — a
generated file that quietly downgraded survivors to "fine" would be the same
failure as a mutation config that never ran.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from collections import defaultdict

RESULT = re.compile(r"^\s+([\w.]+)\.x_(\w+)__mutmut_(\d+): (\w[\w ]*)$")

#: mutant name -> why no test can kill it. Each needs an argument from the source,
#: not an assertion. Anything not listed here is reported as a gap.
EQUIVALENT: dict[str, str] = {
    "retrieval.fusion.x_reciprocal_rank_fusion__mutmut_22": (
        "`score=0.0` is a placeholder on a freshly built `Retrieved`; "
        "`record.score = score` overwrites it unconditionally before return, so "
        "the value is never observable."
    ),
    "retrieval.fusion.x_reciprocal_rank_fusion__mutmut_40": (
        "Same placeholder as mutant 22, set to `1.0` instead of `None`."
    ),
}

#: Every surviving mutant must land in exactly one of these, and each carries a
#: WRITTEN DECISION — accept it and why, or the work that would kill it. A
#: sibling project shipped 192 of 192 survivors marked "undecided", which is a
#: list, not an assessment: it tells a reader the tool ran and nobody looked.
#:
#: `main()` returns 1 if any survivor falls through to UNDECIDED, so this table
#: cannot silently go stale as the code moves.
DECISIONS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "ACCEPTED — report payload key",
        re.compile(r'"XX|XX"'),
        "Mutates a JSON key or string literal in the report payload (e.g. "
        "`expected_doc_id` -> `XXexpected_doc_idXX`). The metric VALUES are "
        "asserted by tests; the report SCHEMA is not. Accepted for now, and the "
        "gap is real rather than harmless: `regression_gate.extract()` reads "
        '`probes[group]["rank1_doc_accuracy"]` by name, so a silent key rename '
        "would break the gate. Killing these needs one test that round-trips a "
        "probe report through the gate — recorded in `mutation.md` as owed work.",
    ),
    (
        "ACCEPTED — prose inside an explanatory note",
        re.compile(r"^\s*[-+]\s*[\"']\s*[A-Za-z].{25,}"),
        "Mutates human-readable prose in a `note`/`description` field that exists "
        "to explain a number to a reader, not to be computed on. Asserting the "
        "wording of an explanation would pin the docs to the tests in the wrong "
        "direction — the next person improving a sentence would have to update a "
        "test. Deliberately not killed.",
    ),
    (
        "ACCEPTED — CLI plumbing",
        re.compile(r"add_argument|help=|metavar|prog=|description="),
        "Mutates argparse help text, metavars or the program name. Tests call "
        "`main([...])` and assert exit codes and behaviour, which is the contract "
        "that matters; asserting help strings tests argparse, not this code.",
    ),
    (
        "ACCEPTED — unobservable default",
        re.compile(r"=\s*(None|\(\)|\[\]|\{\}|0|1|0\.0|True|False)\s*[,)]"),
        "Mutates a default or placeholder that is overwritten or never read on "
        "any path a test can reach. Distinct from the two provably-equivalent "
        "mutants in `EQUIVALENT`: those have a line-level proof, these are "
        "judged unreachable rather than proven so, which is why they are "
        "ACCEPTED and not EQUIVALENT.",
    ),
    (
        "OWED — logic not covered by a test",
        re.compile(r".", re.S),
        "Mutates real logic and survived, so a test is missing. These are the "
        "ones worth spending on. Tracked in `mutation.md`; the largest cluster "
        "is `eval/scoring.py`, whose `score_rows` computes every published "
        "retrieval metric and has no direct unit test.",
    ),
)


def _results() -> list[tuple[str, str]]:
    out = subprocess.run(
        ["mutmut", "results", "--all", "true"], capture_output=True, text=True, check=True
    ).stdout.replace("\r", "\n")
    rows = []
    for line in out.splitlines():
        m = RESULT.match(line)
        if m:
            rows.append((f"{m.group(1)}.x_{m.group(2)}__mutmut_{m.group(3)}", m.group(4).strip()))
    return rows


def _diff(name: str) -> list[str]:
    out = subprocess.run(
        ["mutmut", "show", name], capture_output=True, text=True, check=False
    ).stdout.replace("\r", "\n")
    return [
        line
        for line in out.splitlines()
        if line[:1] in "-+" and not line.startswith(("---", "+++"))
    ]


def _decide(diff: list[str]) -> tuple[str, str]:
    """(decision, written reason) for one survivor. Never silently 'undecided'."""
    joined = "\n".join(diff)
    for label, pattern, reason in DECISIONS:
        if pattern.search(joined):
            return label, reason
    return "UNDECIDED", "no rule in DECISIONS matched this mutant — it must not ship undecided"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tools.mutation_survivors",
        description="Print the full survivor inventory, or gate on the mutation score.",
    )
    parser.add_argument(
        "--check-score",
        type=float,
        default=None,
        metavar="MIN",
        help=(
            "print the score and exit 1 if killed/(killed+survived) falls below MIN "
            "percent. A floor rather than an exact pin: adding a test should never "
            "fail the build, and the number moves whenever the mutated code does."
        ),
    )
    parser.add_argument(
        "--write-json",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help=(
            "write the machine-readable score to PATH. Without it the mutation "
            "score lives only in prose and a CI log, while every other number "
            "this project publishes traces to a file under eval/reports/."
        ),
    )
    args = parser.parse_args(argv)

    rows = _results()
    if not rows:
        print("no mutmut results — run `uv run mutmut run` first", file=sys.stderr)
        return 2

    survived = [name for name, status in rows if status == "survived"]
    counts: defaultdict[str, int] = defaultdict(int)
    for _name, status in rows:
        counts[status] += 1

    killed, surv, none = counts["killed"], counts["survived"], counts["no tests"]

    if args.write_json is not None:
        by_counts: dict[str, dict[str, int]] = {}
        for name, status in rows:
            module = name.rsplit(".x_", 1)[0]
            by_counts.setdefault(module, {"killed": 0, "survived": 0, "no tests": 0})
            by_counts[module][status] += 1
        cov = killed + surv
        args.write_json.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "tool": "mutmut",
                    "total_mutants": len(rows),
                    "killed": killed,
                    "survived": surv,
                    "no_tests": none,
                    "score": round(100 * killed / cov, 1) if cov else None,
                    "score_including_uncovered": (
                        round(100 * killed / len(rows), 1) if rows else None
                    ),
                    "by_module": by_counts,
                    "survivors": sorted(survived),
                    "note": (
                        "Regenerated by `tools.mutation_survivors --write-json`. `score` is "
                        "killed/(killed+survived); `score_including_uncovered` counts mutants "
                        "no in-scope test imports as unkilled. Both are published because "
                        "quoting either alone flatters or damns."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.write_json}")
        return 0

    if args.check_score is not None:
        covered = killed + surv
        if not covered:
            print(
                "MUTATION GATE FAILED: no mutant had a covering test, so the score "
                "is undefined. This is what a config that mutates the wrong paths "
                "looks like — it is a failure, not a pass.",
                file=sys.stderr,
            )
            return 1
        score = 100 * killed / covered
        print(
            f"mutation score {score:.1f}%  ({killed} killed / {covered} covered, "
            f"{none} uncovered, {len(rows)} mutants)"
        )
        if score < args.check_score:
            print(
                f"MUTATION GATE FAILED: {score:.1f}% is below the {args.check_score:.1f}% floor",
                file=sys.stderr,
            )
            return 1
        print(f"at or above the {args.check_score:.1f}% floor")
        return 0

    print("# Surviving mutants — the complete list\n")
    print("Generated by `uv run python -m tools.mutation_survivors`. Do not hand-edit;")
    print("re-run it after `uv run mutmut run` so the list cannot drift from the score.\n")
    print(
        f"**{killed} killed · {surv} survived · {none} uncovered** "
        f"of {len(rows)} mutants — see [`mutation.md`](mutation.md) for the score,"
    )
    print("the scope, and what is deliberately not fixed.\n")
    print("Every survivor is listed. Two are marked EQUIVALENT with the argument for why")
    print("no test can kill them; everything else is a gap, including the cheap ones.\n")

    by_module: dict[str, list[str]] = defaultdict(list)
    for name in survived:
        by_module[name.rsplit(".x_", 1)[0]].append(name)

    print("| module | survivors |\n|---|--:|")
    for module in sorted(by_module):
        print(f"| `{module}` | {len(by_module[module])} |")
    print()

    undecided: list[str] = []
    for module in sorted(by_module):
        print(f"\n## `{module}`\n")
        for name in by_module[module]:
            diff = _diff(name)
            short = name.rsplit(".x_", 1)[1]
            if name in EQUIVALENT:
                decision, reason = "EQUIVALENT — no test can kill it", EQUIVALENT[name]
            else:
                decision, reason = _decide(diff)
                if decision == "UNDECIDED":
                    undecided.append(name)
            print(f"### `{short}` — {decision}\n")
            print(f"{reason}\n")
            print("```diff")
            print("\n".join(diff) if diff else "(diff unavailable)")
            print("```\n")

    if undecided:
        print(
            f"\n{len(undecided)} survivors are UNDECIDED. A list of survivors nobody "
            "assessed is not an assessment.",
            file=sys.stderr,
        )
        for name in undecided:
            print(f"  UNDECIDED: {name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
