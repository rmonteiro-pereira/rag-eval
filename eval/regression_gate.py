"""The CI gate: fail the build when retrieval quality drops.

    uv run python -m eval.regression_gate \
        --baseline eval/reports/ablation.json \
        --candidate eval/reports/ablation.json \
        --arm hybrid+rerank+metadata

Exit 0 = no regression. Exit 1 = a metric fell further than its threshold. Exit 2
= the comparison could not be made at all (missing file, unknown arm, absent
metric), which is a failure too — a gate that passes because it could not find
the numbers is worse than no gate.

## What it gates on, and why the probes matter more than the averages

Two families, both required:

* **Aggregate retrieval metrics** — `recall@k`, `hit_rate@k`, `nDCG@k`, `MRR`.
  The usual dashboard numbers.
* **Probe metrics** — rank-1 correct-document accuracy on the wrong-meeting
  trap (`eval/probes.py`).

The second family is the one worth having. The M4 ablation showed that the
metadata filter moves `meeting_disambiguation` rank-1 accuracy from 0.098 to
1.000 — a 0.902 swing — while moving nDCG@10 by 0.46. If someone breaks the
meeting resolver, the probe collapses to near zero and is impossible to miss;
the averages would drop too, but they would drop into a range that still looks
like "a retrieval system". **Gate on the metric that is specific to your known
defect, not only on the metric that is easy to average.**

## On thresholds

Retrieval here is deterministic: same collection, same gold set, same arm gives
bit-identical rankings. So the honest default tolerance is *small* — 0.02 is
already generous, and it exists to absorb a gold set that grew by a row, not
model noise, because there is none.

Thresholds are declared per metric rather than as one global epsilon. A 0.02
drop in `recall@10` and a 0.02 drop in `meeting_disambiguation` rank-1 accuracy
are not the same event and should not share a number.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from rag.config import REPO_ROOT

DEFAULT_BASELINE = REPO_ROOT / "eval" / "reports" / "ablation.json"
DEFAULT_ARM = "hybrid+rerank+metadata"

#: `metric -> maximum tolerated absolute drop vs the baseline`.
AGGREGATE_THRESHOLDS: dict[str, float] = {
    "recall@1": 0.02,
    "recall@5": 0.02,
    "hit_rate@1": 0.03,
    "hit_rate@5": 0.03,
    "ndcg@5": 0.02,
    "ndcg@10": 0.02,
    "mrr": 0.02,
}

#: Tighter, because these are the defect-specific guards. `meeting_disambiguation`
#: sits at 1.000 and any real drop means the meeting resolver broke.
PROBE_THRESHOLDS: dict[str, float] = {
    "meeting_disambiguation": 0.02,
    "reverse_lookup": 0.13,  # n=8, so one query is 0.125 — anything less is noise
}


class GateError(Exception):
    """The comparison could not be made. Exit 2, not exit 1."""


@dataclass(frozen=True)
class Check:
    name: str
    baseline: float
    candidate: float
    threshold: float

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    @property
    def passed(self) -> bool:
        return self.delta >= -self.threshold

    def to_json(self) -> dict:
        return {
            "metric": self.name,
            "baseline": round(self.baseline, 6),
            "candidate": round(self.candidate, 6),
            "delta": round(self.delta, 6),
            "threshold": self.threshold,
            "passed": self.passed,
        }


def extract(report: dict, arm: str | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """`(aggregate, probes)` from either report schema.

    `eval.ablation` writes many arms; `eval.run_eval` writes one and has no
    probes. Both are accepted so the gate can run against whichever report CI
    happens to produce.
    """
    if "arms" in report:
        arms = {entry["name"]: entry for entry in report["arms"]}
        name = arm or DEFAULT_ARM
        if name not in arms:
            raise GateError(f"arm {name!r} not in report; available: {', '.join(sorted(arms))}")
        entry = arms[name]
        probes = {
            group: values["rank1_doc_accuracy"] for group, values in entry.get("probes", {}).items()
        }
        return entry["aggregate"], probes

    if "aggregate" in report:
        return report["aggregate"], {}

    raise GateError("report has neither `arms` nor `aggregate` — not an eval report")


def compare(
    baseline: dict,
    candidate: dict,
    arm: str | None = None,
    aggregate_thresholds: dict[str, float] | None = None,
    probe_thresholds: dict[str, float] | None = None,
) -> list[Check]:
    aggregate_thresholds = aggregate_thresholds or AGGREGATE_THRESHOLDS
    probe_thresholds = probe_thresholds or PROBE_THRESHOLDS

    base_agg, base_probes = extract(baseline, arm)
    cand_agg, cand_probes = extract(candidate, arm)

    checks: list[Check] = []
    for metric, threshold in aggregate_thresholds.items():
        if metric not in base_agg:
            continue
        if metric not in cand_agg:
            # A metric the baseline has and the candidate does not is a
            # regression in the harness, not an absence to shrug at.
            raise GateError(f"candidate report is missing metric {metric!r}")
        checks.append(Check(metric, base_agg[metric], cand_agg[metric], threshold))

    for group, threshold in probe_thresholds.items():
        if group not in base_probes:
            continue
        if group not in cand_probes:
            raise GateError(f"candidate report is missing probe group {group!r}")
        checks.append(Check(f"probe:{group}", base_probes[group], cand_probes[group], threshold))

    if not checks:
        raise GateError("no comparable metrics found — refusing to pass vacuously")
    return checks


def render(checks: list[Check], arm: str) -> str:
    failures = [c for c in checks if not c.passed]
    lines = [
        "",
        f"regression gate — arm `{arm}`",
        f"{'metric':<34} {'baseline':>10} {'candidate':>10} {'delta':>9} {'tol':>6}  status",
        "-" * 78,
    ]
    for check in checks:
        lines.append(
            f"{check.name:<34} {check.baseline:>10.4f} {check.candidate:>10.4f} "
            f"{check.delta:>+9.4f} {check.threshold:>6.3f}  "
            f"{'ok' if check.passed else 'REGRESSION'}"
        )
    lines += [
        "-" * 78,
        (
            f"FAIL — {len(failures)} metric(s) regressed beyond tolerance"
            if failures
            else f"PASS — {len(checks)} metrics within tolerance"
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.regression_gate",
        description="Fail CI when retrieval quality drops against a committed baseline.",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_BASELINE,
        help="the fresh report; defaults to the baseline, which self-checks the gate",
    )
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        for path in (args.baseline, args.candidate):
            if not path.exists():
                raise GateError(f"report not found: {path}")
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        checks = compare(baseline, candidate, arm=args.arm)
    except GateError as exc:
        print(f"gate error: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"gate error: report is not valid JSON — {exc}", file=sys.stderr)
        return 2

    failed = [c for c in checks if not c.passed]
    if args.json:
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "passed": not failed,
                    "checks": [c.to_json() for c in checks],
                },
                indent=2,
            )
        )
    elif not args.quiet:
        print(render(checks, args.arm))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
