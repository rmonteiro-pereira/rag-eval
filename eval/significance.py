"""Is that difference real? — confidence intervals and a paired test.

    uv run python -m eval.significance --out eval/reports/significance.json

The ablation reports MRR 0.191 for `dense` and 0.741 for `hybrid+rerank+metadata`
to three decimals, and also 0.381 for `hybrid` against 0.382 for `bm25`. Printed
the same way, at the same precision, those two comparisons look like the same
kind of statement. They are not: one is enormous and one is nothing. Until this
module existed, the repository gave a reader no way to tell them apart.

**n = 49.** One query is 2% of the question set, so a two-point gap is one query
changing its mind.

## What is computed, and why this shape

Every arm is scored on the *same* 49 queries by the same code, which makes the
comparison **paired** — and paired is worth a great deal at this sample size,
because the query-to-query variance (some questions are simply hard for every
arm) cancels out of the difference.

* **Per-arm 95% CI** — percentile bootstrap over queries. Answers "how precisely
  do we know this arm's score", and it is deliberately wide.
* **Per-contrast 95% CI on the delta** — *paired* bootstrap: one resample of the
  query indices, both arms evaluated on it, the difference taken per replicate.
  A CI that excludes zero is the claim worth making.
* **Two-sided p-value** — paired randomisation (permutation) test: under the null
  the two arms are interchangeable per query, so each query's pair is flipped
  with probability 0.5. This is the standard significance test for IR
  effectiveness comparisons (Smucker, Allan & Carterette, CIKM 2007), which found
  the t-test and randomisation agree and the bootstrap-shift and sign tests
  disagree with both.

## What this does NOT buy

The queries are 49 draft rows written by an agent. A confidence interval
describes sampling variability **given that question set**; it says nothing about
whether the question set measures the right thing. A tight CI on an unvalidated
gold set is a precise measurement of an unknown quantity, and no amount of
resampling fixes that. The interval is about noise, not about validity.

No multiple-comparison correction is applied, and with six contrasts reported a
p < 0.05 somewhere is not surprising on its own. The headline contrast does not
need one — it survives any correction — but the marginal ones should be read as
descriptive.

Everything here is computed from the committed `ablation.json`, which stores the
per-query metrics for every arm. No retrieval, no Qdrant, no models: the numbers
are re-derivable by anyone who clones the repo, and they are deterministic given
the seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rag.config import REPO_ROOT

DEFAULT_ABLATION = REPO_ROOT / "eval" / "reports" / "ablation.json"
DEFAULT_OUT = REPO_ROOT / "eval" / "reports" / "significance.json"

#: Fixed so the published intervals reproduce exactly. This repo's claim is that
#: its numbers come back; a seeded resample is part of that.
SEED = 20260731
N_BOOTSTRAP = 10_000
N_PERMUTATIONS = 10_000

#: The metrics worth an interval. `mrr` and `hit_rate@5` are the two the README
#: leads with. `rank1_doc_correct` is the wrong-meeting probe — the one the whole
#: project is about, and the one the reranker claim partly rests on, so it needs
#: an interval as much as the headline does.
METRICS = ("mrr", "hit_rate@5", "ndcg@10", "recall@5", "rank1_doc_correct")

#: Same pairs as `CONTRASTS` in `eval/ablation.py` — the single-component
#: differences the ablation's argument rests on.
CONTRASTS = (
    ("sparse (BM25 fused into dense)", "dense", "hybrid"),
    ("metadata filter, on dense", "dense", "dense+metadata"),
    ("metadata filter, on hybrid", "hybrid", "hybrid+metadata"),
    ("reranker, without the metadata filter", "hybrid", "hybrid+rerank"),
    ("reranker, with the metadata filter", "hybrid+metadata", "hybrid+rerank+metadata"),
    ("the whole stack", "dense", "hybrid+rerank+metadata"),
)


#: The probe metric is not measured over all 49 queries. `meeting_disambiguation`
#: is the 41 rows that name a meeting, and the published 0.098 -> 1.000 figures
#: are means over *those*. Computing it over all 49 would produce a different
#: quantity that happens to share a name — the first version of this module did
#: exactly that and reported a delta of -0.102 where the ablation says -0.146.
PROBE_METRIC = "rank1_doc_correct"
PROBE_GROUP = "meeting_disambiguation"


def probe_ids(arm: dict) -> set[str]:
    return set(arm["probes"][PROBE_GROUP]["ids"])


def per_query_values(arm: dict, metric: str, only: set[str] | None = None) -> np.ndarray:
    """One value per query, in the report's own order.

    `rank1_doc_correct` is a per-row boolean rather than a metric, so it is read
    from the row itself; everything else comes out of the metrics block. `only`
    restricts to a probe's query subset, which is required for the probe metric
    to mean what the ablation says it means.
    """
    rows = arm["per_query"]
    if only is not None:
        rows = [r for r in rows if r["id"] in only]
    if metric == PROBE_METRIC:
        return np.array([float(bool(r["rank1_doc_correct"])) for r in rows])
    return np.array([float(r["metrics"][metric]) for r in rows])


def bootstrap_ci(
    values: np.ndarray, n: int = N_BOOTSTRAP, seed: int = SEED, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_delta(
    a: np.ndarray, b: np.ndarray, n: int = N_BOOTSTRAP, seed: int = SEED, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(delta, lo, hi) for mean(b) - mean(a), resampling QUERIES not scores.

    The same resampled indices are applied to both arms, which is what makes it
    paired: a query that is hard for everyone contributes to both sides of the
    difference and cancels.
    """
    if len(a) != len(b):
        raise ValueError(f"arms scored different query counts: {len(a)} vs {len(b)}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    deltas = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(b.mean() - a.mean()), float(lo), float(hi)


def randomisation_p(
    a: np.ndarray, b: np.ndarray, n: int = N_PERMUTATIONS, seed: int = SEED
) -> float:
    """Two-sided paired randomisation test.

    Under the null the arm labels carry no information, so swapping a query's two
    scores is a valid relabelling. Flip each pair with probability 0.5, rebuild
    the mean difference, and count how often |difference| is at least as extreme
    as observed. The +1s are Davison & Hinkley's correction, which keeps the
    p-value from ever being exactly 0 — with 10,000 permutations the smallest
    reportable value is 1e-4, and reporting `p = 0` would claim a precision the
    procedure does not have.
    """
    if len(a) != len(b):
        raise ValueError(f"arms scored different query counts: {len(a)} vs {len(b)}")
    diff = b - a
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n, len(diff)))
    permuted = np.abs((signs * diff).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (n + 1))


def build_report(ablation: dict) -> dict:
    arms = {a["name"]: a for a in ablation["arms"]}
    n_queries = len(next(iter(arms.values()))["per_query"])

    per_arm: dict[str, dict] = {}
    for name, arm in arms.items():
        arm_entry: dict[str, dict] = {}
        for metric in METRICS:
            subset = probe_ids(arm) if metric == PROBE_METRIC else None
            values = per_query_values(arm, metric, subset)
            lo, hi = bootstrap_ci(values)
            arm_entry[metric] = {
                "mean": round(float(values.mean()), 4),
                "ci95_low": round(lo, 4),
                "ci95_high": round(hi, 4),
            }
        per_arm[name] = arm_entry

    contrasts: list[dict[str, Any]] = []
    for label, without, with_ in CONTRASTS:
        if without not in arms or with_ not in arms:
            raise KeyError(f"contrast {label!r} names an arm not in the report")
        entry: dict[str, Any] = {
            "component": label,
            "without": without,
            "with": with_,
            "metrics": {},
        }
        for metric in METRICS:
            # Both arms must be restricted to the SAME query subset, or the
            # pairing is broken and the delta compares different questions.
            subset = probe_ids(arms[without]) if metric == PROBE_METRIC else None
            if subset is not None:
                assert subset == probe_ids(arms[with_]), "probe membership differs between arms"
            a = per_query_values(arms[without], metric, subset)
            b = per_query_values(arms[with_], metric, subset)
            delta, lo, hi = paired_bootstrap_delta(a, b)
            p = randomisation_p(a, b)
            entry["metrics"][metric] = {
                "delta": round(delta, 4),
                "ci95_low": round(lo, 4),
                "ci95_high": round(hi, 4),
                "p_value": round(p, 4),
                # The claim a reader can act on: does the interval exclude no
                # difference at all?
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            }
        contrasts.append(entry)

    return {
        "schema_version": 1,
        "source_report": "eval/reports/ablation.json",
        "n_queries": n_queries,
        "n_probe_queries": len(probe_ids(next(iter(arms.values())))),
        "probe_note": (
            f"`{PROBE_METRIC}` is measured over the {PROBE_GROUP} subset only — the "
            "rows that name a meeting — matching how the ablation reports it. Every "
            "other metric is over all answerable queries."
        ),
        "method": {
            "per_arm": f"percentile bootstrap over queries, {N_BOOTSTRAP} resamples",
            "per_contrast": (
                f"paired bootstrap over queries ({N_BOOTSTRAP} resamples) for the "
                f"interval; two-sided paired randomisation test "
                f"({N_PERMUTATIONS} permutations) for the p-value"
            ),
            "seed": SEED,
            "multiple_comparisons": (
                "No correction applied. Six contrasts are reported, so an isolated "
                "p < 0.05 is not surprising on its own; the headline contrast "
                "survives any correction, the marginal ones are descriptive."
            ),
        },
        "caveat": (
            "These intervals describe sampling variability GIVEN a question set of "
            "56 draft rows that no human has validated. They say nothing about "
            "whether that question set measures the right thing. A tight interval "
            "on an unvalidated gold set is a precise measurement of an unknown "
            "quantity."
        ),
        "per_arm": per_arm,
        "contrasts": contrasts,
    }


def render(report: dict) -> str:
    lines = [
        "",
        f"n = {report['n_queries']} answerable queries · "
        f"{N_BOOTSTRAP} bootstrap resamples · seed {SEED}",
        "",
        "per arm, 95% CI",
        "-" * 58,
        f"{'arm':<26}{'MRR':>8}{'95% CI':>24}",
    ]
    for name, metrics in report["per_arm"].items():
        m = metrics["mrr"]
        lines.append(f"{name:<26}{m['mean']:>8.3f}   [{m['ci95_low']:.3f}, {m['ci95_high']:.3f}]")

    lines += [
        "",
        "contrasts — paired, on the same queries",
        "-" * 78,
        f"{'component':<40}{'dMRR':>8}{'95% CI':>22}{'p':>8}",
    ]
    for c in report["contrasts"]:
        m = c["metrics"]["mrr"]
        mark = "" if m["ci_excludes_zero"] else "   (CI includes 0)"
        lines.append(
            f"{c['component']:<40}{m['delta']:>+8.3f}   "
            f"[{m['ci95_low']:+.3f}, {m['ci95_high']:+.3f}]{m['p_value']:>8.4f}{mark}"
        )
    lines += ["", report["caveat"], ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.significance",
        description="Bootstrap CIs and a paired randomisation test over a committed ablation.",
    )
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.ablation.exists():
        print(f"no ablation report at {args.ablation}", file=sys.stderr)
        return 2

    report = build_report(json.loads(args.ablation.read_text(encoding="utf-8")))
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report -> {args.out}", file=sys.stderr)
    if not args.quiet:
        print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
