"""The human-in-the-loop gate in front of every SQL execution.

## The problem with demo-mode gates

An HITL gate that becomes `return True` when nobody is watching is not a control,
it is a screenshot. But a demo has to run unattended, so "just prompt the user"
is not an option either.

The resolution here: the gate **classifies risk first, and the policy decides per
risk level.** `auto` — the policy the demo runs under — approves `low` risk
automatically and **refuses `high` risk outright**, without a human to ask. So
the gate still blocks things in an unattended run, and the transcript shows it
blocking them. It is a weaker control than a human, and it is not nothing.

`interactive` is the real thing: it prints the SQL and waits on stdin.
`deny` refuses everything, which is what makes the "does the gate actually stop
execution" test meaningful.

## What counts as risky

Risk is assessed on the **normalised** SQL — the exact text that would execute,
after `LIMIT` injection — because approving one string and running another is the
oldest confused-deputy bug there is.

* `high` — anything `normalise_sql` rejected, or a cartesian join, or a scan with
  no `WHERE` over one of the large marts (`mart_futures_curve` is 1.6M rows).
* `medium` — joins across several marts, or aggregation over a large mart.
* `low` — a bounded read of a small monthly mart, which is almost everything the
  agent legitimately needs.

Every decision, including the automatic ones, lands in the transcript with its
reason. A gate whose approvals are invisible cannot be audited.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from agent.tools import SqlRejected, normalise_sql

#: Marts big enough that an unbounded scan is a genuine mistake.
LARGE_MARTS = ("mart_futures_curve", "mart_equity_daily", "mart_yield_curve", "mart_open_interest")

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"

POLICY_INTERACTIVE = "interactive"
POLICY_AUTO = "auto"
POLICY_DENY = "deny"

_FROM_TABLE = re.compile(r"\b(?:from|join)\s+\"?(\w+)\"?", re.IGNORECASE)
_JOIN = re.compile(r"\bjoin\b", re.IGNORECASE)
_WHERE = re.compile(r"\bwhere\b", re.IGNORECASE)
_AGG = re.compile(r"\b(sum|avg|count|min|max|stddev|median)\s*\(", re.IGNORECASE)

#: Severity order, so escalation is a `max` rather than a chain of conditionals.
#: The first version of this used `HIGH if level == HIGH else MEDIUM`, which can
#: only ever *hold* a level and never raise one — a two-join query capped at
#: medium and was auto-approved. Risk must only ever ratchet upward.
_ORDER: dict[str, int] = {RISK_LOW: 0, RISK_MEDIUM: 1, RISK_HIGH: 2}


def _escalate(current: str, candidate: str) -> str:
    return candidate if _ORDER[candidate] > _ORDER[current] else current


@dataclass(frozen=True)
class RiskAssessment:
    level: str
    reasons: tuple[str, ...]
    tables: tuple[str, ...]

    def to_json(self) -> dict:
        return {
            "level": self.level,
            "reasons": list(self.reasons),
            "tables": list(self.tables),
        }


@dataclass(frozen=True)
class GateDecision:
    approved: bool
    policy: str
    risk: RiskAssessment
    reason: str
    sql: str

    def to_json(self) -> dict:
        return {
            "approved": self.approved,
            "policy": self.policy,
            "reason": self.reason,
            "risk": self.risk.to_json(),
            "sql_presented": self.sql,
        }


def assess(sql: str) -> RiskAssessment:
    """Classify the statement that would actually run."""
    try:
        statement = normalise_sql(sql)
    except SqlRejected as exc:
        return RiskAssessment(
            level=RISK_HIGH,
            reasons=(f"rejected by the SQL validator: {exc}",),
            tables=(),
        )

    tables = tuple(sorted(set(_FROM_TABLE.findall(statement))))
    reasons: list[str] = []
    level = RISK_LOW

    large = [t for t in tables if t in LARGE_MARTS]
    joins = len(_JOIN.findall(statement))
    has_where = bool(_WHERE.search(statement))

    if large and not has_where:
        level = _escalate(level, RISK_HIGH)
        reasons.append(f"unbounded scan of a large mart: {', '.join(large)}")
    elif large:
        level = _escalate(level, RISK_MEDIUM)
        reasons.append(f"reads a large mart: {', '.join(large)}")

    if joins >= 2:
        level = _escalate(level, RISK_HIGH)
        reasons.append(f"{joins} joins")
    elif joins == 1:
        level = _escalate(level, RISK_MEDIUM)
        reasons.append("joins two marts")

    if _AGG.search(statement) and large:
        level = _escalate(level, RISK_MEDIUM)
        reasons.append("aggregation over a large mart")

    if not reasons:
        reasons.append("bounded read of a small mart")

    return RiskAssessment(level=level, reasons=tuple(reasons), tables=tables)


class ConfirmationGate:
    """Approves or refuses a SQL statement before it reaches the database."""

    def __init__(self, policy: str = POLICY_AUTO, auto_max_risk: str = RISK_MEDIUM) -> None:
        if policy not in (POLICY_INTERACTIVE, POLICY_AUTO, POLICY_DENY):
            raise ValueError(f"unknown gate policy {policy!r}")
        self.policy = policy
        self.auto_max_risk = auto_max_risk
        self.decisions: list[GateDecision] = []

    def _record(self, decision: GateDecision) -> GateDecision:
        self.decisions.append(decision)
        return decision

    def confirm(self, sql: str, question: str = "") -> GateDecision:
        risk = assess(sql)

        if self.policy == POLICY_DENY:
            return self._record(
                GateDecision(False, self.policy, risk, "policy denies all SQL", sql)
            )

        if self.policy == POLICY_AUTO:
            allowed = _ORDER[risk.level] <= _ORDER[self.auto_max_risk]
            reason = (
                f"auto-approved at risk={risk.level} (ceiling {self.auto_max_risk})"
                if allowed
                else (
                    f"REFUSED: risk={risk.level} exceeds the unattended ceiling "
                    f"{self.auto_max_risk}; this statement needs a human"
                )
            )
            return self._record(GateDecision(allowed, self.policy, risk, reason, sql))

        # Interactive: show the operator exactly what will run.
        print("\n" + "=" * 66, file=sys.stderr)
        print("SQL CONFIRMATION REQUIRED", file=sys.stderr)
        if question:
            print(f"question: {question}", file=sys.stderr)
        print(f"risk    : {risk.level} ({'; '.join(risk.reasons)})", file=sys.stderr)
        print("-" * 66, file=sys.stderr)
        print(sql, file=sys.stderr)
        print("=" * 66, file=sys.stderr)
        answer = input("execute this statement? [y/N] ").strip().lower()
        approved = answer in ("y", "yes", "s", "sim")
        return self._record(
            GateDecision(
                approved,
                self.policy,
                risk,
                "approved by operator" if approved else "refused by operator",
                sql,
            )
        )

    def summary(self) -> dict:
        return {
            "policy": self.policy,
            "auto_max_risk": self.auto_max_risk,
            "n_presented": len(self.decisions),
            "n_approved": sum(d.approved for d in self.decisions),
            "n_refused": sum(not d.approved for d in self.decisions),
            "by_risk": {
                level: sum(1 for d in self.decisions if d.risk.level == level)
                for level in (RISK_LOW, RISK_MEDIUM, RISK_HIGH)
            },
        }
