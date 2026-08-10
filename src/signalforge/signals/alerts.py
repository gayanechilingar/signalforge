"""Alert rules over signals.

Alerting is where a signal pipeline either earns trust or loses it. The design
constraint is not "detect everything" but "be worth reading": an alert stream with
a poor precision rate gets muted, after which the good alerts are lost too.

So the rules here are deliberately conservative and each one states its own
threshold in code rather than in a config file, because a threshold without the
reasoning next to it gets tuned arbitrarily. Rules are pure functions of a signal,
which makes them unit-testable without a model in the loop.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..db import connect, upsert
from .score import Signal


@dataclass(slots=True)
class Alert:
    rule: str
    severity: str  # info | warn | critical
    headline: str
    detail: str
    signal: Signal

    @property
    def alert_id(self) -> str:
        basis = f"{self.rule}|{self.signal.signal_id}"
        return hashlib.sha256(basis.encode()).hexdigest()[:24]

    def row(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "signal_id": self.signal.signal_id,
            "rule": self.rule,
            "severity": self.severity,
            "headline": self.headline,
            "detail": self.detail,
        }


Rule = Callable[[Signal], Alert | None]


def _label(signal: Signal) -> str:
    return signal.ticker or signal.cik


def rule_guidance_withdrawn(signal: Signal) -> Alert | None:
    """Withdrawn or lowered guidance — the single highest-conviction text signal.

    Companies do not withdraw guidance casually; it almost always precedes a
    material revision. Confidence floor is high because a false positive here is
    expensive in attention.
    """
    if signal.name != "guidance_tone" or signal.score > -0.5 or signal.confidence < 0.6:
        return None
    return Alert(
        rule="guidance_withdrawn_or_cut",
        severity="critical",
        headline=f"{_label(signal)}: forward guidance deteriorated",
        detail=signal.rationale,
        signal=signal,
    )


def rule_material_risk_escalation(signal: Signal) -> Alert | None:
    """A risk-factor section that got materially worse.

    Requires both a strong score and decent confidence: risk sections are long and
    largely boilerplate, so a weak bearish read is usually re-wording.
    """
    if signal.name != "risk_delta" or signal.score > -0.55 or signal.confidence < 0.55:
        return None
    return Alert(
        rule="material_risk_escalation",
        severity="warn",
        headline=f"{_label(signal)}: risk disclosure escalated materially",
        detail=signal.rationale,
        signal=signal,
    )


def rule_high_materiality_event(signal: Signal) -> Alert | None:
    """A directional 8-K that a reasonable investor would act on."""
    if signal.name != "event_class" or abs(signal.score) < 0.6:
        return None
    way = "positive" if signal.score > 0 else "negative"
    return Alert(
        rule="high_materiality_event",
        severity="critical" if signal.score < 0 else "warn",
        headline=f"{_label(signal)}: material {way} 8-K event",
        detail=signal.rationale,
        signal=signal,
    )


def rule_confident_positive(signal: Signal) -> Alert | None:
    """Strong positive signals, surfaced at low severity.

    Separate from the bearish rules and deliberately informational: upside
    disclosure is usually already priced by the time it is filed, so it belongs in
    a digest rather than an interrupt.
    """
    if signal.score < 0.65 or signal.confidence < 0.6:
        return None
    return Alert(
        rule="confident_positive",
        severity="info",
        headline=f"{_label(signal)}: positive disclosure signal ({signal.name})",
        detail=signal.rationale,
        signal=signal,
    )


RULES: tuple[Rule, ...] = (
    rule_guidance_withdrawn,
    rule_material_risk_escalation,
    rule_high_materiality_event,
    rule_confident_positive,
)


def evaluate(signals: list[Signal], *, rules: tuple[Rule, ...] = RULES) -> list[Alert]:
    """Run every rule over every signal, most severe first."""
    alerts = [a for s in signals for r in rules if (a := r(s)) is not None]
    order = {"critical": 0, "warn": 1, "info": 2}
    return sorted(alerts, key=lambda a: (order.get(a.severity, 3), -abs(a.signal.score)))


def persist_alerts(alerts: list[Alert]) -> int:
    if not alerts:
        return 0
    with connect() as con:
        return upsert(con, "alerts", [a.row() for a in alerts], key="alert_id")
