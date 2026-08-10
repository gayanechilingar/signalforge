"""Signal scoring and alert rules.

Pure functions of an extraction payload, so these run with no model at all — which
is the point: scoring rules are the part of the system that must be reviewable
line by line by someone who does not trust the model.
"""

from __future__ import annotations

from datetime import date

import pytest

from signalforge.extract.grounding import GroundingResult
from signalforge.extract.runner import ExtractionResult
from signalforge.extract.schemas import (
    Direction,
    EventClass,
    GuidanceChange,
    GuidanceTone,
    RiskDelta,
    Severity,
)
from signalforge.prompts.registry import Prompt
from signalforge.signals.alerts import RULES, evaluate
from signalforge.signals.score import (
    Signal,
    composite_score,
    direction_of,
    score_payload,
    signal_from_extraction,
)

PROMPT = Prompt(name="t", version="v1", template="x")


def tone(direction=Direction.bearish, change=GuidanceChange.withdrawn, conf=0.9):
    return GuidanceTone(
        direction=direction,
        guidance_change=change,
        confidence=conf,
        rationale="because",
        evidence=["quote here that is long enough"],
    )


def result(payload, *, grounded=1.0, valid=True, task="guidance_tone"):
    total, ok = (2, 2) if grounded == 1.0 else (2, int(2 * grounded))
    return ExtractionResult(
        task=task,
        accession="acc-1",
        cik="c1",
        section_id="s1",
        payload=payload,
        prompt=PROMPT,
        model="stub",
        provider="stub",
        valid=valid,
        grounding=GroundingResult(total=total, grounded=ok, ungrounded_quotes=[], methods=[]),
    )


class TestScoring:
    def test_bearish_is_negative_and_bullish_positive(self):
        assert score_payload(tone()) < 0
        assert score_payload(tone(Direction.bullish, GuidanceChange.raised)) > 0

    def test_confidence_discounts_magnitude_without_flipping_sign(self):
        strong = score_payload(tone(conf=1.0))
        weak = score_payload(tone(conf=0.1))
        assert strong < weak < 0, "both bearish, weak one smaller"

    def test_low_confidence_signal_is_not_erased(self):
        """Dropping weak reads biases the corpus toward unambiguous documents —
        exactly the ones carrying no information."""
        assert abs(score_payload(tone(conf=0.0))) >= 0.2

    def test_score_stays_in_range(self):
        for conf in (0.0, 0.5, 1.0):
            for d, c in [
                (Direction.bullish, GuidanceChange.raised),
                (Direction.bearish, GuidanceChange.withdrawn),
            ]:
                assert -1.0 <= score_payload(tone(d, c, conf)) <= 1.0

    def test_tasks_are_comparable_on_one_scale(self):
        event = EventClass(
            event_type="earnings_miss",
            direction=Direction.bearish,
            materiality=Severity.high,
            confidence=0.9,
            rationale="r",
        )
        risk = RiskDelta(
            direction=Direction.bearish,
            severity=Severity.high,
            confidence=0.9,
            rationale="r",
        )
        assert all(-1.0 <= score_payload(p) <= 1.0 for p in (event, risk, tone()))

    def test_direction_deadband_avoids_overclaiming(self):
        assert direction_of(0.02) == "neutral"
        assert direction_of(0.5) == "bullish"
        assert direction_of(-0.5) == "bearish"


class TestSignalConstruction:
    def test_grounded_extraction_becomes_a_signal(self):
        sig = signal_from_extraction(result(tone()), ticker="TST")
        assert sig is not None
        assert sig.direction == "bearish" and sig.ticker == "TST"

    def test_ungrounded_extraction_is_excluded_entirely(self):
        """The one hard gate: an unverifiable finding is not evidence of anything."""
        assert signal_from_extraction(result(tone(), grounded=0.0)) is None

    def test_invalid_extraction_is_excluded(self):
        assert signal_from_extraction(result(None, valid=False)) is None

    def test_signal_id_is_deterministic(self):
        a = signal_from_extraction(result(tone()))
        b = signal_from_extraction(result(tone()))
        assert a.signal_id == b.signal_id


class TestComposite:
    def _sig(self, name, score):
        return Signal(
            name=name,
            cik="c1",
            accession="a",
            as_of=date(2026, 1, 1),
            score=score,
            confidence=0.9,
            direction=direction_of(score),
            rationale="r",
        )

    def test_blends_components(self):
        out = composite_score([self._sig("event_class", -0.8), self._sig("risk_delta", -0.4)])
        assert out["score"] < 0
        assert set(out["components"]) == {"event_class", "risk_delta"}

    def test_weights_renormalise_so_missing_signals_do_not_penalise(self):
        """A company with only an 8-K should not be scored as if its 8-K were
        diluted by absent signals."""
        only = composite_score([self._sig("event_class", -1.0)])
        assert only["score"] == pytest.approx(-1.0)
        assert only["coverage"] < 1.0, "but coverage reports the thinness"

    def test_full_coverage_is_reported(self):
        out = composite_score(
            [
                self._sig("event_class", -0.5),
                self._sig("guidance_tone", -0.5),
                self._sig("risk_delta", -0.5),
            ]
        )
        assert out["coverage"] == pytest.approx(1.0)

    def test_empty_is_neutral_not_an_error(self):
        assert composite_score([])["direction"] == "neutral"


class TestAlerts:
    def _sig(self, name, score, conf=0.9, ticker="TST"):
        return Signal(
            name=name,
            cik="c1",
            accession="a",
            as_of=date(2026, 1, 1),
            score=score,
            confidence=conf,
            direction=direction_of(score),
            rationale="Guidance pulled.",
            ticker=ticker,
        )

    def test_withdrawn_guidance_is_critical(self):
        alerts = evaluate([self._sig("guidance_tone", -0.9)])
        assert any(
            a.rule == "guidance_withdrawn_or_cut" and a.severity == "critical" for a in alerts
        )

    def test_low_confidence_does_not_alert(self):
        """A muted alert stream loses the good alerts too, so precision beats
        recall here."""
        assert evaluate([self._sig("guidance_tone", -0.9, conf=0.3)]) == []

    def test_mild_signal_does_not_alert(self):
        assert evaluate([self._sig("risk_delta", -0.2)]) == []

    def test_positive_signals_are_informational_only(self):
        alerts = evaluate([self._sig("guidance_tone", 0.9)])
        assert alerts and all(a.severity == "info" for a in alerts)

    def test_alerts_are_ordered_by_severity(self):
        alerts = evaluate(
            [
                self._sig("guidance_tone", 0.9),  # info
                self._sig("risk_delta", -0.9),  # warn
                self._sig("event_class", -0.9),  # critical
            ]
        )
        assert [a.severity for a in alerts] == ["critical", "warn", "info"]

    def test_alert_id_is_deterministic(self):
        a = evaluate([self._sig("guidance_tone", -0.9)])[0]
        b = evaluate([self._sig("guidance_tone", -0.9)])[0]
        assert a.alert_id == b.alert_id

    def test_uses_ticker_when_available_else_cik(self):
        assert "TST" in evaluate([self._sig("guidance_tone", -0.9)])[0].headline
        assert "c1" in evaluate([self._sig("guidance_tone", -0.9, ticker=None)])[0].headline

    def test_every_rule_tolerates_every_signal_type(self):
        """A rule must never raise on a signal it does not apply to."""
        for name in ("guidance_tone", "risk_delta", "event_class", "unknown_task"):
            for rule in RULES:
                rule(self._sig(name, -0.9))
                rule(self._sig(name, 0.9))
