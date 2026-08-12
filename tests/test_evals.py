"""Evaluation metrics — the aggregation that decides whether a gate passes."""

from __future__ import annotations

import pytest

from signalforge.evals.metrics import CaseOutcome, summarise


def _case(
    case_id: str,
    *,
    correct: bool,
    valid: bool = True,
    confidence: float | None = 0.8,
    expected: str = "bearish",
    actual: str | None = "bearish",
) -> CaseOutcome:
    return CaseOutcome(
        case_id=case_id,
        expected={"direction": expected},
        actual=None if actual is None else {"direction": actual},
        correct=correct,
        direction_correct=correct,
        confidence=confidence,
        grounded_ratio=1.0,
        hallucinated=False,
        valid=valid,
        repair_attempts=0,
        latency_ms=10.0,
        cost_usd=0.0,
    )


class TestOverconfidence:
    def test_calibrated_model_reports_no_overconfidence(self):
        """Confidence 0.8 with 80% accuracy is honest, and must read as ~0."""
        outcomes = [
            _case(f"c{i}", correct=i < 8, actual="bearish" if i < 8 else "bullish")
            for i in range(10)
        ]
        metrics = summarise(outcomes)

        assert metrics["mean_confidence"] == pytest.approx(0.8)
        assert metrics["overconfidence"] == pytest.approx(0.0, abs=1e-9)

    def test_schema_failures_are_not_charged_against_the_confidence_pool(self):
        """Regression: both terms of the gap must come from the same population.

        A schema violation carries no confidence, so it is absent from the
        confidence pool — but the gap used to be measured against accuracy over
        *all* cases, which charged those failures to the pool anyway and inflated
        overconfidence by roughly the schema violation rate.
        """
        # Eight valid cases, all correct, all stating 0.8 confidence: the model is
        # in fact *under*-confident on everything it managed to answer.
        valid = [_case(f"ok{i}", correct=True) for i in range(8)]
        # Two responses that never validated, and so never stated a confidence.
        invalid = [
            _case(f"bad{i}", correct=False, valid=False, confidence=None, actual=None)
            for i in range(2)
        ]
        metrics = summarise(valid + invalid)

        assert metrics["schema_violation_rate"] == pytest.approx(0.2)
        assert metrics["accuracy"] == pytest.approx(0.8)
        assert metrics["mean_confidence"] == pytest.approx(0.8)
        # Accuracy *within the confidence pool* is 1.0, so a model stating 0.8 is
        # underconfident. The old computation reported 0.8 - 0.8 = 0.0 instead.
        assert metrics["overconfidence"] == pytest.approx(-0.2)

    def test_genuine_overconfidence_is_still_positive(self):
        """The dangerous direction must not be masked by the fix."""
        outcomes = [
            _case(f"c{i}", correct=i < 5, confidence=0.95, actual="bearish" if i < 5 else "bullish")
            for i in range(10)
        ]
        metrics = summarise(outcomes)

        assert metrics["overconfidence"] == pytest.approx(0.45)

    def test_no_cases_is_not_an_error(self):
        assert summarise([]) == {"n": 0}
