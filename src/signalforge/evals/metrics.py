"""Evaluation metrics.

The set here is chosen to answer the questions that actually decide whether a
signal is safe to trade on, which are not the questions a single accuracy number
answers:

``accuracy`` / ``macro_f1``
    Is it right? Macro-F1 rather than micro, because the interesting classes are
    the rare ones — a model that never predicts "withdrawn" can still post high
    accuracy on a corpus where most filings reaffirm.

``hallucination_rate``
    Does it make things up? Measured as the share of extractions containing at
    least one quote absent from the source. This is a *per-extraction* rate, not
    per-quote: one fabricated citation contaminates the whole finding.

``calibration`` (ECE + Brier)
    Does its confidence mean anything? A model that is 60% accurate while
    reporting 0.95 confidence is more dangerous than one that is 60% accurate and
    says so, because the second can be gated on confidence and the first cannot.
    This is the metric that determines whether ``score.py``'s confidence weighting
    is meaningful or decorative.

``schema_violation_rate`` / ``repair_rate``
    How much machinery does it need to stay in-contract? The clearest separator
    between local models in practice.

``p50/p95 latency`` and ``cost_per_1k``
    What does it cost to run? p95 rather than mean, since the tail is what breaks
    a batch window.

Directional agreement is reported alongside exact-match accuracy because for
several tasks getting the *sign* right is most of the value — confusing "lowered"
with "withdrawn" is a minor error, confusing either with "raised" is not.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseOutcome:
    """One graded evaluation case."""

    case_id: str
    expected: dict[str, Any]
    actual: dict[str, Any] | None
    #: Exact match on the task's primary label(s).
    correct: bool
    #: Sign agreement only.
    direction_correct: bool
    confidence: float | None
    grounded_ratio: float
    hallucinated: bool
    valid: bool
    repair_attempts: int
    latency_ms: float
    cost_usd: float
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None
    #: Which model actually served this case. Recorded per case rather than per
    #: run because a fallback chain can serve different cases from different
    #: models, and a run labelled with only the first link would misattribute.
    model: str = ""
    provider: str = ""


def summarise(outcomes: list[CaseOutcome], *, primary_field: str = "direction") -> dict[str, Any]:
    """Aggregate graded cases into the metric block stored on an eval run."""
    n = len(outcomes)
    if n == 0:
        return {"n": 0}

    valid = [o for o in outcomes if o.valid]
    n_valid = len(valid)

    # Accuracy is computed over *all* cases, not just valid ones. Scoring only the
    # responses a model managed to format correctly would flatter a model that
    # fails often, which is the opposite of what a production metric should do.
    correct = sum(1 for o in outcomes if o.correct)
    direction_correct = sum(1 for o in outcomes if o.direction_correct)

    latencies = [o.latency_ms for o in valid] or [0.0]
    confs = [(o.confidence, o.correct) for o in valid if o.confidence is not None]

    metrics: dict[str, Any] = {
        "n": n,
        "n_valid": n_valid,
        "accuracy": round(correct / n, 4),
        "direction_accuracy": round(direction_correct / n, 4),
        "macro_f1": round(macro_f1(outcomes, primary_field), 4),
        "schema_violation_rate": round((n - n_valid) / n, 4),
        "repair_rate": round(sum(1 for o in outcomes if o.repair_attempts > 0) / n, 4),
        "mean_repairs": round(sum(o.repair_attempts for o in outcomes) / n, 3),
        "hallucination_rate": round(sum(1 for o in valid if o.hallucinated) / n_valid, 4)
        if n_valid
        else None,
        "mean_grounded_ratio": round(sum(o.grounded_ratio for o in valid) / n_valid, 4)
        if n_valid
        else None,
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(percentile(latencies, 0.95), 1),
        "total_cost_usd": round(sum(o.cost_usd for o in outcomes), 6),
        "cost_per_1k_usd": round(sum(o.cost_usd for o in outcomes) / n * 1000, 4),
        "total_tokens": sum(o.tokens_in + o.tokens_out for o in outcomes),
    }

    if confs:
        metrics["ece"] = round(expected_calibration_error(confs), 4)
        metrics["brier"] = round(brier_score(confs), 4)
        metrics["mean_confidence"] = round(sum(c for c, _ in confs) / len(confs), 4)
        # The gap between stated confidence and realised accuracy: positive means
        # overconfident, which is the dangerous direction.
        metrics["overconfidence"] = round(metrics["mean_confidence"] - (correct / n), 4)

    metrics["confusion"] = confusion_matrix(outcomes, primary_field)
    return metrics


def macro_f1(outcomes: list[CaseOutcome], field_name: str) -> float:
    """Unweighted mean F1 across classes present in the ground truth.

    Unweighted on purpose: the rare labels ("withdrawn", "high") are the ones that
    carry the signal, and a weighted average would let a model ignore them.
    """
    labels = {str(o.expected.get(field_name)) for o in outcomes if field_name in o.expected}
    labels.discard("None")
    if not labels:
        return 0.0

    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()

    for o in outcomes:
        exp = str(o.expected.get(field_name))
        got = str((o.actual or {}).get(field_name))
        if exp == got:
            tp[exp] += 1
        else:
            fn[exp] += 1
            # An invalid or missing response is a false negative for the true
            # class but not a false positive for any class.
            if got != "None":
                fp[got] += 1

    scores = []
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
    return sum(scores) / len(scores)


def confusion_matrix(outcomes: list[CaseOutcome], field_name: str) -> dict[str, dict[str, int]]:
    """Expected -> predicted counts. Where the *shape* of the errors shows up."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for o in outcomes:
        exp = str(o.expected.get(field_name, "?"))
        got = str((o.actual or {}).get(field_name, "INVALID"))
        matrix[exp][got] += 1
    return {k: dict(v) for k, v in matrix.items()}


def expected_calibration_error(pairs: list[tuple[float, bool]], *, bins: int = 10) -> float:
    """Expected Calibration Error over equal-width confidence bins.

    ECE is the weighted average gap between stated confidence and observed
    accuracy. 0 is perfect; anything above ~0.15 means confidence should not be
    used as a gate without recalibration.
    """
    if not pairs:
        return 0.0
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for conf, ok in pairs:
        idx = min(int(conf * bins), bins - 1)
        buckets[idx].append((conf, ok))

    total = len(pairs)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
        error += (len(bucket) / total) * abs(avg_conf - accuracy)
    return error


def brier_score(pairs: list[tuple[float, bool]]) -> float:
    """Mean squared error of confidence against outcome. Lower is better."""
    if not pairs:
        return 0.0
    return sum((conf - (1.0 if ok else 0.0)) ** 2 for conf, ok in pairs) / len(pairs)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass
class RegressionCheck:
    """One threshold a run must satisfy to pass CI."""

    metric: str
    #: ``min`` for metrics where higher is better, ``max`` where lower is better.
    bound: str
    value: float

    def evaluate(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        actual = metrics.get(self.metric)
        if actual is None:
            # An absent metric is a failure, not a pass. A gate that silently
            # skips when its input disappears is worse than no gate.
            return False, f"{self.metric}: not reported"
        ok = actual >= self.value if self.bound == "min" else actual <= self.value
        arrow = ">=" if self.bound == "min" else "<="
        return ok, f"{self.metric}={actual} {arrow} {self.value} -> {'PASS' if ok else 'FAIL'}"


@dataclass
class GateResult:
    passed: bool
    lines: list[str] = field(default_factory=list)


def check_regressions(metrics: dict[str, Any], checks: list[RegressionCheck]) -> GateResult:
    result = GateResult(passed=True)
    for check in checks:
        ok, line = check.evaluate(metrics)
        result.lines.append(line)
        result.passed = result.passed and ok
    return result
