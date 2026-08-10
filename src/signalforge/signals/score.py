"""Extractions to signal scores.

A signal score is what a portfolio manager would actually look at, so the scoring
rules are explicit and auditable rather than learned. Three properties matter more
than sophistication here:

**Confidence discounts magnitude, it does not gate it.** A low-confidence bearish
read is a small bearish score, not a discarded one and not a full-strength one. The
alternative — thresholding confidence and dropping everything below it — throws away
the weak-but-real signals that aggregate into something useful.

**Ungrounded extractions are excluded, not discounted.** If the model could not
produce a verifiable quote, the finding is not evidence of anything. This is the
one hard gate in the pipeline, and it is the reason grounding is computed at
extraction time rather than as an afterthought.

**Scores are comparable across tasks.** Everything lands in [-1, 1] with a shared
sign convention (negative is bearish), so a risk-factor score and an 8-K event
score can be combined into a composite without per-task calibration constants
floating around the codebase.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..db import connect, query, upsert
from ..extract.runner import ExtractionResult
from ..extract.schemas import SCHEMAS, ExtractionBase

log = logging.getLogger(__name__)

#: Extractions whose evidence is less than this fraction verifiable are excluded.
#: Set at 1.0 for single-quote extractions to be strict, but tolerant of one bad
#: quote among several.
MIN_GROUNDED_RATIO = 0.5

#: Signals are floored rather than dropped at low confidence, so that weak reads
#: still contribute to a composite instead of vanishing.
MIN_CONFIDENCE_WEIGHT = 0.25

#: Relative weight of each signal in the composite. Encodes a view about
#: information content: a hard 8-K event is more decision-relevant than a shift in
#: MD&A adjectives, and both are more timely than an annual risk-factor diff.
COMPOSITE_WEIGHTS = {
    "event_class": 0.45,
    "guidance_tone": 0.35,
    "risk_delta": 0.20,
}

PIPELINE_VERSION = "score-v1"


@dataclass
class Signal:
    name: str
    cik: str
    accession: str
    as_of: date
    score: float
    confidence: float
    direction: str
    rationale: str
    evidence: list[str] = field(default_factory=list)
    extraction_ids: list[str] = field(default_factory=list)
    ticker: str | None = None
    pipeline_version: str = PIPELINE_VERSION

    @property
    def signal_id(self) -> str:
        basis = f"{self.name}|{self.cik}|{self.accession}|{self.pipeline_version}"
        return hashlib.sha256(basis.encode()).hexdigest()[:24]

    def row(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "name": self.name,
            "cik": self.cik,
            "ticker": self.ticker,
            "accession": self.accession,
            "as_of": self.as_of,
            "score": self.score,
            "confidence": self.confidence,
            "direction": self.direction,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "extraction_ids": self.extraction_ids,
            "pipeline_version": self.pipeline_version,
        }


def score_payload(payload: ExtractionBase) -> float:
    """Signed score in [-1, 1], magnitude discounted by confidence.

    The confidence weight is floored so that a genuine finding the model is
    unsure about still registers — dropping it entirely would bias the corpus
    toward unambiguous documents, which are exactly the ones that carry no alpha.
    """
    polarity = payload.polarity()
    weight = MIN_CONFIDENCE_WEIGHT + (1.0 - MIN_CONFIDENCE_WEIGHT) * payload.confidence
    return round(max(-1.0, min(1.0, polarity * weight)), 4)


def direction_of(score: float, *, deadband: float = 0.1) -> str:
    """Label a score, with a deadband so near-zero scores read as neutral.

    Without the deadband a score of 0.02 would be reported as "bullish", which
    overstates what the pipeline actually found.
    """
    if score > deadband:
        return "bullish"
    if score < -deadband:
        return "bearish"
    return "neutral"


def signal_from_extraction(
    result: ExtractionResult, *, ticker: str | None = None, as_of: date | None = None
) -> Signal | None:
    """Convert one validated, grounded extraction into a signal.

    Returns ``None`` when the extraction is not usable, which the caller counts
    rather than treating as an error — exclusion rate is itself a metric worth
    watching over time.
    """
    if not result.valid or result.payload is None:
        return None
    if result.grounded_ratio < MIN_GROUNDED_RATIO:
        log.debug("excluding %s: grounded_ratio=%.2f", result.extraction_id, result.grounded_ratio)
        return None

    score = score_payload(result.payload)
    return Signal(
        name=result.task,
        cik=result.cik,
        accession=result.accession,
        as_of=as_of or _filing_date(result.accession),
        score=score,
        confidence=result.payload.confidence,
        direction=direction_of(score),
        rationale=result.payload.rationale,
        evidence=result.payload.evidence,
        extraction_ids=[result.extraction_id],
        ticker=ticker,
    )


def score_stored_extractions(*, task: str | None = None, cik: str | None = None) -> list[Signal]:
    """Re-score extractions already in the warehouse.

    Kept separate from extraction so scoring rules can be revised and replayed
    without re-paying for inference — the difference between a cheap iteration
    loop and an expensive one.
    """
    clauses = ["e.valid = TRUE", f"e.grounded_ratio >= {MIN_GROUNDED_RATIO}"]
    params: list[Any] = []
    if task:
        clauses.append("e.task = ?")
        params.append(task)
    if cik:
        clauses.append("e.cik = ?")
        params.append(cik)

    rows = query(
        f"""
        SELECT e.extraction_id, e.task, e.cik, e.accession, e.payload,
               coalesce(f.report_date, f.filing_date) AS as_of,
               c.ticker
        FROM extractions e
        JOIN filings f ON f.accession = e.accession
        LEFT JOIN companies c ON c.cik = e.cik
        WHERE {" AND ".join(clauses)}
        """,
        params,
    )

    signals: list[Signal] = []
    for r in rows:
        model_cls = SCHEMAS.get(r["task"])
        if model_cls is None:
            continue
        try:
            payload = model_cls.model_validate_json(r["payload"])
        except Exception as exc:
            # A stored payload that no longer validates means the schema changed
            # under it. Skip loudly rather than crash a scoring run.
            log.warning("stale payload for %s: %s", r["extraction_id"], exc)
            continue

        score = score_payload(payload)
        signals.append(
            Signal(
                name=r["task"],
                cik=r["cik"],
                accession=r["accession"],
                as_of=r["as_of"],
                score=score,
                confidence=payload.confidence,
                direction=direction_of(score),
                rationale=payload.rationale,
                evidence=payload.evidence,
                extraction_ids=[r["extraction_id"]],
                ticker=r["ticker"],
            )
        )
    return signals


def persist_signals(signals: list[Signal]) -> int:
    if not signals:
        return 0
    with connect() as con:
        return upsert(con, "signals", [s.row() for s in signals], key="signal_id")


def composite_score(signals: list[Signal]) -> dict[str, Any]:
    """Weighted blend of a company's signals into one number.

    Weights are renormalised over the signals actually present, so a company with
    only an 8-K is not penalised for lacking a risk-factor diff — its composite
    reflects what is known, not what is missing.
    """
    if not signals:
        return {"score": 0.0, "direction": "neutral", "components": {}, "coverage": 0.0}

    total_weight = 0.0
    weighted = 0.0
    components: dict[str, float] = {}
    for s in signals:
        w = COMPOSITE_WEIGHTS.get(s.name, 0.1)
        weighted += s.score * w
        total_weight += w
        components[s.name] = s.score

    score = round(weighted / total_weight, 4) if total_weight else 0.0
    return {
        "score": score,
        "direction": direction_of(score),
        "components": components,
        # How much of the intended signal set contributed — a composite built on
        # one signal deserves less trust than one built on three.
        "coverage": round(total_weight / sum(COMPOSITE_WEIGHTS.values()), 3),
    }


def _filing_date(accession: str) -> date:
    rows = query(
        "SELECT coalesce(report_date, filing_date) AS d FROM filings WHERE accession = ?",
        [accession],
    )
    if rows and rows[0]["d"]:
        return rows[0]["d"]
    return date.today()
