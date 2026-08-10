"""Typed extraction schemas.

Every extraction task declares a Pydantic model, and that model does three jobs
at once:

1. **Generates the JSON Schema** sent to the model, so the prompt and the
   validator can never drift apart.
2. **Validates the response**, converting "the model said something JSON-shaped"
   into "the model said something usable".
3. **Carries evidence**, so a numeric score can always be traced back to
   verbatim filing text.

The evidence field is the part that makes hallucination measurable rather than
vibes-based. A model asked for a label and a confidence will always produce them,
and they will always look plausible. A model asked to quote the sentence it based
the label on can be *checked*: either the quote is in the source document or it
is invented. That check is the hallucination metric in ``evals/metrics.py``, and
it is why every schema here requires evidence.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Direction(StrEnum):
    bullish = "bullish"
    bearish = "bearish"
    neutral = "neutral"

    @property
    def sign(self) -> int:
        return {"bullish": 1, "bearish": -1, "neutral": 0}[self.value]


class GuidanceChange(StrEnum):
    raised = "raised"
    lowered = "lowered"
    withdrawn = "withdrawn"
    reaffirmed = "reaffirmed"
    none = "none"


class Severity(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class ExtractionBase(BaseModel):
    """Common shape for every extraction.

    ``extra="forbid"`` is deliberate: a model that invents a field is a model
    that has stopped following the schema, and silently dropping the field would
    hide that. Better to fail validation and let the repair loop correct it.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    confidence: float = Field(
        description="How clearly the text supports this reading, 0-1.",
    )
    rationale: str = Field(description="One or two sentences justifying the call.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim quotes from the source document supporting the call.",
    )

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        # Local models frequently return 85 when asked for 0-1. Rescaling is
        # kinder than rejecting, and the intent is unambiguous.
        if v > 1.0:
            v = v / 100.0 if v <= 100.0 else 1.0
        return max(0.0, min(1.0, float(v)))

    @field_validator("evidence")
    @classmethod
    def _tidy_evidence(cls, quotes: list[str]) -> list[str]:
        out: list[str] = []
        for q in quotes:
            q = " ".join(str(q).split())
            if len(q) >= 15:  # a fragment is not a citation
                out.append(q[:500])
        return out[:5]

    #: Subclasses set this to the direction of the finding, so scoring can be
    #: written once rather than per task.
    def polarity(self) -> float:
        raise NotImplementedError


class GuidanceTone(ExtractionBase):
    """Management's forward-looking tone in an MD&A or results section."""

    direction: Direction
    guidance_change: GuidanceChange

    def polarity(self) -> float:
        # An explicit guidance change is a harder signal than adjectives, so it
        # dominates the tone label rather than merely nudging it.
        explicit = {
            GuidanceChange.raised: 1.0,
            GuidanceChange.lowered: -1.0,
            GuidanceChange.withdrawn: -0.85,
            GuidanceChange.reaffirmed: 0.15,
            GuidanceChange.none: 0.0,
        }[self.guidance_change]
        if explicit:
            return explicit
        return float(self.direction.sign) * 0.6


class RiskDelta(ExtractionBase):
    """Change in risk disclosure between two consecutive filings."""

    direction: Direction = Field(
        description="bearish if risk disclosure worsened, bullish if it eased."
    )
    severity: Severity = Field(description="Materiality of the change.")
    new_risks: list[str] = Field(
        default_factory=list, description="Risks present now that were absent before."
    )
    removed_risks: list[str] = Field(
        default_factory=list, description="Risks previously disclosed that are now absent."
    )
    escalated_risks: list[str] = Field(
        default_factory=list, description="Risks whose language intensified."
    )

    @field_validator("new_risks", "removed_risks", "escalated_risks")
    @classmethod
    def _cap_lists(cls, v: list[str]) -> list[str]:
        return [" ".join(str(x).split())[:300] for x in v][:10]

    def polarity(self) -> float:
        weight = {Severity.low: 0.3, Severity.medium: 0.65, Severity.high: 1.0}[self.severity]
        return float(self.direction.sign) * weight


class EventClass(ExtractionBase):
    """Classification of an 8-K event and its likely price impact."""

    event_type: str = Field(description="Short label, e.g. 'earnings_beat', 'auditor_change'.")
    direction: Direction
    materiality: Severity

    @field_validator("event_type")
    @classmethod
    def _slugify(cls, v: str) -> str:
        import re

        return re.sub(r"[^a-z0-9_]+", "_", str(v).lower().strip())[:60].strip("_")

    def polarity(self) -> float:
        weight = {Severity.low: 0.25, Severity.medium: 0.6, Severity.high: 1.0}[self.materiality]
        return float(self.direction.sign) * weight


#: Task name -> schema. The runner and the eval harness both resolve through this
#: single mapping, so adding a task never means editing two places.
SCHEMAS: dict[str, type[ExtractionBase]] = {
    "guidance_tone": GuidanceTone,
    "risk_delta": RiskDelta,
    "event_class": EventClass,
}


def json_schema_for(task: str) -> dict[str, Any]:
    """JSON Schema for a task, with ``$defs`` inlined.

    Local models handle a flat schema markedly better than one with ``$ref``
    indirection, and Anthropic's constrained decoding accepts either — so
    flattening costs nothing and raises the floor.
    """
    model = SCHEMAS[task]
    return _inline_refs(model.model_json_schema())


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.split("/")[-1], {})
                merged = {**walk(target)}
                # Keep any sibling keys (description, default) alongside the
                # resolved definition.
                for k, v in node.items():
                    if k != "$ref":
                        merged[k] = walk(v)
                return merged
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)
