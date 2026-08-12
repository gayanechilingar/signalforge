"""Extraction: schemas, grounding, the repair loop, and provenance."""

from __future__ import annotations

import json

import pytest

from signalforge.db import query
from signalforge.extract.grounding import check_grounding
from signalforge.extract.runner import extract
from signalforge.extract.schemas import (
    Direction,
    GuidanceChange,
    GuidanceTone,
    RiskDelta,
    Severity,
    json_schema_for,
)

SOURCE = (
    "Revenue for the quarter declined 12% year over year. "
    "Given continued softness in enterprise demand, we are withdrawing our "
    "full-year outlook. We expect to provide an update next quarter."
)


def _tone_json(**over) -> str:
    base = {
        "direction": "bearish",
        "guidance_change": "withdrawn",
        "confidence": 0.9,
        "rationale": "Guidance was withdrawn.",
        "evidence": ["we are withdrawing our full-year outlook"],
    }
    base.update(over)
    return json.dumps(base)


class TestSchemas:
    def test_schema_is_flat(self):
        """Local models handle a flat schema far better than $ref indirection."""
        schema = json.dumps(json_schema_for("guidance_tone"))
        assert "$ref" not in schema and "$defs" not in schema

    def test_enums_survive_inlining(self):
        props = json_schema_for("guidance_tone")["properties"]
        assert set(props["direction"]["enum"]) == {"bullish", "bearish", "neutral"}

    def test_percentage_confidence_is_rescaled(self):
        """Local models routinely answer 85 when asked for 0-1."""
        assert GuidanceTone.model_validate_json(_tone_json(confidence=85)).confidence == 0.85

    def test_confidence_is_clamped(self):
        assert GuidanceTone.model_validate_json(_tone_json(confidence=-3)).confidence == 0.0

    def test_extra_fields_are_rejected(self):
        """A model inventing fields has stopped following the schema; the repair
        loop should see that rather than have it silently dropped."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GuidanceTone.model_validate_json(_tone_json(commentary="extra"))

    def test_fragment_evidence_is_discarded(self):
        payload = GuidanceTone.model_validate_json(_tone_json(evidence=["yes", "n/a"]))
        assert payload.evidence == []

    def test_guidance_change_dominates_direction_in_polarity(self):
        """An explicit guidance change is harder evidence than tone adjectives."""
        raised_but_bearish = GuidanceTone(
            direction=Direction.bearish,
            guidance_change=GuidanceChange.raised,
            confidence=1.0,
            rationale="",
        )
        assert raised_but_bearish.polarity() == 1.0

    def test_severity_scales_risk_polarity(self):
        def mk(sev):
            return RiskDelta(
                direction=Direction.bearish, severity=sev, confidence=1.0, rationale=""
            ).polarity()

        assert mk(Severity.high) < mk(Severity.medium) < mk(Severity.low) < 0


class TestGrounding:
    def test_exact_quote_is_grounded(self):
        r = check_grounding(["we are withdrawing our full-year outlook"], SOURCE)
        assert r.ratio == 1.0
        assert r.methods == ["exact"]

    def test_punctuation_and_whitespace_drift_is_tolerated(self):
        r = check_grounding(["we  are withdrawing our full–year outlook!"], SOURCE)
        assert r.grounded == 1

    def test_smart_quotes_are_normalised(self):
        src = "The company said “demand softened materially” during the period."
        assert check_grounding(["demand softened materially"], src).grounded == 1

    def test_elided_word_still_matches_fuzzily(self):
        r = check_grounding(
            ["Given continued softness in demand, we are withdrawing our full-year outlook"], SOURCE
        )
        assert r.grounded == 1
        assert r.methods == ["fuzzy"]

    def test_repeated_words_do_not_make_fuzzy_matching_unreachable(self):
        """Regression: the fuzzy threshold is a share of *distinct* quote tokens.

        Scaling it by the raw token count made the bar exceed the maximum a set
        intersection can reach as soon as a quote repeated a word, so stage 3
        never fired on quotes of realistic length and both cases it exists for
        were reported as fabrications.
        """
        source = (
            "We expect revenue growth in the second half of the year to be driven by "
            "strength in the services segment and by the continued expansion of the "
            "installed base of our products in the enterprise market."
        )
        footnote = source.replace("services segment", "services segment(1)")
        elided = source.replace("the enterprise market", "the market")

        assert check_grounding([footnote], source).methods == ["fuzzy"]
        assert check_grounding([elided], source).methods == ["fuzzy"]

    def test_relaxed_threshold_still_rejects_topical_invention(self):
        """The looser bar must not turn 'same subject matter' into 'cited'."""
        source = (
            "We expect revenue growth in the second half of the year to be driven by "
            "strength in the services segment and by the continued expansion of the "
            "installed base of our products in the enterprise market."
        )
        invented = "We expect revenue in the second half of the year to decline sharply overseas"
        assert check_grounding([invented], source).methods == ["none"]

    def test_invented_quote_is_caught(self):
        r = check_grounding(["we expect revenue to double next year"], SOURCE)
        assert r.ratio == 0.0
        assert r.hallucinated is True
        assert r.ungrounded_quotes == ["we expect revenue to double next year"]

    def test_partial_hallucination_is_reported_as_a_ratio(self):
        r = check_grounding(
            [
                "Revenue for the quarter declined 12% year over year",
                "the CEO resigned effective immediately",
            ],
            SOURCE,
        )
        assert r.total == 2 and r.grounded == 1
        assert r.ratio == 0.5

    def test_no_evidence_is_not_a_hallucination(self):
        """No claims made means no claims fabricated. Whether an evidence-free
        extraction is acceptable is a review-policy question, not a metric one."""
        r = check_grounding([], SOURCE)
        assert r.ratio == 1.0 and r.hallucinated is False

    def test_very_short_quote_is_not_fuzzy_matched(self):
        """A two-word 'quote' would match almost any document, which would make
        the whole metric meaningless."""
        assert check_grounding(["the company"], SOURCE).methods == ["none"]


class TestRunner:
    def test_valid_response_is_persisted_with_provenance(self, router, stub, warehouse):
        stub.register("<document", _tone_json())
        result = extract(
            "guidance_tone",
            source_text=SOURCE,
            variables={
                "company": "Testco",
                "form": "10-Q",
                "period": "2026-06-30",
                "section_text": SOURCE,
            },
            accession="acc-1",
            cik="0000000001",
            section_id="acc-1:mdna",
            router=router,
            chain="ci",
        )
        assert result.valid
        assert result.payload.guidance_change == GuidanceChange.withdrawn
        assert result.grounded_ratio == 1.0

        rows = query("SELECT * FROM extractions")
        assert len(rows) == 1
        row = rows[0]
        # Provenance: the prompt version AND its content hash, so an uncommitted
        # prompt edit is detectable after the fact.
        assert row["prompt_name"] == "guidance_tone"
        assert row["prompt_hash"]
        assert row["model"] == "stub"
        assert row["valid"] is True

    def test_extraction_id_is_stable_across_reruns(self, router, stub, warehouse):
        stub.register("<document", _tone_json())
        kwargs = dict(
            source_text=SOURCE,
            variables={"company": "T", "form": "10-Q", "period": "p", "section_text": SOURCE},
            accession="acc-1",
            cik="c1",
            section_id="s1",
            router=router,
            chain="ci",
        )
        a = extract("guidance_tone", **kwargs)
        b = extract("guidance_tone", **kwargs)
        assert a.extraction_id == b.extraction_id
        assert len(query("SELECT * FROM extractions")) == 1, "rerun must not duplicate"

    def test_repair_loop_recovers_from_invalid_enum(self, router, stub, warehouse):
        """Local models violate enums constantly; discarding those responses would
        bias the corpus toward easy documents."""
        stub.register("<document", _tone_json(direction="very bearish"))
        stub.register("Schema validation failed", _tone_json())

        result = extract(
            "guidance_tone",
            source_text=SOURCE,
            variables={"company": "T", "form": "10-Q", "period": "p", "section_text": SOURCE},
            accession="acc-1",
            cik="c1",
            router=router,
            chain="ci",
        )
        assert result.valid is True
        assert result.repair_attempts == 1
        # Cost accrues across attempts — the price of a valid answer includes the
        # failed tries.
        assert result.tokens_out > 0

    def test_unrepairable_response_is_recorded_as_invalid(self, router, stub, warehouse):
        # Both the initial turn and the repair turn must fail, or the repair
        # succeeds and there is nothing to assert about.
        stub.register("<document", "I cannot answer that.")
        stub.register("Schema validation failed", "Still cannot answer that.")
        stub.register("Response was not valid JSON", "Still cannot answer that.")
        result = extract(
            "guidance_tone",
            source_text=SOURCE,
            variables={"company": "T", "form": "10-Q", "period": "p", "section_text": SOURCE},
            accession="acc-1",
            cik="c1",
            router=router,
            chain="ci",
            max_repairs=1,
        )
        assert result.valid is False
        assert result.error
        assert result.needs_review and result.review_reason() == "invalid"
        assert query("SELECT * FROM extractions")[0]["valid"] is False

    def test_ungrounded_extraction_is_queued_for_review(self, router, stub, warehouse):
        stub.register("<document", _tone_json(evidence=["we expect revenue to double"]))
        result = extract(
            "guidance_tone",
            source_text=SOURCE,
            variables={"company": "T", "form": "10-Q", "period": "p", "section_text": SOURCE},
            accession="acc-1",
            cik="c1",
            router=router,
            chain="ci",
        )
        assert result.valid is True, "schema-valid but fabricated"
        assert result.grounded_ratio == 0.0
        assert result.review_reason() == "ungrounded"
        review = query("SELECT * FROM review_queue")[0]
        assert review["reason"] == "ungrounded"
        assert "we expect revenue to double" in review["proposed"]

    def test_low_confidence_is_queued_at_lower_priority(self, router, stub, warehouse):
        stub.register("<document", _tone_json(confidence=0.2))
        extract(
            "guidance_tone",
            source_text=SOURCE,
            variables={"company": "T", "form": "10-Q", "period": "p", "section_text": SOURCE},
            accession="acc-1",
            cik="c1",
            router=router,
            chain="ci",
        )
        rows = query("SELECT reason, priority FROM review_queue")
        assert rows[0]["reason"] == "low_confidence"
        assert rows[0]["priority"] < 9

    def test_confident_grounded_extraction_is_not_queued(self, router, stub, warehouse):
        stub.register("<document", _tone_json())
        extract(
            "guidance_tone",
            source_text=SOURCE,
            variables={"company": "T", "form": "10-Q", "period": "p", "section_text": SOURCE},
            accession="acc-1",
            cik="c1",
            router=router,
            chain="ci",
        )
        assert query("SELECT * FROM review_queue") == []

    def test_unknown_task_is_rejected(self, router):
        with pytest.raises(KeyError, match="unknown task"):
            extract("nope", source_text="x", variables={}, accession="a", cik="c", router=router)
