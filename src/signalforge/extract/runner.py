"""Extraction runner — the path from filing text to a validated, attributed record.

Sequence, and why each step exists:

    render prompt (versioned, hashed)
      -> route to model (chain, fallback, cost cap, cache)
        -> parse JSON loosely (local models add prose)
          -> validate against the Pydantic schema
            -> on failure, repair: show the model its own error, once or twice
              -> check evidence against the source text
                -> persist with full provenance
                  -> queue for human review if weak

The repair loop is the interesting part. Schema-constrained decoding makes it
almost unnecessary on hosted models, but local open-source models violate schemas
routinely — wrong enum casing, confidence as a percentage, an extra commentary
field. Rather than discard those responses (which biases the corpus toward easy
documents) or accept them loosely (which corrupts the data), the runner feeds the
validation error back and asks for a correction. The number of repairs used is
recorded per extraction, which makes "how well does this model follow a schema"
a measured property rather than an impression — and it is one of the clearest
discriminators in the model bake-off.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..db import connect, upsert
from ..llm.base import Message
from ..llm.router import Router
from ..obs.tracing import Tracer, default_tracer
from ..prompts.registry import Prompt, PromptRegistry, get_prompts
from .grounding import GroundingResult, check_grounding
from .schemas import SCHEMAS, ExtractionBase, json_schema_for

log = logging.getLogger(__name__)

MAX_REPAIRS = 2

#: Below this confidence, or with any ungrounded quote, an extraction goes to a
#: human. Chosen to keep the queue small enough that someone actually works it.
REVIEW_CONFIDENCE_FLOOR = 0.45


@dataclass
class ExtractionResult:
    task: str
    accession: str
    cik: str
    section_id: str | None
    payload: ExtractionBase | None
    prompt: Prompt
    model: str
    provider: str
    valid: bool
    repair_attempts: int = 0
    grounding: GroundingResult | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    error: str | None = None
    raw_text: str = field(default="", repr=False)

    @property
    def extraction_id(self) -> str:
        """Deterministic ID over everything that defines this experiment.

        Re-running the same task, on the same text, with the same prompt version
        and model overwrites its previous row rather than accumulating duplicates
        — so the warehouse holds current results, and history lives in
        ``eval_runs``.
        """
        basis = "|".join(
            [
                self.task,
                self.accession,
                self.section_id or "",
                self.prompt.key,
                self.prompt.hash,
                self.model,
            ]
        )
        return hashlib.sha256(basis.encode()).hexdigest()[:24]

    @property
    def grounded_ratio(self) -> float:
        return self.grounding.ratio if self.grounding else 0.0

    @property
    def needs_review(self) -> bool:
        if not self.valid:
            return True
        if self.grounding and self.grounding.hallucinated:
            return True
        return bool(self.payload and self.payload.confidence < REVIEW_CONFIDENCE_FLOOR)

    def review_reason(self) -> str:
        if not self.valid:
            return "invalid"
        if self.grounding and self.grounding.hallucinated:
            return "ungrounded"
        return "low_confidence"


def extract(
    task: str,
    *,
    source_text: str,
    variables: dict[str, Any],
    accession: str,
    cik: str,
    section_id: str | None = None,
    router: Router | None = None,
    prompts: PromptRegistry | None = None,
    prompt_version: str | None = None,
    chain: str | list[str] = "extract_default",
    max_tokens: int = 1024,
    max_repairs: int = MAX_REPAIRS,
    tracer: Tracer | None = None,
    persist: bool = True,
) -> ExtractionResult:
    """Run one extraction task over one piece of text.

    ``source_text`` is what evidence is checked against, and is kept separate
    from ``variables`` because a prompt may legitimately include context (a prior
    filing, company metadata) that quotes must *not* be drawn from.
    """
    if task not in SCHEMAS:
        raise KeyError(f"unknown task {task!r}; known: {sorted(SCHEMAS)}")

    router = router or Router()
    prompts = prompts or get_prompts()
    tracer = tracer or default_tracer
    prompt = prompts.get(task, prompt_version)
    schema_model = SCHEMAS[task]
    schema = json_schema_for(task)

    with tracer.span(
        f"extract.{task}", kind="pipeline", accession=accession, section=section_id or ""
    ) as span:
        span.set(prompt=prompt.key, prompt_hash=prompt.hash)

        rendered = prompt.render(**variables)
        messages = [Message(role="user", content=rendered)]

        result = ExtractionResult(
            task=task,
            accession=accession,
            cik=cik,
            section_id=section_id,
            payload=None,
            prompt=prompt,
            model="",
            provider="",
            valid=False,
        )

        last_error = ""
        for attempt in range(max_repairs + 1):
            try:
                resp = router.complete(
                    messages,
                    chain=chain,
                    json_schema=schema,
                    max_tokens=max_tokens,
                    span_name=f"llm.{task}",
                    attempt=attempt,
                )
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                span.set(status="error")
                break

            # Accumulate across repair attempts: the cost of getting a valid
            # answer is the cost of *all* the attempts, not just the last one.
            result.model = resp.model
            result.provider = resp.provider
            result.tokens_in += resp.usage.tokens_in
            result.tokens_out += resp.usage.tokens_out
            result.cost_usd += resp.cost_usd
            result.latency_ms += resp.latency_ms
            result.cached = resp.cached
            result.raw_text = resp.text
            result.repair_attempts = attempt

            try:
                payload = schema_model.model_validate(resp.json_payload())
            except (ValidationError, ValueError) as exc:
                last_error = _explain(exc)
                if attempt < max_repairs:
                    messages = [
                        Message(role="user", content=rendered),
                        Message(role="assistant", content=resp.text),
                        Message(role="user", content=_repair_instruction(last_error, schema)),
                    ]
                    continue
                result.error = last_error
                break

            result.payload = payload
            result.valid = True
            result.error = None
            break

        if result.payload is not None:
            result.grounding = check_grounding(result.payload.evidence, source_text)

        span.set(
            valid=result.valid,
            repair_attempts=result.repair_attempts,
            grounded_ratio=round(result.grounded_ratio, 3),
            confidence=result.payload.confidence if result.payload else None,
        )

    if persist:
        persist_extraction(result)
        if result.needs_review:
            queue_review(result)
    return result


def _explain(exc: Exception) -> str:
    """Turn a validation failure into something a model can act on.

    Pydantic's default rendering is verbose and includes URLs; a model repairs
    far more reliably from a short list of field-level problems.
    """
    if isinstance(exc, ValidationError):
        lines = []
        for err in exc.errors()[:6]:
            loc = ".".join(str(p) for p in err["loc"]) or "(root)"
            lines.append(f"- field '{loc}': {err['msg']}")
        return "Schema validation failed:\n" + "\n".join(lines)
    return f"Response was not valid JSON: {exc}"


def _repair_instruction(error: str, schema: dict[str, Any]) -> str:
    return (
        f"{error}\n\n"
        "Return the corrected JSON object only — no explanation, no markdown "
        "fence. It must conform exactly to this schema, using only the fields "
        "listed and only the allowed enum values:\n"
        f"{json.dumps(schema, indent=None)}"
    )


def persist_extraction(result: ExtractionResult) -> None:
    payload = result.payload.model_dump(mode="json") if result.payload else {}
    row = {
        "extraction_id": result.extraction_id,
        "task": result.task,
        "accession": result.accession,
        "cik": result.cik,
        "section_id": result.section_id,
        "prompt_name": result.prompt.name,
        "prompt_version": result.prompt.version,
        "prompt_hash": result.prompt.hash,
        "model": result.model,
        "provider": result.provider,
        "config_hash": result.prompt.hash,
        "payload": payload,
        "valid": result.valid,
        "repair_attempts": result.repair_attempts,
        "grounded_ratio": result.grounded_ratio,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms,
        "cached": result.cached,
        "error": result.error,
    }
    with connect() as con:
        upsert(con, "extractions", [row], key="extraction_id")


def queue_review(result: ExtractionResult) -> None:
    """Route a weak extraction to a human.

    Priority orders the queue by how bad the failure is: an invalid response is
    a bug, an ungrounded quote is a hallucination, low confidence is merely a
    hard document.
    """
    reason = result.review_reason()
    priority = {"invalid": 9, "ungrounded": 7, "low_confidence": 3}[reason]
    proposed: dict[str, Any] = {
        "payload": result.payload.model_dump(mode="json") if result.payload else None,
        "model": result.model,
        "prompt": result.prompt.key,
        "error": result.error,
        "raw": result.raw_text[:2000] if not result.valid else None,
    }
    if result.grounding and result.grounding.ungrounded_quotes:
        proposed["ungrounded_quotes"] = result.grounding.ungrounded_quotes

    with connect() as con:
        upsert(
            con,
            "review_queue",
            [
                {
                    "review_id": f"{result.task}:{result.extraction_id}",
                    "extraction_id": result.extraction_id,
                    "task": result.task,
                    "reason": reason,
                    "priority": priority,
                    "status": "open",
                    "proposed": proposed,
                }
            ],
            key="review_id",
        )
