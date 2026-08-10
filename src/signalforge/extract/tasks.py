"""Task definitions — what text each extraction reads, and how it is framed.

Each task declares how to assemble its inputs from the warehouse. That work is
non-trivial and task-specific: ``risk_delta`` needs the *prior* comparable filing
for the same company and form, ``event_class`` needs the 8-K item codes, and every
task needs its source text truncated to fit the model's context without cutting
mid-sentence.

Keeping this separate from ``runner.py`` means the runner stays a
schema-and-provenance engine that knows nothing about SEC forms, and adding a new
signal is a new entry here plus a prompt file.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ..db import query
from ..ingest.chunk import CHARS_PER_TOKEN
from ..llm.router import Router
from ..obs.tracing import Tracer
from ..prompts.registry import PromptRegistry
from .runner import ExtractionResult, extract

log = logging.getLogger(__name__)

#: Leave room for the schema, instructions, and the response itself.
PROMPT_OVERHEAD_TOKENS = 1200


@dataclass(slots=True)
class TaskUnit:
    """One extraction to run: the text, the prompt variables, and its provenance."""

    task: str
    accession: str
    cik: str
    section_id: str | None
    source_text: str
    variables: dict[str, Any]


def guidance_tone_units(
    *, cik: str | None = None, limit: int | None = None, budget_tokens: int = 8000
) -> list[TaskUnit]:
    """MD&A sections from periodic reports, plus 8-K results sections."""
    rows = query(
        f"""
        SELECT s.section_id, s.text, s.slug, f.accession, f.cik, f.form,
               coalesce(f.report_date, f.filing_date) AS period, c.name
        FROM sections s
        JOIN filings f ON f.accession = s.accession
        JOIN companies c ON c.cik = f.cik
        WHERE s.slug IN ('mdna', 'results_of_operations')
          {"AND f.cik = ?" if cik else ""}
        ORDER BY f.filing_date DESC
        {f"LIMIT {int(limit)}" if limit else ""}
        """,
        [cik] if cik else [],
    )
    return [
        TaskUnit(
            task="guidance_tone",
            accession=r["accession"],
            cik=r["cik"],
            section_id=r["section_id"],
            source_text=_fit(r["text"], budget_tokens),
            variables={
                "company": r["name"],
                "form": r["form"],
                "period": str(r["period"]),
                "section_text": _fit(r["text"], budget_tokens),
            },
        )
        for r in rows
    ]


def risk_delta_units(
    *, cik: str | None = None, limit: int | None = None, budget_tokens: int = 8000
) -> list[TaskUnit]:
    """Consecutive risk-factor sections, paired within company *and form*.

    Pairing across forms would compare a 10-K's full risk section against a 10-Q's
    abbreviated update and report a fabricated improvement every year — the
    disclosure did not change, only the form did. ``PARTITION BY form`` is what
    prevents that.
    """
    rows = query(
        f"""
        WITH risk AS (
            SELECT s.section_id, s.text, f.accession, f.cik, f.form,
                   f.filing_date,
                   coalesce(f.report_date, f.filing_date) AS period,
                   c.name,
                   lag(s.text)     OVER w AS prior_text,
                   lag(coalesce(f.report_date, f.filing_date)) OVER w AS prior_period
            FROM sections s
            JOIN filings f ON f.accession = s.accession
            JOIN companies c ON c.cik = f.cik
            WHERE s.slug = 'risk_factors' AND f.form IN ('10-K', '10-Q')
              {"AND f.cik = ?" if cik else ""}
            WINDOW w AS (PARTITION BY f.cik, f.form ORDER BY f.filing_date)
        )
        SELECT * FROM risk
        WHERE prior_text IS NOT NULL
        ORDER BY filing_date DESC
        {f"LIMIT {int(limit)}" if limit else ""}
        """,
        [cik] if cik else [],
    )
    units = []
    for r in rows:
        # The two documents share the budget; the current filing gets more of it
        # because that is what evidence must be quoted from.
        current = _fit(r["text"], int(budget_tokens * 0.6))
        prior = _fit(r["prior_text"], int(budget_tokens * 0.4))
        units.append(
            TaskUnit(
                task="risk_delta",
                accession=r["accession"],
                cik=r["cik"],
                section_id=r["section_id"],
                # Grounding is checked against the current filing only, matching
                # what the prompt instructs.
                source_text=current,
                variables={
                    "company": r["name"],
                    "current_period": str(r["period"]),
                    "prior_period": str(r["prior_period"]),
                    "current_text": current,
                    "prior_text": prior,
                },
            )
        )
    return units


def event_class_units(
    *, cik: str | None = None, limit: int | None = None, budget_tokens: int = 6000
) -> list[TaskUnit]:
    """8-K filings, one extraction per filing (not per item).

    An 8-K reports one event even when it lists several items — 2.02 plus 9.01 is
    an earnings release with exhibits, not two events. Concatenating the item
    sections keeps the classification aligned with the thing being classified.
    """
    rows = query(
        f"""
        SELECT f.accession, f.cik, f.form, f.items, f.filing_date, c.name,
               string_agg(s.heading || '\n' || s.text, '\n\n' ORDER BY s.ordinal) AS text
        FROM filings f
        JOIN sections s ON s.accession = f.accession
        JOIN companies c ON c.cik = f.cik
        WHERE f.form = '8-K'
          {"AND f.cik = ?" if cik else ""}
        GROUP BY f.accession, f.cik, f.form, f.items, f.filing_date, c.name
        ORDER BY f.filing_date DESC
        {f"LIMIT {int(limit)}" if limit else ""}
        """,
        [cik] if cik else [],
    )
    return [
        TaskUnit(
            task="event_class",
            accession=r["accession"],
            cik=r["cik"],
            section_id=None,
            source_text=_fit(r["text"], budget_tokens),
            variables={
                "company": r["name"],
                "items": r["items"] or "(none reported)",
                "filing_date": str(r["filing_date"]),
                "section_text": _fit(r["text"], budget_tokens),
            },
        )
        for r in rows
    ]


UNIT_BUILDERS = {
    "guidance_tone": guidance_tone_units,
    "risk_delta": risk_delta_units,
    "event_class": event_class_units,
}


def build_units(task: str, **kwargs: Any) -> list[TaskUnit]:
    try:
        return UNIT_BUILDERS[task](**kwargs)
    except KeyError:
        raise KeyError(f"unknown task {task!r}; known: {sorted(UNIT_BUILDERS)}") from None


def run_task(
    task: str,
    *,
    router: Router | None = None,
    prompts: PromptRegistry | None = None,
    prompt_version: str | None = None,
    chain: str | list[str] = "extract_default",
    cik: str | None = None,
    limit: int | None = None,
    tracer: Tracer | None = None,
    persist: bool = True,
) -> list[ExtractionResult]:
    """Run one extraction task across everything in the warehouse that qualifies."""
    return list(
        iter_task(
            task,
            router=router,
            prompts=prompts,
            prompt_version=prompt_version,
            chain=chain,
            cik=cik,
            limit=limit,
            tracer=tracer,
            persist=persist,
        )
    )


def iter_task(
    task: str,
    *,
    router: Router | None = None,
    prompts: PromptRegistry | None = None,
    prompt_version: str | None = None,
    chain: str | list[str] = "extract_default",
    cik: str | None = None,
    limit: int | None = None,
    tracer: Tracer | None = None,
    persist: bool = True,
) -> Iterator[ExtractionResult]:
    """Streaming variant, so a long backfill reports progress as it goes."""
    router = router or Router()
    units = build_units(task, cik=cik, limit=limit)
    log.info("task=%s units=%d chain=%s", task, len(units), chain)

    for unit in units:
        yield extract(
            unit.task,
            source_text=unit.source_text,
            variables=unit.variables,
            accession=unit.accession,
            cik=unit.cik,
            section_id=unit.section_id,
            router=router,
            prompts=prompts,
            prompt_version=prompt_version,
            chain=chain,
            tracer=tracer,
            persist=persist,
        )


def _fit(text: str, budget_tokens: int) -> str:
    """Truncate to a token budget at a paragraph or sentence boundary.

    Cutting mid-sentence is worse than it looks: the trailing fragment reads as a
    complete-but-garbled statement, and models will happily quote it as evidence —
    which then fails grounding against the untruncated source.
    """
    text = (text or "").strip()
    max_chars = int(max(budget_tokens - PROMPT_OVERHEAD_TOKENS, 500) * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text

    window = text[:max_chars]
    for boundary in ("\n\n", ". ", "\n"):
        cut = window.rfind(boundary)
        # Only accept a boundary in the last quarter, or we would discard most of
        # the section to find a tidy break.
        if cut > max_chars * 0.75:
            return window[: cut + len(boundary)].strip() + "\n\n[truncated]"
    return window.strip() + "\n\n[truncated]"
