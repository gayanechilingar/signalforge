"""Ingest orchestration: EDGAR -> parse -> chunk -> warehouse.

The one design decision worth stating: ingest **records how well it parsed** and
routes thin parses to the human review queue instead of failing or, worse,
silently storing a filing whose Item 1A is actually a cross-reference.

That matters because filers are heterogeneous in ways no parser fully absorbs —
JPMorgan incorporates MD&A by reference to its annual report; Tesla's 10-Q risk
section points back at the 10-K. If those land in the warehouse indistinguishable
from a clean parse, every downstream signal inherits the defect and the eval
suite blames the model for a data bug. Coverage is therefore a first-class,
queryable property of an ingested filing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..db import connect, upsert
from ..obs.tracing import Tracer, default_tracer
from .chunk import chunk_text
from .edgar import SUPPORTED_FORMS, Company, EdgarClient, Filing
from .parse import ParsedFiling, parse_filing

log = logging.getLogger(__name__)

#: Minimum share of document text that must land in named sections, per form.
#:
#: This is form-specific because the forms are shaped differently. A 10-K or 10-Q
#: is almost entirely numbered items, so low coverage means the parser lost
#: content. An 8-K is a two-page event notice wrapped in a cover page — registrant
#: address, four checkbox rows, exhibit index, signature block — none of which is
#: an item, so 30% coverage is a *correct* parse of an 8-K.
#:
#: Applying the periodic-report threshold to 8-Ks flagged every single one, which
#: is worse than not flagging at all: a review queue that cries wolf gets ignored,
#: and the real defects hide among the noise.
MIN_SECTION_COVERAGE = {"10-K": 0.60, "10-Q": 0.60, "8-K": 0.10}
DEFAULT_MIN_COVERAGE = 0.30

#: Sections a signal pipeline actually needs. A periodic report missing all of
#: them parsed badly, however good the coverage number looks.
KEY_SECTIONS = ("risk_factors", "mdna")


@dataclass
class IngestReport:
    companies: int = 0
    filings: int = 0
    sections: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "companies": self.companies,
            "filings": self.filings,
            "sections": self.sections,
            "chunks": self.chunks,
            "skipped": len(self.skipped),
            "flagged": len(self.flagged),
        }


def ingest_company(
    cik_or_ticker: str,
    *,
    client: EdgarClient | None = None,
    forms: tuple[str, ...] = SUPPORTED_FORMS,
    since: date | None = None,
    limit: int | None = 12,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    tracer: Tracer | None = None,
) -> IngestReport:
    """Ingest one company's recent filings into the warehouse."""
    tracer = tracer or default_tracer
    own_client = client is None
    client = client or EdgarClient()
    report = IngestReport()

    try:
        with tracer.span("ingest.company", kind="pipeline", cik=cik_or_ticker) as span:
            cik = _resolve(client, cik_or_ticker)
            company = client.company(cik)
            _write_company(company)
            report.companies = 1

            filings = client.filings(cik, forms=forms, since=since, limit=limit)
            span.set(filings_found=len(filings))

            for filing, html in client.iter_documents(filings):
                try:
                    n_sections, n_chunks, flagged = _ingest_filing(
                        filing,
                        html,
                        max_tokens=max_tokens,
                        overlap_tokens=overlap_tokens,
                        tracer=tracer,
                    )
                except Exception as exc:
                    # One malformed filing must not abort a backfill; the skip is
                    # recorded so it is visible rather than lost.
                    log.warning("failed to ingest %s: %s", filing.accession, exc)
                    report.skipped.append(f"{filing.accession}: {exc}")
                    continue
                report.filings += 1
                report.sections += n_sections
                report.chunks += n_chunks
                if flagged:
                    report.flagged.append(filing.accession)

            span.set(**report.as_dict())
    finally:
        if own_client:
            client.close()
    return report


def _resolve(client: EdgarClient, value: str) -> str:
    """Accept either a CIK or a ticker, transparently."""
    try:
        return client._normalise_cik(value)
    except Exception:
        return client.resolve_ticker(value)


def _write_company(company: Company) -> None:
    with connect() as con:
        upsert(
            con,
            "companies",
            [
                {
                    "cik": company.cik,
                    "ticker": company.ticker,
                    "name": company.name,
                    "sic": company.sic,
                    "sic_description": company.sic_description,
                    "exchange": company.exchange,
                }
            ],
            key="cik",
        )


def _ingest_filing(
    filing: Filing,
    html: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    tracer: Tracer,
) -> tuple[int, int, bool]:
    with tracer.span(
        "ingest.filing", kind="pipeline", accession=filing.accession, form=filing.form
    ) as span:
        parsed = parse_filing(html, form=filing.form)
        if not parsed.text:
            raise ValueError("document produced no text")

        quality = assess_parse(parsed, form=filing.form)
        span.set(**quality)

        section_rows = []
        chunk_rows = []
        for section in parsed.sections:
            section_id = f"{filing.accession}:{section.slug}"
            section_rows.append(
                {
                    "section_id": section_id,
                    "accession": filing.accession,
                    "slug": section.slug,
                    "heading": section.heading,
                    "ordinal": section.ordinal,
                    "char_len": section.char_len,
                    "text": section.text,
                }
            )
            for chunk in chunk_text(
                section.text, max_tokens=max_tokens, overlap_tokens=overlap_tokens
            ):
                chunk_rows.append(
                    {
                        "chunk_id": f"{section_id}:{chunk.ordinal}",
                        "section_id": section_id,
                        "accession": filing.accession,
                        "cik": filing.cik,
                        "ordinal": chunk.ordinal,
                        "token_estimate": chunk.token_estimate,
                        "text": chunk.text,
                    }
                )

        with connect() as con:
            upsert(
                con,
                "filings",
                [
                    {
                        "accession": filing.accession,
                        "cik": filing.cik,
                        "form": filing.form,
                        "filing_date": filing.filing_date,
                        "report_date": filing.report_date,
                        "items": filing.items,
                        "primary_doc": filing.primary_doc,
                        "url": filing.url,
                        "size_bytes": filing.size_bytes,
                    }
                ],
                key="accession",
            )
            # Replace rather than accumulate: re-ingesting a filing after a parser
            # change must not leave the previous parse's sections behind.
            con.execute("DELETE FROM sections WHERE accession = ?", [filing.accession])
            con.execute("DELETE FROM chunks WHERE accession = ?", [filing.accession])
            upsert(con, "sections", section_rows, key="section_id")
            upsert(con, "chunks", chunk_rows, key="chunk_id")

        flagged = not quality["parse_ok"]
        if flagged:
            _queue_parse_review(filing, quality)

        span.set(sections=len(section_rows), chunks=len(chunk_rows))
        return len(section_rows), len(chunk_rows), flagged


def assess_parse(parsed: ParsedFiling, *, form: str) -> dict[str, Any]:
    """Score how much of a filing we actually understood.

    Returned as data rather than raised as an error, because a low-coverage parse
    is still worth storing — it just must not be mistaken for a good one.
    """
    total = max(len(parsed.text), 1)
    covered = sum(s.char_len for s in parsed.sections)
    coverage = min(covered / total, 1.0)
    slugs = {s.slug for s in parsed.sections}
    fallback_only = all(s.is_fallback for s in parsed.sections) if parsed.sections else True

    form_key = form.upper()
    is_periodic = form_key in ("10-K", "10-Q")
    has_key = bool(slugs & set(KEY_SECTIONS))
    threshold = MIN_SECTION_COVERAGE.get(form_key, DEFAULT_MIN_COVERAGE)

    reasons = []
    if fallback_only:
        reasons.append("no_sections_detected")
    if coverage < threshold:
        reasons.append(f"low_coverage_{coverage:.2f}")
    if is_periodic and not has_key:
        reasons.append("missing_key_sections")

    return {
        "section_coverage": round(coverage, 3),
        "coverage_threshold": threshold,
        "n_sections": len(parsed.sections),
        "found_key_sections": sorted(slugs & set(KEY_SECTIONS)),
        "parse_ok": not reasons,
        "parse_issues": reasons,
    }


def _queue_parse_review(filing: Filing, quality: dict[str, Any]) -> None:
    """Surface a suspect parse for a human rather than burying it in a log line."""
    with connect() as con:
        upsert(
            con,
            "review_queue",
            [
                {
                    "review_id": f"parse:{filing.accession}",
                    "extraction_id": "",
                    "task": "ingest_parse",
                    "reason": "invalid",
                    "priority": 5,
                    "status": "open",
                    "proposed": {
                        "accession": filing.accession,
                        "form": filing.form,
                        "url": filing.url,
                        **quality,
                    },
                }
            ],
            key="review_id",
        )
