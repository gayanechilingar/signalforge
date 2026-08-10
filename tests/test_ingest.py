"""Parsing, chunking, and parse-quality assessment.

The parser tests encode filing quirks that actually broke this code against real
EDGAR documents — drop-cap headings, running page headers, table-of-contents
duplication, and 10-Q vs 10-K item renumbering. They are regression tests in the
literal sense: each one failed once.
"""

from __future__ import annotations

import pytest

from signalforge.ingest.chunk import chunk_text, estimate_tokens
from signalforge.ingest.edgar import EdgarClient, EdgarError, Filing, RateLimiter
from signalforge.ingest.parse import html_to_text, parse_filing
from signalforge.ingest.store import assess_parse

BODY = " ".join(["The Company faces substantial competitive pressure."] * 40)


def _filing_html(items: list[tuple[str, str]], *, toc: bool = True) -> str:
    parts = ["<html><body>"]
    if toc:
        parts += [f"<div>Item {code}. {title}</div>" for code, title in items]
    for code, title in items:
        parts.append(f"<p>Item {code}. {title}</p><p>{BODY}</p>")
    parts.append("</body></html>")
    return "".join(parts)


class TestHtmlToText:
    def test_strips_script_and_style(self):
        text = html_to_text(
            "<html><body><style>p{color:red}</style><p>Hello</p>"
            "<script>alert(1)</script></body></html>"
        )
        assert text == "Hello"

    def test_block_tags_become_line_breaks(self):
        text = html_to_text("<body><p>One</p><p>Two</p></body>")
        assert text.split("\n") == ["One", "Two"]

    def test_inline_spans_do_not_split_words(self):
        """Drop caps are markup: <span>R</span><span>ISK</span> is one word.

        Splitting here is what made a 350KB 10-K parse into a 5KB risk section.
        """
        text = html_to_text("<body><p><span>ITEM 1A. R</span><span>ISK FACTORS</span></p></body>")
        assert text == "ITEM 1A. RISK FACTORS"

    def test_removes_repeated_page_furniture(self):
        html = (
            "<body>"
            + "".join(
                f"<p>PART I</p><p>Item 1</p><p>Real content paragraph {i} here.</p>"
                for i in range(8)
            )
            + "</body>"
        )
        text = html_to_text(html)
        assert "PART I" not in text
        assert "Real content paragraph 3 here." in text

    def test_keeps_repeated_substantive_lines(self):
        """Table labels legitimately repeat; only page furniture is furniture."""
        html = "<body>" + "<p>Total net sales by segment</p>" * 8 + "</body>"
        assert "Total net sales by segment" in html_to_text(html)

    def test_empty_document(self):
        assert html_to_text("") == ""


class TestSectionSplitting:
    def test_finds_named_sections(self):
        html = _filing_html([("1A", "Risk Factors"), ("7", "Management's Discussion and Analysis")])
        parsed = parse_filing(html, form="10-K")
        assert {s.slug for s in parsed.sections} == {"risk_factors", "mdna"}

    def test_slug_comes_from_title_not_item_number(self):
        """A 10-Q numbers MD&A as Item 2; a 10-K numbers it Item 7.

        Keying slugs on the number mislabels every 10-Q — the text is real but
        filed under the wrong name, which no downstream test would catch.
        """
        parsed = parse_filing(
            _filing_html([("2", "Management's Discussion and Analysis")]), form="10-Q"
        )
        assert [s.slug for s in parsed.sections] == ["mdna"]
        assert (
            parse_filing(_filing_html([("2", "Properties")]), form="10-K").sections[0].slug
            == "properties"
        )

    def test_table_of_contents_is_not_mistaken_for_content(self):
        items = [
            ("1", "Business"),
            ("1A", "Risk Factors"),
            ("2", "Properties"),
            ("3", "Legal Proceedings"),
            ("7", "Management's Discussion and Analysis"),
        ]
        parsed = parse_filing(_filing_html(items, toc=True), form="10-K")
        risk = parsed.section("risk_factors")
        assert risk is not None
        # Real body text, not the one-line TOC entry.
        assert risk.char_len > 1000

    def test_eightk_uses_item_codes(self):
        html = _filing_html([("2.02", "Results of Operations and Financial Condition")])
        parsed = parse_filing(html, form="8-K")
        assert parsed.sections[0].slug == "results_of_operations"

    def test_inline_item_reference_is_ignored(self):
        html = (
            "<body><p>Item 1A. Risk Factors</p><p>" + BODY + "</p>"
            "<p>As further described in Item 7 of this report, revenue grew, and "
            "the trends discussed in Item 1A above remain relevant to our outlook "
            "for the coming fiscal year and beyond.</p></body>"
        )
        parsed = parse_filing(html, form="10-K")
        assert [s.slug for s in parsed.sections] == ["risk_factors"]

    def test_falls_back_to_whole_body_when_no_items_found(self):
        parsed = parse_filing(f"<body><p>{BODY}</p></body>", form="10-K")
        assert len(parsed.sections) == 1
        assert parsed.sections[0].slug == "body"
        assert parsed.sections[0].is_fallback is True

    def test_stub_sections_are_dropped(self):
        html = (
            "<body><p>Item 1B. Unresolved Staff Comments</p><p>None.</p>"
            f"<p>Item 1A. Risk Factors</p><p>{BODY}</p></body>"
        )
        parsed = parse_filing(html, form="10-K")
        assert "unresolved_staff_comments" not in {s.slug for s in parsed.sections}


class TestParseQuality:
    def test_clean_parse_passes(self):
        parsed = parse_filing(
            _filing_html([("1A", "Risk Factors"), ("7", "Management's Discussion and Analysis")]),
            form="10-K",
        )
        q = assess_parse(parsed, form="10-K")
        assert q["parse_ok"] is True
        assert set(q["found_key_sections"]) == {"risk_factors", "mdna"}

    def test_fallback_only_parse_is_flagged(self):
        parsed = parse_filing(f"<body><p>{BODY}</p></body>", form="10-K")
        q = assess_parse(parsed, form="10-K")
        assert q["parse_ok"] is False
        assert "no_sections_detected" in q["parse_issues"]

    def test_periodic_report_without_key_sections_is_flagged(self):
        parsed = parse_filing(_filing_html([("3", "Legal Proceedings")]), form="10-Q")
        q = assess_parse(parsed, form="10-Q")
        assert "missing_key_sections" in q["parse_issues"]

    def test_eightk_is_not_expected_to_have_key_sections(self):
        parsed = parse_filing(
            _filing_html([("2.02", "Results of Operations and Financial Condition")]),
            form="8-K",
        )
        assert "missing_key_sections" not in assess_parse(parsed, form="8-K")["parse_issues"]

    def test_eightk_cover_page_boilerplate_does_not_trip_the_coverage_floor(self):
        """A real 8-K is ~30% items and ~70% cover page.

        Judging it against the periodic-report threshold flagged every 8-K in the
        corpus — a review queue that cries wolf gets ignored, so the threshold is
        per-form.
        """
        cover = "<p>" + " ".join(["Registrant address and checkbox boilerplate."] * 60) + "</p>"
        html = (
            "<body>"
            + cover
            + "<p>Item 2.02. Results of Operations and Financial Condition</p>"
            + f"<p>{BODY}</p></body>"
        )
        parsed = parse_filing(html, form="8-K")
        quality = assess_parse(parsed, form="8-K")
        assert quality["section_coverage"] < 0.60, "precondition: cover page dominates"
        assert quality["parse_ok"] is True

    def test_periodic_report_is_still_held_to_the_strict_floor(self):
        cover = "<p>" + " ".join(["Cover page boilerplate text here."] * 200) + "</p>"
        html = "<body>" + cover + "<p>Item 1A. Risk Factors</p>" + f"<p>{BODY}</p></body>"
        quality = assess_parse(parse_filing(html, form="10-K"), form="10-K")
        assert quality["parse_ok"] is False
        assert any(r.startswith("low_coverage") for r in quality["parse_issues"])


class TestChunking:
    def test_respects_max_tokens(self):
        chunks = chunk_text(BODY * 6, max_tokens=100, overlap_tokens=10)
        assert len(chunks) > 1
        # Allow slack for the final merge of a short tail.
        assert all(c.token_estimate <= 160 for c in chunks)

    def test_chunks_overlap(self):
        chunks = chunk_text(
            " ".join(f"Sentence number {i} about disclosure." for i in range(200)),
            max_tokens=80,
            overlap_tokens=24,
        )
        assert len(chunks) > 2
        overlap = set(chunks[0].text.split()) & set(chunks[1].text.split())
        assert len(overlap) > 2

    def test_does_not_split_mid_sentence_when_avoidable(self):
        text = " ".join(f"This is sentence {i}." for i in range(60))
        for c in chunk_text(text, max_tokens=60, overlap_tokens=8):
            assert c.text.strip().endswith(".")

    def test_short_tail_is_merged_not_orphaned(self):
        chunks = chunk_text(BODY + " Tiny.", max_tokens=200, overlap_tokens=20, min_tokens=40)
        assert all(c.token_estimate >= 20 for c in chunks)

    def test_oversized_single_sentence_is_hard_split(self):
        chunks = chunk_text("x" * 5000, max_tokens=100, overlap_tokens=0)
        assert len(chunks) > 1

    def test_empty_input(self):
        assert chunk_text("   ") == []

    def test_offsets_are_monotonic(self):
        chunks = chunk_text(BODY * 4, max_tokens=100, overlap_tokens=16)
        starts = [c.start_char for c in chunks]
        assert starts == sorted(starts)

    def test_overlap_must_be_smaller_than_max(self):
        with pytest.raises(ValueError, match="overlap_tokens"):
            chunk_text("hello", max_tokens=10, overlap_tokens=10)

    def test_token_estimate_is_positive(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("a" * 360) == 100


class TestEdgarClient:
    def test_cik_normalisation(self):
        assert EdgarClient._normalise_cik("320193") == "0000320193"
        assert EdgarClient._normalise_cik("0000320193") == "0000320193"
        assert EdgarClient._normalise_cik("CIK0000320193") == "0000320193"

    def test_ticker_is_rejected_not_guessed(self):
        with pytest.raises(EdgarError, match="not a CIK"):
            EdgarClient._normalise_cik("AAPL")

    def test_filing_url_construction(self):
        f = Filing(
            accession="0000320193-26-000020",
            cik="0000320193",
            form="10-Q",
            filing_date=__import__("datetime").date(2026, 7, 31),
            report_date=None,
            primary_doc="aapl-20260627.htm",
        )
        assert f.url == (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019326000020/aapl-20260627.htm"
        )

    def test_rate_limiter_enforces_spacing(self):
        import time

        limiter = RateLimiter(per_second=50)
        limiter.wait()
        t0 = time.monotonic()
        limiter.wait()
        assert time.monotonic() - t0 >= 0.015
