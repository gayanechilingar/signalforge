"""Filing HTML to section-aware plain text.

Why sections instead of whole documents
---------------------------------------
A 10-K runs 100k+ tokens and most of it is boilerplate and financial tables. Two
consequences drive this module:

1. **Signals are section-specific.** "Did risk disclosure get worse?" is a
   question about Item 1A. Feeding the whole filing in dilutes the signal and
   multiplies cost.
2. **Comparability requires alignment.** Measuring a risk-factor delta between
   two filings means comparing Item 1A to Item 1A, not document to document.

Parsing filings is genuinely messy — inline XBRL, nested tables, headings that
are styled `<div>`s rather than `<h*>` tags, and item labels that appear in the
table of contents before they appear as real headings. The approach here is
deliberately conservative: find item headings by regex over the extracted text,
require them to look like headings (short line, at a line start), and drop
candidates that are obviously table-of-contents entries. When section detection
fails, fall back to a single ``body`` section rather than silently returning
nothing — a filing with no usable sections should still be ingestible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from selectolax.parser import HTMLParser

#: Tags whose contents are markup, not prose.
_DROP_TAGS = ("script", "style", "head", "noscript", "svg")

#: Tags that end a line of text. Everything else is treated as inline.
#:
#: This distinction is load-bearing, not cosmetic. Filers style the first letter
#: of a heading as a drop cap, so "ITEM 1A. RISK FACTORS" is markup like
#: ``<span>ITEM 1A. R</span><span>ISK FACTORS</span>``. Inserting a newline
#: between *every* node splits that heading into "ITEM 1A. R" and "ISK FACTORS",
#: and heading detection then finds an item with no readable title — which is how
#: a 350KB 10-K ends up with a 5KB "risk factors" section.
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "hr",
        "table",
        "tr",
        "td",
        "th",
        "tbody",
        "thead",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
        "section",
        "article",
        "header",
        "footer",
        "blockquote",
        "pre",
        "body",
    }
)

#: Canonical slugs resolved from *heading text*, not item number.
#:
#: Item numbers are not stable across forms — in a 10-K, Item 1 is Business and
#: Item 7 is MD&A; in a 10-Q, Item 1 is Financial Statements and Item 2 is MD&A.
#: A number-keyed map therefore silently mislabels every 10-Q, which would corrupt
#: signals in a way no downstream test would catch (the text is real, just filed
#: under the wrong name). Heading titles say what a section actually is, so they
#: are the primary key and the item number is only a fallback.
#:
#: Ordered: the first matching pattern wins, so more specific patterns come first.
TITLE_SLUGS: tuple[tuple[str, str], ...] = (
    (r"risk factors", "risk_factors"),
    (r"unresolved staff comments", "unresolved_staff_comments"),
    (r"management.s discussion and analysis", "mdna"),
    (r"quantitative and qualitative disclosures? about market risk", "market_risk"),
    (r"controls and procedures", "controls_and_procedures"),
    (r"legal proceedings", "legal_proceedings"),
    (r"financial statements and supplementary data", "financial_statements"),
    (r"financial statements", "financial_statements"),
    (r"market for (the )?registrant", "market_for_equity"),
    (r"^business$", "business"),
    (r"^properties$", "properties"),
    (r"other information", "other_information"),
    (r"^exhibits", "exhibits"),
    (r"mine safety", "mine_safety"),
)

#: Fallback slugs by 10-K item number, used only when the heading carries no
#: usable title (some filers emit a bare "Item 1A." on its own line).
TENK_ITEM_SLUGS = {
    "1": "business",
    "1a": "risk_factors",
    "1b": "unresolved_staff_comments",
    "2": "properties",
    "3": "legal_proceedings",
    "5": "market_for_equity",
    "7": "mdna",
    "7a": "market_risk",
    "8": "financial_statements",
    "9a": "controls_and_procedures",
}

#: 8-K item codes worth naming; these are the event types that move prices.
EIGHTK_SLUGS = {
    "1.01": "material_agreement",
    "1.03": "bankruptcy",
    "2.01": "completed_acquisition",
    "2.02": "results_of_operations",
    "2.05": "exit_or_disposal_costs",
    "2.06": "material_impairment",
    "3.01": "delisting_notice",
    "4.01": "auditor_change",
    "4.02": "non_reliance_on_financials",
    "5.02": "director_officer_change",
    "7.01": "reg_fd_disclosure",
    "8.01": "other_events",
}

# "Item 1A." / "ITEM 7 -" / "Item 2.02" — the leading anchor is what makes this
# usable; matching "item" anywhere in prose produces mostly false positives.
_ITEM_RE = re.compile(
    r"^\s*item\s+(\d{1,2}(?:\.\d{2})?[a-c]?)\s*[.:—\-–]?\s*(.{0,120})$",
    re.I | re.M,
)


@dataclass(slots=True)
class Section:
    slug: str
    heading: str
    ordinal: int
    text: str
    #: Set when the section came from the whole-document fallback rather than a
    #: detected heading. Recorded so eval reports can separate "the model was
    #: wrong" from "we fed it the wrong text".
    is_fallback: bool = False

    @property
    def char_len(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class ParsedFiling:
    text: str
    sections: list[Section] = field(default_factory=list)

    def section(self, slug: str) -> Section | None:
        for s in self.sections:
            if s.slug == slug:
                return s
        return None


def html_to_text(html: str) -> str:
    """Extract readable text, breaking lines only at block boundaries.

    Line structure has to be right for heading detection to work at all: too few
    newlines and a filing is one unsearchable line, too many and styled headings
    get torn in half (see :data:`_BLOCK_TAGS`).
    """
    tree = HTMLParser(html)
    for tag in _DROP_TAGS:
        for node in tree.css(tag):
            node.decompose()

    root = tree.body or tree.root
    if root is None:
        return ""

    parts: list[str] = []
    for node in root.traverse(include_text=True):
        if node.tag == "-text":
            chunk = node.text(deep=False)
            if chunk:
                parts.append(chunk)
        elif node.tag in _BLOCK_TAGS:
            parts.append("\n")
    text = "".join(parts)

    text = text.replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse the long runs of empty lines that table markup leaves behind,
    # while keeping paragraph breaks.
    text = re.sub(r"\n\s*\n\s*(\n\s*)+", "\n\n", text)
    text = re.sub(r"\n +", "\n", text)
    return _strip_running_headers(text).strip()


def _strip_running_headers(text: str, *, min_repeats: int = 5) -> str:
    """Remove page headers and footers that repeat throughout a filing.

    Paginated filings stamp lines like "PART I", "Item 1", or the company name on
    every page. Left in, they fabricate dozens of spurious item headings, and the
    table-of-contents heuristic cannot tell them apart from real ones. Short lines
    that recur many times are page furniture, not content — a real heading appears
    once or twice.
    """
    lines = text.split("\n")
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if 0 < len(stripped) <= 40:
            counts[stripped] = counts.get(stripped, 0) + 1

    furniture = {
        line for line, n in counts.items() if n >= min_repeats and not _is_substantive(line)
    }
    if not furniture:
        return text
    return "\n".join(line for line in lines if line.strip() not in furniture)


def _is_substantive(line: str) -> bool:
    """True for repeated lines worth keeping even so.

    Guards against deleting genuine repeated content — a table label like "Total
    revenue" legitimately appears on many pages and carries meaning.
    """
    low = line.lower().strip(" .:|")
    if re.fullmatch(r"(part\s+[ivx]+|item\s+\d{1,2}[a-c]?|page\s*\d*|\d+|[ivxlc]+)", low):
        return False
    # Anything with several words and no page-furniture shape is probably content.
    return len(low.split()) >= 4


def parse_filing(html: str, *, form: str) -> ParsedFiling:
    """Parse a filing into named sections.

    ``form`` selects the slug vocabulary — 8-K items are numbered differently
    from 10-K items and mean entirely different things.
    """
    text = html_to_text(html)
    if not text:
        return ParsedFiling(text="", sections=[])

    is_8k = form.upper().startswith("8-K")
    sections = _split_items(text, is_8k=is_8k)

    if not sections:
        sections = [
            Section(
                slug="body",
                heading=form,
                ordinal=0,
                text=text,
                is_fallback=True,
            )
        ]
    return ParsedFiling(text=text, sections=sections)


def _split_items(text: str, *, is_8k: bool) -> list[Section]:
    matches = [m for m in _ITEM_RE.finditer(text) if _looks_like_heading(m, text)]
    matches = _drop_table_of_contents(matches)
    if not matches:
        return []

    sections: list[Section] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # A "section" of a few hundred characters is almost always a cross
        # reference or a stub ("Item 1B. None."), not content worth extracting.
        if len(body) < 200:
            continue

        code = m.group(1).lower().rstrip(".")
        title = (m.group(2) or "").strip(" .:-–—")
        sections.append(
            Section(
                slug=_resolve_slug(code, title, is_8k=is_8k),
                heading=f"Item {m.group(1)}" + (f". {title}" if title else ""),
                ordinal=len(sections),
                text=body,
            )
        )
    return _dedupe_keep_longest(sections)


def _resolve_slug(code: str, title: str, *, is_8k: bool) -> str:
    """Name a section by what it says it is, falling back to its number.

    8-K item codes are globally unique and semantically stable (2.02 is always
    results of operations), so for 8-Ks the code is authoritative. For periodic
    reports the title wins, because the number is form-dependent.
    """
    if is_8k:
        return EIGHTK_SLUGS.get(code, f"item_{code.replace('.', '_')}")

    normalised = re.sub(r"\s+", " ", title).strip().lower()
    if normalised:
        for pattern, slug in TITLE_SLUGS:
            if re.search(pattern, normalised):
                return slug
    return TENK_ITEM_SLUGS.get(code, f"item_{code.replace('.', '_')}")


def _looks_like_heading(m: re.Match[str], text: str) -> bool:
    """Reject in-prose references like "as described in Item 1A above".

    The test is the shape of the line, not the words: a real heading occupies a
    short line of its own, whereas a cross-reference sits mid-sentence.
    """
    line_start = text.rfind("\n", 0, m.start()) + 1
    line_end = text.find("\n", m.start())
    line = text[line_start : line_end if line_end != -1 else len(text)].strip()
    if len(line) > 140:
        return False
    # A heading line does not end mid-sentence.
    return not line.endswith((",", ";", "and", "or", "in", "of", "to"))


def _drop_table_of_contents(matches: list[re.Match[str]]) -> list[re.Match[str]]:
    """Discard the TOC run at the top of the document.

    Filings list every item twice: once in the table of contents and once as the
    real heading. Keeping both would produce a set of near-empty leading sections
    and misattribute the real Item 1A text. The TOC is identifiable as a dense
    cluster of item matches separated by very little text.
    """
    if len(matches) < 4:
        return matches

    gaps = [matches[i + 1].start() - matches[i].end() for i in range(len(matches) - 1)]
    # Walk forward while consecutive matches are packed together.
    cut = 0
    for i, gap in enumerate(gaps):
        if gap < 200:
            cut = i + 1
        else:
            break
    # Only treat it as a TOC if it accounts for several entries and leaves real
    # headings behind; otherwise we would delete the document's only sections.
    if cut >= 3 and len(matches) - cut >= 2:
        return matches[cut:]
    return matches


def _dedupe_keep_longest(sections: list[Section]) -> list[Section]:
    """Keep the longest occurrence of each slug.

    Amended filings and exhibit-heavy documents repeat item headings. The longest
    span is the substantive one; the short ones are cross references or exhibit
    lists.
    """
    best: dict[str, Section] = {}
    for s in sections:
        current = best.get(s.slug)
        if current is None or s.char_len > current.char_len:
            best[s.slug] = s
    ordered = sorted(best.values(), key=lambda s: s.ordinal)
    for i, s in enumerate(ordered):
        s.ordinal = i
    return ordered
