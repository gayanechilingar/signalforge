"""Citation grounding — the hallucination check.

The question this answers is narrow and checkable: *is the text the model quoted
actually in the document it was given?* A quote that isn't there is fabricated,
regardless of how reasonable the surrounding label looks.

The comparison has to be fuzzy, because exact string matching would report
fabrication for a model that got the substance right and the whitespace wrong.
Filings are full of non-breaking spaces, smart quotes, ligatures, and footnote
markers spliced mid-sentence, and models normalise these silently. So matching
proceeds in three widening stages:

1. Exact match on aggressively normalised text.
2. Contiguous match after stripping all non-alphanumerics — catches punctuation
   and whitespace drift.
3. Token-subsequence overlap above a threshold — catches an elided word or an
   inserted footnote marker, while still rejecting invented sentences.

Stage 3 is where the judgement lies. Set the threshold too low and paraphrase
passes as citation; too high and legitimate quotes are called hallucinations. It
is set at 0.85 and exposed as a parameter, and the eval suite reports grounding
rate per model so the number can be re-tuned against evidence rather than taste.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

DEFAULT_THRESHOLD = 0.85

_QUOTE_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        "​": "",
        "…": "...",
    }
)


@dataclass(slots=True)
class GroundingResult:
    total: int
    grounded: int
    ungrounded_quotes: list[str]
    #: How each quote matched: exact | normalised | fuzzy | none.
    methods: list[str]

    @property
    def ratio(self) -> float:
        """Share of quotes found in the source.

        An extraction with *no* quotes returns 1.0 rather than 0.0: it made no
        citation claims, so it cannot have fabricated one. Whether a
        no-evidence extraction is acceptable is a separate policy question,
        handled by the review queue, not by this metric.
        """
        return 1.0 if self.total == 0 else self.grounded / self.total

    @property
    def hallucinated(self) -> bool:
        return self.grounded < self.total


def check_grounding(
    quotes: list[str],
    source: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> GroundingResult:
    """Verify every quote appears in ``source``."""
    if not quotes:
        return GroundingResult(total=0, grounded=0, ungrounded_quotes=[], methods=[])

    norm_source = _normalise(source)
    alnum_source = _alnum(norm_source)
    source_tokens = norm_source.split()
    # Index token positions once; a linear rescan per quote would make this
    # quadratic on long filings.
    token_index: dict[str, list[int]] = {}
    for i, tok in enumerate(source_tokens):
        token_index.setdefault(tok, []).append(i)

    grounded = 0
    ungrounded: list[str] = []
    methods: list[str] = []

    for quote in quotes:
        method = _match(quote, norm_source, alnum_source, source_tokens, token_index, threshold)
        methods.append(method)
        if method == "none":
            ungrounded.append(quote)
        else:
            grounded += 1

    return GroundingResult(
        total=len(quotes),
        grounded=grounded,
        ungrounded_quotes=ungrounded,
        methods=methods,
    )


def _match(
    quote: str,
    norm_source: str,
    alnum_source: str,
    source_tokens: list[str],
    token_index: dict[str, list[int]],
    threshold: float,
) -> str:
    nq = _normalise(quote)
    if not nq:
        return "none"

    if nq in norm_source:
        return "exact"
    if _alnum(nq) and _alnum(nq) in alnum_source:
        return "normalised"
    if _fuzzy_contains(nq.split(), source_tokens, token_index, threshold):
        return "fuzzy"
    return "none"


def _fuzzy_contains(
    quote_tokens: list[str],
    source_tokens: list[str],
    token_index: dict[str, list[int]],
    threshold: float,
) -> bool:
    """Does a window of the source contain ``threshold`` of the quote's tokens?

    Anchored on the quote's rarest token so only plausible windows are examined,
    rather than sliding over the whole document.
    """
    if len(quote_tokens) < 3:
        # Too short to fuzzy-match responsibly: a 2-word "quote" would match
        # almost any document and would make the metric meaningless.
        return False

    anchor = min(
        (t for t in quote_tokens if t in token_index),
        key=lambda t: len(token_index[t]),
        default=None,
    )
    if anchor is None:
        return False

    span = len(quote_tokens)
    want = set(quote_tokens)
    # Threshold applies to *distinct* tokens, because the overlap below is a set
    # intersection and so can never exceed len(want). Scaling it by the raw token
    # count instead made the bar unreachable for any quote that repeats a word:
    # a 36-token quote with 26 distinct tokens needed 30.6 of a possible 26, so
    # stage 3 silently never fired and both cases in this module's docstring — an
    # elided word, a spliced footnote marker — were reported as fabricated.
    needed = threshold * len(want)

    for pos in token_index[anchor][:200]:  # cap work on pathological repetition
        start = max(0, pos - span)
        window = source_tokens[start : start + 2 * span]
        if len(set(window) & want) >= needed:
            return True
    return False


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(_QUOTE_MAP)
    return re.sub(r"\s+", " ", text).strip().lower()


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
