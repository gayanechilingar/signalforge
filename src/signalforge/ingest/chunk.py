"""Chunking for retrieval and extraction.

Two consumers with different needs share this module:

* **Retrieval** wants chunks small enough that a hit is precise, with overlap so
  that a fact spanning a boundary is still findable.
* **Extraction** wants the largest span that fits the model's context, because
  splitting a risk-factor section into fragments destroys the comparison the
  extraction is trying to make.

Both are served by one splitter with different sizes rather than two
implementations, so a change to boundary handling can't apply to only one path.

Boundaries respect paragraphs first, then sentences, and only split mid-sentence
as a last resort — a chunk that starts mid-clause reads as gibberish to both an
embedding model and a reader checking a citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Filings are dense prose with many numbers and abbreviations; measured against
#: nomic-embed-text and llama3 tokenizers, ~3.6 chars/token is closer than the
#: usual 4.0 rule of thumb. Used only for sizing, never for billing.
CHARS_PER_TOKEN = 3.6

_PARA = re.compile(r"\n\s*\n")
# Split on sentence enders followed by whitespace + a capital or digit. The
# lookbehind exclusions keep common filing abbreviations intact.
_SENT = re.compile(
    r"(?<!\bNo)(?<!\bInc)(?<!\bCorp)(?<!\bLtd)(?<!\bCo)(?<!\bU\.S)(?<!\bi\.e)"
    r"(?<!\be\.g)(?<!\bvs)(?<!\bFig)(?<!\bApprox)(?<![A-Z])"
    r"(?<=[.!?])\s+(?=[A-Z0-9])"
)


@dataclass(slots=True)
class Chunk:
    ordinal: int
    text: str
    #: Character offset within the source section — lets a citation be traced
    #: back to its exact location, not merely to "somewhere in Item 1A".
    start_char: int

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.text)


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def chunk_text(
    text: str,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    min_tokens: int = 32,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks of at most ``max_tokens``."""
    text = text.strip()
    if not text:
        return []
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)

    units = _split_units(text, max_chars)

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    start = 0
    cursor = 0

    for unit in units:
        unit_len = len(unit)
        if buf and buf_len + unit_len > max_chars:
            body = "".join(buf).strip()
            if body:
                chunks.append(Chunk(ordinal=len(chunks), text=body, start_char=start))
            # Carry the tail forward so a fact straddling the boundary survives
            # in at least one chunk intact.
            tail = _tail(buf, overlap_chars)
            buf = list(tail)
            buf_len = sum(len(t) for t in tail)
            start = cursor - buf_len
        buf.append(unit)
        buf_len += unit_len
        cursor += unit_len

    body = "".join(buf).strip()
    if body:
        # A short trailing fragment is appended to the previous chunk rather than
        # kept alone: a 20-token chunk is noise in a vector index.
        if chunks and estimate_tokens(body) < min_tokens:
            prev = chunks[-1]
            merged = (prev.text + " " + body).strip()
            chunks[-1] = Chunk(prev.ordinal, merged, prev.start_char)
        else:
            chunks.append(Chunk(ordinal=len(chunks), text=body, start_char=start))
    return chunks


def _split_units(text: str, max_chars: int) -> list[str]:
    """Break text into atoms small enough to pack: paragraphs, then sentences."""
    units: list[str] = []
    for para in _keep_split(text, _PARA):
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sent in _keep_split(para, _SENT):
            if len(sent) <= max_chars:
                units.append(sent)
            else:
                # A single sentence over the limit means a table row or a run-on;
                # a hard character split is the only option left.
                units.extend(sent[i : i + max_chars] for i in range(0, len(sent), max_chars))
    return [u for u in units if u.strip()]


def _keep_split(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split while retaining the separators, so offsets stay meaningful."""
    out: list[str] = []
    prev = 0
    for m in pattern.finditer(text):
        out.append(text[prev : m.end()])
        prev = m.end()
    if prev < len(text):
        out.append(text[prev:])
    return out


def _tail(units: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0:
        return []
    out: list[str] = []
    total = 0
    for unit in reversed(units):
        out.insert(0, unit)
        total += len(unit)
        if total >= overlap_chars:
            break
    # Never carry the entire buffer forward — that would loop forever on a
    # single oversized unit.
    return out if len(out) < len(units) else out[1:]
