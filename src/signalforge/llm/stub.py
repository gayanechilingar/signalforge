"""Deterministic stub provider.

Why this exists
---------------
CI must be able to exercise the *whole* pipeline — prompt rendering, schema
validation, the repair loop, scoring, alerting, the agent loop — without a GPU,
a network, or a bill. A stub provider makes those tests hermetic and fast, so
the regression gate runs on every push.

It is deliberately not a mock that returns a constant. It has three modes:

``scripted``
    Responses keyed by prompt-content substring, registered by the test. Used
    when a test asserts on a specific model output.
``heuristic`` (default)
    A tiny keyword-rule engine that emits *schema-shaped* output for the real
    extraction tasks. Good enough that end-to-end assertions about scoring and
    alerting are meaningful.
``failing``
    Raises on the first N calls to exercise retry and fallback paths.

The contract is: whatever the stub returns must be something a real model could
plausibly return, *including* being wrong. Tests that only pass against a
perfect oracle do not protect production.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .base import LLMClient, LLMError, LLMRequest, LLMResponse, Usage

_BEARISH = (
    "decline",
    "declined",
    "decrease",
    "withdrew",
    "withdrawn",
    "impairment",
    "material weakness",
    "going concern",
    "restructuring",
    "layoff",
    "shortfall",
    "miss",
    "headwind",
    "litigation",
    "investigation",
    "downgrade",
    "delay",
    "weaker",
    "lowered",
    "cut",
    "loss",
    "adverse",
)
_BULLISH = (
    "growth",
    "grew",
    "increase",
    "increased",
    "record",
    "raised",
    "raising",
    "beat",
    "exceeded",
    "expansion",
    "accelerat",
    "margin improvement",
    "tailwind",
    "strong demand",
    "upgrade",
    "buyback",
    "dividend increase",
)


class StubClient(LLMClient):
    provider = "stub"

    def __init__(
        self,
        mode: str = "heuristic",
        *,
        scripted: dict[str, str] | None = None,
        fail_times: int = 0,
        latency_ms: float = 1.0,
    ) -> None:
        self.mode = mode
        self.scripted = scripted or {}
        self.fail_times = fail_times
        self.latency_ms = latency_ms
        self.calls: list[LLMRequest] = []

    def register(self, needle: str, response: str) -> None:
        """Return ``response`` for any request whose text contains ``needle``."""
        self.scripted[needle] = response

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(req)

        if self.fail_times > 0:
            self.fail_times -= 1
            raise LLMError("stub induced failure", provider=self.provider, retryable=True)

        prompt = "\n".join(m.content for m in req.messages)
        # Scripted lookup keys on the *last* user message, not the whole
        # conversation. On a repair turn the original prompt is still present in
        # history, so matching the concatenation would keep returning the first
        # turn's answer and the repair loop could never converge.
        turn = next((m.content for m in reversed(req.messages) if m.role == "user"), prompt)

        text: str | None = None
        for needle, response in self.scripted.items():
            if needle in turn:
                text = response
                break
        if text is None:
            # The heuristic reads the whole prompt: the document it must score
            # lives in the first turn, not in the repair instruction.
            text = self._heuristic(prompt, req)

        return LLMResponse(
            text=text,
            model=req.model,
            provider=self.provider,
            usage=Usage(tokens_in=_approx_tokens(prompt), tokens_out=_approx_tokens(text)),
            latency_ms=self.latency_ms,
            finish_reason="stop",
        )

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Deterministic pseudo-embeddings.

        Hash-derived so they are stable across runs and machines, and lightly
        keyword-biased so that semantically related strings actually land closer
        together — otherwise retrieval tests would be pure noise.
        """
        dim = 64
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * dim
            for tok in _tokens(t):
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                vec[h % dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    def health(self) -> tuple[bool, str]:
        return True, f"stub ok (mode={self.mode})"

    # -- heuristic response generation -------------------------------------
    def _heuristic(self, prompt: str, req: LLMRequest) -> str:
        if not req.json_mode and not req.json_schema:
            return "Stub narrative response."

        schema = req.json_schema or {}
        props: dict[str, Any] = schema.get("properties") or {}
        # Score only the source document, not the instructions, so prompt
        # wording does not leak into the "prediction".
        body = _document_section(prompt)
        polarity = _polarity(body)

        obj: dict[str, Any] = {}
        for key, spec in props.items():
            obj[key] = self._field(key, spec, polarity, body)
        return _dumps(obj)

    def _field(self, key: str, spec: dict[str, Any], polarity: float, body: str) -> Any:
        enum = spec.get("enum")
        typ = spec.get("type")

        if enum:
            return _pick_enum(key, enum, polarity)
        if typ == "number" or typ == "integer":
            lo = spec.get("minimum", -1.0)
            hi = spec.get("maximum", 1.0)
            if "confidence" in key:
                val = 0.55 + min(abs(polarity), 1.0) * 0.35
            elif lo < 0 <= hi:
                val = max(lo, min(hi, polarity))
            else:
                val = lo + (hi - lo) * (0.5 + polarity / 4)
            return int(round(val)) if typ == "integer" else round(float(val), 3)
        if typ == "boolean":
            return polarity < -0.3
        if typ == "array":
            item = spec.get("items") or {}
            if (item.get("type") or "string") == "string":
                return _evidence(body, negative=polarity < 0)[:3]
            return []
        if typ == "object":
            return {}
        # string
        if "rationale" in key or "reason" in key or "summary" in key:
            direction = (
                "negative" if polarity < -0.15 else "positive" if polarity > 0.15 else "neutral"
            )
            return f"Stub rationale: language scans {direction} on balance."
        ev = _evidence(body, negative=polarity < 0)
        return ev[0] if ev else ""


# -- shared helpers --------------------------------------------------------
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _document_section(prompt: str) -> str:
    """Isolate the source text from the instruction scaffold.

    Prompts in ``prompts/`` wrap the filing excerpt in ``<document>`` tags for
    exactly this reason: it lets both the stub and any grounding check see what
    the model was actually asked to read.
    """
    blocks = re.findall(r"<document[^>]*>(.*?)</document>", prompt, re.S | re.I)
    return "\n".join(blocks) if blocks else prompt


def _polarity(text: str) -> float:
    low = text.lower()
    neg = sum(low.count(w) for w in _BEARISH)
    pos = sum(low.count(w) for w in _BULLISH)
    if neg + pos == 0:
        return 0.0
    return round((pos - neg) / (pos + neg), 3)


def _pick_enum(key: str, enum: list[Any], polarity: float) -> Any:
    """Map polarity onto an enum by matching known label vocabularies."""
    labels = [str(e).lower() for e in enum]

    def find(*cands: str) -> Any | None:
        for c in cands:
            for i, lab in enumerate(labels):
                if lab == c:
                    return enum[i]
        return None

    if polarity < -0.15:
        hit = find("bearish", "negative", "deteriorating", "lowered", "high", "down")
    elif polarity > 0.15:
        hit = find("bullish", "positive", "improving", "raised", "low", "up")
    else:
        hit = find("neutral", "unchanged", "stable", "medium", "flat")
    if hit is not None:
        return hit
    # Unknown vocabulary: pick deterministically by key so results are stable.
    idx = int(hashlib.md5(key.encode()).hexdigest()[:4], 16) % len(enum)
    return enum[idx]


def _evidence(body: str, *, negative: bool) -> list[str]:
    """Return verbatim sentences from the document.

    Verbatim matters: the hallucination metric checks that quoted evidence is
    actually present in the source, so a stub that invented quotes would make
    that metric untestable.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if 25 < len(s.strip()) < 400]
    words = _BEARISH if negative else _BULLISH
    hits = [s for s in sentences if any(w in s.lower() for w in words)]
    return hits or sentences[:1]


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
