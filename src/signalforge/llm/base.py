"""Provider-agnostic LLM interface.

Design notes
------------
* Every call returns a :class:`LLMResponse` carrying *usage, cost, and latency*
  alongside the text. Cost and latency are first-class outputs, not something
  bolted on by a wrapper later — you cannot manage what you do not measure.
* ``json_schema`` is honoured on a best-effort basis. Local open-source models
  support "give me syntactically valid JSON" but not full grammar-constrained
  schema adherence, so schema *conformance* is enforced downstream by Pydantic
  with a repair loop (see ``extract/runner.py``). The interface therefore
  promises well-formed JSON, not valid-per-schema JSON, and the pipeline is
  built to expect that distinction.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """Static facts about a model, loaded from ``configs/models.yaml``."""

    name: str
    provider: str
    #: Provider-side identifier (e.g. ``llama3.1:latest``).
    model_id: str
    context_tokens: int
    usd_per_mtok_in: float = 0.0
    usd_per_mtok_out: float = 0.0
    #: Rough capability tier, used by the router for escalation ordering.
    tier: int = 1
    supports_json: bool = True
    supports_tools: bool = False
    #: Sampling params (temperature/top_p/top_k) were removed on the current
    #: frontier Claude models and now return 400. The router drops them when this
    #: is False — a per-model fact, not a per-provider one.
    supports_sampling: bool = True
    #: Provider-side ceiling on output tokens, when it is lower than context.
    max_output_tokens: int | None = None
    notes: str = ""

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        return (
            tokens_in / 1_000_000 * self.usd_per_mtok_in
            + tokens_out / 1_000_000 * self.usd_per_mtok_out
        )


class Message(BaseModel):
    role: str  # system | user | assistant | tool
    content: str
    #: Present on assistant messages that requested tool calls.
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    #: Present on tool-result messages.
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(slots=True)
class LLMRequest:
    messages: list[Message]
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    #: Request well-formed JSON output.
    json_mode: bool = False
    #: Advisory schema, included in the prompt and used downstream for validation.
    json_schema: dict[str, Any] | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    stop: list[str] = field(default_factory=list)

    def cache_key(self) -> str:
        """Stable digest of everything that can change the output.

        Used both for the response cache and for reproducibility claims: two runs
        with the same key are the same experiment.
        """
        payload = {
            "messages": [m.model_dump(exclude_none=True) for m in self.messages],
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "json_schema": self.json_schema,
            "tools": self.tools,
            "stop": self.stop,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


class Usage(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out


class LLMResponse(BaseModel):
    text: str
    model: str
    provider: str
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    #: True when served from the response cache (cost and latency are then ~0).
    cached: bool = False
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    def json_payload(self) -> Any:
        """Parse ``text`` as JSON, tolerating the usual model chatter.

        Local models fenced in markdown or preceded by "Here is the JSON:" are
        common enough that stripping them belongs here rather than in every
        caller.
        """
        return parse_json_loose(self.text)


class LLMError(RuntimeError):
    """Provider call failed in a way worth retrying or falling back from."""

    def __init__(self, message: str, *, provider: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class CostCapExceeded(RuntimeError):
    """Raised when a run would push spend past the configured ceiling."""


def parse_json_loose(text: str) -> Any:
    """Best-effort JSON extraction from a model response.

    Tries, in order: the whole string; a ```json fenced block; the widest
    balanced ``{...}`` or ``[...]`` span. Raises ``ValueError`` if none parse,
    which the extraction repair loop catches and feeds back to the model.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if "```" in text:
        for block in _fenced_blocks(text):
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"no parseable JSON in response: {text[:200]!r}")


def _fenced_blocks(text: str) -> list[str]:
    out: list[str] = []
    parts = text.split("```")
    # Fenced content sits at odd indices: pre ``` body ``` post.
    for chunk in parts[1::2]:
        body = chunk
        if "\n" in body:
            first, rest = body.split("\n", 1)
            if first.strip().lower() in {"json", "js", ""}:
                body = rest
        out.append(body.strip())
    return out


class LLMClient(ABC):
    """One concrete implementation per provider."""

    provider: str

    @abstractmethod
    def complete(self, req: LLMRequest) -> LLMResponse: ...

    @abstractmethod
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...

    def health(self) -> tuple[bool, str]:
        """Cheap reachability probe. Used by the CLI doctor and API healthcheck."""
        return True, "ok"
