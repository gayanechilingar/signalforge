"""Anthropic provider, built on the official SDK.

Inert until ``SF_ANTHROPIC_API_KEY`` is set — the point is that adding a key is
the *only* step needed to route hard extraction tasks to Claude. The router,
prompt registry, cost accounting, and eval harness are provider-agnostic, so a
paid tier slots in without touching pipeline code.

Three details here are model-specific rather than provider-specific, and getting
them wrong is a 400 rather than a degradation:

* **Sampling params are gone** on the current frontier models. ``temperature``,
  ``top_p``, and ``top_k`` are rejected outright, so they are sent only when the
  model's :class:`~.base.ModelSpec` says ``supports_sampling``.
* **Structured output is a response-format constraint**, not a prompt trick:
  ``output_config.format`` with a JSON Schema. That is strictly better than the
  local models' "please emit JSON" — conformance is enforced server-side — which
  is why the extraction repair loop is a no-op on this path in practice.
* **Large ``max_tokens`` must stream**, or the request outlives the HTTP timeout.

Thinking is left at its default (on, adaptive) and ``effort`` is exposed as a
config knob rather than hardcoded, since it is the main quality/cost dial.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..settings import get_settings
from .base import LLMClient, LLMError, LLMRequest, LLMResponse, Usage

#: Above this, use the streaming API. Non-streaming requests with a large output
#: budget hit the SDK's HTTP timeout guard and raise before ever reaching Claude.
STREAM_THRESHOLD_TOKENS = 16_000


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        effort: str = "high",
        supports_sampling: bool = False,
    ) -> None:
        s = get_settings()
        self.api_key = api_key or s.anthropic_api_key
        self.effort = effort
        self.supports_sampling = supports_sampling
        self._client: Any = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _sdk(self) -> Any:
        """Import and construct lazily.

        ``anthropic`` is an optional dependency: a clone with no API key should
        install and test without it, and the error when it *is* needed should say
        so plainly rather than surfacing as an ImportError at module load.
        """
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise LLMError(
                "anthropic provider selected but the SDK is not installed; "
                "run: uv pip install 'signalforge[hosted]'",
                provider=self.provider,
                retryable=False,
            ) from exc
        if not self.configured:
            raise LLMError(
                "anthropic provider selected but SF_ANTHROPIC_API_KEY is unset",
                provider=self.provider,
                retryable=False,
            )
        self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    # -- api ---------------------------------------------------------------
    def complete(self, req: LLMRequest) -> LLMResponse:
        import anthropic

        client = self._sdk()

        system = "\n\n".join(m.content for m in req.messages if m.role == "system")
        kwargs: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": [self._msg(m) for m in req.messages if m.role != "system"],
            "output_config": {"effort": self.effort},
        }
        if system:
            kwargs["system"] = system
        if req.stop:
            kwargs["stop_sequences"] = req.stop
        if self.supports_sampling:
            kwargs["temperature"] = req.temperature
        if req.tools:
            kwargs["tools"] = [self._tool_def(t) for t in req.tools]
        elif req.json_mode and req.json_schema:
            # Schema-constrained decoding. Unlike the Ollama path, conformance is
            # guaranteed here, so the repair loop downstream should never fire.
            kwargs["output_config"]["format"] = {
                "type": "json_schema",
                "schema": _strict(req.json_schema),
            }

        t0 = time.perf_counter()
        try:
            if req.max_tokens > STREAM_THRESHOLD_TOKENS:
                with client.messages.stream(**kwargs) as stream:
                    msg = stream.get_final_message()
            else:
                msg = client.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"anthropic {exc.status_code}: {exc.message}",
                provider=self.provider,
                retryable=exc.status_code in (408, 409, 429, 500, 502, 503, 529),
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"anthropic connection error: {exc}", provider=self.provider) from exc
        latency_ms = (time.perf_counter() - t0) * 1000

        # A refusal is a successful HTTP 200 with an empty or partial body.
        # Reading content[0] unconditionally would crash here, so branch first.
        if msg.stop_reason == "refusal":
            detail = getattr(msg, "stop_details", None)
            raise LLMError(
                f"anthropic refused the request (category={getattr(detail, 'category', None)})",
                provider=self.provider,
                retryable=False,
            )

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in msg.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "args": block.input or {}})

        return LLMResponse(
            text="\n".join(text_parts),
            model=msg.model,
            provider=self.provider,
            usage=Usage(
                tokens_in=msg.usage.input_tokens or 0,
                tokens_out=msg.usage.output_tokens or 0,
            ),
            latency_ms=latency_ms,
            finish_reason=msg.stop_reason,
            tool_calls=tool_calls,
            raw={"id": msg.id},
        )

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        raise LLMError(
            "anthropic exposes no embeddings endpoint; embeddings are served by "
            "the ollama provider (nomic-embed-text)",
            provider=self.provider,
            retryable=False,
        )

    def health(self) -> tuple[bool, str]:
        if not self.configured:
            return False, "anthropic: no API key configured (optional)"
        return True, "anthropic: key present"

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _msg(m: Any) -> dict[str, Any]:
        if m.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id or "",
                        "content": m.content,
                    }
                ],
            }
        if m.tool_calls:
            content: list[dict[str, Any]] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            for c in m.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": c["id"],
                        "name": c["name"],
                        "input": c.get("args") or {},
                    }
                )
            return {"role": "assistant", "content": content}
        return {"role": m.role, "content": m.content}

    @staticmethod
    def _tool_def(t: dict[str, Any]) -> dict[str, Any]:
        """Translate our internal tool shape onto Anthropic's."""
        if "input_schema" in t:
            return t
        fn = t.get("function", t)
        return {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            "strict": True,
        }


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a Pydantic-generated schema acceptable for constrained decoding.

    Structured outputs require ``additionalProperties: false`` on every object and
    reject numeric/length constraints. Pydantic emits both, so strip and stamp
    rather than hand-maintaining a parallel schema — the constraints are still
    enforced, just client-side by the same Pydantic model on the way back in.
    """
    unsupported = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
    }

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items() if k not in unsupported}
            if out.get("type") == "object":
                out["additionalProperties"] = False
                # Constrained decoding requires every property be listed as
                # required; optionality is expressed as a nullable type instead.
                props = out.get("properties") or {}
                if props:
                    out["required"] = list(props.keys())
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return json.loads(json.dumps(walk(schema)))
