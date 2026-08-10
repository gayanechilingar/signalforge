"""Model router — selection, fallback, retries, caching, cost control.

This is the single choke point every LLM call in the system goes through. Putting
it here rather than in each pipeline means five properties hold everywhere by
construction rather than by discipline:

1. **Selection is config-driven.** Callers ask for a *chain name*
   (``extract_default``), not a model. Swapping models is a YAML edit.
2. **Failures degrade instead of exploding.** Retry with backoff within a model,
   then fall down the chain. A non-retryable error (unknown model, bad schema)
   stops immediately — burning the whole chain on a config bug wastes money and
   hides the real error.
3. **Every call is cached** by a digest of its inputs, so re-running an eval after
   a scoring change is free and reproducible.
4. **Cost is capped.** Spend is accumulated per router instance and checked
   *before* dispatch; exceeding the cap raises rather than continuing quietly.
5. **Every call is traced** with model, tokens, cost, latency, and cache status.

Provider clients are constructed lazily and memoised per model, because building
an Ollama client is cheap but building it 10,000 times is not.
"""

from __future__ import annotations

import random
import time
from typing import Any

from ..obs.tracing import Tracer, default_tracer
from ..settings import get_settings
from .anthropic import AnthropicClient
from .base import (
    CostCapExceeded,
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    ModelSpec,
)
from .cache import ResponseCache
from .ollama import OllamaClient
from .registry import Registry, get_registry
from .stub import StubClient


class Router:
    def __init__(
        self,
        *,
        registry: Registry | None = None,
        tracer: Tracer | None = None,
        cache: ResponseCache | None = None,
        cost_cap_usd: float | None = None,
        max_retries: int | None = None,
        stub: StubClient | None = None,
    ) -> None:
        s = get_settings()
        self.registry = registry or get_registry()
        self.tracer = tracer or default_tracer
        self.cache = cache if cache is not None else ResponseCache()
        self.cost_cap_usd = s.run_cost_cap_usd if cost_cap_usd is None else cost_cap_usd
        self.max_retries = s.max_retries if max_retries is None else max_retries
        #: Injected by tests so they can script or fail specific calls.
        self._stub = stub
        self._clients: dict[str, LLMClient] = {}
        self.spend_usd = 0.0
        self.calls = 0
        self.cache_hits = 0

    # -- client construction ----------------------------------------------
    def client_for(self, spec: ModelSpec) -> LLMClient:
        if spec.name in self._clients:
            return self._clients[spec.name]

        client: LLMClient
        if spec.provider == "ollama":
            client = OllamaClient()
        elif spec.provider == "anthropic":
            client = AnthropicClient(supports_sampling=spec.supports_sampling)
        elif spec.provider == "stub":
            client = self._stub or StubClient()
        else:
            raise LLMError(
                f"unknown provider {spec.provider!r} for model {spec.name!r}",
                provider=spec.provider,
                retryable=False,
            )
        self._clients[spec.name] = client
        return client

    # -- the main entry point ---------------------------------------------
    def complete(
        self,
        messages: list[Message] | str,
        *,
        chain: str | list[str] = "extract_default",
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        tools: list[dict[str, Any]] | None = None,
        stop: list[str] | None = None,
        span_name: str = "llm.complete",
        **attrs: Any,
    ) -> LLMResponse:
        """Run a completion against the first model in ``chain`` that succeeds."""
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        specs = self.registry.chain(chain)
        if not specs:
            raise LLMError("empty model chain", provider="router", retryable=False)

        errors: list[str] = []
        for i, spec in enumerate(specs):
            req = LLMRequest(
                messages=messages,
                model=spec.model_id,
                temperature=temperature if spec.supports_sampling else 0.0,
                max_tokens=min(max_tokens, spec.max_output_tokens or max_tokens),
                json_mode=json_schema is not None,
                json_schema=json_schema,
                tools=tools or [],
                stop=stop or [],
            )
            try:
                return self._attempt(spec, req, span_name, attrs, fallback_depth=i)
            except CostCapExceeded:
                # A cost cap is a deliberate stop, never something to fall back
                # from — trying a cheaper model next would defeat the point.
                raise
            except LLMError as exc:
                errors.append(f"{spec.name}: {exc}")
                if not exc.retryable:
                    break
                continue

        raise LLMError(
            "all models in chain failed -> " + " | ".join(errors),
            provider="router",
            retryable=False,
        )

    def _attempt(
        self,
        spec: ModelSpec,
        req: LLMRequest,
        span_name: str,
        attrs: dict[str, Any],
        *,
        fallback_depth: int,
    ) -> LLMResponse:
        with self.tracer.span(span_name, kind="llm", **attrs) as span:
            span.set(
                model=spec.name,
                provider=spec.provider,
                fallback_depth=fallback_depth,
                json_mode=req.json_mode,
            )

            cached = self.cache.get(req)
            if cached is not None:
                self.cache_hits += 1
                self.calls += 1
                span.set(
                    cached=True,
                    tokens_in=cached.usage.tokens_in,
                    tokens_out=cached.usage.tokens_out,
                )
                return cached

            self._check_cap(spec, req)
            client = self.client_for(spec)
            resp = self._with_retries(client, req, spec)

            resp.cost_usd = spec.cost_usd(resp.usage.tokens_in, resp.usage.tokens_out)
            self.spend_usd += resp.cost_usd
            self.calls += 1
            # Report the model's registry name, not the provider-side ID, so that
            # traces and eval rows join against configs/models.yaml.
            resp.model = spec.name

            span.set(
                tokens_in=resp.usage.tokens_in,
                tokens_out=resp.usage.tokens_out,
                cost_usd=resp.cost_usd,
                cached=False,
                finish_reason=resp.finish_reason,
            )
            self.cache.put(req, resp)
            return resp

    def _with_retries(self, client: LLMClient, req: LLMRequest, spec: ModelSpec) -> LLMResponse:
        """Retry transient failures with jittered exponential backoff.

        Jitter matters here specifically: a batch pipeline fires many concurrent
        requests, and un-jittered backoff makes them retry in lockstep and
        re-overload whatever just failed.
        """
        last: LLMError | None = None
        for attempt in range(self.max_retries):
            try:
                return client.complete(req)
            except LLMError as exc:
                if not exc.retryable:
                    raise
                last = exc
                if attempt == self.max_retries - 1:
                    break
                delay = min(2.0**attempt + random.uniform(0, 0.4), 20.0)
                time.sleep(delay)
        assert last is not None
        raise last

    def _check_cap(self, spec: ModelSpec, req: LLMRequest) -> None:
        """Refuse a call that would plausibly breach the cap.

        Estimated pessimistically — full ``max_tokens`` of output — because the
        useful behaviour of a cost cap is to stop *before* the spend, not to
        report it afterwards.
        """
        if self.cost_cap_usd <= 0:
            return
        est_in = sum(len(m.content) for m in req.messages) // 4
        projected = self.spend_usd + spec.cost_usd(est_in, req.max_tokens)
        if projected > self.cost_cap_usd:
            raise CostCapExceeded(
                f"run cost cap ${self.cost_cap_usd:.2f} would be exceeded "
                f"(spent ${self.spend_usd:.4f}, next call ~"
                f"${projected - self.spend_usd:.4f}); raise SF_RUN_COST_CAP_USD "
                f"to continue"
            )

    # -- embeddings --------------------------------------------------------
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        spec = self.registry.embedding(model)
        with self.tracer.span("llm.embed", kind="llm", n=len(texts)) as span:
            span.set(model=spec.name, provider=spec.provider)
            if spec.provider == "stub":
                client: LLMClient = self._stub or StubClient()
            else:
                client = OllamaClient()
            vecs = client.embed(texts, model=spec.model_id)
            if vecs and len(vecs[0]) != spec.dim:
                # A dimension mismatch means the registry and the running model
                # disagree; writing those vectors would corrupt the index.
                raise LLMError(
                    f"embedding dim mismatch for {spec.name}: registry says "
                    f"{spec.dim}, provider returned {len(vecs[0])}",
                    provider=spec.provider,
                    retryable=False,
                )
            span.set(tokens_in=sum(len(t) for t in texts) // 4)
            return vecs

    # -- reporting ---------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / self.calls, 3) if self.calls else 0.0,
            "spend_usd": round(self.spend_usd, 6),
            "cost_cap_usd": self.cost_cap_usd,
        }
