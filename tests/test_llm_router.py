"""Router behaviour: selection, fallback, retries, caching, cost caps.

These are the properties the rest of the system assumes hold, so they are tested
directly rather than incidentally through a pipeline.
"""

from __future__ import annotations

import pytest

from signalforge.llm.base import CostCapExceeded, LLMError, Message, parse_json_loose
from signalforge.llm.cache import ResponseCache
from signalforge.llm.router import Router
from signalforge.llm.stub import StubClient


def test_registry_loads_expected_models(registry):
    assert "llama31-8b" in registry.models
    assert registry.spec("opus-5").provider == "anthropic"
    # The frontier Claude models reject sampling params; the registry must say so,
    # because the router relies on that flag to avoid a 400.
    assert registry.spec("opus-5").supports_sampling is False
    assert registry.spec("haiku-4.5").supports_sampling is True
    assert [s.name for s in registry.chain("ci")] == ["stub"]


def test_cost_is_computed_from_registry_pricing(registry):
    spec = registry.spec("opus-5")
    # 1M in + 1M out at $5/$25.
    assert spec.cost_usd(1_000_000, 1_000_000) == pytest.approx(30.0)
    assert registry.spec("llama31-8b").cost_usd(1_000_000, 1_000_000) == 0.0


def test_completion_records_usage_and_reports_registry_name(router):
    resp = router.complete("What happened?", chain="ci")
    assert resp.provider == "stub"
    # The registry key, not the provider-side model ID — traces and eval rows
    # join on this.
    assert resp.model == "stub"
    assert resp.usage.tokens_in > 0
    assert router.stats()["calls"] == 1


def test_second_identical_call_is_served_from_cache(router):
    first = router.complete("Revenue declined sharply.", chain="ci")
    second = router.complete("Revenue declined sharply.", chain="ci")

    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text
    assert router.stats()["cache_hits"] == 1
    # A replay must not be billed or credited with latency, or every cost
    # dashboard silently inflates on re-runs.
    assert second.cost_usd == 0.0
    assert second.latency_ms == 0.0


def test_cache_key_changes_with_schema(router):
    schema = {"type": "object", "properties": {"direction": {"type": "string"}}}
    router.complete("Revenue grew.", chain="ci")
    resp = router.complete("Revenue grew.", chain="ci", json_schema=schema)
    assert resp.cached is False, "adding a schema must not hit the plain-text entry"


def test_retries_transient_failure_then_succeeds(registry, tracer, tmp_path):
    stub = StubClient(fail_times=1)
    router = Router(
        registry=registry,
        tracer=tracer,
        cache=ResponseCache(path=tmp_path / "c.duckdb"),
        stub=stub,
        max_retries=3,
    )
    resp = router.complete("Guidance was raised.", chain="ci")
    assert resp.text  # recovered on the retry


def test_falls_down_the_chain_when_first_model_is_exhausted(registry, tracer, tmp_path):
    """A model whose retries are all consumed should hand off, not fail the run."""
    stub = StubClient(fail_times=99)
    router = Router(
        registry=registry,
        tracer=tracer,
        cache=ResponseCache(path=tmp_path / "c.duckdb"),
        stub=stub,
        max_retries=2,
    )
    with pytest.raises(LLMError) as exc:
        router.complete("anything", chain=["stub", "stub"])
    # Both links attempted, and the error names them rather than being opaque.
    assert exc.value.args[0].count("stub:") == 2


def test_zero_retries_still_makes_one_attempt(registry, tracer, tmp_path):
    """`max_retries=0` means "do not retry", not "do not call the provider".

    Regression: the retry loop ran `range(max_retries)`, so a 0 skipped the body
    entirely and fell through to an assertion, failing with a bare AssertionError
    before any request was made.
    """
    stub = StubClient()
    router = Router(
        registry=registry,
        tracer=tracer,
        cache=ResponseCache(path=tmp_path / "c.duckdb"),
        stub=stub,
        max_retries=0,
    )
    assert router.complete("Guidance was raised.", chain="ci").text


def test_zero_retries_does_not_retry_a_transient_failure(registry, tracer, tmp_path):
    """One attempt only — the failure must surface as an LLMError, not an assert."""
    router = Router(
        registry=registry,
        tracer=tracer,
        cache=ResponseCache(path=tmp_path / "c.duckdb"),
        stub=StubClient(fail_times=1),
        max_retries=0,
    )
    with pytest.raises(LLMError):
        router.complete("anything", chain="ci")


def test_non_retryable_error_stops_immediately(registry, tracer, tmp_path):
    router = Router(
        registry=registry,
        tracer=tracer,
        cache=ResponseCache(path=tmp_path / "c.duckdb"),
        stub=StubClient(),
    )
    with pytest.raises(KeyError, match="unknown model"):
        router.complete("hi", chain=["no-such-model"])


def test_cost_cap_blocks_before_spending(registry, tracer, tmp_path):
    router = Router(
        registry=registry,
        tracer=tracer,
        cache=ResponseCache(path=tmp_path / "c.duckdb", enabled=False),
        stub=StubClient(),
        cost_cap_usd=0.0000001,
    )
    # Priced against a real model so the projection is non-zero.
    with pytest.raises(CostCapExceeded, match="cost cap"):
        router.complete("x" * 4000, chain=["opus-5"], max_tokens=4096)
    assert router.spend_usd == 0.0, "nothing should have been dispatched"


def test_cost_cap_is_not_swallowed_by_chain_fallback(registry, tracer, tmp_path):
    """A cap breach must abort the run, not quietly retry on a cheaper model."""
    router = Router(
        registry=registry,
        tracer=tracer,
        cache=ResponseCache(path=tmp_path / "c.duckdb", enabled=False),
        stub=StubClient(),
        cost_cap_usd=0.0000001,
    )
    with pytest.raises(CostCapExceeded):
        router.complete("x" * 4000, chain=["opus-5", "stub"], max_tokens=4096)


def test_spans_capture_cost_and_model(registry, tracer, tmp_path):
    router = Router(
        registry=registry,
        tracer=tracer,
        cache=ResponseCache(path=tmp_path / "c.duckdb"),
        stub=StubClient(),
    )
    router.complete("Revenue grew strongly.", chain="ci", span_name="extract.tone")
    span = tracer.spans[-1]
    assert span.name == "extract.tone"
    assert span.kind == "llm"
    assert span.model == "stub"
    assert span.tokens_out > 0
    assert tracer.summary()["llm_calls"] == 1


def test_embeddings_are_deterministic_and_dimension_checked(registry, tracer, tmp_path):
    router = Router(registry=registry, tracer=tracer, stub=StubClient(), cache=None)
    a = router.embed(["material weakness"], model="stub-embed")
    b = router.embed(["material weakness"], model="stub-embed")
    assert a == b
    assert len(a[0]) == registry.embedding("stub-embed").dim


def test_system_messages_are_passed_through(router):
    resp = router.complete(
        [
            Message(role="system", content="You are terse."),
            Message(role="user", content="Revenue declined."),
        ],
        chain="ci",
    )
    assert resp.text


class TestLooseJsonParsing:
    """Local models wrap JSON in prose and fences; the parser must cope, because
    the alternative is a repair round-trip on almost every call."""

    def test_bare_object(self):
        assert parse_json_loose('{"a": 1}') == {"a": 1}

    def test_fenced_block(self):
        assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}

    def test_preamble_and_trailing_prose(self):
        text = 'Sure! Here is the JSON:\n{"direction": "bearish"}\nHope that helps.'
        assert parse_json_loose(text) == {"direction": "bearish"}

    def test_array_payload(self):
        assert parse_json_loose("noise [1, 2, 3] noise") == [1, 2, 3]

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="no parseable JSON"):
            parse_json_loose("I'm afraid I can't do that.")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty response"):
            parse_json_loose("   ")
