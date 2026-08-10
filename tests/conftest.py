"""Shared fixtures.

Every test runs against a temp warehouse, a temp cache, and the stub provider, so
the suite is hermetic: no network, no Ollama, no API key, no shared state between
tests. That is what lets CI run the same regression gate a laptop does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signalforge import db
from signalforge.llm.cache import ResponseCache
from signalforge.llm.registry import load_registry
from signalforge.llm.router import Router
from signalforge.llm.stub import StubClient
from signalforge.obs.tracing import Tracer
from signalforge.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Point every path-valued setting at a tmp dir for the duration of a test."""
    get_settings.cache_clear()
    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SF_DB_PATH", str(tmp_path / "data" / "test.duckdb"))
    monkeypatch.setenv("SF_DEFAULT_PROVIDER", "stub")
    monkeypatch.setenv("SF_DETERMINISTIC", "true")
    settings = get_settings()
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def registry():
    return load_registry()


@pytest.fixture
def stub() -> StubClient:
    return StubClient()


@pytest.fixture
def tracer() -> Tracer:
    # Tracing disabled by default in tests: spans are asserted on in memory, and
    # writing them would add a DuckDB round-trip to every single test.
    return Tracer(enabled=False)


@pytest.fixture
def router(registry, stub, tracer, tmp_path) -> Router:
    return Router(
        registry=registry,
        tracer=tracer,
        cache=ResponseCache(path=tmp_path / "cache.duckdb"),
        stub=stub,
        max_retries=2,
    )


@pytest.fixture
def warehouse(isolated_settings) -> Path:
    return db.init_db()
