"""API contract tests.

Endpoint shapes are worth pinning because the dashboard and any downstream
consumer depend on them, and because the cost and eval endpoints are the ones an
operator reaches for when something looks wrong — they must not be the thing that
is broken.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from signalforge.api.main import app
from signalforge.db import connect, upsert


@pytest.fixture
def client(warehouse) -> TestClient:
    return TestClient(app)


@pytest.fixture
def seeded(warehouse):
    with connect() as con:
        upsert(
            con, "companies", [{"cik": "0000000001", "name": "Testco", "ticker": "TST"}], key="cik"
        )
        upsert(
            con,
            "filings",
            [
                {
                    "accession": "acc-1",
                    "cik": "0000000001",
                    "form": "10-Q",
                    "filing_date": "2026-06-30",
                    "primary_doc": "d.htm",
                }
            ],
            key="accession",
        )
        upsert(
            con,
            "signals",
            [
                {
                    "signal_id": "sig-1",
                    "name": "guidance_tone",
                    "cik": "0000000001",
                    "ticker": "TST",
                    "accession": "acc-1",
                    "as_of": "2026-06-30",
                    "score": -0.72,
                    "confidence": 0.85,
                    "direction": "bearish",
                    "rationale": "Guidance withdrawn.",
                    "evidence": ["withdrawing our full-year outlook"],
                    "extraction_ids": ["e1"],
                    "pipeline_version": "score-v1",
                },
                {
                    "signal_id": "sig-2",
                    "name": "event_class",
                    "cik": "0000000001",
                    "ticker": "TST",
                    "accession": "acc-1",
                    "as_of": "2026-06-30",
                    "score": 0.05,
                    "confidence": 0.4,
                    "direction": "neutral",
                    "rationale": "Routine.",
                    "evidence": [],
                    "extraction_ids": ["e2"],
                    "pipeline_version": "score-v1",
                },
            ],
            key="signal_id",
        )
        upsert(
            con,
            "alerts",
            [
                {
                    "alert_id": "al-1",
                    "signal_id": "sig-1",
                    "rule": "guidance_withdrawn_or_cut",
                    "severity": "critical",
                    "headline": "TST: forward guidance deteriorated",
                    "detail": "Guidance withdrawn.",
                }
            ],
            key="alert_id",
        )
    return True


def test_health_reports_status_and_counts(client):
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    assert "filings" in body["warehouse"]


def test_signals_endpoint_shape(client, seeded):
    rows = client.get("/signals").json()
    assert len(rows) == 2
    row = next(r for r in rows if r["signal_id"] == "sig-1")
    assert row["ticker"] == "TST"
    assert row["direction"] == "bearish"
    # Evidence must arrive as a list, not a JSON string, or the dashboard breaks.
    assert row["evidence"] == ["withdrawing our full-year outlook"]


def test_signals_filters(client, seeded):
    assert len(client.get("/signals?ticker=TST").json()) == 2
    assert client.get("/signals?ticker=NOPE").json() == []
    assert len(client.get("/signals?name=guidance_tone").json()) == 1
    assert len(client.get("/signals?direction=bearish").json()) == 1


def test_min_abs_score_filter_excludes_noise(client, seeded):
    """The neutral 0.05 signal should drop out at a 0.1 threshold."""
    rows = client.get("/signals?min_abs_score=0.1").json()
    assert [r["signal_id"] for r in rows] == ["sig-1"]


def test_alerts_are_ordered_by_severity(client, seeded):
    rows = client.get("/alerts").json()
    assert rows[0]["severity"] == "critical"
    assert rows[0]["ticker"] == "TST"


def test_alerts_severity_filter(client, seeded):
    assert client.get("/alerts?severity=info").json() == []


def test_composite_endpoint(client, seeded):
    body = client.get("/companies/tst/composite").json()
    assert body["ticker"] == "TST"
    assert body["score"] < 0
    assert 0 < body["coverage"] <= 1
    assert len(body["signals"]) == 2


def test_composite_unknown_ticker_is_404(client, seeded):
    assert client.get("/companies/ZZZZ/composite").status_code == 404


def test_search_endpoint_on_empty_corpus(client):
    assert client.get("/search?q=anything").json() == []


def test_search_rejects_bad_mode(client):
    assert client.get("/search?q=x&mode=telepathy").status_code == 422


def test_cost_metrics_shape(client, seeded):
    body = client.get("/metrics/cost?days=30").json()
    assert set(body) >= {"total_cost_usd", "total_calls", "cache_hit_rate", "by_model"}


def test_eval_runs_endpoint(client, seeded):
    assert client.get("/metrics/runs").json() == []


def test_review_queue_endpoint(client, seeded):
    assert client.get("/review").json() == []


def test_config_endpoint_exposes_what_is_deployed(client):
    """An operator must be able to see which prompt hash and models are live."""
    body = client.get("/config").json()
    assert "llama31-8b" in body["models"]
    assert body["chains"]["ci"] == ["stub"]
    assert all({"name", "version", "hash"} <= set(p) for p in body["prompts"])


def test_agent_request_validation(client, seeded):
    assert client.post("/agent", json={"question": "x", "max_steps": 999}).status_code == 422
    assert client.post("/agent", json={}).status_code == 422


def test_dashboard_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "SignalForge" in resp.text


def test_openapi_schema_is_valid(client):
    spec = client.get("/openapi.json").json()
    assert "/signals" in spec["paths"]
    assert "/agent" in spec["paths"]
    json.dumps(spec)  # must be serialisable
