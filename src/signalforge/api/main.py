"""FastAPI service — signals, alerts, search, agent, and observability.

Two design points worth stating:

**Cost and latency are exposed, not hidden.** ``/metrics/cost`` and
``/metrics/runs`` read straight from the trace and eval tables, so the operational
questions ("what did today cost", "did the last eval regress") are answerable from
the same surface that serves the data. An LLM system where cost is invisible is an
LLM system that surprises someone at the end of the month.

**The agent endpoint is bounded and synchronous.** It caps steps and cost per
request and returns the trace alongside the answer. Returning the steps is not a
debugging nicety — an answer from a probabilistic system is only actionable if the
reader can see which tools produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .. import db
from ..llm.registry import get_registry
from ..obs.tracing import Tracer
from ..prompts.registry import get_prompts
from ..settings import get_settings

app = FastAPI(
    title="SignalForge",
    description="LLM extraction pipelines turning SEC filings into investment signals.",
    version="0.1.0",
)


# ---------------------------------------------------------------- healthcheck
@app.get("/health")
def health() -> dict[str, Any]:
    from ..llm.ollama import OllamaClient

    ok, msg = OllamaClient().health()
    try:
        counts = db.query(
            """
            SELECT (SELECT count(*) FROM filings)    AS filings,
                   (SELECT count(*) FROM signals)    AS signals,
                   (SELECT count(*) FROM extractions) AS extractions
            """
        )[0]
        warehouse_ok = True
    except Exception as exc:
        counts, warehouse_ok = {"error": str(exc)}, False

    return {
        "status": "ok" if (ok and warehouse_ok) else "degraded",
        "provider": {"ollama": msg},
        "warehouse": counts,
    }


# -------------------------------------------------------------------- signals
class SignalOut(BaseModel):
    signal_id: str
    name: str
    cik: str
    ticker: str | None
    accession: str
    as_of: str
    score: float
    confidence: float | None
    direction: str
    rationale: str | None
    evidence: list[str] = Field(default_factory=list)


@app.get("/signals", response_model=list[SignalOut])
def list_signals(
    ticker: str | None = None,
    cik: str | None = None,
    name: str | None = Query(None, description="Signal name, e.g. guidance_tone."),
    direction: str | None = None,
    min_abs_score: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, le=500),
) -> list[dict[str, Any]]:
    clauses: list[str] = ["abs(s.score) >= ?"]
    params: list[Any] = [min_abs_score]
    if ticker:
        clauses.append("upper(coalesce(s.ticker, c.ticker)) = ?")
        params.append(ticker.upper())
    if cik:
        clauses.append("s.cik = ?")
        params.append(cik)
    if name:
        clauses.append("s.name = ?")
        params.append(name)
    if direction:
        clauses.append("s.direction = ?")
        params.append(direction)

    rows = db.query(
        f"""
        SELECT s.signal_id, s.name, s.cik, coalesce(s.ticker, c.ticker) AS ticker,
               s.accession, s.as_of, s.score, s.confidence, s.direction,
               s.rationale, s.evidence
        FROM signals s
        LEFT JOIN companies c ON c.cik = s.cik
        WHERE {" AND ".join(clauses)}
        ORDER BY s.as_of DESC, abs(s.score) DESC
        LIMIT {int(limit)}
        """,
        params,
    )
    return [_signal_row(r) for r in rows]


@app.get("/companies/{ticker}/composite")
def company_composite(ticker: str) -> dict[str, Any]:
    """Blended view across a company's signals."""
    from ..signals.score import composite_score

    rows = db.query(
        """
        SELECT s.name, s.cik, s.accession, s.as_of, s.score, s.confidence,
               s.direction, s.rationale
        FROM signals s
        LEFT JOIN companies c ON c.cik = s.cik
        WHERE upper(coalesce(s.ticker, c.ticker)) = ?
        """,
        [ticker.upper()],
    )
    if not rows:
        raise HTTPException(404, f"no signals for {ticker}")

    from datetime import date as _date

    from ..signals.score import Signal

    signals = [
        Signal(
            name=r["name"],
            cik=r["cik"],
            accession=r["accession"],
            as_of=r["as_of"] or _date.today(),
            score=r["score"],
            confidence=r["confidence"] or 0.0,
            direction=r["direction"],
            rationale=r["rationale"] or "",
            ticker=ticker.upper(),
        )
        for r in rows
    ]
    return {
        "ticker": ticker.upper(),
        **composite_score(signals),
        "signals": [{"name": s.name, "score": s.score, "as_of": str(s.as_of)} for s in signals],
    }


@app.get("/alerts")
def list_alerts(
    severity: str | None = None, limit: int = Query(50, le=500)
) -> list[dict[str, Any]]:
    clause = "WHERE a.severity = ?" if severity else ""
    return db.query(
        f"""
        SELECT a.alert_id, a.rule, a.severity, a.headline, a.detail, a.created_at,
               s.name AS signal_name, s.score, coalesce(s.ticker, c.ticker) AS ticker,
               s.accession
        FROM alerts a
        JOIN signals s ON s.signal_id = a.signal_id
        LEFT JOIN companies c ON c.cik = s.cik
        {clause}
        ORDER BY CASE a.severity WHEN 'critical' THEN 0 WHEN 'warn' THEN 1 ELSE 2 END,
                 a.created_at DESC
        LIMIT {int(limit)}
        """,
        [severity] if severity else [],
    )


# --------------------------------------------------------------------- search
@app.get("/search")
def search_corpus(
    q: str,
    k: int = Query(8, le=25),
    mode: str = Query("hybrid", pattern="^(hybrid|vector|keyword)$"),
    cik: str | None = None,
    section: str | None = None,
) -> list[dict[str, Any]]:
    from ..retrieval.index import search

    hits = search(q, k=k, mode=mode, cik=cik, slug=section, tracer=Tracer(enabled=False))
    return [
        {
            "chunk_id": h.chunk_id,
            "accession": h.accession,
            "cik": h.cik,
            "section": h.slug,
            "score": h.score,
            "sources": list(h.sources),
            "text": h.text,
        }
        for h in hits
    ]


# ---------------------------------------------------------------------- agent
class AgentRequest(BaseModel):
    question: str
    max_steps: int = Field(6, ge=1, le=20)
    chain: str = "agent"
    cost_cap_usd: float = Field(1.0, gt=0, le=50)


@app.post("/agent")
def ask_agent(req: AgentRequest) -> dict[str, Any]:
    """Answer a research question, returning the trace alongside the answer."""
    from ..agent.loop import run_agent
    from ..llm.router import Router

    tracer = Tracer()
    router = Router(tracer=tracer, cost_cap_usd=req.cost_cap_usd)
    result = run_agent(
        req.question,
        router=router,
        chain=req.chain,
        max_steps=req.max_steps,
        tracer=tracer,
    )
    tracer.flush()
    return {
        "question": result.question,
        "answer": result.answer,
        "stop_reason": result.stop_reason,
        "steps": [
            {
                "n": s.n,
                "tool": s.tool,
                "args": s.args,
                "ok": s.ok,
                "duration_s": round(s.duration_s, 3),
                "observation": s.observation[:2000],
            }
            for s in result.steps
        ],
        "stats": result.stats(),
    }


# -------------------------------------------------------------- observability
@app.get("/metrics/cost")
def cost_metrics(days: int = Query(7, ge=1, le=365)) -> dict[str, Any]:
    """Spend, tokens, latency, and cache hit rate by model."""
    by_model = db.query(
        f"""
        SELECT model, provider, count(*) AS calls,
               sum(CASE WHEN cached THEN 1 ELSE 0 END) AS cache_hits,
               sum(tokens_in) AS tokens_in, sum(tokens_out) AS tokens_out,
               round(sum(cost_usd), 6) AS cost_usd,
               round(avg(duration_ms), 1) AS avg_ms,
               round(quantile_cont(duration_ms, 0.95), 1) AS p95_ms,
               sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors
        FROM traces
        WHERE kind = 'llm' AND started_at > now() - INTERVAL {int(days)} DAY
        GROUP BY model, provider
        ORDER BY cost_usd DESC
        """
    )
    total = sum(r["cost_usd"] or 0 for r in by_model)
    calls = sum(r["calls"] for r in by_model)
    hits = sum(r["cache_hits"] or 0 for r in by_model)
    return {
        "window_days": days,
        "total_cost_usd": round(total, 6),
        "total_calls": calls,
        "cache_hit_rate": round(hits / calls, 3) if calls else 0.0,
        "by_model": by_model,
    }


@app.get("/metrics/runs")
def eval_runs(task: str | None = None, limit: int = Query(20, le=200)) -> list[dict[str, Any]]:
    """Recent eval runs, so quality history is queryable alongside the data."""
    clause = "WHERE task = ?" if task else ""
    return db.query(
        f"""
        SELECT run_id, suite, task, model, prompt_name, prompt_version, prompt_hash,
               n_cases, metrics, git_sha, started_at, duration_s, total_cost_usd
        FROM eval_runs {clause}
        ORDER BY started_at DESC
        LIMIT {int(limit)}
        """,
        [task] if task else [],
    )


@app.get("/review")
def review_queue(limit: int = Query(25, le=200)) -> list[dict[str, Any]]:
    """The human-in-the-loop queue, worst first."""
    return db.query(
        f"""
        SELECT review_id, extraction_id, task, reason, priority, status, proposed,
               created_at
        FROM review_queue
        WHERE status = 'open'
        ORDER BY priority DESC, created_at ASC
        LIMIT {int(limit)}
        """
    )


@app.get("/config")
def config() -> dict[str, Any]:
    """What is actually deployed: models, chains, prompt versions and hashes."""
    reg = get_registry()
    return {
        "default_provider": get_settings().default_provider,
        "models": {
            name: {
                "provider": s.provider,
                "context_tokens": s.context_tokens,
                "usd_per_mtok_in": s.usd_per_mtok_in,
                "usd_per_mtok_out": s.usd_per_mtok_out,
                "tier": s.tier,
            }
            for name, s in reg.models.items()
        },
        "chains": reg.chains,
        "prompts": get_prompts().manifest(),
    }


# ------------------------------------------------------------------ dashboard
@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).parent / "static" / "index.html"
    if not path.exists():
        return (
            "<h1>SignalForge</h1><p>Dashboard not found. API docs at <a href='/docs'>/docs</a>.</p>"
        )
    return path.read_text()


def _signal_row(r: dict[str, Any]) -> dict[str, Any]:
    import json

    evidence = r.get("evidence")
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except json.JSONDecodeError:
            evidence = []
    return {**r, "as_of": str(r["as_of"]), "evidence": evidence or []}
