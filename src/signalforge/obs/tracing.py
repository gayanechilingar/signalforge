"""Lightweight tracing.

A span tree per pipeline or agent run, persisted to the warehouse so that cost,
latency, and failure questions are answerable with SQL after the fact:

    SELECT model, count(*), avg(duration_ms), sum(cost_usd)
    FROM traces WHERE kind='llm' GROUP BY 1;

Deliberately not OpenTelemetry. OTel is the right answer once there is a
collector to ship to; until then it is a dependency and a running service in
exchange for a table we can already query. ``docs/adr/0004-tracing.md`` records
the swap-out path — the ``Tracer`` surface is small enough to re-implement over
an OTel exporter without touching call sites.
"""

from __future__ import annotations

import contextvars
import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..db import connect, upsert
from ..settings import get_settings

_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sf_current_span", default=None
)
_current_trace: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sf_current_trace", default=None
)


@dataclass(slots=True)
class Span:
    name: str
    kind: str = "internal"
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""
    parent_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = 0.0
    status: str = "ok"
    error: str | None = None
    model: str | None = None
    provider: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    attrs: dict[str, Any] = field(default_factory=dict)

    def set(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.attrs[k] = v

    def row(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "provider": self.provider,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "cached": self.cached,
            "attrs": self.attrs,
            "error": self.error,
        }


class Tracer:
    """Buffers spans in memory and flushes them in one write.

    Batching matters: a single pipeline run produces hundreds of spans, and a
    DuckDB write per span would dominate the runtime of the cheap stub provider.
    """

    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = get_settings().trace_enabled if enabled is None else enabled
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, kind: str = "internal", **attrs: Any) -> Iterator[Span]:
        trace_id = _current_trace.get() or uuid.uuid4().hex[:16]
        sp = Span(
            name=name,
            kind=kind,
            trace_id=trace_id,
            parent_id=_current_span.get(),
            attrs=dict(attrs),
        )
        t_trace = _current_trace.set(trace_id)
        t_span = _current_span.set(sp.span_id)
        t0 = time.perf_counter()
        try:
            yield sp
        except Exception as exc:
            sp.status = "error"
            sp.error = f"{type(exc).__name__}: {exc}"[:2000]
            raise
        finally:
            sp.duration_ms = (time.perf_counter() - t0) * 1000
            _current_span.reset(t_span)
            _current_trace.reset(t_trace)
            self.spans.append(sp)

    # -- aggregates used by cost caps and run summaries --------------------
    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_in + s.tokens_out for s in self.spans)

    def summary(self) -> dict[str, Any]:
        llm = [s for s in self.spans if s.kind == "llm"]
        errs = [s for s in self.spans if s.status == "error"]
        return {
            "spans": len(self.spans),
            "llm_calls": len(llm),
            "cache_hits": sum(1 for s in llm if s.cached),
            "errors": len(errs),
            "tokens": self.total_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
            "wall_ms": round(sum(s.duration_ms for s in self.spans if s.parent_id is None), 1),
        }

    def flush(self) -> int:
        if not self.enabled or not self.spans:
            self.spans.clear()
            return 0
        rows = [s.row() for s in self.spans]
        try:
            with connect() as con:
                upsert(con, "traces", rows, key="span_id")
        except Exception:
            # Observability must never take down the thing it observes.
            return 0
        finally:
            self.spans.clear()
        return len(rows)

    def to_json(self) -> str:
        return json.dumps([s.row() for s in self.spans], default=str)


#: Process-wide default tracer. Pipelines create their own so that a run's cost
#: total is scoped to that run.
default_tracer = Tracer()
