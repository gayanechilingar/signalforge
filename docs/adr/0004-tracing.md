# 4. Tracing to a table, not OpenTelemetry

**Status:** accepted

## Context

Cost, latency, cache-hit rate, and error rate per model need to be observable.
The industry answer is OpenTelemetry.

## Decision

A ~150-line `Tracer` that buffers spans and batch-writes them to a `traces` table
in the same warehouse as the data.

## Consequences

**Good.** Operational questions are SQL, joined against the data they describe:
"what did each model cost this week", "which prompt version has the worst p95".
That is what `/metrics/cost` reads. No collector to run, and batching keeps the
overhead below the cost of a single LLM call.

**Bad.** No distributed context propagation, no sampling, no existing dashboard
ecosystem. Trace volume grows unbounded in the warehouse.

**Migration path.** The surface is one context manager (`Tracer.span`) plus
`Span.set`. Re-implementing it over an OTel exporter touches no call sites. That
is the right move as soon as there is more than one service to correlate; until
then OTel is a running dependency in exchange for a table we can already query.
