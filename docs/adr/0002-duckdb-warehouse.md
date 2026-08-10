# 2. DuckDB as the warehouse, including vector search

**Status:** accepted

## Context

The system needs a corpus store, a vector index, an extraction record, a signal
store, and a trace/eval log. The conventional stack is Postgres plus pgvector plus
an object store plus a metrics backend.

## Decision

One embedded DuckDB file holds all of it. Vector search is an exact scan using
`array_cosine_similarity`, with a NumPy fallback for older builds.

## Consequences

**Good.** Zero infrastructure: `git clone && make demo` works. Analytical queries
over filings, extractions, signals, and traces are single-file joins, which is
what makes `/metrics/cost` and the agent's SQL tool trivial rather than a
cross-service problem. An exact vector scan has no recall parameter to tune and
cannot silently degrade.

**Bad.** DuckDB is single-writer, so the access pattern is short-lived
connections per unit of work rather than a pool; concurrent writers block. Exact
scan is linear — fine at ~10⁴–10⁵ chunks, not at 10⁷.

**Migration path.** The `db.py` surface is small (`connect`, `upsert`, `query`)
and retrieval is behind `retrieval/index.py`. Moving to Postgres + pgvector means
reimplementing those two modules, not the pipeline.
