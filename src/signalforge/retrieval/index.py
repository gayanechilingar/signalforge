"""Embedding index and hybrid retrieval over the warehouse.

Vector search is done in DuckDB with ``array_cosine_similarity`` where the build
supports it, falling back to NumPy. At this corpus size (tens of thousands of
chunks) an exact scan is milliseconds and is *correct*, whereas an approximate
index adds a recall parameter to tune and a failure mode to debug. Reaching for
a dedicated vector database here would be architecture for its own sake; the
interface below is narrow enough to swap when the corpus justifies it.

Retrieval is hybrid by default. Pure vector search on filings is weaker than it
looks: much of what an analyst searches for is an exact term — a product name, a
statute, "material weakness" — and embeddings smear those into their neighbours.
Keyword scoring catches the exact hits, vectors catch the paraphrases, and the
two are combined with reciprocal rank fusion, which needs no score calibration
between the two systems.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

from ..db import connect, upsert
from ..llm.router import Router
from ..obs.tracing import Tracer, default_tracer

log = logging.getLogger(__name__)

#: Embedding calls dominate index time; batching cuts the round-trips.
EMBED_BATCH = 32


@dataclass(slots=True)
class Hit:
    chunk_id: str
    accession: str
    cik: str
    section_id: str
    text: str
    score: float
    #: Which retriever(s) surfaced this chunk — useful when debugging why a
    #: relevant passage was or wasn't found.
    sources: tuple[str, ...] = ()

    @property
    def slug(self) -> str:
        return self.section_id.rsplit(":", 1)[-1]


def build_index(
    *,
    router: Router | None = None,
    model: str | None = None,
    accessions: list[str] | None = None,
    tracer: Tracer | None = None,
    batch_size: int = EMBED_BATCH,
) -> int:
    """Embed chunks that have no vector yet for the given model.

    Incremental by construction: re-running after ingesting more filings embeds
    only the new chunks. Switching embedding model adds rows rather than
    overwriting, so a model change is reversible and comparable.
    """
    router = router or Router()
    tracer = tracer or default_tracer
    spec = router.registry.embedding(model)

    with tracer.span("retrieval.build_index", kind="pipeline", model=spec.name) as span:
        params: list[Any] = [spec.name]
        sql = """
            SELECT c.chunk_id, c.text
            FROM chunks c
            LEFT JOIN embeddings e
              ON e.chunk_id = c.chunk_id AND e.model = ?
            WHERE e.chunk_id IS NULL
        """
        if accessions:
            placeholders = ", ".join("?" for _ in accessions)
            sql += f" AND c.accession IN ({placeholders})"
            params.extend(accessions)
        sql += " ORDER BY c.chunk_id"

        with connect() as con:
            pending = con.execute(sql, params).fetchall()

        if not pending:
            span.set(embedded=0)
            return 0

        total = 0
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            vectors = router.embed([text for _, text in batch], model=spec.name)
            rows = [
                {
                    "chunk_id": chunk_id,
                    "model": spec.name,
                    "dim": len(vec),
                    "vec": vec,
                }
                for (chunk_id, _), vec in zip(batch, vectors, strict=True)
            ]
            with connect() as con:
                upsert(con, "embeddings", rows, key="chunk_id")
            total += len(rows)

        span.set(embedded=total)
        return total


def search(
    query: str,
    *,
    router: Router | None = None,
    model: str | None = None,
    k: int = 8,
    cik: str | None = None,
    accession: str | None = None,
    slug: str | None = None,
    mode: str = "hybrid",
    tracer: Tracer | None = None,
) -> list[Hit]:
    """Retrieve the ``k`` most relevant chunks.

    ``mode`` is one of ``hybrid`` (default), ``vector``, or ``keyword``. The
    filters narrow the candidate set *before* scoring, which is what makes
    "what did Apple say about supply chains in this 10-K" a cheap query.
    """
    router = router or Router()
    tracer = tracer or default_tracer

    with tracer.span("retrieval.search", kind="tool", mode=mode, k=k) as span:
        filters, params = _filters(cik=cik, accession=accession, slug=slug)

        vector_hits: list[Hit] = []
        keyword_hits: list[Hit] = []

        if mode in ("hybrid", "vector"):
            vector_hits = _vector_search(
                query, router=router, model=model, k=k * 3, filters=filters, params=params
            )
        if mode in ("hybrid", "keyword"):
            keyword_hits = _keyword_search(query, k=k * 3, filters=filters, params=params)

        if mode == "vector":
            results = vector_hits[:k]
        elif mode == "keyword":
            results = keyword_hits[:k]
        else:
            results = _fuse(vector_hits, keyword_hits)[:k]

        span.set(results=len(results), vector=len(vector_hits), keyword=len(keyword_hits))
        return results


def _filters(*, cik: str | None, accession: str | None, slug: str | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if cik:
        clauses.append("c.cik = ?")
        params.append(cik)
    if accession:
        clauses.append("c.accession = ?")
        params.append(accession)
    if slug:
        # Slug lives at the tail of section_id; matching the suffix avoids a join.
        clauses.append("c.section_id LIKE ?")
        params.append(f"%:{slug}")
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def _vector_search(
    query: str,
    *,
    router: Router,
    model: str | None,
    k: int,
    filters: str,
    params: list[Any],
) -> list[Hit]:
    spec = router.registry.embedding(model)
    qvec = router.embed([query], model=spec.name)[0]

    sql = f"""
        SELECT c.chunk_id, c.accession, c.cik, c.section_id, c.text,
               array_cosine_similarity(
                   e.vec::FLOAT[{len(qvec)}], ?::FLOAT[{len(qvec)}]
               ) AS score
        FROM chunks c
        JOIN embeddings e ON e.chunk_id = c.chunk_id
        WHERE e.model = ? {filters}
        ORDER BY score DESC
        LIMIT ?
    """
    try:
        with connect() as con:
            rows = con.execute(sql, [qvec, spec.name, *params, k]).fetchall()
    except Exception as exc:
        # Older DuckDB builds lack array_cosine_similarity. Falling back keeps a
        # fresh clone working rather than demanding a specific patch version.
        log.debug("native cosine unavailable (%s); using numpy fallback", exc)
        rows = _numpy_cosine(qvec, spec.name, filters, params, k)

    return [
        Hit(
            chunk_id=r[0],
            accession=r[1],
            cik=r[2],
            section_id=r[3],
            text=r[4],
            score=float(r[5]),
            sources=("vector",),
        )
        for r in rows
    ]


def _numpy_cosine(
    qvec: list[float], model: str, filters: str, params: list[Any], k: int
) -> list[tuple]:
    import numpy as np

    sql = f"""
        SELECT c.chunk_id, c.accession, c.cik, c.section_id, c.text, e.vec
        FROM chunks c
        JOIN embeddings e ON e.chunk_id = c.chunk_id
        WHERE e.model = ? {filters}
    """
    with connect() as con:
        rows = con.execute(sql, [model, *params]).fetchall()
    if not rows:
        return []

    mat = np.asarray([r[5] for r in rows], dtype=np.float32)
    q = np.asarray(qvec, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(q)
    norms[norms == 0] = 1e-9
    scores = (mat @ q) / norms

    order = np.argsort(-scores)[:k]
    return [(*rows[i][:5], float(scores[i])) for i in order]


def _keyword_search(query: str, *, k: int, filters: str, params: list[Any]) -> list[Hit]:
    """Term-overlap scoring with IDF-style weighting.

    Written in SQL rather than pulled into Python so it scales with the warehouse
    and stays usable from the agent's SQL tool. Not BM25 — but the ranking only
    needs to be good enough to feed rank fusion, which is insensitive to score
    scale.
    """
    terms = _terms(query)
    if not terms:
        return []

    # One scored LIKE per term; DuckDB handles this efficiently at this scale.
    score_expr = " + ".join(
        f"(CASE WHEN lower(c.text) LIKE ? THEN {weight:.4f} ELSE 0 END)" for _, weight in terms
    )
    like_params = [f"%{term}%" for term, _ in terms]

    # Scored in a subquery so the alias can be filtered on: DuckDB's QUALIFY is
    # only valid alongside a window function, and WHERE cannot see a SELECT alias.
    sql = f"""
        SELECT chunk_id, accession, cik, section_id, text, score
        FROM (
            SELECT c.chunk_id, c.accession, c.cik, c.section_id, c.text,
                   ({score_expr}) AS score,
                   length(c.text) AS text_len
            FROM chunks c
            WHERE 1=1 {filters}
        )
        WHERE score > 0
        ORDER BY score DESC, text_len ASC
        LIMIT ?
    """
    with connect() as con:
        rows = con.execute(sql, [*like_params, *params, k]).fetchall()

    return [
        Hit(
            chunk_id=r[0],
            accession=r[1],
            cik=r[2],
            section_id=r[3],
            text=r[4],
            score=float(r[5]),
            sources=("keyword",),
        )
        for r in rows
    ]


_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "what",
        "which",
        "how",
        "why",
        "when",
        "do",
        "does",
        "did",
        "about",
    ]
)


def _terms(query: str) -> list[tuple[str, float]]:
    """Content terms with a length-based weight.

    Longer terms are more discriminative than short ones ("impairment" says more
    than "cost"), which is a cheap stand-in for a corpus IDF and avoids
    maintaining term statistics that go stale as the warehouse grows.
    """
    words = [w for w in re.findall(r"[a-z][a-z0-9'-]{2,}", query.lower()) if w not in _STOP]
    seen: dict[str, float] = {}
    for w in words:
        seen[w] = 1.0 + math.log(len(w))
    return sorted(seen.items(), key=lambda kv: -kv[1])[:12]


def _fuse(*rankings: list[Hit], k_rrf: int = 60) -> list[Hit]:
    """Reciprocal rank fusion.

    Combines rankings by position rather than by score, so a cosine similarity in
    [-1, 1] and an unbounded keyword score can be merged without normalising
    either — the usual source of subtle bugs in hybrid retrieval.
    """
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    sources: dict[str, set[str]] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k_rrf + rank + 1)
            sources.setdefault(hit.chunk_id, set()).update(hit.sources)
            best.setdefault(hit.chunk_id, hit)

    out: list[Hit] = []
    for chunk_id, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        hit = best[chunk_id]
        out.append(
            Hit(
                chunk_id=hit.chunk_id,
                accession=hit.accession,
                cik=hit.cik,
                section_id=hit.section_id,
                text=hit.text,
                score=round(score, 6),
                sources=tuple(sorted(sources[chunk_id])),
            )
        )
    return out


def index_stats(model: str | None = None) -> dict[str, Any]:
    """Chunk and embedding counts, optionally narrowed to one embedding model."""
    # `model` was accepted and then ignored, so asking for one model's coverage
    # silently reported every model's.
    clause, params = ("WHERE model = ?", [model]) if model else ("", [])
    with connect() as con:
        row = con.execute("SELECT count(*) FROM chunks").fetchone()
        rows = con.execute(
            f"SELECT model, count(*), min(dim) FROM embeddings {clause} GROUP BY model",
            params,
        ).fetchall()
    return {
        "chunks": row[0] if row else 0,
        "embeddings": {r[0]: {"count": r[1], "dim": r[2]} for r in rows},
    }
