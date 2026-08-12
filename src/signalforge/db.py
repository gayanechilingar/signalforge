"""DuckDB warehouse access.

DuckDB is a single-writer embedded engine, so the access pattern here is
deliberately short-lived connections opened per unit of work rather than one
long-lived global handle. That keeps the CLI, the API, and the eval harness from
deadlocking each other, at the cost of a few milliseconds per open — irrelevant
next to an LLM call.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

from .settings import get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@contextmanager
def connect(
    path: Path | str | None = None, *, read_only: bool = False
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a connection with the schema applied.

    ``read_only`` is used by the agent's SQL tool: a second line of defence
    behind statement parsing, so a prompt injection that slips a DELETE past the
    parser still cannot write.
    """
    s = get_settings()
    db_path = Path(path) if path is not None else s.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if read_only and not db_path.exists():
        # DuckDB refuses read_only on a missing file; create it first.
        duckdb.connect(str(db_path)).close()

    con = duckdb.connect(str(db_path), read_only=read_only)
    try:
        if not read_only:
            con.execute(SCHEMA_PATH.read_text())
        yield con
    finally:
        con.close()


def init_db(path: Path | str | None = None) -> Path:
    with connect(path) as con:
        con.execute("SELECT 1")
    return Path(path) if path is not None else get_settings().db_path


def upsert(
    con: duckdb.DuckDBPyConnection,
    table: str,
    rows: Sequence[dict[str, Any]],
    *,
    key: str,
) -> int:
    """Insert rows, replacing any existing row with the same primary key.

    DuckDB supports ``INSERT OR REPLACE``; JSON-typed columns need dict/list
    values serialised on the way in.

    ``key`` documents the caller's intent only — replacement is always resolved by
    the table's declared PRIMARY KEY, not by this argument. Read it as a comment,
    never as a guarantee: ``embeddings`` is keyed on ``(chunk_id, model)``, so a
    call passing ``key="chunk_id"`` still keeps one row per model.
    """
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    payload = [[_encode(r[c]) for c in cols] for r in rows]
    con.executemany(sql, payload)
    return len(payload)


def _encode(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def query(
    sql: str, params: Sequence[Any] | None = None, *, path: Path | str | None = None
) -> list[dict[str, Any]]:
    """Run a read query and return dict rows. Convenience for API/CLI layers."""
    with connect(path) as con:
        cur = con.execute(sql, list(params or []))
        cols = [d[0] for d in cur.description or []]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
