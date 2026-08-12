"""Content-addressed LLM response cache.

Serves two purposes that are usually conflated:

1. **Cost and latency.** Re-running an eval suite after changing the scoring
   code should not re-pay for inference.
2. **Reproducibility.** The cache key is a digest of every input that can change
   the output (messages, model, temperature, schema, tools). A cached run is
   therefore a *replay* of an experiment, not an approximation of it — which is
   what makes "same prompt version, same model, same numbers" a claim we can
   actually make.

Kept in its own DuckDB file so that wiping the cache never risks the warehouse.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from ..settings import get_settings
from .base import LLMRequest, LLMResponse

_DDL = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key         VARCHAR PRIMARY KEY,
    provider    VARCHAR,
    model       VARCHAR,
    response    JSON NOT NULL,
    created_at  TIMESTAMP DEFAULT current_timestamp,
    hits        INTEGER DEFAULT 0
);
"""


class ResponseCache:
    def __init__(self, path: Path | None = None, *, enabled: bool | None = None) -> None:
        s = get_settings()
        self.path = path or s.cache_path
        self.enabled = s.cache_enabled if enabled is None else enabled
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._con() as con:
                con.execute(_DDL)

    def _con(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.path))

    def get(self, req: LLMRequest) -> LLMResponse | None:
        if not self.enabled:
            return None
        key = req.cache_key()
        try:
            with self._con() as con:
                row = con.execute("SELECT response FROM llm_cache WHERE key = ?", [key]).fetchone()
                if not row:
                    return None
                con.execute("UPDATE llm_cache SET hits = hits + 1 WHERE key = ?", [key])
        except duckdb.Error:
            return None

        payload = json.loads(row[0])
        resp = LLMResponse.model_validate(payload)
        # A replay costs nothing and takes no time; reporting the original
        # figures would inflate every cost dashboard.
        resp.cached = True
        resp.cost_usd = 0.0
        resp.latency_ms = 0.0
        return resp

    def put(self, req: LLMRequest, resp: LLMResponse) -> None:
        if not self.enabled or resp.cached:
            return
        try:
            with self._con() as con:
                con.execute(
                    "INSERT OR REPLACE INTO llm_cache (key, provider, model, response) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        req.cache_key(),
                        resp.provider,
                        resp.model,
                        resp.model_dump_json(),
                    ],
                )
        except duckdb.Error:
            # A cache that fails to write must not fail the call.
            pass

    def stats(self) -> dict[str, int]:
        if not self.enabled:
            return {"entries": 0, "hits": 0}
        with self._con() as con:
            row = con.execute("SELECT count(*), coalesce(sum(hits), 0) FROM llm_cache").fetchone()
        if row is None:
            return {"entries": 0, "hits": 0}
        return {"entries": int(row[0]), "hits": int(row[1])}

    def clear(self) -> None:
        if not self.enabled:
            return
        with self._con() as con:
            con.execute("DELETE FROM llm_cache")
