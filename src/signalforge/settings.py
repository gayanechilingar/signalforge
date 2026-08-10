"""Central configuration.

Everything that varies between a laptop, CI, and a deployed box lives here and
only here. Defaults are chosen so that `sf` works on a fresh clone with no
environment variables set at all.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths -------------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    prompt_dir: Path = REPO_ROOT / "prompts"
    config_dir: Path = REPO_ROOT / "configs"
    eval_dir: Path = REPO_ROOT / "evals"

    # --- warehouse ---------------------------------------------------------
    db_path: Path = REPO_ROOT / "data" / "signalforge.duckdb"

    # --- providers ---------------------------------------------------------
    #: Provider used when a pipeline config does not name one.
    default_provider: str = "ollama"
    ollama_host: str = "http://localhost:11434"
    ollama_timeout_s: float = 300.0
    embed_model: str = "nomic-embed-text"

    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com"

    # --- reliability / cost ------------------------------------------------
    #: Hard ceiling per pipeline run. Exceeding it raises rather than silently
    #: burning money — cost bugs should be loud.
    run_cost_cap_usd: float = 5.0
    max_retries: int = 3
    cache_enabled: bool = True

    # --- SEC EDGAR ---------------------------------------------------------
    #: The SEC requires a descriptive UA with contact info, and caps clients at
    #: 10 req/s. Both are enforced in ingest/edgar.py.
    sec_user_agent: str = "SignalForge/0.1 (research; contact@example.com)"
    sec_rate_limit_per_s: float = 6.0

    log_level: str = "INFO"
    trace_enabled: bool = True

    # Set by tests / CI to force determinism.
    deterministic: bool = Field(default=False)

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "llm_cache.duckdb"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
