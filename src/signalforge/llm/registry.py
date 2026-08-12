"""Model registry — loads ``configs/models.yaml`` into typed specs.

Pipelines name models by registry key, never by provider-side ID. That one level
of indirection is what makes "benchmark llama3.2 against llama3.1 against Sonnet"
a config edit instead of a refactor, and it keeps pricing in exactly one place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..settings import get_settings
from .base import ModelSpec


class EmbedSpec(BaseModel):
    name: str
    provider: str
    model_id: str
    dim: int
    usd_per_mtok_in: float = 0.0


class Registry(BaseModel):
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    embeddings: dict[str, EmbedSpec] = Field(default_factory=dict)
    default_embedding: str = "nomic-embed-text"
    chains: dict[str, list[str]] = Field(default_factory=dict)

    def spec(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError:
            raise KeyError(
                f"unknown model {name!r}; registry has: {', '.join(sorted(self.models))}"
            ) from None

    def chain(self, name_or_chain: str | list[str]) -> list[ModelSpec]:
        """Resolve a chain name, a single model name, or an explicit list."""
        if isinstance(name_or_chain, list):
            names = name_or_chain
        elif name_or_chain in self.chains:
            names = self.chains[name_or_chain]
        else:
            names = [name_or_chain]
        return [self.spec(n) for n in names]

    def embedding(self, name: str | None = None) -> EmbedSpec:
        """Resolve an embedding model: explicit argument, then env, then YAML.

        The env layer exists so that ``SF_EMBED_MODEL=stub-embed`` makes the
        *whole* system hermetic in one variable. Without it, any code path that
        embeds without naming a model — retrieval called from the API, for
        instance — silently reaches for real Ollama even when the completion
        provider is stubbed. That is a test-isolation bug that passes on a laptop
        with Ollama running and fails in CI, which is the worst combination.
        """
        key = name or get_settings().embed_model or self.default_embedding
        try:
            return self.embeddings[key]
        except KeyError:
            raise KeyError(f"unknown embedding model {key!r}") from None

    def by_provider(self, provider: str) -> list[ModelSpec]:
        return [m for m in self.models.values() if m.provider == provider]


def load_registry(path: Path | None = None) -> Registry:
    path = path or get_settings().config_dir / "models.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}

    models = {}
    for entry in raw.get("models", []):
        spec = ModelSpec.model_validate(entry)
        models[spec.name] = spec

    emb_raw = raw.get("embeddings") or {}
    embeddings = {}
    for entry in emb_raw.get("models", []):
        e = EmbedSpec.model_validate(entry)
        embeddings[e.name] = e

    return Registry(
        models=models,
        embeddings=embeddings,
        default_embedding=emb_raw.get("default", "nomic-embed-text"),
        chains=raw.get("chains") or {},
    )


@lru_cache
def get_registry() -> Registry:
    return load_registry()
