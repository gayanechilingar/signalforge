"""Versioned, content-hashed prompt registry.

A prompt is code. Treating it as a string literal buried in a function makes two
things impossible that this project depends on:

* **Attributing a result.** Every extraction row records the prompt name, its
  declared version, *and* the SHA-256 of its rendered template. The version is
  what a human maintains; the hash is what actually shipped. If someone edits
  ``v2.md`` without bumping the version, the hash changes and the regression gate
  notices — a silent prompt edit is the single easiest way to invalidate an eval
  suite without anyone realising.
* **Comparing two prompts fairly.** Because versions live side by side as files,
  ``sf eval run --prompt-version v1`` vs ``v2`` is a real A/B on the same
  ground truth and the same model.

Format: markdown with YAML frontmatter.

    ---
    name: risk_delta
    version: v2
    task: risk_delta
    description: Compare risk-factor sections across consecutive filings.
    variables: [company, current_text, prior_text]
    ---
    You are a securities analyst...
    <document name="current">{{ current_text }}</document>

Rendering is intentionally a dumb ``{{ var }}`` substitution rather than Jinja:
prompt templates that can branch and loop become programs, and programs need
their own tests. Every variable must be supplied — a silently-empty ``{{ }}``
would produce a plausible-looking prompt with a missing document in it.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..settings import get_settings

_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)

#: Subdirectories of the prompt tree that hold markdown which is *not* a prompt.
#: Agent research skills live in ``prompts/skills/`` and have their own, looser
#: frontmatter contract; without this exclusion the registry tries to parse them
#: as prompts and every command that loads prompts dies on a missing 'version'.
EXCLUDED_DIRS = frozenset({"skills"})


def _is_excluded(path: Path) -> bool:
    return bool(EXCLUDED_DIRS & set(path.parts))


class Prompt(BaseModel):
    name: str
    version: str
    task: str = ""
    description: str = ""
    variables: list[str] = Field(default_factory=list)
    template: str
    path: Path | None = None

    @property
    def hash(self) -> str:
        """Digest of the template body — the identity of what actually ran."""
        return hashlib.sha256(self.template.encode()).hexdigest()[:16]

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    def declared_vars(self) -> set[str]:
        return set(_VAR.findall(self.template))

    def render(self, **kwargs: object) -> str:
        found = self.declared_vars()
        missing = found - set(kwargs)
        if missing:
            raise KeyError(
                f"prompt {self.key} needs variables {sorted(missing)} that were not supplied"
            )
        extra = set(kwargs) - found
        if extra:
            # Loud rather than ignored: a renamed variable that silently stops
            # being interpolated is a prompt bug that evals may not catch.
            raise KeyError(
                f"prompt {self.key} was given unused variables {sorted(extra)}; "
                f"template uses {sorted(found)}"
            )
        return _VAR.sub(lambda m: str(kwargs[m.group(1)]), self.template)


class PromptRegistry:
    """All prompts on disk, indexed by name and version."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_settings().prompt_dir
        self._by_key: dict[str, Prompt] = {}
        self._by_name: dict[str, list[Prompt]] = {}
        self._load()

    def _load(self) -> None:
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob("*.md")):
            if _is_excluded(path):
                continue
            prompt = _parse(path)
            if prompt.key in self._by_key:
                raise ValueError(
                    f"duplicate prompt {prompt.key} at {path} and {self._by_key[prompt.key].path}"
                )
            self._by_key[prompt.key] = prompt
            self._by_name.setdefault(prompt.name, []).append(prompt)

    def get(self, name: str, version: str | None = None) -> Prompt:
        """Fetch a prompt; ``version=None`` resolves to the highest version.

        "Highest" is a natural sort on the version string, so ``v10`` sorts after
        ``v9`` rather than before it.
        """
        if version:
            try:
                return self._by_key[f"{name}@{version}"]
            except KeyError:
                available = [p.version for p in self._by_name.get(name, [])]
                raise KeyError(
                    f"no prompt {name}@{version}; available versions: {available}"
                ) from None
        versions = self._by_name.get(name)
        if not versions:
            raise KeyError(f"no prompt named {name!r}; registry has: {sorted(self._by_name)}")
        return max(versions, key=lambda p: _version_key(p.version))

    def versions(self, name: str) -> list[str]:
        return sorted((p.version for p in self._by_name.get(name, [])), key=_version_key)

    def for_task(self, task: str) -> list[Prompt]:
        return [p for p in self._by_key.values() if p.task == task]

    def all(self) -> list[Prompt]:
        return list(self._by_key.values())

    def manifest(self) -> list[dict[str, str]]:
        """Name/version/hash triples — printed by ``sf prompts list`` and
        embedded in eval reports so a report is self-describing."""
        return [
            {"name": p.name, "version": p.version, "hash": p.hash, "task": p.task}
            for p in sorted(self._by_key.values(), key=lambda p: (p.name, p.version))
        ]


def _parse(path: Path) -> Prompt:
    raw = path.read_text()
    m = _FRONTMATTER.match(raw)
    if not m:
        raise ValueError(f"prompt {path} is missing YAML frontmatter")
    meta = yaml.safe_load(m.group(1)) or {}
    body = m.group(2).strip()

    for field in ("name", "version"):
        if field not in meta:
            raise ValueError(f"prompt {path} frontmatter is missing required '{field}'")

    prompt = Prompt(
        name=meta["name"],
        version=str(meta["version"]),
        task=meta.get("task", ""),
        description=meta.get("description", ""),
        variables=list(meta.get("variables") or []),
        template=body,
        path=path,
    )

    # If the author listed variables, hold them to it — a drifted list is a
    # documentation lie, and this file is the documentation.
    if prompt.variables:
        declared = set(prompt.variables)
        used = prompt.declared_vars()
        if declared != used:
            raise ValueError(
                f"prompt {path}: frontmatter declares variables {sorted(declared)} "
                f"but the template uses {sorted(used)}"
            )
    return prompt


def _version_key(version: str) -> tuple[int, str]:
    m = re.match(r"v?(\d+)", version)
    return (int(m.group(1)) if m else 0, version)


@lru_cache
def get_prompts() -> PromptRegistry:
    return PromptRegistry()
