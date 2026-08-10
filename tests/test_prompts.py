"""Prompt registry: versioning, hashing, and strict rendering.

The hash tests are the load-bearing ones — the regression gate's ability to
notice an uncommitted prompt edit depends on them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signalforge.prompts.registry import PromptRegistry, get_prompts

FIXTURE = """---
name: demo
version: v1
task: demo_task
description: A demo prompt.
variables: [subject]
---
Analyse {{ subject }} carefully.
"""


@pytest.fixture
def reg(tmp_path: Path) -> PromptRegistry:
    (tmp_path / "demo_v1.md").write_text(FIXTURE)
    (tmp_path / "demo_v2.md").write_text(
        FIXTURE.replace("version: v1", "version: v2").replace("carefully", "rigorously")
    )
    (tmp_path / "demo_v10.md").write_text(FIXTURE.replace("version: v1", "version: v10"))
    return PromptRegistry(root=tmp_path)


def test_loads_all_versions(reg):
    assert reg.versions("demo") == ["v1", "v2", "v10"]


def test_default_resolves_to_highest_version_numerically(reg):
    # Lexical sort would pick v2; the registry must sort numerically.
    assert reg.get("demo").version == "v10"


def test_explicit_version_selection(reg):
    assert "carefully" in reg.get("demo", "v1").template
    assert "rigorously" in reg.get("demo", "v2").template


def test_hash_is_stable_and_distinguishes_versions(reg):
    v1, v2 = reg.get("demo", "v1"), reg.get("demo", "v2")
    assert v1.hash == reg.get("demo", "v1").hash
    assert v1.hash != v2.hash, "an edited template must change the hash"


def test_render_substitutes(reg):
    assert reg.get("demo", "v1").render(subject="Apple") == "Analyse Apple carefully."


def test_render_rejects_missing_variable(reg):
    with pytest.raises(KeyError, match="needs variables"):
        reg.get("demo", "v1").render()


def test_render_rejects_unused_variable(reg):
    """A renamed variable silently dropping out of the prompt is a real bug that
    evals may not surface, so it fails loudly at render time instead."""
    with pytest.raises(KeyError, match="unused variables"):
        reg.get("demo", "v1").render(subject="Apple", sbject="typo")


def test_frontmatter_variable_list_must_match_template(tmp_path: Path):
    (tmp_path / "bad.md").write_text(FIXTURE.replace("[subject]", "[subject, extra]"))
    with pytest.raises(ValueError, match="declares variables"):
        PromptRegistry(root=tmp_path)


def test_skills_subdirectory_is_not_parsed_as_prompts(tmp_path: Path):
    """Agent skills are markdown under prompts/ but are not prompts.

    Without the exclusion the registry raises on their looser frontmatter, which
    takes down every command that loads prompts — including the eval harness.
    """
    (tmp_path / "demo_v1.md").write_text(FIXTURE)
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "playbook.md").write_text("---\nname: playbook\n---\nSome guidance.\n")

    reg = PromptRegistry(root=tmp_path)
    assert [p.name for p in reg.all()] == ["demo"]


def test_missing_frontmatter_is_rejected(tmp_path: Path):
    (tmp_path / "raw.md").write_text("Just a prompt with no metadata.")
    with pytest.raises(ValueError, match="missing YAML frontmatter"):
        PromptRegistry(root=tmp_path)


def test_duplicate_version_is_rejected(tmp_path: Path):
    (tmp_path / "a.md").write_text(FIXTURE)
    (tmp_path / "b.md").write_text(FIXTURE)
    with pytest.raises(ValueError, match="duplicate prompt"):
        PromptRegistry(root=tmp_path)


def test_unknown_prompt_error_lists_alternatives(reg):
    with pytest.raises(KeyError, match="available versions"):
        reg.get("demo", "v99")
    with pytest.raises(KeyError, match="registry has"):
        reg.get("nope")


def test_manifest_shape(reg):
    manifest = reg.manifest()
    assert {"name", "version", "hash", "task"} == set(manifest[0])


def test_shipped_prompts_are_valid():
    """The real prompts/ directory must parse — this is the guard that keeps a
    malformed prompt from reaching CI as a runtime error."""
    prompts = get_prompts()
    assert prompts.all(), "expected at least one shipped prompt"
    for p in prompts.all():
        assert p.task, f"{p.key} has no task declared"
        assert p.hash
