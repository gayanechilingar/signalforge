"""Research skills — domain playbooks loaded into the agent's context.

A skill is a markdown file describing *how an analyst approaches a class of
question*: which tools in which order, what the domain traps are, what a good
answer looks like. They live as files rather than in the system prompt for the
same reasons prompts do — they are versioned, reviewable in a diff, and editable
by someone who knows investing without touching Python.

The tradeoff is context cost: every skill is charged on every step of every agent
run. So they are short, and each one earns its place by encoding knowledge the
model demonstrably lacks — that a 10-Q's Item 2 is MD&A while a 10-K's Item 7 is,
that "reaffirmed guidance after a weak quarter" is not bearish, that risk-factor
sections are mostly boilerplate. General reasoning advice is left out; the model
already has it.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from ..settings import get_settings

SKILLS_DIRNAME = "skills"


def skills_dir() -> Path:
    return get_settings().prompt_dir / SKILLS_DIRNAME


@lru_cache
def load_skills() -> str:
    """Concatenate the skill library into a single context block."""
    directory = skills_dir()
    if not directory.exists():
        return ""

    blocks = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text().strip()
        # Strip frontmatter if present; the agent needs the body, not metadata.
        text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.S)
        if text:
            blocks.append(text)

    if not blocks:
        return ""
    return "Research playbooks:\n\n" + "\n\n---\n\n".join(blocks)


def list_skills() -> list[str]:
    directory = skills_dir()
    return sorted(p.stem for p in directory.glob("*.md")) if directory.exists() else []
