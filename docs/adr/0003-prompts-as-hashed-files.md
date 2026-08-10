# 3. Prompts as versioned, content-hashed files

**Status:** accepted

## Context

Prompts are the highest-leverage and least-controlled part of an LLM system. As
string literals in Python they cannot be diffed meaningfully, A/B tested, or
attributed to a result — and an eval suite that cannot attribute a result cannot
detect a regression caused by an unrecorded prompt edit.

## Decision

Every prompt is a markdown file with YAML frontmatter declaring `name`, `version`,
`task`, and `variables`. Versions coexist as files (`guidance_tone_v1.md`,
`_v2.md`). The registry validates that declared variables match the template,
and computes a SHA-256 of the rendered body.

Every extraction and every eval run records prompt **name, version, and hash**.

## Consequences

**Good.** Prompt A/B is a real experiment: `sf eval ab guidance_tone` runs both
versions over identical ground truth. This measured v2 as +0.17 accuracy and
+0.13 calibration over v1. The hash catches the failure mode a version string
cannot: editing a file without bumping its version.

**Bad.** Rendering is deliberately dumb `{{ var }}` substitution — no
conditionals, no loops. A prompt that can branch is a program and needs its own
tests, so that capability is withheld on purpose.

**Also.** Strict rendering rejects both missing *and* unused variables, which has
caught renamed-variable bugs that would otherwise have silently produced a prompt
with an empty document in it.
