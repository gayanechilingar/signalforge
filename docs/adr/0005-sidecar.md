# 5. Python sidecar sandbox: honest about its limits

**Status:** accepted

## Context

The agent needs arithmetic. An LLM asked to compute a growth rate produces a
plausible-looking wrong number, so it gets a Python tool instead. But that means
executing model-authored code, and the agent's inputs include filing text, which
is attacker-influenceable in principle.

## Decision

Five layers: static AST rejection (import allowlist, no dunder access, no
`exec`/`eval`/`open`), a separate process, `setrlimit` caps on CPU / address space
/ file size, a wall-clock timeout enforced by the parent, and a scrubbed
environment with a temp CWD.

## Consequences

**Good.** Genuine defence against a confused agent and casual misuse. Timeouts,
memory bombs, and filesystem access are all blocked, and each failure returns as
an observation the agent can correct from rather than an exception.

**Bad, and stated plainly.** This is **not** a security boundary against a
determined adversary sharing the host. A CPython in-process sandbox is not
robust; there is a long history of escapes via attribute traversal.

**A lesson worth recording.** The first implementation nulled dangerous builtins
in the child — including `__import__`, `getattr`, `exec`, and `setattr`. That
broke CPython's own import machinery (importlib uses `getattr`;
`collections.namedtuple` uses `exec`), and it failed as an opaque
`'NoneType' object is not callable` that looked like a bug in the user's code. The
runtime disarm list is now deliberately small (`open`, `input`, `breakpoint`) and
the *static* AST pass is the real control. Layered defences must be checked
individually or a broken layer masquerades as a working one.

**When to replace.** Anything running genuinely untrusted code should move
execution into a container or gVisor. `sidecar.run_python` is one function with a
narrow signature, precisely so that swap is contained.
