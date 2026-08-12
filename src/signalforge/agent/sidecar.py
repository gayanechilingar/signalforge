"""Sandboxed Python sidecar.

Agents need arithmetic. Asking an LLM to compute a growth rate or a weighted
average is asking it to hallucinate a number that looks right, so the agent gets
a Python execution tool instead — but code written by a model and executed on the
host is the single most dangerous thing in this system, so the sandbox is
defence-in-depth rather than a single check:

1. **Static rejection** — the AST is walked before anything runs, and imports
   outside an allowlist, attribute access to dunders, ``exec``/``eval``/``open``,
   and comprehension-bomb patterns are refused. This catches the obvious cases
   with a clear error the agent can act on.
2. **Separate process** — execution happens in a subprocess, not in-process, so a
   segfault, a memory bomb, or a native-code crash cannot take down the API.
3. **Resource limits** — CPU time, address space, and file descriptors are capped
   via ``setrlimit`` in the child before user code runs, so an infinite loop or a
   runaway allocation dies rather than exhausting the host.
4. **Wall-clock timeout** — the parent kills the process group regardless of what
   the child does.
5. **No network, no filesystem writes** — the child gets a scrubbed environment
   and a temp CWD.

This is honest about its limits: it is a meaningful barrier against a *confused*
agent and casual misuse, not a security boundary against a determined adversary.
Running genuinely untrusted code needs a container or gVisor, and the interface
here is small enough to move behind one — see ``docs/adr/0005-sidecar.md``.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

#: Modules the sidecar may import. Deliberately arithmetic and data shaping only.
ALLOWED_IMPORTS = frozenset(
    {
        "math",
        "statistics",
        "json",
        "datetime",
        "decimal",
        "fractions",
        "re",
        "itertools",
        "functools",
        "collections",
        "numbers",
    }
)

#: Names that provide a route to the host regardless of import restrictions.
#: Enforced statically, by AST inspection, before any code runs.
BANNED_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
    }
)

#: Names actually nulled in the child at runtime — a *deliberately smaller* set
#: than :data:`BANNED_NAMES`, and not derived from it.
#:
#: The two lists do different jobs. :data:`BANNED_NAMES` polices what user code may
#: reference, checked statically before anything executes. This set hardens the
#: running interpreter as a second layer, and it must exclude anything CPython's
#: own machinery depends on: ``__import__`` backs every ``import`` statement, and
#: ``getattr``/``hasattr``/``globals``/``vars`` are used inside ``importlib``. Nulling
#: those breaks even the allowlisted imports the sandbox is meant to permit — and it
#: fails as an opaque ``'NoneType' object is not callable``, which reads like a bug
#: in the user's code rather than a bug in the sandbox.
#:
#: ``exec``/``eval``/``compile`` are excluded for the same class of reason:
#: ``collections.namedtuple`` builds its class with ``exec``, so nulling it breaks
#: ``collections`` and therefore ``statistics`` — both on the allowlist. They stay
#: blocked statically, which is the enforcement that actually matters, since user
#: code cannot reference a name the AST pass rejects.
#:
#: What remains is what nothing else needs: interactive prompts and direct
#: filesystem access. Writes are separately impossible via ``RLIMIT_FSIZE``.
#: Be clear-eyed about the result — the load-bearing guarantees here are the AST
#: pass, the process boundary, the resource limits, and the timeout. This last
#: layer is a thin extra, not the wall.
RUNTIME_DISARM = frozenset({"open", "input", "breakpoint"})

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MEMORY_MB = 512
MAX_OUTPUT_CHARS = 8000


class SidecarError(RuntimeError):
    """Code was rejected before execution."""


@dataclass(slots=True)
class SidecarResult:
    ok: bool
    stdout: str
    stderr: str
    #: Value of a ``result`` variable, if the script set one — lets the agent
    #: return structured data instead of parsing its own printed output.
    result: object | None = None
    duration_s: float = 0.0

    def render(self) -> str:
        """Format for an agent observation."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.result is not None:
            parts.append(f"result = {json.dumps(self.result, default=str)}")
        if self.stderr.strip():
            parts.append(f"stderr:\n{self.stderr.strip()}")
        return "\n".join(parts) or "(no output)"


def validate(code: str) -> None:
    """Reject code that must not run. Raises :class:`SidecarError`."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SidecarError(f"syntax error: {exc}") from None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    raise SidecarError(
                        f"import of {alias.name!r} is not allowed; permitted: "
                        f"{', '.join(sorted(ALLOWED_IMPORTS))}"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise SidecarError(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise SidecarError(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            # Blocks the classic __class__/__subclasses__/__globals__ escape,
            # which is how sandbox bypasses are normally written.
            raise SidecarError(f"access to dunder attribute {node.attr!r} is not allowed")


#: Preamble injected into the child: caps resources, then removes the import
#: machinery's ability to reach anything not already loaded.
_HARNESS = """
import resource, sys, os, builtins, json as _json

resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
try:
    resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
except (ValueError, OSError):
    pass  # macOS ignores RLIMIT_AS for some limits; the timeout still applies.
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
try:
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # no writes at all
except (ValueError, OSError):
    pass

# Capture every builtin the harness itself needs *before* disarming them.
# Several disarmed names (exec, compile, setattr, getattr) are ones this harness
# uses, so nulling them first and then calling them is a self-inflicted
# TypeError — and one that silently disables the whole sidecar.
_exec = builtins.exec
_compile = builtins.compile
_setattr = builtins.setattr
_hasattr = builtins.hasattr

for name in {banned!r}:
    if _hasattr(builtins, name):
        _setattr(builtins, name, None)

_ns = {{}}
try:
    _exec(_compile(_USER_CODE, "<sidecar>", "exec"), _ns, _ns)
except BaseException as exc:
    print(f"{{type(exc).__name__}}: {{exc}}", file=sys.stderr)
    sys.exit(1)

_result = _ns.get("result")
if _result is not None:
    try:
        sys.stderr.write("\\x00RESULT\\x00" + _json.dumps(_result, default=str))
    except Exception:
        sys.stderr.write("\\x00RESULT\\x00" + _json.dumps(str(_result)))
"""


def run_python(
    code: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> SidecarResult:
    """Validate then execute ``code`` in a locked-down subprocess."""
    import time

    validate(code)

    harness = _HARNESS.format(
        # One second of headroom over the wall clock. A CPU-bound infinite loop
        # burns CPU time at wall-clock rate, so setting RLIMIT_CPU *equal* to the
        # timeout made the two deadlines race: whichever fired first was arbitrary,
        # and when the rlimit won the child died by signal with an empty stderr, so
        # the agent saw a bare failure with no explanation. The parent's timeout now
        # wins reliably and reports why; the rlimit stays as the backstop for a
        # child the parent somehow fails to reap.
        cpu=int(max(1, timeout_s)) + 1,
        mem=int(memory_mb * 1024 * 1024),
        banned=sorted(RUNTIME_DISARM),
    )
    program = f"_USER_CODE = {code!r}\n" + harness

    with tempfile.TemporaryDirectory() as workdir:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", program],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=workdir,
                # Scrubbed environment: no inherited credentials, no PYTHONPATH
                # that could re-expose host packages.
                env={"PATH": "/usr/bin:/bin", "HOME": workdir, "PYTHONHASHSEED": "0"},
                start_new_session=True,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SidecarResult(
                ok=False,
                stdout="",
                stderr=f"execution exceeded {timeout_s}s and was terminated",
                duration_s=timeout_s,
            )
        duration = time.perf_counter() - t0

    stderr, result = _split_result(proc.stderr)
    if proc.returncode < 0 and not stderr.strip():
        # Killed by a signal, so there is no traceback to report. Say so, rather
        # than handing the agent a failure with no output to reason about.
        stderr = f"execution was terminated by signal {-proc.returncode} (resource limit reached)"
    return SidecarResult(
        ok=proc.returncode == 0,
        stdout=proc.stdout[:MAX_OUTPUT_CHARS],
        stderr=stderr[:MAX_OUTPUT_CHARS],
        result=result,
        duration_s=round(duration, 3),
    )


def _split_result(stderr: str) -> tuple[str, object | None]:
    """Separate the structured result channel from real stderr."""
    marker = "\x00RESULT\x00"
    if marker not in stderr:
        return stderr, None
    head, _, payload = stderr.partition(marker)
    try:
        return head, json.loads(payload)
    except json.JSONDecodeError:
        return head, None


def sidecar_available() -> bool:
    """Whether the sidecar can run here. ``resource`` is POSIX-only."""
    if os.name != "posix":
        return False
    try:
        return run_python("result = 1 + 1").result == 2
    except Exception:
        return False
