"""The agentic research loop.

An agent loop is easy to write and hard to make trustworthy. The parts that
matter here are the bounds, not the reasoning:

**Every run is bounded three ways** — step count, wall clock, and cost. Any one of
them can terminate the run, and hitting a bound produces a *partial answer with a
stated reason*, never a silent truncation or an infinite spin. An agent that can
loop forever is an agent that will, usually at 3am.

**Tool calls are parsed, not trusted.** Local models emit tool calls in whatever
shape they feel like — the native ``tool_calls`` field, a JSON object in prose, a
fenced code block. The loop accepts all three, because insisting on one means the
loop simply fails on most open models. Unknown tool names and bad arguments come
back as observations the model can correct from, rather than exceptions.

**Repeated identical calls are short-circuited.** The most common local-model
failure mode is calling the same tool with the same arguments forever. The loop
detects that and tells the model, which is far more effective than raising the
step limit.

**Every step is traced** with its tool, duration, and result size, so a run can be
audited after the fact — which is the difference between an agent you can put in
front of the firm and a demo.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import CostCapExceeded, LLMError, Message, parse_json_loose
from ..llm.router import Router
from ..obs.tracing import Tracer, default_tracer
from .skills import load_skills
from .tools import Tool, ToolResult, build_tools

log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 8
DEFAULT_TIME_BUDGET_S = 300.0

SYSTEM_PROMPT = """You are Aion, a research analyst assistant with direct access \
to a warehouse of SEC filings and computed investment signals.

You answer questions by using tools, not from memory. You have no knowledge of \
any company's filings except what the tools return.

How to work:
- Start by orienting yourself if you don't know what data exists.
- Ground every factual claim in a tool result. Quote filing text where it matters \
and name the accession number it came from.
- Compute numbers with the python tool. Never estimate arithmetic yourself.
- If the data cannot answer the question, say so plainly and say what is missing. \
An honest "the corpus has no 2024 filings for this company" is far more useful \
than a confident guess.
- Do not repeat a tool call you have already made with the same arguments.

To call a tool, respond with a JSON object and nothing else:
{"tool": "<name>", "args": {...}}

When you have enough information, respond with a JSON object and nothing else:
{"answer": "<your complete answer, citing accession numbers>"}

Available tools:
%s

%s"""


@dataclass(slots=True)
class Step:
    n: int
    tool: str
    args: dict[str, Any]
    observation: str
    duration_s: float
    ok: bool = True

    @property
    def args_summary(self) -> str:
        parts = []
        for k, v in self.args.items():
            text = str(v)
            parts.append(f"{k}={text[:60] + '...' if len(text) > 60 else text}")
        return ", ".join(parts)


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = "answered"
    duration_s: float = 0.0
    cost_usd: float = 0.0
    llm_calls: int = 0

    def stats(self) -> dict[str, Any]:
        return {
            "steps": len(self.steps),
            "llm_calls": self.llm_calls,
            "stop_reason": self.stop_reason,
            "duration_s": round(self.duration_s, 2),
            "cost_usd": round(self.cost_usd, 6),
        }


def run_agent(
    question: str,
    *,
    router: Router | None = None,
    chain: str | list[str] = "agent",
    max_steps: int = DEFAULT_MAX_STEPS,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    tools: dict[str, Tool] | None = None,
    tracer: Tracer | None = None,
    skills: bool = True,
) -> AgentResult:
    """Answer ``question`` using tools, within bounds."""
    router = router or Router()
    tracer = tracer or default_tracer
    tools = tools or build_tools(router)

    tool_docs = "\n".join(
        f"- {t.name}({', '.join((t.parameters.get('properties') or {}).keys())}): {t.description}"
        for t in tools.values()
    )
    skill_docs = load_skills() if skills else ""
    system = SYSTEM_PROMPT % (tool_docs, skill_docs)

    messages = [
        Message(role="system", content=system),
        Message(role="user", content=question),
    ]

    result = AgentResult(question=question, answer="")
    seen_calls: set[str] = set()
    started = time.perf_counter()

    with tracer.span("agent.run", kind="agent", question=question[:200]) as run_span:
        for step_n in range(1, max_steps + 1):
            elapsed = time.perf_counter() - started
            if elapsed > time_budget_s:
                result.stop_reason = "time_budget_exhausted"
                break

            try:
                resp = router.complete(
                    messages,
                    chain=chain,
                    max_tokens=1024,
                    json_schema={"type": "object"},
                    span_name="agent.think",
                    step=step_n,
                )
            except CostCapExceeded as exc:
                result.stop_reason = "cost_cap_exceeded"
                result.answer = f"Stopped before completing: {exc}"
                break
            except LLMError as exc:
                result.stop_reason = "llm_error"
                result.answer = f"The model was unreachable: {exc}"
                break

            result.llm_calls += 1
            result.cost_usd += resp.cost_usd
            messages.append(Message(role="assistant", content=resp.text))

            decision = _parse_decision(resp)
            if decision is None:
                # Unparseable output is a correctable mistake, not a fatal one.
                messages.append(
                    Message(
                        role="user",
                        content=(
                            "Your response was not a valid JSON object. Reply with "
                            'exactly {"tool": "<name>", "args": {...}} or '
                            '{"answer": "..."} and nothing else.'
                        ),
                    )
                )
                continue

            if "answer" in decision:
                result.answer = str(decision["answer"]).strip()
                result.stop_reason = "answered"
                break

            name = str(decision.get("tool", "")).strip()
            args = decision.get("args") or {}
            if not isinstance(args, dict):
                args = {}

            step = _dispatch(name, args, tools, seen_calls, step_n, tracer)
            result.steps.append(step)
            messages.append(
                Message(
                    role="user",
                    content=f"Observation from {name}:\n{step.observation}",
                )
            )
        else:
            result.stop_reason = "max_steps_reached"

        result.duration_s = time.perf_counter() - started

        # A bounded run must still say something useful. Synthesising from the
        # observations gathered is better than returning nothing, and stating the
        # bound is what keeps it honest.
        if not result.answer:
            result.answer = _partial_answer(result, router, chain, messages, tracer)

        run_span.set(**result.stats())

    return result


def _dispatch(
    name: str,
    args: dict[str, Any],
    tools: dict[str, Tool],
    seen_calls: set[str],
    step_n: int,
    tracer: Tracer,
) -> Step:
    t0 = time.perf_counter()

    tool = tools.get(name)
    if tool is None:
        return Step(
            n=step_n,
            tool=name,
            args=args,
            ok=False,
            observation=(
                f"There is no tool named {name!r}. Available tools: {', '.join(sorted(tools))}."
            ),
            duration_s=time.perf_counter() - t0,
        )

    # Identical repeated calls are the dominant local-model loop failure; naming
    # it explicitly breaks the cycle far more reliably than a bigger step budget.
    fingerprint = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
    if fingerprint in seen_calls:
        return Step(
            n=step_n,
            tool=name,
            args=args,
            ok=False,
            observation=(
                "You already made this exact call and received its result above. "
                "Either use a different tool or different arguments, or give your "
                "final answer."
            ),
            duration_s=time.perf_counter() - t0,
        )
    seen_calls.add(fingerprint)

    with tracer.span(f"tool.{name}", kind="tool", **{"args": _safe(args)}) as span:
        try:
            result: ToolResult = tool.fn(**args)
        except TypeError as exc:
            # Wrong arguments: return the signature so the model can fix it.
            result = ToolResult(
                text=(
                    f"Invalid arguments for {name}: {exc}. Expected parameters: "
                    f"{json.dumps(tool.parameters.get('properties') or {})}"
                ),
                ok=False,
            )
        except Exception as exc:
            log.exception("tool %s failed", name)
            result = ToolResult(text=f"Tool {name} failed: {type(exc).__name__}: {exc}", ok=False)

        observation = result.truncated()
        span.set(ok=result.ok, chars=len(observation))

    return Step(
        n=step_n,
        tool=name,
        args=args,
        observation=observation,
        duration_s=time.perf_counter() - t0,
        ok=result.ok,
    )


def _parse_decision(resp: Any) -> dict[str, Any] | None:
    """Extract a tool call or answer from a model response.

    Accepts the provider's native ``tool_calls``, a bare JSON object, or JSON
    embedded in prose. Being permissive here is not sloppiness — insisting on one
    format means the loop fails outright on most open models, and the strictness
    buys nothing because the result is validated on dispatch anyway.
    """
    if getattr(resp, "tool_calls", None):
        call = resp.tool_calls[0]
        return {"tool": call.get("name"), "args": call.get("args") or {}}

    text = (resp.text or "").strip()
    if not text:
        return None

    try:
        payload = parse_json_loose(text)
    except ValueError:
        # Last resort: a model that narrated its intent in prose but included a
        # recognisable call somewhere in it.
        m = re.search(r'"tool"\s*:\s*"(\w+)"', text)
        return {"tool": m.group(1), "args": {}} if m else None

    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return None

    if "answer" in payload or "tool" in payload:
        return payload
    # Some models wrap the call: {"function": {"name": ..., "arguments": {...}}}
    fn = payload.get("function") or payload.get("tool_call")
    if isinstance(fn, dict) and fn.get("name"):
        args = fn.get("arguments") or fn.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return {"tool": fn["name"], "args": args}
    return None


def _partial_answer(
    result: AgentResult,
    router: Router,
    chain: str | list[str],
    messages: list[Message],
    tracer: Tracer,
) -> str:
    """Synthesise the best available answer after hitting a bound."""
    if not result.steps:
        return (
            f"I could not answer this within the configured bounds "
            f"({result.stop_reason}) and gathered no usable data."
        )
    try:
        resp = router.complete(
            messages
            + [
                Message(
                    role="user",
                    content=(
                        "You have run out of budget for further tool calls. Using "
                        "only the observations above, give your best answer now as "
                        "plain prose. State explicitly what you could not verify."
                    ),
                )
            ],
            chain=chain,
            max_tokens=800,
            span_name="agent.synthesise",
        )
        result.llm_calls += 1
        result.cost_usd += resp.cost_usd
        answer = (resp.text or "").strip()
    except Exception:
        answer = ""

    prefix = f"[stopped early: {result.stop_reason}] "
    if answer:
        return prefix + answer
    tools_used = ", ".join(sorted({s.tool for s in result.steps}))
    return prefix + f"Gathered data using {tools_used} but could not synthesise an answer."


def _safe(args: dict[str, Any]) -> str:
    return json.dumps(args, default=str)[:500]
