"""Eval harness — grade a task against ground truth, and gate CI on the result.

The distinguishing property of this harness is that a *run is fully attributed*:
every stored run records the model, the prompt name, version, and content hash, the
case count, the git SHA, and the full metric block. That makes three otherwise
impossible things routine:

* **Prompt A/B.** Run ``guidance_tone`` at ``v1`` and ``v2`` over the same cases
  and compare directly.
* **Model bake-off.** Same cases, same prompt, N models — the comparison that
  answers "which model for this task", including whether a 3B local model is
  good enough to avoid paying for a frontier one.
* **Regression gates.** A run's metrics are checked against declared thresholds,
  and CI fails on a drop. Without this, a prompt "improvement" that quietly
  raises the hallucination rate ships.

Grading is exact-match on the task's labelled fields, plus a separate directional
score, because for several tasks the sign carries most of the value and a
severity confusion is a much smaller error than a direction flip.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..db import connect, upsert
from ..extract.runner import extract
from ..extract.schemas import SCHEMAS
from ..llm.router import Router
from ..obs.tracing import Tracer
from ..prompts.registry import PromptRegistry, get_prompts
from ..settings import get_settings
from .metrics import CaseOutcome, GateResult, RegressionCheck, check_regressions, summarise

log = logging.getLogger(__name__)

#: Which field carries the primary label, per task. Used for macro-F1 and the
#: confusion matrix.
PRIMARY_FIELD = {
    "guidance_tone": "direction",
    "risk_delta": "direction",
    "event_class": "direction",
}

#: Regression thresholds enforced in CI.
#:
#: Calibrated against the stub provider, which is what CI runs — they gate the
#: *pipeline* (schema conformance, grounding, plumbing), not model quality, since
#: CI has no GPU and no API key. Real-model thresholds are asserted by the
#: bake-off report, which a human reads.
DEFAULT_GATES: dict[str, list[RegressionCheck]] = {
    "guidance_tone": [
        RegressionCheck("schema_violation_rate", "max", 0.0),
        RegressionCheck("hallucination_rate", "max", 0.0),
        RegressionCheck("direction_accuracy", "min", 0.40),
    ],
    "event_class": [
        RegressionCheck("schema_violation_rate", "max", 0.0),
        RegressionCheck("hallucination_rate", "max", 0.0),
    ],
    "risk_delta": [
        RegressionCheck("schema_violation_rate", "max", 0.0),
        RegressionCheck("hallucination_rate", "max", 0.0),
    ],
}


@dataclass(slots=True)
class Case:
    case_id: str
    task: str
    label: dict[str, Any]
    variables: dict[str, Any]
    notes: str = ""

    @property
    def source_text(self) -> str:
        """The text evidence must be grounded in.

        For ``risk_delta`` this is the *current* filing only — the prompt forbids
        quoting the prior period, so grading grounding against both would let a
        prompt violation pass.
        """
        for key in ("section_text", "current_text"):
            if key in self.variables:
                return str(self.variables[key])
        return ""


def load_cases(task: str, *, path: Path | None = None) -> list[Case]:
    path = path or get_settings().eval_dir / "datasets" / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no ground-truth dataset for task {task!r} at {path}")

    cases: list[Case] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i} is not valid JSON: {exc}") from None
        cases.append(
            Case(
                case_id=raw["case_id"],
                task=raw.get("task", task),
                label=raw["label"],
                variables=raw["variables"],
                notes=raw.get("notes", ""),
            )
        )
    if not cases:
        raise ValueError(f"dataset {path} is empty")
    return cases


@dataclass
class EvalRun:
    run_id: str
    suite: str
    task: str
    model: str
    provider: str
    prompt_name: str
    prompt_version: str
    prompt_hash: str
    metrics: dict[str, Any]
    outcomes: list[CaseOutcome]
    duration_s: float
    started_at: datetime
    git_sha: str = ""
    notes: str = ""

    @property
    def total_cost_usd(self) -> float:
        return sum(o.cost_usd for o in self.outcomes)


def run_eval(
    task: str,
    *,
    chain: str | list[str] = "extract_default",
    prompt_version: str | None = None,
    router: Router | None = None,
    prompts: PromptRegistry | None = None,
    cases: list[Case] | None = None,
    limit: int | None = None,
    suite: str = "default",
    tracer: Tracer | None = None,
    persist: bool = True,
    notes: str = "",
) -> EvalRun:
    """Grade one (task, prompt version, model chain) against ground truth."""
    if task not in SCHEMAS:
        raise KeyError(f"unknown task {task!r}; known: {sorted(SCHEMAS)}")

    router = router or Router()
    prompts = prompts or get_prompts()
    tracer = tracer or Tracer(enabled=False)
    cases = cases or load_cases(task)
    if limit:
        cases = cases[:limit]

    prompt = prompts.get(task, prompt_version)
    started = datetime.now(UTC)
    t0 = time.perf_counter()

    outcomes: list[CaseOutcome] = []
    for case in cases:
        result = extract(
            task,
            source_text=case.source_text,
            variables=case.variables,
            # Namespaced so eval extractions never collide with, or pollute, the
            # real corpus in the extractions table.
            accession=f"eval:{case.case_id}",
            cik="eval",
            section_id=None,
            router=router,
            prompts=prompts,
            prompt_version=prompt_version,
            chain=chain,
            tracer=tracer,
            persist=False,
        )
        outcomes.append(_grade(case, result))

    duration = time.perf_counter() - t0
    metrics = summarise(outcomes, primary_field=PRIMARY_FIELD.get(task, "direction"))

    run = EvalRun(
        run_id=uuid.uuid4().hex[:16],
        suite=suite,
        task=task,
        model=_served_by(outcomes, "model") or _chain_label(chain),
        provider=_served_by(outcomes, "provider"),
        prompt_name=prompt.name,
        prompt_version=prompt.version,
        prompt_hash=prompt.hash,
        metrics=metrics,
        outcomes=outcomes,
        duration_s=round(duration, 2),
        started_at=started,
        git_sha=_git_sha(),
        notes=notes,
    )
    if persist:
        persist_run(run)
    return run


def _grade(case: Case, result: Any) -> CaseOutcome:
    payload = result.payload.model_dump(mode="json") if result.payload else None

    # Exact match across every labelled field. Partial credit is deliberately
    # withheld here and reported separately as direction_accuracy, so that a
    # single number cannot hide a severity-vs-direction confusion.
    correct = bool(payload) and all(str(payload.get(k)) == str(v) for k, v in case.label.items())
    direction_correct = bool(payload) and (
        "direction" not in case.label
        or str(payload.get("direction")) == str(case.label["direction"])
    )

    return CaseOutcome(
        case_id=case.case_id,
        expected=case.label,
        actual=payload,
        correct=correct,
        direction_correct=direction_correct,
        confidence=result.payload.confidence if result.payload else None,
        grounded_ratio=result.grounded_ratio,
        hallucinated=bool(result.grounding and result.grounding.hallucinated),
        valid=result.valid,
        repair_attempts=result.repair_attempts,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        error=result.error,
        model=result.model,
        provider=result.provider,
    )


def persist_run(run: EvalRun) -> None:
    with connect() as con:
        upsert(
            con,
            "eval_runs",
            [
                {
                    "run_id": run.run_id,
                    "suite": run.suite,
                    "task": run.task,
                    "model": run.model,
                    "provider": run.provider,
                    "prompt_name": run.prompt_name,
                    "prompt_version": run.prompt_version,
                    "prompt_hash": run.prompt_hash,
                    "n_cases": len(run.outcomes),
                    "metrics": run.metrics,
                    "git_sha": run.git_sha,
                    "started_at": run.started_at,
                    "duration_s": run.duration_s,
                    "total_cost_usd": run.total_cost_usd,
                    "notes": run.notes,
                }
            ],
            key="run_id",
        )
        upsert(
            con,
            "eval_results",
            [
                {
                    "result_id": f"{run.run_id}:{o.case_id}",
                    "run_id": run.run_id,
                    "case_id": o.case_id,
                    "expected": o.expected,
                    "actual": o.actual,
                    "correct": o.correct,
                    "scores": {
                        "direction_correct": o.direction_correct,
                        "confidence": o.confidence,
                        "grounded_ratio": o.grounded_ratio,
                        "repair_attempts": o.repair_attempts,
                    },
                    "latency_ms": o.latency_ms,
                    "cost_usd": o.cost_usd,
                    "error": o.error,
                }
                for o in run.outcomes
            ],
            key="result_id",
        )


def gate(run: EvalRun, checks: list[RegressionCheck] | None = None) -> GateResult:
    """Check a run against its regression thresholds."""
    return check_regressions(run.metrics, checks or DEFAULT_GATES.get(run.task, []))


def bakeoff(
    task: str,
    models: list[str],
    *,
    prompt_version: str | None = None,
    router: Router | None = None,
    prompts: PromptRegistry | None = None,
    limit: int | None = None,
    tracer: Tracer | None = None,
) -> list[EvalRun]:
    """Run the same cases and prompt against several models.

    Identical cases and identical prompt hash across runs is what makes the
    comparison valid — the whole point is that only the model varies.
    """
    cases = load_cases(task)
    if limit:
        cases = cases[:limit]

    runs = []
    for model in models:
        log.info("bakeoff task=%s model=%s", task, model)
        runs.append(
            run_eval(
                task,
                chain=[model],
                prompt_version=prompt_version,
                router=router,
                prompts=prompts,
                cases=cases,
                suite="bakeoff",
                tracer=tracer,
                notes=f"bakeoff:{task}",
            )
        )
    return runs


def prompt_ab(
    task: str,
    versions: list[str],
    *,
    chain: str | list[str] = "extract_default",
    router: Router | None = None,
    prompts: PromptRegistry | None = None,
    limit: int | None = None,
    tracer: Tracer | None = None,
) -> list[EvalRun]:
    """Compare prompt versions on one model over identical cases."""
    cases = load_cases(task)
    if limit:
        cases = cases[:limit]
    return [
        run_eval(
            task,
            chain=chain,
            prompt_version=v,
            router=router,
            prompts=prompts,
            cases=cases,
            suite="prompt_ab",
            tracer=tracer,
            notes=f"prompt_ab:{task}",
        )
        for v in versions
    ]


def _served_by(outcomes: list[CaseOutcome], attr: str) -> str:
    """Which model(s) actually served the run.

    Usually one, but a fallback chain can spread a run across models. Reporting
    all of them keeps a run from being mislabelled with a model that only served
    part of it — the sort of silent mislabel that makes a bake-off table lie.
    """
    names = sorted({getattr(o, attr) for o in outcomes if getattr(o, attr)})
    return "+".join(names)


def _chain_label(chain: str | list[str]) -> str:
    return chain if isinstance(chain, str) else "+".join(chain)


def _git_sha() -> str:
    """Record the commit a run was produced at, so results stay traceable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[3],
        ).stdout.strip()
    except Exception:
        return ""
