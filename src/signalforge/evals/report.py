"""Rendering eval results for humans and for CI logs.

A bake-off table that reports only accuracy invites the wrong decision. The
comparison a person actually needs is accuracy *against* hallucination rate,
calibration, schema conformance, latency, and cost — because the cheapest model
that clears the quality bar is usually the right answer, and the most accurate
model is sometimes disqualified by an overconfidence problem that accuracy alone
hides.

So the table below always shows those columns together, and ``recommend`` states
a pick with its reasoning rather than leaving a grid of numbers for someone to
squint at.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from .harness import EvalRun
from .metrics import GateResult


def render_run(run: EvalRun, *, console: Console | None = None) -> None:
    console = console or Console()
    m = run.metrics

    console.print(
        f"\n[bold]{run.task}[/bold]  model=[cyan]{run.model}[/cyan]  "
        f"prompt=[magenta]{run.prompt_name}@{run.prompt_version}[/magenta] "
        f"([dim]{run.prompt_hash}[/dim])  n={m.get('n')}  "
        f"{run.duration_s}s  ${run.total_cost_usd:.4f}"
        + (f"  git={run.git_sha}" if run.git_sha else "")
    )

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in (
        "accuracy",
        "direction_accuracy",
        "macro_f1",
        "hallucination_rate",
        "mean_grounded_ratio",
        "schema_violation_rate",
        "repair_rate",
        "ece",
        "brier",
        "mean_confidence",
        "overconfidence",
        "latency_p50_ms",
        "latency_p95_ms",
        "cost_per_1k_usd",
    ):
        if m.get(key) is not None:
            table.add_row(key, _fmt(m[key]))
    console.print(table)

    if m.get("confusion"):
        console.print("[dim]confusion (expected -> predicted):[/dim]")
        for expected, preds in sorted(m["confusion"].items()):
            got = ", ".join(f"{k}:{v}" for k, v in sorted(preds.items()))
            console.print(f"  [dim]{expected:>10}[/dim] -> {got}")


def render_failures(run: EvalRun, *, console: Console | None = None, limit: int = 8) -> None:
    """Show the cases that failed.

    The aggregate metric says how much is wrong; only the individual failures say
    *what kind* of wrong, which is what actually drives the next prompt revision.
    """
    console = console or Console()
    failures = [o for o in run.outcomes if not o.correct]
    if not failures:
        console.print("[green]all cases correct[/green]")
        return

    console.print(f"\n[bold red]{len(failures)} failing case(s)[/bold red]")
    for o in failures[:limit]:
        exp = ", ".join(f"{k}={v}" for k, v in o.expected.items())
        if o.actual:
            got = ", ".join(f"{k}={o.actual.get(k)}" for k in o.expected)
        else:
            got = f"INVALID ({o.error})"
        flag = " [yellow](ungrounded)[/yellow]" if o.hallucinated else ""
        console.print(f"  [dim]{o.case_id}[/dim] expected: {exp}")
        console.print(f"  {'':>{len(o.case_id)}}  actual:   {got}{flag}")


def render_comparison(
    runs: list[EvalRun], *, console: Console | None = None, title: str = "comparison"
) -> None:
    """Side-by-side table across models or prompt versions."""
    console = console or Console()
    if not runs:
        return

    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("model")
    table.add_column("prompt")
    table.add_column("acc", justify="right")
    table.add_column("dir acc", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("halluc", justify="right")
    table.add_column("schema err", justify="right")
    table.add_column("repairs", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("$/1k", justify="right")

    best_acc = max((r.metrics.get("accuracy") or 0) for r in runs)
    for run in runs:
        m = run.metrics
        acc = m.get("accuracy") or 0
        table.add_row(
            f"[bold]{run.model}[/bold]" if acc == best_acc else run.model,
            f"{run.prompt_version}",
            _fmt(acc),
            _fmt(m.get("direction_accuracy")),
            _fmt(m.get("macro_f1")),
            _risk(m.get("hallucination_rate")),
            _risk(m.get("schema_violation_rate")),
            _fmt(m.get("mean_repairs")),
            _fmt(m.get("ece")),
            _fmt(m.get("latency_p95_ms")),
            _fmt(m.get("cost_per_1k_usd")),
        )
    console.print(table)


def recommend(
    runs: list[EvalRun],
    *,
    min_direction_accuracy: float = 0.75,
    max_hallucination_rate: float = 0.05,
    max_ece: float = 0.20,
) -> dict[str, Any]:
    """Pick a model for a task: cheapest that clears the quality bar.

    Cheapest-that-qualifies rather than most-accurate, because per-task model
    selection is a cost decision constrained by quality — if a 3B local model
    clears the bar, spending on a frontier model for that task buys nothing.
    Disqualification reasons are returned alongside, so a rejected model's problem
    is legible instead of implicit.
    """
    qualified: list[EvalRun] = []
    rejected: list[dict[str, Any]] = []

    for run in runs:
        m = run.metrics
        reasons = []
        if (m.get("direction_accuracy") or 0) < min_direction_accuracy:
            reasons.append(
                f"direction_accuracy {m.get('direction_accuracy')} < {min_direction_accuracy}"
            )
        if (m.get("hallucination_rate") or 0) > max_hallucination_rate:
            reasons.append(
                f"hallucination_rate {m.get('hallucination_rate')} > {max_hallucination_rate}"
            )
        if (m.get("schema_violation_rate") or 0) > 0.1:
            reasons.append(f"schema_violation_rate {m.get('schema_violation_rate')} > 0.1")
        if m.get("ece") is not None and m["ece"] > max_ece:
            # Overconfidence is disqualifying on its own: it makes the
            # confidence-weighted scoring in signals/score.py meaningless.
            reasons.append(f"ece {m['ece']} > {max_ece} (confidence not usable as a gate)")

        if reasons:
            rejected.append({"model": run.model, "reasons": reasons})
        else:
            qualified.append(run)

    if not qualified:
        return {
            "pick": None,
            "reason": "no model cleared the quality bar",
            "rejected": rejected,
        }

    pick = min(
        qualified,
        key=lambda r: (
            r.metrics.get("cost_per_1k_usd") or 0.0,
            r.metrics.get("latency_p95_ms") or 0.0,
        ),
    )
    return {
        "pick": pick.model,
        "prompt": f"{pick.prompt_name}@{pick.prompt_version}",
        "reason": (
            f"cheapest model clearing the bar: direction_accuracy="
            f"{pick.metrics.get('direction_accuracy')}, hallucination_rate="
            f"{pick.metrics.get('hallucination_rate')}, ece={pick.metrics.get('ece')}, "
            f"${pick.metrics.get('cost_per_1k_usd')}/1k"
        ),
        "qualified": [r.model for r in qualified],
        "rejected": rejected,
    }


def render_gate(result: GateResult, *, console: Console | None = None) -> None:
    console = console or Console()
    for line in result.lines:
        colour = "green" if line.endswith("PASS") else "red"
        console.print(f"  [{colour}]{line}[/{colour}]")
    verdict = (
        "[bold green]GATE PASSED[/bold green]"
        if result.passed
        else "[bold red]GATE FAILED[/bold red]"
    )
    console.print(verdict)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
    return str(value)


def _risk(value: Any) -> str:
    """Colour a metric where any non-zero value is a problem."""
    if value is None:
        return "-"
    text = f"{value:.3f}"
    return f"[green]{text}[/green]" if value == 0 else f"[red]{text}[/red]"
