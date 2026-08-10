"""``sf`` — the command line surface.

Every stage of the pipeline is runnable independently, in order:

    sf doctor                     # check providers, warehouse, prompts
    sf ingest AAPL MSFT           # EDGAR -> warehouse
    sf index                      # embed chunks
    sf search "material weakness" # hybrid retrieval
    sf extract guidance_tone      # run a signal pipeline
    sf score                      # extractions -> signals -> alerts
    sf eval run guidance_tone     # grade against ground truth
    sf eval bakeoff guidance_tone # compare models
    sf eval ab guidance_tone      # compare prompt versions
    sf eval gate                  # CI regression gate
    sf review                     # human-in-the-loop queue
    sf agent "question"           # agentic research
    sf serve                      # API + dashboard

Separate commands rather than one orchestrator because each stage has a different
cost and failure profile: ingest is network-bound and cheap, extract is the
expensive one, and scoring is free and worth re-running often.
"""

from __future__ import annotations

import json
import logging

import typer
from rich.console import Console
from rich.table import Table

from . import db
from .llm.registry import get_registry
from .llm.router import Router
from .obs.tracing import Tracer
from .prompts.registry import get_prompts
from .settings import get_settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="SignalForge — SEC filings to investment signals.",
)
eval_app = typer.Typer(no_args_is_help=True, help="Evaluations, bake-offs, and CI gates.")
app.add_typer(eval_app, name="eval")

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else getattr(logging, get_settings().log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )


# --------------------------------------------------------------------- doctor
@app.command()
def doctor() -> None:
    """Check that providers, the warehouse, and prompts are all usable."""
    from .ingest.edgar import EdgarClient
    from .llm.anthropic import AnthropicClient
    from .llm.cache import ResponseCache
    from .llm.ollama import OllamaClient

    settings = get_settings()
    console.print(f"[bold]SignalForge[/bold]  data={settings.data_dir}")

    checks: list[tuple[str, bool, str]] = []

    ok, msg = OllamaClient().health()
    checks.append(("ollama", ok, msg))
    ok, msg = AnthropicClient().health()
    checks.append(("anthropic", ok, msg))

    try:
        path = db.init_db()
        counts = db.query(
            """
            SELECT (SELECT count(*) FROM companies)  AS companies,
                   (SELECT count(*) FROM filings)    AS filings,
                   (SELECT count(*) FROM chunks)     AS chunks,
                   (SELECT count(*) FROM embeddings) AS embeddings,
                   (SELECT count(*) FROM extractions) AS extractions,
                   (SELECT count(*) FROM signals)    AS signals
            """
        )[0]
        checks.append(
            ("warehouse", True, f"{path.name}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        )
    except Exception as exc:
        checks.append(("warehouse", False, str(exc)))

    try:
        prompts = get_prompts()
        checks.append(("prompts", True, f"{len(prompts.all())} loaded"))
    except Exception as exc:
        checks.append(("prompts", False, str(exc)))

    try:
        reg = get_registry()
        checks.append(("models.yaml", True, f"{len(reg.models)} models, {len(reg.chains)} chains"))
    except Exception as exc:
        checks.append(("models.yaml", False, str(exc)))

    stats = ResponseCache().stats()
    checks.append(("llm cache", True, f"{stats['entries']} entries, {stats['hits']} hits"))

    ok, msg = EdgarClient().health()
    checks.append(("edgar", ok, msg))

    for name, ok, msg in checks:
        mark = "[green]ok[/green]  " if ok else "[red]FAIL[/red]"
        console.print(f"  {mark} [bold]{name:12s}[/bold] {msg}")

    # Anthropic being unconfigured is expected, not a failure — the local path is
    # the default. Only hard failures should affect the exit code.
    hard = [n for n, ok, _ in checks if not ok and n != "anthropic"]
    if hard:
        raise typer.Exit(1)


# --------------------------------------------------------------------- ingest
@app.command()
def ingest(
    tickers: list[str] = typer.Argument(..., help="Tickers or CIKs."),
    limit: int = typer.Option(12, help="Max filings per company."),
    forms: str = typer.Option("10-K,10-Q,8-K", help="Comma-separated forms."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fetch filings from EDGAR and load them into the warehouse."""
    from .ingest.store import ingest_company

    _setup_logging(verbose)
    form_tuple = tuple(f.strip() for f in forms.split(",") if f.strip())
    tracer = Tracer()

    for ticker in tickers:
        report = ingest_company(ticker, forms=form_tuple, limit=limit, tracer=tracer)
        console.print(f"[bold]{ticker}[/bold] {report.as_dict()}")
        for note in report.skipped:
            console.print(f"  [yellow]skipped[/yellow] {note}")
        for acc in report.flagged:
            console.print(f"  [yellow]flagged for review[/yellow] {acc}")
    tracer.flush()


@app.command()
def index(
    model: str | None = typer.Option(None, help="Embedding model name."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Embed any chunks that do not yet have a vector."""
    from .retrieval.index import build_index, index_stats

    _setup_logging(verbose)
    tracer = Tracer()
    n = build_index(model=model, tracer=tracer)
    tracer.flush()
    console.print(f"embedded [bold]{n}[/bold] chunks")
    console.print(index_stats())


@app.command()
def search(
    query: str,
    k: int = typer.Option(8),
    mode: str = typer.Option("hybrid", help="hybrid | vector | keyword"),
    cik: str | None = typer.Option(None),
    slug: str | None = typer.Option(None, help="Section slug, e.g. risk_factors."),
) -> None:
    """Search the filing corpus."""
    from .retrieval.index import search as do_search

    hits = do_search(query, k=k, mode=mode, cik=cik, slug=slug, tracer=Tracer(enabled=False))
    if not hits:
        console.print("[yellow]no results[/yellow]")
        return
    for h in hits:
        console.print(
            f"[cyan]{h.score:.4f}[/cyan] [dim]{h.accession}[/dim] "
            f"[magenta]{h.slug}[/magenta] {'/'.join(h.sources)}"
        )
        console.print(f"   {h.text[:260].strip()}\n")


# -------------------------------------------------------------------- extract
@app.command()
def extract(
    task: str = typer.Argument(..., help="guidance_tone | risk_delta | event_class"),
    chain: str = typer.Option("extract_default", help="Model chain from models.yaml."),
    prompt_version: str | None = typer.Option(None, help="Pin a prompt version."),
    cik: str | None = typer.Option(None),
    limit: int | None = typer.Option(None),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run an extraction task over the warehouse."""
    from .extract.tasks import iter_task

    _setup_logging(verbose)
    tracer = Tracer()
    router = Router(tracer=tracer)

    n = valid = repairs = 0
    for result in iter_task(
        task,
        router=router,
        prompt_version=prompt_version,
        chain=chain,
        cik=cik,
        limit=limit,
        tracer=tracer,
    ):
        n += 1
        valid += result.valid
        repairs += result.repair_attempts
        mark = "[green]ok[/green]" if result.valid else "[red]invalid[/red]"
        console.print(
            f"  {mark} {result.accession} grounded={result.grounded_ratio:.2f} "
            f"repairs={result.repair_attempts}"
            + (f" [red]{result.error}[/red]" if result.error else "")
        )
    tracer.flush()
    console.print(f"\n[bold]{task}[/bold]: {valid}/{n} valid, {repairs} repairs, {router.stats()}")


@app.command()
def score(
    task: str | None = typer.Option(None),
    cik: str | None = typer.Option(None),
) -> None:
    """Turn stored extractions into signals and alerts."""
    from .signals.alerts import evaluate, persist_alerts
    from .signals.score import persist_signals, score_stored_extractions

    signals = score_stored_extractions(task=task, cik=cik)
    persist_signals(signals)
    alerts = evaluate(signals)
    persist_alerts(alerts)

    table = Table(title=f"{len(signals)} signals", box=None, header_style="bold")
    for col in ("ticker", "signal", "score", "conf", "direction", "as of"):
        table.add_column(col)
    for s in sorted(signals, key=lambda s: s.score)[:20]:
        colour = "red" if s.score < -0.1 else "green" if s.score > 0.1 else "white"
        table.add_row(
            s.ticker or s.cik,
            s.name,
            f"[{colour}]{s.score:+.3f}[/{colour}]",
            f"{s.confidence:.2f}",
            s.direction,
            str(s.as_of),
        )
    console.print(table)

    if alerts:
        console.print(f"\n[bold]{len(alerts)} alerts[/bold]")
        for a in alerts:
            colour = {"critical": "red", "warn": "yellow", "info": "dim"}[a.severity]
            console.print(f"  [{colour}][{a.severity}][/{colour}] {a.headline}")


# ----------------------------------------------------------------- evaluation
@eval_app.command("run")
def eval_run(
    task: str,
    chain: str = typer.Option("extract_default"),
    prompt_version: str | None = typer.Option(None),
    limit: int | None = typer.Option(None),
    show_failures: bool = typer.Option(True, "--failures/--no-failures"),
) -> None:
    """Grade a task against its ground-truth dataset."""
    from .evals.harness import gate, run_eval
    from .evals.report import render_failures, render_gate, render_run

    tracer = Tracer(enabled=False)
    run = run_eval(
        task,
        chain=chain,
        prompt_version=prompt_version,
        router=Router(tracer=tracer),
        limit=limit,
        tracer=tracer,
    )
    render_run(run, console=console)
    if show_failures:
        render_failures(run, console=console)
    render_gate(gate(run), console=console)


@eval_app.command("bakeoff")
def eval_bakeoff(
    task: str,
    models: str = typer.Option("llama32-3b,llama31-8b", help="Comma-separated model names."),
    prompt_version: str | None = typer.Option(None),
    limit: int | None = typer.Option(None),
) -> None:
    """Compare models on identical cases with an identical prompt."""
    from .evals.harness import bakeoff
    from .evals.report import recommend, render_comparison

    tracer = Tracer(enabled=False)
    names = [m.strip() for m in models.split(",") if m.strip()]
    runs = bakeoff(
        task,
        names,
        prompt_version=prompt_version,
        router=Router(tracer=tracer),
        limit=limit,
        tracer=tracer,
    )
    render_comparison(runs, console=console, title=f"{task}: model bake-off")

    rec = recommend(runs)
    if rec["pick"]:
        console.print(f"\n[bold green]pick:[/bold green] {rec['pick']}  [dim]{rec['reason']}[/dim]")
    else:
        console.print(f"\n[bold red]no pick:[/bold red] {rec['reason']}")
    for r in rec["rejected"]:
        console.print(f"  [red]rejected[/red] {r['model']}: {'; '.join(r['reasons'])}")


@eval_app.command("ab")
def eval_ab(
    task: str,
    versions: str | None = typer.Option(None, help="Comma-separated, default: all."),
    chain: str = typer.Option("extract_default"),
    limit: int | None = typer.Option(None),
) -> None:
    """A/B prompt versions on one model."""
    from .evals.harness import prompt_ab
    from .evals.report import render_comparison

    prompts = get_prompts()
    vs = [v.strip() for v in versions.split(",")] if versions else prompts.versions(task)
    if len(vs) < 2:
        console.print(f"[yellow]only one version of {task} exists: {vs}[/yellow]")
        raise typer.Exit(0)

    tracer = Tracer(enabled=False)
    runs = prompt_ab(
        task, vs, chain=chain, router=Router(tracer=tracer), limit=limit, tracer=tracer
    )
    render_comparison(runs, console=console, title=f"{task}: prompt A/B on {chain}")


@eval_app.command("gate")
def eval_gate(
    tasks: str = typer.Option("guidance_tone,event_class,risk_delta"),
    chain: str = typer.Option("ci", help="Use 'ci' for the hermetic stub provider."),
) -> None:
    """Run the regression gate. Exits non-zero on failure — this is the CI hook."""
    from .evals.harness import gate, run_eval
    from .evals.report import render_gate, render_run

    tracer = Tracer(enabled=False)
    passed = True
    for task in (t.strip() for t in tasks.split(",") if t.strip()):
        run = run_eval(
            task,
            chain=chain,
            router=Router(tracer=tracer),
            tracer=tracer,
            suite="gate",
            persist=False,
        )
        render_run(run, console=console)
        result = gate(run)
        render_gate(result, console=console)
        passed = passed and result.passed

    if not passed:
        console.print("\n[bold red]regression gate failed[/bold red]")
        raise typer.Exit(1)
    console.print("\n[bold green]all gates passed[/bold green]")


# --------------------------------------------------------------------- review
@app.command()
def review(
    limit: int = typer.Option(10),
    task: str | None = typer.Option(None),
) -> None:
    """Show the human-in-the-loop review queue, worst first."""
    rows = db.query(
        f"""
        SELECT review_id, task, reason, priority, status, proposed
        FROM review_queue
        WHERE status = 'open' {"AND task = ?" if task else ""}
        ORDER BY priority DESC, created_at ASC
        LIMIT {int(limit)}
        """,
        [task] if task else [],
    )
    if not rows:
        console.print("[green]review queue is empty[/green]")
        return
    for r in rows:
        console.print(
            f"[bold]{r['reason']}[/bold] [dim]p{r['priority']}[/dim] "
            f"{r['task']} [dim]{r['review_id']}[/dim]"
        )
        proposed = json.loads(r["proposed"]) if isinstance(r["proposed"], str) else r["proposed"]
        for key in ("ungrounded_quotes", "error", "parse_issues"):
            if proposed and proposed.get(key):
                console.print(f"   {key}: {proposed[key]}")


@app.command()
def prompts() -> None:
    """List prompts with their versions and content hashes."""
    table = Table(box=None, header_style="bold")
    for col in ("name", "version", "hash", "task"):
        table.add_column(col)
    for row in get_prompts().manifest():
        table.add_row(row["name"], row["version"], row["hash"], row["task"])
    console.print(table)


@app.command()
def models() -> None:
    """List registered models with pricing and capabilities."""
    reg = get_registry()
    table = Table(box=None, header_style="bold")
    for col in ("name", "provider", "ctx", "$/Mtok in", "$/Mtok out", "tier", "sampling"):
        table.add_column(col)
    for spec in sorted(reg.models.values(), key=lambda s: (s.provider, s.tier)):
        table.add_row(
            spec.name,
            spec.provider,
            f"{spec.context_tokens:,}",
            f"{spec.usd_per_mtok_in:g}",
            f"{spec.usd_per_mtok_out:g}",
            str(spec.tier),
            "yes" if spec.supports_sampling else "no",
        )
    console.print(table)
    console.print("\n[bold]chains[/bold]")
    for name, chain in reg.chains.items():
        console.print(f"  {name}: {' -> '.join(chain)}")


@app.command()
def agent(
    question: str,
    chain: str = typer.Option("agent"),
    max_steps: int = typer.Option(8),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ask the research agent a question about the corpus."""
    from .agent.loop import run_agent

    _setup_logging(verbose)
    tracer = Tracer()
    result = run_agent(
        question, chain=chain, max_steps=max_steps, router=Router(tracer=tracer), tracer=tracer
    )
    tracer.flush()

    for step in result.steps:
        console.print(f"[dim]step {step.n}[/dim] [cyan]{step.tool}[/cyan]({step.args_summary})")
        if verbose:
            console.print(f"   [dim]{step.observation[:400]}[/dim]")
    console.print(f"\n{result.answer}\n")
    console.print(f"[dim]{result.stats()}[/dim]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False),
) -> None:
    """Run the API and dashboard."""
    import uvicorn

    uvicorn.run("signalforge.api.main:app", host=host, port=port, reload=reload)


@app.command("init-db")
def init_db_cmd() -> None:
    """Create the warehouse and apply the schema."""
    console.print(f"warehouse ready at {db.init_db()}")


if __name__ == "__main__":
    app()
