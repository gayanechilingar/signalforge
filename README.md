# SignalForge

LLM extraction pipelines that turn SEC filings into investment signals — with
evaluations, model selection, an agentic research layer, and the LLMOps
scaffolding to keep all of it honest.

Runs end to end on a laptop with **no API key**: the default provider is local
open-source models via Ollama. Adding `SF_ANTHROPIC_API_KEY` routes the hard tasks
to Claude without touching pipeline code.

```bash
git clone <repo> && cd signalforge
make install
export SF_SEC_USER_AGENT="SignalForge/0.1 (research; you@example.com)"   # SEC requires this
ollama pull llama3.1 && ollama pull nomic-embed-text
make demo          # ingest → embed → extract → score, on AAPL and MSFT
make serve         # dashboard + API on :8000
```

---

## What it does

Filings are messy, long, and mostly boilerplate. The pipeline turns them into
typed, attributed, verifiable signals:

```
EDGAR ──▶ parse ──▶ chunk ──▶ embed ──▶ hybrid retrieval
             │                              │
             ▼                              ▼
      section-aware text            versioned prompt  ──▶ model router
                                           │              (chain, fallback,
                                           ▼               cache, cost cap)
                                    typed extraction
                                    + validation + repair loop
                                           │
                                           ▼
                                  citation grounding check
                                           │
                     ┌─────────────────────┼──────────────────────┐
                     ▼                     ▼                      ▼
                signal score          review queue           eval harness
                     │                (human in loop)       (ground truth,
                     ▼                                       bake-off, gates)
                 alert rules
                     │
                     ▼
              API + dashboard ◀── agentic research layer (Aion)
```

Three signals ship today:

| Signal | Source | Question it answers |
|---|---|---|
| `guidance_tone` | MD&A, 8-K results | Did management's forward outlook improve or deteriorate? |
| `risk_delta` | Item 1A across consecutive filings | Did risk disclosure materially worsen? |
| `event_class` | 8-K | What happened, and which way does it cut? |

Adding a fourth is a prompt file, a Pydantic schema, and a unit-builder function.

---

## Measured results

All numbers below are reproducible with `make bakeoff` / `make ab`; raw output in
[`docs/results/`](docs/results/).

### Model selection is empirical, not assumed

`guidance_tone`, 12 hand-labelled cases, identical prompt (`v2`), three local models:

| model | accuracy | direction acc | macro F1 | **hallucination** | ECE | verdict |
|---|---|---|---|---|---|---|
| `llama3.2:3b` | 0.500 | 0.750 | 0.750 | **0.083** | 0.250 | rejected — fabricates citations, overconfident |
| **`llama3.1:8b`** | **0.917** | **1.000** | **1.000** | **0.000** | 0.183 | **selected** |
| `llama3:8b` (8k ctx) | 0.833 | 0.833 | 0.833 | **0.167** | 0.083 | rejected — fabricates citations |

The interesting column is hallucination, not accuracy. Two models that look
respectable on accuracy invent quotes that are not in the source document — and
`recommend()` disqualifies them for it. A free 8B local model clears the bar for
this task, so there is nothing to buy.

### Prompt changes are A/B tested, not asserted

Same model (`llama3.1:8b`), same cases, `v1` vs `v2` of the prompt:

| prompt | accuracy | direction acc | macro F1 | ECE (calibration) |
|---|---|---|---|---|
| `guidance_tone@v1` | 0.750 | 0.833 | 0.806 | 0.317 |
| `guidance_tone@v2` | **0.917** | **1.000** | **1.000** | **0.183** |

`v2` adds worked examples pinning the distinction v1 kept failing — a weak quarter
with *unchanged* guidance is not bearish — plus an explicit decision order. Both
accuracy and calibration improved, and the change is attributable because every
run records the prompt's content hash.

### The ground-truth set is adversarial on purpose

Roughly half the 32 labelled cases are traps drawn from real filing patterns:
record results paired with a guidance cut; a weak quarter with reaffirmed guidance;
safe-harbour boilerplate full of future-tense verbs that says nothing; an 8-K that
furnishes a press release without including any numbers; a risk-factor diff that
is pure re-wording. A model that pattern-matches sentiment words fails these; a
model that reads forward-looking statements passes.

---

## What makes it production-shaped

**Every extraction is attributable.** Prompt name, version, *and* content hash;
model; provider; token counts; cost; latency; repair attempts; grounding ratio.
An extraction you cannot attribute is one you cannot regress against.

**Hallucination is a gated number, not a vibe.** Every schema requires verbatim
evidence; every quote is checked against the source. Extractions below 50%
grounded are excluded from scoring outright — the pipeline's one hard gate. See
[ADR 6](docs/adr/0006-grounding-as-hallucination-metric.md).

**Calibration is measured.** ECE and Brier per run, because a model that is 60%
accurate while claiming 0.95 confidence is more dangerous than one that says it is
unsure — and `signals/score.py` weights by confidence, which is only meaningful if
confidence means something.

**Cost is capped and visible.** A per-run ceiling checked *before* dispatch, a
content-addressed response cache (reproducibility as much as thrift), and
`/metrics/cost` reading straight from the trace table.

**Failure degrades instead of exploding.** Retries with jittered backoff, then
fallback down a model chain; non-retryable errors stop immediately rather than
burning the whole chain on a config bug; a cost-cap breach aborts rather than
quietly retrying somewhere cheaper.

**Bad data is flagged, not buried.** Parse coverage is scored per filing, with
form-specific thresholds, and thin parses go to a review queue instead of entering
the corpus indistinguishable from clean ones.

**Humans are in the loop where it matters.** Invalid, ungrounded, and
low-confidence extractions are queued with a priority, so the queue stays small
enough that someone actually works it.

**CI gates signal quality.** `sf eval gate` runs the full pipeline on a
deterministic stub provider — no GPU, no key, no network — and fails the build on a
hallucination-rate or schema-conformance regression. 199 tests, hermetic.

---

## The agentic layer ("Aion")

A tool-using research loop over the warehouse, bounded three ways — step count,
wall clock, and cost — where hitting a bound produces a partial answer *with the
reason stated*, never a silent truncation or an infinite spin.

Tools: guarded read-only SQL, hybrid filing search, signal lookup, schema
introspection, and a sandboxed Python sidecar. Domain knowledge lives in
[research skills](prompts/skills/) as markdown — that a 10-Q's Item 2 is MD&A while
a 10-K's Item 7 is; that reaffirmed guidance after a weak quarter is not bearish.

```bash
sf agent "Which companies show deteriorating guidance, and what exactly did they say?"
```

**Safety is in the tools, not the prompt.** A prompt saying "only read data" is a
suggestion to a probabilistic system whose inputs include attacker-influenceable
filing text. So the SQL tool strips comments before applying a denylist, refuses
anything but `SELECT`/`WITH`, blocks DuckDB's filesystem and network functions,
rejects stacked statements, forces a `LIMIT`, *and* opens the connection
read-only. The sidecar adds AST rejection, a separate process, `setrlimit` caps,
and a wall-clock kill — and [ADR 5](docs/adr/0005-sidecar.md) is explicit that this
is a barrier against a confused agent, not a boundary against a determined
adversary.

**Honest assessment:** the loop is production-shaped, but `llama3.1:8b` is a weak
multi-step reasoner — it sometimes invents a CIK or searches unproductively before
correctly reporting it could not verify something. The bounded-and-honest failure
mode is the design working; the reasoning quality is a model limitation, and the
registry exists so the agent can be pointed at a stronger model when one is
available.

---

## Layout

```
src/signalforge/
  llm/         provider clients (ollama · anthropic · stub), router, cache, registry
  prompts/     hash-pinned prompt registry
  ingest/      EDGAR client, filing parser, chunker, warehouse loader
  retrieval/   embeddings + hybrid vector/keyword search with rank fusion
  extract/     schemas, grounding check, runner with repair loop, task definitions
  signals/     scoring, composites, alert rules
  evals/       metrics, harness, bake-off, prompt A/B, reporting
  agent/       loop, tools, skills, sandboxed sidecar
  api/         FastAPI service + dashboard
configs/models.yaml   model registry: pricing, context, capabilities, chains
prompts/              versioned prompts + agent skills
evals/datasets/       hand-labelled ground truth
docs/adr/             architecture decisions, including the tradeoffs accepted
```

## Commands

```bash
sf doctor                        # check providers, warehouse, prompts, EDGAR
sf ingest AAPL MSFT --limit 8    # EDGAR → warehouse
sf index                         # embed new chunks
sf search "material weakness" --slug risk_factors
sf extract guidance_tone --chain extract_default
sf score                         # extractions → signals → alerts
sf eval run guidance_tone        # grade vs ground truth + gate
sf eval bakeoff guidance_tone --models llama32-3b,llama31-8b
sf eval ab guidance_tone         # prompt A/B
sf eval gate --chain ci          # hermetic CI gate
sf review                        # human-in-the-loop queue
sf agent "..."                   # agentic research
sf models / sf prompts           # what's deployed, with hashes
sf serve                         # API + dashboard
```

## Configuration

Copy [`.env.example`](.env.example) to `.env`. Every value has a working default
except `SF_SEC_USER_AGENT`, which the SEC requires to carry real contact
information — requests without it risk an IP block.

To route work to Claude: set `SF_ANTHROPIC_API_KEY`, `uv pip install -e '.[hosted]'`,
and point a pipeline at the `extract_hosted` or `judge` chain in
[`configs/models.yaml`](configs/models.yaml). No pipeline code changes.

## Known limitations

Stated rather than hidden:

- **Filer heterogeneity in parsing.** JPMorgan incorporates MD&A by reference to
  its annual report; Tesla's 10-Q risk section cross-references its 10-K. Section
  coverage runs 73–90% across the filers tested. These are flagged into the review
  queue rather than silently ingested, but they are not solved.
- **Ground truth is small (32 cases) and partly synthetic.** The passages are
  written to mirror real filing patterns and the traps are drawn from real ones,
  but a production system needs a few hundred hand-labelled real excerpts with
  inter-annotator agreement.
- **No market data.** Signals say what management *said*, not what the stock did.
  There is no return attribution, so predictive value is unmeasured — the evals
  measure extraction accuracy, which is a different claim.
- **Exact vector scan.** Correct and fast at ~10⁴–10⁵ chunks; needs an ANN index
  beyond that. See [ADR 2](docs/adr/0002-duckdb-warehouse.md).
- **Single-writer warehouse.** DuckDB serialises writers, so concurrent ingest
  blocks. Fine for batch; Postgres is the move for concurrent write load.
- **CI gates the pipeline, not model quality.** Thresholds are calibrated against
  the stub provider so CI stays hermetic; real-model quality is the bake-off,
  which a human reads.

## License

MIT
