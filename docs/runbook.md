# Runbook

Operating the pipeline: what to run, what breaks, and how to tell which.

## Daily flow

```bash
sf ingest AAPL MSFT NVDA --limit 8   # new filings only; re-ingest is idempotent
sf index                             # embeds new chunks only
sf extract event_class               # cheapest and most time-sensitive first
sf extract guidance_tone
sf extract risk_delta                # needs two comparable filings per company
sf score                             # extractions → signals → alerts
```

Order matters in one place: `risk_delta` produces nothing until a company has two
filings of the *same form*, so a fresh corpus yields no risk signals until the next
quarter's filing lands.

`sf score` is free — it re-reads stored extractions rather than re-running
inference. Re-run it freely after changing scoring rules or alert thresholds.

## Diagnosing a bad signal

Work outward from the data, because the model is the *last* thing to suspect:

1. **Was the filing parsed correctly?**
   ```sql
   SELECT slug, char_len FROM sections WHERE accession = '<acc>' ORDER BY char_len DESC;
   ```
   A `body` slug means section detection failed entirely and the whole filing was
   fed in as one blob. A `risk_factors` section of a few hundred characters means
   the filing cross-references another document rather than restating it.

2. **Was it flagged at ingest?**
   ```bash
   sf review --task ingest_parse
   ```

3. **What did the model actually see and say?**
   ```sql
   SELECT model, prompt_version, prompt_hash, repair_attempts, grounded_ratio,
          payload, error
   FROM extractions WHERE accession = '<acc>' AND task = '<task>';
   ```

4. **Was the evidence real?** `grounded_ratio < 1.0` means at least one quote was
   not in the source. Below 0.5 the extraction is excluded from scoring already.

5. **Only now suspect the model.** Reproduce against ground truth:
   ```bash
   sf eval run <task> --chain llama31-8b
   ```

## Symptoms and causes

| Symptom | Likely cause | Check |
|---|---|---|
| `sf extract` produces 0 units | Nothing in the corpus matches the task's source sections | `SELECT slug, count(*) FROM sections GROUP BY slug` |
| All signals score 0.0 / neutral | The section text fed in is financial tables, not prose | Read `sections.text` for that accession |
| `risk_delta` returns nothing | Only one filing per company/form | `SELECT cik, form, count(*) FROM filings GROUP BY 1,2` |
| High `repair_rate` | Model follows schemas poorly, or the prompt's JSON example drifted from the schema | `sf eval run <task>` and compare across models |
| Hallucination rate climbing | Prompt weakened its verbatim-quote instruction, or the model changed | Diff the prompt hash against the last good `eval_runs` row |
| Search returns nothing | Chunks not embedded | `sf index`; then `SELECT count(*) FROM embeddings` |
| Every 8-K flagged at ingest | Coverage threshold applied to the wrong form | `MIN_SECTION_COVERAGE` in `ingest/store.py` is per-form for this reason |
| Agent loops on one tool | Model is a weak multi-step reasoner | The loop already short-circuits repeats; raise the model tier rather than `max_steps` |
| `CostCapExceeded` | Working as designed | Raise `SF_RUN_COST_CAP_USD` deliberately, having looked at `/metrics/cost` |

## Cost and latency

```sql
-- Spend and tail latency by model, last 7 days
SELECT model, count(*) AS calls,
       sum(CASE WHEN cached THEN 1 ELSE 0 END) AS cache_hits,
       round(sum(cost_usd), 4) AS usd,
       round(quantile_cont(duration_ms, 0.95), 0) AS p95_ms
FROM traces
WHERE kind = 'llm' AND started_at > now() - INTERVAL 7 DAY
GROUP BY 1 ORDER BY usd DESC;
```

Or `GET /metrics/cost?days=7`.

A **falling cache hit rate** usually means a prompt changed — the cache key covers
the rendered prompt, so an edit invalidates every entry for that task. That is
correct behaviour, and worth knowing before concluding the cache is broken.

## Changing a prompt safely

1. Copy the current version to a new file, bump `version` in the frontmatter.
2. Edit the new file. Never edit a released version in place — the hash changes and
   historical `eval_runs` rows stop being comparable.
3. A/B it: `sf eval ab <task> --chain llama31-8b`
4. Ship only on improvement in **both** accuracy and hallucination rate. A prompt
   that gains accuracy while fabricating more citations is a regression.
5. Update the chain or default in `configs/models.yaml`.

## Changing a model

```bash
sf eval bakeoff <task> --models llama32-3b,llama31-8b,sonnet-5
```

`recommend()` picks the cheapest model clearing the quality bar and prints why each
rejected model was rejected. Then edit the relevant chain in
`configs/models.yaml` — no code change.

## Adding a signal

1. Pydantic schema in `extract/schemas.py`, registered in `SCHEMAS`. It must carry
   `evidence`, or grounding cannot be measured.
2. Prompt file in `prompts/`, with `task` matching the schema key.
3. Unit builder in `extract/tasks.py`, registered in `UNIT_BUILDERS`.
4. Ground truth in `evals/datasets/<task>.jsonl` — **including traps**, or the evals
   will overstate quality.
5. Weight in `COMPOSITE_WEIGHTS` and, if warranted, an alert rule.
6. Gate thresholds in `evals/harness.py:DEFAULT_GATES`.

## Backfill

```bash
for t in AAPL MSFT NVDA GOOGL AMZN; do sf ingest $t --limit 40; done
sf index
sf extract guidance_tone   # ~7-20s per unit on local models
```

Local inference dominates: budget roughly 10-20 seconds per extraction. The
response cache makes re-runs after a scoring change effectively free, but a prompt
change re-pays in full.

## SEC etiquette

Non-negotiable, because a block affects everyone behind the IP:

- `SF_SEC_USER_AGENT` must carry a real contact email.
- Rate limit stays at or below 10 req/s (`SF_SEC_RATE_LIMIT_PER_S`, default 6).
- Filing documents are cached on disk; do not clear `data/edgar_cache/` casually.
