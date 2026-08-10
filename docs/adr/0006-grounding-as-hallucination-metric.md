# 6. Citation grounding as the hallucination metric

**Status:** accepted

## Context

"Does the model hallucinate" needs to be a number in a regression gate, not an
impression. Asking a model to self-report confidence does not work: a model asked
for a label and a confidence always produces both, and both always look
reasonable.

## Decision

Every extraction schema **requires verbatim evidence quotes**. After extraction,
each quote is checked against the source text it was drawn from. The
hallucination rate is the share of extractions containing at least one quote that
is not in the source, and an extraction below 50% grounded is **excluded from
scoring entirely** — the one hard gate in the pipeline.

Matching widens in three stages: exact on normalised text, contiguous on
alphanumerics only, then token-subsequence overlap at a 0.85 threshold.

## Consequences

**Good.** Hallucination becomes a measured, gated quantity. It found a real
difference between models that accuracy alone hid: on `guidance_tone`,
`llama3.2-3b` fabricated citations on 8.3% of cases and `llama3-8b` on 16.7%,
while `llama3.1-8b` was at 0% — which disqualified two otherwise-plausible models.

**Bad.** The fuzzy threshold is a judgement call. Too low and paraphrase passes
as citation; too high and legitimate quotes are called fabrications. It is a named
constant, reported per run, and re-tunable against evidence.

**Deliberate choice.** An extraction with *no* quotes scores 1.0, not 0.0 — it made
no citation claims, so it fabricated none. Whether an evidence-free extraction is
acceptable is a separate policy question, handled by the review queue.
