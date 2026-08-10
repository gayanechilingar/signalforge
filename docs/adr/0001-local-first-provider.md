# 1. Local-first model provider, hosted as an upgrade

**Status:** accepted

## Context

The system needs an LLM for every extraction. The obvious default is a frontier
hosted model, but this project had to be buildable and runnable with no API key,
and a portfolio artefact that cannot be run by whoever clones it is worth much
less than one that can.

## Decision

Ollama with local open-source models is the **default** provider. Anthropic is a
first-class but env-gated provider: setting `SF_ANTHROPIC_API_KEY` and pointing a
chain at `extract_hosted` is the only change needed to route work to Claude. A
third `stub` provider serves CI.

All three implement the same `LLMClient` interface, and selection is a name in
`configs/models.yaml` rather than a code path.

## Consequences

**Good.** A clone runs end to end for free. The three-provider split forces the
abstraction to be real rather than nominal — mistakes like sending `temperature`
to a model that rejects it surface immediately. It also makes the open-source
question empirical: the bake-off measures whether a local 8B model is good enough
for a given task, and for `guidance_tone` it is (1.00 direction accuracy, zero
hallucination — `docs/results/`).

**Bad.** Local models violate schemas far more often than schema-constrained
hosted decoding, which is why the repair loop in `extract/runner.py` exists at
all. Local latency is also ~7-10s per extraction versus sub-second, so batch
windows are sized around the local path.

**Accepted risk.** Two decoding paths means two behaviours to test. Mitigated by
running the same eval suite against both.
