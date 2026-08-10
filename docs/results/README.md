# Measured results

Raw output from the eval harness, reproducible with the commands shown. All runs
used the hand-labelled ground truth in [`../../evals/datasets/`](../../evals/datasets/).

| File | Command | What it shows |
|---|---|---|
| `bakeoff_guidance_tone.txt` | `sf eval bakeoff guidance_tone --models llama32-3b,llama31-8b,llama3-8b` | Model selection: two of three local models are disqualified for fabricating citations, not for accuracy |
| `prompt_ab_guidance_tone.txt` | `sf eval ab guidance_tone --chain llama31-8b` | Prompt v2 vs v1 on identical cases: +0.17 accuracy, +0.13 calibration |

## Reading the columns

- **acc** — exact match on every labelled field.
- **dir acc** — sign agreement only. For most of these tasks getting the direction
  right is most of the value; confusing `lowered` with `withdrawn` is a minor error,
  confusing either with `raised` is not.
- **halluc** — share of extractions containing at least one quote that is not in
  the source document. Any non-zero value is a problem, which is why it is
  coloured. This is the column that decided the bake-off.
- **schema err** — share of responses that never validated, even after the repair
  loop. The clearest separator between local models in general, though all three
  reached zero here.
- **ECE** — expected calibration error. A model that is 60% accurate while claiming
  0.95 confidence is more dangerous than one that admits uncertainty, because
  `signals/score.py` weights by confidence.
- **$/1k** — cost per thousand extractions. Zero for local models; the accounting
  path still runs so it is exercised before a paid provider is wired in.

## Caveat on cached latency

`p95 ms` reads 0.000 for a run whose responses were served from the response
cache — a replay costs no time and no money, and reporting the original figures
would inflate every cost dashboard. Compare latency only across uncached runs.
