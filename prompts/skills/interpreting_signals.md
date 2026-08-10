---
name: interpreting_signals
description: How to read this system's signal scores, and what they do not mean.
---

## Interpreting signals

**Scale.** Every signal score runs -1 (bearish) to +1 (bullish) and is already
discounted by the model's confidence. A score of -0.25 is a weak bearish read, not
a quarter of a catastrophe. Scores between -0.1 and +0.1 are reported as neutral.

**Signal meanings.** `guidance_tone` reads management's forward-looking language
in MD&A or a results release. `risk_delta` compares risk disclosure against the
prior comparable filing. `event_class` classifies a discrete 8-K event.

**Confidence is not accuracy.** Confidence is the model's own stated certainty.
Check the calibration figures in `eval_runs` before treating it as reliable; a
model with high ECE is overconfident and its confidence should not be used to gate
anything.

**Every signal is text-derived.** These signals say what management *said*, not
what happened. They carry no price, volume, or fundamental data. Never present a
signal as a prediction of returns, and never describe one as a recommendation to
buy or sell — describe what the disclosure said and let the reader draw the
conclusion.

**Aggregate carefully.** A composite over one signal is not comparable to a
composite over three; check the `coverage` figure. Absence of a signal means the
filing type was absent, not that the news was neutral.

**Interpretation traps.** Reaffirmed guidance after a weak quarter is *not*
bearish — the outlook did not change. Record results alongside a guidance cut *is*
bearish; forward statements dominate historical ones. Risk sections are mostly
boilerplate, so a small bearish risk_delta is usually re-wording rather than news.
