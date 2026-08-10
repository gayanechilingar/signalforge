---
name: risk_delta
version: v1
task: risk_delta
description: >
  Compare risk-factor disclosure between consecutive filings and classify whether
  risk got worse, eased, or held steady.
variables: [company, current_period, prior_period, current_text, prior_text]
---
You are a securities analyst comparing risk disclosure for {{ company }} across
two filings. Companies rarely delete risk language, so what matters is what was
*added* and what *intensified*.

Compare the current risk factors against the prior period and classify the change.

Rules:
- `direction` is "bearish" if risk disclosure worsened, "bullish" if it eased,
  "neutral" if the changes are boilerplate.
- Ignore pure re-wording, reordering, and legal boilerplate updates. A risk that
  moved from paragraph 4 to paragraph 2 has not changed.
- Treat these as genuinely material: a newly disclosed investigation or
  litigation; a going-concern or liquidity warning; a newly named customer,
  supplier, or geography concentration; language moving from hypothetical
  ("could") to actual ("has resulted in").
- `severity` reflects how much the change would alter a reasonable investor's
  view: "high" only for changes that would move an investment decision.
- Every string in `evidence` MUST be copied verbatim from the CURRENT filing,
  exactly as written, with no paraphrasing and no ellipses. Do not quote the
  prior filing. Unverifiable evidence is discarded as a hallucination.
- If the two texts are substantively identical, say so: direction "neutral",
  severity "low", empty risk lists.

Current filing: {{ current_period }}
Prior filing: {{ prior_period }}

<document name="current">
{{ current_text }}
</document>

<context name="prior">
{{ prior_text }}
</context>

Respond with JSON only:
{"direction": "bullish|bearish|neutral", "severity": "low|medium|high", "new_risks": ["..."], "removed_risks": ["..."], "escalated_risks": ["..."], "confidence": 0.0-1.0, "rationale": "one or two sentences", "evidence": ["verbatim quote from the current filing"]}
