---
name: guidance_tone
version: v2
task: guidance_tone
description: >
  v1 plus two changes measured against the ground-truth set: worked examples that
  pin the tone-vs-results distinction (v1's most common error was reading a weak
  quarter as bearish guidance), and an explicit decision order so the guidance
  label drives the direction label rather than the two being judged separately.
variables: [company, form, period, section_text]
---
You are a securities analyst reading a filing from {{ company }}.

Classify management's **forward-looking tone**: how they characterise the business
going forward, not how it performed historically.

Decide in this order:

1. Find any explicit statement about future guidance or outlook. Set
   `guidance_change` to raised, lowered, withdrawn, reaffirmed, or none.
2. Set `direction` consistently with step 1. If guidance was raised, direction is
   bullish; lowered or withdrawn, bearish; reaffirmed or none, judge from the
   forward-looking language alone.
3. Quote the passage that drove step 1.

Worked examples of the distinction that matters most:

- "Revenue declined 12% year over year. We continue to expect full-year revenue
  growth of 5-7%." -> guidance_change "reaffirmed", direction "neutral". Weak
  results, unchanged outlook. Do not mark this bearish.
- "We delivered record quarterly revenue. Given softening demand, we are
  withdrawing our full-year outlook." -> guidance_change "withdrawn", direction
  "bearish". Strong results, deteriorating outlook.
- "We now expect full-year earnings per share of $4.10 to $4.20, up from our
  prior range of $3.85 to $4.00." -> guidance_change "raised", direction
  "bullish".
- "This quarter's results reflect continued execution against our strategy."
  -> guidance_change "none", direction "neutral". Says nothing about the future.

Rules:
- Judge only the text provided. If the section does not discuss the future, return
  direction "neutral", guidance_change "none", low confidence, and say so.
- Every string in `evidence` MUST be copied verbatim from the document, exactly as
  written, with no paraphrasing and no ellipses. Unverifiable evidence is
  discarded as a hallucination.
- Quote 1-3 passages, each under 300 characters.
- `confidence` reflects how clearly the text supports your call, not how strongly
  you feel about the company.

Filing: {{ form }} for the period {{ period }}.

<document name="section">
{{ section_text }}
</document>

Respond with JSON only:
{"direction": "bullish|bearish|neutral", "guidance_change": "raised|lowered|withdrawn|reaffirmed|none", "confidence": 0.0-1.0, "rationale": "one or two sentences", "evidence": ["verbatim quote"]}
