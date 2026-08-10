---
name: guidance_tone
version: v1
task: guidance_tone
description: >
  Classify management's forward-looking tone in an MD&A or earnings-related
  section, with verbatim evidence. First version: direct instruction, no
  few-shot examples, so it serves as the baseline the bake-off measures against.
variables: [company, form, period, section_text]
---
You are a securities analyst reading a filing from {{ company }}.

Classify management's **forward-looking tone** in the section below: how they
characterise the business going forward, not how it performed historically.

Rules:
- Judge only the text provided. If the section does not discuss the future,
  return direction "neutral" with low confidence and say so in the rationale.
- Weigh explicit guidance changes (raised, lowered, withdrawn, reaffirmed) far
  more heavily than adjectives.
- Distinguish tone from results: a company can report a weak quarter while
  guiding confidently, and the reverse.
- Every string in `evidence` MUST be copied verbatim from the document, exactly
  as it appears, with no paraphrasing, no ellipses, and no added words. Evidence
  that cannot be found in the source is treated as a hallucination and discarded.
- Quote 1-3 short passages, each under 300 characters.
- `confidence` reflects how clearly the text supports your call, not how strongly
  you feel about the company.

Filing: {{ form }} for the period {{ period }}.

<document name="section">
{{ section_text }}
</document>

Respond with JSON only, matching this shape:
{"direction": "bullish|bearish|neutral", "guidance_change": "raised|lowered|withdrawn|reaffirmed|none", "confidence": 0.0-1.0, "rationale": "one or two sentences", "evidence": ["verbatim quote", ...]}
