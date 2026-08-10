---
name: reading_filings
description: Domain facts about SEC filing structure that models reliably get wrong.
---

## Reading SEC filings

**Item numbers mean different things in different forms.** In a 10-K, Item 1 is
Business, Item 1A is Risk Factors, Item 7 is MD&A. In a 10-Q, Item 1 is Financial
Statements and Item 2 is MD&A. Never infer a section's content from its item
number alone — in this corpus sections are stored by name (`risk_factors`,
`mdna`, `results_of_operations`), so use those.

**8-K item codes are stable and meaningful.** 2.02 is results of operations,
4.01 an auditor change, 4.02 a non-reliance (restatement), 5.02 a
director/officer change, 2.06 a material impairment, 3.01 a delisting notice.
An 8-K reports one event even when it lists several items — 2.02 plus 9.01 is an
earnings release with exhibits, not two events.

**A 10-Q's risk factors are an update, not a restatement.** Companies file the
full risk section annually in the 10-K and only material changes quarterly. A
short or absent 10-Q risk section usually means "no change," not "risk
disappeared." Never compare a 10-K risk section against a 10-Q one and conclude
risk eased.

**Filings incorporate by reference.** "Refer to pages 34-58 of our Annual Report"
means the content is not in this document. If a section is a cross-reference, say
the data is unavailable rather than analysing the cross-reference.

**Furnished vs filed.** Item 2.02 and 7.01 content is *furnished*, often as an
exhibit that is not in the primary document. An 8-K that says "a copy of the press
release is furnished as Exhibit 99.1" contains no numbers — do not infer the
earnings outcome from its absence.
