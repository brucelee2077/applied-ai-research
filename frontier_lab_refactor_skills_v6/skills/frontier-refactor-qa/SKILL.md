---
name: frontier-refactor-qa
description: Use after curriculum edits to validate coverage contracts, visual contracts, artifact paths, stale commands, JS/shell behavior, legacy skill references, and optional held-out evaluation.
---

# Frontier Refactor QA

You are the quality gate after curriculum refactor.

Be strict on learning quality.
Be relaxed on cosmetic polish.

## P0 — fix immediately

- missing must-cover concept
- incorrect math
- misleading visual
- P0 visual not implemented or not justified
- broken JS
- broken navigation
- broken quiz/playground completion
- stale run command
- Produce path mismatch
- duplicated paragraph
- truncated title/finale
- missing artifact path
- legacy skill reference to uninstalled skill

## P1 — fix before merge if easy

- weak acceptance criteria
- unclear visual caption
- minor title spacing
- terminology inconsistency
- missing comparison table for an important concept
- coverage too shallow for a should-cover concept

## P2 — final polish

- Markdown line wrapping
- raw report formatting
- non-blocking wording polish

## Required checks

1. Scope boundary
2. Coverage contract satisfaction
3. Visual contract satisfaction
4. Artifact contract satisfaction
5. Run command correctness
6. Legacy skill references
7. Duplicate paragraphs
8. Title/finale integrity
9. Math correctness
10. Visual correctness
11. JS/shell behavior
12. Audit/jsdom if available

## Legacy skill reference check

Search for old skill names:

- frontier-experiment-lab
- frontier-session-coach
- frontier-math-unfogger
- frontier-lesson-humanizer
- frontier-d3-visual-lab
- frontier-visual-auditor
- frontier-artifact-reviewer
- frontier-curriculum-rollout

If these are no longer installed, replace with current v6 skills or plain instructions.

## Optional held-out eval

Only run held-out evaluation if user explicitly asks.

When running held-out eval:

- do not edit lessons
- compare generated lessons against held-out notebooks/sources
- identify where generated lessons win/lose
- identify missing skill rules
- recommend skill patches, not just lesson patches

## Report format

Create:

```text
sessions/<module_or_theme>_qa_report.md
```

Include:

```markdown
# QA Report

## Scope

## Coverage contract result

## Visual contract result

## Artifact contract result

## Checks run

## P0 findings

## P1 findings

## P2 findings

## Fixes applied

## Remaining risks

## Merge recommendation
```

## Merge recommendation

Use:

- Pass
- Pass with P1
- Blocked

Do not block on P2.
