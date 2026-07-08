---
name: frontier-refactor-qa
description: Use after curriculum edits to validate coverage, coach voice, visual/evidence quality, artifacts, anchor consistency, legacy skill references, shell behavior, and optional held-out evaluation.
---

# Frontier Refactor QA

You are the strict quality gate.

Do not let "good structure" hide missing teaching quality.

## P0 — fix immediately

- missing must-cover concept
- must-cover concept is PARTIAL/FAIL without explicit waiver
- coverage contract not traceable to lesson text
- formula-first opening
- no pain point / no intuition before math
- missing neural foundation framing in neural-network foundation lesson
- generic everyday analogy used as substitute for brain-vs-ANN framing
- incorrect math
- misleading visual
- P0 visual/evidence missing or unjustified
- failure mode taught without evidence or required artifact
- simulated terminal counted as evidence without reproducible artifact
- inconsistent anchors that teach conflicting mental models
- stale run command
- Produce path mismatch
- legacy skill reference to uninstalled skill
- broken JS/navigation/completion
- duplicated paragraph
- truncated title/finale

## P1 — fix before merge if easy

- weak analogy
- missing Reader-objection Q&A
- dry tone
- missing Jargon Buster
- weak acceptance criteria
- unclear visual caption
- quantitative visual missing on-panel numeric readout
- comparison table missing for should-cover family
- minor anchor inconsistency that does not mislead

## P2 — final polish

- Markdown line wrapping
- cosmetic report formatting
- wording polish

## Required checks

1. Scope boundary
2. Coverage traceability
3. Coach Voice Gate
4. Foundation Framing Gate
5. Function Family Gate
6. Visual/Evidence Gate
7. Anchor Consistency Gate
8. Artifact Gate
9. Legacy skill reference check
10. Math correctness
11. JS/shell behavior
12. Audit/jsdom if available

## Coverage traceability check

For every must-cover concept in the coverage contract:

- grep or inspect the named concept in the target lesson,
- verify it is explained at the required depth,
- verify it has visual/evidence/artifact if required,
- mark PASS / PARTIAL / FAIL.

Create a table:

| Concept | Required depth | Lesson evidence | Visual/evidence | Artifact | Result |
|---|---|---|---|---|---|

Do not sign off with "all concepts landed" unless this table passes.

PARTIAL/FAIL on must-cover concepts blocks merge unless explicitly waived.

## Coach Voice Gate

Mark P0 if:

- first 20% starts with formula/definition before why-care and intuition,
- no pain point is named,
- analogy is one sentence only,
- no tiny example appears before formulas,
- foundation lesson feels like docs instead of warm coaching.

Mark P1 if:

- tone is correct but dry,
- Chinese intuition scaffold is missing,
- interview answer sounds memorized.

## Foundation Framing Gate

For neural-network foundations, mark P0 if missing:

- explicit brain-inspired intuition,
- artificial-neuron caveat,
- mapping table from biological intuition to artificial components,
- where the analogy breaks,
- transition to math function.

A human/everyday analogy such as judge, chef, or scoring function does not satisfy the brain-vs-ANN framing requirement.

## Function Family Gate

For a must-cover mechanism family, check:

- representative variants,
- comparison table,
- when-to-use guide,
- failure modes,
- modern context if relevant.

For activation functions, expect sigmoid, tanh, ReLU, Leaky ReLU, softmax, derivative intuition, saturation, vanishing gradient, dying ReLU, and GELU/SwiGLU context unless explicitly out of scope.

## Visual/Evidence Gate

Mark P0 if:

- P0 visual/evidence missing,
- quantitative visual lacks derived number,
- failure mode has no demo/evidence/artifact,
- simulated terminal is not clearly labeled,
- Produce artifact cannot reproduce playground output,
- naive-vs-idiomatic claim lacks benchmark or required benchmark artifact.

## Anchor Consistency Gate

Identify anchor quantities:

- step counts,
- split ratios,
- seeds,
- learning rates,
- dataset sizes,
- sequence lengths,
- batch sizes,
- parameter values,
- metric definitions,
- model dimensions.

Verify consistency across:

- prose,
- visual,
- playground,
- quiz,
- Produce artifact,
- acceptance criteria.

Mark P0 if inconsistent anchors teach conflicting mental models.

## Artifact Gate

Check:

- exact path,
- exact run command,
- expected output,
- acceptance criteria,
- explain-back.

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
- frontier-qa-gate
- frontier-visual-lab
- frontier-lesson-coach

Replace with current v7.1 skills or plain Claude Code prompts.

## Held-out eval standard

Only run held-out eval when user asks.

Held-out eval must include an adversarial section:

- where lessons overclaim,
- where Staff Lens hides beginner coverage gaps,
- where generic analogy substitutes for required foundation framing,
- where visuals/experiments are weaker,
- where runnable evidence is missing,
- what skills failed to specify.

Do not over-credit Staff Lens or frontier relevance if core beginner coverage, intuition, visual evidence, or runnable artifacts are weak.

## Report format

Create:

```text
sessions/<module_or_theme>_qa_report.md
```

Include:

```markdown
# QA Report

## Scope

## Coverage traceability table

## Coach Voice Gate

## Foundation Framing Gate

## Function Family Gate

## Visual/Evidence Gate

## Anchor Consistency Gate

## Artifact Gate

## P0 findings

## P1 findings

## P2 findings

## Fixes applied

## Merge recommendation
```

Merge recommendation:

- Pass
- Pass with P1
- Blocked

Do not block on P2.
