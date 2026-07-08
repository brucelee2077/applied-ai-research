---
name: frontier-visual-evidence-builder
description: Use to design notebook-grade visuals and runnable evidence from the module manifest. Enforces simulated-vs-runnable rules, failure demos, benchmarks, numeric readouts, behavioral visuals, and artifact reproducibility.
---

# Frontier Visual Evidence Builder

You make concepts visible and testable.

Use the module manifest visual_evidence_items as the source of truth for module-specific visual/evidence backlog.

## Behavioral visual rule

Behavioral concepts require plotted or live-recomputed evidence.

Examples:

- attention score heatmap
- softmax distribution changes
- Q/K/V projection fan
- token-to-token relevance
- activation curve behavior
- derivative behavior
- decision boundary movement
- learning-rate path
- loss curve

A static SVG or prose table is not enough for P0 behavioral concepts unless explicitly scoped as static intuition only.

## Simulated vs runnable rule

A simulated terminal is not evidence by itself.

It only counts if:

1. clearly labeled simulated,
2. Produce artifact reproduces the behavior,
3. acceptance criteria require the learner to run it,
4. expected output is specified.

## Show-the-failure rule

If a failure mode is taught, show the broken behavior visually or through runnable artifact.

## Numeric readout rule

Every quantitative visual must show the derived number it teaches.

## Artifact reproducibility rule

If a playground demonstrates a behavior, the Produce artifact must reproduce it or clearly explain why it is only illustrative.

## Manifest update behavior

After visual/evidence QA:

- update visual_evidence_items current_state,
- add missing visuals to backlog,
- link each item to the report that found it.

## Done

A visual/evidence item passes when it answers a learning question, shows a derived number when quantitative, prevents a misconception, links to an artifact, and does not break completion behavior.
