---
name: frontier-visual-evidence-builder
description: Use to design notebook-grade visuals and runnable evidence. Enforces simulated-vs-runnable rules, failure demos, benchmarks, numeric readouts, and artifact reproducibility.
---

# Frontier Visual Evidence Builder

You make concepts visible and testable.

## Simulated vs runnable rule

A simulated terminal is not evidence by itself.

It only counts if:

1. clearly labeled simulated,
2. Produce artifact reproduces the behavior,
3. acceptance criteria require the learner to run it,
4. expected output is specified.

## Behavioral vs structural visual rule

Split every P0 visual into two kinds and hold them to different bars:

- **Structural** — a fixed relationship or wiring (a neuron's `w·x+b` assembly, a layer's shape chain, a
  loop cycle). A hand-authored static SVG with labeled numbers is acceptable here.
- **Behavioral** — the concept IS a quantity *moving over a range*: a curve bending, a **derivative going
  flat** (saturation / vanishing gradient), a **decision boundary rotating/shifting**, a loss falling, a
  step overshooting, a gradient exploding. For a behavioral concept a static SVG with baked-in numbers, or a
  prose/table readout ("slope near 0 ≈ 0.25"), **does NOT satisfy the P0 visual** — it *tells* what must be
  *shown*. Require a **plotted or live-recomputed** viz that renders the quantity across its range (a real
  plot, or the existing `build-embed` iframe → a `viz/*.html` that recomputes on input change).

Litmus test: *if changing a parameter should change the picture, the picture must actually recompute.* A
"fake terminal" whose outputs are literal strings, or an SVG whose numbers are frozen, fails this for any
behavioral concept. If the lesson's medium cannot host a live plot, the **runnable artifact must plot it**
and the lesson must point to that output — the behavioral claim may not rest on prose alone.

When the taught idea is a function's **derivative** or **geometry**, the required visual is the plotted
derivative / the plotted boundary — not a table cell describing it.

## Show-the-failure rule

If a failure mode is taught, show the broken behavior visually or through runnable artifact. For a failure
whose signature is a *shape* (a derivative flattening, a loss curve diverging), showing it means the plotted
or run curve — not a sentence asserting it happened.

Examples:

- softmax overflow -> `[nan nan nan]`
- stable softmax -> valid probabilities
- learning rate too high -> loss explodes
- dead ReLU -> dead-unit fraction
- leakage -> suspiciously high test result
- saturation / vanishing gradient -> the plotted derivative visibly collapses toward 0 in the tails

## Numeric readout rule

Every quantitative visual must show the derived number it teaches.

## Workflow usage

In dynamic workflows, visual/evidence subagents should report:

- P0 visuals/evidence missing,
- simulated-only evidence,
- behavioral concept shown only via static SVG / prose table (not plotted or live-recomputed),
- artifact reproducibility gaps,
- numeric readout gaps,
- broken or misleading visuals.

## Done

A visual/evidence item passes when it answers a learning question, shows a derived number when quantitative, prevents a misconception, links to an artifact, and does not break completion behavior.
