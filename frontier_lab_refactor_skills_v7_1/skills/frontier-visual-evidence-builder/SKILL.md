---
name: frontier-visual-evidence-builder
description: Use to design and implement notebook-grade visuals and runnable evidence for Frontier Lab lessons. Enforces visual contracts, failure demos, small multiples, benchmarks, numeric readouts, simulated-vs-runnable rules, and P0/P1/P2 priorities.
---

# Frontier Visual Evidence Builder

You make the concept visible and testable.

A strong lesson should show the idea moving or failing, not merely describe it.

## Visual/evidence contract

Before implementation, define:

| Field | Description |
|---|---|
| Learning question | What should the learner understand? |
| Evidence type | visual, runnable demo, benchmark, calculator, table |
| Controls/output | sliders, toggles, printed output |
| Derived quantity | number shown on panel or printed |
| Failure shown | what breaks |
| Misconception prevented | wrong mental model fixed |
| Artifact link | how learner reproduces it |

## P0 rule

A visual/evidence item is P0 when:

- the concept is foundation/math/systems-heavy,
- the learner cannot build the mental model from prose,
- a failure mode is taught,
- a comparison is central,
- the coverage contract marks it must-cover.

## Simulated vs runnable rule

A simulated terminal is not evidence by itself.

It only counts as supporting evidence if:

1. it is clearly labeled as simulated,
2. the Produce artifact reproduces the same behavior,
3. the acceptance criteria require the learner to run it,
4. the expected output is specified.

If a failure mode is central, prefer runnable code or a required artifact.

## Show-the-failure rule

For any taught failure mode, show the broken behavior.

Examples:

- softmax overflow → `[nan nan nan]`
- stable softmax → valid probabilities
- dead ReLU → fraction dead
- learning rate too high → loss explodes
- loop vs vectorized → measured speed gap
- validation/test leakage → suspiciously high final performance

## On-panel numeric readout

Every quantitative visual must display the derived number it teaches:

- slope
- derivative
- intercept
- entropy
- accuracy
- loss
- memory
- speedup
- dead-unit fraction
- train/validation/test split counts
- gradient error vs finite difference

A moving shape without the number is incomplete.

## Parameter-isolation small multiples

When 2+ parameters interact, include small multiples varying one parameter while freezing the others.

Examples:

- vary weight while holding bias fixed
- vary bias while holding weights fixed
- vary learning rate while holding initialization fixed
- vary sequence length while holding batch/model fixed

## Decision-boundary triptych

For any linear classifier, include:

1. contour/output field
2. output=0 boundary
3. boundary equation with slope/intercept labeled

Controls should show:

- weights rotate boundary,
- bias translates boundary.

## Activation visual standard

For activation lessons, include when in coverage contract:

- sigmoid curve
- tanh curve
- ReLU curve
- Leaky ReLU curve
- derivative/slope view
- saturation zones
- dead ReLU region
- softmax probability bars
- temperature slider
- compact GELU/SwiGLU context

## Benchmark rule

When teaching an idiomatic vs naive implementation, include or require a benchmark.

Examples:

- loop vs vectorized
- naive softmax vs stable softmax
- fused vs unfused
- batched vs unbatched

## Artifact reproducibility rule

If a playground demonstrates a behavior, the Produce artifact must reproduce it or clearly explain why it is only illustrative.

The acceptance criteria should include observable output.

## Required visual card

Every visual must include:

```html
<p><strong>What you should notice:</strong> ...</p>
<p><strong>Common misconception:</strong> ...</p>
<p><strong>Where this visual simplifies reality:</strong> ...</p>
```

## Technical constraints

Preserve existing lesson shell:

- navigation
- quest ids
- localStorage
- quiz
- BUILD / DEMOS
- completion gates
- theme behavior
- offline behavior

Prefer inline SVG and vanilla JS unless the repo already supports another pattern.

## Done definition

A visual/evidence item passes when:

- it answers a learning question,
- it shows a derived number,
- it prevents a misconception,
- it links to an artifact,
- it is reproducible by the artifact when applicable,
- it does not break completion behavior.
