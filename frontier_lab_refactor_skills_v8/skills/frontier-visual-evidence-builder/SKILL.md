---
name: frontier-visual-evidence-builder
description: Use to design visuals and runnable evidence that reduce learning barrier, support curiosity, demonstrate mechanisms, and ground staff-depth reasoning.
---

# Frontier Visual Evidence Builder

Visuals are not only proof. They are part of the learning experience.

## Visual sequence

Prefer:

```text
mental picture visual
→ mechanism visual
→ failure/limit visual
→ artifact reproduction
```

## Behavioral visual rule

Behavioral concepts require plotted or live-recomputed evidence.

Examples:

- attention score heatmap,
- Q/K/V projection fan,
- softmax distribution changes,
- activation curve behavior,
- derivative behavior,
- decision boundary movement,
- learning-rate path,
- loss curve.

## Curiosity visual rule

The first visual in a hard concept should reduce barrier, not add machinery.

## Show-the-failure rule

If a failure mode is taught, show the broken behavior visually or through a runnable artifact.

## Artifact reproducibility rule

If a playground demonstrates a behavior, the Produce artifact should reproduce it or clearly explain why it is illustrative.

## Seed propagation visual risk

If a seed lesson uses static or late visuals in a way that future modules may copy, record this as seed-stabilization P1 when appropriate.

## v8 — Source-first behavioral viz

In `source.md`, a behavioral block is flagged `behavioral=true` and names a `viz/*.html`.

The compiler inserts the auto-resizing iframe + `postMessage` height sender. A behavioral block with no viz is a **compile error**, not a late QA finding.

Keep claim and moving evidence adjacent in `source.md`. Do not let prose and its plotted evidence drift into separate files — that decoupling is why prose-only behavioral gaps recur in hand-authored HTML.
