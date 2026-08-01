---
name: frontier-visual-evidence-builder
description: Use to design visuals and runnable evidence that reduce learning barrier, support curiosity, demonstrate mechanisms, and ground staff-depth reasoning.
---

# Frontier Visual Evidence Builder

Visuals are not only proof. They are part of the learning experience.

> **Authoritative V9 authoring grammar + compile/gate loop:**
> `sessions/_compiler/AUTHORING.md` (source of truth — matches the shipped compiler).
> The `%%% svg` / `%%% viz` widget grammar and the per-concept visual gate live there,
> not in the obsolete `v8_source_first_authoring_plan.md`.

## Visual sequence

Prefer:

```text
mental picture visual
→ mechanism visual
→ failure/limit visual
→ artifact reproduction
```

The "mental picture visual" means a picture of the OBJECT itself (the curve, distribution, boundary, shape), not an analogy card standing in for it.

## Depict-in-unit rule (v9)

Every concept unit carries its OWN inline visual: a static labeled figure at minimum, an interactive viz where the behavior is explorable. Analogy is NOT depiction — a "valve"/"dimmer" card does not satisfy "show the ReLU curve." A lesson must not defer a concept's picture to a later unit or to a single shared late "build" section. The moment a concept names a visual object (a curve, distribution, boundary, geometric/matrix shape), that same unit must show it.

Mechanical backstop: the reader_flow_gate concept branch (every @@@ concept must contain a visual marker) + concept_shell_gate (every concept section has a visual). These gates only confirm a visual marker is PRESENT, not that it renders and correctly depicts the object. Opus 5 vision is strong: render the produced %%% svg / %%% viz output and visually inspect it (crop and re-check iteratively) rather than reasoning about the markup alone — verifying the rendered figure beats trusting the marker.

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

**This skill does not own the doing leg.** Despite this skill's "runnable evidence" framing, the learner-facing runnable artifact `sessions/<module>/<day>/experiment.py` belongs to `frontier-lesson-builder` → "The doing leg", is gated by `frontier-refactor-qa` → "Doing-Leg Gates", and is contracted in `sessions/_compiler/AUTHORING.md` section 10. Build it with `sessions/_compiler/workflows/doing_leg_build.js`, not from here.

**A `%%% demo`'s `out:` block is a hand-written CLAIM about what that artifact prints, and no gate compares the two.** The coupling is one character wide: an `out:` line reading `loss: 0.0470` and an artifact printing `loss: 0.047` disagree on screen while both pass every gate. When a demo shows an output the doing leg also produces, copy it from a REAL run of the artifact, and re-copy it whenever either side changes. If the demo is deliberately illustrative rather than a transcript, say so in the demo body so a later reviewer does not "fix" the artifact to match a number that was never real.

## Seed propagation visual risk

If a seed lesson uses static or late visuals in a way that future modules may copy, record this as seed-stabilization P1 when appropriate.

## v8 — Source-first behavioral viz

In `source.md`, a behavioral visual is a `%%% viz src=../../viz/<file>.html title="…" caption="…" %%%`
widget naming an EXISTING `sessions/viz/*.html`; a static shape is a `%%% svg … %%%`
widget holding a closed `<svg>…</svg>`. (See AUTHORING.md §4/§6 for the exact grammar —
the earlier `behavioral=true` flag is obsolete.)

The compiler renders `%%% viz` as the auto-resizing `.build-embed` iframe + `postMessage`
height sender. A concept unit with no visual is a **compile error** (Reader Flow + Concept
Shell gates), not a late QA finding.

Keep claim and moving evidence adjacent in `source.md`. Do not let prose and its plotted evidence drift into separate files — that decoupling is why prose-only behavioral gaps recur in hand-authored HTML.

## Deliverable length

Match the length of written deliverables (especially Markdown files) to what the task needs: cover the substance, but do not pad documents with filler sections, redundant summaries, or boilerplate.
