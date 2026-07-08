# Frontier Lab Refactor Skills v8

v8 turns lesson authoring into a **source-first** pipeline and removes the notebook dependency.

## The problem v8 fixes

Hand-refactoring `lesson.html` felt rougher than authoring a notebook — even after a *successful* v7.6 seed stabilization. Diagnosis (`sessions/_refactor/v8_exemplar_distillation_report.md`): the notebook is smooth because **authoring surface = reader surface**; `lesson.html` conflates authoring with rendering, interactivity, and state, so reader order ≠ source order and ~15 frozen invariants tax every edit.

## What v8 adds (all additive over v7.6)

```text
Source-First Authoring Rule    author source.md, compiler emits lesson.html
Reader Flow Blueprint          one canonical top-to-bottom reader sequence
Six authoring rules            Jargon Ladder, Narrative Spine, Curiosity-before-Rigor,
                               Artifact-as-Discovery, Staff-Depth-after-Orientation
Three Authoring Modes          Exemplar / Reference / First-Principles
No-Notebook Rollout Rule       notebook is calibration, not a build dependency
New gates                      Reader Flow · Notebook Smoothness (pilot) ·
                               No-Notebook Authoring · Shell Invariant · Staff Depth
```

## The notebook is NOT required for future modules

The exemplar notebook (`00-neural-networks/fundamentals/01_what_is_a_neural_network.ipynb`) only donates the **Reader Flow Blueprint**. Future modules author `source.md` fresh:

- **JAX (m07)** → Reference mode, from JAX docs.
- **Scaling Laws (m10a/m13)** → Reference mode, from arXiv papers via the `arxiv` MCP.
- Modules with neither → First-Principles mode, from the Learning Barrier Lens + Blueprint.

The Notebook Smoothness Gate is **skipped (N/A)** for these — never failed.

## State files (unchanged)

```text
sessions/_refactor/rollout_tracker.yaml
sessions/<module>/_refactor/manifest.yaml
```

## Spec

- Design + gates + rollout loop: `sessions/_refactor/v8_source_first_authoring_plan.md`
- Diagnosis + distilled rules: `sessions/_refactor/v8_exemplar_distillation_report.md`

## Philosophy

The author writes reader-flow. The compiler owns the invariants. The notebook is a yardstick, not a dependency.
