---
name: frontier-refactor-qa
description: Use to validate autonomous rollout loops, seed stabilization, learning experience, coverage, visual/evidence, artifacts, manifest/tracker state, and goal completion.
---

# Frontier Refactor QA

You are the strict quality gate and rollout-state validator.

> **Authoritative V9 authoring grammar + compile/gate loop:**
> `sessions/_compiler/AUTHORING.md` (source of truth — matches the shipped compiler).
> Validate authored `source.md` against it, not the obsolete
> `v8_source_first_authoring_plan.md`.

## State files

Use both:

```text
sessions/_refactor/rollout_tracker.yaml
sessions/<module>/_refactor/manifest.yaml
```

## Autonomous QA Role

When the goal is to continue rollout, validate the whole pipeline:

- tracker read,
- seed propagation risk check,
- seed stabilization if required,
- target module rollout,
- manifest updates,
- tracker updates,
- reports,
- audit/check evidence.

Do not treat audit, fix, and rollout as separate user-driven tasks if the tracker provides enough information.

## Seed Propagation Risk Gate

A seed P1 is required-to-fix before rollout if:

- seed_module is true,
- issue affects template/structure,
- issue would be P0 for a fresh build under current skills,
- future modules are likely to copy it,
- targeted low-risk fix exists.

Record it as P1, not retroactive P0.

## Learning Experience Gate

Evaluate whether the lesson makes the learner want to continue.

Check:

- opening invitation,
- learner barrier named,
- curiosity hook,
- familiar human situation,
- mental picture before terms/formulas,
- narrative spine,
- artifact as discovery,
- staff depth as empowerment.

## P0 Learning Experience Findings

Mark P0 if:

- opening feels task-like or checklist-like,
- foundation lesson starts engineering-first,
- first technical term arrives before mental picture,
- analogy is decorative rather than structural,
- staff/frontier relevance appears before orientation,
- artifact feels like homework before curiosity,
- learner would reasonably prefer a beginner notebook for readability.

## P1 Learning Experience Findings

Mark P1 if:

- opening is acceptable but middle becomes dry,
- curiosity weakens after first section,
- staff lens is useful but abrupt,
- artifact is correct but not discovery-oriented,
- bilingual warmth is missing,
- shipped seed has a propagation-risk pattern.

## Manifest Update Gate

Every P0/P1/P2 finding must map to module manifest backlog.

Seed-stabilization findings should use IDs such as:

```text
SEED-LX-D1-PICTURE-ORDER
SEED-LX-D2-FORMULA-FIRST
SEED-LX-ARTIFACT-AS-DISCOVERY
```

## Tracker Update Gate

After every phase:

- update module status,
- update counts,
- record report paths,
- append rollout_history,
- keep current_target accurate.

## Reports

Reports must include:

- Scope
- Phase
- Tracker status
- Manifest status
- Seed Propagation Risk Gate
- Learning Experience Gate
- Coverage Traceability
- Visual/Evidence Gate
- Artifact Gate
- P0/P1/P2 findings
- Edits applied
- Audit/check evidence
- Recommendation

Recommendation:

```text
Pass
Pass with P1
Blocked
```

## v8 Gates (run on the COMPILED lesson)

### Reader Flow Gate (mechanical)

Check the compiled lesson against the blueprint:

- curiosity before any definition/formula,
- jargon glossary present; every term used after its gloss,
- mental picture before first technical term,
- spine analogy in ≥3 blocks including staff,
- artifact has predict + observe (discovery register),
- staff/frontier relevance after mechanism,
- every staff claim traces to an earlier block.

### Notebook Smoothness Gate (pilot / exemplar ONLY)

Only when a `notebook_yardstick` is set. Compare compiled reader flow to the notebook on the barrier axis (first move, picture-vs-term order, why-it-exists framing, goal framing). The compiled lesson must read at least as smoothly.

Skip (record `N/A`) for reference / first-principles modules. Never fail a module for lacking a notebook.

### No-Notebook Authoring Gate

For reference / first-principles modules:

- mode declared; a missing notebook is not a defect (Notebook Smoothness = N/A),
- reference mode: ≥1 authoritative source cited; staff claims trace to it,
- first-principles mode: a self-authored worked numeric example exists,
- Reader Flow + Staff Depth still pass.

### Shell Invariant Gate

- recompile is idempotent (byte-identical),
- quest-id, 7 sections, `data-target` links, DEMOS/BUILD/QS shapes, playground ≥3, quiz `q:4 o:16`, localStorage keys, `.fin`, nav intact,
- migration: diff compiled shell + interactive regions vs shipped `lesson.html` — no regression,
- backstop: `lesson_audit.py`, `nav_audit.py`, `node --check`, jsdom if available.

### Staff Depth Gate

- one named failure mode (silent-failure preferred), one trade-off, one grounded interview line, one diagnostic quiz question. Backstop: `staff_lens_audit.js`.

### V9 Concept Gates (mode: concept)

- Gate dispatch: compile_lesson.py branches on meta['mode']; concept mode runs reader_flow_gate (concept branch) + concept_shell_gate + notebook_smoothness. Non-concept lessons use the existing v8 path unchanged.
- concept_shell_gate (on compiled HTML): ≥3 concept units, each with a visual (<svg or .build-embed) + exactly one gotit; exactly one quiz section (4 questions) + one produce section (references experiment.py); sidebar nav parity (data-targets == section ids); idempotent recompile; no leaked markers.
- reader_flow_gate concept branch (on source): hero human-first / no frontier-pressure; every @@@ concept ships a visual (P0 if not); spine word across ≥3 blocks; produce discovery-framed.
- "concept unit without a visual" and "a concept's picture deferred to a later unit" are P0 for a fresh V9 build.

Spec: `sessions/_compiler/AUTHORING.md` (source of truth — matches the shipped compiler).
The older `sessions/_refactor/v8_source_first_authoring_plan.md` documents historical/obsolete
grammar (`:::` fences, `{{anchor}}`, `[[term:weight]]`) the compiler now rejects — do not author from it.
