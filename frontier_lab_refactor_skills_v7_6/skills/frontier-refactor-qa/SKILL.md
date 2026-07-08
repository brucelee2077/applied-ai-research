---
name: frontier-refactor-qa
description: Use to validate autonomous rollout loops, seed stabilization, learning experience, coverage, visual/evidence, artifacts, manifest/tracker state, and goal completion.
---

# Frontier Refactor QA

You are the strict quality gate and rollout-state validator.

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
