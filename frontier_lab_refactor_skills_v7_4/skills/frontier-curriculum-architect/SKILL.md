---
name: frontier-curriculum-architect
description: Use to design/refactor Frontier Lab curriculum modules from first principles, create/update rollout trackers and module manifests, and orchestrate goal-driven/dynamic workflow rollout loops.
---

# Frontier Curriculum Architect

You design complete learning experiences from first principles while preserving the existing sessions shell.

## Source-free default

Do not read old notebooks or prior courseware unless the user explicitly asks or the goal enters a held-out eval phase.

Existing sessions are shell constraints and rough drafts.

## Rollout Tracker Rule

Every curriculum rollout should maintain a repo-level tracker:

```text
sessions/_refactor/rollout_tracker.yaml
```

The rollout tracker is the curriculum-wide source of truth for:

- current target
- module order
- module statuses
- module manifest paths
- global constraints
- default success gates
- report paths
- open P0/P1/P2 counts
- skill version
- rollout history
- skill history

Before each module rollout:

1. read the tracker if it exists,
2. create it if missing,
3. ensure the target module has an entry,
4. set `current_target`,
5. create/read the module manifest,
6. derive the goal loop from the tracker defaults.

After each module rollout:

1. update module status,
2. update open P0/P1/P2 counts,
3. record report paths,
4. append rollout_history,
5. update current_target or suggested next target if obvious.

Do not leave rollout state only in prose reports.

## Minimal user command behavior

If the user says:

```text
Roll out sessions/m03-attention.
```

or:

```text
Roll out the next module.
```

Use the rollout tracker to expand this into the full goal loop.

Ask a clarification only if:
- no tracker exists and no module can be inferred,
- multiple equally valid targets exist,
- the requested module path does not exist.

## Module Refactor Manifest Rule

Every module refactor must maintain:

```text
sessions/<module>/_refactor/manifest.yaml
```

The module manifest is the durable module source of truth for:

- constraints
- quality gates
- coverage items
- visual/evidence items
- artifact items
- backlog
- held-out eval findings
- skill-gap candidates
- loop history

## Goal-driven orchestration

For substantial refactors, prefer a `/goal` loop with a measurable completion condition.

A good goal condition must specify:

- tracker path,
- manifest path,
- target module,
- report files,
- required P0 count,
- required gate results,
- required audit/test checks,
- stop-after turn bound,
- what to do if blocked.

## Dynamic workflow rule

Assume dynamic workflow is enabled if the user says so.

Use subagents for:

- rollout tracker reconciliation
- coverage traceability
- coach voice / foundation framing
- function family coverage
- visual/evidence
- artifacts / anchor consistency
- held-out eval
- adversarial integration

## Required protocol

Before editing a module:

```text
1. Read/create rollout tracker
2. Read/create module manifest
3. Structural audit
4. Coverage discovery
5. Update manifest coverage items
6. Update manifest visual/evidence items
7. Update manifest artifact items
8. Update manifest backlog
9. Update rollout tracker status to in_progress
10. Refactor plan
```

Then edit, QA, update manifest, update tracker, and loop until the active goal condition is satisfied or blocked.

## Coverage Engine

Derive coverage using:

1. Concept Anatomy Lens
2. Mechanism Chain Lens
3. Variant / Family Lens
4. Foundation Framing Lens
5. Failure Mode Lens
6. Empirical Evidence Lens
7. Systems / Research Relevance Lens
8. Interview Readiness Lens
9. Anchor Consistency Lens
10. Capability Limits & Geometry Lens

## Done

A rollout passes when:

- active goal condition is met,
- module manifest status is pass or pass_with_p1,
- open P0 backlog is empty,
- audit/test evidence is reported,
- module findings are reflected in the manifest,
- rollout tracker records final status and reports.
