---
name: frontier-refactor-qa
description: Use after curriculum edits to validate coverage, learning experience, coach voice, foundation framing, function family coverage, visual/evidence, artifacts, tracker/manifest state, and goal-loop completion.
---

# Frontier Refactor QA

You are the strict quality gate.

Do not let a technically correct lesson pass if it creates unnecessary learning barrier.

## State files

Use both:

```text
sessions/_refactor/rollout_tracker.yaml
sessions/<module>/_refactor/manifest.yaml
```

## Learning Experience Gate

Evaluate whether the lesson makes the learner want to continue.

Check:

- opening invitation,
- learner barrier named,
- curiosity hook,
- familiar human situation,
- mental picture before technical terms,
- narrative spine,
- warmth through technical sections,
- artifact as discovery,
- staff depth as empowerment.

## P0 Learning Experience Findings

Mark P0 if:

- opening feels task-like or checklist-like,
- foundation lesson starts engineering-first,
- first technical term arrives before a mental picture,
- analogy is decorative rather than structural,
- staff/frontier relevance appears before learner orientation,
- artifact feels like homework before curiosity,
- learner would reasonably prefer the reference notebook for readability.

## P1 Learning Experience Findings

Mark P1 if:

- opening is acceptable but middle becomes dry,
- curiosity weakens after first section,
- staff lens is useful but abrupt,
- artifact is correct but not discovery-oriented,
- bilingual warmth is missing.

## Coverage and rigor still matter

Do not let warmth hide missing depth.

A pass requires both:

```text
learning pull + technical depth
```

## Manifest Update Gate

Every P0/P1/P2 finding must map to the module manifest backlog.

Learning experience findings should use IDs such as:

```text
LX-D1-OPENING
LX-D2-NARRATIVE-SPINE
LX-D3-ARTIFACT-AS-DISCOVERY
```

## Tracker Update Gate

After QA/fix/eval, update rollout tracker status, counts, and report paths.

## Dynamic workflow pattern

Use subagents for:

1. Learning Experience Agent
2. Coverage Traceability Agent
3. Coach Voice/Foundation Framing Agent
4. Visual/Evidence Agent
5. Artifact/Anchor Consistency Agent
6. Manifest/Tracker Agent
7. Integrator/Adversarial Critic

## Held-out eval

If comparing to a notebook/reference, distinguish:

- content coverage,
- learning experience,
- runnable evidence,
- visual evidence,
- staff depth.

Do not over-credit a lesson because it is more advanced if it is less inviting.

## Reports

Reports must include:

- Learning Experience Gate
- Coverage Traceability
- Visual/Evidence
- Artifact Gate
- Staff Depth
- Manifest Update
- Tracker Update
- P0/P1/P2
- recommendation

Recommendation:

```text
Pass
Pass with P1
Blocked
```
