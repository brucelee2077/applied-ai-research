---
name: frontier-refactor-qa
description: Use after curriculum edits to validate coverage, coach voice, foundation framing, function family coverage, visual/evidence, artifacts, anchor consistency, rollout tracker state, module manifest state, legacy skill references, and goal-loop completion.
---

# Frontier Refactor QA

You are the strict quality gate.

Use both state files:

```text
sessions/_refactor/rollout_tracker.yaml
sessions/<module>/_refactor/manifest.yaml
```

## Tracker Update Gate

After QA, fix, held-out eval, or skill patch:

- update the module entry in `sessions/_refactor/rollout_tracker.yaml`,
- record status,
- record open P0/P1/P2 counts,
- record manifest path,
- record report paths,
- append rollout_history.

Do not leave rollout state only in prose reports.

## Manifest Update Gate

After QA, held-out eval, or skill patch:

- update `sessions/<module>/_refactor/manifest.yaml`,
- every P0/P1/P2 finding must map to a backlog item,
- every coverage finding must map to coverage_items,
- every visual/evidence finding must map to visual_evidence_items,
- every artifact finding must map to artifact_items,
- every systemic skill gap must map to skill_gap_candidates,
- report files must be recorded in loop_history.

## Goal-loop role

When `/goal` is active, produce evidence that the goal is or is not satisfied.

Do not claim the goal is met without:

- tracker status,
- manifest status,
- open P0 backlog count,
- report files created,
- P0 count shown,
- gate results shown,
- audit/test results shown,
- remaining P1/P2 listed.

## Dynamic workflow pattern

For substantial QA, use subagents:

1. Rollout Tracker Agent
2. Manifest Consistency Agent
3. Coverage Traceability Agent
4. Coach Voice/Foundation Framing Agent
5. Function Family Agent
6. Visual/Evidence Agent
7. Artifact/Anchor Consistency Agent
8. Legacy Skill Agent
9. Integrator/Adversarial Critic

## P0

- tracker missing during rollout loop
- manifest missing during module loop
- missing must-cover concept
- must-cover concept PARTIAL/FAIL without waiver
- formula-first opening
- P0 visual/evidence missing
- behavioral P0 concept lacks plotted/live evidence
- failure mode without evidence/artifact
- simulated output counted as evidence without reproducible artifact
- anchor inconsistency that teaches conflicting model
- stale run command
- Produce path mismatch
- legacy skill reference to uninstalled skill
- broken JS/nav/completion
- QA findings not reflected in manifest/tracker

## Coverage traceability table

For every must-cover concept:

| Concept | Required depth | Lesson evidence | Visual/evidence | Artifact | Result |
|---|---|---|---|---|---|

Result: PASS / PARTIAL / FAIL.

Named is not covered. A concept is covered only if the lesson explains it at the required depth and provides required evidence/artifact.

## Visual/Evidence Gate

Mark P0 if:

- P0 behavioral concept has no plotted/live evidence,
- P0 failure mode has no evidence/artifact,
- simulated output is counted as evidence without reproducible artifact,
- quantitative visual lacks required numeric readout.

## Anchor Consistency Gate

Check consistency across prose, visual, playground, quiz, Produce, and acceptance criteria.

## Reports

Create reports with:

- Scope
- Tracker status
- Manifest status
- Coverage traceability table
- Coach Voice Gate
- Foundation Framing Gate if applicable
- Function Family Gate
- Visual/Evidence Gate
- Anchor Consistency Gate
- Artifact Gate
- Tracker Update Gate
- Manifest Update Gate
- P0/P1/P2 findings
- Fixes applied if any
- Merge recommendation

Recommendation: Pass / Pass with P1 / Blocked.
