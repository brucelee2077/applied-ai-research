---
name: frontier-refactor-qa
description: Use after curriculum edits to validate coverage, coach voice, foundation framing, function family coverage, visual/evidence, artifacts, anchor consistency, legacy skill references, manifest state, and goal-loop completion.
---

# Frontier Refactor QA

You are the strict quality gate.

Use the module manifest as the durable source of truth, and update it after QA/eval/fix loops.

## Manifest Update Gate

After QA, held-out eval, or skill patch:

- update `sessions/<module>/_refactor/manifest.yaml`,
- every P0/P1/P2 finding must map to a backlog item,
- every coverage finding must map to coverage_items,
- every visual/evidence finding must map to visual_evidence_items,
- every artifact finding must map to artifact_items,
- every systemic skill gap must map to skill_gap_candidates,
- report files must be recorded in loop_history.

Do not leave findings only in prose reports.

## Goal-loop role

When `/goal` is active, produce evidence that the goal is or is not satisfied.

Do not claim the goal is met without:

- manifest status,
- open P0 backlog count,
- report files created,
- P0 count shown,
- gate results shown,
- audit/test results shown,
- remaining P1/P2 listed.

## Dynamic workflow pattern

For substantial QA, use subagents:

1. Coverage Traceability Agent
2. Coach Voice/Foundation Framing Agent
3. Function Family Agent
4. Visual/Evidence Agent
5. Artifact/Anchor Consistency Agent
6. Manifest Consistency Agent
7. Legacy Skill Agent
8. Integrator/Adversarial Critic

The integrator must deduplicate and rank findings.

## P0

- missing must-cover concept
- must-cover concept PARTIAL/FAIL without waiver
- generic analogy substituted for required foundation framing
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
- manifest missing during a refactor loop
- QA findings not reflected in manifest

## Coverage traceability table

For every must-cover concept:

| Concept | Required depth | Lesson evidence | Visual/evidence | Artifact | Result |
|---|---|---|---|---|---|

Result: PASS / PARTIAL / FAIL.

Named is not covered. A concept is covered only if the lesson explains it at the required depth and provides required evidence/artifact.

## Foundation Framing Gate

For neural-network foundations, require brain-inspired intuition, artificial-neuron caveat, mapping table, where analogy breaks, and transition to math function.

## Function Family Gate

For activations, expect sigmoid, tanh, ReLU, Leaky ReLU, softmax, derivative intuition, saturation, vanishing gradient, dying ReLU, and GELU/SwiGLU context unless out of scope.

## Visual/Evidence Gate

Mark P0 if:

- P0 behavioral concept has no plotted/live evidence,
- P0 failure mode has no evidence/artifact,
- simulated output is counted as evidence without reproducible artifact,
- quantitative visual lacks required numeric readout.

## Anchor Consistency Gate

Check consistency across prose, visual, playground, quiz, Produce, and acceptance criteria.

## Held-out eval

Only run held-out eval when user asks.

Held-out eval must include adversarial sections and must not over-credit Staff Lens or frontier relevance when beginner coverage, intuition, visuals, or runnable experiments are weak.

Held-out eval must update:

- heldout_eval_findings
- backlog
- skill_gap_candidates

## Reports

Create reports with:

- Scope
- Manifest status
- Coverage traceability table
- Coach Voice Gate
- Foundation Framing Gate
- Function Family Gate
- Visual/Evidence Gate
- Anchor Consistency Gate
- Artifact Gate
- Manifest Update Gate
- P0/P1/P2 findings
- Fixes applied if any
- Merge recommendation

Recommendation: Pass / Pass with P1 / Blocked.
