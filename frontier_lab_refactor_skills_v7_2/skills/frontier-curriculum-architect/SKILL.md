---
name: frontier-curriculum-architect
description: Use to design/refactor Frontier Lab curriculum modules from first principles, create contracts, and orchestrate goal-driven/dynamic workflow refactor loops.
---

# Frontier Curriculum Architect

You design complete learning experiences from first principles while preserving the existing sessions shell.

## Source-free default

Do not read old notebooks or prior courseware unless the user explicitly asks. Existing sessions are shell constraints and rough drafts.

## Goal-driven orchestration

For substantial refactors, prefer a `/goal` loop with a measurable completion condition.

A good goal condition must specify:

- target report files,
- required P0 count,
- required gate results,
- required audit/test checks,
- stop-after turn bound,
- what to do if blocked.

Example:

```text
/goal Module 2 reaches Pass or Pass with P1: sessions/m02_v7_2_qa_report.md reports 0 P0 findings, lesson_audit.py passes, nav/jsdom checks pass if available, and sessions/m02_v7_2_fix_report.md summarizes remaining P1/P2 only; stop after 8 turns if not met.
```

## Dynamic workflow rule

Use a dynamic workflow when a task benefits from parallel subagents:

- one subagent per lesson,
- one for coverage,
- one for coach voice,
- one for visual/evidence,
- one for artifacts,
- one adversarial critic,
- one integrator.

Do not use workflow for tiny single-file fixes.

## Required protocol

Before editing a module:

```text
1. Structural audit
2. Coverage discovery
3. Coverage contract
4. Coach voice contract
5. Visual/evidence contract
6. Artifact contract
7. Refactor plan
```

Then edit, QA, and loop until the active goal condition is satisfied or blocked.

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
10. Capability-Limits & Geometry Lens

## Capability-Limits & Geometry Lens

A coverage contract must not enumerate only what a mechanism *does* (its forward formula). For every core
mechanism it must also require:

- **Capability limit** — the defining thing the mechanism *cannot* do, taught as a must-cover or
  should-cover with evidence. (A single neuron cannot solve XOR / is limited to a linear boundary; one
  linear layer buys no expressivity; a small LR crawls forever.) The limit is often the whole reason the
  *next* concept exists — omitting it leaves the learner without the motivation for what follows.
- **Geometric interpretation** — what the parameters *do to a picture*, required as a **plotted/behavioral**
  visual (not prose): `w` orients a boundary, `b` shifts/thresholds it, an activation bends a line, a loss
  is a surface. If the contract names geometry, it must pair it with a plotted-boundary / plotted-curve
  visual requirement (coordinate with the visual contract's behavioral-visual rule), never prose-only.

When you write the coverage contract, add a row for the limit and a row (or visual-contract entry) for the
geometry. A contract that lists `output = f(w·x+b)` but not "what it can't do" and not "what w/b do to the
boundary" is incomplete.

## Foundation Framing Lens

For neural-network foundations, must-cover:

- brain-inspired intuition,
- artificial-neuron caveat,
- biological-to-artificial mapping,
- where analogy breaks,
- transition to math function.

Generic everyday analogies do not satisfy this.

## Function Family Lens

For mechanism families, require representative variants, comparison table, when-to-use guide, failure modes, and modern context.

For activations, expect sigmoid, tanh, ReLU, Leaky ReLU, softmax, derivative intuition, saturation, vanishing gradient, dying ReLU, and GELU/SwiGLU context unless explicitly out of scope.

## Anchor Consistency Lens

Track step counts, seeds, split ratios, learning rates, dataset sizes, sequence lengths, batch sizes, parameter values, metric definitions, and model dimensions across prose, visuals, playgrounds, quiz, Produce, and acceptance criteria.

## Done

A refactor passes when the active goal condition is met, not when the lesson merely looks better.
