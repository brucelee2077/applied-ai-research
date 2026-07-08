---
name: frontier-lesson-builder
description: Use to build or stabilize lessons so learning experience, mechanism, and staff-depth are integrated rather than checklist-driven.
---

# Frontier Lesson Builder

Your job is to make the learner want to continue while still building staff-level depth.

## Non-negotiables

No formula before felt intuition.

No checklist before invitation in foundation lessons.

No decorative analogy. Use a narrative spine.

## Three-layer architecture

```text
1. Experience Layer
   reduce fear, build curiosity, familiar human situation, mental picture

2. Mechanism Layer
   definitions, formula, step-traced example, visual/evidence, implementation

3. Staff Depth Layer
   failure mode, scale implication, debugging signal, artifact, interview answer
```

## Seed Stabilization Writing Rule

When stabilizing a seed module, do not over-polish.

Only rewrite enough to prevent a bad pattern from propagating.

Allowed targeted changes:

- move mental picture before technical vocabulary,
- reframe section openings from task/checklist to curiosity,
- carry an analogy through mechanism/staff/artifact sections,
- reframe Produce as discovery,
- delay frontier/staff pressure until learner orientation.

Not allowed unless explicitly needed:

- broad redesign of all lessons,
- unrelated prose polish,
- shell/navigation/quest-id changes,
- changing artifact requirements without QA reason.

## Learning Barrier Gate

Mark P0 if:

- opening feels like work before wonder,
- first technical term appears before mental picture,
- foundation lesson starts engineering-first,
- staff/frontier relevance arrives before orientation,
- artifact feels like homework before curiosity.

Mark seed-stabilization P1 if this issue exists in a seed module and is likely to be copied forward.

## Artifact as discovery

Produce should feel like:

```text
Run this and observe something surprising.
```

not:

```text
Submit this homework.
```

Acceptance criteria should include observation and explain-back.

## Staff courage rule

Staff depth should feel empowering:

```text
You can understand the small mechanism, then scale the reasoning up.
```

## v8 — author in source, not HTML

Under v8 you write `source.md` in reader-flow order; a compiler emits `lesson.html`.

Reordering reader flow is moving a block, not splicing HTML. Do not hand-edit `lesson.html`.

## Reader Flow Blueprint

Author `source.md` in this order (the compiler maps it onto the shell):

```text
hook → jargon buster → human situation → spine analogy → mental picture →
worked example → mechanism → frontier payoff → failure/misconception →
artifact-as-discovery → checkpoint → takeaways
```

## Jargon Ladder Rule

Front-load a plain-English gloss of every term used today.

Then introduce each term once, defined-before-use. No term appears before its gloss.

This is the checkable form of "no formula before felt intuition."

## Curiosity-before-Rigor Rule

First contact is wonder — never a definition or a prerequisites checklist.

Prerequisites read as "just curiosity."

## Staff-Depth-after-Orientation Rule

Failure modes, trade-offs, scale, and frontier relevance appear only after hook → picture → mechanism.

Frame depth as empowerment, not pressure. Every staff/interview claim traces to something taught earlier.
