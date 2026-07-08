# What Changed in v2.1

## Diagnosis

v2 solved the first important problem:

> The sessions felt too dry and math-heavy.

It added:

- Coach Layer
- Math Ladder
- bilingual support
- artifact review
- curriculum refactor workflow

But the pilot revealed a second problem:

> Many playgrounds are still simulated terminal walkthroughs, not strong visual learning tools.

They are useful, but not always enough to create the “I can see it now” moment.

## v2.1 changes

### 1. Added `frontier-visual-auditor`

This skill reviews existing lesson visuals and scores whether they are good enough for concept understanding.

It classifies each lesson as:

- Visual mechanism
- Interactive calculator
- Terminal walkthrough
- Static explanation
- Missing visual

### 2. Strengthened `frontier-d3-visual-lab`

The skill now requires every visualization to answer one learning question and include:

- what you should notice
- misconception note
- where the visual lies / simplifies reality
- controls only when they teach something

It explicitly warns that terminal-style playgrounds are not enough for abstract math and systems concepts.

### 3. Strengthened `frontier-curriculum-refactor`

Rollout should now include a visual audit before batch edits.

Each batch should identify:

- which lessons already have acceptable visuals
- which need interactive calculators
- which need D3/SVG mechanism visuals
- which should remain text/code focused

### 4. Strengthened `frontier-lesson-humanizer`

When a lesson has a terminal-only playground, the humanizer should flag whether it needs a visual upgrade instead of silently accepting it.

## Recommended next workflow

```text
/frontier-visual-auditor
→ /frontier-d3-visual-lab for the top 1–2 hard concepts
→ /frontier-curriculum-refactor for batch rollout
```

## Definition of a good visualization

A good visualization should help the learner answer:

1. What is moving, growing, routing, shrinking, or transforming?
2. Which variable controls it?
3. What should I notice?
4. What misconception does this prevent?
5. What frontier-lab decision does this affect?
