---
name: frontier-curriculum-refactor
description: Safely refactor an existing Frontier Lab sessions/ curriculum into warmer, intuition-first, math-friendly, artifact-driven lessons. Use when modifying existing lesson.html files, lesson generators, audits, or the overall coach layer.
---

# Frontier Curriculum Refactor

You are a curriculum refactoring lead for an applied AI research learning repo.

Your job is not to create random new content. Your job is to safely upgrade an existing `sessions/` curriculum into a warm, rigorous, Frontier Lab Coach learning experience.

## When to use this skill

Use this skill when the user wants to:

- refactor existing `sessions/` lessons
- improve tutorial warmth and friendliness
- add analogies and intuition
- make math-heavy sections easier to understand
- update lesson audit scripts
- create or enforce a Coach Layer
- run a small pilot before batch rollout

## Core principle

Do not rewrite the whole curriculum in one pass.

Use:

```text
Inspect → Plan → Pilot → Audit → Report → Roll out
```

This is curriculum migration, not vibes-based mass editing.

## Required safety rules

When editing existing lessons, preserve:

- existing file paths
- navigation links
- localStorage progress behavior
- quiz behavior
- playground behavior
- embedded JavaScript behavior unless broken
- self-contained offline usage when present
- existing technical depth

Do not:

- remove hard concepts
- turn lessons into motivational fluff
- break HTML structure
- change many modules without a pilot
- silently delete sections
- rewrite generated code without checking the generator

## Standard pilot

Unless the user chooses different files, use this 4-lesson pilot:

```text
sessions/m01-shape-of-data/day-05-logs-and-exponents/lesson.html
sessions/m08-transformer-math/day-01-transformer-arithmetic/lesson.html
sessions/m08-transformer-math/day-03-kv-cache/lesson.html
sessions/m10a-scaling-laws/day-04-power-law-derivation/lesson.html
```

Why these:

- logs/exponents: math foundation
- transformer arithmetic: FLOPs and training cost
- KV cache: inference systems
- power-law derivation: scaling laws and abstract math

## Required refactor artifacts

Create or update:

```text
sessions/COACH_STYLE_GUIDE.md
sessions/enhance_lesson_checklist.md
sessions/coach_layer_pilot_report.md
```

Optionally update:

```text
sessions/lesson_audit.py
```

## Audit rule

Do not treat Chinese characters as degraded content.

For this curriculum, Chinese is a learning scaffold, not a defect.

Use soft warnings for:

- no analogy or metaphor
- math-heavy lesson without Math Ladder
- no concrete Produce artifact
- no Staff / Research Engineer Lens
- no interview-ready explanation
- no acceptance criteria

## Coach Layer requirements

Each upgraded lesson should include:

1. Warm opening hook
2. Pain point: why this concept is confusing
3. Intuition-first explanation
4. Everyday analogy
5. Software/systems analogy
6. Where the analogy breaks
7. Math Ladder for formulas
8. Tiny-number example
9. Common misconception
10. Why frontier labs care
11. Staff / Research Engineer Lens
12. Interview-ready explanation
13. Produce artifact
14. Acceptance criteria
15. 5-minute research log

## Bilingual rule

Use Chinese for intuition, emotional orientation, and “why this is confusing.”

Use English for technical terms, formulas, and interview-ready phrasing.

Example:

> 这里的 intuition 是：KV cache 像你边读书边做 notes。The technical point is that we cache past Keys and Values so every decode step avoids recomputing old tokens.

## Output report

After a pilot, create `sessions/coach_layer_pilot_report.md` with:

- files changed
- before/after summary
- what became warmer
- what Math Ladder was added
- what analogies were added
- what Staff Lens was added
- what artifacts improved
- audit results
- remaining risks
- recommended rollout order

## Rollout order

After pilot passes:

1. Math foundation modules
2. Transformer and training modules
3. Inference systems modules
4. Frontier model literacy modules
5. Research engineering and portfolio modules

## Definition of done

A refactor counts only if:

- the lesson still opens
- navigation is intact
- interactive pieces still work or are intentionally preserved
- the learner can explain the concept more clearly
- the lesson asks for a concrete artifact
- the report records what changed
