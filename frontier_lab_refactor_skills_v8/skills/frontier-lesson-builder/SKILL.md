---
name: frontier-lesson-builder
description: Use to build or stabilize lessons so learning experience, mechanism, and staff-depth are integrated rather than checklist-driven.
---

# Frontier Lesson Builder

Your job is to make the learner want to continue while still building staff-level depth.

> **Authoritative V9 authoring grammar + compile/gate loop:**
> `sessions/_compiler/AUTHORING.md` (source of truth — matches the shipped compiler).
> Author every `source.md` from that file, not from memory or the obsolete
> `v8_source_first_authoring_plan.md`.

## Non-negotiables

No formula before felt intuition.

No checklist before invitation in foundation lessons.

No decorative analogy. Use a narrative spine.

## Beginner Intuition Register (per concept — the notebook's tone)

Match the register of a strong beginner notebook: warm, plain-language, analogy-first, heavy on *why*. A staff-depth lesson is NOT an excuse to go straight into mechanism — the intuition layer must carry every concept, not just the hero. This is the difference between "correct but cold" and a lesson a beginner actually wants to read. The eval gate for this is the **beginner-friendliness judge** (`coverage_judge.py` tone axis), which grades the lesson against the notebook on warmth / analogy_quality / intuition_depth / plain_language / curiosity / pace — drive it to MATCHES_NOTEBOOK.

Every concept unit opens with **felt intuition before any formula, range, or notation**, in this order:

```text
1. Plain-words "what is it"   one sentence a 12-year-old gets, no symbols
2. Full analogy SCAFFOLD      not a one-word metaphor. Give all four beats:
                              (a) a concrete everyday thing (switch, dimmer, valve),
                              (b) what the analogy gets RIGHT,
                              (c) the concept restated in plain words,
                              (d) where the analogy BREAKS DOWN (one sentence).
3. Why it matters             the problem it solves, one line
4. THEN the mechanism         narrate EVERY symbol in words before/beside the formula
                              ("1 over 1 plus e-to-the-minus-z" — not a bare glyph); its own visual
5. When to use it             plain-language pros & cons — the notebook's "When to Use This"
```

**Warmth & encouragement (sustained, not just the hero):**
- Voice = a brilliant friend rooting for the reader, never an exam or a textbook. Prefer "let's…", "you", "notice".
- Normalize confusion: when something is hard, SAY so ("this part trips almost everyone up the first time — that's normal"). Do NOT frame the learner as error-prone.
- Victory laps: after a hard idea, tell the reader what they just unlocked ("you just discovered why depth needs a bend").
- Keep required staff/interview depth, but frame it as empowerment ("now you can see why…"), and let it FOLLOW the warm intuition — never lead with "this is the staff-level part" or "say this in an interview" as the concept's opening move.

**Plain language & pace:**
- Short sentences, one idea each. Break run-ons. Active voice. No idioms.
- Front-load a Jargon Ladder (`%%% jargon`) glossing every scary term the day uses (saturation, sparsity, XOR, …) BEFORE those terms appear — mirror the notebook's Jargon Buster.
- Let ideas breathe: a short recap or a one-line "so far…" between big concepts; don't pour them out in one stream.

Mark **P0 (intuition-inverted)** if a concept opens with a formula, a range like `(0,1)`, `max(0,z)`, or notation before the felt picture; if a concept is all mechanism with no analogy and no "why"; if an analogy is given WITHOUT its "where it breaks down" beat; or if the dominant voice is interview/textbook rather than warm-beginner. The inline visual anchors the *intuition*, not the algebra. See repo `CLAUDE.md` §5 (analogy scaffold, simple words, no idioms) and §7 (curiosity hook, normalize confusion, victory laps).

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
- a concept unit that introduces a visual object (curve, distribution, boundary, shape) but is text-only (no inline visual),
- a concept's picture deferred to a later unit, or all visuals dumped in one late "build" section.

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

## Reader Flow Blueprint (v9 — concept-driven)

A lesson body is a sequence of **concept units**, not a fixed template. Author `source.md` (mode: concept) as:

    hero (curiosity hook)
    → concept unit 1  (plain-words intro → its OWN inline visual → build-up)
    → concept unit 2  (…)
    → … as many concept units as the topic needs
    quiz (one section, all questions)
    produce (discovery artifact)
    fin

Each concept unit MUST carry its own inline visual, placed immediately after the intro and before the build-up. Never defer a concept's picture to a later unit or to one shared "build" section. Front-load a jargon gloss; introduce each term defined-before-use. Keep the narrative spine word running through hero + most concepts.

WITHIN a single concept unit, order the beats like this (this is unit-level ordering, not fixed body sections):

```text
plain-words intro → inline visual (its own) → worked example → mechanism →
frontier payoff → failure/misconception → build-up
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

## V9 authoring essentials (full grammar in `sessions/_compiler/AUTHORING.md`)

Required front-matter for a `mode: concept` lesson:

```yaml
quest_id: <frozen id>       # replaces the donor __QUEST_ID__ token
mode: concept
donor: v9-base.donor
module_label: "…"           # KeyError if missing; drives sidebar group + hero kicker
title: "…"                  # KeyError if missing
subtitle: "…"               # KeyError if missing
brand_sub: "…"              # set it or the donor's old brand line leaks
spine: "<one word>"         # narrative-spine word; gate needs it in >=3 blocks
page_title / nav_prev_* / nav_next_* / fin_title / fin_body   # optional
notebook_yardstick: <path or null>   # null => Notebook Smoothness gate is N/A
```

Compile + iterate:

```bash
python3 sessions/_compiler/compile_lesson.py <source.md>              # exit 0 = pass
python3 sessions/_compiler/compile_lesson.py <source.md> --check-only # run gates, write nothing
# exit codes: 0 pass · 2 reader-flow fail (nothing written) · 3 shell/concept gate fail · 1 usage/parse error
```

Every concept unit must ship a real visual (`%%% svg` with a closed `<svg>…</svg>`, or
`%%% viz src=…` pointing at an EXISTING `sessions/viz/*.html`) or the gate FAILS. See
AUTHORING.md for the full block grammar, all `%%%` widgets, callouts, and a minimal
compiling example.
