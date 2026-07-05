# Staff-lens teaching principle — design

## Problem

Comparing `attention-lab-course/modules/04-the-math.html` against built sessions (e.g.
`sessions/week-m9b/day-01-float-formats.html`) shows both already follow the Layer-1
scaffolding `TEACHING_PRINCIPLES.md` requires: analogy before math, "where the analogy
breaks," a symbol-labeled formula build-up, a worked numeric example, a production/relevance
callout.

Attention Lab goes further in ways the skill does not currently require:
1. Explicitly maps each analogy element to its matching math symbol (fader → attention
   weight, master trim → √d_k), not just an analogy paragraph.
2. Pairs a named **"Failure mode"** card with a named **"Trade-off"** card, and states the
   staff-engineer framing outright ("the wrong-axis bug is what a senior engineer catches in
   code review; the O(n²) trade-off is what they raise in a design review").
3. Quiz questions are diagnostic ("a teammate's code runs without errors, but training
   stalls — where do you look first?"), not recall ("what does symbol X mean?").

Built sessions have analogous pieces (playground demo, a "why X happens" explanation) but
never name the staff-engineer framing, and quiz questions test recall of the mechanism rather
than debugging-style reasoning.

## Decision

Close the gap on (2) and (3) — failure-mode/trade-off naming and diagnostic quiz questions —
as a new required rule. (1), the explicit symbol-to-analogy table, is not required; existing
prose-form analogy mapping is judged sufficient.

## Changes

### 1. `.claude/skills/frontier-session-coach/TEACHING_PRINCIPLES.md` — new "Staff Lens" rule

Add a new required block, sibling to the existing "Keep depth" rule:

- Every lesson with a Mechanism/math section (section 4, "机制/为什么") must include two
  explicitly named callouts:
  - **Failure mode** — a concrete way this breaks silently (no crash, wrong behavior),
    framed as "what a senior engineer catches in code review."
  - **Trade-off** — what you give up for what you gain, framed as "what gets raised in a
    design review."
- At least one of the day's 4 quiz questions must be **diagnostic** in style — "a
  teammate's code/run does X, where do you look first / why did this happen" — not pure
  term-recall.

This is the source of truth. `SESSION_TEMPLATE.md` and `SKILL.md` reference it rather than
duplicating the rule text.

### 2. `.claude/skills/frontier-session-coach/SESSION_TEMPLATE.md` and `SKILL.md`

- Section 4 description gains one sentence pointing at the Staff Lens rule in
  `TEACHING_PRINCIPLES.md`.
- Section 6 (Quiz) description gains one sentence: "at least one diagnostic-style question
  — see Staff Lens in `TEACHING_PRINCIPLES.md`."

### 3. `ROADMAP.md`

- **Global rules** section gains one line pointing at the new rule, dated, so a builder
  scanning the roadmap knows the bar exists without opening the skill folder.
- A new small subsection near the Build queue, **"Staff-lens retrofit watchlist"** — not a
  work package, just a deferred note: when a future session picks up a math-heavy module in
  Phase 0–4 (attention math, transformer arithmetic, scaling laws, MoE, quantization), it
  should check the already-built lesson against the Staff Lens rule and upgrade
  opportunistically. No blanket retrofit is scheduled now.

## Scope

- Enforced going forward for any newly authored/reused/converted lesson.
- No retroactive edits to the ~140 existing lesson files in this pass. The watchlist note in
  `ROADMAP.md` is the only trace that this may need revisiting later, and only for
  math-heavy modules not yet built.

## Out of scope

- Rewriting `lesson_audit.py` to mechanically check for the failure-mode/trade-off callouts
  or diagnostic quiz style — this is a qualitative writing bar, not a structural one, so it
  isn't a good fit for the existing regex-based structural audit.
- Symbol-to-analogy mapping table requirement (Attention Lab's item 1) — judged not worth a
  hard requirement; prose analogies with an explicit "where it breaks" line are sufficient.
