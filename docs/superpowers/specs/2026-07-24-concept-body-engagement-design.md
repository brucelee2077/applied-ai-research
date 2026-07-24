# Concept-Body Engagement — Design

**Date:** 2026-07-24
**Branch:** `build/capability-spiral`
**Status:** Approved (design shape confirmed via AskUserQuestion: full toolkit + full author-loop rebuild)

## Problem

Lesson **intros/heroes** are now engaging — warm voice, curiosity hooks, the 4-beat
analogy scaffold, "normalize confusion," victory laps — and these are *enforced* for
the concept opening. But the **body of each concept** (the "build-up": mechanism, math,
worked example that follows the intro/analogy) is still, in the user's words, "hard to
digest, tedious, and a little bit boring." There is an **engagement cliff** right after
the intro closes.

### Root cause (from the 4-reader diagnosis, evidence-grounded)

Engagement is prescribed and enforced **only for the opening**; the build-up gets only
*structural* treatment. Three concrete gaps:

1. **The builder skill goes silent on body voice.** `frontier-lesson-builder/SKILL.md`
   has a "Beginner Intuition Register" (5 named beats) governing the *intro*. For the
   build-up it prescribes only structure — "visualize it," "narrate symbols," "math
   restraint," "show dimensions," "one idea per equation." No rules on keeping the body
   *engaging*. Result: bodies come out **correct but cold**.

2. **The judges never grade body voice.** The interest & tone judges score the *whole
   lesson*, so a brilliant hero carries a tedious body. The concept-structure judge
   grades four axes — `intuition_first` / `analogy` / `buildup` / `buildup_visualized` —
   all of which certify the build-up is *present, followable, and visualized*, but
   **none grades whether the body PROSE reads engagingly** (voice, re-hook, narration,
   discovery).

3. **The real bodies confirm it.** After the warm intro, bodies switch to a flat
   textbook register: the spine analogy is dropped once mechanism starts (m02 d05
   backprop, m03 d03 attention), worked examples are cold symbol-pushing, and 3–5 dense
   blocks run with no re-hook / "why this bites" / discovery beat. There is also **dead
   scaffolding** (`.build` / `.build-step` scroll-reveal CSS + a `__revealBuild()` call
   wired to nothing).

This is the same *shape* as the interest-floor hole fixed earlier: engagement existed
but was **unenforced where it mattered**.

## Goal

A beginner should find each concept's **body** as engaging and digestible as its intro —
and this must be **skill-driven and judge-enforced** so it reproduces on every future
module, not hand-edited one-offs.

## Design — four layers (mirrors the interest-floor fix)

### Layer 1 — Builder skill: the "Build-Up Register"

Add a **"Build-Up Register (per concept — keep the body alive)"** section to
`frontier-lesson-builder/SKILL.md` (and the v8 mirror tree), the body-side twin of the
existing "Beginner Intuition Register." Named body beats the author must hit:

1. **Bridge from the analogy** — the build-up's first move re-invokes the opening
   analogy ("let's go back to the ruler and watch what happens when we stack two…"),
   never a cold switch into mechanism.
2. **Re-hook / "why this bites"** — a one-line stakes beat *inside* the body so the
   mechanism answers a live question ("this collapse is sneaky — the net runs fine but
   secretly does nothing").
3. **Narrate the architecture, not the steps** — causal connectors (*therefore / which
   means / that's why*) and semantic step names ("the collapse", "the fix"), never bare
   "Step 1 / Step 2 / Step 3" enumeration or silent symbol-pushing.
4. **Predict-then-reveal** in worked examples — invite a guess before the number.
5. **Micro-recap + breath** — a one-line "so we've shown that…" landing after a dense run.
6. **Normalize struggle + a victory lap DURING the hard part** — not only in the intro.
7. **Keep the spine analogy alive to the end of the unit**, and mark where it breaks down.

Plus a **P0 rule (body-cold):** mark P0 if a concept's build-up drops voice/analogy into
a flat mechanism dump (Step 1/2/3, symbol-pushing) with no re-hook, no narration, no
discovery. Point the rule at the `body_engagement` judge axis (Layer 2) as its gate.

### Layer 2 — Judge: a standalone `body_engagement` judge (per concept)

**Its own judge, not a 5th axis on the structure judge.** The structure judge already
emits 4 axes + note + fix across ~13 concepts and its file comments warn it truncated at
`max_tokens` when the 4th axis was added (the salvage path silently drops the *last*
concept's verdict). Piling a 5th axis on would re-open that hazard. Instead add
`judge_body_engagement(lesson_text, concept_titles)` in `coverage_judge.py`, built on the
existing **`_chat()` mock-seam** (like `judge_interest_absolute`) so it is unit-testable
without a network and has its own token budget.

It scores each concept's **build-up prose** (not structure): does the body keep the spine
analogy alive, re-hook "why this matters," narrate the mechanism with voice/causal
connectors (vs flat "Step 1/2/3" enumeration or silent symbol-pushing), invite
prediction/discovery, and normalize struggle?

- Scale per concept: `GOOD | WEAK | MISSING | NA`.
- **`MISSING` is reserved for a genuinely COLD body** — a mechanism/symbol dump with none
  of the register beats. A body with voice via *any* beat is GOOD/WEAK. This mirrors the
  interest floor (a single weak lever is tolerable); it never requires padding or added
  length — a cold body is always fixable by adding voice, not words.
- **`NA` (never penalized), with explicit triggers spelled out in the prompt:** the recap
  unit (`tag="Recap"`), or a concept whose body is a single narrated definitional line
  with no real build-up to make cold.
- `run_from_paths` adds `'body_engagement': judge_body_engagement(...)`; new CLI section
  "Concept Body Engagement".
- Wire into `lesson_build.js` as a **new `body` lens**: `MISSING → P0`, `WEAK → P1`
  (kind `body_engagement`). Add `body_engagement` to `JUDGE_SCHEMA.kind`'s enum
  description. This is the hard floor the user asked for; **`MAX_ROUNDS` is the existing
  backstop** — a dense day that can't clear stops at the round cap as a judge-flagged
  checkpoint (the lesson still compiles), never an infinite loop.

### Layer 3 — Widgets: the full body toolkit

Give authors the *tools* to express body engagement, not just the instruction. Chosen to
minimize donor churn (reuse existing CSS / inline styles where possible):

1. **`%%% insight`** — an inline "why this matters / notice this" re-hook callout. Body is
   prose. Renders reusing the existing `.takeaway` class (💡 icon) — **no donor CSS change**.
2. **Predict-then-reveal for `%%% demo`** — new optional body field `predict:`. When
   present, render a "🤔 Predict first: …" prompt above the code (inline-styled). **When
   absent, output is byte-identical** to today's demos (guards the recompile-idempotence
   and any shipped-output snapshot test).
3. **`%%% steps`** — a narrated stepped worked-example. Body = repeated `step:` (the work)
   + `why:` (plain-English gloss) pairs. Renders using the **existing dead `.build` /
   `.build-step` / `.build-num` / `.build-note` CSS** already in the donor — a `.build`
   wrapper containing one `.build-step` per pair, each with a `.build-num` badge and a
   `.build-note` (work + gloss). `render_widget` MUST gain a `steps` case (it raises
   `ValueError` on unknown types today — no silent fallback). Body run through `inline()`.
   - **Gate fix (required, or the hard gate bounces the loop):** `concept_structure_gate.py`
     strips every `%%%…%%%` widget before measuring build-up prose (`_MIN_PROSE=40`), so a
     concept whose build-up is *only* a `%%% steps` block would FAIL "has build-up after
     its visual" — a hard gate that short-circuits `lesson_build.js` back to the author.
     Fix: the build-up floor is satisfied if there is ≥40 chars of build-up prose **OR** a
     build-up widget (`%%% steps` / `%%% demo` / `%%% mathladder`, or a 2nd `svg`/`viz`)
     appears after the opening visual. A `%%% steps` block IS substantial build-up content.
4. **Wire the dead scroll-reveal — a NEW multi-container implementation** (not a port). The
   `__revealBuild()` that ships in other (v8) shells is hardwired to a single
   `#build`/`#s5` container and couples `.gotit` unlock to scrolling — wrong for v9, where
   `%%% steps` can appear multiple times, inside arbitrary concept sections, and must not
   gate `.gotit`. Write a fresh `window.__revealBuild()` in `v9-base.donor`:
   `querySelectorAll('.build')`, arm each independently, observe its `.build-step`s with an
   IntersectionObserver, add `.revealed` on scroll-in; **graceful** — if reduced-motion or
   no IntersectionObserver, reveal all (never arm). Decoupled from `.gotit`/`#s5`. This
   makes the already-present nav-click call to `__revealBuild()` real. A jsdom test asserts
   arm→reveal marks every `.build-step` `.revealed`.

**Donor-change consequence:** adding `__revealBuild()` changes the donor's embedded JS, so
**every concept lesson using `v9-base.donor` must be recompiled** to stay consistent
(content unchanged; only the donor JS grows). This **re-baselines `test_v8_regression.py`**,
which (despite its name) recompiles the v9 concept lesson `m02-the-neuron/day-03/source.md`
and asserts byte-equality with its shipped `lesson.html` — it fails transiently until
day-03 is recompiled, then re-passes; not a regression. m02+m03 are rebuilt anyway; the
other concept modules (m06, m07, …) get a mechanical recompile pass. Verified safe by
recompiling one non-target concept module first (gates re-run on identical content).

### Layer 4 — Content rerun: full author-loop rebuild of m02 (9) + m03 (5)

Rebuild each of the 14 days through `lesson_build.js` under the new Build-Up Register +
`body_engagement` floor, with the new widgets available. **Frozen invariants preserved**
(quest-ids, nav chain, module labels, spine, `notebook_yardstick`) via `args.frozen`.
The `lesson_build.js` author prompt gains a "keep the body alive" paragraph pointing at
the register + new widgets, mirroring the existing inlined structural rules. Bounded
per-day workflow (one agent per day) with an independent verification pass — the proven
pattern from the interest rerun.

## Non-goals / preserved constraints

- **Length is a feature** — never cut coverage to make a body livelier; interleave voice,
  re-hooks, and widgets instead.
- **Cost is no object** — judges run every round.
- No changes to quest-ids, nav, localStorage keys, or the 7-section shell shape.
- Intros are already good — this design does not touch the Beginner Intuition Register
  except to add its body-side twin.

## Testing (TDD)

- `test_insight_widget.py` — `%%% insight` renders the re-hook callout (`.takeaway`, 💡),
  body run through `inline()` (so `[[term||gloss]]`/`**bold**` work).
- `test_demo_predict.py` — `predict:` renders the prompt **and** a real pre-change demo
  fixture (no `predict:`) is byte-identical to the pre-change output (regression diff).
- `test_steps_widget.py` — `%%% steps` renders `.build` > `.build-step` (each with
  `.build-num` + `.build-note`); `render_widget('steps',…)` no longer raises.
- `test_concept_structure_buildup_widget.py` — a concept whose build-up is only a
  `%%% steps` block PASSES the build-up floor (the gate fix).
- `test_body_engagement.py` — `judge_body_engagement` mocks `_chat`, returns per-concept
  GOOD/WEAK/MISSING/NA, defaults safely on parse error, and its prompt names the NA
  triggers; CLI prints a "Concept Body Engagement" section.
- Donor: a jsdom check that `window.__revealBuild` is defined, arms every `.build`, and
  reveals every `.build-step` (and that reduced-motion / no-IO reveals all without arming).
- `lesson_build.js`: `JUDGE_SCHEMA.kind` enum includes `body_engagement`; a `body` lens
  routes MISSING→P0 / WEAK→P1.
- Regression: full compiler suite stays green (169 baseline; the 1 pre-existing
  m02-manifest failure is out of scope); `test_v8_regression.py` re-passes after the m02
  day-03 recompile; recompile idempotence holds; `test_no_dead_lab` stays 0.
- Verification on the 14 rebuilt lessons: `body_engagement` P0-clear on all 14; interest
  floor still `FLOOR_MET` on all 14; gates pass; idempotent recompile.

## Files touched

| File | Change |
|---|---|
| `.claude/skills/frontier-lesson-builder/SKILL.md` + v8 mirror | + Build-Up Register section, + body-cold P0 rule |
| `sessions/_compiler/gates/coverage_judge.py` | + standalone `judge_body_engagement` (via `_chat` seam) + `run_from_paths` + CLI section |
| `sessions/_compiler/gates/concept_structure_gate.py` | build-up floor satisfied by a build-up widget (`steps`/`demo`/`mathladder`/2nd svg-viz) |
| `sessions/_compiler/v8lib.py` | + `render_insight`, `render_steps`, `predict:` in `render_demo`, register both in `render_widget` |
| `sessions/_compiler/shells/v9-base.donor` | + new multi-container `window.__revealBuild()` scroll-reveal JS |
| `sessions/_compiler/workflows/lesson_build.js` | + `body` lens (body_engagement P0/P1), + `JUDGE_SCHEMA.kind` enum, + "keep the body alive" author paragraph |
| `sessions/_compiler/AUTHORING.md` | + document `%%% insight`, `%%% steps`, `demo predict:` |
| `sessions/_compiler/tests/*` | new tests above |
| `sessions/_compiler/tests/test_v8_regression.py` | re-baseline (m02 day-03 recompiles under new donor) |
| all concept `lesson.html` using `v9-base.donor` | recompile (m02+m03 rebuilt; others mechanical) |
| m02 (9) + m03 (5) `source.md` + `lesson.html` | full author-loop rebuild |
