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

### Layer 2 — Judge: `body_engagement` axis (per concept)

Add a **fifth axis** to the Concept-Structure Judge (`coverage_judge.judge_concept_structure`),
graded per concept so a strong hero can't carry a dead body. It scores the **build-up
prose** (not structure): does the body keep the spine analogy alive, re-hook "why this
matters," narrate the mechanism with voice/causal connectors (vs flat enumeration or
symbol-pushing), invite prediction/discovery, and normalize struggle?

- Scale: `GOOD | WEAK | MISSING | NA` (NA only for a concept with essentially no build-up
  prose — e.g. a one-line narrated definition or the recap unit).
- Wire into `lesson_build.js`'s `structure` lens: `body_engagement` **MISSING → P0**,
  **WEAK → P1** (kind `body_engagement`). This is the hard floor, mirroring how `analogy`
  and `intuition_first` already gate the loop.
- CLI prints the new axis in the per-concept line.

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
   `.build-step` / `.build-num` / `.build-note` CSS** already in the donor — numbered
   steps, each with its "why" gloss.
4. **Wire the dead scroll-reveal** — add the missing `window.__revealBuild()` to the donor
   (an IntersectionObserver that arms `.build` containers and reveals `.build-step`s on
   scroll; graceful — steps stay visible if JS/IO absent or reduced-motion). This turns
   the `%%% steps` widget into a satisfying assemble-as-you-scroll device and makes the
   already-present nav-click call to `__revealBuild()` real.

**Donor-change consequence:** adding `__revealBuild()` changes the donor's embedded JS,
so **every concept lesson using `v9-base.donor` must be recompiled** to stay consistent
(content unchanged; only the donor JS grows). m02+m03 are rebuilt anyway; the other
concept modules (m06, m07, …) get a mechanical recompile pass, and any shipped-output
snapshot test is refreshed. Verified safe by recompiling one non-target concept module
first (gates re-run on identical content).

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

- `test_insight_widget.py` — `%%% insight` renders the re-hook callout.
- `test_demo_predict.py` — `predict:` renders the prompt **and** absence is byte-identical
  to the pre-change demo output (regression guard).
- `test_steps_widget.py` — `%%% steps` renders `.build`/`.build-step` numbered steps + why.
- `test_body_engagement.py` — the `body_engagement` axis is present in the judge's system
  prompt + user prompt, flows through `_extract_json`, and prints in the CLI line.
- Donor: a node/jsdom check that `window.__revealBuild` is defined and arms `.build`.
- Regression: full compiler suite stays green (169 baseline; the 1 pre-existing
  m02-manifest failure is out of scope); recompile idempotence holds; `test_no_dead_lab`
  stays 0.
- Verification on the 14 rebuilt lessons: `body_engagement` GOOD/absent-of-P0 on all 14;
  interest floor still `FLOOR_MET` on all 14; gates pass; idempotent recompile.

## Files touched

| File | Change |
|---|---|
| `.claude/skills/frontier-lesson-builder/SKILL.md` + v8 mirror | + Build-Up Register section, + body-cold P0 rule |
| `sessions/_compiler/gates/coverage_judge.py` | + `body_engagement` axis (SYS, prompt, setdefault, CLI) |
| `sessions/_compiler/v8lib.py` | + `render_insight`, `render_steps`, `predict:` in `render_demo`, register in `render_widget` |
| `sessions/_compiler/shells/v9-base.donor` | + `window.__revealBuild()` scroll-reveal JS |
| `sessions/_compiler/workflows/lesson_build.js` | + `body_engagement` routing in structure lens, + "keep the body alive" author paragraph |
| `sessions/_compiler/AUTHORING.md` | + document `%%% insight`, `%%% steps`, `demo predict:` |
| `sessions/_compiler/tests/*` | new tests above |
| all concept `lesson.html` using `v9-base.donor` | recompile (m02+m03 rebuilt; others mechanical) |
| m02 (9) + m03 (5) `source.md` + `lesson.html` | full author-loop rebuild |
