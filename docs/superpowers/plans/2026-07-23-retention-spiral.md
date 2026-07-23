# Retention & the Spiral Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring spaced, active retrieval into the daily HTML lessons: a browser SM-2 mirror over the existing localStorage, a daily "recall" warm-up, cumulative review gates, foundational concept banks, and a "due for review" hub tile.

**Architecture:** Port COACH's SM-2 recurrence (`coach/core.py:641–714`) to vanilla JS in the donor, extending the `frontier-lesson:<qid>` localStorage object with `sr:{concept:{next,ease,interval,reps}}` (no server). Warm-up + gate questions reuse the review-gate scored-quiz JS and are drawn from bridge-generated concept banks in the existing `coach/quizzes` schema.

**Tech Stack:** vanilla JS in `v9-base.donor` (node/jsdom tests), Python generators (`_lesson_gen.py`, `_review_shell_migrate.py`), the LLM bridge for banks, system python3.

**Spec:** `docs/superpowers/specs/2026-07-23-retention-spiral-design.md`

---

### Task 1: Browser SM-2 mirror (pure JS, unit-tested)

**Files:**
- Create: `sessions/_compiler/shells/js/sr.js` (authored module, later inlined into the donor)
- Test: `sessions/_compiler/tests/test_sr.mjs` (node, no DOM needed)

- [ ] **Step 1: Failing test.** Port the SM-2 semantics from `coach/core.py:667–714`. Assert: fresh concept → `interval=1`; correct (quality≥3) twice → intervals `1 → 6`; a third correct → `round(6*ease)`; a wrong answer (quality<3) → `interval` resets to `1`; `dueConcepts(state, today)` returns concepts whose `next <= today`, most-overdue first.
```js
// test_sr.mjs — run: node sessions/_compiler/tests/test_sr.mjs
import { review, dueConcepts } from '../shells/js/sr.js'
import assert from 'node:assert'
let s = {}
s = review(s, 'c1', 5, 0)      // day 0, perfect
assert.equal(s.c1.interval, 1)
s = review(s, 'c1', 5, 1)      // next review
assert.equal(s.c1.interval, 6)
s = review(s, 'c1', 1, 7)      // failed
assert.equal(s.c1.interval, 1)
assert.deepEqual(dueConcepts({a:{next:0},b:{next:5}}, 3), ['a'])
console.log('ok')
```
- [ ] **Step 2: Run, expect FAIL.** `node sessions/_compiler/tests/test_sr.mjs` → error (module missing).
- [ ] **Step 3: Implement `sr.js`** — `review(state, conceptId, quality, today)` mirrors `core.py:667–714` **exactly**: track `reps` (review_count) per concept and gate the interval on it — `reps==0 → interval=1`, `reps==1 → interval=6`, `reps>=2 → interval=round(interval*ease)` (NOT keyed on the running interval); ease update `ease + 0.1 - (5-q)*(0.08+(5-q)*0.02)`, floor 1.3; on `q<3` reset `reps=0` and `interval=1`; `next = today + interval`; increment `reps` on success. `dueConcepts(state, today)` returns concepts with `next <= today`, most-overdue first. **Omit** the XP side-effects (`core.py:702–707`).
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit.** `git add sessions/_compiler/shells/js/sr.js sessions/_compiler/tests/test_sr.mjs && git commit -m "feat(retention): browser SM-2 mirror (sr.js) + unit tests"`

### Task 2: Wire `sr:` into the lesson localStorage + inline sr.js into the donor

**Files:**
- Modify: `sessions/_compiler/shells/v9-base.donor` (inline `sr.js`; extend the `state` read/write to carry an `sr` sub-object under the existing `frontier-lesson:<qid>` key)
- Test: `sessions/_compiler/tests/test_donor_sr.mjs` (jsdom)

- [ ] **Step 1: Failing jsdom test** — after a simulated warm-up answer, `localStorage['frontier-lesson:<qid>']` parses to an object containing `sr.<concept>.next`.
- [ ] **Step 2–4:** Inline `sr.js`; on warm-up answer call `review(...)` and persist. Keep `done`/`theme` untouched (hub still reads them). Verify jsdom test + recompile idempotence.
- [ ] **Step 5: Commit.** `git commit -am "feat(retention): persist SM-2 sr state in lesson localStorage"`

### Task 3: Daily "Warm-up: recall from earlier" block

**Files:**
- Modify: `sessions/_compiler/v8lib.py` + `sessions/_compiler/shells/v9-base.donor` (emit + render a mandatory warm-up section at the top of a concept-mode lesson, reusing the review-gate `QS=[{q,opts,ans,fb}]` JS) — the V9 compile path, NOT the legacy `_lesson_gen.py`; the warm-up content is authored/populated from front-matter (`warmup:`)
- Test: `sessions/_compiler/tests/test_warmup.py` + jsdom render check

- [ ] **Step 1: Failing test** — a compiled concept-mode lesson contains a `#warmup` section BEFORE the first `@@@ concept`, with ≥2 questions, answered-before-revealed.
- [ ] **Step 2–4:** Add the warm-up block: 2–3 recall questions on due/prior concepts (source from the day's `warmup:` front-matter list, populated in Task 4). Answering updates `sr` (Task 2). Right answer → pushes `next` out; wrong → shows `fb` + resets.
- [ ] **Step 5: Commit.** `git commit -am "feat(retention): mandatory daily recall warm-up (active retrieval)"`

### Task 4: Concept banks for the foundational spine (bridge-generated + judged)

**Files:**
- Create: `coach/quizzes/m01_shape_of_data.py`, `coach/quizzes/m02_the_neuron.py`, … (M04–M28 spine), matching the existing schema (`concept_id / question / choices / correct / hint / explanation`) — where **`correct` is a choice LETTER (`"A"`/`"B"`/`"C"`/`"D"`), not a numeric index** (confirmed against `transformers.py`)
- Reference: `coach/quizzes/transformers.py` (schema exemplar)

- [ ] **Step 1:** For each module, bridge-generate 1–2 questions per concept (concept_ids = the lesson `coverage_topics`), judged for correctness + beginner phrasing (reuse the tone/interest judge bar). Gate: JSON parses, fields present, and `correct` is a **letter that maps to a real choice**.
- [ ] **Step 2:** Wire each day's `warmup:` front-matter list = a due-scheduled sample of prior-day/prior-module concept_ids.
- [ ] **Step 3: Verify** banks import cleanly (`python3 -c "import coach.quizzes.m02_the_neuron"`) and every `correct` letter indexes a real choice.
- [ ] **Step 4: Commit.** `git commit -m "feat(retention): concept banks for m01/m02 + spine (bridge-generated, judged)"`

### Task 5: Cumulative review gates

**Files:**
- Modify: `sessions/_review_shell_migrate.py` (the review-gate generator) — mix 2–3 prior-module concepts into each gate's `QS`
- Test: `sessions/_compiler/tests/test_cumulative_gate.py`

- [ ] **Step 1: Failing test** — the m03 review gate `QS` includes ≥1 concept_id owned by m01 or m02.
- [ ] **Step 2–4:** Sample prior-module concepts (by SM-2 due-ness) into each gate; regenerate the 29 gates. Verify test + gate still scores.
- [ ] **Step 5: Commit.** `git commit -am "feat(retention): cumulative cross-module review gates"`

### Task 6: Hub "due for review" tile

**Files:**
- Modify: `sessions/index.html` (read `sr` from each `frontier-lesson:<qid>`, compute due-count via `dueConcepts`, render a tile + CTA)
- Test: `sessions/_compiler/tests/test_hub_due.mjs` (jsdom)

- [ ] **Step 1: Failing test** — seed localStorage with an overdue concept; assert the hub renders a due-count > 0 and a "Review now" CTA.
- [ ] **Step 2–4:** Add the tile (mirror `dashboard.py:283` Spaced-Repetition-Queue). Verify jsdom.
- [ ] **Step 5: Commit.** `git commit -am "feat(retention): hub due-for-review tile"`

---
**Done when:** `sr.js` unit tests pass; a rebuilt lesson opens with a working recall warm-up that persists SM-2 state; review gates include prior-module items; the hub shows a due count; `python3 -m pytest sessions/_compiler/tests/ -q` + the node tests are green.
**Deferred (out of scope):** JS↔`coach/state.json` server sync (one source of truth) — noted, not built.
