# Retention & the Spiral — Design Spec

**Date:** 2026-07-23 · **Author:** ruifengli (+ Claude) · **Status:** Draft for review
**Scope:** Make content **stick** for a 12-year-old by bringing **spaced, active retrieval** into the daily HTML lessons — porting the existing (but never-fired) SM-2 engine across the World-A/World-B wall, adding a daily recall warm-up, making review gates cumulative, and surfacing due-reviews in the hub. Foundation audit: `sessions/learning_loop_gap_audit.md` (Finding 1, CRITICAL). One of three loop-fix specs.

---

## 1. Problem

Every retention mechanism the learner touches is one-shot and module-local; the one true spaced system is disconnected (verified):
- **SM-2 has never fired.** `coach/core.py:667–714` is a complete SM-2 engine, but `record_quiz_answer` is reachable **only** from the Jupyter widget; `coach/state.json` after 28 sessions has `concept_next_review = {}`. Grep across `sessions/` for SM-2 = 0.
- **No spaced resurfacing.** A concept learned Day 2 is never scheduled for Day 5/12/30. The "capability spiral" is topic **ordering**, not retrieval **reuse**.
- **Openings hand over the answer.** *"Yesterday you built the network's score…"* is a narrative callback, not effortful recall.
- **Review gates are module-local.** The m03 gate never re-tests M1 shapes or M2 backprop.
- **No foundational question bank.** `coach/quizzes/` has no file for `m01`/`m02` or the M04–M28 spine.

## 2. Goal

A concept a kid learns is **actively re-retrieved on a spacing schedule** inside the daily lessons they actually open — so Day 2's idea resurfaces on Day 5, ~12, ~30 — with cumulative review gates and a visible "due today" nudge.

## 3. Design

**① Browser SM-2 mirror (cross the wall without a server).** Port the SM-2 recurrence + due-selection (`core.py:641–714`: `get_due_concepts` at 641–665 for due selection, and `record_quiz_answer`'s recurrence at 667–714 — ease update, interval `1→6→interval*ease`, reset-on-fail) to **vanilla JS in `v9-base.donor`**. The XP side-effects (`core.py:702–707`) are intentionally **out of the browser mirror's scope**. Extend the existing `frontier-lesson:<qid>` localStorage object (already read by the hub) with `sr:{concept_id:{next,ease,interval,reps}}`. No server, works offline.

**② Daily "Warm-up: recall from earlier" block.** Every lesson **opens** with 2–3 active-recall questions on **due/prior** concepts, answered **before** any new content, reusing the review-gate scored-quiz JS (`QS=[{q,opts,ans,fb}]`). Answering updates the SM-2 state (a right answer pushes `next` out; wrong resets). This converts the narrative callback into effortful retrieval — the single strongest retention driver.

**③ Concept banks for the spine.** Bridge-generate + judge concept-tagged question banks for the modules missing from `coach/quizzes/` (m01, m02, and the M04–M28 spine), in the **existing schema** (`concept_id / question / choices / correct / hint / explanation`). The warm-up and gates draw from these; the `hint`/`explanation` fields double as get-unstuck content.

**④ Cumulative review gates.** Extend the 29 `review*.html` gates so each mixes **2–3 questions from prior modules** (the m03 gate re-tests one M1 shape + one M2 backprop item), scheduled by the same SM-2 state.

**⑤ Hub "due for review" tile.** Surface a due-count tile + CTA in `sessions/index.html` (which already reads the `{done}` storage), mirroring `dashboard.py`'s Spaced Repetition Queue — so the kid is pulled back to what's due.

**Deferred (nice-to-have, needs a server):** a JS↔`coach/state.json` sync so browser SR, the notebook dashboard, and `get_due_concepts` share one source of truth. Not required for the learner-facing win.

**Generation strategy:** bridge-generate the concept banks + warm-up/gate questions, judged for correctness **and** beginner phrasing (reuse the tone/interest judge bar).

## 4. Reusable assets
SM-2 recurrence + due-selection (`core.py:641–714` — `get_due_concepts` + `record_quiz_answer`) · review-gate scored-quiz JS (`QS=[…]`, instant `fb`, score tally) · `coach/quizzes/` concept-tag schema · `dashboard.py:283` Spaced-Repetition-Queue renderer · the `{done}` localStorage contract + hub dashboard (`index.html`).

## 5. Files touched
`v9-base.donor` (JS SM-2 mirror + warm-up render) · `_lesson_gen.py` (mandatory warm-up block) · the `review*.html` generator (`_review_shell_migrate.py`) for cumulative items · `coach/quizzes/` (new banks) · `sessions/index.html` (due tile).

## 6. Testing
JS SM-2 unit tests (interval progression, reset-on-fail, due-selection) via a node/jsdom harness · warm-up renders and persists `sr:` state across reloads · a review gate includes ≥1 prior-module item · hub renders a correct due-count · generated banks pass the correctness + beginner-phrasing judge. Recompile idempotence preserved.

## 7. Out of scope
Full COACH state unification / server sync (deferred) · adaptive difficulty beyond SM-2 · changing the notebook COACH path (it keeps working).
