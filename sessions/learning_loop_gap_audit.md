# Learning-Loop Gap Audit — Retention · Real Doing · Get-Unstuck

**Scope:** whether the curriculum *teaches* a curious 12-year-old (ESL, easily overwhelmed) — i.e. whether content **sticks** and the kid can **act** on it — as opposed to being an engaging one-time read. Companion to the interest audit (`sessions/m02_v9_interest_audit.md`), which found the lessons *are* interesting and warm. This audit covers the three things beyond interest that real teaching needs.
**Method:** mechanical grep scan (curriculum-wide) + 3 dimension deep-dives + adversarial verification of each + a reusable-asset catalog (workflow `m0x-learning-loop-audit`, 7 agents). Every load-bearing claim independently re-verified against files.
**Date:** 2026-07-23 · **Mode:** report-only.

---

## TL;DR

Interest gets a kid *in the door*; these three loops decide whether they *learn*. **All three are broken or absent in the daily HTML lessons the learner actually opens** — and, strikingly, most of the machinery to fix them **already exists but sits on the wrong side of a wall.**

| Dimension | Severity (verified) | One-line verdict |
|---|---|---|
| **Retention** (spaced recall, the spiral) | **CRITICAL** | The SM-2 spaced-repetition engine is code-complete but has **literally never fired**; a concept learned Day 2 is never scheduled to return. |
| **Real doing** (build/run/predict-check) | **HIGH** | All **115 `experiment.py` are empty stubs**; the "do it" button points at a **dead skill**; predict-then-check is broken because answers are printed in the open. |
| **Get-unstuck** (help when confused) | **HIGH** | **Zero** in-the-moment interactive help across 159 lessons; a stalled kid can only re-read or close the tab. |

**The root cause is one structural fact:** the curriculum is **two parallel worlds sharing only two `localStorage` keys.**
- **World A — the Python COACH engine** (`coach/core.py`, `dashboard.py`, `notebook_widgets.py`): rich — full SM-2, XP, badges, a scored quiz widget with hints + self-rating, concept-tagged quiz banks. Reaches learners **only inside Jupyter notebooks.**
- **World B — the static `sessions/` HTML lessons** the 12-year-old actually opens: persists **only** `frontier-theme` and `frontier-lesson:<qid> = {done:{…}}`.

The retention/doing machinery is largely *built* — it just never crosses into World B. That makes the fixes far cheaper than they look.

---

## Finding 1 — Retention: CRITICAL

**The gap:** every retention mechanism the learner touches is one-shot and module-local; the one true spaced system is disconnected.
- **SM-2 never fires.** `coach/core.py:667–714` implements full SM-2 (ease factor, interval doubling, reset-on-fail); `get_due_concepts` exists; `dashboard.py:283` renders a "Spaced Repetition Queue." But `record_quiz_answer` is called **only** from the Jupyter widget (`notebook_widgets.py:530`). `coach/state.json` after **28 sessions** shows `concept_next_review = {}`, `concept_review_count = {}` — **it has never executed.** Grep across `sessions/` for SM-2 / `concept_next_review` / `record_quiz_answer` = **0 hits.**
- **No foundational question bank.** `coach/quizzes/` has files for ML-Design / rnn / transformers but **none for `m01-shape-of-data` or `m02-the-neuron`** — the beginner spine has no tagged concepts to schedule.
- **Openings are narrative callbacks, not retrieval.** Day 5 opens *"Yesterday you built the network's score — one number for how wrong its guess was"* — it *hands over* the answer instead of asking the kid to recall it. Effortful retrieval (the single strongest driver of retention) is absent.
- **Review gates never test cross-module recall.** The m03 attention gate quizzes only attention concepts; it never re-tests M1 shapes or M2 backprop. So the "capability **spiral**" is topic **ordering**, not retrieval reuse — nothing older than the current module is ever re-checked.

**Net:** the curriculum teaches each day, celebrates completion, and then **structurally never checks whether Day 2 survived to Day 20.**

## Finding 2 — Real doing: HIGH

**The gap:** the "making" leg is a facade.
- **All 115 `experiment.py` are empty placeholder stubs** — verified 115/115, each 5–6 lines, zero code: a comment saying *"Placeholder. Fill this from the lesson's PRODUCE step … Option A: write it yourself, or Option B: paste the frontier-experiment-lab prompt."*
- **The "do it" path is dead.** **133 lesson.html** files send the kid to `/frontier-experiment-lab` — a skill that is **not installed**. So the two options are: hand-type code from a blank page (too high a cliff for this learner, no scaffold, no run-and-see-green) or paste a prompt that **silently does nothing.**
- **Predict-then-check is broken — it's anti-doing.** The expected outputs ("What you should see": `z=−1.0, sigmoid≈0.269`, shapes `(5,3)/(5,)`) sit in a plain **ungated list directly below the prompt** — the answer is on screen *before* the kid predicts.
- **No cross-day build-up.** The learner never extends yesterday's artifact into today's (0 real instances). They never get the most motivating builder pattern: *watching one thing they made grow every day.*

**Net:** the curriculum currently teaches by **reading + watching, not doing** — the opposite of what a learn-by-doing 12-year-old needs.

## Finding 3 — Get-unstuck: HIGH

**The gap:** when comprehension breaks mid-section, there is a dead end.
- **Zero in-the-moment interactive recourse.** 0/159 lessons use progressive hint disclosure; no "explain this differently" affordance; no worked micro-example on demand; no mid-lesson check. A kid stuck on *"the running product … is called the upstream gradient"* can only hover a tagged term, re-read, or scroll back.
- **The only diagnostic feedback arrives at the end.** The quiz `fb` strings *do* name specific misconceptions (a real strength), but they fire only at the end-of-lesson quiz on ~4 canned questions — useless to a kid stalled on section 3 of 12.
- **Passive support is genuinely good** (and should be kept): warm normalize-confusion callouts, a full term-tooltip glossary, and the review gates double as a weak "which day to revisit" router. But passive warmth *validates* confusion without giving the kid anything to *do*.
- **A live tutor is architecturally blocked.** Lessons run as `file://` and the deployed site is `https://` GitHub Pages, so neither can `fetch()` the keyless `localhost:11211` bridge (CORS + mixed-content). The fix must be **offline hint ladders / mid-lesson micro-checks / a paste-to-Claude handoff**, not a live in-page call — until there's a served build.

---

## Reusable assets (the fixes are cheaper than they look)

| Asset | Where | Reusable for | Reaches HTML learner today? |
|---|---|---|---|
| SM-2 recurrence (~15 lines) | `coach/core.py:667–714` | Retention — port to vanilla JS over existing localStorage | ❌ |
| Scored-quiz JS pattern (`QS=[{q,opts,ans,fb}]`) | `sessions/*/review.html` | Retention — reuse for daily warm-up + cross-module gate | ✅ |
| Concept-tagged quiz banks (concept_id/hint/explanation) | `coach/quizzes/` | Retention + unstuck (hints pre-written) — but no m01/m02 bank yet | ❌ |
| `%%% demo` predict-then-reveal widget | `_lesson_gen.py` / AUTHORING.md | Doing + retrieval micro-checks | ✅ |
| Hub live dashboard (reads `{done}` storage) | `sessions/index.html` | Retention — natural home for a "N due for review" tile | ✅ |
| Passive unstuck kit (jargon ladder, tooltips, `c-warn`, quiz `fb`) | `_lesson_gen.py` GLOSSARY_SCRIPT | Unstuck — the layer to build escalation on top of | ✅ |
| Donor shell (`v9-base.donor`) | `sessions/_compiler/` | All three — one edit lands across all 159 lessons | ✅ |

---

## Recommended build order

Fixing what's **broken** outranks enforcing what's already **good**. The parked interest+momentum spec is real and ready, but interest is *already achieved* (audit: MATCHES); these loops are *absent*.

1. **Real doing — cheapest, most embarrassing, mostly mechanical.** Replace the 115 empty stubs with real starter scaffolds (imports + TODO bodies + a self-check `assert` that turns "what you should see" into an *earned* reveal), fix the 133 dead `/frontier-experiment-lab` links, and gate the answer-list behind a predict step. Regenerate from `_lesson_gen.py` + the `_build_m*.py` generators so it's not per-file. *(Effort L, but mechanical.)*
2. **Retention — highest learning-science leverage.** Add a required daily **"Warm-up: recall from earlier"** retrieval block (2–3 prior-concept questions answered *before* new content, reusing the review-gate quiz JS) + a **pure-JS SM-2 mirror** over the existing `{done}` localStorage, so Day 2's concept resurfaces on Day 5/12/30; make review gates **cumulative** across modules. Author the missing m01/m02 concept banks. *(Effort L.)*
3. **Get-unstuck — donor-level, lands everywhere at once.** Add tiered-hint disclosure + mid-lesson micro-checks + a **"Stuck? copy this to Claude"** paste widget to `v9-base.donor`. Defer any live tutor to a future served build. *(Effort L.)*
4. **Interest+momentum spec** (already approved) — fold in alongside, or ship first since it's small and ready; it prevents regression and extends interest enforcement to the 76% no-notebook days.

Each is its own spec → plan → build cycle. The unifying move behind 1–3 is **crossing the World-A/World-B wall**: either surface COACH state into the HTML lessons, or mirror its ~15-line SM-2 recurrence in browser JS keyed off the localStorage the hub already reads.
