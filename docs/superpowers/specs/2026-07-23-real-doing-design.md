# Real Doing — Design Spec

**Date:** 2026-07-23 · **Author:** ruifengli (+ Claude) · **Status:** Draft for review
**Scope:** Make the "making" leg of the learning loop real: every daily lesson ships a **runnable, self-checking** artifact; predict-then-check becomes genuine; the dead `/frontier-experiment-lab` path is removed; cross-day build-up is enabled where natural. Foundation audit: `sessions/learning_loop_gap_audit.md` (Finding 2, HIGH). One of three loop-fix specs (with retention + get-unstuck).

---

## 1. Problem

For a learn-by-doing 12-year-old, "doing" is a facade (verified curriculum-wide):
- **All 115 `sessions/**/experiment.py` are empty stubs** — 5–6 lines, zero code: *"Placeholder. Fill this from the lesson's PRODUCE step … Option B: paste the frontier-experiment-lab prompt."*
- **The "do it" path is dead** — **133 lesson.html** send the kid to `/frontier-experiment-lab`, a skill **not installed**. Option A is a blank page (too high a cliff, no scaffold, no run-and-see-green); Option B silently does nothing.
- **Predict-then-check is broken** — the produce block prints expected outputs (`z=−1.0, sigmoid≈0.269`, shapes) in an **ungated list right under the prompt**, so the answer is on screen before the kid predicts.
- **No cross-day build-up** — the learner never extends yesterday's artifact; they never watch one thing they made grow.

## 2. Goal

Every day ends in a real, runnable artifact the learner can fill in and **run to see green**, with a genuine predict-then-check, no dead links, and — where topically natural — building on yesterday's code.

## 3. Design

**① Real, run-verified `experiment.py` per day (reuse `evidence_build.js`).**
`sessions/_compiler/workflows/evidence_build.js` already writes **and runs** a real `python3` `experiment.py` (into `portfolio/<module>/<day>/`), fixing it until it exits 0, judge-gated on `ran_ok` + `numbers_match` (6 real ones already exist under `portfolio/m04-first-model-mlp/`). Extend/retarget this producer to emit the artifact into the **lesson dir** (`sessions/<module>/<day>/experiment.py`), replacing the stub. Same producer+judge loop; no new engine.

**② Scaffold, not blank page.** The shipped `experiment.py` is a **starter scaffold** for the ESL/easily-overwhelmed learner: imports + `# TODO` function bodies with 1-line hints + a self-check block — fill-the-blanks and run, not from-scratch.

**③ Self-check that earns the reveal.** The artifact ends with an `assert`-and-print block that prints `✅ you got it` / `❌ not yet — expected X` when run — so "what you should see" is **confirmed by running**, not read off the page.

**④ Kill the dead path.** Replace the produce "Option B: paste the frontier-experiment-lab prompt" block: **Option A** → *"open `experiment.py` — it's scaffolded; fill the TODOs and run it"*; **Option B** → the **paste-to-Claude handoff** (shared with the get-unstuck spec), not the dead skill. `/frontier-experiment-lab` appears in **453 locations** (133 `lesson.html` + 115 `experiment.py` + 43 `source.md` + 2 `manifest.yaml` + historical reports); this spec rewrites it in the **learner-facing + generating** files (the 133 `lesson.html`, the 115 regenerated `experiment.py`, and the 43 `source.md` that compile them) and leaves historical reports/trackers untouched.

**⑤ Gate the answer (real predict-then-check).** Move the produce "What you should see" list behind a **predict-then-reveal gate** — a small record-a-prediction-then-reveal control built on the existing quiz-reveal JS (the `%%% demo` console shows output on click but does not itself force a prediction, so this is a thin gate on top, not a ready-made widget). The kid records a prediction, then reveals — the answer is no longer pre-printed.

**⑥ Opt-in cross-day build-up.** Where topically natural (Day 3's layer reuses Day 1's neuron; Day 6's loop reuses Day 5's gradient), today's `experiment.py` imports/extends yesterday's. Recorded per-module in the module manifest (`sessions/<module>/_refactor/manifest.yaml`, a new `build_on: <prior-day>` key on the day entry), **not forced** on every day.

**Generation strategy (cross-cutting):** bridge-generate at build time via the `evidence_build.js` producer + judge, invoked from the module generators (`_lesson_gen.py` / `_build_m*.py`). Judge gate: artifact **exits 0**, **self-check passes**, and its numbers **match the lesson**. Hand-authored fallback if the bridge is down.

## 4. Reusable assets
`evidence_build.js` (writes+runs+judges a real experiment.py) · `evidence_compile.py` / `evidence_judge.py` · the 6 real `portfolio/m04-*/experiment.py` as templates · the `%%% demo` predict-then-reveal JS · `_lesson_gen.py` + `_build_m*.py` generators.

## 5. Files touched
115 `experiment.py` (regenerated) · 133 `lesson.html` produce blocks (dead-link removal + answer-gating) · `_lesson_gen.py` + `_build_m*.py` · `evidence_build.js` (retarget to lesson dir) · module manifests (`build_on`).

## 6. Testing
Every regenerated `experiment.py` exits 0 and its self-check passes (the producer loop already enforces `ran_ok`); grep-assert **0** `/frontier-experiment-lab` references remain in the learner-facing + generating files (`lesson.html`, `experiment.py`, `source.md`) — historical reports excluded; the produce answer list is gated behind a reveal; a `build_on` day's artifact imports the prior day's without error. Recompile idempotence preserved.

## 7. Out of scope
In-browser code execution; full auto-grading beyond the self-check assert; forcing cross-day build-up on unrelated days.
