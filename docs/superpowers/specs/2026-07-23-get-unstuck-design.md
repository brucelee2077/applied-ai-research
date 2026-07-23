# Get-Unstuck — Design Spec

**Date:** 2026-07-23 · **Author:** ruifengli (+ Claude) · **Status:** Draft for review
**Scope:** Give a stalled 12-year-old something to **do** at the point of confusion — offline-safe, since a live in-page tutor is architecturally blocked. Adds tiered hints, mid-lesson micro-checks, and a paste-to-Claude handoff on top of the existing (good) passive support. Foundation audit: `sessions/learning_loop_gap_audit.md` (Finding 3, HIGH). One of three loop-fix specs.

---

## 1. Problem

When comprehension breaks mid-section, there is a dead end (verified):
- **Zero in-the-moment interactive help** across all 159 lessons (0 use progressive hint disclosure). A kid stuck on *"the running product … is called the upstream gradient"* can only hover a tagged term, re-read, or scroll back.
- **Diagnostic feedback arrives too late** — the quiz `fb` strings *do* name specific misconceptions (a real strength), but only at the **end-of-lesson** quiz (~4 canned questions), useless to a kid stalled on section 3 of 12.
- **A live tutor is blocked** — lessons run `file://` and the site is `https://` GitHub Pages, so neither can `fetch()` the keyless `localhost:11211` bridge (CORS + mixed-content). The fix must be offline, not a live in-page call.
- Passive support (tooltips, normalize-confusion callouts, the review-gate "which day to revisit" router) is **genuinely good and must be kept** — but it validates confusion without giving the kid an action.

## 2. Goal

At the hardest moments, the learner has an **escalating, in-page way to get help** — a nudge, then a worked micro-step, then a one-sentence idea, plus a check that catches a misconception *where it happens*, plus a one-click handoff to a real tutor — all working offline.

## 3. Design

**① Tiered hint ladder (`%%% hint` widget).** A new authoring block compiled into a *"💡 Stuck? reveal a hint"* control on each hard section that progressively discloses: **hint-1** (a nudge) → **hint-2** (a worked micro-step) → **"still stuck? the idea in one sentence."** Authored in the donor + a `%%% hint` grammar so it lands across lessons; reuses the existing callout CSS + quiz-reveal JS. Hint content bridge-generated from the section (seeded by the `coach/quizzes` `hint` fields where they exist).

**② Mid-lesson micro-checks.** Promote the diagnostic-`fb` quiz engine into **optional inline "quick checks"** right after the 2–3 hardest sections (not only end-of-lesson), so misconception-naming feedback lands **where** the kid stalls (e.g. day-05 chain-rule; day-02 vanishing-gradient / dead-ReLU). These double as retrieval events feeding the retention spec's SM-2 state.

**③ Paste-to-Claude handoff.** A per-section *"Stuck? copy this to Claude"* button that pre-fills a **scoped** prompt — e.g. *"I'm 12 and on M2 Day 5, section 'Trace the blame'. I don't understand `<term>`. Explain it a simpler way with a small example."* Reuses the produce copy-to-clipboard widget. This is the pragmatic "tutor" given the `file://`/CORS constraint, and it replaces the dead `/frontier-experiment-lab` Option-B block (shared with the real-doing spec).

**Deferred:** a live `localhost:11211` tutor — requires a **served build** (local server + CORS/allow-listing). Note as future work; do not gate this spec on it.

**Generation strategy:** bridge-generate the hint ladders + micro-check questions per hard section, judged for correctness and beginner phrasing (reuse the tone/interest judge bar). Author once in the donor/compiler so the widgets render across all 159 lessons.

## 4. Reusable assets
`v9-base.donor` + compiler (one edit → all lessons) · callout CSS + quiz-reveal JS · the diagnostic-`fb` quiz engine · the produce copy-to-clipboard widget · `coach/quizzes` `hint`/`explanation` fields (pre-written unstuck content) · the GLOSSARY tooltip system + normalize-confusion callouts (the passive layer this builds on).

## 5. Files touched
**First**, register the new widget type so the compiler accepts it (AUTHORING.md §4: unknown `%%%` types raise `ValueError`): teach the parser the `hint` case in `sessions/_compiler/` + `_lesson_gen.py`, and document `%%% hint` in `sessions/_compiler/AUTHORING.md` (the stated grammar source of truth) — *before* any donor/lesson authors it. Then: `v9-base.donor` (hint + micro-check + paste-to-Claude widgets + JS) · pilot the two hardest lessons first (`sessions/m02-the-neuron/day-02-activations` and `sessions/m02-the-neuron/day-05-gradients-backprop`), then roll donor-wide.

## 6. Testing
Hint ladder reveals progressively (hint-1 → hint-2 → one-liner), never all at once · mid-lesson micro-check scores + shows the correct `fb` · paste button copies the correctly-scoped prompt to the clipboard · **offline-safe**: grep-assert no `fetch()`/network call in the **donor + widget JS regions** (scope the grep to the widgets, not lesson body prose — some m26 lessons legitimately mention `localhost:11211` in produce text) · widgets render across a sample of regenerated lessons; recompile idempotence preserved.

## 7. Out of scope
A live in-page bridge tutor (deferred to a served build) · full adaptive branching of lesson content · replacing the passive support layer (it stays).
