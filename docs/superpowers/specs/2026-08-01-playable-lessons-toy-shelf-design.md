# Playable Lessons and the Toy Shelf — Design

**Date:** 2026-08-01
**Branch:** `build/capability-spiral`
**Status:** Approved by user, pending implementation plan
**Scope:** `sessions/` only. `coach/` is explicitly untouched.

---

## 1. What this is, and what it replaced

The user asked for "a reward system to make the learning more fun and engaging." An audit
(11 agents, every load-bearing claim adversarially verified) showed that a reward system
was the wrong answer, and the user agreed. This document specifies what we build instead.

**The conversation converged in four steps:**

1. Focus on `sessions/` only, not the `coach/` notebook engine.
2. The goal is **"make the hour itself fun"** — not proving learning, not building a
   daily-return habit, not driving module completion.
3. Within "fun," the chosen register is **play — things to manipulate.**
4. Coverage strategy is **just-in-time**, chosen with the reason *"because I don't know
   what we will build."*
5. Progress should be **playable too** — specifically the **Toy Shelf** (option A of three
   mockups).

Step 4's stated reason is load-bearing. It rules out designing a widget taxonomy up front,
which an earlier draft of this design proposed. See §4.3.

---

## 2. Evidence this is the right problem

All figures independently re-verified against the working tree at commit `ddfadae`.

| Finding | Evidence |
|---|---|
| **211 of 266 lesson pages have nothing to manipulate** | `grep -lE "<iframe\|<input[^>]*type=[\"']?range"` over 159 nested `lesson.html` → 54 hit; over 107 flat `week-m*/day-*.html` → 1 hit. 55 interactive, 211 inert. |
| Whole modules are inert | m05a 0/8, m05b 0/4, m06 0/8, m07 0/6, m12 0/11, m14b 0/6, m15b 0/6, m26 0/12, m29 0/8 |
| The daily surface pays nothing that accumulates | `grep -rlE '\bXP\b'` over all 266 lesson pages → **0 files**. Lesson state is literally `{done:{<section>:true}}` (`lesson.html:497`). |
| Play is the lever the project already committed to | `sessions/_compiler/workflows/lesson_build.js:80` — "CULTIVATING THE READER'S INTEREST IS THE #1 GOAL at this early stage" |
| The infrastructure already exists | 22 standalone viz pages in `sessions/viz/`, `d3.v7.min.js` vendored locally, `%%% viz` widget at `sessions/_compiler/v8lib.py:224` emitting the auto-resizing `.build-embed` iframe |
| **All 22** viz pages already send their height | `grep -c postMessage sessions/viz/*.html` → 22/22. No autoresize retrofit needed. |
| The user is at m02 Day 2 | Chrome Profile 1 localStorage: `frontier-lesson:wf2-d02-activations`; `wf2-d01-neuron` has `c1`–`c11` all true; files mtime 2026-08-01 14:52 |

**A methodological note worth preserving:** two naive greps during this audit produced
false results — `grep -i xp` matched *"experiment"*, and `grep "build-embed"` matched the
CSS *class definition* present in every lesson's stylesheet, which reported all 23 modules
as fully interactive. Both were caught only by re-running against real tags. Any future
interactivity census must match `<iframe` and `<input ... type=range`, never class names.

---

## 3. Goals and non-goals

### Goals

- **G1.** A learner opening any lesson in the module they are currently on can change
  something with their hands and watch the consequence.
- **G2.** Progress, seen on the hub, is itself something to play with rather than a
  percentage to read.
- **G3.** No work is done for lessons the learner has not reached.

### Non-goals — deliberately excluded

- **No XP, levels, tokens, streaks, shields, badges, or boss battles.** The user stated
  they dislike the existing `coach/` system. None of it is ported.
- **No fabricated social proof.** The hub's own comment (`index.html:825`, "every value
  below is derived live from localStorage") is the standard; `coach/`'s
  `fictional_cohort_size: 1247` is the anti-pattern.
- **No changes to `coach/`.** Not extended, not bridged, not deleted. Out of scope.
- **No spaced-repetition or retention work.** `state.sr` is written by 46 pages and read by
  nothing; that is a real gap, but it serves the "come back tomorrow" goal the user
  explicitly did not choose. Recorded in §8 as a known deferral.
- **No coverage or curriculum changes.** Play is added to existing lessons; no lesson is
  rewritten, resequenced, or removed.

---

## 4. Design

Three components. Component C is the only one that is mostly new code.

### 4.1 Component A — the just-in-time rollout rule

Not code. A rule governing when play gets built.

- **Trigger:** the learner finishing a module. Not a schedule, not a batch.
- **Increment:** exactly one module — the next one. Never "all inert pages."
- **Horizon:** nothing is built for a module the learner has not reached. `m17b` stays
  inert until they are working through `m17a`.

**First increment, from the verified census:**

| Module | Playable now | Lessons needing play |
|---|---|---|
| m02 ← current position | 9/9 | 0 — already the best module in the repo |
| m03 · attention | 4/5 | **1** (`day-05-positional`) |
| m04 · first-model-mlp | 2/6 | **4** (`day-02`, `day-04`, `day-05`, `day-06`) |

Five lessons of work covers the learner through the end of m04. The next wall
(m05a 0/8, m05b 0/4, m06 0/8, m07 0/6 = 26 lessons) is three modules away and is not
planned now, by design.

### 4.2 Component B — the bar for "playable"

A widget counts only if all five hold. This exists to stop play drifting into decoration.

1. **Immediate consequence.** Drag, don't submit. No button gates the feedback.
2. **It is the lesson's one mechanism** — not an illustration beside it. Reference
   implementation: `m02/day-01-single-neuron` (4 sliders; drag the weight, the boundary tilts).
3. **It can be broken.** The learner can drive it into a state where the mechanism visibly
   fails. This is what static images cannot do, and where the learning is.
4. **It works offline.** Plain inline SVG, or the vendored `sessions/viz/d3.v7.min.js`.
   No CDN — public CDNs return 403 in this environment.
5. **It lands through the compiler.** A `%%% viz` block plus a page under `sessions/viz/`.
   No hand-editing of `lesson.html`. `data-quest-id` values are never touched, so existing
   progress survives (`sessions/COACH_STYLE_GUIDE.md:151-179`).

### 4.3 Component C — the Toy Shelf

The hub's progress surface becomes a shelf of the widgets the learner has met, each
replayable in place.

#### Data model

One new array in `sessions/index.html`, a sibling of `MODULES` (declared at `:335`):

```js
// name, module label, unlocking quest-id, what you can do with it, source page
var TOYS = [
  ['Broadcasting',       'm01 · Day 3', 'wf1-d03-broadcasting','stretch two shapes to fit',   'viz/broadcasting.html'],
  ['Matmul shapes',      'm01 · Day 4', 'wf1-d04-matmul',     'multiply matrices',            'viz/matmul.html'],
  ['Activation curves',  'm02 · Day 2', 'wf2-d02-activations','compare ReLU vs sigmoid',      'viz/activation-derivatives.html'],
  ['XOR limit',          'm02 · Day 2', 'wf2-d02-activations','break a single neuron',        'viz/xor-limit.html'],
  // ...
];
```

`wire_index.py` only rewrites fields 3 and 4 of entries matching `wNN-dMM` inside the
`WEEKS`/`MODULES` structure and is idempotent, so a separate `TOYS` array is not clobbered
by re-runs. Verified by reading `sessions/wire_index.py:1-30`.

#### Unlock rule

A toy is unlocked when its lesson has **any** completed section:

```js
var s = JSON.parse(localStorage.getItem('frontier-lesson:'+qid) || 'null');
var unlocked = !!(s && s.done && Object.keys(s.done).length > 0);
```

This deliberately differs from the hub's existing `pillStatus()` (`index.html:677-688`),
which requires `done.produce || done.verdict`. Rationale: a learner meets a toy in the
middle of a lesson, so requiring full completion would lock a toy they have already played
with. The cost is that a toy can unlock slightly before its section is reached. Chosen for
generosity, consistent with the "fun" goal. `pillStatus()` is left unchanged — the ring and
capability bars keep their stricter definition.

#### Rendering

- A grid of cards at the top of the existing `.dash` section (`index.html:274`).
- Heading: `<n> toys on your shelf`.
- Subtitle: **carried over from mockup option C** — the forward-looking gate line
  (`"5 to clear the m02 gate"`), derived from the same data the existing goal cards use
  (`index.html:858-901`). The audit rated this the best existing mechanic in the repo; it
  survives into the shelf rather than being replaced.
- Locked toys render dashed and dimmed with the module that unlocks them — visible as
  anticipation, not as a scold.
- Clicking an unlocked toy opens it **inline** in a `.build-embed` iframe, reusing the
  existing auto-resize receiver. All 22 viz pages already post their height, so no
  retrofit is needed. A plain link to the standalone page is the no-JS fallback.

#### Two kinds of toy, and the extraction backlog

- **Standalone** (`sessions/viz/*.html`) — shelf-able immediately. 22 exist.
- **Inline sliders** — 24 lessons build their widget directly in `lesson.html` (m02 Day 1
  has 4 sliders, m03 Day 2 has 8). These are *not* shelf-able until lifted into their own
  file under `sessions/viz/`.

Extraction is itself just-in-time: a toy is extracted when its module gets its play pass.
**One exception:** m02's six slider lessons are already complete work by the learner, and
extracting them is what makes the shelf feel populated on day one rather than showing two
toys. Treated as a small, explicit one-off in the plan.

---

## 5. Constraints the implementation must respect

Each is sourced, not assumed.

| Constraint | Source |
|---|---|
| No server. Lessons run `file://` locally and `https://` on GitHub Pages; neither can `fetch()` anything. | `sessions/learning_loop_gap_audit.md:53`; 0 `fetch(` in `sessions/` |
| Public CDNs are 403. Vendor or inline all JS. | memory `cdn-blocked-vendor-js-locally.md` |
| Browser state is localStorage-only, per-origin, per-machine, and gitignored. | `.gitignore:60`; verified single `file://` storage area in Chrome Profile 1 |
| `data-quest-id` is frozen. Changing one wipes that lesson's progress. | `sessions/COACH_STYLE_GUIDE.md:151-179` |
| Lesson edits go through the compiler (`v8lib.py`, `_compiler/shells/*.donor`, `lesson_build.js`), not by hand. | verified file layout |
| `sessions/index.html` is hand-maintained and safe to edit directly. | last 5 commits to it are hand edits; `wire_index.py` is a field-filler, not a generator |
| Regeneration is expensive: ~$300/day ceiling ≈ 5–6 lesson-days. | `sessions/concept_body_engagement_rerun.md:156-162` |
| Interest-first outranks rigor and coverage at this stage. | `lesson_build.js:80`; memory `interest-first-at-early-stage.md` |
| Lesson bodies are Reader A only — no interview register. | `CLAUDE.md:9-10, 20, 39` |

---

## 6. Verification

Manual browser checking is not available — there is no usable browser automation in this
sandbox (`chrome-devtools-axi`, Playwright, and AppleScript are all blocked; both
`start-server.sh` and `lavish-axi` fail with `listen EPERM`). Verification is therefore
static plus jsdom, matching established practice in this repo.

1. **`node --check`** on every new or modified viz page's script block.
2. **jsdom harness** for the shelf: seed a fake `localStorage` with known
   `frontier-lesson:<qid>` values, render, and assert the unlocked/locked partition and the
   toy count. Pattern already used in this repo (`/tmp/hubverify`, `sessions/nav_audit.py`).
3. **Interactivity census** re-run after each module's pass, using real-tag matching, to
   confirm the target module moved to n/n. The naive-grep failure mode in §2 is the reason
   this check is specified precisely.
4. **`scripts/validate_notebook.py`** is not applicable — no notebooks are touched.
5. **Toy-shelf link integrity:** assert every `TOYS[i][4]` path exists on disk and every
   `TOYS[i][2]` quest-id appears in some lesson page, so a shelf entry can never point at
   nothing.
6. **Regression:** confirm `pillStatus()`, the ring, capability bars, and the 12 existing
   milestone badges still compute identically after the shelf is added.

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **The shelf is only as good as the toys on it.** Charming at 6 toys; unproven at 40. | High — it is the core bet | The user was shown this risk explicitly and accepted it. Re-evaluate after m04, when the shelf holds ~12. If it feels thin, mockup option C (ring + can-do strip) is the fallback and is nearly free. |
| Just-in-time means the learner meets inert lessons if they move faster than the build. | Medium | The rule triggers on module completion, giving one module of lead time. m02→m04 is only 5 lessons of work. |
| Extracting inline sliders could alter lesson rendering. | Medium | Extraction must be byte-additive to `lesson.html` (a `%%% viz` block replacing the inline block), verified by the existing shell-invariant gate. |
| A widget satisfies bar rule 1 but not rule 3 (can't be broken) and quietly becomes decoration. | Medium | Rule 3 is the reviewable one; make it an explicit checklist item per widget, not a vibe. |
| Unlock-on-any-section makes the shelf feel loose. | Low | Deliberate choice, documented above. Trivially revertible to `pillStatus()`. |

---

## 8. Known deferrals

Recorded so they are not lost, and explicitly not built now:

1. **The retention loop is write-only.** `srReview()` writes `{ease, interval, next, reps}`
   into `frontier-lesson:<qid>` on 46 pages; `dueConcepts()` exists only at
   `sessions/_compiler/shells/js/sr.js:44` and is inlined into zero lessons. The learner
   currently has `prereq-multiply` due 2026-08-07 that nothing will surface. Deferred
   because it serves the goal the user did not choose.
2. **4 orphaned quest-ids.** `w03-d06-examrehearsal`, `w02-d06-distckpt`,
   `w04-d06-consolidation`, `w17-d01-consolidation` appear on lesson pages but in no hub
   row, so m08/m09a/m09c/m14b can never reach 100%. A ~4-line fix. Included in the plan as
   a standalone bug fix because it is cheap, sits in `sessions/`, and corrupts the shelf's
   own module labels if left.
3. **69 stub `experiment.py` files** and 0-of-115 written `log.md`. Tracked separately in
   the doing-leg backfill; not part of this design.
4. **`coach/` fossils** — dead token shop, frozen streak readout in
   `01-transformers/days/check.py:37`, 64 RL notebooks that `KeyError` on cell 0, a
   3-month-stale `DASHBOARD.md`. Out of scope by user decision.

---

## 9. Files touched

| File | Change |
|---|---|
| `sessions/index.html` | add `TOYS` array, shelf renderer, shelf CSS; reuse `.build-embed` receiver |
| `sessions/viz/<new>.html` | one new page per widget built, plus extractions from inline sliders |
| `sessions/m03-attention/day-05-positional/lesson.html` | `%%% viz` block (via compiler) |
| `sessions/m04-first-model-mlp/day-0{2,4,5,6}/lesson.html` | `%%% viz` block (via compiler) |
| `sessions/index.html` (separate commit) | 4 orphaned quest-id corrections |

Not touched: `coach/**`, any notebook, any `experiment.py`, any `data-quest-id`.
