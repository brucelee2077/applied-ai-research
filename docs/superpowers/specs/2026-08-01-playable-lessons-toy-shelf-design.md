# Playable Lessons and the Toy Shelf — Design

**Date:** 2026-08-01
**Branch:** `build/capability-spiral`
**Status:** Approved by user; revised after spec review
**Scope:** `sessions/` only. `coach/` is explicitly untouched.

---

## 1. What this is, and what it replaced

The user asked for "a reward system to make the learning more fun and engaging." An audit
(11 agents, every load-bearing claim adversarially verified) showed that a reward system
was the wrong answer, and the user agreed. This document specifies what we build instead.

**The conversation converged in five steps:**

1. Focus on `sessions/` only, not the `coach/` notebook engine.
2. The goal is **"make the hour itself fun"** — not proving learning, not building a
   daily-return habit, not driving module completion.
3. Within "fun," the chosen register is **play — things to manipulate.**
4. Coverage strategy is **just-in-time**, chosen with the reason *"because I don't know
   what we will build."*
5. Progress should be **playable too** — specifically the **Toy Shelf** (option A of three
   mockups).

Step 4's stated reason is load-bearing. It rules out designing a widget taxonomy up front,
which an earlier draft of this design proposed. See §4.1.

---

## 2. Evidence this is the right problem

All figures re-derived against the working tree at commit `4df9e69`, counting **tracked
files only**.

| Finding | Evidence |
|---|---|
| **211 of 262 lesson pages have nothing to manipulate** | `grep -lE "<iframe\|<input[^>]*type=[\"']?range"` over the 155 tracked `sessions/m*/day-*/lesson.html` → 50 hit; over 107 flat `sessions/week-m*/day-*.html` → 1 hit. **51 interactive, 211 inert, 262 total.** |
| Whole modules are inert | m05a 0/8, m05b 0/4, m06 0/8, m07 0/6, m12 0/11, m14b 0/6, m15b 0/6, m26 0/12, m29 0/8 |
| The daily surface pays nothing that accumulates | `grep -rlE '\bXP\b'` over all lesson pages → **0 files**. Lesson state is `{done:{<section>:true}}` (`sessions/m02-the-neuron/day-01-single-neuron/lesson.html:816`), plus a `sr` key on 46 pages (§8.1) that nothing reads. |
| Play is the lever the project already committed to | `sessions/_compiler/workflows/lesson_build.js:80` — "CULTIVATING THE READER'S INTEREST IS THE #1 GOAL at this early stage" |
| The widget infrastructure exists | 22 viz pages in `sessions/viz/` (21 tracked), `d3.v7.min.js` vendored locally, `%%% viz` widget at `sessions/_compiler/v8lib.py:224` emitting a `.build-embed` iframe plus a visible "open full screen" link (`v8lib.py:229-230`) |
| **All 22** viz pages already send their height | `grep -c postMessage sessions/viz/*.html` → 22/22. No sender retrofit needed. |
| The user is at m02 Day 2 | Chrome Profile 1 localStorage: `frontier-lesson:wf2-d02-activations`; `wf2-d01-neuron` has `c1`–`c13` sections done; files mtime 2026-08-01 14:52 |

**Two methodological traps, both hit during this work and both to be avoided by any future
census:**

1. `grep -i xp` matches *"experiment"*. `grep "build-embed"` matches the CSS **class
   definition** present in every lesson's own `<style>` block (e.g.
   `m07-thinking-in-jax/day-04-jit/lesson.html:226`), which reports all 23 modules as fully
   interactive. Match real tags — `<iframe`, `<input ... type=range` — never class names.
2. `find sessions -name lesson.html` returns **159**, four of which are untracked scratch
   copies (`sessions/_compare/m02-day02-{gold,engine}/`,
   `sessions/_coldgen/{mlp-mnist,m02-day02-activations}/`). All four match the interactivity
   regex. Use `git ls-files`.

---

## 3. Goals and non-goals

### Goals

- **G1.** A learner opening any lesson in the module they are currently on can change
  something with their hands and watch the consequence.
- **G2.** Progress, seen on the hub, is itself something to play with rather than a
  percentage to read.
- **G3.** No work is done for lessons the learner has not reached.

### Non-goals — deliberately excluded

- **No XP, levels, tokens, streaks, shields, or boss battles.** The user stated they
  dislike the existing `coach/` system. None of it is ported.
- **No *new* badges.** The 12 existing hub milestone badges (`sessions/index.html:915-927`)
  stay exactly as they are; §6 requires them to keep computing identically.
- **No fabricated social proof.** The hub's own comment (`index.html:825`, "every value
  below is derived live from localStorage") is the standard; `coach/`'s
  `fictional_cohort_size: 1247` is the anti-pattern.
- **No changes to `coach/`.** Not extended, not bridged, not deleted.
- **No spaced-repetition or retention work.** See §8.1.
- **No teaching content is rewritten.** No lesson is resequenced or removed and no prose is
  re-authored. Six m02 lessons have an existing inline widget **relocated**, which is
  content-preserving; that is the only edit to already-complete lessons (§4.3).

---

## 4. Design

Three components. Component C is the only one that is mostly new code.

### 4.1 Component A — the just-in-time rollout rule

Not code. A rule governing when play gets built.

- **Trigger:** the learner finishing a module. Not a schedule, not a batch.
- **Increment:** exactly one module — the next one. Never "all inert pages."
- **Horizon:** nothing is built for a module the learner has not reached. `m17b` stays
  inert until they are working through `m17a`.
- **Reuse is discovered, not designed.** Build what each lesson needs. Only on the *third*
  time the same widget shape is needed does it become a settings-driven reusable one. This
  follows directly from the user's stated reason for choosing just-in-time.

**First increment, from the verified census:**

| Module | Playable now | Lessons needing play |
|---|---|---|
| m02 ← current position | 9/9 | 0 — already the best module in the repo |
| m03 · attention | 4/5 | **1** — `day-05-positional` |
| m04 · first-model-mlp | 2/6 | **4** — `day-02-backward-pass`, `day-04-training-loss-dropout`, `day-05-pytorch-version`, `day-06-why-pytorch` |

Five lessons covers the learner through the end of m04. The next wall (m05a 0/8, m05b 0/4,
m06 0/8, m07 0/6 = 26 lessons) is three modules away and is not planned now, by design.

### 4.2 Component B — the bar for newly built widgets

These five rules govern **widgets we author from now on**. They are an authoring standard,
**not** a membership test for the shelf — §4.3 explains why five existing viz pages that
fail rule 1 still belong on it.

1. **Immediate consequence.** Drag, don't submit. No button gates the feedback.
2. **It is the lesson's one mechanism** — not an illustration beside it. Reference
   implementation: `m02/day-01-single-neuron`, whose 4 range sliders drive `<svg id="db-svg">`
   via `db-line`/`db-out` — drag the weight, the boundary tilts.
3. **It can be broken.** The learner can drive it into a state where the mechanism visibly
   fails. This is what static images cannot do, and where the learning is. This is the
   reviewable rule; it goes on the per-widget checklist explicitly.
4. **It works offline.** Plain inline SVG, or the vendored `sessions/viz/d3.v7.min.js`.
   No CDN — public CDNs return 403 in this environment.
5. **It lands through the deterministic compiler, not the authoring loop.** Concretely:
   hand-edit the lesson's `source.md` to add a `%%% viz` block, then run
   ```
   python3 sessions/_compiler/compile_lesson.py <path>/source.md
   ```
   which is free, offline, and *"Deterministic + idempotent: same (source.md, donor) ->
   byte-identical output."* **Do not** re-run `sessions/_compiler/workflows/lesson_build.js`
   — that is the ~$300/day LLM authoring loop, and it re-authors the whole lesson body,
   which would violate §3's no-rewrite non-goal. All five target lessons already have a
   `source.md` (verified: m03 5/5, m04 6/6), so the cheap path is available for every one.
   `data-quest-id` is never touched, so existing progress survives
   (`sessions/COACH_STYLE_GUIDE.md:151-179`).

### 4.3 Component C — the Toy Shelf

The hub gains a shelf of the widgets the learner has met, each replayable in place. It is
an **addition** at the top of the existing `.dash` section — the ring, capability bars,
goal cards and 12 badges all remain, pushed down, and must keep computing identically (§6).

#### Data model

One new array in `sessions/index.html`, a sibling of `MODULES` (declared at `:335`):

```js
// name, module label, unlocking quest-id, what you can do with it, source page
var TOYS = [
  ['Broadcasting',      'm01 · Day 3', 'wf1-d03-broadcasting','stretch two shapes to fit', 'viz/broadcasting.html'],
  ['Matmul shapes',     'm01 · Day 4', 'wf1-d04-matmul',      'multiply matrices',         'viz/matmul.html'],
  ['Activation curves', 'm02 · Day 2', 'wf2-d02-activations', 'compare ReLU vs sigmoid',   'viz/activation-derivatives.html'],
  ['XOR limit',         'm02 · Day 2', 'wf2-d02-activations', 'break a single neuron',     'viz/xor-limit.html'],
  // ...
];
```

`wire_index.py` cannot clobber this: its id regex is `\['(w\d{2}-d\d{2})'`, matching only
entries whose *first* field is `wNN-dMM`. `TOYS` rows begin with a display name.

**Shelf inventory: 21 pages, not 22.** `sessions/viz/leaky-slope.html` is untracked and
referenced by zero lessons, so it has no unlocking quest-id and would 404 on GitHub Pages.
It is excluded until committed and embedded.

**Five committed viz pages have no range slider** — `attention-heatmap`,
`attention-multihead`, `attention-pipeline`, `broadcasting`, `softmax-scaling` — and are
button-driven click-throughs (`sessions/batch_1_rollout_plan.md:32` describes
`broadcasting` exactly that way). They **do** go on the shelf: they are still something the
learner manipulates, and excluding them would empty the shelf of m01 entirely. §4.2's rule 1
constrains what we *build next*, not what already exists. `broadcasting.html` is
deliberately kept as the first `TOYS` entry despite failing rule 1.

**Shared-toy ownership.** `viz/xor-limit.html` is embedded by three m02 lessons;
`viz/matmul.html` by two. A `TOYS` row holds exactly one quest-id and one hardcoded module
label, so the rule is: **the earliest lesson that embeds it owns it** — first encounter
wins. `matmul` is therefore labelled `m01 · Day 4`, not `m02 · Day 3`.

#### Unlock rule

A toy is unlocked when its owning lesson has **any** completed section:

```js
var s = JSON.parse(localStorage.getItem('frontier-lesson:'+qid) || 'null');
var unlocked = !!(s && s.done && Object.keys(s.done).length > 0);
```

This deliberately differs from the hub's `pillStatus()` (`index.html:677-688`), which
requires `done.produce || done.verdict`. Rationale: a learner meets a toy mid-lesson, so
requiring completion would lock a toy they have already played with. The cost is that a toy
can unlock slightly before its section is reached. Chosen for generosity. `pillStatus()` is
**not** modified — the ring and capability bars keep their stricter definition.

#### Rendering and interaction

- **New markup:** a `<div id="shelf">` container plus a `<div id="toyPlay">` panel, inserted
  as the first children of `<section class="dash" id="dash">` (`index.html:274`).
- Heading `<n> toys on your shelf`; subtitle is the forward-looking gate line
  (`"5 to clear the m02 gate"`). See the hoist note below.
- Locked toys render dashed and dimmed with the module that unlocks them — anticipation,
  not a scold.
- **Interaction, specified because it changes the layout at 40 toys:** clicking an unlocked
  toy opens **one** panel (`#toyPlay`) *below* the grid — the grid never reflows. Only one
  toy is open at a time; clicking the open toy again collapses it. The `<iframe>` is created
  **lazily on click** and destroyed on collapse, so N toys cost N zero iframes at rest.

#### The iframe receiver is new code

`sessions/index.html` currently contains **zero** occurrences of `iframe`, `build-embed`,
`postMessage`, or a `message` listener — all verified by grep. The auto-resize receiver is
per-lesson boilerplate, not shared infrastructure. The shelf must therefore **implement its
own receiver**, matching the senders exactly:

```js
// sender, in all 22 viz pages: window.parent.postMessage({ type:'viz-height', px:h }, '*')
// receiver the shelf must add:
window.addEventListener('message', function(e){
  var d = e.data;
  if (!d || d.type !== 'viz-height' || typeof d.px !== 'number') return;
  frame.style.height = Math.min(Math.max(d.px, 320), 3200) + 'px';   // same clamp as lessons
});
```

#### Fallback for blocked `file://` iframes

A nested `file://` iframe can be blocked by the browser regardless of correct code —
recorded in-repo at `sessions/_compiler/compile_lesson.py:96-98` ("serve over http to
view") and at `m01-shape-of-data/day-03-broadcasting-dtypes/lesson.html:751-756` (Safari
treats each local `file://` path as its own origin and suppresses the paint). Every toy card
therefore carries a visible **"open full screen"** link to the standalone page, mirroring
what `render_viz` already emits (`v8lib.py:229-230`). This is the real fallback; a "no-JS
fallback" would be meaningless here because the entire hub is rendered from JS arrays.

#### Two kinds of toy, and the extraction backlog

- **Standalone** (`sessions/viz/*.html`) — shelf-able immediately. 21 committed.
- **Inline** — 24 lesson pages build their widget directly in the page (m02 Day 1 has 4
  sliders, m03 Day 2 has 8). Not shelf-able until lifted into their own file.

**Extraction is not a pure lift.** In `m02-the-neuron/day-01-single-neuron/source.md:427-470`
the `%%% svg` block holds the `<svg id="db-svg">`, the `<input type="range">` labels, **and**
the `(function(){ var w1=document.getElementById('db-w1'); … })()` that drives them — and it
relies on the lesson's own palette (inline `accent-color:#9A5A12`, `color:#5A544E`, hardcoded
SVG fills) which a standalone page does not have. Each extraction is therefore: move the
block → add the `viz-height` sender → **re-establish page chrome and theme variables** →
replace the source block with `%%% viz` → recompile.

Extraction is just-in-time like everything else, with **one exception**: m02's six slider
lessons (`day-01`, `day-04`, `day-05`, `day-07`, `day-08`, `day-09`) are already-completed
work, and extracting them is what makes the shelf feel populated on day one instead of
showing two toys. That is an explicit, scoped one-off — and it is why §3's no-rewrite
non-goal is worded as *teaching content*.

#### The gate-line subtitle requires a small hoist

The subtitle's data is the `gate` object (`{m, remaining, total, href}`) computed inside an
anonymous IIFE at `sessions/index.html:859-904` and never exposed. `render()` (`:785`) is one
monolithic function. Two options; **we take (b)**:

- (a) duplicate the gate-scan loop in the shelf renderer — no edit to existing code, but two
  copies of the same logic that can drift.
- **(b) hoist `gate` out of the IIFE into `render()`'s scope** so both the existing goal card
  and the shelf read one value. This *is* a modification to existing hub JS, so it is listed
  in §9 and carved out explicitly in §6.

---

## 5. Constraints the implementation must respect

Each is sourced, not assumed.

| Constraint | Source |
|---|---|
| No server. Lessons run `file://` locally and `https://` on GitHub Pages; **no lesson page calls `fetch(`** (the string does occur in `viz/d3.v7.min.js` and in test/audit files). | `sessions/learning_loop_gap_audit.md:53` |
| Public CDNs are 403. Vendor or inline all JS. | memory `cdn-blocked-vendor-js-locally.md` |
| Browser state is localStorage-only, per-origin, per-machine, and lives in the Chrome profile — **not in the repo**, so it is unbacked-up rather than "gitignored." | verified single `file://` storage area in Chrome Profile 1 leveldb |
| `data-quest-id` is frozen. Changing one wipes that lesson's progress. | `sessions/COACH_STYLE_GUIDE.md:151-179` |
| Lesson edits go through `compile_lesson.py` (free, deterministic), not `lesson_build.js`. | §4.2 rule 5 |
| `sessions/index.html` is hand-maintained and safe to edit directly. | last 8 commits to it are hand edits; `wire_index.py` is a field-filler that is currently a no-op on this file |
| Regeneration via the authoring loop is expensive: $300/day ceiling, and each quota-day buys ~5–6 days of rebuild. | `sessions/concept_body_engagement_rerun.md:156-162` (ceiling) and `:186` (throughput) |
| Interest-first outranks rigor and coverage at this stage. | `lesson_build.js:80`; memory `interest-first-at-early-stage.md` |
| Lesson bodies are Reader A only — no interview register. | `CLAUDE.md:9-10, 20, 39` |

---

## 6. Verification

There is no usable browser automation in this sandbox — `chrome-devtools-axi`, Playwright,
and AppleScript are all blocked, and both `start-server.sh` and `lavish-axi` fail with
`listen EPERM`. Verification is static plus jsdom.

1. **`node --check`** on every new or modified viz page's script block, and on the shelf
   renderer extracted from `index.html`.
2. **New jsdom harness for the shelf** (written fresh — the previously used `/tmp/hubverify`
   no longer exists; `sessions/nav_audit.py` does and is the closest live pattern). Seed a
   fake `localStorage` with known `frontier-lesson:<qid>` values; assert the unlocked/locked
   partition, the toy count, single-panel-open behaviour, and that collapse destroys the
   iframe.
3. **Compiler gates on every recompiled lesson.** m02/m03/m04 are V9 `mode: concept`, so
   `compile_lesson.py:56,69-71` dispatches to **`concept_shell_gate.py`**, not
   `shell_invariant_gate.py` — the latter hard-asserts exactly 7 `module-section`s and 7
   `.gotit` buttons (`shell_invariant_gate.py:45-46,60`) and would fail on
   `day-01-single-neuron`, which has 15 of each. **`visual_integrity_gate.py`**
   (`compile_lesson.py:93-101`) is the gate that actually verifies an embed's file exists and
   its height-sender protocol has not drifted; it must pass on every extraction.
4. **Interactivity census** re-run after each module's pass, using `git ls-files` plus
   real-tag matching, confirming the target module moved to n/n. Both traps in §2 are the
   reason this is specified precisely.
5. **Toy-shelf integrity:** assert every `TOYS[i][4]` path exists on disk **and is tracked
   by git**, and every `TOYS[i][2]` quest-id appears as a `data-quest-id` in some lesson
   page — so a shelf entry can never point at nothing. (This check would have caught a
   `wf1-d03-broadcast` / `wf1-d03-broadcasting` typo made while drafting this spec.)
6. **Regression:** the ring, the 8 capability bars, the 2 goal cards, and the 12 milestone
   badges must render identical values before and after. **One sanctioned exception:** the
   `gate` object moves scope per §4.3's hoist. Its *value* must be unchanged; assert the
   goal card's rendered text is byte-identical across the change.
7. **`scripts/validate_notebook.py`** is not applicable — no notebooks are touched.

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **The shelf is only as good as the toys on it.** Charming at 6; unproven at 40. | High — it is the core bet | The user was shown this risk explicitly and accepted it. Re-evaluate after m04, when the shelf holds ~12. Fallback is mockup option C (ring + can-do strip), which is nearly free. |
| Extraction changes lesson rendering. It is a **replacement**, not additive. | Medium | Verified by `concept_shell_gate.py` (every concept section still has a real visual) plus `visual_integrity_gate.py` (the embed resolves and its sender protocol holds). Compare compiled output before/after; only the widget region may differ. |
| Extracted widgets lose the lesson's palette and render unstyled. | Medium | Treat page chrome + theme variables as a required extraction step, not an afterthought (§4.3). Visual check via the full-screen link, since iframes may be blocked locally. |
| Just-in-time means meeting inert lessons if the learner outruns the build. | Medium | The rule triggers on module completion, giving one module of lead time; m02→m04 is only 5 lessons. |
| A widget satisfies rule 1 but not rule 3 and becomes decoration. | Medium | Rule 3 is an explicit per-widget checklist item. |
| Unlock-on-any-section feels loose. | Low | Deliberate; trivially revertible to `pillStatus()`. |
| The hoisted `gate` silently changes the existing goal card. | Low | §6.6 asserts byte-identical rendered text. |

---

## 8. Known deferrals

Recorded so they are not lost, and explicitly not built now:

1. **The retention loop is write-only.** `srReview()` writes `{ease, interval, next, reps}`
   into `frontier-lesson:<qid>` on 46 pages; `dueConcepts()` exists only at
   `sessions/_compiler/shells/js/sr.js:44` and is inlined into **zero** lessons. The learner
   currently has `prereq-multiply` due 2026-08-07 that nothing will surface. Deferred
   because it serves the "come back tomorrow" goal the user explicitly did not choose.
2. **4 orphaned quest-ids.** `w03-d06-examrehearsal`, `w02-d06-distckpt`,
   `w04-d06-consolidation`, `w17-d01-consolidation` each appear as `data-quest-id` on exactly
   one lesson and **zero** times in `index.html`, so m08/m09a/m09c/m14b can never reach 100%.
   A ~4-line fix, included in the plan as a **standalone commit** because it is cheap and
   sits in `sessions/`. *Correction to an earlier draft:* this does **not** corrupt shelf
   labels — those are hardcoded literals and the unlock read goes straight to localStorage.
   The real effect is that the hub's "next review gate" goal card is stuck at m08 forever.
3. **69 stub `experiment.py` files** and 0-of-115 written `log.md`. Tracked separately in the
   doing-leg backfill; not part of this design.
4. **Gamification fossils outside `sessions/`.** A dead token shop and unreachable badges in
   `coach/`, plus two **root-level** artifacts: `01-transformers/days/check.py:37`, which
   prints a frozen `Streak: 1 day(s)`, and a 3-month-stale `DASHBOARD.md`. All out of scope
   by user decision; noting the paths because they are not under `coach/` and so fall outside
   this document's scope line either way.

---

## 9. Files touched

| File | Change |
|---|---|
| `sessions/index.html` | **new:** `TOYS` array, `#shelf` + `#toyPlay` markup inside `.dash`, shelf renderer, shelf CSS, and its own `viz-height` message receiver. **modified:** hoist `gate` out of the `:859-904` IIFE. |
| `sessions/viz/<new>.html` | one new page per widget built (5 for m03+m04), plus 6 extracted from m02's inline sliders |
| `sessions/m03-attention/day-05-positional/source.md` + recompiled `lesson.html` | add `%%% viz` block |
| `sessions/m04-first-model-mlp/day-0{2,4,5,6}/source.md` + recompiled `lesson.html` | add `%%% viz` block |
| `sessions/m02-the-neuron/day-0{1,4,5,7,8,9}/source.md` + recompiled `lesson.html` | replace inline `%%% svg` widget block with `%%% viz` pointing at the extracted page — content-preserving |
| `sessions/index.html` (separate commit) | 4 orphaned quest-id corrections |
| new jsdom test harness | shelf unlock/count/interaction assertions |

Not touched: `coach/**`, any notebook, any `experiment.py`, any `data-quest-id`, any lesson
prose.
