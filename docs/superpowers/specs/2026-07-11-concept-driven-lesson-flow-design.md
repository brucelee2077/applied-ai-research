# Concept-Driven Lesson Shell (V9) — Design Spec

**Date:** 2026-07-11
**Author:** ruifengli (+ Claude)
**Status:** Draft for review (supersedes the V8.1 "re-purpose 7 slots" spec of the same date)
**Scope of this session:** Build a new concept-driven lesson shell (V9): a lesson body = N per-concept units, each `intro → its own visual → build-up`, instead of the fixed 7-section template. Deliver the compiler mode + donor template + gates, and rebuild **Module 2 Day 2 (Activation Functions)** as the first V9 lesson to prove it. **Day 2 only** as the shipped lesson; m02 Days 1/3–9, m03, and the paused m04 are out of scope this session (roll-forward is future work).

---

## 1. Problem

V8 lessons are built on a **fixed 7-section template**: `What is it → Intuition → Playground → Mechanism → Build it up → Quiz → Produce` (DOM ids `s1…s7`). This is *template-driven*: concepts are introduced up front as **text** (What-is-it, Intuition), and the **visuals + build-up are quarantined in one late section** (`s5`, "Build it up"). A reader meets a concept's *name and analogy* long before they *see* it.

Concrete failure — Module 2 Day 2, "Activation Functions": the reader meets `ReLU`, `sigmoid`, `tanh` as words (§1 vocab table), then as analogies (§2 🚰 valve / 🎛️ dimmer cards), then as numbers (§3 playground `array([0,0,0,2,5])`) — and only **sees the actual curves in §4/§5**, after all of that. This contradicts the repo's own `CLAUDE.md` ("Every new concept needs an analogy before any math"; "Every major concept needs at least one diagram") and the user's stated preference for concept-driven, blog/coaching-style material.

### 1.1 Root cause

The failure is **structural, not editorial**. The fixed template has exactly one "build-up + visuals" section (`s5`) for the whole lesson, so *every* concept's picture is deferred to one late block regardless of when the concept is introduced. Re-labeling the 7 slots (the rejected V8.1 approach) cannot fix this because it preserves the single-`s5`-buildup and the 7-section cap. The fix is to make the body a **sequence of concept units**, each carrying its own visual and build-up.

### 1.2 Why now / why it's feasible

- The **m04 rollout is paused mid-loop**, blocked on exactly "a user-flagged visualization issue to fix first" (per memory index). This work is that blocker; fixing the shell here unblocks m04.
- **Feasibility confirmed by code inspection.** The shipped lesson JavaScript is ~90% generic already: progress tracking uses `document.querySelectorAll('.module-section')` keyed on each section's `data-sec` attribute; sidebar nav matches `data-target`↔section `id`; scroll-spy iterates over "all sections"; completion counts sections generically. **None of these hardcode `s1–s7` or the number 7.** A variable number of concept sections works for free for nav, scroll-spy, progress, and completion. The only id-coupled pieces are three widget engines (`getElementById('s3')` playground, `getElementById('s5')`/`#build` build-reveal, `getElementById('s6')` quiz) and the gates (which assert exactly 7). Those are the surgical targets.

---

## 2. Goal

Introduce a **V9 concept-driven shell** in which a lesson is:

```
hero (curiosity hook)
  → concept unit 1   (intro → visual → build-up  [+ optional inline run-demo])
  → concept unit 2   (intro → visual → build-up)
  → …                (as many concept units as the topic needs)
quiz (one section, all questions)
produce (discovery artifact)
fin (completion banner)
```

Each **concept unit** is a normal tracked `.module-section` (so the existing progress/gamification works and improves — more, smaller "got it" steps). Inside every concept unit, in order:
1. **intro** — the concept in plain words (may include a jargon gloss).
2. **visual** — its own depiction, **inline**, appearing immediately: a static labeled SVG, or an interactive iframe where the behavior is explorable. *This is mandatory for every concept unit.*
3. **build-up** — the mechanism/why, building on the intro + visual.
4. **(optional) inline run-demo** — a click-to-run mini-console folded into the concept it belongs to (e.g. ReLU's `▶ run` lives in the ReLU concept), not a separate playground section.

### 2.1 Decisions locked (from brainstorming)

- **Break the fixed shell** into a concept-driven body (not re-label 7 slots).
- **Inline visual per concept** (not scroll-reveal) — simplest, most robust, delivers intro→visual→buildup directly.
- **Distribute run-demos into their concepts**; the **quiz stays a single section** at the end.
- **Every concept unit must ship a visual.** (A concept unit introduces a concept; if a unit is purely a code-run or a definition with no visual object, it still must carry at least a minimal diagram — the whole point is concept+picture together. This is stricter than V8.1's "only visual-object beats," and is appropriate because in V9 the author chooses what becomes a concept unit.)

### 2.2 Success criteria

1. Day 2 renders as `hero → 7 concept units → quiz → produce → fin` (see §6 for the 7 concepts), each concept unit showing its own curve/diagram **inline, immediately after its intro**, before any later concept.
2. The reader sees ReLU, sigmoid, and tanh curves **in their own concept units**, each next to where the concept is named — never as "words now, picture much later."
3. The V9 JS engine drives nav, scroll-spy, progress, completion, inline demos, and quiz over a **variable** number of sections with no hardcoded `s1–s7`.
4. New/updated gates PASS on Day 2: structural gate (V9 invariants), reader-flow gate (every concept unit has a visual), notebook-smoothness gate (hero vs notebook), and recompile is idempotent (byte-identical for same source+donor).
5. The three skills carry the V9 concept-unit rule, in **both** skill locations, kept byte-consistent.
6. **No regression:** shipped m02 (Days 1/3–9) and m03 lessons are untouched and byte-identical (they keep their V8 donors + verbatim mode); V9 changes are additive.
7. Independent adversarial verification confirms 1–6, including that the gates actually FAIL on a deliberately-broken V9 variant (a concept unit with no visual).

---

## 3. Non-Goals (explicit)

- **Not** re-flowing m02 Days 1/3–9 or m03 to V9 this session (future backport).
- **Not** resuming the m04 rollout this session (V9 unblocks it; the rollout itself is later).
- **Not** deleting or rewriting the V8 verbatim path — it stays for the already-shipped lessons. V9 is an additional mode.
- **Not** authoring brand-new interactive viz assets from scratch — Day 2 reuses existing inline SVGs (ReLU/sigmoid/tanh/collapse/cure) and existing iframes (`viz/activation-derivatives.html`, `viz/xor-limit.html`).
- **Not** scroll-reveal animation for the per-concept visuals (inline was chosen).
- **Not** changing the shipped lesson's quest-id for Day 2 (`wf2-d02-activations` stays; the page content changes but its identity and localStorage key do not).

---

## 4. Architecture

### 4.1 Source format (V9 mode)

`source.md` front-matter declares `mode: concept` (new). The body is authored as typed blocks parsed by the existing `v8lib.parse_blocks` (`@@@ ` delimited):

```
@@@ hero
@kicker Module 2 · Train · Day 2
@lede …curiosity hook…
@goal …by the end you can…

@@@ concept id=c1 tag="The problem" title="Straight + straight is still straight" gotit="Got it"
intro prose (plain words; may use [[term||gloss]] and %%% jargon)
%%% svg
<svg …two-rulers-collapse…></svg>
%%%
build-up prose (the mechanism/why)

@@@ concept id=c2 tag="The fix" title="A bend between the layers" gotit="Got the bend"
intro prose
%%% svg
<svg …bent shape…></svg>
%%%
build-up prose

@@@ concept id=c3 tag="Meet ReLU" title="ReLU — a one-way valve" gotit="Met ReLU"
intro prose
%%% svg
<svg …ReLU curve…></svg>
%%%
build-up prose
%%% demo id=relu label="run it"
code: relu(np.array([-3,-1,0,2,5]))
out:  array([0, 0, 0, 2, 5])
take: ReLU zeros negatives, passes positives — one cheap max.
%%%

… concepts c4 (sigmoid), c5 (tanh), c6 (when bends break — %%% viz activation-derivatives),
   c7 (XOR — %%% viz xor-limit) …

@@@ quiz id=quiz tag="Check" title="Four questions" gotit="Checked"
%%% quiz
q: … | a:1 | opt: … | opt: … | opt: … | opt: … | fb: …
q: … (×4 total)
%%%

@@@ produce id=produce tag="Produce" title="Watch the collapse, then break it" gotit="Done"
discovery-framed artifact prose (predict → run → observe), reusing the Option-A/Option-B pattern

@@@ fin
@title Module 2 · Day 2 complete! 🏆
@body Nice work …
```

New block types / widgets to add to `v8lib`:
- `@@@ concept` — renders a `.module-section` with `sec-head` (num auto-incremented, tag, title) + `sec-body` (rendered markdown) + one `.gotit`. `data-sec` = the `id` (e.g. `c1`).
- `@@@ hero`, `@@@ quiz`, `@@@ produce`, `@@@ fin` — section wrappers (hero already exists in clean mode; quiz/produce are `.module-section`s; fin already exists).
- `%%% svg` — inline raw-SVG visual widget (passes the SVG through, wrapped in `.build-viz` for consistent styling). This is the mandatory per-concept static visual.
- `%%% viz src=… caption=…` — existing behavioral-iframe widget (already implemented as `render_viz`), reused for interactive concepts (c6, c7).
- `%%% demo id=… label=…` — inline run-demo widget: renders a small console with a run button that reveals `out` + `take` (one demo, scoped to its concept).
- `%%% quiz` — quiz data widget: renders the 4 Q blocks (replaces the old `var QS` JS-data approach with authored source; the quiz engine reads the rendered DOM).

### 4.2 Compiler (`compile_lesson.py` + `v8lib.compile_html`)

- Add a **concept mode** path: when `meta.mode == 'concept'`, build the body by concatenating rendered `hero` + each `concept` + `quiz` + `produce` + `fin` into the donor's content area, and generate the sidebar nav from the concept/quiz/produce list (reuse `render_sidebar_nav`, which already iterates a `meta['sidebar']`-style list — here derived from the concept blocks).
- The V8 verbatim/clean paths remain unchanged for existing lessons.
- Determinism preserved: same `(source.md, donor)` → byte-identical output (existing guarantee; keep it).

### 4.3 Donor template (new V9 shell)

Create `sessions/_compiler/shells/v9-base.donor` (one reusable base, not per-day) containing:
- The existing head/CSS (reused verbatim from the V8 donor — themes, sidebar, `.module-section`, `.gotit`, `.build-viz`, `.build-embed`, quiz, prompt styles all already exist).
- A content placeholder region the compiler fills with hero + concepts + quiz + produce + fin.
- A **generalized JS engine** (see §4.4).
- The shared glossary-tooltip script and the viz-iframe auto-resize script (both already generic — reused verbatim).

### 4.4 Generalized JS engine (the only real JS work)

Start from the existing engine (already generic for progress/nav/scrollspy/completion — keep as-is) and change the three id-coupled widget engines to be selector-driven:

- **Inline run-demos:** replace the `getElementById('s3')` playground engine with a generic one: `querySelectorAll('.demo')`, each demo self-contained (its own run button reveals its own `out`+`take`). No "saw all 3" cross-section gating; each concept's `.gotit` is enabled normally (or on demo-run if present).
- **Build-reveal:** inline visuals need **no** reveal engine (they're always visible). Drop the `#build`/`s5` scroll-reveal engine entirely for V9. (Static SVGs render immediately; iframes keep their existing auto-resize script.)
- **Quiz:** change `getElementById('s6')` to select the quiz section by a class/`data-sec="quiz"` marker so it is not tied to a fixed id. The quiz question DOM is emitted by the compiler from `%%% quiz` (not injected from a `var QS`), and the engine wires click-to-answer over `.q` blocks found within the quiz section.

### 4.5 Gates

- **New `concept_shell_gate.py`** (V9 replacement for `shell_invariant_gate.py`; the V8 gate stays for V8 lessons). Asserts V9 invariants on the compiled HTML:
  - quest-id present and matches donor/front-matter (frozen).
  - `≥3` concept sections (`.module-section` whose `data-sec` starts `c`), each containing **at least one visual** (`<svg` or `.build-embed` iframe) **and exactly one `.gotit`**.
  - exactly one quiz section with 4 questions (4 `.q` blocks / 4 option groups) and one produce section.
  - sidebar `data-target`s are in 1:1 correspondence with section `id`s (nav parity) — generic, not "8 fixed".
  - localStorage keys (`frontier-lesson:`, `frontier-theme`), `.fin`, footer present; no unresolved `@@@`/`%%%`/marker leakage.
  - recompile idempotent (byte-identical).
- **Extend `reader_flow_gate.py`** with the V9 **concept-visual check**: in concept mode, every `@@@ concept` block's raw source must contain a visual marker (`<svg`, `%%% svg`, `%%% viz`, or `.build-embed`). A concept block with no visual → FAIL (blocks the write, exit 2). In non-concept modes this check is a no-op (existing lessons unaffected). This is the mechanical form of "every concept ships its picture."
  - Reuse the existing `run(meta, blocks, spine_word=None)` signature, `ok=[True]`/`fail()` accumulator, and `pass `/`FAIL `/`warn `/`N/A ` message style. The check reads **raw** block source (concept blocks expose their raw lines directly via `parse_blocks`, so no tag-stripping problem — unlike the V8 region path).
- **`notebook_smoothness_gate.py`** — unchanged; still compares the hero's first screen to the notebook yardstick. Day 2 keeps `notebook_yardstick: 00-neural-networks/fundamentals/03_activation_functions.ipynb`.

---

## 5. Component: skill edits

Three skills, two byte-identical locations each (6 files), kept in sync:
- Active: `.claude/skills/<skill>/SKILL.md`
- Bundle: `frontier_lab_refactor_skills_v8/skills/<skill>/SKILL.md`

- **`frontier-lesson-builder`** — replace the fixed "Reader Flow Blueprint" template with the **V9 concept-unit model**: a lesson body = N concept units, each authored as `intro (plain words) → its own inline visual → build-up`, plus hero / quiz / produce / fin. Add to the Learning Barrier Gate P0 list: "a concept unit with no visual" and "a concept's picture deferred to a later unit."
- **`frontier-visual-evidence-builder`** — add the **depict-in-unit rule**: every concept unit carries its own inline visual (static labeled figure minimum; interactive where behavior is explorable); analogy is not depiction; a lesson must not defer a concept's picture to a later unit or to one shared late "build" section.
- **`frontier-refactor-qa`** — update the Visual/Evidence + Shell gates to the V9 invariants: "concept unit without a visual" is P0; reference `concept_shell_gate.py` and the reader-flow concept-visual check as the mechanical backstops; the "exactly 7 sections" expectation is replaced by "≥3 concept units + quiz + produce, nav parity."

---

## 6. Day 2 rebuild (first V9 lesson)

Author `sessions/m02-the-neuron/day-02-activations/source.md` in V9 concept mode with **7 concept units**, reusing existing assets:

| Unit | Concept (title) | Inline visual (reused asset) | Build-up |
|---|---|---|---|
| c1 | Straight + straight is still straight | two-rulers/collapse SVG (from current BUILD[2]) | `(x@W1)@W2 = x@(W1@W2)`; depth without a bend is an illusion |
| c2 | The fix is a bend | cure SVG (from current BUILD[3]) | an activation between layers stops the collapse |
| c3 | Meet ReLU | ReLU curve SVG (from current BUILD[0]) + `%%% demo relu` | `max(0,z)`, one-way valve, cheap default |
| c4 | Meet sigmoid | sigmoid curve SVG (from current BUILD[1]) + `%%% demo sigmoid` | squash to (0,1), saturates at the ends |
| c5 | Meet tanh | tanh curve SVG (from current s4 inline SVG) | zero-centered cousin; range (−1,1) |
| c6 | When bends break | `%%% viz activation-derivatives.html` (interactive) | dead ReLU / saturation; log activation stats (staff lens) |
| c7 | One line can't do XOR | `%%% viz xor-limit.html` (interactive) | why depth + a bend earn their keep; frontier payoff (GELU/SwiGLU) |

Then `quiz` (the existing 4 questions, incl. the dead-ReLU diagnostic) and `produce` (the existing collapse-then-cure discovery artifact, Option-A/Option-B). The staff-depth content (failure mode, trade-off, interview line) folds into c6's build-up so Staff Depth still passes.

**Identity preserved:** quest-id `wf2-d02-activations`, localStorage key, nav prev/next hrefs. The page content is intentionally **not** byte-identical to the shipped V8 Day 2 — replacing it is the deliverable.

---

## 7. Re-test / evaluation loop

Heuristic gates BLOCK; LLM judge evaluates qualitatively (decided in brainstorming).

1. Compile: `python3 sessions/_compiler/compile_lesson.py sessions/m02-the-neuron/day-02-activations/source.md --donor sessions/_compiler/shells/v9-base.donor`
2. Reader-flow gate incl. concept-visual check — PASS (every concept has a visual).
3. `concept_shell_gate.py` — PASS (V9 invariants; idempotent recompile).
4. Notebook-smoothness gate — PASS or N/A.
5. Backstops: `node --check` on the emitted `<script>`s; jsdom smoke (nav parity, progress over N sections, one demo runs, quiz answers) if available.
6. **LLM judge** reads compiled `lesson.html`: "Does every concept appear with its own picture, inline, at the moment it's introduced? Is there any concept named without a visual, or any picture deferred?" → `{pass: bool, reasons:[…]}`.
7. If any gate FAILs or judge says no → edit source, recompile, repeat.
8. **Adversarial verification (final):** independent agents confirm (a) each of the 7 concepts shows its visual inline in the compiled HTML in order; (b) V9 invariants intact + idempotent; (c) shipped m02 Days 1/3–9 and m03 are byte-identical (unchanged); (d) the gates actually FAIL on a broken variant (a concept unit stripped of its visual) — proving teeth, not green-by-default.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| New shell breaks shipped m02/m03 lessons | V9 is a separate mode + separate donor. Existing lessons keep V8 donors + verbatim mode; verify byte-identity of Days 1/3–9 + m03 in the adversarial step. |
| Generalized JS engine regresses nav/progress/quiz | Progress/nav/scrollspy are already generic (confirmed by inspection); only 3 widget engines change. Add a jsdom smoke test over a variable-N page + `node --check`. |
| Gate green-by-default (never catches a missing visual) | Concept-visual check reads raw block source (no tag-stripping); adversarial step tests it against a visual-stripped variant to confirm FAIL. |
| Quiz/demo engine coupling to fixed ids leaks through | Select by class/`data-sec` marker, never by `s3`/`s5`/`s6`; the gate asserts nav parity generically. |
| Losing staff-depth / interview grounding in the re-flow | c6 build-up carries the failure mode + trade-off + interview line; Staff Depth gate still runs. |
| Two skill copies drift | Every skill edit applied to both paths in one step; diff-check parity. |
| iframe height sticks (past bug) | Reuse the existing postMessage auto-resize script verbatim; the two iframes already ship the height sender. |

---

## 9. Deliverables (this session)

1. `v8lib.py` extended: `@@@ concept`/`@@@ quiz`/`@@@ produce` block rendering + `%%% svg`/`%%% demo`/`%%% quiz` widgets + concept-mode `compile_html` path.
2. `sessions/_compiler/shells/v9-base.donor` — the reusable V9 shell with the generalized JS engine.
3. `concept_shell_gate.py` (new) + `reader_flow_gate.py` extended with the concept-visual check + tests against a broken variant.
4. `sessions/m02-the-neuron/day-02-activations/source.md` rebuilt in V9 concept mode (+ recompiled `lesson.html`).
5. 6 edited `SKILL.md` files (3 skills × 2 locations), diff-verified identical.
6. Green run of all gates + jsdom/`node --check` backstops + LLM-judge approval + adversarial-verification notes.
7. This spec, committed.

Out of scope (future sessions): backport m02 Days 1/3–9 and m03 to V9; resume the m04 rollout on the V9 shell.
