# Concept-Driven Lesson Flow (V8.1) — Design Spec

**Date:** 2026-07-11
**Author:** ruifengli (+ Claude)
**Status:** Draft for review
**Scope of this session:** Update 3 skills + extend the reader-flow gate + rebuild Module 2 Day 2 (Activation Functions) into concept-beats + prove it with a re-test loop. **Day 2 only** — the other 8 m02 days and m03 are out of scope this session (see Non-Goals).

---

## 1. Problem

The V8 lesson-authoring skill produces lessons on a **fixed 7-slot template**: `What is it → Intuition → Playground → Mechanism → Build it up → Quiz → Produce` (DOM section ids `s1…s7`). This template is *template-driven*, not *concept-driven*: it forces **concepts to the front as text** and **visual depictions to the back**.

Concrete failure (Module 2 Day 2, "Activation Functions"):

| Reader position | What the lesson does | Where the actual picture is |
|---|---|---|
| §1 "What is it" | Names `sigmoid`, `ReLU`, `tanh` in a vocab table (words only) | — |
| §2 "Intuition" | Analogy cards 🚰 valve / 🎛️ dimmer (words only) | — |
| §3 "Playground" | Prints `array([0,0,0,2,5])` (numbers only) | — |
| §4 "Mechanism" | **First curve appears** (tanh SVG, derivatives iframe, comparison table) | `s4` |
| §5 "Build it up" | ReLU / sigmoid / collapse SVG curves | `s5` |

A beginner therefore reads the *names*, the *analogies*, and the *numeric output* of ReLU/sigmoid/tanh **before ever seeing what any of the three curves look like**. This directly contradicts the repo's own `CLAUDE.md` ("Every new concept needs an analogy before any math"; "Every major concept needs at least one diagram") and the user's stated preference for concept-driven, blog/coaching-style material.

### 1.1 Root cause (two layers)

1. **Skill layer.** `frontier-visual-evidence-builder` states the right *principles* — "mental picture visual → mechanism visual", "the first visual in a hard concept should reduce barrier", and lists "activation curve behavior" as requiring plotted evidence — but nothing enforces **adjacency**: that a concept and its depiction ship *in the same beat*. "Mental picture" was satisfied by an **analogy** (🚰🎛️ emoji cards) instead of an actual **picture of the curve**.

2. **Structural layer.** The compiler authors only 4 prose regions (`s1, s2, s4, s7`). The visual-heavy sections are auto-generated from JS data blocks: `s3` (Playground) from `var DEMOS`, `s5` (Build-it-up) from `var BUILD` (which holds the SVG curves), `s6` (Quiz) from `var QS`. In the frozen DOM order `s1→s2→s3→s4→s5→s6→s7`, the curve figures live in `s4`/`s5`, so the earliest a curve can currently appear is *after* the reader has already passed the `s3` playground numbers.

### 1.2 Why this matters beyond Day 2

Per the memory index, the **m04 rollout is paused mid-loop, blocked on "a user-flagged visualization issue to fix first."** This concept-driven-flow work is that blocker. Fixing the skill + gate here is what unblocks m04 cleanly afterward. m02 is a **seed module**: a defect left in it propagates by copy to future modules, so the fix belongs in the skill/gate, not only in one lesson.

---

## 2. Goal

Encode and enforce a single principle:

> **Depict-on-introduction (adjacency rule).** The moment a lesson names or analogizes a *visual object* (a curve, a distribution, a boundary, a geometric/matrix shape), that same concept-beat must **show** it. Each body section becomes a self-contained **concept-beat**: `plain-words idea → its own diagram → what to notice`. A visual object must not be named in an earlier beat than the one that depicts it. An analogy (🚰 valve) is **not** a substitute for a depiction (the actual curve).

**Which beats must ship a visual (decided):** only beats that introduce an inherently *visual object* (curve / distribution / boundary / geometric or matrix shape). A purely definitional or code-run beat is not forced to carry a decorative diagram.

We achieve this **without tearing down the shell**: keep the compiler, the 7 shell sections, the DEMOS/BUILD/QS blocks, the frozen quest-ids, and both existing gates. We *re-purpose* the 7 slots into concept-beats and add one new enforceable check.

### 2.1 Success criteria

1. A reader sees the shape of ReLU, sigmoid, and tanh **in `s1`/`s2`** (before the `s3` playground numbers).
2. `frontier-visual-evidence-builder`, `frontier-lesson-builder`, and `frontier-refactor-qa` each carry the depict-on-introduction rule, in **both** skill locations (`.claude/skills/` and `frontier_lab_refactor_skills_v8/skills/`), kept byte-consistent.
3. `reader_flow_gate.py` has a new **depict-on-introduction check** that FAILs a re-flowed lesson where a declared visual object is named in an earlier beat than the one that depicts it.
4. Day 2 recompiles: reader-flow gate PASS (incl. new check), shell-invariant gate PASS (byte-identity of the shell preserved), notebook-smoothness gate PASS/N-A as before.
5. An LLM judge, reading the compiled `lesson.html`, answers "does a beginner see each concept's picture at the moment it's introduced?" = **yes**, with reasons.
6. Independent adversarial verification confirms 1–5 with no regression to the other 8 m02 days.

---

## 3. Non-Goals (explicit)

- **Not** rebuilding the shell/compiler into variable-length beats (the "new concept-driven shell" option was considered and rejected as too high-blast-radius this session).
- **Not** re-flowing m02 Days 1, 3–9, or m03 this session. They remain byte-identical; the new gate check must **not** retroactively fail them (see §5.3 mode handling).
- **Not** changing quest-ids, sidebar `data-target` ids, DEMOS/BUILD/QS block shapes, the 4-question quiz, `.fin`, or nav wiring.
- **Not** authoring new interactive viz assets from scratch — Day 2 reuses the existing `viz/activation-derivatives.html`, `viz/xor-limit.html`, and inline SVGs already present.
- **Not** unrelated prose polish beyond what the re-flow needs.

---

## 4. Component 1 — Skill edits

All three skills exist in two byte-identical locations that MUST stay in sync:
- Active: `.claude/skills/<skill>/SKILL.md`
- Bundle: `frontier_lab_refactor_skills_v8/skills/<skill>/SKILL.md`

Every edit below is applied to both copies (6 file edits total).

### 4.1 `frontier-visual-evidence-builder`

Add a new top-level rule section, **Depict-on-introduction (adjacency rule)**:

- When a beat names or analogizes a visual object (curve, distribution, boundary, geometric/matrix shape), that beat MUST depict it — a static labeled figure at minimum; an interactive viz where the behavior is explorable.
- A visual object MUST NOT be named or analogized in a beat earlier than the one that depicts it.
- Analogy is not depiction. A 🚰 valve card does not satisfy "show the ReLU curve."
- The static figure is the floor, the interactive viz is the ceiling: prefer static-for-instant-recognition **plus** interactive-for-exploration when a behavior is drag-explorable.

Update the existing **Visual sequence** block so "mental picture visual" is explicitly *a picture of the object*, not an analogy card.

### 4.2 `frontier-lesson-builder`

- Reframe the **Reader Flow Blueprint** from a template list into **concept-beats**: each body section is authored as `plain-words idea → its own diagram → what to notice`, self-contained.
- Add to the **Learning Barrier Gate** P0 list: "a concept-beat that introduces a visual object but is text-only (names/analogizes without depicting)."
- Note the structural constraint for authors: in this shell the compiler auto-generates `s3`/`s5`/`s6` from DEMOS/BUILD/QS, and prose is authored in `s1/s2/s4/s7`; therefore *the early depiction of a visual object must be placed in the authored `s1`/`s2` regions*, ahead of the `s3` playground.

### 4.3 `frontier-refactor-qa`

- Add to the **Visual/Evidence Gate**: "named-but-not-shown" (a visual object named/analogized in a beat that lacks its depiction) and "visual-object beat is text-only" are **P0** for a fresh build; **seed-propagation P1** if the lesson already shipped and the fix is targeted/low-risk.
- Reference the new `reader_flow_gate.py` depict-on-introduction check as the mechanical backstop for this finding.

---

## 5. Component 2 — The gate (mechanical, build-blocking)

Extend `sessions/_compiler/gates/reader_flow_gate.py` with a **depict-on-introduction check**.

### 5.1 Declaring visual objects (front-matter)

Add an optional front-matter key to `source.md`:

```yaml
visual_objects: [relu, sigmoid, tanh]   # tokens whose depiction adjacency is checked
```

Each entry is a lowercase token matched case-insensitively against prose. If the key is absent, the check is a no-op (records `N/A — no visual_objects declared`). This keeps every currently-shipped lesson unaffected until it opts in.

### 5.2 The check

The intent is stronger than "same-or-earlier beat": success criterion #1 requires the curve to be **visible in `s1`/`s2`, before the reader reaches the `s3` playground numbers.** The check therefore encodes *"declared visual objects must be depicted in `s1` or `s2`"*, not merely "depicted no later than first mention."

For each token in `visual_objects`:

1. **Naming location.** Find the first authored prose beat (scan `s1`, then `s2`, then `s4` in DOM order) whose text names the token.
2. **Depiction location.** A beat "depicts" if its **raw source** contains any depiction marker: `<svg`, `class="build-viz"`, `class="build-embed"` (behavioral iframe), or a `%%% viz` widget.
3. **PASS** if the token is depicted in `s1` or `s2` (the early beats that precede the `s3` playground).
4. **FAIL** if the token is named anywhere but not depicted in `s1`/`s2` — this catches both the "named-but-not-shown" defect and the subtler "first curve only appears in `s4`, after the playground" defect (a token whose only depiction is in `s4` FAILs, because the reader still met the playground numbers first).

Note on `s3`/`s5`: these are compiler-generated (DEMOS/BUILD) and not authored prose regions, so the check operates on the authored prose. Because `s3` (playground) sits between `s2` and `s4` in DOM order, requiring depiction in `s1`/`s2` is exactly the mechanical form of "curve visible before the playground."

### 5.2a Helper change required (not optional)

`reader_flow_gate._region_texts()` currently returns **tag-stripped** text for `s1`/`s2`/`s4` (via `_text()` → `_TAG.sub`) and exposes a raw copy for **`s1` only** (`t['s1_raw']`). The depiction markers (`<svg`, `build-viz`, `build-embed`) are HTML tags, so they are erased by tag-stripping and are invisible to the current text fields. Therefore the implementation MUST extend `_region_texts()` to also expose `t['s2_raw']` and `t['s4_raw']` (raw, un-stripped source), and the new check MUST read the `*_raw` fields — not the stripped `t['s1']`/`t['s2']`/`t['s4']`. Building the check against the stripped strings would make it green-by-default (find zero markers always) — precisely the failure the §8 adversarial step exists to catch.

### 5.2b Marker relevance in verbatim vs clean mode

Day-2 is `source_mode: verbatim` and embeds depictions as raw HTML (`<div class="build-embed">…`, `<div class="build-viz">…<svg…`). The `%%% viz` widget marker is a **clean-mode** convenience (compiled by `v8lib.render_viz`) and appears **zero** times in the Day-2 source. So for the Day-2 target the load-bearing markers are `<svg` and `build-embed`/`build-viz`; `%%% viz` is retained in the marker list only for future clean-mode lessons. The check must accept any of the four markers so it works in both modes.

### 5.3 Mode handling (must not break shipped days)

- The check runs only when `visual_objects` is present in front-matter. The 8 untouched m02 days and all m03 days do not declare it → unaffected, byte-identical, never failed.
- Day 2's `source.md` opts in by adding `visual_objects: [relu, sigmoid, tanh]`.
- Emit clear `pass`/`FAIL`/`N/A` messages consistent with the gate's existing message style, and wire the FAIL into the gate's existing `ok` accumulator so a real failure exits non-zero (exit 2) and blocks the compile write.

### 5.4 Gate output contract

- Extend the existing `run(meta, blocks, spine_word=None)` function and its message list; add lines prefixed `pass `/`FAIL `/`warn ` / `N/A ` as the current code does, and route a real failure through the existing `fail()` closure so the `ok` accumulator drives exit code 2 and blocks the compile write.
- Requires the helper change in §5.2a (`_region_texts()` must expose `s2_raw`/`s4_raw`). The check reads raw source, not stripped text.
- The new check must be side-effect-free and deterministic (pure function of source), consistent with the compiler's determinism guarantee.

---

## 6. Component 3 — Day-2 rebuild (concept-beats, same shell)

Edit `sessions/m02-the-neuron/day-02-activations/source.md` **authored regions only** (`s1`, `s2`), plus add `visual_objects` to front-matter. Do not touch DEMOS/BUILD/QS/hero/s4/s7 except as needed to remove now-duplicated late-visual redundancy (optional, minimal).

- **`s1` — relabel toward "Meet the bend / meet the three curves."** Keep the jargon ladder (gate requires it). The instant ReLU/sigmoid/tanh are named, insert a compact **static 3-curve figure** (three small labeled SVGs in the existing tanh-SVG visual language) so the reader sees which shape is which immediately. Curve visuals SHOWN here.
- **`s2` — relabel toward "See them behave."** Keep the valve/dimmer analogy cards (they are good, just no longer the *only* thing before a curve), and move the **interactive `viz/activation-derivatives.html`** embed here (drag `z`, watch the slope) so the intuition beat is explorable — instead of first appearing in `s4`.
- `s3` (playground numbers), `s4` (comparison table, XOR via `viz/xor-limit.html`), `s5` (BUILD curves) keep their existing depth; they are simply no longer the *first* sight of a curve.

**Invariants preserved:** 7 sections, 8 `data-target`s, DEMOS/BUILD/QS present, playground ≥3 demos, quiz `q:4 o:16`, 7 `gotit` buttons, `.fin`, nav, quest-id `wf2-d02-activations`, localStorage keys. The shell-invariant gate proves byte-identity of CSS + JS engine + shell; only the authored content regions change.

---

## 7. Component 4 — Re-test / evaluation loop

Both gate mechanisms agreed: **heuristic gate BLOCKS, LLM judge evaluates qualitatively.**

Loop until convergence:

1. Compile: `python3 sessions/_compiler/compile_lesson.py sessions/m02-the-neuron/day-02-activations/source.md`
2. Reader-flow gate (incl. new depict-on-introduction check) — must PASS.
3. Shell-invariant gate (with `--donor`) — must PASS (byte-identity of shell).
4. Notebook-smoothness gate — must PASS or N/A (unchanged from current status).
5. **LLM judge** reads the compiled `lesson.html` and answers: "Does a beginner see each concept's picture at the moment it's introduced? Are ReLU/sigmoid/tanh curves visible before the playground numbers?" → structured `{pass: bool, reasons: [...]}`.
6. If any gate FAILs or the judge says no → edit `source.md`, recompile, repeat.

**Adversarial verification (final):** dispatch independent agents to confirm (a) curves are visible in `s1`/`s2` of the compiled HTML, (b) shell invariants intact, (c) the other 8 m02 days are byte-identical (unchanged), (d) the new gate actually FAILs on a deliberately-broken variant (proving it has teeth, not just green-by-default).

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| New gate retroactively fails 8 shipped m02 days / m03 | Check is opt-in via `visual_objects` front-matter; absent key = N/A. Verify byte-identity of untouched days. |
| Adding SVG to `s1`/`s2` breaks shell byte-identity | Shell-invariant gate masks content regions; only CSS + JS engine must match donor. Run gate with `--donor` to prove. |
| Early static curves duplicate the later BUILD curves (redundant) | Acceptable — repetition of a shape aids beginners. Optionally trim the most redundant late SVG, but not required. |
| Interactive iframe height sticks (known past bug) | `activation-derivatives.html` already ships the postMessage height sender (it's an existing embed just relocated), so no new iframe risk. |
| Gate is green-by-default and never actually catches the defect | Two mitigations: (a) check reads **raw** source (`*_raw` fields per §5.2a), not tag-stripped text where markers vanish; (b) adversarial step explicitly tests the gate against a broken variant to confirm it FAILs. |
| Check passes on `s4`-only depiction (curve still after the playground) | §5.2 requires depiction in `s1`/`s2`, not merely "no later than first mention." A token whose only figure is in `s4` FAILs. |
| Two skill copies drift | Every skill edit applied to both paths in the same step; a diff check confirms parity. |

---

## 9. Deliverables (this session)

1. 6 edited `SKILL.md` files (3 skills × 2 locations), diff-verified identical.
2. Extended `reader_flow_gate.py` with the depict-on-introduction check + tests against a broken variant.
3. Rebuilt `sessions/m02-the-neuron/day-02-activations/source.md` (+ `visual_objects` front-matter) and recompiled `lesson.html`.
4. Green run of all gates + LLM-judge approval + adversarial-verification notes.
5. This spec, committed.

Out of scope (future sessions): re-flow m02 Days 1/3–9 and m03; resume the paused m04 rollout using the improved skill/gate.
