# V9 Concept-Driven Lesson Shell — Session Report

**Date:** 2026-07-11
**Branch:** `build/capability-spiral`
**Spec:** `docs/superpowers/specs/2026-07-11-concept-driven-lesson-flow-design.md`
**Plan:** `docs/superpowers/plans/2026-07-11-concept-driven-lesson-shell-v9.md`
**Recommendation:** **Pass** (0 P0 open; the one adversarially-found gate weakness was fixed in-session)

---

## Scope

Motivated by a user complaint on **Module 2 Day 2 (Activation Functions)**: a learner met the *words* "ReLU / sigmoid / tanh", the valve/dimmer analogies, and the numeric playground output **before ever seeing the curves** — the visuals were deferred to a single late "Build it up" section. Root cause was structural: the fixed 7-section lesson template (`What-is-it → Intuition → Playground → Mechanism → Build-it-up → Quiz → Produce`) forced concepts to the front as text and quarantined all visuals in one late section.

This session built a **concept-driven (V9) lesson shell**: a lesson body is now **N per-concept units**, each `plain intro → its own inline visual → build-up`. Delivered end-to-end and proven on Day 2.

**Delivered this session:** V9 compiler mode + reusable donor + gates + gate dispatch + Day 2 rebuild + skill edits + full verification.
**Deferred (future sessions):** backport m02 Days 1/3–9 and m03 to V9; resume the paused **m04** rollout on the V9 shell (m04 was blocked on exactly this "visualization issue").

---

## What changed (13 commits, `c3e3c7a`…`3d36791`)

### Compiler (`sessions/_compiler/`)
- **`v8lib.py`** — new pure renderers: `%%% svg` (inline static visual), `%%% demo` (inline run-demo), `%%% quiz` (authored quiz DOM), `@@@ concept` block, `concept_nav_items` (auto-numbered sidebar), and a `mode=='concept'` `_compile_concept` path (+ `render_quiz_section`, `render_produce_section`, `render_sidebar_nav_items`). V8 path untouched.
- **`shells/v9-base.donor`** (new) — reusable V9 shell: head/CSS/glossary+resize scripts kept verbatim from the m02 donor; `<!--V9_NAV-->`/`<!--V9_CONTENT-->` placeholders; `data-quest-id="__QUEST_ID__"` template token; **generalized JS engine** — the shipped progress/nav/scrollspy/theme code was already generic over `.module-section`/`data-sec`/`.nav-link`; only the 3 id-coupled engines changed (deleted the `s3` playground / `s5` build-reveal / `s6` quiz engines; added generic `.demo` and `.quiz` engines that work over any number of sections).
- **`compile_lesson.py`** — gate dispatch on `meta['mode']`: concept → reader_flow (concept branch) + `concept_shell_gate` + notebook_smoothness; else → existing V8 path unchanged.

### Gates
- **`gates/concept_shell_gate.py`** (new) — V9 structural invariants on compiled HTML: ≥3 concept sections each with a **real** visual (`<svg>…</svg>` or a `build-embed` iframe) + exactly one gotit; one quiz (4 Qs) + one produce (references `experiment.py`); sidebar nav parity; localStorage keys; `.fin`; no leaked markers.
- **`gates/reader_flow_gate.py`** — added a concept-mode branch (`_run_concept`) that skips the V8 `s1/s2/s4/s7` strict checks and instead enforces: hero human-first / no frontier-pressure; every `@@@ concept` ships a **real** visual; spine word across ≥3 blocks; produce discovery-framed. Non-concept modes unchanged.

### Lesson
- **`sessions/m02-the-neuron/day-02-activations/`** — `source.md` rewritten to V9 concept mode (7 concept units: c1 collapse, c2 the-bend, c3 ReLU+demo, c4 sigmoid+demo, c5 tanh+comparison-table, c6 dead-ReLU/saturation+interactive derivatives, c7 XOR+interactive+frontier payoff), `lesson.html` regenerated. **This is the one lesson intentionally NOT byte-identical** — replacing it is the deliverable. Quest-id `wf2-d02-activations`, all vetted content and numbers preserved (Math Ladder, staff-lens trio, demo outputs, dead-ReLU quiz, bilingual note).

### Skills (both locations, byte-identical)
- `frontier-lesson-builder` (concept-unit blueprint + 2 new P0s), `frontier-visual-evidence-builder` (depict-in-unit rule), `frontier-refactor-qa` (V9 gate subsection). Applied to `.claude/skills/` **and** `frontier_lab_refactor_skills_v8/skills/`; parity verified.

---

## Gate / test evidence

- **`pytest sessions/_compiler/tests/`** → **25 passed** (widgets 7, compile 5, concept-gates 12, v8-regression 1).
- **Day 2 compile** → exit 0; Reader Flow Gate PASS, Concept Shell Gate PASS, **Notebook Smoothness PASS** (not N/A — hero is human-first, no formula wall).
- **JS backstop** → all `<script>` blocks pass `node --check`.
- **Idempotency** → recompiling Day 2 is byte-identical.

## Verification

- **LLM-judge (local bridge, Opus 4.8):** PASS — "each concept including ReLU, sigmoid, and tanh shows its own visual inline where it is introduced, with no deferrals."
- **Adversarial verification (independent agents):**
  1. **No V8 regression** — all 13 shipped V8 days (m02 ×8 non-day-02 + m03 ×5) recompile byte-identical. Claim TRUE.
  2. **Gate teeth** — 6/7 malformed inputs correctly rejected; found **one real weakness**: the visual check used naive substring matching, so a figure-less concept whose *prose* contained the literal string `build-embed` (or `<svg` in inline code) passed. **Fixed in `3d36791`** — both gates now require a real closed `<svg>…</svg>` element or a `build-embed` iframe; the exact bypass now FAILs (`concept c1 has NO visual`), and the real Day-2 (genuine widgets) still passes. Added 3 tests locking it.

## Process
Spec + plan each went through a review loop (2 iterations, approved). Implementation ran subagent-driven: one implementer per cluster + two-stage review (spec compliance → code quality), fix loop until clean. Every fix traced to a review or adversarial finding.

## Reusable facts
- **V9 = additive mode**, not a replacement. `mode: concept` + `donor: v9-base.donor`. Shipped V8 lessons keep their donors + `mode: exemplar/reference` and are byte-identical — a regression test enforces this.
- Authoring: `@@@ hero/concept/quiz/produce/fin`; widgets `%%% svg`, `%%% demo` (code/out/take), `%%% viz src=…`, `%%% quiz` (`q: … | a:IDX | opt×4 | fb: …`). Front-matter needs `module_label`/`title`/`subtitle` for the hero.
- The visual gates require a **real figure element**, not a marker string — do not "trick" them; put an actual `%%% svg`/`%%% viz` in every concept.
- Next: this unblocks the paused m04 rollout (rebuild its 6 days on V9), then backport m02 D1/D3–9 + m03.
