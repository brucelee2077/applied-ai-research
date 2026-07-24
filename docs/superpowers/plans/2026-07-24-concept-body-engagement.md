# Concept-Body Engagement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use
> checkbox (`- [ ]`) syntax. TDD, DRY, YAGNI, frequent commits. Build on
> `build/capability-spiral`, one file at a time (user directive).

**Goal:** Make each concept's BODY (the build-up) as engaging as its intro, enforced
skill-side + judge-side, then rebuild m02 (9) + m03 (5) under the new engine.

**Architecture:** Body-side twin of the interest-floor fix. Builder-skill "Build-Up
Register" (source) + standalone `judge_body_engagement` LLM floor (held-out test) + a
body widget toolkit (`%%% insight`, predict-then-reveal `%%% demo`, `%%% steps` + a wired
scroll-reveal). Then a full author-loop rebuild of the 14 target days.

**Tech stack:** Python compiler (`v8lib.py`, gates), JS donor + `lesson_build.js`
workflow, LLM judges via the local keyless bridge (`localhost:11211`, model
`aws:anthropic.claude-opus-4-8`, no `temperature`). Tests: pytest + node/jsdom.

**Baseline:** compiler suite 169 pass / 1 pre-existing m02-manifest fail (out of scope).
Run tests with system `python3` (the `.venv` is broken):
`cd sessions/_compiler && python3 -m pytest tests/ -q`.

---

### Task 1: `%%% insight` widget (inline "why this matters" re-hook)

**Files:** Modify `sessions/_compiler/v8lib.py`; Test `sessions/_compiler/tests/test_insight_widget.py`

- [ ] **Step 1 — failing test.** Assert `render_widget('insight', {}, ['This is why it **matters**: the [[net||model]] runs but does nothing.'])` returns HTML containing `class="takeaway"`, the 💡 glyph, `<strong>matters</strong>` (proves `inline()` ran), and the `.term` span (proves `[[...]]` glossary ran).
- [ ] **Step 2 — run, expect fail** (`render_widget` raises `ValueError: unknown ... insight`).
- [ ] **Step 3 — implement.** Add `render_insight(lines)`: join body, run through `inline()`, emit `<div class="takeaway">💡 <span>…</span></div>`. Register `if typ == 'insight': return render_insight(lines)` in `render_widget`.
- [ ] **Step 4 — run, expect pass.**
- [ ] **Step 5 — commit** `feat(engage): %%% insight re-hook callout (reuses .takeaway)`.

### Task 2: predict-then-reveal for `%%% demo`

**Files:** Modify `render_demo` in `sessions/_compiler/v8lib.py`; Test `sessions/_compiler/tests/test_demo_predict.py`

- [ ] **Step 1 — failing test.** (a) `render_demo({'id':'d'}, ['predict: what does relu([-3,2]) give?', 'code: relu([-3,2])', 'out: [0,2]', 'take: zeros negatives'])` contains `demo-predict` and the question text and the 🤔 glyph. (b) **Regression:** a demo WITHOUT `predict:` returns a string byte-identical to the current output (hard-code the expected current HTML for a fixed input, or snapshot it before the change).
- [ ] **Step 2 — run, expect (a) fail, (b) pass.**
- [ ] **Step 3 — implement.** In `render_demo`, after `_kv`, read `pred = d.get('predict')`. If truthy, prepend `<div class="demo-predict" style="padding:.6rem .9rem;font-size:.85rem;color:var(--ink2);background:var(--panel);border-bottom:1px solid var(--line)">🤔 <b>Predict first:</b> %s</div>` (run through `inline()`) INSIDE the `.demo` div, before `.demo-code`. When `predict` absent, emit exactly as today (guard with `if pred`).
- [ ] **Step 4 — run, expect both pass.** Also run the full suite to confirm no shipped-demo diff elsewhere.
- [ ] **Step 5 — commit** `feat(engage): %%% demo predict: (predict-then-reveal; byte-identical when absent)`.

### Task 3: `%%% steps` narrated stepped worked-example

**Files:** Modify `sessions/_compiler/v8lib.py`; Test `sessions/_compiler/tests/test_steps_widget.py`

- [ ] **Step 1 — failing test.** `render_widget('steps', {}, ['step: (x·W1)·W2', 'why: two matmuls back to back', 'step: = x·(W1·W2)', 'why: they collapse into ONE'])` returns a `class="build"` wrapper with exactly two `class="build-step"`, each containing a `class="build-num"` (1, 2) and a `class="build-note"`; the `why` text present; `inline()` applied.
- [ ] **Step 2 — run, expect fail.**
- [ ] **Step 3 — implement.** Add `render_steps(lines)`: walk body pairing each `step:` with the following `why:` (tolerate a `step:` with no `why:`). Emit `<div class="build">` + per pair `<div class="build-step"><div class="build-note"><span class="build-num">N</span><b>WORK</b> — GLOSS</div></div>` + `</div>`. Run `step`/`why` through `inline()`. Register `steps` in `render_widget`.
- [ ] **Step 4 — run, expect pass.**
- [ ] **Step 5 — commit** `feat(engage): %%% steps narrated worked-example (reuses .build CSS)`.

### Task 4: `concept_structure_gate` — build-up widget satisfies the floor

**Files:** Modify `sessions/_compiler/gates/concept_structure_gate.py`; Test `sessions/_compiler/tests/test_concept_structure_buildup_widget.py`

- [ ] **Step 1 — failing test.** Build a minimal `source.md` string with 3 concepts; one concept's build-up region (after its `%%% svg`) is ONLY a `%%% steps` block (< 40 chars of surrounding prose). Assert `run(src)` returns `ok == True` (today it returns False on "has build-up after its visual").
- [ ] **Step 2 — run, expect fail** (gate strips the widget → build-up prose < 40).
- [ ] **Step 3 — implement.** In `run()`, after computing `buildup` (stripped prose), also test for a build-up widget in the `after` region: `has_bw = re.search(r'(?m)^%%%\s+(steps|demo|mathladder|svg|viz)\b', after)`. Change the build-up check to `chk(len(buildup) >= _MIN_PROSE or bool(has_bw), 'concept %s has build-up after its visual' % cid)`. (Intro-prose check unchanged.)
- [ ] **Step 4 — run, expect pass;** re-run the existing `test_concept_structure*` tests to confirm no regression.
- [ ] **Step 5 — commit** `fix(gate): build-up floor satisfied by a build-up widget (steps/demo/mathladder)`.

### Task 5: donor `__revealBuild` (new multi-container scroll-reveal) + recompile

**Files:** Modify `sessions/_compiler/shells/v9-base.donor`; Test `sessions/_compiler/tests/test_reveal_build.mjs` (jsdom, if jsdom present per repo convention); recompile all concept lessons.

- [ ] **Step 1 — failing jsdom test.** Load a fixture HTML with two `.build` containers (each with `.build-step`s) + a stub IntersectionObserver, define the donor's reveal IIFE, call `window.__revealBuild()`; assert each `.build` gains `armed` and, after the observer fires "intersecting", each `.build-step` gains `revealed`. Also assert: with `matchMedia('(prefers-reduced-motion)')=true` OR no `IntersectionObserver`, all steps end `revealed` and no container is `armed`.
- [ ] **Step 2 — run, expect fail** (`__revealBuild` undefined in v9-base.donor).
- [ ] **Step 3 — implement.** Add an IIFE in the donor `<script>` (near the demo/quiz handlers) defining `window.__revealBuild = function(){ ... }`: `querySelectorAll('.build')`; if `RM || !('IntersectionObserver' in window)` → add `revealed` to all `.build-step`, return; else for each container add `armed`, create one IntersectionObserver that adds `revealed` to entries that intersect (threshold ~0.25), observe each `.build-step`. Idempotent (skip already-armed). Call `window.__revealBuild()` once at the end of the main IIFE (after `refresh()`), so build steps reveal on load too. Decoupled from `.gotit`/`#s5`.
- [ ] **Step 4 — run jsdom test, expect pass.**
- [ ] **Step 5 — recompile every concept lesson** using `v9-base.donor` (content unchanged; donor JS grew). Script: find all `sessions/**/source.md` with `mode: concept` + `donor: v9-base.donor`, run `compile_lesson.py` on each. First recompile ONE non-target concept module (e.g. an m07 day) and confirm exit 0 + gates pass. Then recompile the rest.
- [ ] **Step 6 — verify** `python3 -m pytest tests/test_v8_regression.py -q` re-passes (m02 day-03 now recompiled) and the full suite is green.
- [ ] **Step 7 — commit** `feat(engage): wire multi-container __revealBuild scroll-reveal + recompile concept lessons`.

### Task 6: `judge_body_engagement` (standalone LLM floor)

**Files:** Modify `sessions/_compiler/gates/coverage_judge.py`; Test `sessions/_compiler/tests/test_body_engagement.py`

- [ ] **Step 1 — failing test.** Monkeypatch `coverage_judge._chat` to return a fixed JSON (`{"concepts":[{"concept":"c1","body_engagement":"MISSING","note":"cold dump","fix":"add voice"}],"overall":"WEAK","summary":"x"}`). Assert `judge_body_engagement('text', ['c1'])` returns `status==OK`, `concepts[0]['body_engagement']=='MISSING'`. Assert `judge_body_engagement('t', [])` returns `status=='N/A'`. Assert the prompt (`_BODY_SYS` + `_body_prompt`) contains "body_engagement", "Recap", and "cold" (NA triggers + the axis). Assert a `_chat` that raises → `status=='BRIDGE_UNAVAILABLE'` (never raises).
- [ ] **Step 2 — run, expect fail.**
- [ ] **Step 3 — implement.** Add `_BODY_SYS`, `_body_prompt(lesson_text, concept_titles)`, and `judge_body_engagement(lesson_text, concept_titles, model, timeout)` using `_chat(_BODY_SYS, _body_prompt(...))` (mirror `judge_interest_absolute`: `_extract_json` → `_salvage_truncated_json` → setdefault → status). Prompt: for each concept, grade `body_engagement` GOOD/WEAK/MISSING/NA on the register beats (analogy alive, re-hook "why this bites", narrated with causal connectors vs flat Step 1/2/3 or symbol-pushing, predict/discovery, normalize-struggle); MISSING only for a genuinely COLD body; NA for the recap unit or a one-line definitional body; default to the lower grade. Add `'body_engagement': judge_body_engagement(lesson_text, concept_titles)` to `run_from_paths`'s return. Add a CLI "Concept Body Engagement" section printing per-concept `[be:GOOD] c1 — note`.
- [ ] **Step 4 — run, expect pass.**
- [ ] **Step 5 — commit** `feat(judge): standalone judge_body_engagement floor (per-concept build-up voice)`.

### Task 7: `lesson_build.js` — `body` lens + enum + author paragraph

**Files:** Modify `sessions/_compiler/workflows/lesson_build.js`; Test `sessions/_compiler/tests/test_lesson_build_body_lens.mjs` (or a grep-style node `--check` assertion).

- [ ] **Step 1 — failing test.** Read the file; assert it contains a `body` lens whose prompt parses the "Concept Body Engagement" section (`body_engagement` MISSING→P0 kind `body_engagement`, WEAK→P1), that `JUDGE_SCHEMA.kind` description includes `body_engagement`, and that the author prompt contains a "keep the body alive" instruction naming `%%% insight` / `%%% steps` / predict-then-reveal.
- [ ] **Step 2 — run, expect fail.**
- [ ] **Step 3 — implement.** (a) Add to `LENSES` a `{ key: 'body', prompt: 'Run coverage_judge.py …; parse "Concept Body Engagement": each concept body_engagement MISSING → P0 finding kind="body_engagement" (a cold mechanism/symbol dump — the user\'s #1 ask 2026-07-24 is a body as engaging as the intro); WEAK → P1; GOOD/NA → no finding. Bridge unavailable → PASS, note it.' }`. (b) Extend `JUDGE_SCHEMA.properties.findings.items.properties.kind.description` to include `body_engagement`. (c) Append a paragraph to the author prompt: "KEEP THE BODY ALIVE (Build-Up Register): the build-up must be as engaging as the intro — bridge from the opening analogy, re-hook 'why this bites' inside the body, narrate the mechanism with causal connectors and semantic step names (NOT flat Step 1/2/3 or bare symbol-pushing), use predict-then-reveal, keep the spine analogy alive to the end. Tools: `%%% insight` (why-this-matters re-hook), `%%% demo` with a `predict:` line (predict-then-reveal), `%%% steps` (narrated worked example). A cold mechanism dump is a P0 body_engagement finding."
- [ ] **Step 4 — run, expect pass;** `node --check sessions/_compiler/workflows/lesson_build.js`.
- [ ] **Step 5 — commit** `feat(loop): body_engagement lens + enum + 'keep the body alive' author rule`.

### Task 8: AUTHORING.md — document the new body grammar

**Files:** Modify `sessions/_compiler/AUTHORING.md`

- [ ] **Step 1 — implement.** In §4, add `insight`, `steps`, and the `demo predict:` field with a minimal example each. In §6, add a "Keep the build-up alive (Build-Up Register)" note pointing at the widgets + the `judge_body_engagement` floor.
- [ ] **Step 2 — verify** no compiling example broke: re-run the suite (AUTHORING has a tested minimal example) — green.
- [ ] **Step 3 — commit** `docs(authoring): %%% insight / %%% steps / demo predict: + Build-Up Register note`.

### Task 9: Builder skill — the Build-Up Register (both trees)

**Files:** Modify `.claude/skills/frontier-lesson-builder/SKILL.md` AND the v8 mirror `frontier_lab_refactor_skills_v8/skills/frontier-lesson-builder/SKILL.md` (find exact path first).

- [ ] **Step 1 — implement.** After the "Beginner Intuition Register" section add a **"Build-Up Register (per concept — keep the body alive)"** section with the 7 named body beats (bridge-from-analogy, re-hook, narrate-the-architecture, predict-then-reveal, micro-recap+breath, normalize-struggle-during, spine-alive-to-the-end), the tools (`%%% insight` / `%%% steps` / `demo predict:`), and a **P0 (body-cold)** rule pointing at the `judge_body_engagement` floor. Add one line to the Learning Barrier Gate: "P0 if a concept's build-up drops into a flat mechanism/symbol dump with no voice, re-hook, or discovery."
- [ ] **Step 2 — apply the identical edit to the v8 mirror tree** (skills-parity rule).
- [ ] **Step 3 — verify** both files contain "Build-Up Register" (grep).
- [ ] **Step 4 — commit** `feat(skill): Build-Up Register — keep the concept body as engaging as the intro (both trees)`.

### Task 10: Rerun — full author-loop rebuild of m02 (9) + m03 (5)

**Files:** m02 (9 days) + m03 (5 days) `source.md` + `lesson.html`.

- [ ] **Step 1 — gather frozen front-matter** for each of the 14 days (read each existing `source.md`'s `---` block verbatim; it holds quest_id, mode, donor, nav_*, module_label, page_title, brand_sub, spine, notebook_yardstick).
- [ ] **Step 2 — run a bounded per-day Workflow** (one `lesson_build.js` invocation per day via `workflow('lesson-build', {module, day, frozen, maxRounds:4})`, or an inline pipeline of 14 author agents), each: author under the new register + body toolkit → compile → judge panel (incl. the `body` lens) → fix until P0-clear → the day's `lesson.html` recompiles clean. Preserve frozen front-matter. **Length is a feature — never cut coverage to fix a body; add voice/widgets.**
- [ ] **Step 3 — independent verification pass** per day: `body_engagement` P0-clear; interest floor `FLOOR_MET`; `concept_structure_gate` + `compile_lesson.py` exit 0; idempotent recompile.
- [ ] **Step 4 — commit** per module (or per day) `feat(m02|m03): rebuild bodies under Build-Up Register + body_engagement floor`.

### Task 11: Verify + report + memory

- [ ] **Step 1 — full suite green** (baseline + new tests; 1 pre-existing m02-manifest fail only).
- [ ] **Step 2 — write** `sessions/concept_body_engagement_rerun.md` (per-day body_engagement + interest verdicts, before/after excerpts).
- [ ] **Step 3 — update memory** `loop-fix-rollout-state.md` + `MEMORY.md` index with the body-engagement dimension.
- [ ] **Step 4 — finishing-a-development-branch** (verify tests, present options).
