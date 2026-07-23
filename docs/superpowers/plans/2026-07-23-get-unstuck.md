# Get-Unstuck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a stalled learner in-page, offline-safe recourse: a tiered hint ladder (`%%% hint`), mid-lesson micro-checks, and a "Stuck? copy this to Claude" handoff — built on the existing passive support.

**Architecture:** Add a `%%% hint` widget to the compiler grammar FIRST (unknown widget types raise `ValueError`), then render progressive-disclosure hints + inline micro-checks + a scoped copy-to-clipboard handoff in the donor so one edit lands across all 159 lessons. No network calls (a live tutor is deferred — `file://`/CORS blocks it).

**Tech Stack:** Python compiler (`sessions/_compiler/`, `_lesson_gen.py`), `v9-base.donor` JS, node/jsdom tests, LLM bridge to generate hint/micro-check content, system python3.

**Spec:** `docs/superpowers/specs/2026-07-23-get-unstuck-design.md`

---

### Task 1: Register the `%%% hint` widget in the compiler grammar (FIRST — before any author uses it)

**Files:**
- Modify: the `%%%` widget dispatcher `render_widget` in `sessions/_compiler/v8lib.py` (add a `hint` case — this is the ONLY required grammar edit; `_lesson_gen.py` is the legacy generator and is not on the V9 path), `sessions/_compiler/AUTHORING.md` §4 (document the block)
- Test: `sessions/_compiler/tests/test_hint_widget.py`

- [ ] **Step 1: Failing test.** A source with a `%%% hint … %%%` block compiles (exit 0) and renders a container with progressive tiers; an unknown widget type still raises `ValueError`.
```python
def test_hint_compiles(tmp_path, compile_source):  # compile_source helper in conftest
    src = ('%%% hint\n'
           't1: A gentle nudge.\n'
           't2: A worked micro-step.\n'
           't3: The idea in one sentence.\n%%%\n')
    html = compile_source(src)
    assert 'data-hint-tier="1"' in html and 'data-hint-tier="3"' in html
def test_unknown_widget_still_errors(compile_source):
    import pytest
    with pytest.raises(ValueError):
        compile_source('%%% nonsense\nx\n%%%\n')
```
- [ ] **Step 2: Run, expect FAIL** (`ValueError: unknown widget 'hint'`).
- [ ] **Step 3: Implement `render_hint(args, body)`** — parse `t1:/t2:/t3:` lines into `<div class="hint" data-hint-tier="N" hidden>` tiers + a "💡 Stuck? reveal a hint" button. Register in the dispatcher next to `demo`/`quiz`/`prompt`. Document in `AUTHORING.md` §4.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(unstuck): %%% hint widget grammar + AUTHORING.md doc"`

### Task 2: Progressive hint-reveal JS in the donor

**Files:**
- Modify: `sessions/_compiler/shells/v9-base.donor` (hint CSS reuse + reveal JS)
- Test: `sessions/_compiler/tests/test_hint_reveal.mjs` (jsdom)

- [ ] **Step 1: Failing jsdom test** — clicking the hint button reveals tier 1 only; clicking again reveals tier 2; then tier 3; never all at once; no `fetch` is called.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the reveal handler (reuse the quiz-reveal pattern): each click unhides the next `[data-hint-tier]`. Pure DOM, no network.
- [ ] **Step 4: Run, expect PASS**; recompile a sample lesson, confirm idempotent.
- [ ] **Step 5: Commit.** `git commit -am "feat(unstuck): progressive hint-ladder reveal JS"`

### Task 3: Mid-lesson micro-checks (diagnostic feedback where the kid stalls)

**Files:**
- Modify: `sessions/_compiler/v8lib.py` (`render_widget`) + `v9-base.donor` — allow a `%%% quiz` (or a lighter `%%% check`) to appear INSIDE a concept unit, not only the end-of-lesson quiz; its `fb` names the misconception; answering also feeds `sr` (retention plan Task 2). (NOT `_lesson_gen.py` — legacy.)
- Test: `sessions/_compiler/tests/test_micro_check.py`

- [ ] **Step 1: Failing test** — a concept unit containing an inline check compiles, renders the diagnostic `fb`, and calls the `sr` `review()` on answer.
- [ ] **Step 2–4:** Permit inline checks after the 2–3 hardest concepts; reuse the diagnostic-`fb` quiz engine. Verify.
- [ ] **Step 5: Commit.** `git commit -am "feat(unstuck): inline mid-lesson micro-checks with diagnostic feedback"`

### Task 4: "Stuck? copy this to Claude" handoff

**Files:**
- Modify: `v9-base.donor` (reuse the produce copy-to-clipboard widget: `navigator.clipboard.writeText` + `execCommand` fallback), `_lesson_gen.py` (emit a per-hard-section button with a scoped, pre-filled prompt)
- Test: `sessions/_compiler/tests/test_paste_handoff.mjs` (jsdom)

- [ ] **Step 1: Failing test** — the button copies a scoped prompt containing the module, day, section title, and the confusing term (e.g. `"I'm 12 and on M2 Day 5, section 'Trace the blame'. Explain <term> a simpler way with a small example."`); asserts no `fetch` anywhere in the widget JS.
- [ ] **Step 2–4:** Implement the button + `data-copy` scoped-prompt payload (template-filled from the section's metadata). This also replaces the produce "Option B" stub from the real-doing plan.
- [ ] **Step 5: Commit.** `git commit -am "feat(unstuck): paste-to-Claude scoped handoff (offline)"`

### Task 5: Generate hint ladders + micro-checks for the two pilot lessons (bridge, judged)

**Files:**
- Modify: `sessions/m02-the-neuron/day-02-activations/source.md`, `sessions/m02-the-neuron/day-05-gradients-backprop/source.md` (author `%%% hint` + inline `%%% check` on the hardest sections), recompile

- [ ] **Step 1:** Bridge-generate hint tiers + micro-check items for the hardest sections (day-05 chain rule / vanishing gradient; day-02 dead ReLU), seeded by the `coach/quizzes` `hint`/`explanation` fields; judge for correctness + beginner phrasing.
- [ ] **Step 2:** Recompile both; confirm exit 0 + gates pass + hint ladder/micro-check render.
- [ ] **Step 3: Commit.** `git commit -am "feat(unstuck): pilot hint ladders + micro-checks on m02 day-02/day-05"`

### Task 6: Roll donor-wide + offline-safety verify

- [ ] **Step 1:** Regenerate all lessons through the updated donor (widgets are opt-in per source, but the JS ships everywhere).
- [ ] **Step 2:** Grep-assert **no `fetch()`/`XMLHttpRequest`/`11211`** in the donor + widget JS regions (lesson body prose may still mention the bridge — scope the grep to the JS/donor). `python3 -m pytest sessions/_compiler/tests/ -q` + node tests green.
- [ ] **Step 3: Commit.** `git commit -am "feat(unstuck): roll hint/micro-check/handoff widgets donor-wide (offline-safe)"`

---
**Done when:** `%%% hint` compiles and reveals progressively; inline micro-checks give diagnostic feedback mid-lesson; the paste-to-Claude button copies a correct scoped prompt with zero network calls; pilots render on m02 day-02/day-05; test suites green.
**Deferred (out of scope):** a live in-page `localhost:11211` tutor (needs a served build).
