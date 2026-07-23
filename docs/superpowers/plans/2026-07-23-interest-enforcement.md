# Interest Enforcement + Momentum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make interest a true, always-on P0 gate on every lesson (notebook or not), add a momentum/failure-cluster signal, make `%%% demo` labels honest, and align the skills — without capping length.

**Architecture:** A notebook-free `judge_interest_absolute` runs every build round and P0-gates the loop; a deterministic advisory warn flags un-interleaved failure clusters; a one-line label default fix; three skill-doc edits (applied to both skill trees for parity).

**Tech Stack:** Python (`sessions/_compiler/`), `pytest`, the LLM bridge (`localhost:11211`), `lesson_build.js`. System python3.

**Spec:** `docs/superpowers/specs/2026-07-23-interest-enforcement-design.md`

---

### Task 1: `judge_interest_absolute` — notebook-free interest floor

**Files:**
- Modify: `sessions/_compiler/gates/coverage_judge.py` (add `judge_interest_absolute`; call it unconditionally in `run_from_paths`, key `interest_absolute`)
- Test: `sessions/_compiler/tests/test_interest_absolute.py`

- [ ] **Step 1: Failing test.** With the bridge call mocked **the same way `test_coverage_judge.py` mocks it** (patch the OpenAI client / `.create` path that `judge_interest` actually uses — there is no `_bridge_json` helper), `judge_interest_absolute(text)` returns `{status:'OK', overall:'FLOOR_MET'|'BELOW_FLOOR', dimensions:[7 levers…], top_fixes, summary}`; and `run_from_paths(lesson, source)` populates `interest_absolute` **even when `notebook_yardstick` is null** (mock `meta` with null yardstick).
```python
# Mirror test_coverage_judge.py's mock (patch the client call judge_interest uses), NOT a _bridge_json symbol.
def test_absolute_runs_without_notebook(monkeypatch):
    import coverage_judge as cj
    # patch the same LLM-call seam test_coverage_judge.py patches:
    monkeypatch.setattr(cj, "<the client-call helper used by judge_interest>", lambda *a, **k: {
        "overall":"FLOOR_MET","dimensions":[{"name":"aspiration_hook","verdict":"GOOD"}],
        "top_fixes":[],"summary":"ok"})
    out = cj.judge_interest_absolute("some lesson text")
    assert out["overall"] in ("FLOOR_MET","BELOW_FLOOR") and out["status"]=="OK"
```
- [ ] **Step 2: Run, expect FAIL.** `cd sessions/_compiler && python3 -m pytest tests/test_interest_absolute.py -q`.
- [ ] **Step 3: Implement.** Clone `judge_interest` (lines ~357–388) but drop the notebook from the prompt; system prompt scores the 7 levers on their own merits against the fixed anchors in the spec's rubric table and returns `FLOOR_MET | BELOW_FLOOR` (bar: no lever MISSING and holistically want-to-continue; `≤1 WEAK` guide). Add the unconditional call in `run_from_paths` → `interest_absolute`. Add a CLI "Absolute Interest Floor" print section.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit.** `git add sessions/_compiler/gates/coverage_judge.py sessions/_compiler/tests/test_interest_absolute.py && git commit -m "feat(interest): notebook-free absolute interest floor judge"`

### Task 2: Loop gates on the floor every round

**Files:**
- Modify: `sessions/_compiler/workflows/lesson_build.js` (the interest lens, ~line 117)
- Test: `sessions/_compiler/tests/test_interest_lens.mjs` (or a small node harness over the lens function if extracted)

- [ ] **Step 1: Failing test** — the interest lens emits a P0 `kind:'interest'` when `interest_absolute.overall === 'BELOW_FLOOR'` regardless of notebook; and an absent/`N/A` interest result surfaces an explicit "unenforced" finding (not a silent pass). The notebook-relative `BELOW_NOTEBOOK`/`WORSE` remains an additional P0 when present.
- [ ] **Step 2–4:** Implement; keep the existing notebook-relative check as an added ceiling. Verify.
- [ ] **Step 5: Commit.** `git commit -am "feat(interest): P0-gate the build loop on the absolute floor every round"`

### Task 3: Failure-cluster momentum advisory (deterministic, never blocks)

**Files:**
- Modify: `sessions/_compiler/gates/concept_structure_gate.py` (add a day-level check after the per-concept loop; append to `msgs`, **never flip `ok[0]`/exit code** — mirror the "visualize the build-up" advisory at ~lines 88–95)
- Test: `sessions/_compiler/tests/test_failure_cluster.py`

- [ ] **Step 1: Failing test.** (Add a conftest helper `run_gate(concepts=[...])` that builds a synthetic concept-mode `source.md` from the titles, calls the gate's real entry `run(source_text)` — which returns `(ok, msgs)` — and returns `(msgs, ok)` normalized for these asserts.)
```python
def test_cluster_warns(run_gate):        # helpers build source with N concepts
    msgs, ok = run_gate(concepts=["intro","dead ReLU trap","vanishing gradient","exploding gradient","recap"])
    assert any("failure-mode cluster" in m for m in msgs) and ok is True   # advisory, never fails
def test_interleaved_no_warn(run_gate):
    msgs, ok = run_gate(concepts=["intro","dead ReLU trap","%%%demo","vanishing gradient","recap"])
    assert not any("failure-mode cluster" in m for m in msgs)
def test_two_failures_no_warn(run_gate):
    msgs, _ = run_gate(concepts=["intro","dead ReLU trap","vanishing gradient","recap"])
    assert not any("failure-mode cluster" in m for m in msgs)
def test_benign_substring_no_warn(run_gate):
    msgs, _ = run_gate(concepts=["The problem with one line","The limit of one neuron","recap"])
    # 'problem'/'limit' are benign titles, not a ≥3 failure run
    assert not any("failure-mode cluster" in m for m in msgs)
```
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Detect **≥3 consecutive** concept units whose title/tag matches the failure pattern (`dead`, `vanish`, `saturat`, `explod`, `collapse`, `overfit`, `underfit`, `-limit`, `trap`, `puzzle`, `cannot`, `wall`) with **no intervening** `%%% demo`/`%%% viz`. Word-boundary matching to avoid benign substrings. Message names the remedy (interleave). Append to `msgs`; do not touch `ok[0]`.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit.** `git commit -am "feat(interest): failure-cluster momentum advisory (never blocks)"`

### Task 4: Honest `%%% demo` label default

**Files:**
- Modify: `sessions/_compiler/v8lib.py` (`render_demo`: default label `'run it'` → a reveal verb)
- Test: `sessions/_compiler/tests/test_demo_label.py`

- [ ] **Step 1: Failing test** — omitted `label:` renders "reveal ▶"-style text; an author-supplied `label:` is preserved verbatim.
- [ ] **Step 2–4:** Change the default only; keep author labels. Verify.
- [ ] **Step 5: Commit.** `git commit -am "fix(interest): honest default demo button label (reveal, not run)"`

### Task 5: Align the skills (both trees, parity-checked)

**Files:** (edit in BOTH `.claude/skills/` and `frontier_lab_refactor_skills_v8/skills/`)
- `frontier-curriculum-architect/SKILL.md` — add the Coverage-Spec-Rule **interleave counterweight** (when 3+ failure units run consecutively, interleave a play/payoff beat; no compression/deferral, keep coverage + length)
- `frontier-lesson-builder/SKILL.md` — **reader-separation** (no scripted interview monologues / Reader-B notation in foundation bodies; staff depth as one plain empowerment line); **interest levers as named P0s** in the Non-negotiables; **real-play rule** (≥2 interactive `%%% viz` front-loaded; `%%% demo` is reveal-not-compute)
- `frontier-refactor-qa/SKILL.md` — document the **two interest enforcers** (absolute floor always-on + notebook ceiling); `notebook_yardstick:null` no longer means unenforced; add the failure-cluster advisory to the gate list; remove the stale "null saves cost" rationale

- [ ] **Step 1:** Make each edit in the `.claude/skills` tree.
- [ ] **Step 2:** Mirror verbatim into `frontier_lab_refactor_skills_v8/skills/`; `diff -r` the two trees for the three files → identical.
- [ ] **Step 3: Commit.** `git commit -am "docs(interest): align architect/builder/refactor-qa skills (both trees)"`

### Task 6: Calibrate the floor on the 9 m02 lessons

- [ ] **Step 1:** Run `python3 sessions/_compiler/gates/coverage_judge.py <lesson.html> --source <source.md>` on all 9 m02 days; record each `interest_absolute.overall`.
- [ ] **Step 2:** Confirm Days 1–6 (rated EXCELLENT/GOOD) read `FLOOR_MET`. If a genuinely-engaging lesson reads `BELOW_FLOOR` or a weak one reads `FLOOR_MET`, tune the rubric anchors and re-run. Day 9 (MIXED) reading `BELOW_FLOOR` is acceptable — it feeds the m02 rebuild.
- [ ] **Step 3:** Record the calibration outcome in the spec/report; commit any anchor tweak. `git commit -am "chore(interest): calibrate absolute floor anchors on m02"`

---
**Done when:** the absolute floor runs + P0-gates on every day (notebook or not), the failure-cluster advisory warns without blocking, demo labels are honest, both skill trees are aligned (identical), and the m02 calibration passes. Full suite: `python3 -m pytest sessions/_compiler/tests/ -q` green.
