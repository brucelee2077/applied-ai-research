# Real Doing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 115 empty `experiment.py` stubs with real, run-verified, self-checking artifacts; kill the dead `/frontier-experiment-lab` path; make the produce answer a real predict-then-reveal; enable opt-in cross-day build-up.

**Architecture:** Reuse `evidence_build.js`'s producer (writes+runs a real `python3` script, judge-gated on exit-0 + numbers) but retarget it to the **lesson dir**, emitting a *scaffold* (`imports + TODO bodies + a self-check `assert` that prints ✅/❌`). A deterministic gate enforces the scaffold+self-check contract. Answer-gating and cross-day `build_on` are donor/generator changes.

**Tech Stack:** Python 3 (system python3 — repo `.venv` is broken), `pytest` under `sessions/_compiler/tests/`, the local LLM bridge (`localhost:11211`) for generation, node/jsdom for the donor JS.

**Spec:** `docs/superpowers/specs/2026-07-23-real-doing-design.md`

---

### Task 1: Define the experiment.py scaffold + self-check contract (deterministic gate)

**Files:**
- Create: `sessions/_compiler/gates/experiment_contract.py`
- Test: `sessions/_compiler/tests/test_experiment_contract.py`

- [ ] **Step 1: Write the failing test.** Assert `check_experiment(path)` returns `ok=False` for a 5-line placeholder stub, and `ok=True` for a script that has imports, at least one `# TODO`, and an `if __name__ == "__main__":` block containing an `assert` and a `print("✅"...)`/`print("❌"...)`.
```python
# test_experiment_contract.py
from gates.experiment_contract import check_experiment  # sys.path add in conftest
STUB = "# day-x - experiment\n# Placeholder. Fill this from the lesson's PRODUCE step\n"
GOOD = ('import numpy as np\n'
        'def forward(x):\n    # TODO: return the weighted sum\n    return x.sum()\n'
        'if __name__ == "__main__":\n'
        '    got = forward(np.array([1,2]))\n'
        '    assert abs(got-3)<1e-9, got\n'
        '    print("✅ you got it" if abs(got-3)<1e-9 else "❌ not yet")\n')
def test_stub_fails(tmp_path):
    p = tmp_path/"e.py"; p.write_text(STUB)
    assert check_experiment(str(p)).ok is False
def test_good_passes(tmp_path):
    p = tmp_path/"e.py"; p.write_text(GOOD)
    r = check_experiment(str(p)); assert r.ok is True
```
- [ ] **Step 2: Run it, expect FAIL.** `cd sessions/_compiler && python3 -m pytest tests/test_experiment_contract.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement `check_experiment`.** Returns a small dataclass `Result(ok: bool, reasons: list[str])`. Checks: (a) not the placeholder text; (b) ≥1 import; (c) contains `if __name__ == "__main__":`; (d) that block contains `assert`; (e) contains a `✅`/`❌` print. Pure string/AST checks, no execution.
- [ ] **Step 4: Run it, expect PASS.**
- [ ] **Step 5: Commit.** `git add sessions/_compiler/gates/experiment_contract.py sessions/_compiler/tests/test_experiment_contract.py && git commit -m "feat(doing): experiment.py scaffold+self-check contract gate"`

### Task 2: Producer that writes+runs a real scaffold into the lesson dir

**Files:**
- Create: `sessions/_compiler/workflows/doing_build.js` (adapted from `evidence_build.js`)
- Reference: `sessions/_compiler/workflows/evidence_build.js:41-54` (producer: write → `python3 … > out.txt` → fix until exit 0)

- [ ] **Step 1:** Copy `evidence_build.js` → `doing_build.js`. Change the target from `portfolio/${module}/${day}/experiment.py` to `sessions/${module}/${day}/experiment.py`. Args `{module, day, maxRounds}`.
- [ ] **Step 2:** Rewrite the producer prompt: *"Write a SCAFFOLD `experiment.py` from this lesson's `@@@ produce` block: real imports; the core function(s) with `# TODO:` bodies AND a reference solution in a hidden `_solution()` so the self-check has a truth to compare; an `if __name__=='__main__':` self-check that runs the learner-fillable path, `assert`s the expected values from the lesson's 'what you should see', and prints `✅ you got it` / `❌ not yet — expected X`. RUN it (`python3 …`), fix until exit 0 AND it prints ✅."*
- [ ] **Step 3:** Judge/gate: after produce, run `python3 sessions/_compiler/gates/experiment_contract.py <path>` (add a `__main__` CLI to Task 1's module) and require exit 0 + `✅` in stdout. Loop `maxRounds`.
- [ ] **Step 4: Verify on ONE day.** Run the workflow for `m02-the-neuron day-01-single-neuron`; confirm `sessions/m02-the-neuron/day-01-single-neuron/experiment.py` now exits 0, prints ✅, and passes `check_experiment`.
- [ ] **Step 5: Commit.** `git add sessions/_compiler/workflows/doing_build.js && git commit -m "feat(doing): producer writes+runs real scaffold experiment.py into lesson dir"`

### Task 3: Kill the dead `/frontier-experiment-lab` path

> **Note (V9 build path):** `sessions/_lesson_gen.py` is the LEGACY V7 generator and is NOT on the V9 path — do not edit it. V9 lessons are compiled from `source.md` + `.donor` shells via `compile_lesson.py`/`v8lib.py`, so the fix is in the **sources + donors**, then recompile to regenerate `lesson.html`.

**Files:**
- Modify: 43 `sessions/**/source.md` and the ~20 `sessions/_compiler/shells/*.donor` shells that carry the dead path (grep to enumerate); then **recompile** to regenerate the 133 `sessions/**/lesson.html`
- Test: `sessions/_compiler/tests/test_no_dead_lab.py`

- [ ] **Step 1: Failing test.** Grep-assert 0 `/frontier-experiment-lab` in learner/source/donor files (exclude reports):
```python
import subprocess
def test_no_dead_lab():
    out = subprocess.run(["grep","-rl","frontier-experiment-lab","sessions",
        "--include=lesson.html","--include=experiment.py","--include=source.md","--include=*.donor"],
        capture_output=True,text=True).stdout.strip()
    assert out == "", f"dead path remains in:\n{out}"
```
- [ ] **Step 2: Run, expect FAIL** (source.md + donor shells + regenerated lesson.html).
- [ ] **Step 3: Fix the sources + donors.** Replace the produce "Option B: paste the frontier-experiment-lab prompt" block in the 43 `source.md` and the ~20 `shells/*.donor`: Option A = *"open the scaffolded `experiment.py`, fill the TODOs, run it"*; Option B = the paste-to-Claude handoff (from the get-unstuck plan; until that lands, a plain "ask Claude Code to help you fill it" line with no skill path). Scripted find/replace across both file sets.
- [ ] **Step 4: Recompile the affected lessons** (`compile_lesson.py` per source) so `lesson.html` regenerates clean; re-run the test → PASS.
- [ ] **Step 5: Commit.** `git commit -am "fix(doing): remove dead /frontier-experiment-lab path from sources+donors, recompile"`

### Task 4: Gate the produce answer behind predict-then-reveal

**Files:**
- Modify: `sessions/_compiler/shells/v9-base.donor` (produce JS + markup), `sessions/_compiler/v8lib.py` (produce-block render, if markup is emitted there) — NOT `_lesson_gen.py` (legacy)
- Test: `sessions/_compiler/tests/test_produce_gate.js` (node/jsdom)

- [ ] **Step 1: Failing jsdom test.** Load a compiled produce section; assert the "what you should see" `<ul>` starts with `hidden`/`aria-hidden` and is revealed only after a "I've predicted — reveal" button click.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Wrap the produce expected-output list in a `.reveal-gate` (reuse the quiz-reveal JS pattern): a prediction prompt + "reveal" button that unhides the list. Keep it in the donor so it lands lesson-wide.
- [ ] **Step 4: Run, expect PASS**; recompile a sample lesson, confirm idempotent.
- [ ] **Step 5: Commit.** `git commit -am "feat(doing): predict-then-reveal gate on produce expected-output"`

### Task 5: Opt-in cross-day build-up (`build_on`)

**Files:**
- Modify: `sessions/<module>/_refactor/manifest.yaml` (add `build_on: <prior-day>` on select day entries), `doing_build.js` (pass prior-day artifact into the producer prompt)
- Test: `sessions/_compiler/tests/test_build_on.py`

- [ ] **Step 1: Failing test.** For a day with `build_on`, assert its generated `experiment.py` imports/references the prior day's module and still exits 0.
- [ ] **Step 2–4:** Add `build_on` to 1–2 natural pairs (m02 day-03 layer ← day-01 neuron); teach the producer to **copy-forward** the reused function into today's scaffold (or `sys.path.insert` the prior day's dir then import — NOT a package-relative `from ..x import`, which fails under a bare `python3 experiment.py` run as in Task 6); verify exit 0 standalone.
- [ ] **Step 5: Commit.** `git commit -m "feat(doing): opt-in cross-day build_on artifact chaining"`

### Task 6: Regenerate all 115 + full verify

- [ ] **Step 1:** Run `doing_build.js` across all 115 lesson days (a loop over the manifest day list; `maxRounds:3`).
- [ ] **Step 2:** Verify: `find sessions -name experiment.py -exec python3 {} \;` — every one exits 0 and prints ✅; `python3 -m pytest sessions/_compiler/tests/ -q` green; the dead-path test green.
- [ ] **Step 3: Commit** the regenerated artifacts. `git commit -am "feat(doing): regenerate all 115 experiment.py as real self-checking artifacts"`

---
**Done when:** 0 stub `experiment.py` remain (all pass `check_experiment` + exit 0 + print ✅), 0 dead-lab references in learner/source files, produce answers are gated, and the compiler test suite is green.
