# Interest Enforcement + Momentum — Design Spec

**Date:** 2026-07-23
**Author:** ruifengli (+ Claude)
**Status:** Draft for review
**Scope of this session:** Close the interest-enforcement hole in the V9 lesson-build engine and remove the structural causes of mid-lesson momentum loss, **without shortening lessons** (long, fully-covered lessons are an explicit goal). Two phases: **(1) engine + skills** (an always-on absolute interest floor, a momentum/failure-cluster check, a reader-separation rule, interest as a first-class skill gate, honest interactive play, and a truthful QA contract); then **(2) rebuild Module 2's 9 days** through the improved engine. Companion audit: `sessions/m02_v9_interest_audit.md`.

---

## 1. Problem

A full 9-day audit of `sessions/m02-the-neuron/` (two independent instruments — the project's own interest+tone judge, and an adversarially-verified 12-year-old-lens agent audit) found the lessons are genuinely beginner-friendly: **all 9 open with a strong hook and put the felt picture before any formula; Days 1–6 score `MATCHES_NOTEBOOK` on every interest and tone lever.** No P0. The interest-first rebuild worked.

But four systemic weaknesses cap a beginner's *want-to-keep-going*, and the engine cannot currently defend against most of them:

1. **Interest is only enforced where a notebook exists.** `judge_interest` / `judge_tone` are notebook-relative and return `N/A` when `notebook_yardstick` is null; the interest lens in `lesson_build.js` only P0s on `BELOW_NOTEBOOK`/`WORSE`, so an `N/A` **passes vacuously**. **35 of 46 built days (76%)** are null (m02 Days 7–9, all of m07/JAX, m03, m05, m06, scaling, …). On those days the stated #1 goal is unenforced. Even where a notebook exists there is **no absolute floor**, so "matches a mediocre baseline" passes.
2. **Late-lesson "failure-mode wall."** On 7 of 9 days, 3–5 "Puzzle → Cause → Remedy / here's a limit" units run back-to-back before the recap with no play or payoff between them; the `momentum` lever is WEAK on 8/9 days. This is where a beginner bounces.
3. **"Play" is largely an illusion.** The `%%% demo` "run it ▶" buttons do not compute — they un-hide a pre-baked `<pre class="demo-out" hidden>` answer. Only ~1 genuinely interactive drag widget exists per lesson, usually landing late. The "run it" label overpromises.
4. **Interview/Reader-B register leaks into the beginner voice.** Day 2 c9 carries a scripted "here's how you'd explain it in an interview" monologue with matrix notation `(x·W₁)·W₂ = x·(W₁·W₂)`; Day 6 drops "non-convex," "representational ceiling." This violates `CLAUDE.md §1` (never mix Reader A and Reader B).

### 1.1 Root causes

- **Enforcement coupled to `notebook_yardstick`.** The interest/tone judges were built as *delta* judges (lesson-vs-notebook). When no notebook exists — the majority case — the axis is undefined and the P0 gate silently disables. This was a deliberate cost choice (`notebook_yardstick:null` to stay under a bridge budget); **cost is no longer a constraint**, so the hole should be closed.
- **Coverage completeness has no momentum counterweight.** The architect's Coverage Spec Rule requires, for every failure, its cause + named remedy (each remedy its own covers topic) + a capability limit; `AUTHORING.md §7` maps each covers topic to one concept unit. That correctly produces *complete* lessons, but with nothing telling the author to *interleave* play between consecutive failure units, the completeness stacks into a late-lesson wall.
- **`%%% demo` is a reveal widget by design, mislabeled as execution.** Genuine interactivity lives only in `%%% viz` iframes, which authors under-use.
- **Unreconciled dual mandate.** The builder's three-layer architecture bakes "interview answer" into every lesson's Staff Depth Layer while also demanding 12-year-old prose, so authors *append* interview register instead of removing it.

### 1.2 Non-goal (explicit)

**No length cap, no concept-count ceiling, no word budget.** Long, fully-covered lessons are wanted. Every change below targets *enforcement, momentum, honesty of play, and reader-separation* — never length. Coverage is never cut or deferred to shorten a day; the momentum fix is always to **interleave**, not to remove.

---

## 2. Goal

Make **interest a true, always-on P0 gate on every lesson** (notebook or not), give the build loop a signal for the failure-mode-wall momentum dip, make interactive play honest and real, and keep the beginner voice free of interview scripting — while lessons stay as long and complete as they are today.

---

## 3. Design

Seven components. ①②⑥ are code (test-driven); ③④⑤⑦ are skill/contract changes. Skill edits must be applied in **both** `.claude/skills/` and the `frontier_lab_refactor_skills_v8/skills/` mirror (parity is required by convention).

### ① Absolute interest floor — always on *(engine)*

**File:** `sessions/_compiler/gates/coverage_judge.py`

New `judge_interest_absolute(lesson_text, model=MODEL, timeout=90)`, modeled on `judge_interest` but with **no notebook in the prompt**. It scores the 7 existing levers on their own merits (each `GOOD` / `WEAK` / `MISSING`) against fixed written anchors:

| Lever | `GOOD` anchor (abbreviated) |
|---|---|
| aspiration_hook | opens with wonder / "this unlocks X", not problem-first ("without this, X is useless") |
| relevance | ties to things a beginner knows (ChatGPT, face unlock, recommendations, games) — and returns after the hook, not once |
| invites_play | ≥1 genuine "change something → see it change" (live widget / predict-then-run), not only static pictures |
| momentum | ideas keep moving; no run of ≥3 failure/limit beats with no play/payoff between them |
| breadth_spark | teases a landscape/"family" to explore, not a single corridor |
| delight_voice | genuine energy, brilliant-friend voice, memorable lines |
| payoff | concrete wow-moments + victory laps |

Returns `{status, overall: FLOOR_MET | BELOW_FLOOR, dimensions:[{name, verdict, reason, fix}], top_fixes, summary}` — same shape as the existing judges (so the panel and CLI render it unchanged).

**Pass bar (holistic + anchored, calibrated):** the judge returns `BELOW_FLOOR` if **any lever is MISSING**, or if it holistically judges that a curious beginner would not want to continue. As a calibration guide, `≤1 WEAK` with no MISSING should read as `FLOOR_MET`. The bar is **calibrated to pass a genuinely engaging lesson and fail a dutiful/overwhelming one — not tuned to rubber-stamp all current lessons.** Empirical calibration step: run it on all 9 m02 lessons; Days 1–6 (rated EXCELLENT/GOOD) must pass; **Day 9 (rated MIXED, multiple WEAK levers) is the expected edge case** — if it falls `BELOW_FLOOR`, that is the floor working (it becomes a phase-2 rebuild target), not a reason to loosen the bar.

`run_from_paths` calls `judge_interest_absolute` **unconditionally** and adds it under a new key `interest_absolute`. The existing notebook-relative `interest` runs additionally when a yardstick exists (cost is not a concern). The CLI prints an "Absolute Interest Floor" section.

### ② Interest lens gates on the floor every round *(engine)*

**File:** `sessions/_compiler/workflows/lesson_build.js`

The `interest` lens reads `interest_absolute` and emits a **P0 (`kind:'interest'`) on `BELOW_FLOOR`, every round, regardless of notebook** — so the author→judge loop self-corrects on interest while writing (per the chosen cadence). Where a notebook exists, the notebook-relative `BELOW_NOTEBOOK`/`WORSE_THAN_NOTEBOOK` remains an *additional* P0. An `N/A`/absent interest result on a day that should be enforced surfaces as an explicit finding, never a silent pass — so "unenforced" can never again be mistaken for "passed."

### ③ Momentum / failure-cluster check *(engine — advisory)*

**File:** `sessions/_compiler/gates/concept_structure_gate.py`

Because ① now runs the `momentum` lever on **all** days, the LLM judge is the real enforcer. Add a cheap **deterministic advisory warn** (offline backstop; mirrors the existing "visualize the build-up" advisory — appends to the gate's message list and **never flips the gate's `ok[0]` / exit code**): flag **≥3 consecutive concept units whose title/tag matches a failure/limit pattern** (`dead-`, `vanish`, `saturat`, `explod`, `collapse`, `overfit`, `underfit`, `-limit`, `trap`, `puzzle`, `cannot`, `wall`) **with no intervening play/payoff widget** — i.e. a `%%% demo` or `%%% viz`, which live *inside* a concept. (`@@@ produce` is excluded: it is always the final content block and can never interrupt a mid-lesson cluster.) The message names the remedy: **interleave a win / live widget between the traps — do not cut or defer coverage.** The keyword list is broad, so it must key on genuine failure clusters, not incidental word matches (see §4 ③).

### ④ Coverage-Spec-Rule counterweight *(architect skill)*

**Files:** `.claude/skills/frontier-curriculum-architect/SKILL.md` (+ mirror), `sessions/_compiler/AUTHORING.md`

Keep the full Coverage Spec Rule unchanged (every failure + cause + remedy + limit; lessons stay long and complete). **Add one sequencing rule:** when 3+ failure/limit units would run consecutively, **interleave a play or payoff beat** (a live widget, a predict-then-run, or a "you just unlocked…" victory lap) between them. This is a sequencing constraint only — **no compression into tables, no deferral of remedies to a later day.**

### ⑤ Reader-separation + interest-as-P0 + real-play rules *(builder skill)*

**Files:** `.claude/skills/frontier-lesson-builder/SKILL.md` (+ mirror), `.claude/skills/frontier-visual-evidence-builder/SKILL.md` (+ mirror)

- **Reader-separation:** ban scripted "here's how you'd say it in an interview" monologues and Reader-B notation (bare matrix identities, terms like "non-convex"/"representational ceiling" without a plain gloss) from foundation lesson bodies. The Staff Depth Gate still requires its depth, but framed as **one plain-language empowerment line** ("now you can explain *why* deep nets need a bend"), not interview scripting. Resolve the dual mandate by stating explicitly: staff insight stays as beginner-framed empowerment; interview *scripting/notation* does not belong in a Reader-A body.
- **Interest as first-class skill P0s:** promote 1–2 interest levers into the builder's Non-negotiables / Learning Barrier Gate (e.g. "each concept opens with aspiration/relevance, not a problem/warning" and "the lesson teases breadth/payoff") so interest is gated by the skill itself, independent of the engine or a notebook.
- **Honest, real play:** require **≥2 genuinely interactive `%%% viz` widgets per lesson, front-loaded** (not one, late). Relabel `%%% demo` buttons from "run it ▶" to a reveal verb ("reveal ▶" / "show the answer ▶") so they stop implying computation. (Compiler note in ⑥.)

### ⑥ Honest `%%% demo` label default *(engine — small)*

**File:** `sessions/_compiler/v8lib.py` (`_compile_concept` demo rendering)

The `%%% demo` `label:` is author-supplied; where omitted, change the default button text from a "run"-style verb to a reveal verb, so a demo that only un-hides a pre-baked output never advertises computation. (The primary fix is the ⑤ authoring rule; this is the safety default.)

### ⑦ Truthful QA contract *(refactor-qa skill)*

**File:** `.claude/skills/frontier-refactor-qa/SKILL.md` (+ mirror)

Document that interest now has **two enforcers**: an **absolute floor (always on, every day)** and a **notebook-relative ceiling (when a yardstick exists)**. State explicitly that `notebook_yardstick: null` **no longer means "interest unenforced."** Add the momentum/failure-cluster advisory to the V9 Concept Gates list. Remove the stale "null keeps interest N/A to save cost" rationale.

---

## 4. Testing

Test-driven for all code changes (the compiler suite has 80+ passing tests under `sessions/_compiler/tests/`).

- **①** `judge_interest_absolute`: unit test the output shape and the `FLOOR_MET`/`BELOW_FLOOR` parsing with a stubbed/mocked bridge response (mirror existing `test_coverage_judge.py` patterns); test that `run_from_paths` populates `interest_absolute` even when `notebook_yardstick` is null.
- **③** failure-cluster detector: unit tests over synthetic source — a 3-failure run with no play → warn; the same run with an interleaved `%%% viz` → no warn; a 2-failure run → no warn; **a benign non-failure title that merely contains a matched substring (e.g. "The problem with one line", "The limit of one neuron") → no false-positive warn.** Assert it never flips `ok[0]` / the exit code.
- **⑥** demo-label default: test that an author-supplied `label:` is preserved verbatim, and an omitted label renders the reveal-verb default.
- **Regression:** full `pytest sessions/_compiler/tests/` green; recompile an unchanged shipped lesson and confirm byte-identical (idempotence invariant).
- **Calibration:** run `coverage_judge.py` (with the new absolute judge) on all 9 m02 lessons; confirm Days 1–6 pass the floor; record Day 9's verdict; tune the rubric anchors only if a genuinely-engaging lesson fails or a genuinely-weak one passes.

---

## 5. Phase 2 — rebuild Module 2 through the improved engine

After the engine + skills land, rebuild m02's 9 days via the standard V9 loop (`lesson_build.js`, `maxRounds` per day), applying the per-day `single_most_important_fix` from the audit: **interleave** the failure walls with wins/live widgets (D2 c7–c10, D6 c4–c8, D9 c6–c8, D5 c6–c7, D7 c3–c5), add ≥2 real `%%% viz` widgets earlier, and **strip the D2 interview monologue** (`(x·W₁)·W₂=x·(W₁·W₂)` + He/Xavier/GELU box). **Keep every day long and fully covered** — sequence and enrich, do not cut. Freeze the V9 invariants (quest-ids `wf2-*`/`wf3-*`, nav chain, module labels). Sign-off: every day clears the new absolute interest floor and the deterministic gates; re-run the interest+tone judge to confirm `MATCHES_NOTEBOOK`/`FLOOR_MET`.

---

## 6. Out of scope / non-goals

- Any length/concept-count/word cap (explicit non-goal).
- Cutting or deferring coverage to reduce density.
- Making `%%% demo` compute in-browser (heavier; the reveal-verb relabel + `%%% viz` requirement is sufficient this session).
- Rebuilding other modules (m03/m05/m06/m07/…). The engine change enforces interest on them going forward; their rebuilds are separate work.
- Changing the notebook-relative judges' behavior where a notebook exists (they remain as the additional ceiling).

---

## 7. Rollout order

1. **① + ②** (absolute floor + loop gating) — the core fix; land and calibrate first.
2. **③ + ⑥** (momentum advisory + honest demo default) — small engine additions with tests.
3. **④ + ⑤ + ⑦** (architect / builder / refactor-qa skill edits, both trees, parity-checked).
4. Re-run the compiler suite + the interest judge on all 9 m02 lessons (calibration gate).
5. **Phase 2:** rebuild m02's 9 days through the improved engine.

Each of 1–3 is independently committable; nothing here changes a shipped `lesson.html` until Phase 2.
