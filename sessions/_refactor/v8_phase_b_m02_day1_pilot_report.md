# v8 Phase B — m02 Day 1 Source-First Compiler Pilot Report

**Loop:** `v8_phase_b_m02_day1_compiler_pilot`
**Date:** 2026-07-08
**Owner skills:** `frontier-curriculum-architect` (source-first authoring), `frontier-lesson-builder` (reader flow), `frontier-refactor-qa` (gates), `frontier-visual-evidence-builder`
**Scope:** `sessions/m02-the-neuron/day-01-single-neuron/` **only** (day 02–09 untouched, still hand-authored v7.6).
**Spec:** `sessions/_refactor/v8_source_first_authoring_plan.md` · **Diagnosis:** `sessions/_refactor/v8_exemplar_distillation_report.md`

---

## TL;DR

- **Verdict: PASS.** The smallest-viable v8 source-first path is live for one day. `source.md` is the canonical authored source; `lesson.html` is now **compiled** by `sessions/_compiler/compile_lesson.py`.
- **Is the compiled Day 1 smoother than the v7.6 HTML on the first screen? YES.** v7.6 opened frontier-first ("GPT-3 has a weight matrix with 49,152 columns…"); the compiled Day 1 opens human-first ("your brain is soaking up thousands of tiny signals… *weigh things, then decide*") and adds a front-loaded **Jargon Ladder** the v7.6 lesson never had. The GPT-3 payoff is preserved but moved to §4, after the mechanism.
- **At least competitive with the notebook on first-screen barrier? YES.** It now matches the notebook's human-first opening and its "Jargon Buster" table, while keeping v7.6's far stronger staff depth later in the lesson.
- **Zero shell regression.** The CSS block and all four `<script>` blocks (incl. the 19,299-char JS engine with `DEMOS/BUILD/QS`) are **byte-identical** to the v7.6 donor. Only the reader-flow prose regions changed.
- **Deterministic.** Compiling twice yields byte-identical output.
- **Remaining issues are P1/P2 only** (see §7).

---

## 1. What was built

| Artifact | Path | Role |
|---|---|---|
| Compiler | `sessions/_compiler/compile_lesson.py` | `source.md → lesson.html`; runs Reader Flow Gate (on source, pre-compile) + Shell Invariant Gate (on output) |
| Shell donor | `sessions/_compiler/shells/m02-day-01.donor` | pristine v7.6 shell snapshot (CSS + JS engine reused verbatim); stored without `.html` so audits skip it |
| v7.6 backup | `sessions/_compiler/shells/m02-day-01.v7_6-backup` | rollback copy of the shipped v7.6 lesson |
| Canonical source | `sessions/m02-the-neuron/day-01-single-neuron/source.md` | the authored reader-flow source (front-matter + typed `@@@` blocks + markdown-lite) |
| Compiled output | `sessions/m02-the-neuron/day-01-single-neuron/lesson.html` | now compiled, not hand-edited |

**Compiler design (smallest viable):** reuse the proven shell verbatim from the donor; marker-replace **only** the reader-flow regions (hero, s1, s2, s4, s7), the `.fin` banner, nav, quest-id, sidebar, title, brand-sub, and the authored `DEMOS/BUILD/QS` data. The CSS and JS engine are never touched — which is what guarantees zero behavioral regression. `s3/s5/s6` static scaffolding is carried from the donor; their *content* (playground steps, build pieces, quiz questions) is authored in `source.md` as `@@@ js` data blocks.

---

## 2. The authored change (reader flow, first screen)

Only the first-screen ordering/framing changed. Mechanism, staff-lens, artifact, and **every anchor number** (`x=[2,3]`, `w=[0.5,−1]`, `b=1`, weighted sum `−2`, pre-bias `−1`, `ReLU→0`; symmetric-init; fan-in) are unchanged.

| First-screen dimension | v7.6 shipped HTML | **v8 compiled** | Notebook `01_what_is_a_neural_network` |
|---|---|---|---|
| Hero opening | **"GPT-3 has a weight matrix with 49,152 columns…"** (frontier-first) | **"Right now, without even trying, your brain is soaking up thousands of tiny signals…"** (human-first) | "Let's start with something familiar: your brain!" (human-first) |
| Jargon | scattered inline tooltips | inline tooltips **+ a front-loaded Jargon Ladder table** | "Jargon Buster" table up front |
| Picture vs terms | picture paragraph, then vocab | **picture heading first → Jargon Ladder → then vocab** | brain story → then terms |
| Frontier relevance | in the hero (before orientation) | **in §4, after the mechanism** | (n/a — no frontier framing) |
| Goal framing | quiet promise | quiet promise | "Prerequisites: just curiosity!" |

**Assessment.** On the first screen, the v8 compiled Day 1 **beats v7.6** (which led with scale pressure) and is **at least competitive with the notebook** (same human-first + jargon-buster move), while retaining the module's superior staff depth (silent-failure init, fan-in trade-off, grounded interview line) that the notebook lacks entirely.

---

## 3. Gate results

### Reader Flow Gate (on source, before compile) — PASS 9/9
```
pass hero has no frontier-pressure opening
pass hero opens on human intuition/curiosity
pass s1 has a front-loaded Jargon Ladder
pass mental picture precedes vocabulary in s1
pass s1 defers frontier payoff
pass frontier payoff lands in s4 (after mechanism)
pass s4 carries a staff/interview grounding
pass spine ('brain') runs through hero+s1+s2
pass produce is discovery-framed (predict + observe)
```

### Shell Invariant Gate (on output) — PASS
quest-id frozen (`wf2-d01-neuron`) · 7 module-sections · 8 sidebar `data-target`s (home,s1–s7) · `DEMOS`/`BUILD`/`QS` present · playground ≥3 · **quiz q:4 o:16** · `frontier-lesson:` + `frontier-theme` keys · `.fin` banner · prev/next nav · no unresolved markers.

### Notebook Smoothness Gate (pilot/exemplar) — PASS
First-move, picture-vs-term order, why-it-exists framing, and goal framing all meet or exceed the notebook (see §2 table).

---

## 4. Determinism & shell byte-identity

- **Determinism:** `compile_lesson.py` run twice → `diff` byte-identical. It is a pure function of (`source.md`, donor); no timestamps or randomness.
- **Shell byte-identity vs v7.6 donor:**
  - CSS block: **identical** (20,967 chars)
  - `<script>` #0 (theme init): identical
  - `<script>` #1 (JS engine incl. `DEMOS/BUILD/QS`): **identical** (19,299 chars)
  - `<script>` #2 (glossary tooltips): identical
  - `<script>` #3 (viz iframe auto-resize): identical
  - head + sidebar preamble: identical · footer + tail scripts: identical
  - **Only** lines ~334–491 differ = hero + s1 + s2 + s4 + s7 (the re-authored reader-flow prose).

---

## 5. Repo audit evidence

| Check | Result |
|---|---|
| `python3 sessions/lesson_audit.py m02-the-neuron` | **9 OK / 0 MISSING / 0 LEFTOVER / 0 DEGRADED** |
| `python3 sessions/nav_audit.py` | **PASS** — 0 CASE, 0 BROKEN, 0 orphans |
| `node sessions/staff_lens_audit.js m02` | **9/9 staff-lens present, gap 0**; day-01 `q:4 o:16 errs:[]` (`render:BROKEN` = documented-benign `.sec` selector mismatch) |
| `node --check` on all 4 inline scripts | **all parse** |

---

## 6. Frozen invariants preserved

`data-quest-id="wf2-d01-neuron"` (asserted by the compiler, never invented) · nav prev/next hrefs · `frontier-lesson:<qid>` + `frontier-theme` localStorage · playground ≥3 gate + quiz-answered gate (`q:4 o:16`) · `BUILD`/`DEMOS` behavior · `.fin` completion banner + wiring · shared shell (sidebar + 4-theme Appearance switcher, dark default) · `experiment.py` + `log.md` untouched · artifact contract (Option A/B + frontier-experiment-lab prompt + 5-minute log) preserved.

---

## 7. Remaining issues (P1 / P2 — none blocking)

- **P1 — partial sourcing:** `s2/s4/s7` still use `~~~html` raw-HTML escapes for the complex widgets (brain↔ANN mapping table, judge cards, formula callout, Math Ladder, produce prompt box). The reader-flow-critical regions (hero, s1 picture, Jargon Ladder, definitions) are clean markdown. Progressive conversion of the raw widgets to typed blocks is Phase C.
- **P1 — one day only:** day 02–09 are still hand-authored v7.6; the pilot proves the path, not the full module. Sourcing all 9 days + running the Notebook Smoothness Gate across them is Phase D.
- **P1 (carried, unchanged):** `BL-VE-D1` / `BL-VE-D2` behavioral-viz upgrades (boundary geometry, plotted derivatives) are unaffected by this pilot.
- **P2 — s3/s5/s6 scaffolding carried from donor:** their static wrapper HTML is preserved shell, not yet authored in `source.md` (their content/data *is*). Fold into Phase C.

**No P0.** m02 remains `pass_with_p1`. No other module status changed. m03 untouched.

---

## 8. What this proves for the rollout

The pilot validates the core v8 claim on a known-good target: **you can author reader-flow in a clean linear source and compile a lesson whose shell is byte-identical to the shipped one, while measurably improving the first screen.** The compiler moved the entire frozen-invariant tax off the author and onto tested code — the exact friction the distillation report diagnosed. Next phases: C (gates as standalone scripts + convert raw widgets to typed blocks), D (source all 9 m02 days, run Notebook Smoothness across them), E (first **no-notebook** rollout — m07 JAX or m10a Scaling Laws in Reference mode).
