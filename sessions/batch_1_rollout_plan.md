# Batch 1 — Coach Layer Rollout Plan

_Plan-first artifact for the `frontier-curriculum-refactor` Coach Layer rollout to Batch 1. Written **before any lesson edit**. Date: 2026-07-06._

Companion docs: [`COACH_STYLE_GUIDE.md`](./COACH_STYLE_GUIDE.md) (blocks + rules) · [`enhance_lesson_checklist.md`](./enhance_lesson_checklist.md) (per-lesson checklist) · [`lesson_audit.py`](./lesson_audit.py) (automated gate) · [`coach_layer_pilot_report.md`](./coach_layer_pilot_report.md) (pilot precedent).

---

## 1. Scope

Batch 1 = three math-foundation / scaling modules (rollout step 1 per the pilot report §13):

| Module | Lessons | Theme |
|--------|:---:|-------|
| `m01-shape-of-data` | 6 | NumPy / tensor foundations |
| `m10a-scaling-laws` | 6 | Scaling laws (Kaplan → Chinchilla → IsoFLOP → data wall) |
| `m13-isoflops-scaling-law` | 6 | Hands-on IsoFLOP experiment build |

**18 lessons total. 2 are already done (pilot) and are excluded from all edits:**

- `m01-shape-of-data/day-05-logs-and-exponents` — Coach Layer + interactive `.vlab`, audits clean.
- `m10a-scaling-laws/day-04-power-law-derivation` — Coach Layer + interactive `.vlab`, audits clean.

**→ 16 lessons in play.** All 16 are on the **new sidebar shell** (`class="module-section"`, Light/Dim/Dark/Midnight switcher). `lesson_audit.py` reports all 18 structurally **OK** (7 sec / 7 got-it / 3 demos / 4 quiz / BUILD present) — no repair work; this is purely a Coach Layer overlay + targeted P0 visual pass.

---

## 2. Method (how this plan was built)

1. **Inspect** — listed all 18 lessons; confirmed all on the new shell; ran `lesson_audit.py` (targeted, so `_recover_set.json` untouched) → 18/18 OK with soft COACH advisories.
2. **Visual audit** — ran the `frontier-visual-auditor` rubric over the 16 non-pilot lessons via 3 read-only module agents (one per module), each citing evidence (`data-demo` keys, element ids, iframe interactivity) and scoring 1–5.
3. **Iframe reality check** — the three embedded viz files were inspected directly: `matmul.html` (3 range sliders + D3) and `scaling-laws.html` (6 range sliders + D3) are **genuinely interactive**; `broadcasting.html` is button-driven click-through (0 sliders). Lessons embedding a slider+D3 viz therefore already score ~4 and are **not** P0.
4. **Precise Coach-element scan** — grepped each lesson for the exact markers (😕 / 🪜 Math Ladder / 🎤 interview / Staff Lens / Acceptance criteria / 📓 research log / Chinese chars) so this plan adds **only what is missing**.

**Upgrade decision rule (from the auditor skill):** P0 = hard math/systems concept **and** current visual score < 4. P1 = medium concept **and** score < 3.5. P2/P3 = nice-to-have or lesson is legitimately code/writing (terminal walkthrough acceptable).

---

## 3. Visual audit results (16 lessons)

| Lesson | Concept difficulty | Visual cat. | Score | Current visual | Priority |
|--------|:---:|:---:|:---:|---|:---:|
| m01 · day-01 · arrays | code/applied | C+D | 3 | fake-terminal DEMOS + static SVG build | P2 |
| m01 · day-02 · indexing-slicing | code/applied | C+D | 3 | fake-terminal DEMOS + static SVG build | P2 |
| m01 · day-03 · broadcasting-dtypes | hard (shapes) | B | 4 | **iframe `broadcasting.html`** (button-driven) + SVG | P1 |
| m01 · day-04 · matmul-and-shapes | hard (shapes) | B | 4 | **iframe `matmul.html`** (3 sliders + D3) + SVG | P1 |
| m01 · day-06 · random-seeds | code/applied | C+D | 3 | fake-terminal DEMOS + static SVG (terminal is apt) | P2 |
| **m10a · day-01 · kaplan-paradigm** | **hard math** | **C** | **2** | **terminal-only** (allocation/perparam/sampleeff) | **P0** |
| **m10a · day-02 · chinchilla-correction** | **hard math** | **C** | **2** | **terminal-only** (gpt3/fit/gopher) | **P0** |
| m10a · day-03 · isoflops-methodology | hard math | A+B | 4 | **iframe `scaling-laws.html`** (6 sliders + D3) + SVG | P1 |
| **m10a · day-05 · data-wall** | **hard math+systems** | **C** | **2** | **terminal-only** (crossing/repetition/synthetic) | **P0** |
| m10a · day-06 · scaling-simulator-visualization | medium | A+B | 4 | **iframe `scaling-laws.html`** + SVG (the viz lesson) | P1 |
| m13 · day-01 · isoflops-experiment-design | medium | C | 2–3 | terminal-only (fixD/grid/fit) — design lesson | P2 |
| m13 · day-02 · automation-scripting | code/systems | C | 2 | terminal-only (beforeafter/launch/monitor) — apt | P2 |
| m13 · day-03 · execution-data-aggregation | code/data | C | 2 | terminal-only (matrix/checkpoint/aggregate) — apt | P2 |
| **m13 · day-04 · plotting-parabolas** | **hard math** | **C/D** | **2** | **terminal-only** (rawspace/fit/p0) — no plot at all | **P0** |
| m13 · day-05 · deriving-scaling-law | hard math | B | 4 | **iframe `scaling-laws.html`** (6 sliders + D3) + SVG | P3 |
| m13 · day-06 · scientific-documentation | writing | C | 1 | terminal-only (vague/fit/symbol) — writing lesson, apt | P3 |

### P0 set — 4 lessons (hard math/systems + terminal-only, score 2)

1. **m10a · day-01 · kaplan-paradigm** — the compute-optimal N/D allocation is taught by three canned console prints. No way to see the allocation split or the power law move.
2. **m10a · day-02 · chinchilla-correction** — the Kaplan→Chinchilla shift and "≈20 tokens/param" rule are text-only; the IsoFLOP valley that justifies the correction is never drawn.
3. **m10a · day-05 · data-wall** — supply-vs-demand crossing, repetition-decay `D'`, and synthetic data are three dense papers reduced to canned text; the crossing year is a fixed assumption the learner can't stress-test.
4. **m13 · day-04 · plotting-parabolas** — teaches an **inherently visual object** (loss-vs-log(N) parabola + vertex) with zero plots. The one lesson in m13 where terminal is clearly not enough.

**Not P0 (already have an interactive viz, score 4):** m01/day-03 (broadcasting.html), m01/day-04 (matmul.html), m10a/day-03, m10a/day-06, m13/day-05 (all scaling-laws.html). These get the **Coach Layer prose only** — no new visual (per "do not blindly convert every playground to D3").

**Terminal-acceptable (code/writing/methodology):** m01/day-01,02,06; m13/day-01,02,03,06 — Coach Layer prose only; keep the terminal playground.

---

## 4. Per-lesson Coach Layer work (add only what is missing)

Every in-play lesson gets the missing Coach blocks inserted **inside `.sec-body`, before the section's `.gotit`**, using the exact templates in `COACH_STYLE_GUIDE.md` §5. Legend: **●** = add, **○** = already present (leave it), **–** = intentionally skip (see notes).

| Lesson | 😕 pain | 🪜 Math Ladder | 🎤 interview | Staff Lens | ✔ Accept. | 📓 log | 中文 (2–4) |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| m01 · day-01 · arrays | ● | – (no core formula) | ● | ● | ● | ● | ● |
| m01 · day-02 · indexing-slicing | ● | – (no core formula) | ● | ● | ● | ● | ● |
| m01 · day-03 · broadcasting-dtypes | ● | ● (broadcast rule) | ● | ○ | ● | ● | ● |
| m01 · day-04 · matmul-and-shapes | ● | ● ((m,k)·(k,n)→(m,n) + 2mnk FLOPs) | ● | ○ | ● | ● | ● |
| m01 · day-06 · random-seeds | ● | – (no core formula) | ● | ○ | ● | ● | ● |
| m10a · day-01 · kaplan-paradigm | ● | ● (N∝C^a, D∝C^b, a+b=1) | ● | ○ | ○ | ● | ● |
| m10a · day-02 · chinchilla-correction | ● | ● (N,D∝C^0.5; ≈20 tok/param) | ● | ○ | ○ | ● | ● |
| m10a · day-03 · isoflops-methodology | ● | ● (parabola vertex → N*) | ● | ○ | ○ | ● | ● |
| m10a · day-05 · data-wall | ● | ● (repetition decay `D'`) | ● | ○ | ○ | ● | ● |
| m10a · day-06 · scaling-simulator-visualization | ● | ● (dollars→FLOPs; C=6ND) | ● | ○ | ○ | ● | ● |
| m13 · day-01 · isoflops-experiment-design | ● | ● (D = C/6N at fixed C) | ● | ● | ○ | ● | ● |
| m13 · day-02 · automation-scripting | ● | – (software lesson) | ● | ● | ○ | ● | ● |
| m13 · day-03 · execution-data-aggregation | ● | – (data lesson) | ● | ● | ○ | ● | ● |
| m13 · day-04 · plotting-parabolas | ● | ● (vertex `N* = −b/2a` in log space) | ● | ○ | ○ | ● | ● |
| m13 · day-05 · deriving-scaling-law | ● | ● (slope of log-log = exponent) | ● | ● | ○ | ● | ● |
| m13 · day-06 · scientific-documentation | ● | – (writing lesson) | ● | ● | ○ | ● | ● |

**Totals to add:** pain-point ×16 · Math Ladder ×12 (skip the 4 non-formula lessons) · interview ×16 · Staff Lens ×6 · Acceptance criteria ×5 (all in m01) · research log ×16 · bilingual scaffold ×16.

**Notes / guardrails:**
- **Math Ladder only for formula-heavy lessons.** The 4 skips (arrays, indexing, random-seeds, automation, aggregation, documentation → the ones marked "–") have no single core equation; forcing a ladder would be noise. `lesson_audit.py`'s math-ladder advisory will remain on those — that is expected and will be noted in the report, not "fixed".
- **Produce artifact — do NOT fabricate `experiment.py`.** m01 lessons already reference `experiment.py`. Several m10a/m13 lessons legitimately produce a **notebook / `.md` / LaTeX** artifact (e.g. `01_scaling_simulator.ipynb`, `day5_data_wall_analysis.md`, `RESULTS.md`, a `\section{Methodology}`). Keep the existing artifact; only add the `<h4>Acceptance criteria</h4>` list where missing (the 5 m01 lessons).
- **Interview + Staff Lens** were re-checked with exact markers (🎤 / "Staff Lens"), not the loose word "interview" — the word appears in shell chrome, so every lesson needs a real 🎤 block added.
- **Chinese dosage:** 2–4 short touches per lesson, concentrated in §1 (pain) and §2 (intuition) only. Never in formulas, code, or the interview phrasing (guide §3, memory `prefers-english-learning-content`).

---

## 5. P0 visual upgrades (4 lessons) — design specs

All four are **dependency-free inline SVG + vanilla JS** using the repo `.vlab` template proven in the pilot (`m01/day-05`, `m10a/day-04`). **No CDN, no new fonts, offline-safe, theme-aware, keyboard-reachable sliders.** Inserted into **Section 3** (`data-sec="play"`) **above** the existing fake-terminal playground; the terminal keeps its 3 `data-demo` buttons and its **completion-gate** role (untouched). Every lab carries the 7 required cards: learning question · labeled axes · clear change on control move · "what you should notice" · misconception note · "where this simplifies reality" · frontier-lab relevance.

### 5.1 m10a · day-01 · kaplan-paradigm — "Allocation dial"
- **Learning question:** given a fixed compute budget `C`, how should it split between parameters `N` and data `D`, and what did Kaplan claim?
- **Control:** one slider = the parameter-exponent `a` in `N ∝ C^a` (with `D ∝ C^(1−a)`), 0 → 1. A preset button snaps to **Kaplan a≈0.73** and **balanced a≈0.5**.
- **Two panels:** (left) a stacked bar showing the N-vs-D share of the budget as `a` moves; (right) a log-log line of `N* vs C` whose **slope readout = a**. Badge reads "parameter-heavy (Kaplan)" vs "balanced".
- **Notice / misconception / simplifies:** Kaplan's slope tilts spend toward giant, under-fed models; the misconception is "bigger model = better" independent of tokens; simplification = single clean power law, no noise.

### 5.2 m10a · day-02 · chinchilla-correction — "IsoFLOP valley"
- **Learning question:** at one fixed budget, why is there an interior best model size, and how does the Kaplan pick differ from Chinchilla's?
- **Control:** slider = model size `N` along one IsoFLOP curve (D forced by `D = C/6N`); the loss curve `L(N)` is drawn as a **U**; a marker shows the current `N`, and a vertical line shows the true valley (`≈20 tokens/param`).
- **Second element:** a small toggle overlays the **Kaplan allocation point** (too far right/parameter-heavy) on the same U so the "same compute, worse loss" gap is a visible vertical distance.
- **Notice / misconception / simplifies:** the valley is interior (not at the extremes); misconception = "Kaplan and Chinchilla disagree about the math" (they disagree about the *experiment* — fixed vs tuned schedule); simplification = one synthetic budget, smooth curve.

### 5.3 m10a · day-05 · data-wall — "Crossing simulator"
- **Learning question:** when does compute-demanded tokens exceed human-text supply, and how sensitive is that year to assumptions?
- **Controls:** 2 sliders — internet-text growth rate, and compute growth / overtraining multiplier. Plot two curves (demand vs supply) on a year axis; a **crossing-year readout** updates live and the uncertainty band widens/narrows.
- **Second element:** a repetition-decay mini-panel — slider = epochs `R_D`, curve shows effective data `D'` saturating toward its asymptote (the lesson's `D' = U + U·R*·(1 − e^(−R_D/R*))`).
- **Notice / misconception / simplifies:** the wall is a *range* (2026–2032), not a date; misconception = "repeating data is free" (returns decay); simplification = smooth exponentials, no policy/licensing shocks.

### 5.4 m13 · day-04 · plotting-parabolas — "Fit the valley"
- **Learning question:** why fit a parabola in **log** model-size space, and where is the compute-optimal `N*`?
- **Two panels:** (left) raw `N` vs loss — points bunched near the origin, bowl hard to see; (right) `log N` vs loss — evenly spaced points with a clear **U**, a fitted parabola overlaid, and the **vertex `N* = exp(−b/2a)`** marked with a dropline.
- **Control:** slider nudges the fit's curvature `a` (or adds one noisy point) so a **too-flat / failed fit** (`a ≤ 0`) is demonstrable — the danger the terminal demo only describes.
- **Notice / misconception / simplifies:** log-space makes the bowl symmetric and fittable; misconception = "pick the lowest sampled point" (the true minimum sits *between* samples); simplification = clean quadratic, few points.

> **Reuse note:** the m10a P0 labs echo the reference `.vlab` in `m10a/day-04`; the m13/day-04 lab echoes the pilot logs lab. Where a lesson's learning question is already served by the embedded `scaling-laws.html` (day-03, day-06, m13/day-05) **no new lab is built** — those are P1/P3 and out of the visual scope of this batch.

---

## 6. Preservation contract (never violate — guide §7)

For every edited lesson, these must be byte-behaviour-identical after the edit:

- `data-quest-id` on `<body>` (the localStorage key).
- Every prev / next / hub `href` (top and sidebar nav).
- 7 sections, their `id`s and `data-sec` keys; 7 `.gotit` buttons; the progress/reset/checklist script.
- `var QS` (4 quiz entries `{q,opts,ans,fb}`), `var DEMOS` (3 keys ↔ 3 `data-demo` buttons), `var BUILD` (scroll-reveal).
- Glossary tooltip script + every `.term[data-tip]` (new terms get a `data-tip`).
- Self-contained / offline: no new external assets, CDN, or fonts.
- Section wrapper tags, `id`s, and `<div class="sec-body">…</div></section>` boundaries kept intact so `_shell_migrate.py` can still extract cleanly.

**Additive-only.** No failure mode, trade-off, equation, or complexity claim is deleted; no precise language is swapped for cheerleading; no motivational fluff.

**P0 visual specifics:** the inline `.vlab` is added **above** the existing terminal playground; the terminal's 3 `data-demo` buttons remain the section's completion gate. New viz height, if iframed, would need the postMessage sender (memory `viz-iframe-autoresize-bug`) — but these labs are **inline** (not iframed), so that bug does not apply.

---

## 7. Execution order (sequential — one file per session, no parallel lesson writing)

Per CLAUDE.md §4 and memory `capability-spiral-build`: **no subagents for parallel writing** (tone/analogy drift). Edit lessons one at a time, in this order (foundation → scaling → hands-on; within a module, foundational day first). Run the per-file verify gate after each before moving on.

1. `m01` — day-01, day-02, day-03, day-04, day-06 (5)
2. `m10a` — day-01 **(+P0)**, day-02 **(+P0)**, day-03, day-05 **(+P0)**, day-06 (5)
3. `m13` — day-01, day-02, day-03, day-04 **(+P0)**, day-05, day-06 (6)

P0 lessons get the Coach Layer prose **and** the inline `.vlab` in the same session.

---

## 8. Verification gate (run after each file; summarized in the report)

1. `node --check` on the lesson's extracted `<script>` blocks (JS syntax).
2. jsdom headless load — no thrown error; asserts 7 sections / 7 got-it / 4 quiz / 3 demos / BUILD present / tooltip controller created; for P0 files, dispatch `input`/`click` and assert the lab SVG re-renders and readouts respond.
3. `python3 sessions/lesson_audit.py <changed paths…>` (targeted — leaves `_recover_set.json` alone) → **OK**, and the six core COACH advisories (pain / ladder / interview / staff / log / bilingual) cleared for that file (except intentional Math-Ladder skips on non-formula lessons).

**Definition of done for the batch:** all 16 lessons OK with no hard failures; each has its full missing-Coach-element set added; the 4 P0 labs render and interact; nav / quest-id / quiz / playground / build / tooltips preserved everywhere; Chinese light (2–4 touches); depth ≥ before. Then write `sessions/batch_1_rollout_report.md`.

---

## 9. Risks

1. **Concurrent edits.** `sessions/` is edited by other sessions; re-read each file immediately before editing (the tooling enforces "file modified since read").
2. **Migrator kicker/finale bug (pilot report §12, risk #1).** Migrated lessons may carry a wrong kicker / `nav-group-label` / finale. Where present in an in-play file, fix it (coherence, in-scope per checklist Step 5) using `<title>` + `.brand-sub` as the authoritative source. Not a curriculum-wide sweep here.
3. **Math-Ladder advisories on non-formula lessons** will persist by design (arrays, indexing, seeds, automation, aggregation, documentation). The report will list these as intentional, not failures.
4. **P0 lab scope creep.** Keep each lab to one core learning question + the 7 required cards; do not gold-plate. Reuse the pilot `.vlab` pattern rather than inventing new markup.
