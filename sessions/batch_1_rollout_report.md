# Batch 1 — Coach Layer Rollout Report

_Completion report for the `frontier-curriculum-refactor` Coach Layer rollout to Batch 1. Date: 2026-07-06._

Companion docs: [`batch_1_rollout_plan.md`](./batch_1_rollout_plan.md) (the plan this executed) · [`COACH_STYLE_GUIDE.md`](./COACH_STYLE_GUIDE.md) · [`enhance_lesson_checklist.md`](./enhance_lesson_checklist.md) · [`lesson_audit.py`](./lesson_audit.py) · [`coach_layer_pilot_report.md`](./coach_layer_pilot_report.md).

---

## 1. TL;DR

Applied the Coach Layer to the **16 non-pilot lessons** of Batch 1 (`m01-shape-of-data`, `m10a-scaling-laws`, `m13-isoflops-scaling-law`), purely **additively**, and built **4 new interactive P0 visual labs**. All edits followed the plan-first workflow: visual audit → P0 identification → written plan → edit → audit → report. The 2 pilot lessons (`m01/day-05`, `m10a/day-04`) were left untouched. Final state: **18/18 OK in `lesson_audit.py`, 0 DEGRADED, 18/18 jsdom-render clean, all 6 interactive labs pass simulated-interaction checks.** No navigation, quest-id, quiz, playground `DEMOS`, `BUILD`, or tooltip machinery was changed.

---

## 2. Files changed

**16 lesson files** (all additive; the 2 pilot lessons are NOT in this list — untouched):

| Module | Lessons edited | P0 lab built |
|--------|----------------|:---:|
| `m01-shape-of-data` | day-01, day-02, day-03, day-04, day-06 | — |
| `m10a-scaling-laws` | day-01, day-02, day-03, day-05, day-06 | day-01, day-02, day-05 |
| `m13-isoflops-scaling-law` | day-01, day-02, day-03, day-04, day-05, day-06 | day-04 |

Plus **new artifact** `sessions/batch_1_rollout_plan.md` (the pre-edit plan) and this report.

**Explicitly NOT touched:** `m01/day-05-logs-and-exponents`, `m10a/day-04-power-law-derivation` (pilot, already done); any lesson outside Batch 1; the `DEMOS`/`BUILD`/`QS` interactive literals; navigation `href`s, `data-quest-id`s, progress/localStorage, quiz, tooltip machinery. Diff verification confirms **no protected line was removed** — the one deletion per file is the old one-line `log.md` nudge, folded into the richer research-log block.

---

## 3. Before → after (Coach markers per lesson)

`lesson_audit.py` + a shell-agnostic jsdom harness confirm every edited lesson kept **7 sections / 7 got-it / 3 demos / 4 quiz / BUILD / tooltip** intact. Coach markers added where missing (● = added this batch, ○ = already present, – = intentionally skipped):

| Lesson | pain😕 | Ladder🪜 | interview🎤 | StaffLens | Accept | log📓 | 中文 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| m01 · day-01 arrays | ● | – | ● | ● | ● | ● | ● |
| m01 · day-02 indexing | ● | – | ● | ● | ● | ● | ● |
| m01 · day-03 broadcasting | ● | ● | ● | ○ | ● | ● | ● |
| m01 · day-04 matmul | ● | ● | ● | ○ | ● | ● | ● |
| m01 · day-06 random-seeds | ● | – | ● | ○ | ● | ● | ● |
| m10a · day-01 kaplan **(P0)** | ● | ● | ● | ○ | ○ | ● | ● |
| m10a · day-02 chinchilla **(P0)** | ● | ● | ● | ○ | ○ | ● | ● |
| m10a · day-03 isoflops-method | ● | ● | ● | ○ | ○ | ● | ● |
| m10a · day-05 data-wall **(P0)** | ● | ● | ● | ○ | ○ | ● | ● |
| m10a · day-06 scaling-simulator | ● | ● | ● | ○ | ○ | ● | ● |
| m13 · day-01 experiment-design | ● | ● | ● | ● | ○ | ● | ● |
| m13 · day-02 automation | ● | – | ● | ● | ○ | ● | ● |
| m13 · day-03 aggregation | ● | – | ● | ● | ○ | ● | ● |
| m13 · day-04 parabolas **(P0)** | ● | ● | ● | ○ | ○ | ● | ● |
| m13 · day-05 deriving | ● | ● | ● | ○ | ○ | ● | ● |
| m13 · day-06 documentation | ● | – | ● | ● | ○ | ● | ● |

**Added this batch:** pain-point ×16 · Math Ladder ×10 · interview ×16 · Staff Lens ×6 · Acceptance criteria ×5 · research log ×16 · bilingual scaffold ×16 · **4 interactive P0 labs**.

---

## 4. What became warmer (Reader A)

- A **bilingual pain-point** (😕 "为什么这里容易卡住 / Why this trips people up") now opens the confusion in §1 of every lesson — e.g. "people memorize `N ∝ C^0.73` without noticing the exponents must add to 1", "people assume a process that *finished* also *succeeded*", "a blind `.dropna()` throws away exactly the diverged runs that were telling you something".
- A **bilingual intuition line** (直觉 / Intuition:) in §2 carries the feeling in Chinese and the mechanism in English — e.g. the data wall as "需求指数上冲，供给几乎是平的，两条线迟早相交".
- Chinese is deliberately **light: 3 touches per lesson** (pain / intuition / research-log), intuition only — per `prefers-english-learning-content`. Never inside formulas, code, or interview phrasing.

## 5. What Math Ladder was added (Reader B)

10 four-rung 🪜 ladders (words → labeled formula → tiny numbers → sanity check) anchor each formula-heavy lesson's core equation:

| Lesson | Ladder target | Sanity-check rung |
|--------|---------------|-------------------|
| m01 · broadcasting | right-aligned axis rule, `out[k]=max(a,b)` | result axis never smaller than either input |
| m01 · matmul | `(m,k)@(k,n)→(m,n)`, `≈2mkn` FLOPs | `k` appears both sides & vanishes; count = m·n |
| m10a · kaplan | `N∝C^a, D∝C^b, a+b=1` | `29×·3.5×≈100×` (a+b=1 restated) |
| m10a · chinchilla | `L=E+A/N^α+B/D^β`, `D=C/6N` | exponents on C sum to 1 |
| m10a · isoflops-method | parabola vertex `x*=−b/2a` | vertex must sit inside sampled sizes |
| m10a · data-wall | repetition decay `D'=U+U·R*(1−e^…)` | `D'→U(1+R*)` ceiling |
| m10a · scaling-simulator | dollars→FLOPs, `N*=√(C/120)` | `D/N≈20`; MFU=1.0 inflates ~1.6× |
| m13 · experiment-design | `D=C/6N` at fixed C | `6ND` must return C |
| m13 · parabolas | vertex `N*=10^{−b/2a}`, `D*=C/6N*` | `a>0` **and** vertex inside window |
| m13 · deriving | slope of log-log = exponent, `polyfit` | `a+b≈1`, else a bad minimum/unlogged axis |

**Intentionally NO Math Ladder** (documented, soft advisory remains): m01 arrays/indexing/random-seeds, m13 automation/aggregation/documentation — none has a single core formula; forcing a ladder would be noise.

## 6. What analogies were added

The lessons already had strong everyday analogies (egg carton, kitchen/pantry, airline bag, road trip, 25 ovens, science fair, suitcase, recipe card). The Coach Layer **added the software/systems framing** where it was thin — arrays as "a fixed-width database column", slicing as "`LIMIT/OFFSET` over sorted rows", broadcasting's size-1 axis as "an infinite stamp", and reinforced each with the bilingual intuition line. Existing "where the analogy breaks" callouts were preserved.

## 7. What Staff Lens was added

10 lessons already had a Staff Lens (preserved intact). Added **6 new Staff-Lens blocks** (named silent failure ⚠️ + trade-off ⚖️, tied to code/design review) to the lessons that lacked one: m01 arrays (silent broadcast bug), m01 indexing (view-vs-copy aliasing), m13 experiment-design (unbracketed valley), m13 automation ("no news is good news" / concurrency-vs-contention), m13 aggregation (survivorship bias from `dropna`), m13 documentation (the omitted decision that changes the answer). Every lesson also got a matching 🎤 **interview-ready block** (20–30s spoken answer: headline + proof + nuance).

## 8. What artifacts improved

- Every lesson keeps its concrete Produce artifact (`experiment.py`, a notebook, `.md` analysis, or a LaTeX Methodology) — untouched.
- Added an explicit **Acceptance criteria** list to the 5 m01 lessons that lacked one.
- Replaced each lesson's one-line `log.md` nudge with a **5-minute research log** block (📓, bilingual) asking for 3 concrete lines (teammate summary / surprising number / open question), folding in each lesson's original log question.

## 9. P0 visual upgrades built (4 labs)

All are **dependency-free inline SVG + vanilla JS** on the repo `.vlab` template (from the pilot) — no CDN, offline-safe, theme-aware, keyboard-reachable sliders, inserted **above** the existing fake-terminal playground (which keeps its completion-gate role). Each carries the 7 required cards (learning question · labeled axes · clear change on control · what-you-should-notice · misconception · where-it-simplifies · frontier relevance).

| Lesson | Lab | Interaction verified (jsdom) |
|--------|-----|------------------------------|
| m10a · day-01 kaplan | **Allocation dial** — exponent-`a` slider splits a 100× compute jump into model vs data growth; log-log slope = a; Kaplan/Balanced presets; regime badge | slider → readout `a=0.73→…`, badge flips parameter-heavy↔balanced ✓ |
| m10a · day-02 chinchilla | **IsoFLOP valley** — model-size slider on `L=E+A/N^α+B/D^β` at fixed C draws the U; valley + Kaplan-pick marked; N/D bars reallocate; presets | slider → badge too-big↔optimal↔over-fed, N/D bars swap ✓ |
| m10a · day-05 data-wall | **Crossing simulator** — demand-growth slider slides the wall year (2025↔2033); epochs slider draws `D'` saturating to its ceiling | growth → "wall ≈ 2028→2025"; epochs 40 → D'=4.00 ✓ |
| m13 · day-04 parabolas | **Fit the valley** — raw-N (bunched) vs log-N panels; window-center slider fits a real least-squares parabola; badge flips **bracketed ✓ / extrapolated** as the sampled window straddles or misses the true vertex | left-arm window → extrapolated; centered → bracketed, fit `a≈0.20` ✓ |

The **not-P0** hard-math lessons already embed genuinely interactive viz (`matmul.html` 3 sliders+D3; `scaling-laws.html` 6 sliders+D3) — per the plan and the "don't blindly convert every playground to D3" rule, those got Coach prose only, no new lab.

---

## 10. Audit results

**`lesson_audit.py` (targeted, `_recover_set.json` left unchanged):**
```
TOTAL lessons audited: 18
  OK: 18   MISSING: 0   LEFTOVER: 0   DEGRADED: 0
```

**jsdom render + JS-syntax gate:** 18/18 render clean, 0 runtime errors, 0 syntax errors; structural counts (7 sec / 7 got-it / 3 demos / 4 quiz / BUILD / tooltip) intact on all. The 6 interactive labs (4 new + 2 pilot) all re-render on simulated `input`/`click`.

**Remaining soft COACH advisories (all expected, none a failure):**
- `math-heavy but no Math Ladder` ×6 — the intentional non-formula skips (arrays, indexing, random-seeds, automation, aggregation, documentation). Correct by design.
- `no experiment.py referenced` ×6 — these lessons legitimately produce a **notes.md / notebook / analysis.md / RESULTS.md / LaTeX methodology** instead of `experiment.py`; the advisory is a literal-string heuristic. No `experiment.py` was fabricated.

---

## 11. Verification method

1. `vm.Script` syntax check on every inline `<script>`.
2. jsdom 29.1.1 headless load (no thrown error) + shell-agnostic structural assertions.
3. For P0 labs: dispatch `input`/`click`, assert the `.vlab` SVG content re-renders and readouts respond; hand-verified the day-04 bracketing logic and day-05 crossing-year math in Node.
4. `lesson_audit.py` per file and per module.
5. `git diff` audit: confirmed only additive callouts + CSS + new IIFE scripts; no `data-quest-id`/`data-demo`/`DEMOS`/`QS`/`BUILD`/nav-`href`/`data-sec` line removed.

Reusable harness: `/tmp/batch1_verify.js` (`node /tmp/batch1_verify.js <lesson.html> [--p0]`).

---

## 12. Remaining risks / notes

1. **No real-browser pixel check.** All verification is headless (jsdom) + math re-derivation; per `no-real-browser-in-sandbox`, a final visual eyeball of the 4 new labs at Light/Dim themes should be a user screenshot if pixel-fidelity matters. The SVGs use the same `.vlab` classes proven in the pilot.
2. **P0 labs are teaching idealizations.** Each lab's `.vlab-note` states its simplification (single clean budget, fixed constants, tiny noise). The day-02 IsoFLOP-valley uses Chinchilla approach-3 constants; its tokens-per-parameter at the valley is scale-dependent (the lesson's own §7 gotcha already flags "20×" as not universal), so the lab teaches the *valley/bracketing*, not a fixed ratio.
3. **`m13` "Week 10 · Day 4" label** on day-01 is **consistent** across title/brand-sub/nav-group-label/kicker/finale — it is the authoritative curriculum label (folder `day-01`, arc "Week 10 Day 4"), not a migration bug. Left as-is.
4. **Concurrent edits** to `sessions/` remain possible; each file was re-read immediately before editing.

---

## 13. Recommended next rollout order

Batch 1 (math-foundation + scaling) is done. Per the pilot's rollout order, proceed:

2. **Transformer & training modules** — `m02`–`m05`, `m08` (remaining days), `m09*`, `m12`.
3. **Inference systems** — `m16a`, `m17a/b`, `m06`.
4. **Frontier model-literacy** — `m14*`, `m15*`, `m19`–`m21`.
5. **Research-engineering & portfolio** — `m23`–`m29`.

Reuse this batch's mechanics: the `.vlab` CSS block + drawing-IIFE pattern for any new P0 lab, and `/tmp/batch1_verify.js` + `lesson_audit.py <path>` as the per-file gate. Keep bilingual light (3 touches), Math Ladder only for genuine formula lessons, and add Staff Lens/interview only where missing.

---

## 14. Definition of done — Batch 1

- [x] Visual audit run (3 module agents, rubric-scored) before any edit; P0 set identified (4 lessons).
- [x] `batch_1_rollout_plan.md` written **before** any lesson was modified.
- [x] All 16 non-pilot lessons: missing Coach elements added; 2 pilots untouched.
- [x] 4 P0 interactive labs built (dependency-free, offline-safe, verified interactive).
- [x] Navigation, quest-ids, quiz, playground, BUILD, tooltips, shell structure preserved (diff-verified additive).
- [x] Bilingual scaffold present but light (3 touches/lesson, intuition only).
- [x] Math Ladder for every formula-heavy lesson; skipped (documented) for the 6 non-formula lessons.
- [x] No motivational fluff; technical depth preserved or increased.
- [x] `lesson_audit.py`: 18/18 OK, 0 DEGRADED. jsdom: 18/18 clean.
- [x] `batch_1_rollout_report.md` written (this file).
