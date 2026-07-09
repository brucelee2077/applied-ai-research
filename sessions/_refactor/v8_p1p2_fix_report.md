# m02 P1/P2 Fix Report

**Loop:** `m02_p1p2_fixes`
**Date:** 2026-07-09
**Trigger:** User approved fixing the full m02 backlog ("Both — everything fixable"), excluding the curriculum-wide `BL-P1-legacy` (won't-fix-per-module).
**Method:** recon workflow (blueprint) → build 3 live viz → fix each day's defects via the v8 source-first pipeline → adversarial verification workflow → fix confirmed issues.
**Commits:** `447e711` (viz + D1), `31f0a81` (D7 table + nits + review + gate), `94e2ac4` (jargon ladders), `f6d9475` (verification fixes). Plus the D2 commit.

---

## TL;DR

- **Every learner-facing P1/P2 defect is fixed and verified.** Behavioral-viz gaps (`BL-VE-D1`, `BL-VE-D2`), the XOR capability limit (`BL-COV-XOR`), the D7 optimizer table (`BL-P1b`), the review-gate scope (`BL-P1e`), and all P2 wording/anchor nits.
- **3 new self-contained live visualizations** (local D3, auto-resize) built and jsdom/node-verified: `neuron-boundary`, `activation-derivatives`, `xor-limit`.
- **Reader-flow pass complete:** front-loaded Jargon Ladders on **all 9 days**; D2 frontier moved s1→s4.
- **Adversarial verification (13 agents)** found 6 real issues — all fixed and re-verified (including a genuine XOR-line geometry bug).
- Every change compiled through the v8 pipeline (shell byte-identical to donors), `lesson_audit` **9 OK**, `nav` **PASS**, `staff_lens` **9/9**, deterministic.

---

## 1. Fixes by backlog item

| ID | Fix | Where | Verified |
|---|---|---|---|
| **BL-VE-D1** | Live decision-boundary viz (`w` rotates, `b` shifts, `b=0` pins to origin) grounding the "bias shifts the boundary" interview line | D1 §4 + `viz/neuron-boundary.html` | jsdom render (945 shapes), 200k-param fuzz by verifier: 0 failures |
| **BL-VE-D2** | Live activation + derivative viz showing slope flatten at tails (saturation / dead-ReLU) | D2 §4 + `viz/activation-derivatives.html` | jsdom render; FUNCS math verified (sigmoid′max .25, tanh′ 1, ReLU′ step) |
| **BL-COV-XOR** | "One line can't do XOR" section (2×2 truth table, linear-separability) + live viz (single line stuck at 3/4; hidden layer fences the band) | D2 §4 + `viz/xor-limit.html` | verifier brute-forced grid: max 3/4 with one line, 4/4 impossible (proven) |
| **BL-P1b** | SGD / Momentum / Adam comparison table (update rule · state per weight · speed vs cost) | D7 §4 | verifier confirmed rows correct |
| **BL-P1e** | Review gate scope made honest (D4–6 core mechanics; D7–9 in their own lessons) — hero, verdict, **and** quiz header | `review.html` | verifier CLEAN after quiz-header fix |
| **BL-P2-D6** | "labeled columns" acceptance bullet (accurate: prints step/w/loss), redundant bullet merged | D6 §7 | verifier CLEAN after reword |
| **BL-P2-D7** | Produce stub → "open lesson.html, section 7" | D7 §7 | landed |
| **BL-P1c / BL-P2-D8** | 4-step diverge preview vs 8-step Produce reconciled explicitly; diverge rows to true 3dp; `toosmall` 13.589→13.59 | D8 §3 | 3dp values computed in Python, applied exactly |
| **BL-P1d / BL-P2-D9** | steps-axis (playground) → capacity-axis (Produce) bridge sentence; "step 300"→"step-200 low" | D9 §7 + gap take | verifier CLEAN |
| **BL-P2-review** | Self-check day-tags Day 1/2/3 → Day 4/5/6 | `review.html` | verifier CLEAN |
| **Reader-flow (all days)** | Front-loaded Jargon Ladder on every day (D1/D2 had one; added D3–D9); D2 frontier callout moved s1→s4 | s1 of all 9 | all 9 compile + audit |

`BL-P1a` (perceptron/XOR historical hook) is subsumed by `BL-COV-XOR`.

---

## 2. New live visualizations (`sessions/viz/`)

All three: self-contained, load D3 from the local `./d3.v7.min.js` (no CDN JS), carry the standard `{type:'viz-height'}` auto-resize sender + a graceful D3-missing fallback, and use the `.build-embed` iframe the lessons' existing receiver script already drives.

| File | Teaches | Embedded in |
|---|---|---|
| `neuron-boundary.html` | a neuron's decision boundary: `w` orients, `b` shifts, `b=0` → through origin | D1 §4 |
| `activation-derivatives.html` | sigmoid/tanh/ReLU + their slopes; where the slope→0 (saturation, dead ReLU) | D2 §4 |
| `xor-limit.html` | XOR is not linearly separable (one line ≤ 3/4); a hidden layer fences the band | D2 §4 |

A reusable `%%% viz` typed block was added to the compiler (`v8lib.render_viz`).

---

## 3. Adversarial verification (13 read-only agents) → 6 issues, all fixed

7/9 days CLEAN on the first pass (D1, D3, D4, D5, D7, D9 + D2). The verify pass caught:

1. **`xor-limit.html` (real bug):** the 2nd hidden-layer line was drawn at `x1+x2=2.0` (endpoints both summed to 2.0) instead of `1.5` — it passed through (1,1) and didn't fence the band. **Fixed** → `(0,1.5)-(1.5,0)`.
2. **`activation-derivatives.html`:** ReLU's solid curve wasn't clamped to the y-domain → left the canvas. **Fixed** (clamp `pathF`).
3. **`neuron-boundary.html`:** intro said "fires when z > 0"; code uses `≥`. **Fixed** wording.
4. **D6:** "labeled stages" bullet over-claimed all four loop steps are printed (code prints step/w/loss) + was redundant. **Fixed** (accurate reword + merge).
5. **D8:** `round(,3)` code label disagreed with 2dp output rows; `toosmall` showed 13.589. **Fixed** (true 3dp rows; 13.59).
6. **`review.html`:** residual quiz-header "across the whole module" overclaim. **Fixed**.

All re-verified after fixing (viz jsdom-render clean, D6/D8 recompile+gates pass, audits green).

---

## 4. Tooling change

The **Shell Invariant Gate** now masks `DEMOS/BUILD/QS` data when comparing the JS engine to the donor. Editing playground/quiz data (D8/D9) previously false-failed the "script #1 byte-identical" check because that data lives inside the main `<script>`; the gate now verifies the *engine* is unchanged while allowing data edits.

---

## 5. What remains (not a defect)

- **Verbatim → clean typed-block source conversion of D3–D9.** These days were *sourced* in verbatim/migration mode (Phase D) and their **defects are now fixed in place** via targeted region edits. Converting their `source.md` from verbatim regions to fully clean typed blocks (like D1/D2) is a **representation** change with **zero learner-visible effect** (the compiled HTML is unchanged in content), so it is deferred as ongoing v8 authoring — not a P1/P2 defect. Verbatim mode remains byte-safe and gated.
- **D4/D6/D8 s1 frontier callouts:** left in place — they sit at the *end* of s1 (after the picture + definitions = post-orientation), so they already satisfy "frontier after orientation." (D2's was mid-s1 and was moved.)
- **`BL-P1-legacy`** (dead `/frontier-experiment-lab`): curriculum-wide, won't-fix-per-module (unchanged).

**No open P0. No open learner-facing P1/P2.** m02 remains `pass_with_p1` (the residual P1 is the curriculum-wide legacy path); m03 untouched.
