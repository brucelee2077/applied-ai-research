# QA Report — Module 2 "The Neuron & How It Learns" (v7.2 Gate Audit)

> **Target:** `sessions/m02-the-neuron` (9 days + 2 gates: `review-part-a.html`, `review.html`)
> **Skill:** `/frontier-refactor-qa` (installed skills verified **byte-identical** to
> `frontier_lab_refactor_skills_v7_2/skills/*` — this audit IS a v7.2 run).
> **Mode:** independent, adversarial, **report-only**. Old notebooks not read (none in-module).
> **Method:** dynamic workflow — 1 read-only subagent per lesson (D1–D9) + 1 gate-file agent,
> integrated + deduped + re-verified by the main thread with direct greps and the audit scripts.
> **Ground truth:** `m02_coverage_contract.md`, `m02_visual_contract.md`, `m02_artifact_contract.md`.
> **Prior state:** `m02_v7_1_p0_fix_report.md` claimed "Pass with P1" (4 P0s fixed). This pass
> **re-derives** the verdict from scratch rather than trusting it.

---

## Scope & method

| Surface | Agent | Gates graded |
|---|---|---|
| D1 single-neuron | per-lesson | Coverage · Coach Voice · **Foundation Framing** · Visual/Evidence · Artifact · Anchor |
| D2 activations | per-lesson | Coverage · Coach Voice · **Function Family** · Visual/Evidence · Artifact · Anchor |
| D3–D9 | per-lesson (×7) | Coverage · Coach Voice · Visual/Evidence · Artifact · Anchor |
| review-part-a + review.html | gate agent | Module-label identity · gate integrity · quiz coverage |

Each finding was ranked with `file:line` evidence. The integrator then **independently re-verified**
the highest-risk claims (module-label leaks, D6 `0.746`, D9 `step 200`, D7 momentum `−9.6`/`−24.0`,
D5 `~3×`, D2 `tanh`, D1 `np.maximum`) by direct grep, and ran all available tooling.

---

## Coverage traceability table (29 must-cover concepts)

Result key: **PASS** = lands at stated depth · **WAIVED** = intentionally reduced, contract-sanctioned.

| # | Concept | Depth | Lesson | Visual | Artifact | Result |
|---|---|---|---|---|---|---|
| 1 | Neuron as affine map + `z=w·x+b` | Core | D1 | build+playground | `neuron()` prints w·x,+b | ✅ PASS |
| 2 | Weights/bias are learned state; `w` orients, `b` shifts | Core | D1 | build (weights label) | random init discussed | ✅ PASS |
| 3 | Bias not optional (no `b` → boundary through origin) | Support | D1 | callout | b changes output (0.0 vs 5.0) | ✅ PASS |
| 4 | Symmetry-breaking initialization | Core (staff) | D1 | callout | explain-back + quiz Q4 | ✅ PASS |
| 5 | Nonlinearity required: `linear∘linear` collapses (scalar → matrix) | Core | D2 | build collapse→cure | `(xW1)W2==x(W1W2)` 1e-10 | ✅ PASS |
| 6 | ReLU/sigmoid/tanh — formulas, ranges, ReLU default | Core | D2 | curves + tanh SVG + table | grid over `linspace(-6,6,13)` | ✅ PASS |
| 7 | Saturation + dead-ReLU counter-failure | Core (staff) | D2 | flat-tail + callout | sigmoid'(0)≈0.25, ReLU grad 0 | ✅ PASS |
| 8 | Layer `Z=xW+b` with explicit shapes | Core | D3 | shape chain build | shapes printed each stage | ✅ PASS |
| 9 | Forward pass composing layers → ŷ; batch; one numeric pass | Core | D3 | build+playground | batched (4,·); row0==single | ✅ PASS |
| 10 | Logits / raw scores | Support | D3→D4 | callout | — | ✅ PASS |
| 11 | MSE — formula, why squared | Core | D4 | pred→err→sq→mean build | MSE=0 at match | ✅ PASS |
| 12 | MSE convex bowl in ŷ | Core | D4 | falling-loss curve | — | 🟡 **WAIVED→P2** (curve present; "bowl-in-ŷ" not drawn — contract-waived) |
| 13 | Cross-entropy (+softmax/sigmoid) & why not MSE | Core | D4 | MSE↔CE table | CE confident≫hesitant | ✅ PASS |
| 14 | MSE vs CE comparison **table** | Core | D4 | real `<table>` L413-421 | — | ✅ PASS |
| 15 | Loss is a **mean** over the batch | Support | D4 | — | mean over N shown | ✅ PASS |
| 16 | Loss as scalar `L(θ)` — the surface | Core | D4→D5 | landscape callout | — | ✅ PASS |
| 17 | Gradient = partials uphill; `−∇L` descends | Core | D5 | slope-on-bowl build | finite-diff ≈ analytic | ✅ PASS |
| 18 | **Vanilla GD named** baseline (`w←w−η·dL/dw`) | Core | D5 | roll-downhill playground | loop drives w→min | ✅ PASS |
| 19 | Backprop = chain rule backward, **real numeric chain** | Core | D5 | backward-reveal build | chain pieces printed | ✅ PASS |
| 20 | Forward-cache → train mem > inference | Support | D5 | callout | explain-back | ✅ PASS |
| 21 | **Non-convexity / local vs global min** | Core (aware) | D5 | bumpy-hill note on build | — | ✅ PASS |
| 22 | Training loop: fwd→loss→bwd→**zero_grad**→update | Core | D6 | loop cycle + falling loss | labeled stages, loss↓ | ✅ PASS |
| 23 | Epoch / iteration / batch (worked 100/25=4) | Support | D6 | batch boxes | — | ✅ PASS |
| 24 | Batch vs mini-batch vs SGD — noisy unbiased estimate | Core | D6 | playground | noise | ✅ PASS |
| 25 | Random-init scale (fan-in) | Support (staff) | D6 | callout | — | ✅ PASS |
| 26 | SGD update `θ←θ−η∇L` stated exactly | Core | D7 | optimizer paths | one-step w=1→1.8 | ✅ PASS |
| 27 | Momentum rule (velocity) + what it fixes | Core | D7 | ball/momentum build | real g=−9.6, v=−24.0 | ✅ PASS |
| 28 | Adam rule (m,v, bias-corr, β₁=.9/β₂=.999/ε=1e-8) | Core | D7 | Adam build step | bias-corrected step | ✅ PASS |
| 29 | LR as step size — small/right/large(→NaN) | Core | D8 | LR slider + curve | 0.005 crawl/0.1 conv/0.6 diverge | ✅ PASS |
| 30 | LR schedules + warmup | Support | D8 | warmup+decay shape | — | ✅ PASS |
| 31 | Train / validation / test roles | Core | D9 | 3-box split build | disjoint 18/6/6, sizes sum | ✅ PASS |
| 32 | Overfitting/underfitting via gap `= val−train` | Core | D9 | two-curve diverge | train↓ while val↑ | ✅ PASS |
| 33 | Early stopping + regularization (L2/dropout) | Core (aware) | D9 | early-stop marker | best-val ≠ last | ✅ PASS |

**Coverage result: 32/33 rows PASS · 0 FAIL · 1 WAIVED (row 12, D4 convex-bowl-in-ŷ → P2, contract-sanctioned).**
Every **must-cover** concept lands at its stated depth or is explicitly waived. The 4 promoted items are
visibly owned: **init** (D1 symmetry-break + D6 fan-in), **named GD** (D5), **non-convexity** (D5),
**logits/log-sum-exp** (D4 🔬 callout).

> Note: **Perceptron/XOR** and **Leaky-ReLU/GELU one-liners** are in the contract's **"Should cover"**
> tier, not "Must cover." Leaky-ReLU/GELU/SwiGLU **are** present (aware). Perceptron/XOR is **absent** —
> a *Should-cover* gap → **P1**, not a coverage FAIL (missing a *must-cover* is the P0 trigger; missing a
> should-cover is not).

---

## Gate results

### Coach Voice Gate — ✅ PASS (9/9)
Every lesson carries the full Coach Layer: `.term` Jargon tooltips, a Math Ladder (words → formula →
worked numbers), **one silent-failure callout + one trade-off callout**, a 🎤 interview answer, and a
diagnostic quiz Q (`q:4 o:16` on every day). No dismissive phrases ("obviously / trivially / as you can
see / recall that") found. Bilingual is light-touch (English-primary), consistent with the user preference.

### Foundation Framing Gate (D1) — ✅ PASS
All five required elements present: (1) brain-inspired intuition (dendrites→axon→firing), (2) artificial-
neuron caveat, (3) **biological→artificial mapping table** (5 rows), (4) where-the-analogy-breaks (no
biological backprop; spikes over time; weights learned by gradient descent), (5) transition to
`output = activation(w·x + b)`. The everyday "judge" analogy is explicitly demoted to *optional/secondary*
— no generic analogy is substituted for the required foundation frame. No formula-first opening.

### Function Family Gate (D2) — ✅ PASS
Present: **ReLU, sigmoid, tanh** (full formula + range + comparison table), **Leaky ReLU** (aware),
**GELU** (aware, GPT context), **SwiGLU** (aware, Llama context), **softmax** (aware; correctly deferred to
D4 as an output-layer function). Derivative intuition ("slope near 0" table row + sigmoid'(0)≈0.25),
saturation, vanishing gradient, and dying ReLU all covered. **Scalar collapse is shown before matrix
collapse** in the Math Ladder. No core family member missing.

### Visual/Evidence Gate — ✅ PASS (9/9)
Every lesson's P0 visual exists and shows a **changing quantity** (loss falling, curve bending, step
overshooting). The three `viz/gradient-descent.html` embeds (D5/D7/D8) carry the `viz-height` postMessage
sender + matching parent receiver → they auto-size (no stuck 520px). The two v7.1 fix targets are closed:
**D3** now shows a concrete numeric forward pass (`[2,3]→[2,−3]→[2,0]→[2]`, not shapes-only); **D5** shows a
numeric multi-link chain (`dL/da·da/dz·dz/dw` with verified numbers). SVGs use the accepted light
`.build-viz` canvas convention consistently.

### Anchor Consistency Gate — ✅ NO P0
Per-day playground numbers == Math-Ladder numbers == Produce acceptance numbers on every day. Independently
re-verified: D1 `−2/−1/0` + `np.maximum`; D6 `0.746`; D7 first-step `w=1.8`, step-2 `g=−9.6`/`v=−24.0`
(phantom `−12.8` absent); D9 kept-model `step 200`; D5 `~3×`. Two residual mismatches are **P1** (they teach
**no conflicting model**): D8 playground `5/5/4` vs Produce `8/8/8` (narrated at `day-08:390`); D9 playground
framing (100 pts / steps-axis) vs Produce (30 pts / degree-axis), **bridged** by the capacity callout at
`day-09:427`.

### Artifact Gate — ✅ PASS (9/9)
Every Produce Option-A prompt + acceptance criteria match the artifact contract's exact numbers; every run
command path == the actual folder; every `experiment.py` stub's commented run path matches its folder; each
lesson carries the 📓 5-minute-log callout with the explain-back question. `experiment.py` files are
intentionally guided stubs (contract §31), not filled solutions.

### Module-label identity (defect class #1) — ✅ CLOSED
`grep` finds **zero** learner-facing self-labels reading "Module 3 / Module 2-3 / Module 4". `M40` in
D6/D7 is an SVG path coordinate (`M400 …`). All `review.html` corrections are in place: "Ready for Module 3?
(Attention)", "(M1–M2)", "M3 · Attention"; review-part-a's forward copy reads "training half of Module 2".
Forward-pointers to Module 3 (Attention) are correct and retained.

---

## Findings

### P0 — **NONE (count = 0)**
No missing must-cover concept, no formula-first opening, no missing P0 visual, no failure mode without
evidence, no simulated-output-as-evidence, no conflicting-model anchor break, no stale run command, no
Produce path mismatch, no broken JS/nav/completion, no module-label self-leak.

### P1 (do not block merge; **out of Phase-1 "fix P0 only" scope**)
| ID | Lesson | Finding | Fix shape (no frozen invariants) |
|---|---|---|---|
| P1-a | D2 | Perceptron/XOR aware historical hook absent (Should-cover) | +1 prose sentence in §1/§4 |
| P1-b | D7 | No SGD/momentum/Adam comparison **table** (S8 general rule; only D4/D6 tables are contract-mandated) | add 3-row table in §4 |
| P1-c | D8 | Playground step counts `5/5/4` vs Produce `8/8/8` — narrated, no conflicting model | extend playground rows to 8 |
| P1-d | D9 | Playground framing (steps-axis) ≠ Produce framing (degree-axis) — bridged by capacity callout, no conflicting model | +1 sentence unifying the two axes |
| P1-e | review.html | End-gate quiz + self-check cover D4–D6 only; copy claims "whole module" (`QS` is frozen → soften copy) | reword L171/L189 |
| P1-legacy | all 9 days | `/frontier-experiment-lab` (uninstalled) in every Produce Option-B + stub | **Contract-sanctioned, curriculum-wide (~243 files) — do NOT fix per-module** |

### P2 (polish; do not block)
- **D1** — determinism criterion ("same input twice") not surfaced in Option A/B; disclosed 2→3 width in layer SVG (contract-intended).
- **D2** — Option A softens the "cure" (relu insertion changes output) to prose vs Option B's numeric diff.
- **D3** — playground uses unseeded `np.random.rand` (prints shapes only; benign).
- **D4** — MSE=0-at-match not printed in playground; bowl-in-ŷ (waived); confident-wrong is prose.
- **D6** — Produce Option-A acceptance omits the "prints labeled stages" bullet.
- **D7** — stub comment "open lesson.html" vs contract "open lesson.html, section 7".
- **D8** — playground `toobig` rounds to 2dp vs 3dp elsewhere.
- **D9** — `gap` take says "around step 300 the validation loss turns around" while the minimum is step 200 (`day-09:623`); kept-model anchor is correctly 200 everywhere else.
- **review.html** — stale self-check day-tags "Day 1/2/3" should read "Day 4/5/6" (`review.html:265-267`); mid-module gate `wf2-review` is not registered as a hub tile (frozen MODULES array — architect decision).

---

## Fixes applied
**None.** Phase 1 scope is "fix P0 only"; P0 count = 0, so no lesson file was edited. No `data-quest-id`,
navigation, localStorage, quiz `QS`, `BUILD`/`DEMOS` array, completion gating, or shared shell file was
touched by this audit (read-only).

---

## Verification run

| Check | Result |
|---|---|
| `python3 sessions/lesson_audit.py m02-the-neuron` | **9 OK / 0 MISSING / 0 LEFTOVER / 0 DEGRADED** |
| `python3 sessions/nav_audit.py` | **0 BROKEN links** — PASS, all pages wired |
| `node sessions/staff_lens_audit.js m02` | **9/9 staff-lens present, gap 0, `errs:[]`, `q:4 o:16`**. `render:BROKEN`/`secs:0` = known-benign `.sec` vs `.module-section` selector mismatch (not a regression) |
| JS-syntax `node --check` (extracted script blocks, 9 lessons + 2 gates) | **ALL 11 PASS** |
| Module-label leak grep | **0 learner-facing self-leaks** (`M40` = SVG coordinate) |
| Anchor re-verify grep (0.746 / step 200 / −9.6 / −24.0 / no −12.8 / ~3× / tanh / np.maximum) | **all confirmed** |
| jsdom render check | not run — jsdom not installed (MODULE_NOT_FOUND); substituted node syntax + structural checks |

---

## Merge recommendation

### ✅ Pass with P1

Module 2 is **clean at the P0 bar** under an independent v7.2 audit: **0 P0** across all 9 lessons and both
gate files, all 33 must-cover rows PASS (1 contract-waived to P2), all six gates pass (Coverage, Coach
Voice, Foundation Framing, Function Family, Visual/Evidence, Artifact) and Anchor Consistency has no P0, all
five defect classes are closed, and every audit script passes. Remaining items are **P1/P2 only** — the
S8-table gaps (D7), the narrated/bridged framing mismatches (D8, D9), the missing XOR aware hook (D2), the
review-gate coverage overclaim, and the curriculum-wide `/frontier-experiment-lab` legacy reference — none
of which teach a conflicting model or break the module.

**Module 2 status: ✅ Pass with P1.** Phase-1 gate satisfied (0 P0).
