# QA Report — Module 2 "The Neuron & How It Learns" (v7 Gate Pass)

> **Target:** `sessions/m02-the-neuron` (9 days + 2 gates)
> **Mode:** v7 QA, xhigh effort, **report-only** (no lesson/contract files were edited).
> **Method:** 22-agent workflow — per-day comprehensive gate check (coverage / coach-voice /
> visual-evidence / artifact / math) → adversarial re-read verify, plus 4 cross-cutting agents
> (review pages, legacy-skill sweep, JS/nav/completion, cross-day numeric consistency). Every finding
> below survived an adversarial re-read of the cited line(s); severities are **post-verify**.
> **Contracts used as ground truth:** `m02_coverage_contract.md`, `m02_artifact_contract.md`,
> `m02_visual_contract.md`, `m02_first_principles_blueprint.md`.

---

## Scope

- **In scope:** the 9 current `day-*/lesson.html` files, their `experiment.py` stubs, and
  `review.html` + `review-part-a.html`, checked against the four existing contracts under the v7 gates.
- **Explicitly excluded (per instruction):** old notebooks (none exist inside the module folder; none
  were read), other modules, and any file edits. This is a gap report, not a refactor.
- **Headline verdict:** 🎯 the interactive lesson bodies are notebook-grade — coach voice, the Coach
  6-pack, math correctness, and all JS/nav/completion behavior pass cleanly. **The v7 blockers are
  concentrated in the Produce/artifact layer plus one genuine teaching hole (D2 `tanh`).**

### Defect-class scorecard (the 5 fixes the refactor claimed to make)

| # | Defect class (from `m02_coverage_contract.md:104-117`) | Status |
|---|---|---|
| 1 | Module-label identity split (Days 4–9 / review pages said "Module 3") | ✅ **Closed** — all labels read "Module 2"; `review.html` correctly says "Ready for Module 3? (Attention)", "(M1–M2)", "M3 · Attention" (verified line-by-line). ⚠️ Residue: `review.html` self-check still tags Days **1/2/3** for Days 4/5/6 content (P2). |
| 2 | Numeric self-consistency (D6 `0.747→0.746`, D9 best model `step 200`, D5 `~3×`) | ✅ **Closed** — D6 reads `0.746`, D9 pins the val minimum at step 200, D5 states `~3×` consistently. |
| 3 | Playground ↔ Produce agreement (D7 momentum, D8 counts, D9 framing) | ⚠️ **Partially open** — D7 momentum `g=−9.6, v=−24.0` ✅ closed; **D8 step-count still open (P0)**; D9 framing still differs (P2) + the deeper D9 no-test-set break (P0). |
| 4 | Worked-example spine (D2 one anchor matrix; D1 `np.maximum`) | ✅ **Closed** — D2 uses the single `W1@W2=[[7,2],[3,1]]` anchor everywhere; D1 uses `np.maximum` in both surfaces. |
| 5 | No premature close (D6 "Foundations done / completes the story") | ✅ **Closed** — the premature-close line is gone from D6. |

---

## Coverage traceability table

Must-cover concepts (`m02_coverage_contract.md` "Must cover" + the 4 promoted items), each traced to its
named lesson. **PASS** = lands at stated depth with required visual/artifact; **PARTIAL** = taught but a
required surface (usually the artifact) is missing; **FAIL** = absent at required depth.

| Concept | Req. depth | Lesson evidence | Visual/evidence | Artifact | Result |
|---|---|---|---|---|---|
| Neuron as affine map `z=w·x+b` | Core | D1 `lesson.html:348-353,402` | Playground 3-step + Build term-by-term (`:589-627`) | `neuron()` Option A `:451` | ✅ PASS |
| Weights/bias = learned state; geometry | Core | D1 `:412,417,404` | Build labels weights `:619` | random-init framing | ✅ PASS |
| Bias not optional (origin) | Support | D1 `:404` callout | Build step 4 `:623` | criterion | ✅ PASS |
| Symmetry-breaking init | Core (staff) | D1 `:417,419` + quiz `:671` | callout | explain-back | ✅ PASS |
| Nonlinearity required; `linear∘linear` collapses (scalar-first) | Core | D2 `:401-402` (scalar then matrix) | Build collapse→cure `:613-616` | Produce collapse `:445` | ✅ PASS |
| **ReLU / sigmoid / tanh — formulas, ranges** | Core | D2 `:395` shows **only ReLU + sigmoid**; `tanh` absent everywhere | no tanh curve/demo | Produce omits tanh | ❌ **FAIL** (D2-F1) |
| Saturation + dead-ReLU | Core (staff) | D2 `:411` + quiz `:664` | build note `:612` | log item `:464` | ✅ PASS |
| Layer `Z=xW+b` with shapes | Core | D3 `:345,401-406` | Build shape chain `:617-623` | Produce `:449` | ✅ PASS |
| Forward pass → `ŷ`; batch; one numeric pass | Core | D3 numeric callout `:407`; batch `:416` | Build + playground `:588-603` | batch-of-4 `:449,461` | ✅ PASS |
| Logits / raw scores | Support | D3 `:398` term + `:597` | tooltip + build note | n/a | ✅ PASS |
| MSE — formula, why squared, **convex bowl in ŷ** | Core | D4 `:401-407`; **no convex-bowl curve** (only loss-vs-steps `:642`) | falling-loss curve only | Produce `:465` | 🟡 PARTIAL (D4-F3) |
| Cross-entropy + softmax/sigmoid & why-not-MSE | Core | D4 `:401,418,432` | playground CE + drop `:609-619` | acceptance `:482` | ✅ PASS |
| MSE vs CE comparison **table** | Core | D4 real `<table>` `:413-421` | table element | n/a | ✅ PASS |
| Loss = MEAN over batch | Support | D4 `:405,407` | build step 4 `:638` | `np.mean` `:465` | ✅ PASS |
| Loss as scalar `L(θ)` — surface | Core | D4 `:425` landscape handoff | build step 6 `:642` | n/a | ✅ PASS |
| Gradient uphill; `−∇L` descends | Core | D5 `:345,398-402` | Build slope + live embed `:431,625` | GD loop `:455` | ✅ PASS |
| **Vanilla GD named** baseline | Core | D5 `:414` ("…is vanilla gradient descent") | build step 3 `:627` | loop `:455` | ✅ PASS |
| **Backprop = real numeric multi-link chain** | Core | D5 Math Ladder `:406-410` (correct) | Build backward-reveal `:631` | **Produce omits part (b)** | 🟡 PARTIAL (D5-F1) |
| Forward-cache → train mem > inference | Support | D5 `:405,420` | callout | explain-back | ✅ PASS |
| Non-convexity / local vs global | Core (aware) | D5 `:415,432` | bumpy-hill build note | n/a | ✅ PASS |
| Training loop w/ `zero_grad`, over epochs/batches | Core | D6 `:346-351,401` (`reset_grads()`) | loop-cycle SVG + falling loss `:633-637` | Produce `:459` | ✅ PASS |
| Epoch/iteration/batch (worked count) | Support | D6 `:411` (`100/25=4`) | batch boxes `:639` | n/a | ✅ PASS |
| Batch/mini-batch/SGD = noisy unbiased | Core | D6 `:411` + batch table `:421-426` | trade-off table | n/a | ✅ PASS |
| Random-init scale (fan-in) | Support (staff) | D6 `:402` (Xavier/He) | callout | n/a | ✅ PASS |
| SGD rule `θ←θ−η∇L` | Core | D7 `:405,420` | playground `:604-611`, build `:643` | Option A `:465` | ✅ PASS |
| Momentum rule + what it fixes | Core | D7 `:410,412,648` | playground `:612-620`, build `:645-648` | loop `:465` | ✅ PASS |
| Adam (m,v, bias-corr, per-param; β/ε defaults) | Core | D7 `:416-417` (β₁.9/β₂.999/ε1e-8) | playground `:621-627`, build `:649-652` | note `:477` | ✅ PASS |
| LR regimes: small / right / large(→NaN) | Core | D8 `:345,362-367,402-404,411-415` | playground 3 rates + build + live embed | Produce `:463` | ✅ PASS |
| LR schedules + warmup | Support | D8 `:417,420-421` | warmup/decay SVGs `:643-646` | n/a | ✅ PASS |
| Train / val / test roles | Core | D9 `:347-349,366-370` | 3-box build `:642`; playground `(70,15,15)` | **Produce drops the test split** | 🟡 PARTIAL (D9-F1) |
| Overfit/underfit via train–val gap | Core | D9 `:404,413-419` | two-curve diverge build `:648` | train/val print `:466` | ✅ PASS |
| Early stopping + L2/dropout (gap-closers) | Core (aware) | D9 `:426-427` | best-val marker `:650` | criterion `:479` | ✅ PASS |

**Result:** 27/32 must-cover rows PASS, **1 FAIL** (D2 `tanh`), **3 PARTIAL** (D4 convex-bowl visual, D5
backprop artifact, D9 test-split artifact). Should-cover gaps: D2 perceptron/XOR (P1), D4 MLE + MAE/Huber
(P2). No out-of-scope material is taught (forward-pointers only). ⚠️ **The table does not pass** —
`tanh` (Core must-cover) is absent, so "all concepts landed" cannot be signed off.

---

## Coach Voice Gate

**PASS on all 9 days.** Every lesson opens with a why-care hook and a named pain point *before* any
formula (first-20% rule holds — formulas live in section 4 on every day), carries a deep multi-sentence
analogy with an explicit "where it breaks" line, and D1 (the neural-network foundation lesson) has the
required brain/human analogy + caveat (the "judge scoring a contestant", `day-01:365-372`). The **Coach
6-pack is complete on all 9 days**: Jargon tooltips, a Math Ladder, one silent-failure + one trade-off
callout, an interview-grade answer, and a diagnostic quiz question — each verified per day. No
duplicated callouts (the m05a-style double-paste defect is absent here).

Minor coach nits (not blockers):
- **P1 — D6** never explicitly reframes the "training is a run-once job that terminates" misconception
  (`blueprint:204`); the loop's *repetition* is taught but "no natural exit / you decide when to stop /
  stochastic" is never named. (D6-F2)
- **P2 — D6** the free-throw analogy's stated breakdown covers overfitting but omits the
  blueprint's "a net has no innate 'done'; it can diverge/oscillate" point. (D6-F1)

---

## Visual/Evidence Gate

**PASS on required P0 visuals.** Every day's P0 visual exists and shows a *changing quantity*; every
**simulated terminal is explicitly labelled** ("a simulated Python terminal", verified on all 9 days);
quantitative panels carry numeric readouts that match the Math Ladder; the D5/D7/D8
`viz/gradient-descent.html` embeds have both the receiver and the `viz-height` sender wired
(`gradient-descent.html:540`) — no iframe sticks at 520px.

Gaps:
- **P1 — D3** the playground prints *shapes only* at each stage (`day-03:592,596`); the visual contract
  asks it to print shape **and** one numeric value. The P0 "concrete numeric forward pass" requirement
  is still satisfied (the S4 math callout `:407`), so this is a surface-quality gap, not a P0. (D3-F2)
- **P2 — D4** the "convex bowl in `ŷ`" sub-requirement of the MSE row is never drawn or stated; the only
  curve is loss-vs-training-steps. (D4-F3)
- **P2 — D7** the Adam playground shows the un-bias-corrected step `m/(√v+ε)` while S4 teaches the
  bias-corrected `m̂/v̂` — cosmetic panel mismatch; coverage is satisfied by the S4 callout. (D7-F3)
- **P2 — D9** the playground take says overfitting turns "around step 300" while the early-stop point is
  pinned at step 200 (numerically consistent; narrative looseness only). (D9-F5)

---

## Artifact Gate

**This is where the v7 blockers live.** Paths and run commands are correct on every day
(`python3 sessions/m02-the-neuron/<folder>/experiment.py`), and every day has the 📓 5-minute-log
callout. But several Produce steps + acceptance criteria diverge from the artifact contract — including
two that leave a **must-cover concept with no runnable evidence** and one **anchor-spine break** the
refactor claimed to have closed:

- **D2 (P0):** Produce omits `tanh`, the `np.linspace(-6,6,13)` grid, `sigmoid'(0)≈0.25`, and the
  "ReLU grad is 0 for negatives" note the contract mandates (`day-02:445,452-462` vs
  `m02_artifact_contract.md:20`). (D2-F2)
- **D5 (P0 ×2):** Produce covers only part (a) `f(w)=w²` GD; the entire contract part (b) — derive
  `dL/dw, dL/db` on `(sigmoid(w·x+b)−y)²`, print the chain pieces, verify vs finite differences to
  `<1e-6` — is absent (`day-05:454-474`). The day's headline concept (backprop) has **no runnable
  artifact evidence**. The 📓 explain-back also drops the contract's finite-difference question. (D5-F1, D5-F2)
- **D8 (P0):** anchor-spine break — the playground runs **5/5/4** steps while Produce (Option A, B, and
  acceptance) mandates an **8-step** loop (`day-08:604,611,618` vs `:463,473,480`). This is the exact
  D8 fix listed in all three contracts and it is still open; the playground is also internally
  inconsistent (5 vs 4). (D8-F1 = numeric `D8-step-count-mismatch`)
- **D9 (P0):** Produce uses a **2-way train/val split with no test set** (`day-09:466,476-484`),
  contradicting the 3-split requirement and *silently re-teaching the exact "val = test" misconception
  the day exists to kill*; it fails 2 of 4 contracted acceptance criteria. (D9-F1)

Lower-severity artifact gaps: D1 acceptance omits the determinism criterion + the log question drops the
bias explain-back (P1, D1-F1/F2); D4 acceptance omits MSE==0-at-match, the clip/no-`log(0)` criterion,
and confident-wrong≫hesitant CE (P1/P2, D4-F1 + miss); D9 acceptance omits the fixed-seed criterion and
D9 Option A is internally incoherent (a single degree-9 lstsq fit cannot produce a "train-over-time"
U-curve) (P1, D9-F2 + miss); D4/D8 `experiment.py` stub comments drift from the canonical template (P2).

The `/frontier-experiment-lab` Option-B reference is present on all 9 days but is **contract-sanctioned**
(`m02_artifact_contract.md:5`) — see the P1 tracking item below.

---

## P0 findings

> Blockers. Real coverage/artifact defects; each confirmed by adversarial re-read.

| ID | Day | Gate | Defect (evidence) |
|---|---|---|---|
| **P0-1** | D2 | coverage | **`tanh` — a Core must-cover activation — is entirely absent** (no formula, range, curve, playground, or quiz). `day-02:395` shows only ReLU + sigmoid; grep for `tanh` = 0 hits. Contract `coverage_contract.md:24`, `blueprint:135`. Module-incomplete per the completeness check. |
| **P0-2** | D2 | artifact | Produce diverges from `artifact_contract.md:20`: omits `tanh`, the `linspace(-6,6,13)` grid, `sigmoid'(0)≈0.25`, and the ReLU-grad-0 note. A required concept can never be produced. `day-02:445,452-462`. |
| **P0-3** | D5 | artifact | Produce omits the **entire backprop chain-rule + finite-difference check** (contract part b). The headline "backprop" concept has no runnable evidence; the numeric chain lives only in the Math Ladder. `day-05:454-474` vs `artifact_contract.md:23`. |
| **P0-4** | D5 | artifact | The 📓 explain-back question does not match the contract and **drops the finite-difference probe** entirely (`day-05:475` vs `artifact_contract.md:23`). Consistent with P0-3. |
| **P0-5** | D8 | artifact / anchor-spine | **Playground 5/5/4 steps vs Produce 8-step loop** — the explicit D8 defect-class fix ("step count == the playground's") is still open, and the playground is internally inconsistent. `day-08:604,611,618` vs `:463,473,480`; `artifact_contract.md:26`, `visual_contract.md:28`, `coverage_contract.md:114`. |
| **P0-6** | D9 | artifact / anchor-spine | **Produce drops the test split** (2-way 20/10 train/val, no test), contradicting the 3-split requirement and silently re-teaching "val = test"; fails acceptance criteria (3) and (4). `day-09:466,476-484` vs `artifact_contract.md:27`. |

> **Note on `/frontier-experiment-lab` (raw gate = P0).** Per the strict v7 list, a reference to an
> uninstalled skill is P0. On every day the reference is the **contract-sanctioned Produce Option-B
> pattern** (`m02_artifact_contract.md:5` prescribes exactly this trigger), and per the documented
> curriculum-wide decision (uninstalled skill in ~243 files, "P1, don't fix per-module") it is **not a
> Module-2 regression**. It is therefore carried as **P1-10 below**, not as a per-day P0 blocker. This
> is the one place where the raw gate and the module's reality diverge, and it is called out
> deliberately.

---

## P1 findings

> Fix before merge if easy.

| ID | Day/Area | Defect (evidence) |
|---|---|---|
| **P1-1** | D1 | Acceptance list omits the contract's determinism criterion ("same input twice → identical output"). `day-01:466-470` vs `artifact_contract.md:19`. |
| **P1-2** | D1 | 📓 log explain-back does not carry the contract's bias question ("what does the bias let it do that weights alone cannot?"). `day-01:471`. |
| **P1-3** | D2 | Should-cover perceptron + XOR historical hook is missing (`coverage_contract.md:57`; grep = 0 hits). |
| **P1-4** | D3 | Playground prints shapes only, no per-stage numeric readout (`day-03:592,596`; `visual_contract.md:23`). |
| **P1-5** | D4 | Acceptance omits MSE==0-within-1e-12-at-match and the clip/no-`log(0)` criterion. `day-04:480-484` vs `artifact_contract.md:22`. |
| **P1-6** | D6 | "Training is run-once / terminates" misconception never explicitly reframed (`blueprint:204`; grep = 0 hits). |
| **P1-7** | D9 | Acceptance omits the fixed-seed / determinism criterion (`day-09:483-487`; `artifact_contract.md:12,27`). |
| **P1-8** | D9 | Produce **Option A is internally incoherent**: a single degree-9 least-squares fit has no training-step axis, so it cannot produce the "loss falling / val bottoms-then-rises as you train" curve its own acceptance bullet requires. `day-09:466,485`. |
| **P1-9** | review.html | The **final D1–D9 gate quiz only tests D4–D6** (loss, gradient, GD, backprop, loop, epoch) — never D1 neuron, D2 activations, D3 forward pass, D7 optimizers, D8 LR, or D9 train/val/test — despite the lead/verdict/fin all claiming a full-engine recap. A learner can pass 6/6 without being checked on half the module. `review.html:286-291` vs `:171,189,202,211`. |
| **P1-10** | all 9 days | `/frontier-experiment-lab` (uninstalled) referenced in every Produce Option-B + `experiment.py:4`. **Contract-sanctioned; curriculum-wide item — do NOT fix per-module.** Track under the existing ~243-file sweep and update `m02_artifact_contract.md:5` in lockstep when it runs. |

---

## P2 findings

> Final polish. Do not block on these.

| ID | Day/Area | Defect |
|---|---|---|
| **P2-1** | D4 | `experiment.py` stub comment drifts from the canonical template (missing "section 7"; bare `frontier-experiment-lab` vs `/frontier-experiment-lab`). `day-04 experiment.py:3-4`. |
| **P2-2** | D4 | Convex-bowl-in-`ŷ` visual not delivered (falling-loss-vs-steps curve present instead). `day-04:642`. |
| **P2-3** | D4 | Should-cover MLE view (MSE⇐Gaussian, CE⇐Bernoulli) and MAE/Huber absent (`coverage_contract.md:60-61`). |
| **P2-4** | D4 | Acceptance omits the "confident-wrong CE ≫ hesitant-wrong CE" criterion (`artifact_contract.md:22`). |
| **P2-5** | D6 | Free-throw analogy breakdown omits the "no innate done / can diverge/oscillate" point (`blueprint:230`). |
| **P2-6** | D7 | 📓 log callout asks 3 custom lines instead of the contract's verbatim explain-back; "what Adam adapts per-parameter" is not asked. `day-07:486`. |
| **P2-7** | D7 | Adam playground omits bias correction (`m/√v` vs taught `m̂/v̂`); no "simplified for brevity" note. `day-07:624`. |
| **P2-8** | D7 | Acceptance list omits 2 of 4 contract criteria (first step == `w=1.8`; both converge). `day-07:482-484`. |
| **P2-9** | D8 | `experiment.py` stub comment drifts from the canonical template (cosmetic; run path correct). |
| **P2-10** | D9 | Playground (100-pt 3-way split, step-sweep) vs Produce (30-pt 2-way, degree-sweep) framing differs; contract asks "pick one framing". `day-09:606-609,478`. |
| **P2-11** | D9 | 📓 log explain-back substitutes a data-leakage question for the contract's "what is the validation set protecting you from?". `day-09:488`. |
| **P2-12** | D9 | Playground take "around step 300" vs step-200 early-stop point — narrative looseness (numerically consistent). `day-09:622`. |
| **P2-13** | review.html | Self-check items tagged "Day 1 / 2 / 3" but their content is loss / gradients / training-loop = actual module Days **4 / 5 / 6** — residue of the internal `wf3-*` numbering. `review.html:265-267`. Do NOT touch `data-quest-id`. |

---

## What passed cleanly (recorded for faithfulness)

- **JS / nav / completion — zero defects.** All 11 pages: 7-section model + "0/7" checklist + `.gotit`
  gating, playground/build/quiz/copy/tooltip behaviors, the one continuous prev/next chain
  (M1-review → D1 → … → D3 → review-part-a → D4 → … → D9 → review → m03), the `frontier-lesson:<QID>`
  completion contract (lessons on `done.produce`, gates on `done.verdict`, hub pill reads both), the
  frozen quest-ids incl. the First-Neuron badge key, and the D5/D7/D8 viz autoresize. `node --check`
  clean on every script block; `nav_audit.py` and `lesson_audit.py m02-the-neuron` both PASS (9 OK / 0
  advisories).
- **Math correctness — PASS on all 9 days** (independently recomputed): D1 `−2/−1/0`, D2
  `W1@W2=[[7,2],[3,1]]`, D4 CE `1.204/0.511/0.105`, D5 chain `dL/dw≈−0.212` + GD trajectory, D6 step-1
  `w=1.8`, D7 `g=−9.6, v=−24.0`, D8 tipping `lr=2/c=0.25`, D9 val-min at step 200.
- **Module-label P0 defect class — fully closed** on the review pages (all forbidden "Module 3/4",
  "M1–M3", "M4 · Attention" strings gone; `review-part-a.html` clean).
- **`review-part-a.html`** passes every check (coach voice, completion, nav, coverage recap of D1–D3, no
  legacy names).
- **Legacy sweep:** 11 of the 12 flagged skill names have zero hits anywhere in the module; both review
  pages are entirely legacy-free.

---

## Fixes applied

**None.** This was a report-only v7 pass (per instruction: "Do not edit lesson files"). No lesson,
contract, or `experiment.py` file was modified. The only file written is this report,
`sessions/m02_v7_gap_report.md`.

---

## Merge recommendation

### 🚫 Blocked (on P0-1 … P0-6)

Six confirmed P0 defects remain, concentrated in the **Produce/artifact layer** plus one **coverage
hole**:

- **Coverage:** D2 is missing `tanh`, a Core must-cover activation (P0-1/P0-2).
- **Artifacts with no runnable evidence for a must-cover concept:** D5 backprop (P0-3/P0-4).
- **Anchor-spine breaks the refactor claimed to close:** D8 step counts (P0-5), D9 missing test split
  (P0-6) — the D9 one silently re-teaches the very misconception the day targets.

Everything else is in strong shape: the lesson bodies, coach voice + full 6-pack, math, all JS/nav/
completion behavior, the module-label unification, and 4 of the 5 numeric/spine defect classes are
verified clean. **Once P0-1 … P0-6 are addressed (all are localized Produce/coverage edits), the module
lands at "Pass with P1"** — the remaining P1s (D1/D4/D9 acceptance criteria, D3 numeric playground
readout, D6 misconception reframe, the review.html capstone quiz coverage) are quick, non-structural
fixes, and the `/frontier-experiment-lab` P1-10 item is deliberately out of per-module scope.

*Note: `data-quest-id` values and the frozen invariants (§0 of the blueprint) must not change during any
follow-up fix — several P0/P1 fixes touch Produce prompts and the review-page self-check, which sit
right next to those frozen keys.*
