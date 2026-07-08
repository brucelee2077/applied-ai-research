# QA Report — Module 2 "The Neuron & How It Learns" (v7.1 Gate Pass)

> **Target:** `sessions/m02-the-neuron` (9 days + 2 gates)
> **Mode:** v7.1 QA, **xhigh** effort, **report-only** — no lesson, contract, or `experiment.py`
> file was edited (per instruction). Old notebooks were not read (none exist inside the module folder).
> **Method:** independent re-read of the P0-bearing lessons (D1, D2, D5, D8, D9) against the current
> file bytes, plus targeted verification of D3/D4/D6/D7 anchors, the `experiment.py` stubs, and a full
> 12-name legacy-skill sweep. The module files are **unchanged since the v7 pass** (git shows no commit
> and no working-tree edit under `day-*/` since `4ef2714`; all `day-*` mtimes precede
> `m02_v7_gap_report.md`), so the v7 findings are the baseline this pass re-grades under the stricter
> v7.1 gates.
> **Contracts used as ground truth:** `m02_coverage_contract.md`, `m02_artifact_contract.md`,
> `m02_visual_contract.md`.

---

## What changed from v7 → v7.1

v7.1 tightens two gates. Applying them faithfully **adds one new P0 and re-grades one prior P0 down**:

| # | Item | v7 verdict | v7.1 verdict | Why it moved |
|---|---|---|---|---|
| A | **D1 Foundation Framing** (brain-vs-ANN intuition + mapping table) | PASS (judge analogy credited) | **P0 FAIL** | v7.1 **explicitly disallows** an everyday analogy (judge/chef/scoring) as the foundation frame and **requires a biological→artificial mapping table**. D1 has neither. |
| B | **D8 step-count** (playground 5/5/4 vs Produce 8) | P0-5 (anchor-spine) | **P1** | v7.1 marks anchors P0 only when they **teach conflicting mental models**. Line 390 now narrates the playground as "a short preview" and the artifact as "more steps," so no wrong model is taught; the residue (internal 5/5/4, artifact-criterion mismatch) is consistency-polish. |

Everything else from v7 re-verifies unchanged: D2 `tanh` (P0), D5 backprop artifact (P0 ×2), D9 test-split (P0).

---

## Scope

- **In scope:** the 9 `day-*/lesson.html`, their `experiment.py` stubs, `review.html` + `review-part-a.html`,
  checked against the three contracts under the seven v7.1 gates + math + JS/legacy.
- **Excluded (per instruction):** old notebooks (none in-module; none read), other modules, all file edits.
- **Headline:** 🎯 The lesson bodies remain notebook-grade (coach voice, the Coach 6-pack, math, and all
  JS/nav/completion pass). The v7.1 blockers sit in **(1) D1 foundation framing** and **(2) the
  Produce/artifact layer** — D2 `tanh`, D5 backprop, D9 test split. These four block merge.

### Defect-class scorecard (the 5 the refactor claimed to close)

| # | Defect class (`m02_coverage_contract.md:104-117`) | Status |
|---|---|---|
| 1 | Module-label identity split | ✅ Closed (labels read "Module 2"; verified D1 kicker/title/brand-sub) |
| 2 | Numeric self-consistency (D6 `0.746`, D9 step-200, D5 `~3×`) | ✅ Closed (D5 `~3×` at `:413`; others re-confirmed) |
| 3 | Playground ↔ Produce agreement (D7/D8/D9) | ⚠️ **Partially open** — D7 `g=−9.6,v=−24.0` ✅; **D8 5/5/4 vs 8** now **P1** (narrated); **D9 split framing + missing test set still P0** |
| 4 | Worked-example spine (D2 one anchor; D1 `np.maximum`) | ✅ Closed (D2 single `W₁@W₂=[[7,2],[3,1]]` at `:402,454,595`; D1 `np.maximum` at `:451,598`) |
| 5 | No premature close (D6) | ✅ Closed |

---

## Coverage traceability table

Must-cover concepts (`m02_coverage_contract.md` "Must cover" + 4 promoted), each traced to its named
lesson. **PASS** = lands at depth with required visual + artifact · **PARTIAL** = taught but a required
surface (usually the artifact) is missing · **FAIL** = absent at required depth. Line refs marked ✔ were
re-read this session; the rest are carried from the v7 line-by-line pass on **identical** file bytes.

| Concept | Req. depth | Lesson evidence | Visual/evidence | Artifact | Result |
|---|---|---|---|---|---|
| Neuron as affine map `z=w·x+b` | Core | ✔ D1 `:348-353,402-403` | ✔ Playground 3-step + build term-by-term `:588-627` | ✔ `neuron()` Opt A `:451` | ✅ PASS |
| Weights/bias = learned state; geometry | Core | ✔ D1 `:412,417,404` | ✔ build weights `:619` | random-init framing | ✅ PASS |
| Bias not optional (origin) | Support | ✔ D1 `:404` callout | ✔ build step 4 `:623` | criterion `:468` | ✅ PASS |
| **Symmetry-breaking init** | Core (staff) | ✔ D1 `:417` + quiz `:671` | ✔ callout | explain-back | ✅ PASS |
| Nonlinearity required; `linear∘linear` collapses (scalar-first) | Core | ✔ D2 `:401-403` (scalar then matrix) | ✔ build collapse→cure `:613-616` | ✔ Produce collapse `:454` | ✅ PASS |
| **ReLU / sigmoid / tanh — formulas, ranges** | Core | ✔ D2 `:395` shows **only ReLU + sigmoid**; `tanh` = 0 hits module-wide | ✔ no tanh curve/demo | ✔ Produce omits tanh `:453` | ❌ **FAIL (P0-2)** |
| Saturation + dead-ReLU | Core (staff) | ✔ D2 `:411` + quiz `:664` | ✔ build/staff note | log item `:464` | ✅ PASS |
| Layer `Z=xW+b` with shapes | Core | ✔ D3 `:351,401` | build shape chain | Produce | ✅ PASS |
| Forward pass → `ŷ`; batch; one numeric pass | Core | ✔ D3 `:401-402` shape chain + S4 numeric callout | build + playground (shapes) | batch-of-4 | ✅ PASS (playground numeric = P1-4) |
| Logits / raw scores | Support | ✔ D3 `:398` term + tooltip | build note | n/a | ✅ PASS |
| MSE — formula, why squared, convex bowl in ŷ | Core | ✔ D4 `:352` | falling-loss curve (no convex-bowl draw) | Produce | 🟡 PARTIAL (bowl visual = P2) |
| Cross-entropy + softmax/sigmoid & why-not-MSE | Core | ✔ D4 `:353,432-433` | playground CE + drop | acceptance `:483` | ✅ PASS |
| **MSE vs CE comparison table** | Core | ✔ D4 real `<table>` `:413-421` | table element | n/a | ✅ PASS |
| Loss = MEAN over batch | Support | D4 `:405` | build step | `np.mean` | ✅ PASS |
| Loss as scalar `L(θ)` — surface | Core | D4 `:425` handoff | build step | n/a | ✅ PASS |
| Gradient uphill; `−∇L` descends | Core | ✔ D5 `:405` | ✔ build slope + live embed `:430` | GD loop `:455` | ✅ PASS |
| **Vanilla GD named** baseline | Core | ✔ D5 `:414` ("…is **vanilla gradient descent**") | build step | loop `:455` | ✅ PASS |
| **Backprop = real numeric multi-link chain** | Core | ✔ D5 Math Ladder `:406-410` (correct) | ✔ build backward-reveal `:631` | ✔ **Produce omits chain part (b)** `:454-466` | 🟡 **PARTIAL → P0-3** (no runnable evidence for the headline concept) |
| Forward-cache → train mem > inference | Support | D5 `:405,420` | callout | explain-back | ✅ PASS |
| Non-convexity / local vs global | Core (aware) | ✔ D5 `:419` (exploding/vanishing) + embed note | bumpy-hill note | n/a | ✅ PASS |
| Training loop w/ `zero_grad`, epochs/batches | Core | ✔ D6 quiz `:686` (accum/reset) | loop-cycle + falling loss | Produce | ✅ PASS |
| Epoch/iteration/batch (worked count) | Support | D6 `100/25=4` | batch boxes | n/a | ✅ PASS |
| Batch/mini-batch/SGD = noisy unbiased | Core | ✔ D6 batch `<table>` `:421` | trade-off table | n/a | ✅ PASS |
| Random-init scale (fan-in) | Support (staff) | D6 (Xavier/He); D1 `:418` | callout | n/a | ✅ PASS |
| SGD rule `θ←θ−η∇L` | Core | D7 `:405,420` | playground + build | Opt A | ✅ PASS |
| Momentum rule + what it fixes | Core | D7 (`g=−9.6,v=−24.0`) | playground + build | loop | ✅ PASS |
| Adam (m,v, bias-corr, per-param; β/ε) | Core | D7 (β₁.9/β₂.999/ε1e-8) | playground + build | note | ✅ PASS |
| LR regimes: small/right/large(→NaN) | Core | ✔ D8 `:404,408-415` | ✔ playground 3 rates `:383-385` + build + live embed | ✔ Produce `:463` | ✅ PASS (step-count = P1) |
| LR schedules + warmup | Support | ✔ D8 `:420-421` | warmup/decay SVGs | n/a | ✅ PASS |
| Train / val / **test** roles | Core | ✔ D9 lesson body (3-box build) | ✔ 3-box build | ✔ **Produce drops the test split** `:463,478-484` | 🟡 **PARTIAL → P0-4** |
| Overfit/underfit via train–val gap | Core | ✔ D9 `:465,485` | two-curve diverge build | train/val print | ✅ PASS |
| Early stopping + L2/dropout (gap-closers) | Core (aware) | ✔ D9 build | best-val marker | criterion | ✅ PASS |

**Result:** 27/32 rows PASS · **1 FAIL** (D2 `tanh`) · **3 PARTIAL** (D4 convex-bowl visual → P2; D5
backprop artifact → P0; D9 test-split artifact → P0). **The table does not pass** — `tanh` (Core
must-cover) is absent, so "all concepts landed" cannot be signed off. (This coverage table is orthogonal
to the Foundation Framing Gate below, which fails on D1 independently of any must-cover row.)

---

## Coach Voice Gate — ✅ PASS (all 9 days)

Every day opens with a why-care hook and a named pain point **before** any formula (the `😕 Why this
trips people up` bilingual callout sits in section 1 on every day; formulas live in section 4). Each
carries a multi-sentence analogy with an explicit "where it breaks" line, a Math Ladder, one
silent-failure + one trade-off callout, an interview answer, and a diagnostic quiz Q. No duplicated
callouts. Minor nit carried from v7: **P1** — D6 never explicitly reframes "training is a run-once job
that terminates."

> ⚠️ Note the boundary: the Coach Voice Gate is about *warmth and ordering*, and D1 passes it — its
> "judge" analogy is warm and lands early. The **Foundation Framing Gate is a separate, stricter test**
> that D1 fails (below). A warm everyday analogy satisfies coach voice but **not** foundation framing.

---

## Foundation Framing Gate — ❌ FAIL (D1) · **P0-1**

D1 is *the* neural-network foundation lesson (the first neuron). v7.1 requires **all five** elements;
D1 has **3 of 5**:

| Required element | Present? | Evidence |
|---|---|---|
| Explicit **brain-inspired intuition** (as the foundation frame) | ❌ **No** | Section 2 intuition is **"A judge scoring a contestant"** (`:365-373`) — a human/everyday analogy. `brain` appears exactly **once** module-wide, buried in the D1 "where the analogy breaks" callout (`:372` "a real brain neuron is far messier"). `dendrite`/`axon`/`synapse`/`biological` = **0 hits**. |
| Artificial-neuron caveat | ✅ Yes | `:372` "one artificial neuron alone is weak. The power comes from **many** neurons in **layers**." |
| **Mapping table** biological → artificial | ❌ **No** | No table anywhere in D1 maps dendrites→inputs, synaptic strength→weights, cell-body-sum→`w·x`, firing threshold→bias/activation. The only D1 tables/cards are the two "judge" `.relate` cards (`:367-370`). |
| Where the analogy breaks | ✅ Yes | `:372` (about the judge/brain). |
| Transition to math function | ✅ Yes | `:402-403` `output = activation(w·x + b)` + worked `−2 / −1 / 0`. |

**Verdict.** The everyday "judge" analogy is used **as a substitute** for brain-vs-ANN framing — which
v7.1 names as a P0 (`SKILL.md`: "generic everyday analogy used as substitute for brain-vs-ANN framing";
"A human/everyday analogy such as judge, chef, or scoring function does not satisfy the brain-vs-ANN
framing requirement"). Two of the five required elements are missing. **P0.**

---

## Function Family Gate — ⚠️ INCOMPLETE (D2 activations) · **P0-2 + P1**

Activation family expected: sigmoid, tanh, ReLU, Leaky ReLU, softmax, derivative intuition, saturation,
vanishing gradient, dying ReLU, GELU/SwiGLU context.

| Family member / facet | Status | Evidence |
|---|---|---|
| ReLU | ✅ formula + curve + playground + quiz | `:395,582,609,663` |
| sigmoid | ✅ formula + curve + playground + range | `:395,586,611` |
| **tanh** | ❌ **absent everywhere** (no formula, range, curve, playground, quiz) | grep `tanh` = **0** module-wide |
| Leaky ReLU | ✅ one-line (Aware, contract-scoped) | `:412` |
| GELU / SwiGLU | ✅ one-line modern context | `:350,412` |
| softmax | ✅ named + tooltip (home is D4) | `:406,412` |
| saturation | ✅ | `:411` staff lens + build step 2 |
| vanishing gradient | ✅ term + tooltip | `:406,620` |
| dead / dying ReLU | ✅ staff lens + quiz Q4 | `:411,664` |
| derivative intuition (e.g. `sigmoid'(0)≈0.25`) | 🟡 qualitative only ("barely moves") — no derivative value in-lesson | `:411` |
| **activation comparison table** | ❌ **absent** | no `<table>` in D2 (curves in build + prose in staff lens) |

**Verdict.** The family is **incomplete**: `tanh` — a Core must-cover representative variant — is
entirely missing (**P0-2**, same root as the coverage FAIL and the D2 artifact gap). Separately, D2 has
**no comparison table** for the family (S8 + Function Family Gate ask for one) — **P1**. Leaky ReLU /
GELU / SwiGLU are correctly present as Aware one-liners.

> For completeness: the **optimizer** family (D7: SGD/momentum/Adam) is well-covered in prose + build +
> rules, but also has **no comparison `<table>`** — **P1** (the contract does not mandate a D7 table, so
> this is should-cover formatting, not a coverage failure).

---

## Visual/Evidence Gate — ✅ PASS on required P0 visuals

Every day's P0 visual exists and shows a changing quantity; every simulated terminal is explicitly
labelled "a simulated Python terminal"; the D5/D7/D8 `viz/gradient-descent.html` embeds carry the
`viz-height` autoresize receiver (`:760-772`). Gaps: **P1** — D3 playground prints shapes only, no
per-stage numeric readout (the P0 "concrete numeric pass" is satisfied by the S4 callout). **P2** — D4
"convex bowl in ŷ" never drawn (falling-loss-vs-steps curve instead).

> ⚠️ **Evidence-for-failure-modes check.** Two must-cover failure modes have their *evidence* only in a
> **simulated terminal / prose**, not in a reproducible artifact: **D5 backprop / vanishing gradients**
> (the numeric chain lives in the Math Ladder, and the Produce artifact never reproduces it — see
> Artifact Gate) and **D2 saturation / dead-ReLU** (`sigmoid'(0)≈0.25` and "ReLU grad = 0 on negatives"
> are contracted for the artifact but the Produce omits them). Per v7.1, a failure mode taught without a
> reproducible artifact is a defect — these roll up into P0-2 (D2) and P0-3 (D5).

---

## Anchor Consistency Gate — ⚠️ one P0 (D9), one P1 (D8)

Anchors tracked: step counts, split ratios, seeds, learning rates, dataset sizes, param values, metric
definitions. Cross-checked prose ↔ visual ↔ playground ↔ quiz ↔ Produce ↔ acceptance.

- ✅ **D1** `x=[2,3], w=[0.5,−1], b=1 → −2/−1/0` identical across S3 playground, S4 ladder, build, Produce.
- ✅ **D2** single anchor pair `W₁@W₂=[[7,2],[3,1]]` identical across playground (`:595`), ladder (`:402`),
  Produce (`:454`).
- ✅ **D5** chain `dL/dw≈−0.212`; D8 tipping `lr=2/c=0.25` (c=8); D7 `g=−9.6,v=−24.0` — all internally consistent.
- ⚠️ **D8 (P1):** playground runs **5 / 5 / 4** steps (`range(5)`, `range(5)`, `range(4)` at script lines
  for `toosmall`/`justright`/`toobig`) while Produce mandates **8 / 8 / 8** (`:463,478`). Line 390 now
  narrates the playground as "a short preview" and the artifact as "more steps," so **no conflicting
  mental model is taught** → **P1**, not P0. Residue: the internal 5-vs-4 split is arbitrary, and Produce
  violates `artifact_contract.md:26` criterion (4) "step count == the playground's."
- ⚠️ **D9 (P0-4):** the Produce **teaches a different split than the lesson** — lesson body = train/val/**test**
  (3 boxes); Produce = **2-way 20-train/10-val, no test** (`:463,478-484`). This is a **conflicting
  mental model** on the exact concept the day exists to establish → **P0** (detailed under Artifact Gate).

---

## Artifact Gate — ❌ three P0 gaps + P1/P2 residue

Path + run command are correct on every day (`python3 sessions/m02-the-neuron/<folder>/experiment.py`,
verified in all 9 stubs and all 9 Produce steps); every day has the 📓 5-minute-log callout. But three
Produce steps break the artifact contract, two of them leaving a **must-cover concept with no runnable
evidence**:

- **D2 (P0-2):** Produce omits `tanh`, the `np.linspace(-6,6,13)` grid, `sigmoid'(0)≈0.25`, and the
  "ReLU grad = 0 for negatives" note that `artifact_contract.md:20` mandates. `day-02:453-460`. A
  required activation can never be produced.
- **D5 (P0-3):** Produce covers only part (a) `f(w)=w²` GD (`day-05:454-466`). The **entire contract
  part (b)** — derive `dL/dw, dL/db` on `(sigmoid(w·x+b)−y)²`, print the chain pieces, verify vs finite
  differences to `<1e-6` — is **absent**. The day's headline concept (backprop) has **no runnable
  artifact**; the numeric chain lives only in the Math Ladder prose. The 📓 explain-back (`:475`) asks
  "what a gradient is / lr=1.0 / vanishing gradients" and **drops the contract's finite-difference
  probe** entirely.
- **D9 (P0-4):** Produce (Option A `:463`, Option B `:478-484`, acceptance `:487-489`) uses a **2-way
  20/10 train/val split with no test set**, contradicting the 3-split requirement and **silently
  re-teaching the "val = test" misconception the day exists to kill**. It fails `artifact_contract.md:27`
  criteria (3) disjoint-splits-sum-to-N and (4) test-loss-at-best-val. **P1 rider:** Option A also says
  "print the training loss and validation loss **as you train**" for a one-shot degree-9 `lstsq` fit,
  which has no training-step axis (Option B's degree-sweep is the coherent version).

Lower-severity: **P1** — D1 acceptance omits the determinism criterion + the log drops the bias
explain-back (`:466-471`); D4 acceptance omits MSE==0-at-match and clip/no-`log(0)` (`:487`). **P2** —
all 9 `experiment.py` stubs say "open lesson.html" (contract template says "section 7") and bare
`frontier-experiment-lab` (no slash); D4 convex-bowl visual; D7 no bias-correction in the Adam playground.

---

## Legacy skill reference check — ✅ clean except the sanctioned pattern

Full 12-name sweep across `day-*/lesson.html`, `experiment.py`, and both review pages:
**11 of 12 legacy names = 0 hits** (`frontier-session-coach`, `-math-unfogger`, `-lesson-humanizer`,
`-d3-visual-lab`, `-visual-auditor`, `-artifact-reviewer`, `-curriculum-rollout`, `-qa-gate`,
`-visual-lab`, `-lesson-coach`, `-visual-builder`). Only **`frontier-experiment-lab`** appears — in 18
files (9 Produce Option-B blocks + 9 stub comments). Per the strict gate a reference to an uninstalled
skill is raw-P0, **but** it is the **contract-sanctioned** Produce Option-B pattern
(`m02_artifact_contract.md:5` prescribes exactly this trigger) and a **documented curriculum-wide** item
(~243 files, "P1, don't fix per-module"). Carried as **P1-legacy**, not a Module-2 regression.

---

## P0 findings

> Blockers. Each confirmed by reading the current file bytes this session.

| ID | Day | Gate | Defect (evidence) |
|---|---|---|---|
| **P0-1** | D1 | Foundation Framing | **Missing brain-vs-ANN foundation framing.** Section-2 intuition is the everyday "judge scoring a contestant" (`:365-373`); no explicit brain-inspired intuition (only one `brain` mention in the break line `:372`) and **no biological→artificial mapping table**. v7.1 disallows an everyday analogy as the foundation frame. 2 of 5 required elements missing. |
| **P0-2** | D2 | Coverage + Function Family + Artifact | **`tanh` — a Core must-cover activation — is entirely absent** (0 hits: no formula, range, curve, playground, quiz). The Produce also omits tanh + the `linspace(-6,6,13)` grid + `sigmoid'(0)≈0.25` + the ReLU-grad-0 note (`:453-460` vs `artifact_contract.md:20`). Module-incomplete per the completeness check. |
| **P0-3** | D5 | Artifact / Evidence | **Backprop has no runnable artifact.** Produce covers only `f(w)=w²` GD; the contract's chain-rule + finite-difference check (part b) is absent (`:454-466`), and the 📓 explain-back drops the finite-difference probe (`:475`). The headline concept's only evidence is Math-Ladder prose. |
| **P0-4** | D9 | Artifact / Anchor | **Produce drops the test split** (2-way 20/10 train/val, no test — `:463,478-489`), contradicting the 3-split lesson and silently re-teaching "val = test"; fails 2 of 4 contract criteria. |

---

## P1 findings

> Fix before merge if easy.

| ID | Day/Area | Defect (evidence) |
|---|---|---|
| **P1-1** | D8 | Anchor/artifact: playground **5/5/4** vs Produce **8/8/8**; internal 5-vs-4 is arbitrary; Produce breaks `artifact_contract.md:26` criterion (4). Narrated at `:390` so not P0. |
| **P1-2** | D2 | No **activation comparison table** (S8 + Function Family Gate expect one for the family). |
| **P1-3** | D7 | No **optimizer comparison table** (SGD/momentum/Adam covered in prose+build only). |
| **P1-4** | D3 | Playground prints shapes only, no per-stage numeric readout (`visual_contract.md:23`). |
| **P1-5** | D9 | Produce Option A is internally incoherent: a one-shot degree-9 `lstsq` fit has no "as you train" axis (`:463`). |
| **P1-6** | D1 | Acceptance omits the determinism criterion; 📓 log drops the bias explain-back (`:466-471`). |
| **P1-7** | D4 | Acceptance omits MSE==0-within-1e-12-at-match and the clip/no-`log(0)` criterion (`:487`). |
| **P1-8** | D6 | "Training is run-once / terminates" misconception never explicitly reframed. |
| **P1-9** | review.html | (Carried, unverified this pass) final gate quiz reportedly tests only D4–D6; re-check before sign-off. |
| **P1-legacy** | all 9 days | `/frontier-experiment-lab` (uninstalled) in every Produce Option-B + stub. **Contract-sanctioned; curriculum-wide — do NOT fix per-module.** |

---

## P2 findings

> Polish. Do not block.

| ID | Day/Area | Defect |
|---|---|---|
| **P2-1** | all stubs | `experiment.py` comment says "open lesson.html" (template: "section 7") and bare `frontier-experiment-lab` (no slash). |
| **P2-2** | D4 | Convex-bowl-in-`ŷ` visual not drawn (falling-loss-vs-steps instead). |
| **P2-3** | D2 | No explicit derivative value in-lesson (`sigmoid'(0)≈0.25` is contracted for the artifact only). |
| **P2-4** | D7 | Adam playground shows `m/√v` not bias-corrected `m̂/v̂`; no "simplified" note (coverage met by S4). |
| **P2-5** | D9 | Playground "around step 300" vs step-200 early-stop point — narrative looseness (numerically consistent). |
| **P2-6** | review.html | (Carried) self-check items tagged Day 1/2/3 but content = Days 4/5/6 — residue of internal `wf3-*` numbering. Do NOT touch `data-quest-id`. |

---

## What passed cleanly (recorded for faithfulness)

- **Coach voice + full Coach 6-pack** on all 9 days.
- **Math correctness** on the re-read days: D1 `−2/−1/0`; D2 `W₁@W₂=[[7,2],[3,1]]`, `sigmoid([-2,0,2])≈[0.119,0.5,0.881]`;
  D5 chain `dL/dw≈−0.212` + `f(w)=w²` GD toward 0; D8 tipping `lr=2/c=0.25` with `c=8`, factors `0.96/0.2/−3.8`.
- **Module-label unification** (D1 kicker/title/brand-sub all "Module 2").
- **experiment.py run paths** correct in all 9 stubs.
- **Legacy sweep** clean but for the sanctioned `frontier-experiment-lab`.
- Anchor spine intact on D1, D2, D5, D7 (the days it was previously broken on, minus D8/D9).

---

## Fixes applied

**None.** Report-only pass (per instruction). The only file written is this report,
`sessions/m02_v7_1_gap_report.md`.

---

## Merge recommendation

### 🚫 Blocked (on P0-1 … P0-4)

Four confirmed P0 defects:

- **Foundation framing:** D1 substitutes an everyday "judge" analogy for the required brain-vs-ANN
  framing + mapping table (P0-1) — the new v7.1 finding.
- **Coverage + family:** D2 is missing `tanh`, a Core must-cover activation, everywhere (P0-2).
- **No runnable evidence for a must-cover concept:** D5 backprop (P0-3).
- **Anchor conflict re-teaching the day's own misconception:** D9 Produce drops the test split (P0-4).

The lesson bodies, coach voice + 6-pack, math, all JS/nav/completion, the module-label unification, and
4 of 5 numeric/spine defect classes are clean. **Once P0-1 … P0-4 are addressed (all localized to one
D1 section + three Produce blocks), the module lands at "Pass with P1."** The D8 step-count item is now
P1 (narrated), and `/frontier-experiment-lab` (P1-legacy) is deliberately out of per-module scope.

> `data-quest-id` values and the frozen invariants must not change during any fix — several P0/P1 fixes
> touch Produce prompts and the review-page self-check, which sit next to those frozen keys.

---

## P0 fix plan (from failed gates only)

Ordered by gate; each is a localized, non-structural edit. **Do not touch `data-quest-id`.**

### P0-1 — D1 Foundation Framing (Foundation Framing Gate)
- **Where:** `day-01-single-neuron/lesson.html`, section 2 (`:363-376`) — keep the warm "judge" card as a
  *secondary* everyday hook, but lead with brain framing.
- **Add:** (1) a short **brain-inspired intuition** paragraph (a biological neuron sums signals from many
  others and "fires" past a threshold); (2) a **mapping table** — biological → artificial: dendrites/inputs →
  `x`, synaptic strength → `w`, cell-body summation → `w·x`, firing threshold → `b` + activation, axon
  output → the neuron's output; (3) keep the existing artificial-neuron caveat + "where it breaks" line
  (already at `:372`), extending "where it breaks" to name what the brain analogy specifically gets wrong
  (no biological backprop; real neurons spike in time; weights here are learned by gradient descent).
- **Accept:** the five Foundation-Framing elements all present; the everyday analogy is clearly secondary
  to the brain frame; `lesson_audit.py m02-the-neuron` still 9 OK.

### P0-2 — D2 `tanh` + activation artifact (Coverage / Function Family / Artifact)
- **Where:** `day-02-activations/lesson.html` — S4 formulas (`:394-395`), the playground `DEMOS`
  (`:581-597`), the build curves (`:608-620`), one quiz option, and the Produce block (`:452-462`).
- **Add:** `tanh(z)=(e^z−e^−z)/(e^z+e^−z)`, range `(−1,1)`, zero-centered vs sigmoid; a `tanh` curve (build)
  and a `tanh` playground demo; extend the Produce/acceptance to print `tanh` on the grid, plus
  `np.linspace(-6,6,13)`, `sigmoid'(0)≈0.25`, and "ReLU grad = 0 on negatives" per `artifact_contract.md:20`.
- **Also (P1, cheap while here):** add a **ReLU/sigmoid/tanh comparison table** (range · zero-centered? ·
  derivative behavior · saturates? · typical use) — closes the Function-Family table gap.
- **Accept:** `grep tanh day-02` > 0 across S4/playground/build/produce; the coverage row flips FAIL→PASS.

### P0-3 — D5 backprop artifact (Artifact / Evidence Gate)
- **Where:** `day-05-gradients-backprop/lesson.html` Produce (`:454-466`) + 📓 log (`:475`).
- **Add part (b)** to Option A and Option B: on `(sigmoid(w·x+b)−y)²` with the Math-Ladder anchors
  (`x=2, w=0.5, b=0, y=1`), compute `dL/dw, dL/db` analytically, **print the chain pieces**
  (`dL/da · da/dz · dz/dw`, reproducing `≈−0.212`), and **verify vs finite differences to `<1e-6`**.
  Restore the contract explain-back: "Why is backprop just the chain rule backward, and what does the
  finite-difference check prove?"
- **Accept:** the backprop concept has a runnable artifact whose numbers match the Math Ladder; the
  finite-difference probe is in the log question.

### P0-4 — D9 test split (Artifact / Anchor Gate)
- **Where:** `day-09-train-val-test/lesson.html` Produce (`:463`, `:478-489`).
- **Fix:** make the artifact a **3-way disjoint split** (train / val / test) with sizes printed that sum
  to N; evaluate **test loss only at the best-val checkpoint**; keep the degree-sweep framing (Option B)
  and drop the "as you train" phrasing from Option A (fixes P1-5 at the same time). Align the acceptance
  bullets to `artifact_contract.md:27` criteria (1)-(4).
- **Accept:** three disjoint splits sum to the dataset; best-val ≠ final; test used once; the artifact no
  longer re-teaches "val = test."

**Not in this P0 plan (correctly excluded):** D8 step-count (P1 — narrated), the `/frontier-experiment-lab`
reference (P1-legacy — curriculum-wide), and all P2 polish.
