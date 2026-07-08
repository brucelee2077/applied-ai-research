# Module 2 — "The Neuron & How It Learns" — First-Principles Blueprint

> **Mode:** source-free, first-principles. This blueprint was designed by reasoning from scratch about
> what an ideal module *should* teach, then reconciled against the existing `sessions/m02-the-neuron`
> code (treated as a shell + rough draft, **not** as the source of truth). Old notebooks under
> `00-neural-networks/fundamentals/` were **not** read.
>
> **Authoritative status:** this file + the four contracts beside it
> (`m02_coverage_contract.md`, `m02_visual_contract.md`, `m02_artifact_contract.md`, `m02_refactor_plan.md`)
> govern the refactor. Lesson edits must satisfy them.

---

## 0. Frozen invariants (read before touching anything)

These are load-bearing and MUST NOT change:

| Invariant | Value | Why frozen |
|---|---|---|
| Module folder | `sessions/m02-the-neuron/` | Path referenced by hub + nav |
| Lesson count / order | 9 days + 2 gates | `review-part-a.html` after Day 3, `review.html` after Day 9 |
| Quest-ids | `wf2-d01-neuron`, `wf2-d02-activations`, `wf2-d03-forward`, `wf3-d01-loss`, `wf3-d02-backprop`, `wf3-d03-training-loop`, `wf3-d04-optimizers`, `wf3-d05-lr`, `wf3-d06-split`, `wf2-review`, `wf3-review` | localStorage keys (`frontier-lesson:<QID>`) + hub progress pills + the **"First Neuron" badge** keys on `wf2-d01-neuron` (`index.html:910`) |
| Shell | sidebar + 4-theme switcher (dim default) + progress bar + checklist + scrollspy | Whole-repo consistency (see MEMORY: sessions-shell-migration) |
| 7-section model | `what / intuition / play / why / build / quiz / produce` (`data-sec`) | `.gotit` gating + checklist + `count` "0/7" |
| Behaviors | playground `DEMOS`, scroll-reveal `BUILD`, `QS` quiz, Produce copy-button, `.term` tooltips, viz-iframe autoresize, review gates' `done.verdict` completion | Established interaction contracts |
| Nav chain | prev/next exactly as wired today (Day3→review-part-a→Day4 … Day9→review→m03) | Verified continuous (MEMORY: sessions-nav-wiring) |

**What the hub already believes (authoritative, `index.html:344`):**
`{n:2, title:'The Neuron & How It Learns', goal:'Build a neuron and a layer, then learn how it
measures error and updates its weights — one continuous story.'}`. All learner-facing labels must
match this. Quest-ids `wf3-*` stay as-is (an internal workflow sub-arc); only the human-facing
"Module 3 / Module 2-3" **text** is wrong and gets unified to **"Module 2 · Day 1–9"**.

---

## 1. Learner profile

A **senior software engineer**, strong at programming, data structures, systems, and reading code —
but **new to ML training**. Reads English fluently (this repo prefers English; Chinese only where a
single bilingual line genuinely helps a stuck point). They are prepping toward **Staff/Principal MLE**
depth (Reader B), so every day must land both the plain intuition (Reader A voice) *and* a precise
staff lens + interview answer.

What this profile changes about the teaching:
- **Kill mysticism fast.** This reader over-respects "neural" and "backprop" as magic. Reframe
  everything as ordinary functions, arithmetic, and loops they already trust.
- **Lean on their existing mental models** — pure functions, pipelines, control loops, config vs
  discovered state — and then mark exactly where ML breaks those models (weights are *discovered*, the
  loop has *no natural exit*, "passing tests" (low train loss) can mean *failure*).
- **No hand-waving survives.** A senior engineer will ask "why does the framework take logits, not
  probabilities?" and "do we find the global minimum?" — so the module must own those answers, not
  defer them to faith.

---

## 2. Why this module matters

This is the **spine of all of deep learning**. Every later module (attention, transformers, vision,
JAX, training systems, inference) is *scale and specialization on top of exactly this loop*: a neuron
is `activation(w·x + b)`; a layer is that as one matmul; a network is stacked layers; a loss turns the
output into one scalar; the gradient of that scalar says which way to nudge every weight; an optimizer
does the nudging; the learning rate sizes it; and train/val/test tells you whether it *learned* or just
*memorized*. If a candidate cannot narrate that chain crisply and name where each step silently breaks,
they are not ready for a frontier-lab interview. Module 2 is where that chain is built once, from atoms.

---

## 3. Prerequisites

- **Module 1** — arrays & shapes (Day 1), indexing (Day 2), broadcasting & dtypes (Day 3), **matmul &
  the dot product** (Day 4), **logs & exponents** (Day 5). The dot product (M1 D4) and `log`/`exp`
  (M1 D5) are used directly. No neural-network background is assumed — Day 1 is the first NN piece.
- **Programming fluency** — Python, NumPy, functions, loops. Assumed, not taught.

**Genuine cross-module references to keep:** "M1 Day 4" (dot product/matmul), "M1 Day 5"
(logs/exponents, scaling-laws framing), "M1 Day 1" (GPT-3 `(12288, 49152)` weight-matrix teaser).
**Cross-references to fix:** anything inside these lessons that calls an *earlier day of this same
module* "M2" or "M3" (e.g. Day 5 calling the forward pass "M2") — the forward pass is **Day 3 of this
module**, so reword to "Day 3 (earlier this module)".

---

## 4. Concept map

```
        REPRESENT THE FUNCTION                  BUILD THE LEARNING SIGNAL              DRIVE THE MACHINE                    TRUST IT
  ┌───────────────────────────────┐      ┌───────────────────────────────┐   ┌──────────────────────────────┐   ┌───────────────────┐
  │ D1 neuron  z = w·x + b         │      │ D4 loss  L(θ) = one scalar     │   │ D6 loop: fwd→loss→back→      │   │ D9 train/val/test │
  │    (affine map; weights learned│──┐   │    MSE (regression)            │   │     zero_grad→update, ×epochs│   │    the train–val   │
  │     not config; symmetry break)│  │   │    cross-entropy (classif.)    │   │     batch = noisy estimate   │   │    gap = overfit   │
  │ D2 activation = the bend       │  │   │    logits→softmax→prob         │   │ ──── vanilla gradient descent│   │    early stopping, │
  │    linear∘linear collapses     │  ├──▶│    loss = MEAN over batch      │──▶│      is the baseline ───▶    │──▶│    regularization  │
  │    ReLU / sigmoid / tanh       │  │   │ D5 gradient ∇L points uphill;  │   │ D7 optimizer = the hand:     │   │    (concept)       │
  │ D3 layer = matmul, forward pass│──┘   │    backprop = chain rule back  │   │     SGD → momentum → Adam    │   │    data leakage    │
  │    shapes chain; init scale    │      │    (real numeric chain);       │   │ D8 learning rate = step size │   └───────────────────┘
  │    ── REVIEW-PART-A (gate) ──  │      │    forward-cache; non-convex   │   │     too small/right/too big  │            ▲
  └───────────────────────────────┘      └───────────────────────────────┘   └──────────────────────────────┘   ── REVIEW (gate) ──
         "what the network IS"                  "how we score & blame"              "how it actually moves"          "did it learn?"
```

Each day supplies exactly the object the next day operates on. There is no earlier day whose content
could move later without breaking a prerequisite, and no gap where a needed object appears
un-introduced. The two gates bracket the two natural halves: **what the network is** (after D3) and
**how it learns and whether it learned the right thing** (after D9).

---

## 5. Learning sequence (per-day one-line goal + what it introduces / assumes)

| Day | Folder / quest-id | One-line goal | Introduces | Assumes |
|---|---|---|---|---|
| 1 | `day-01-single-neuron` · `wf2-d01-neuron` | A neuron is a tiny tunable function `z = w·x + b`. | neuron as `activation(w·x+b)`; pre-activation `z`; weights **learned** vs config; bias shifts (not optional); dot product = weighted sum; **symmetry-breaking init** (staff) | M1 dot product, shapes |
| 2 | `day-02-activations` · `wf2-d02-activations` | Why a neuron needs a bend, or a whole stack collapses to one line. | linearity-collapse (**scalars first**, then matrices); ReLU/sigmoid/tanh (formulas, ranges); ReLU default; saturation "curve goes flat"; **dead ReLU** (staff) | D1 neuron + placeholder activation |
| 3 | `day-03-layers-forward-pass` · `wf2-d03-forward` | Compose neurons into layers and run the forward pass to a prediction `ŷ`. | layer `Z=xW+b` with shapes; stacking; forward pass end-to-end (one **numeric** pass, not shapes-only); batch dim; logits/"raw scores"; **missing-nonlinearity + broadcasting** (staff) | D1 neuron, D2 why-nonlinearity |
| — | `review-part-a.html` · `wf2-review` | Gate on the static forward model (mid-module checkpoint). | — | D1–D3 |
| 4 | `day-04-loss` · `wf3-d01-loss` | Turn error into one scalar; pick MSE (regression) vs cross-entropy (classification). | MSE + CE (with **table**); loss = **MEAN** over batch; logits→softmax→prob + **log-sum-exp** (should); loss surface `L(θ)`; **double-softmax / MSE-on-classification** (staff) | D3 `ŷ`, D2 sigmoid/softmax |
| 5 | `day-05-gradients-backprop` · `wf3-d02-backprop` | Compute `∇L` for every weight by running the chain rule backward. | gradient points uphill, `−∇L` descends; **name "gradient descent"** baseline; backprop = chain rule with a **real numeric multi-link chain**; forward-cache → training mem > inference; **non-convexity / local vs global**; **missing `zero_grad` + vanishing/exploding** (staff) | D3 forward+cache, D4 scalar loss, D2 activation slopes |
| 6 | `day-06-training-loop` · `wf3-d03-training-loop` | Assemble the repeating loop: forward→loss→backward→**zero_grad**→update, over batches/epochs. | loop skeleton (with zeroing); epoch/iteration/batch (worked count); full/mini/stochastic batch as a **noisy unbiased** estimate; **random-init scale** (fan-in intuition); track a held-out metric; **eval-mode/no_grad + loss≠metric** (staff) | D3 fwd, D4 loss, D5 gradient+GD |
| 7 | `day-07-optimizers` · `wf3-d04-optimizers` | Exact SGD, momentum, Adam — each as "what it ADDS" to the GD baseline. | SGD `θ←θ−ηg`; momentum `v←μv+g, θ←θ−ηv`; Adam `m,v`+bias-correction+per-param step (β₁=.9, β₂=.999, ε=1e-8); **AdamW weight-decay coupling** (staff) | D5 gradient + named GD baseline, D6 update slot |
| 8 | `day-08-learning-rate` · `wf3-d05-lr` | The step size: crawl vs converge vs diverge, plus schedules. | η regimes with a concrete divergence; `(1−ηc)` recurrence intuition (c=curvature); warmup + decay (concept); LR–batch coupling (should); **LR-too-high silent plateau** (staff) | D7 rules contain η, D5 surface/conditioning |
| 9 | `day-09-train-val-test` · `wf3-d06-split` | Split data; define the train–val gap; detect & fix overfitting. | train/val/test roles; overfit/underfit via gap `=val−train`; two-curve plot + turn-up; early stopping (best checkpoint); L2/dropout as gap-closers (one-line mechanism); **data leakage** (staff) | D6 loop tracks val, D4 loss metric |
| — | `review.html` · `wf3-review` | Final gate: the full learning loop + generalization. | — | D1–D9 |

---

## 6. Must-cover concepts

The chain is incomplete if any of these is missing. Full table with depth + owner in
`m02_coverage_contract.md`. Headline list:

1. Neuron as an affine map + pre-activation `z = w·x + b` (D1)
2. Weights & bias are **learned state**, geometry: `w` orients, `b` shifts the boundary (D1)
3. **Symmetry-breaking initialization** — why weights can't all start equal/zero (D1 staff)
4. Why a nonlinearity is required: `linear ∘ linear` collapses (**scalar proof first**) (D2)
5. ReLU / sigmoid / tanh — formulas, ranges, ReLU as hidden default (D2)
6. Saturation failure ("curve goes flat" → dead learning) + dead-ReLU counter-failure (D2)
7. Layer as `Z = xW + b` with explicit shapes (D3)
8. Forward pass composing layers to `ŷ`, batch dimension, one **numeric** end-to-end pass (D3)
9. Logits / "raw scores" (last layer usually un-squashed) (D3→D4)
10. MSE for regression — formula, why squared, convex bowl in `ŷ` (D4)
11. Cross-entropy for classification (with softmax/sigmoid) + **why not MSE** (D4)
12. Loss is a **MEAN over the batch** (earns D6's unbiased-estimate claim) (D4)
13. Loss as a scalar `L(θ)` — the loss surface (D4→D5)
14. Gradient = vector of partials pointing uphill; `−∇L` descends (D5)
15. **Vanilla gradient descent named** as the baseline algorithm `w ← w − η·dL/dw` (D5)
16. Backprop = chain rule backward with a **real numeric multi-link worked chain** (D5)
17. Forward-cache: forward values needed for backward → training memory > inference (D5)
18. **Non-convexity / local vs global minimum** — we seek a good-enough basin (D5)
19. Training loop: forward→loss→backward→**zero_grad**→update, over epochs/batches (D6)
20. Batch vs mini-batch vs SGD — the batch gradient is a noisy estimate (D6)
21. **Random-init scale** intuition (fan-in; too big saturates, too small vanishes) (D6)
22. SGD update rule `θ ← θ − η∇L`, stated exactly (D7)
23. Momentum update rule (velocity), stated exactly + what it fixes (D7)
24. Adam update rule (`m`, `v`, bias correction, per-param step), stated exactly (D7)
25. Learning rate as step size — too-small / just-right / too-large (NaN) regimes (D8)
26. LR schedules + warmup at a conceptual level (D8)
27. Train / validation / test — the precise role of each (D9)
28. Overfitting/underfitting via the train–val gap `= val − train`, defined precisely (D9)
29. Early stopping + regularization (L2/dropout) as gap-closers, one-line mechanism (D9)

## 7. Should-cover concepts

- Perceptron + XOR limitation (D2 hook) · Leaky-ReLU / GELU one-line mentions (D2)
- MAE / Huber as MSE alternatives (D4) · **logits→prob + log-sum-exp numerical stability** (D4)
- MLE view (MSE⇐Gaussian, CE⇐Bernoulli/categorical) (D4)
- Computational-graph / reverse-mode autodiff framing (D5) · gradient checking (D5)
- Exploding/vanishing gradients through depth (**owned on D5** so D8's clipping has a referent)
- AdaGrad/RMSProp → Adam lineage (D7) · LR–batch linear-scaling rule (D8)
- Input feature scaling / ill-conditioning ↔ LR (D8) · data leakage (D9) · k-fold CV (D9)

## 8. Optional concepts

Bias-variance framing (name only) · double descent (awareness) · second-order methods (one line) ·
Nesterov · softmax temperature · universal approximation (name) · gradient clipping (mention on D8) ·
AdamW vs Adam decoupled decay · softmax outputs are **not** calibrated confidences (D4 one-liner).

## 9. Explicitly out of scope

Conv/RNN/attention/transformer architectures · general vectorized L-layer backprop derivation ·
batch/layer normalization mechanics · full regularization math (dropout-as-ensemble proof, L1 vs L2
derivations, augmentation) · hyperparameter-search machinery · distributed / mixed-precision /
kernel-level training · building a full autograd tape engine · probability/information-theory
foundations (entropy/KL/MLE derivations). Each is a later module; naming them as forward-pointers is
fine, teaching them here violates the "only use what was taught" rule.

---

## 10. Common misconceptions (the reframes each day must perform)

| Day | Wrong model the reader arrives with | The correction to land |
|---|---|---|
| 1 | "A neuron is a brain cell / organic magic." | It's a pure function `activation(w·x+b)` — dot, add, squash. Three lines of NumPy. |
| 1 | "Weights are config constants a human sets." | Weights are **discovered** state, random at start, updated every step. Config = human-set; weights = machine-set. |
| 1 | "The bias is a minor fudge factor, droppable." | Without `b` the boundary is forced through the origin (can rotate, never shift) — a real expressivity loss. |
| 2 | "The activation is optional decoration." | It's the entire reason depth works: `W₂(W₁x)=(W₂W₁)x` — no bend ⇒ any stack is one line. |
| 2 | "Activations are interchangeable; use sigmoid everywhere." | The choice shapes trainability: sigmoid/tanh saturate (flat tails), ReLU keeps slope 1 but can die. |
| 3 | "More layers = strictly better." | Depth trades capacity against optimization difficulty, overfitting, and compute. Sweet spot, not a free dial. |
| 3 | "Forward pass = the whole computation." | In training it also **caches** activations the backward pass needs; it's a batched matmul, not a scalar loop. |
| 4 | "Loss is the same as accuracy." | Loss is a smooth **surrogate** you optimize; accuracy is a discrete **metric** you report (zero gradient a.e.). |
| 4 | "Any loss works; MSE for everything." | Loss must match the output type: MSE⇒regression, CE⇒classification; MSE-on-classification stalls (saturated gradient). |
| 5 | "The gradient tells you the answer — jump there." | It's a **local** direction + slope. Take a small step, recompute, repeat. Iterative descent, not solving an equation. |
| 5 | "Backprop is a separate magic algorithm." | It **is** the chain rule, organized to reuse computation (reverse-mode autodiff). No new math. |
| 5 | "Gradient descent finds the GLOBAL minimum." | On a non-convex surface it finds *a* good-enough basin — and that is usually fine. |
| 6 | "Training is a run-once batch job that terminates." | It's an open-ended iterative loop with no natural exit; **you** decide when to stop; it's stochastic. |
| 6 | "Batch size is just a speed knob." | It changes gradient **noise** (the optimization), interacts with LR — not just throughput. |
| 7 | "The optimizer computes the gradients." | Backprop **computes** gradients; the optimizer **decides the update** (step size, momentum, per-param scale). |
| 7 | "Adam is strictly better than SGD." | No optimizer dominates: Adam converges fast/forgiving; well-tuned SGD+momentum often generalizes better. |
| 8 | "A bigger learning rate is always faster." | Past the sweet spot bigger = broken: overshoot, oscillate, diverge to NaN. Right-sized, not maximal. |
| 8 | "The learning rate is a fixed constant." | The ideal step size changes during training — hence schedules and warmup. |
| 9 | "The validation set is the test set." | Three roles: train fits θ; val tunes/selects (slowly leaks); test is touched **once** for an honest number. |
| 9 | "Overfitting means the model is broken." | It's the model working *exactly as optimized* — memorizing noise; the objective just isn't your real goal. |
| 9 | "Lower loss always means a better model." | Loss is a proxy; driving it to zero on train can *hurt* the real objective (generalization). |

---

## 11. Analogy map (DECISION: keep the current concrete analogies; add the structural fixes)

The design workflow proposed a single "mixing-console" theme; we **reject** forcing it (it would rewrite
every strong existing analogy and lower quality). Instead we keep the current per-day analogies, which
already chain as **skilled-task (D1–D4, D6) → terrain-descent (D5, D7, D8) → generalization (D9)**, and
adopt the design's *structural* insights.

| Day | Analogy (kept) | What it gets right | Where it breaks (state it) |
|---|---|---|---|
| 1 | A **judge scoring a contestant** | each input gets a weight = how much the judge cares; bias = a grumpy judge's baseline | a real neuron/brain is messier; one judge alone is weak — power is many, in layers |
| 2 | **Valve** (ReLU) + **dimmer** (sigmoid) | passes/blocks or smoothly squashes the summed signal | it's a smooth differentiable curve, not a hard on/off; a hard gate would block learning |
| 3 | An **assembly line** | layers = stations; forward pass = the finished product rolling off | a factory can't run backward; a net can (that's backprop, D5) |
| 4 | A **golf score** | loss = strokes from the hole; lower is better; 0 is perfect | golf just records; loss must be **smooth** so it can hand back a slope |
| 5 | A **hiker in the fog** on a hill (**the loss landscape**) | feel the steepest-down direction under your feet = the gradient; step, re-feel, repeat | the real surface has millions of dims and is **bumpy** (local minima); backprop feels all dims at once |
| 6 | **Practicing free throws** | shoot→see the miss→adjust→repeat; thousands of reps = training | a net has no innate "done"; it can diverge/oscillate if the step behavior is wrong |
| 7 | **Rolling a ball downhill, three ways** | SGD = careful footsteps; momentum = a ball that coasts over bumps; Adam = a smart ball | an optimizer only feels the slope once per step from one batch — velocity is a stored number, not physics |
| 8 | **Crossing a stream on stepping stones** | step too big → splash (diverge); too small → sunset first (crawl); just right → land squarely | the ground is curved and changes as you move — a safe step near the bank is too big at the start |
| 9 | **Studying for an exam** | train = practice problems; val = mock exam; test = the real exam sat once | rooms/exams differ knowably; splits differ only by random sampling — a peeked-at val set silently becomes another practice set |

**Structural fixes to script (the design's real contribution):**
- **The D4→D5 handoff** introduces the **loss landscape** explicitly: "we've been scoring a golf shot;
  now picture the loss as a whole landscape you descend." From D5 on, the mental picture is terrain.
- **One gradient picture only** on D5: the hiker feeling the slope (terrain). Do not also run a
  second, incompatible gradient metaphor.
- **Non-convexity** rides on the hiker analogy: the hill is bumpy — you reach *a* valley, not
  necessarily the deepest one.

---

## 12. Visual map (summary; full P0 table in `m02_visual_contract.md`)

Two surfaces already exist and are strong: the **playground fake-terminal** (step-by-step arithmetic)
and the **scroll-reveal `BUILD`** (annotated SVGs that assemble the idea). Days 5/7/8 additionally embed
the live `viz/gradient-descent.html` iframe. **P0 = keep these + fix the specific per-day defects**;
we do not mandate brand-new interactive iframes. Every day's P0 visual must show a **quantity changing**
(a loss falling, a curve bending, a step overshooting), not a static diagram. Day 9 is the one thin spot
(static SVG only) → its P0 is the two-curve diverge build; a live embed is **P1** (justified deferral).

## 13. Experiment / artifact map (summary; full table in `m02_artifact_contract.md`)

Every day ends with a Produce step whose artifact is a tiny **NumPy-only, deterministic** `experiment.py`
at `sessions/m02-the-neuron/<day-folder>/experiment.py`, run with
`python3 sessions/m02-the-neuron/<day-folder>/experiment.py`. `experiment.py` stays a **guided stub**
(the learner fills it via Produce Option A, or Option B triggers `/frontier-experiment-lab`). The
refactor's job: make each Produce prompt + acceptance criteria **exact and numerically consistent with
that day's playground** (fixing the current drifts on Days 7/8/9), and give each a one-line explain-back
into `log.md`.

## 14. Staff / Research-Engineer lens (one silent failure + one trade-off per day)

| Day | Silent failure mode (no crash) | Design trade-off |
|---|---|---|
| 1 | **Symmetric init** — all weights equal ⇒ identical gradients ⇒ a 512-unit layer acts as 1 | init **scale** (fan-in): too big saturates, too small vanishes (He vs Xavier) |
| 2 | **Dead ReLU** — unit driven negative for all inputs, gradient 0, never recovers | ReLU (cheap, can die) vs smooth GELU/SiLU (recoverable, costlier) |
| 3 | **Missing nonlinearity** between Linears (collapses to one line) + silent **broadcasting** | width vs depth at a fixed parameter budget |
| 4 | **Double softmax** / logits-vs-probabilities into the loss; MSE on classification | MSE vs cross-entropy (task fit + gradient scaling) |
| 5 | **Missing `zero_grad`** (gradients accumulate) + **vanishing/exploding** through depth | gradient clipping / normalize-or-not |
| 6 | Eval without **`model.eval()`/`no_grad`** (Dropout/BN misbehave) + loss ≠ reported metric | batch size ↔ gradient noise ↔ LR coupling |
| 7 | **AdamW bug** — L2 folded into the Adam gradient is silently rescaled per-param | SGD+momentum (generalizes) vs Adam/AdamW (fast, forgiving) |
| 8 | **LR too high** — silent high plateau / slow divergence (not always a clean NaN) | LR schedule (warmup+decay) vs constant |
| 9 | **Data leakage** — scaler fit on all data / dup rows / temporal bleed → inflated offline score | split ratio + strategy (random / stratified / grouped / k-fold) |

## 15. Interview readiness targets (a Staff MLE candidate should nail all six after this module)

1. Narrate the training loop as a data-flow graph (forward builds it → loss reduces to a scalar →
   backprop reverses to fill `.grad` → `optimizer.step()` consumes it → `zero_grad()` clears it) and
   name what silently breaks if any step is missing or misordered.
2. Reason about gradient flow: why sigmoid/tanh saturate & vanish, why deep unnormalized stacks
   explode, how ReLU dies, and which fixes (init scale, normalization, residuals, activation) address
   each — plus how to *instrument* for it (per-layer grad-norm histograms).
3. Choose & justify a loss from the task; explain why MSE-on-classification trains slowly; catch the
   double-softmax / logits-vs-probabilities bug on sight.
4. Compare optimizers + the LR as the top hyperparameter: SGD+momentum vs Adam/AdamW trade-offs, why
   Adam can generalize worse, the AdamW decoupling fix, and how LR-too-high diverges/plateaus silently.
5. Diagnose generalization from evidence: read train-vs-val curves to tell underfit / overfit / broken
   pipeline apart; explain why a leaky split looks great offline and collapses in production.
6. Distinguish the metric you **optimize** (differentiable loss) from the metric you **report**
   (accuracy/F1/AUC), and why watching only one hides real regressions.
