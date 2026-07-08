# Module 2 — Coverage Contract

> Governs what Module 2 (`sessions/m02-the-neuron`, 9 days + 2 gates) must teach. Derived
> source-free from `m02_first_principles_blueprint.md` and hardened by the completeness critic.
> **A concept is "covered" only when the named lesson lands it at the stated depth.** The four
> **promoted** items (init, named gradient descent, non-convexity, logits/log-sum-exp) and the
> five defect classes below are the delta this refactor must add on top of the current strong draft.
>
> Depth key: **Core** = full intuition + math + worked numbers + staff lens; **Support** = solid
> treatment, no separate math ladder; **Aware** = named with a one-line mechanism + forward-pointer.

---

## Must cover
*Concepts that make the module incomplete if missing.*

| Concept | Depth | Why it matters | Lesson | Visual? | Artifact? |
|---|---|---|---|---|---|
| Neuron as affine map + pre-activation `z = w·x + b` | Core | The atom of everything; a plain parametric function, not magic | D1 | ✅ build + playground | ✅ `neuron()` prints w·x, +b |
| Weights & bias are **learned state**; geometry (`w` orients, `b` shifts the boundary) | Core | Kills "weights = config"; "learning" = moving the boundary | D1 | ✅ build (weights label) | ✅ random init printed |
| Bias is **not** optional (without `b`, boundary forced through origin) | Support | Corrects "bias = fudge factor"; real expressivity point | D1 | ➖ prose/callout | ✅ criterion (b changes output) |
| **Symmetry-breaking initialization** (weights can't all start equal/zero) | Core (staff) | Load-bearing hole today; top interview Q; assumed by D5/D6 | D1 staff-lens | ➖ callout | ➖ explain-back |
| Why a nonlinearity is required: `linear ∘ linear` collapses (**scalar proof first**, then matrix) | Core | THE reason activations exist; the hinge from "a neuron" to "a useful network" | D2 | ✅ build (collapse→cure) | ✅ `(xW1)W2 == x(W1W2)` to 1e-10 |
| ReLU / sigmoid / tanh — formulas, ranges, ReLU as hidden default | Core | The activations they'll actually use and read in code | D2 | ✅ build + playground curves | ✅ table over an input grid |
| Saturation ("curve goes flat" → stalled learning) + **dead ReLU** counter-failure | Core (staff) | First real trade-off; previews the gradient story without gradient vocab | D2 | ✅ flat-tail curve | ✅ ReLU=0 on negatives; sigmoid'≈0.25 at 0 |
| Layer as `Z = xW + b` with explicit shapes | Core | Scale one neuron → a layer without hand-waving; it's a matmul | D3 | ✅ build (shape chain) | ✅ shape printed each stage |
| Forward pass composing layers → `ŷ`; batch dimension; **one numeric end-to-end pass** | Core | The `F(x)` whose error we measure; fixes "shapes-only" gap | D3 | ✅ build + playground | ✅ batched (4,·); row0 == single |
| Logits / "raw scores" (last layer usually un-squashed) | Support | Removes a dangling term; seams into D4 CE | D3→D4 | ➖ callout | ➖ |
| MSE for regression — formula, why squared, convex bowl in `ŷ` | Core | Half the objective; the scalar the gradient is taken of | D4 | ✅ loss-vs-prediction curve | ✅ MSE=0 at match, grows away |
| Cross-entropy for classification (+ softmax/sigmoid) & **why not MSE** | Core | The other half + explicit "don't use MSE for classification" | D4 | ✅ MSE↔CE toggle curve | ✅ CE confident-wrong ≫ hesitant-wrong |
| **MSE vs CE comparison table** (task fit, gradient behavior, output pairing) | Core | S8 requires a table when alternatives exist; today it's prose only | D4 | ✅ table | ➖ |
| Loss is a **MEAN over the batch** | Support | Earns D6's "unbiased estimate" claim; ties batch→gradient | D4 | ➖ | ✅ mean over N shown |
| Loss as a scalar `L(θ)` — the loss surface | Core | Bridge to D5; makes "gradient of the loss" the obvious next question | D4→D5 | ✅ bowl diagram | ➖ |
| Gradient = vector of partials pointing uphill; `−∇L` descends | Core | The core of learning; local direction + slope, not a destination | D5 | ✅ slope-on-bowl build | ✅ finite-diff ≈ analytic |
| **Vanilla gradient descent named** as the baseline (`w ← w − η·dL/dw`, repeated) | Core | Missing baseline before optimizers; makes D7 "what they ADD" | D5 (→ referenced D7) | ✅ roll-downhill playground | ✅ loop drives w→min |
| Backprop = chain rule backward, **real numeric multi-link chain** | Core | Mechanical heart; not the slogan — actual `dL/da·da/dz·dz/dw` with numbers | D5 | ✅ backward-reveal build | ✅ chain pieces printed |
| Forward-cache: forward values needed for backward → training mem > inference | Support | Explains autograd mechanics + a real memory cost | D5 | ➖ callout | ➖ explain-back |
| **Non-convexity / local vs global minimum** — seek a good-enough basin | Core (aware) | D7 momentum, D8 schedules, "we don't need the global optimum" all assume it | D5 | ✅ bumpy-hill note on build | ➖ |
| Training loop: forward→loss→backward→**zero_grad**→update, over epochs/batches | Core | The skeleton every run shares; give the engineer THE loop | D6 | ✅ loop cycle + falling loss | ✅ 4/5 labeled stages, loss↓ |
| Epoch / iteration / batch defined (worked count e.g. 100/25=4 steps) | Support | Makes "batch" concrete instead of verbal | D6 | ✅ batch boxes | ➖ |
| Batch vs mini-batch vs SGD — batch gradient is a **noisy unbiased estimate** | Core | Why training is noisy; sets up SGD; batch is a statistics knob | D6 | ✅ (playground) | ✅ (optional noise) |
| **Random-init scale** intuition (fan-in; too big saturates, too small vanishes) | Support (staff) | Second half of the init hole; ties D1 symmetry → D5 gradient scale | D6 | ➖ callout | ➖ |
| SGD update rule `θ ← θ − η∇L`, stated exactly | Core | The update link + baseline all others modify | D7 | ✅ optimizer paths | ✅ one-step w=1→1.8 |
| Momentum rule (velocity), stated exactly + what it fixes | Core | First fix to SGD; damps zig-zag, coasts flat spots | D7 | ✅ ball/momentum build | ✅ momentum reaches target faster |
| Adam rule (`m`,`v`, bias correction, per-param step; β₁=.9, β₂=.999, ε=1e-8) | Core | The default they'll reach for; why it "just works" | D7 | ✅ Adam build step | ✅ bias-corrected step shown |
| LR as step size — too-small / just-right / too-large (→NaN) regimes | Core | The single most-tuned hyperparameter | D8 | ✅ LR slider + loss curve | ✅ 3 rates: crawl/converge/diverge |
| LR schedules + warmup (conceptual) | Support | Standard in every real run; LR is a policy over time | D8 | ✅ warmup+decay shape | ➖ |
| Train / validation / test — precise role of each | Core | The generalization link; confusing them is a classic bug | D9 | ✅ 3-box split build | ✅ disjoint splits, sizes sum |
| Overfitting/underfitting via train–val gap `= val − train` | Core | The primary diagnostic read off every curve | D9 | ✅ two-curve diverge | ✅ train↓ while val↑ visible |
| Early stopping + regularization (L2/dropout) as gap-closers (one-line mechanism) | Core (aware) | Makes D9 actionable; L2 = smaller weights→smoother, dropout ≈ ensemble | D9 | ✅ early-stop marker | ✅ best-val checkpoint ≠ last |

## Should cover
*Concepts that deepen understanding.*

| Concept | Depth | Why it matters | Lesson | Visual? | Artifact? |
|---|---|---|---|---|---|
| Perceptron + XOR limitation (historical hook) | Aware | Motivates why one linear neuron isn't enough | D2 | ➖ | ➖ |
| Leaky-ReLU / GELU one-line mentions (GELU in transformers) | Aware | They'll see these in modern code | D2 | ➖ | ➖ |
| **Logits→prob + log-sum-exp numerical stability** (why the loss takes logits) | Support | Concrete staff-interview trap; today only optional | D4 | ➖ callout | ✅ CE clips/offsets, no log(0) |
| MLE view (MSE⇐Gaussian, CE⇐Bernoulli/categorical) | Aware | Shows the losses aren't arbitrary | D4 | ➖ | ➖ |
| MAE / Huber as MSE alternatives (outlier robustness) | Aware | Rounds out the loss picture | D4 | ➖ | ➖ |
| Computational-graph / reverse-mode autodiff framing | Aware | Connects the worked chain rule to PyTorch/JAX | D5 | ➖ | ➖ |
| Gradient checking (numerical vs analytic) | Support | Engineer-friendly way to trust a hand-derived gradient | D5 | ➖ | ✅ finite-diff check (already the D5 artifact core) |
| **Exploding / vanishing gradients through depth** | Support | D8's clipping needs this referent; ties D2 saturation → D5 chain | D5 | ➖ callout | ➖ |
| AdaGrad / RMSProp → Adam lineage | Aware | Shows Adam = RMSProp + momentum, not magic | D7 | ➖ | ➖ |
| LR–batch-size linear-scaling rule | Aware | Knobs aren't independent; connects D6↔D8 | D8 | ➖ | ➖ |
| Input feature scaling / ill-conditioning ↔ LR | Support | Common real reason training is slow/diverges | D8 | ➖ callout | ➖ |
| Data leakage (scaler fit before split, dup rows, temporal bleed) | Support (staff) | The most common real way splits go wrong | D9 | ➖ | ✅ disjoint-index criterion |
| K-fold cross-validation | Aware | For small datasets a single val split is noisy | D9 | ➖ | ➖ |

## Optional
*Nice-to-have; include only if it doesn't crowd a Core concept.*

- Bias-variance decomposition (name only, defer algebra) — D9
- Double descent as a modern nuance to the U-curve (awareness) — D9
- Second-order optimization (Newton/L-BFGS) one-line contrast — D7
- Nesterov accelerated gradient (look-ahead momentum) — D7
- Softmax temperature — D4
- Universal approximation theorem (name + intuition) — D3/D2
- Gradient clipping (mention alongside D8 divergence) — D8
- AdamW vs Adam (decoupled weight decay) — D7 (already the D7 staff-lens; keep as awareness)
- Softmax outputs are **not** calibrated confidences (one-line note) — D4

## Explicitly out of scope
*Avoid for now — later modules own these; a forward-pointer is fine, teaching them here is not.*

| Concept | Why excluded |
|---|---|
| Conv / RNN / attention / transformer architectures | This module teaches the generic learning mechanism on plain FC layers; architectures are their own modules (M3+) |
| General vectorized L-layer backprop derivation (per-layer dW/db Jacobians) | Heavy linear algebra drowns the intuition; a short single-path worked chain suffices; autograd makes the general form unnecessary here |
| Batch / layer normalization mechanics | Fixes for training-dynamics problems that presuppose the whole loop is understood |
| Full regularization math (dropout-as-ensemble proof, L1 vs L2 derivation, augmentation zoo) | D9 names them conceptually; full mechanics deserve their own treatment |
| Hyperparameter-search machinery (grid/random/Bayesian, scheduler frameworks) | MLOps/tooling layered on top of understanding WHICH knobs exist |
| Distributed / mixed-precision / GPU-kernel training | Systems concerns downstream of a single-device loop |
| Building a full autograd tape/graph engine (micrograd-style) | Excellent but separate deep-dive; not needed to understand how a net learns |
| Probability / information-theory foundations (entropy/KL/MLE derivations) | CE is taught operationally; the full framework is a prerequisites-module concern |

---

## Defect classes this refactor must also close (verified in the audit)

These are not new concepts — they are correctness/consistency fixes required for "notebook-quality":

1. **Module-label identity split (P0, systemic).** Days 4–9 render "Module 3" / "Module 2-3"; `review.html`
   renders "Module 3". Unify **all** learner-facing labels (kicker, `<title>`, `brand-sub`, `nav-group`,
   `.fin` banner, Produce prompt headers) to **"Module 2 · Day 1–9"**. Fix `review.html` "Ready for
   Module 4?"→"Ready for Module 3? (Attention)", "(M1–M3)"→"(M1–M2)", "M4 · Attention"→"M3 · Attention",
   and review-part-a's forward copy ("Gate Before Module 3" → the training half of Module 2).
   **Do NOT touch any `data-quest-id`.** Fix intra-module cross-refs that call an earlier day "M2/M3".
2. **Numeric self-consistency.** Day 6 loss `0.747`→`0.746`; Day 9 "best model near step 100" →
   **step 200** (the actual val minimum); Day 5 forward-pass cost "≈2–3x" vs "≈3x" → pick **~3×**.
3. **Playground ↔ Produce agreement.** Day 7 momentum uses a hypothetical continuation gradient (−12.8)
   the real 15-step loop never produces → use the **real** continuation (`g=−9.6`, `v=−24.0`). Day 8
   align step counts. Day 9 make the playground toy and the Produce toy the **same framing**.
4. **Worked-example spine.** Day 2 must carry **one** anchor matrix pair through playground/math-ladder/
   produce (today three drift). Day 1 unify `np.maximum` across playground + Produce.
5. **No premature close.** Day 6 must drop "Foundations done / completes the story" — Days 7–9 remain.

## Coverage completeness check (all must be ✅ before QA sign-off)

- [ ] All 29 must-cover concepts land at stated depth on their named lesson
- [ ] The 4 promoted items are visibly owned: init (D1+D6), named GD (D5), non-convexity (D5), logits/LSE (D4)
- [ ] Each day carries the full Coach Layer: Jargon tooltips · Math Ladder · one silent-failure + one trade-off · interview answer · diagnostic quiz Q
- [ ] MSE↔CE table (D4) and batch-size table (D6) exist as **tables**, not prose
- [ ] All 5 defect classes closed
- [ ] Out-of-scope items appear only as forward-pointers, never taught
