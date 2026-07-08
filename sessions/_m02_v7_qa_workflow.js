export const meta = {
  name: 'm02-v7-qa',
  description: 'v7 QA pass over sessions/m02-the-neuron (report-only): per-day coverage/coach/visual/artifact/math gates + adversarial verify, plus review-pages, legacy-skill sweep, JS/nav, and cross-day numeric consistency.',
  phases: [
    { title: 'PerDayQA', detail: 'comprehensive v7 gate check per lesson (9 days)' },
    { title: 'Verify', detail: 'adversarially re-read cited lines; confirm/refute/adjust + catch misses' },
    { title: 'CrossCutting', detail: 'review pages, legacy-skill sweep, JS/nav/completion, cross-day numeric consistency' },
  ],
}

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------
const MOD = 'sessions/m02-the-neuron'
const CONTRACTS = [
  'sessions/m02_coverage_contract.md',
  'sessions/m02_artifact_contract.md',
  'sessions/m02_visual_contract.md',
  'sessions/m02_first_principles_blueprint.md',
]

const READ_ONLY = `HARD RULES (do not violate):
- READ-ONLY. Never call Edit/Write/NotebookEdit on any file. This is a report-only QA pass.
- Only read files under ${MOD}/ and the four contract files (${CONTRACTS.join(', ')}). Do NOT read old notebooks anywhere (e.g. 00-neural-networks/). Do NOT read other modules.
- Every finding MUST cite exact evidence: file path + line number + a short verbatim quote from the file. No finding without a locatable quote.
- If something the contract requires is genuinely present and correct, say PASS — do not invent problems. If genuinely missing/wrong, say so plainly. Faithful reporting only.`

const V7_GATES = `v7 QA GATE DEFINITIONS (classify every finding P0/P1/P2 using these):

P0 (blocker): missing must-cover concept; coverage not traceable to lesson text; formula-first opening (formula/definition before why-care + intuition in the first ~20%); no pain point / no intuition before math; missing brain/human analogy + caveat in a neural-network foundation lesson; incorrect math; a P0 visual missing or unjustified; a failure mode taught with NO runnable evidence or required artifact; a quantitative visual with no derived/numeric readout; simulated terminal not clearly labeled as simulated; a stale run command; a Produce path mismatch; a legacy skill reference to an uninstalled skill; broken JS / navigation / completion; a duplicated paragraph; a truncated title or finale.

P1 (fix before merge if easy): weak/one-line analogy; missing Reader-objection Q&A; dry/docs-like tone; missing Jargon Buster; weak acceptance criteria; unclear visual caption; no on-panel numeric readout; a should-cover comparison-family with no comparison table.

P2 (polish): markdown wrapping; cosmetic formatting; wording polish.

COACH VOICE GATE — mark P0 if: first ~20% opens with formula/definition before why-care and intuition; no pain point named; analogy is a single sentence only; no tiny worked example before formulas; a neural-network foundation lesson lacks a brain/human analogy + its caveat; the lesson reads like docs not a warm coach. Mark P1 if: tone correct but dry; interview answer sounds memorized.

COACH LAYER 6-PACK (coverage_contract line 123) — each day must carry ALL of: (1) Jargon tooltips (.term spans), (2) a Math Ladder (words -> formula -> worked numbers), (3) one silent-failure callout + one trade-off callout (staff lens), (4) an interview-grade answer, (5) a diagnostic quiz question. A missing member of the 6-pack is at least P1 (P0 if it is the required brain analogy, the pain point, or a must-cover staff concept).

VISUAL/EVIDENCE GATE — mark P0 if: a P0 visual is missing; a quantitative visual lacks a derived number; a failure mode has no demo/evidence/artifact; the simulated terminal is not clearly labeled as simulated (a "playground" fake-terminal must read as simulated, not as a live Python REPL); the Produce artifact cannot reproduce the playground output (numbers must match).

ARTIFACT GATE — check the Produce step against the artifact contract row: exact path, exact run command, expected output, acceptance criteria (concrete numbers/behaviors), and an explain-back question written to log.md. A number in the Produce acceptance criteria that disagrees with the day's playground or Math Ladder is a P0 (anchor-spine break).`

const LEGACY_NAMES = [
  'frontier-experiment-lab', 'frontier-session-coach', 'frontier-math-unfogger',
  'frontier-lesson-humanizer', 'frontier-d3-visual-lab', 'frontier-visual-auditor',
  'frontier-artifact-reviewer', 'frontier-curriculum-rollout', 'frontier-qa-gate',
  'frontier-visual-lab', 'frontier-lesson-coach', 'frontier-visual-builder',
]

// ---------------------------------------------------------------------------
// Per-day briefs (extracted from the four contracts)
// ---------------------------------------------------------------------------
const DAYS = [
  {
    id: 'D1', folder: 'day-01-single-neuron', qid: 'wf2-d01-neuron',
    goal: 'A neuron is a tiny tunable function z = w·x + b.',
    mustCover: [
      'Neuron as affine map + pre-activation z=w·x+b [Core] — plain parametric function, not magic',
      'Weights & bias are LEARNED state; geometry (w orients, b shifts the boundary) [Core]',
      'Bias is NOT optional (without b, boundary forced through origin) [Support]',
      'Symmetry-breaking initialization (weights can\'t all start equal/zero) [Core, staff]',
    ],
    artifact: `Path folder: day-01-single-neuron. Run: python3 ${MOD}/day-01-single-neuron/experiment.py
neuron(x,w,b)=relu(np.dot(w,x)+b) using np.maximum. Anchor x=[2,3], w=[0.5,-1], b=1 -> weighted-sum -2, +bias -1, output 0.0; positive case w=[1,1], b=0 -> 5.0; same input twice -> identical (determinism).
Acceptance: (1) 3 labeled steps (weighted sum, +bias, output); (2) anchor outputs 0.0, positive case positive; (3) same input twice identical; (4) uses np.maximum (not built-in max) — consistent with the playground.
Explain-back (-> log.md): Why is a neuron just a weighted sum + bias, and what does the bias let it do that weights alone cannot?`,
    visual: 'P0: BUILD assembles w·x -> +b -> z term-by-term with running numbers; Playground 3 steps (weighted sum -> +bias -> z) for anchor x=[2,3]. Prevents "neuron=black box/brain cell" and "bias is cosmetic". Known fix to verify: the width 2->3 jump at the layer step must be noted/labeled.',
    analogy: 'A judge scoring a contestant (each input gets a weight = how much the judge cares; bias = a grumpy judge\'s baseline). Must state where it breaks: a real neuron/brain is messier; one judge alone is weak — power is many, in layers.',
    misconceptions: '"neuron = brain cell/organic magic" -> pure function activation(w·x+b); "weights = config a human sets" -> discovered state, random then updated; "bias = minor droppable fudge factor" -> without b the boundary is forced through the origin.',
    staffLens: 'Silent failure: symmetric init (all weights equal -> identical gradients -> 512-unit layer acts as 1). Trade-off: init SCALE (fan-in; too big saturates, too small vanishes; He vs Xavier).',
    defectChecks: 'np.maximum used in BOTH playground and Produce; width 2->3 note present at the layer build step; all learner-facing labels say "Module 2" (not "Module 3"/"Module 2-3"). This is a neural-network FOUNDATION lesson -> the brain/human analogy + caveat is REQUIRED (P0 if missing).',
    isFoundation: true,
  },
  {
    id: 'D2', folder: 'day-02-activations', qid: 'wf2-d02-activations',
    goal: 'Why a neuron needs a bend, or a whole stack collapses to one line.',
    mustCover: [
      'Why a nonlinearity is required: linear∘linear collapses (SCALAR proof first, then matrix) [Core]',
      'ReLU / sigmoid / tanh — formulas, ranges, ReLU as hidden default [Core]',
      'Saturation ("curve goes flat" -> stalled learning) + dead-ReLU counter-failure [Core, staff]',
    ],
    shouldCover: ['Perceptron + XOR limitation (historical hook) [Aware]', 'Leaky-ReLU / GELU one-line mentions (GELU in transformers) [Aware]'],
    artifact: `Path folder: day-02-activations. Run: python3 ${MOD}/day-02-activations/experiment.py
relu/sigmoid/tanh over a fixed grid (np.linspace(-6,6,13)) as a table. Then the ONE anchor matrix pair (same as playground): (x@W1)@W2 == x@(W1@W2) to 1e-10 (collapse), then relu(x@W1)@W2 differs (cure).
Acceptance: (1) table: ReLU 0 on negatives / x on positives, sigmoid in (0,1), tanh in (-1,1); (2) stacked-linear == single-combined-linear within 1e-10; (3) inserting ReLU changes the output (nonzero diff); (4) prints sigmoid'(0)≈0.25 and notes ReLU grad is 0 for negatives.
Explain-back: Why does stacking linear layers with no activation give no more power than one linear layer?`,
    visual: 'P0: BUILD ReLU/sigmoid curves -> collapse (linear∘linear = one line) -> cure (insert ReLU) -> many-layer bent shape; Playground relu/sigmoid on a vector + the collapse demo. Prevents "activation is optional decoration" and "depth alone buys power". Fix to verify: ONE anchor matrix pair through playground/math-ladder/produce; SCALAR collapse shown BEFORE the matrix collapse.',
    analogy: 'Valve (ReLU) + dimmer (sigmoid). Breaks: it is a smooth differentiable curve, not a hard on/off; a hard gate would block learning.',
    misconceptions: '"activation is optional decoration" -> it is the entire reason depth works (W2(W1x)=(W2W1)x); "activations interchangeable, use sigmoid everywhere" -> sigmoid/tanh saturate (flat tails), ReLU keeps slope 1 but can die.',
    staffLens: 'Silent failure: dead ReLU (unit driven negative for all inputs, gradient 0, never recovers). Trade-off: ReLU (cheap, can die) vs smooth GELU/SiLU (recoverable, costlier).',
    defectChecks: 'ONE anchor matrix pair used EVERYWHERE (playground == math-ladder == produce; the audit flagged three drifting versions); scalar collapse shown before the matrix version; module labels say "Module 2".',
    isFoundation: false,
  },
  {
    id: 'D3', folder: 'day-03-layers-forward-pass', qid: 'wf2-d03-forward',
    goal: 'Compose neurons into layers and run the forward pass to a prediction ŷ.',
    mustCover: [
      'Layer as Z = xW + b with explicit shapes [Core]',
      'Forward pass composing layers -> ŷ; batch dimension; ONE numeric end-to-end pass [Core]',
      'Logits / "raw scores" (last layer usually un-squashed) [Support, D3->D4]',
    ],
    artifact: `Path folder: day-03-layers-forward-pass. Run: python3 ${MOD}/day-03-layers-forward-pass/experiment.py
2-layer MLP: x -> relu(x@W1+b1)=h -> h@W2+b2 = ŷ with fixed small matrices; prints shape AND values each stage. Recomputes one hidden unit by hand to match the vectorized entry. Runs a batch of 4.
Acceptance: (1) shape printed each stage and they chain (e.g. (2,)->(3,)->(1,)); (2) one hand-computed hidden unit matches W1@x+b1 to 1e-10; (3) batched run prints (4, out) and row 0 == single-input result; (4) post-ReLU hidden values all ≥ 0.
Explain-back: How does a layer turn one matmul into many neurons at once, and why is the batch just an extra dimension?`,
    visual: 'P0: BUILD 2-layer net lighting up the forward pass with shape labels at each hop; Playground push one input through, printing shape AND one numeric value at each stage ((1,3)->(1,4)->(1,2)). Prevents "forward pass is a scalar for-loop" and "shapes just work out". Fix to verify: add ONE concrete NUMERIC forward pass (not shapes-only).',
    analogy: 'An assembly line (layers = stations; forward pass = finished product rolling off). Breaks: a factory can\'t run backward; a net can (backprop, D5).',
    misconceptions: '"more layers = strictly better" -> depth trades capacity vs optimization difficulty/overfitting/compute; "forward pass = the whole computation" -> in training it caches activations the backward pass needs; batched matmul not scalar loop.',
    staffLens: 'Silent failure: missing nonlinearity between Linears (collapses to one line) + silent broadcasting. Trade-off: width vs depth at a fixed parameter budget.',
    defectChecks: 'ONE concrete numeric forward pass present (not shapes-only); the D3 Produce Option-B prompt has NO authoring artifact left in it (the audit flagged text like "W2 (4,4)... use W2 (4,2)"); logits/"raw scores" callout present; module labels say "Module 2".',
    isFoundation: false,
  },
  {
    id: 'D4', folder: 'day-04-loss', qid: 'wf3-d01-loss',
    goal: 'Turn error into one scalar; pick MSE (regression) vs cross-entropy (classification).',
    mustCover: [
      'MSE for regression — formula, why squared, convex bowl in ŷ [Core]',
      'Cross-entropy for classification (+ softmax/sigmoid) & WHY NOT MSE [Core]',
      'MSE vs CE COMPARISON TABLE (task fit, gradient behavior, output pairing) [Core] — must be a TABLE, not prose',
      'Loss is a MEAN over the batch [Support]',
      'Loss as a scalar L(θ) — the loss surface [Core, D4->D5]',
    ],
    shouldCover: ['Logits->prob + log-sum-exp numerical stability (why the loss takes logits) [Support]', 'MLE view (MSE⇐Gaussian, CE⇐Bernoulli/categorical) [Aware]', 'MAE / Huber as MSE alternatives [Aware]'],
    artifact: `Path folder: day-04-loss. Run: python3 ${MOD}/day-04-loss/experiment.py
mse(pred,target) and cross_entropy(p_correct)=-np.log(p_correct) (clipped to avoid log(0)). MSE close vs far; CE for p_true∈[0.3,0.6,0.9] -> ≈1.204, 0.511, 0.105 (SAME ORDER as playground). confident-wrong ≫ hesitant-wrong.
Acceptance: (1) MSE==0 (within 1e-12) at match, larger otherwise; (2) CE for p=[0.3,0.6,0.9] prints ≈1.204/0.511/0.105; (3) confident-wrong CE ≫ hesitant-wrong CE; (4) inputs clipped/offset so no log(0) NaN/inf.
Explain-back: What does a loss turn mistakes into, and why must it be smallest when the prediction is exactly right?`,
    visual: 'P0: BUILD prediction->error->square->mean->MSE, then CE -log(p), then a falling-loss curve; Playground MSE + CE + "loss falls" sweep; MSE↔CE comparison TABLE. Prevents "loss = accuracy" and "any loss works / MSE for everything". Fix to verify: the MSE↔CE table exists as a TABLE + MSE math ladder; CE ordering matches the criteria (1.204/0.511/0.105).',
    analogy: 'A golf score (loss = strokes from the hole; 0 perfect). Breaks: golf just records; loss must be SMOOTH so it can hand back a slope.',
    misconceptions: '"loss = accuracy" -> loss is a smooth surrogate you optimize, accuracy a discrete metric you report; "any loss works / MSE for everything" -> MSE⇒regression, CE⇒classification; MSE-on-classification stalls (saturated gradient).',
    staffLens: 'Silent failure: double softmax / logits-vs-probabilities into the loss; MSE on classification. Trade-off: MSE vs cross-entropy (task fit + gradient scaling).',
    defectChecks: 'MSE↔CE comparison is a real TABLE (not prose); CE ordering 1.204/0.511/0.105 consistent playground↔produce; log-sum-exp / logits->prob stability callout present; module labels say "Module 2".',
    isFoundation: false,
  },
  {
    id: 'D5', folder: 'day-05-gradients-backprop', qid: 'wf3-d02-backprop',
    goal: 'Compute ∇L for every weight by running the chain rule backward.',
    mustCover: [
      'Gradient = vector of partials pointing uphill; −∇L descends [Core]',
      'VANILLA GRADIENT DESCENT NAMED as the baseline (w ← w − η·dL/dw, repeated) [Core]',
      'Backprop = chain rule backward, REAL NUMERIC multi-link chain (actual dL/da·da/dz·dz/dw with numbers) [Core]',
      'Forward-cache: forward values needed for backward -> training mem > inference [Support]',
      'Non-convexity / local vs global minimum — seek a good-enough basin [Core, aware]',
    ],
    shouldCover: ['Computational-graph / reverse-mode autodiff framing [Aware]', 'Gradient checking (numerical vs analytic) [Support]', 'Exploding / vanishing gradients through depth (OWNED here so D8 clipping has a referent) [Support]'],
    artifact: `Path folder: day-05-gradients-backprop. Run: python3 ${MOD}/day-05-gradients-backprop/experiment.py
(a) named GD baseline: loss=w**2, loop w = w - 0.1*(2*w) from w=3, print w,loss falling toward 0. (b) chain-rule check on loss=(sigmoid(w*x+b)-y)**2: derive dL/dw, dL/db analytically, print the chain pieces, verify vs finite differences.
Acceptance: (1) GD loop loss strictly decreasing, w->0; (2) analytic vs numerical gradient for w and b agree to < 1e-6; (3) prints intermediate chain pieces (dL/dpred, dpred/dz, dz/dw); (4) fixed eps + fixed inputs -> reproducible.
Explain-back: Why is backprop just the chain rule backward, and what does the finite-difference check prove?`,
    visual: 'P0: BUILD forward, then reveal the backward pass arrow-by-arrow with local slopes multiplying; LIVE gradient-descent.html embed (lr slider + Step/Run). Prevents "gradient = the answer, jump there", "backprop is separate magic", and "GD finds the global min". Fix to verify: a NUMERIC multi-link chain computation is shown; a "bumpy hill / local minima" note rides on the descent; the gradient-descent.html iframe renders and auto-sizes (no stuck 520px blank box).',
    analogy: 'A hiker in the fog on a hill (the loss landscape): feel the steepest-down direction = the gradient; step, re-feel, repeat. Breaks: the real surface has millions of dims and is bumpy (local minima); backprop feels all dims at once.',
    misconceptions: '"gradient tells you the answer — jump there" -> local direction+slope, iterative; "backprop is separate magic" -> it IS the chain rule (reverse-mode autodiff); "GD finds the GLOBAL minimum" -> a good-enough basin.',
    staffLens: 'Silent failure: missing zero_grad (gradients accumulate) + vanishing/exploding through depth. Trade-off: gradient clipping / normalize-or-not.',
    defectChecks: 'REAL numeric multi-link chain present; vanilla GD explicitly NAMED as the baseline; non-convexity/local-minima note present; forward-pass cost stated as ~3x (audit flagged "2-3x" vs "3x" inconsistency — should be ~3×); any cross-ref that calls the forward pass "M2" is reworded to "Day 3 (earlier this module)"; module labels say "Module 2".',
    isFoundation: false,
  },
  {
    id: 'D6', folder: 'day-06-training-loop', qid: 'wf3-d03-training-loop',
    goal: 'Assemble the repeating loop: forward->loss->backward->zero_grad->update, over batches/epochs.',
    mustCover: [
      'Training loop: forward->loss->backward->ZERO_GRAD->update, over epochs/batches [Core]',
      'Epoch / iteration / batch defined (worked count e.g. 100/25=4 steps) [Support]',
      'Batch vs mini-batch vs SGD — batch gradient is a NOISY UNBIASED estimate [Core]',
      'Random-init SCALE intuition (fan-in; too big saturates, too small vanishes) [Support, staff]',
    ],
    artifact: `Path folder: day-06-training-loop. Run: python3 ${MOD}/day-06-training-loop/experiment.py
Full loop wiring D1–D5: fit pred=w*x to x=2, y=6 (ideal w=3) from w=1.0. Each step forward->loss->grad->(zero accumulator)->update w -= lr*grad. Prints step,w,loss for ~20 steps; w->3, loss->0.
Acceptance: (1) loss step0 ≫ final (final < ~0.01); (2) |w-3| < 0.1; (3) prints labeled stages (forward, loss, backward, update) at least once; (4) deterministic: two runs identical final loss.
Explain-back: What are the four (with zeroing, five) repeating steps of a training loop, and which one actually changes the weights?`,
    visual: 'P0: Playground/BUILD — each click runs one full iteration and drops a point onto a falling loss-vs-step curve; loop-cycle SVG + batch boxes; batch-size trade-off TABLE. Prevents "training is one call that solves it" and "batch = pure speed knob". Fixes to verify: the loss value reads 0.746 (audit: 0.747 was wrong); zero_grad IS in the loop skeleton (4/5 labeled stages); the batch-size trade-off exists as a TABLE; the premature-close line ("Foundations done" / "completes the story") is REMOVED (Days 7–9 still remain).',
    analogy: 'Practicing free throws (shoot->see miss->adjust->repeat). Breaks: a net has no innate "done"; it can diverge/oscillate.',
    misconceptions: '"training is a run-once batch job that terminates" -> open-ended iterative loop, no natural exit, you decide when to stop, stochastic; "batch size is just a speed knob" -> it changes gradient noise, interacts with LR.',
    staffLens: 'Silent failure: eval without model.eval()/no_grad (Dropout/BN misbehave) + loss ≠ reported metric. Trade-off: batch size ↔ gradient noise ↔ LR coupling.',
    defectChecks: 'Loss reads 0.746 (NOT 0.747); zero_grad present in skeleton; batch-size trade-off is a TABLE (not prose); NO premature-close language ("Foundations done"/"completes the story"); module labels say "Module 2".',
    isFoundation: false,
  },
  {
    id: 'D7', folder: 'day-07-optimizers', qid: 'wf3-d04-optimizers',
    goal: 'Exact SGD, momentum, Adam — each as "what it ADDS" to the GD baseline.',
    mustCover: [
      'SGD update rule θ ← θ − η∇L, stated exactly [Core]',
      'Momentum rule (velocity) stated exactly + what it fixes [Core]',
      'Adam rule (m, v, bias correction, per-param step; β₁=.9, β₂=.999, ε=1e-8) [Core]',
    ],
    shouldCover: ['AdaGrad / RMSProp -> Adam lineage (Adam = RMSProp + momentum) [Aware]'],
    artifact: `Path folder: day-07-optimizers. Run: python3 ${MOD}/day-07-optimizers/experiment.py
Plain SGD vs SGD+momentum (β=0.9, v=0) on the SAME problem (pred=w*x, x=2, y=6, w=1.0, lr=0.05), two 15-step loops printing step,w,loss. Momentum reaches w≈3 in fewer steps. First step both give w=1.8; momentum's real continuation uses g=−9.6 at w=1.8.
Acceptance: (1) both start w=1.0/same seed printed once; (2) first step of each == w=1.8; (3) momentum reaches target loss in fewer steps than plain SGD; (4) both converge.
Explain-back: What does momentum fix that plain SGD struggles with, and what extra thing does Adam adapt per-parameter?`,
    visual: 'P0: BUILD same loop, only the UPDATE changes -> SGD step -> momentum velocity -> Adam per-param; LIVE gradient-descent.html embed; Playground SGD vs momentum on the same weight. Prevents "all optimizers are the same" and "Adam is strictly better". Fixes to verify: momentum step-2 uses the REAL continuation (g=−9.6, v=−24.0) that the 15-step loop actually produces (audit: a hypothetical −12.8 was wrong); Adam gives β₂=.999 and ε=1e-8 defaults; the gradient-descent.html iframe auto-sizes.',
    analogy: 'Rolling a ball downhill, three ways (SGD = careful footsteps; momentum = ball that coasts over bumps; Adam = a smart ball). Breaks: an optimizer feels the slope once per step from one batch — velocity is a stored number, not physics.',
    misconceptions: '"the optimizer computes the gradients" -> backprop computes gradients, the optimizer decides the update; "Adam is strictly better than SGD" -> no optimizer dominates; well-tuned SGD+momentum often generalizes better.',
    staffLens: 'Silent failure: AdamW bug (L2 folded into the Adam gradient is silently rescaled per-param). Trade-off: SGD+momentum (generalizes) vs Adam/AdamW (fast, forgiving).',
    defectChecks: 'Momentum step-2 continuation is g=−9.6, v=−24.0 (NOT −12.8); first step of each == w=1.8; Adam β₂=.999 & ε=1e-8 present; module labels say "Module 2".',
    isFoundation: false,
  },
  {
    id: 'D8', folder: 'day-08-learning-rate', qid: 'wf3-d05-lr',
    goal: 'The step size: crawl vs converge vs diverge, plus schedules.',
    mustCover: [
      'LR as step size — too-small / just-right / too-large (->NaN) regimes [Core]',
      'LR schedules + warmup (conceptual) [Support]',
    ],
    shouldCover: ['LR–batch-size linear-scaling rule [Aware]', 'Input feature scaling / ill-conditioning ↔ LR [Support]'],
    artifact: `Path folder: day-08-learning-rate. Run: python3 ${MOD}/day-08-learning-rate/experiment.py
Same loop, same init (w=1.0), three learning rates 0.005 / 0.1 / 0.6, each an N-step loop (N matches the lesson's playground), printing step,w,loss and labeling each run crawl/converge/diverge. Comment the |1 − lr·c| < 1 rule with c=8 -> tipping point lr = 2/c = 0.25.
Acceptance: (1) lr=0.005 ends far from converged; (2) lr=0.1 reaches w≈3/low loss; (3) lr=0.6 diverges (loss grows/oscillates) and is labeled; (4) same seed + init printed once; step count == the playground's.
Explain-back: What happens when the LR is too small vs too large, and why tune it first?`,
    visual: 'P0: BUILD lr scales every step -> too-small crawls -> just-right -> too-big explodes -> warmup/decay shapes; LIVE gradient-descent.html embed with the lr slider. Prevents "bigger LR is always faster" and "LR is a fixed constant". Fixes to verify: playground and Produce step counts are ALIGNED (one count); a symbol-disambiguation note distinguishes c (curvature) from v (velocity/value); the iframe auto-sizes.',
    analogy: 'Crossing a stream on stepping stones (too big -> splash/diverge; too small -> sunset first/crawl; just right -> land squarely). Breaks: the ground is curved and changes as you move — a safe step near the bank is too big at the start.',
    misconceptions: '"a bigger learning rate is always faster" -> past the sweet spot bigger = broken (overshoot/oscillate/diverge to NaN); "the learning rate is a fixed constant" -> ideal step size changes -> schedules & warmup.',
    staffLens: 'Silent failure: LR too high — silent high plateau / slow divergence (not always a clean NaN). Trade-off: LR schedule (warmup+decay) vs constant.',
    defectChecks: 'Playground & Produce step counts aligned; c-vs-v symbol disambiguation note present; three rates labeled crawl/converge/diverge; module labels say "Module 2".',
    isFoundation: false,
  },
  {
    id: 'D9', folder: 'day-09-train-val-test', qid: 'wf3-d06-split',
    goal: 'Split data; define the train–val gap; detect & fix overfitting.',
    mustCover: [
      'Train / validation / test — precise role of each [Core]',
      'Overfitting/underfitting via train–val gap = val − train [Core]',
      'Early stopping + regularization (L2/dropout) as gap-closers (one-line mechanism) [Core, aware]',
    ],
    shouldCover: ['Data leakage (scaler fit before split, dup rows, temporal bleed) [Support, staff]', 'K-fold cross-validation [Aware]'],
    artifact: `Path folder: day-09-train-val-test. Run: python3 ${MOD}/day-09-train-val-test/experiment.py
MATCH the playground framing: synthetic noisy data, split train/val/test (disjoint), fit a model flexible enough to overfit; print train vs val loss at each checkpoint showing train↓ while val↓-then-↑; report the best-val checkpoint (NOT the last) and the test loss there.
Acceptance: (1) prints train & val loss per checkpoint; a point where train↓ continues while val↑; (2) best-val checkpoint is NOT the final one; (3) three splits disjoint, sizes printed and sum to dataset size; (4) test loss evaluated only at best-val; deterministic.
Explain-back: Why does training loss alone lie about a model, and what is the validation set protecting you from?`,
    visual: 'P0: BUILD one dataset -> 3 boxes -> both losses fall -> the gap opens (val turns up) -> early-stop marker -> test used once. Prevents "lower train loss is always better", "one dataset is enough", "overfitting = a bug". Fixes to verify: best model is at step 200 (audit: "near step 100" was wrong — the val minimum is step 200); overfitting is tied to CAPACITY. A live complexity/epoch slider is P1 (justified deferral — the static two-curve build is adequate for the capstone).',
    analogy: 'Studying for an exam (train = practice problems; val = mock exam; test = the real exam sat once). Breaks: splits differ only by random sampling — a peeked-at val set silently becomes another practice set.',
    misconceptions: '"the validation set is the test set" -> three roles (train fits θ; val tunes/selects, slowly leaks; test touched ONCE); "overfitting means the model is broken" -> it is the model working exactly as optimized (memorizing noise); "lower loss always means a better model" -> loss is a proxy.',
    staffLens: 'Silent failure: data leakage (scaler fit on all data / dup rows / temporal bleed -> inflated offline score). Trade-off: split ratio + strategy (random / stratified / grouped / k-fold).',
    defectChecks: 'Best model reported at step 200 (NOT step 100); overfitting tied to model capacity; three disjoint splits with sizes summing; module labels say "Module 2".',
    isFoundation: false,
  },
]

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------
const QA_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['day', 'coverage', 'gateVerdicts', 'findings', 'legacyRefsInFile'],
  properties: {
    day: { type: 'string' },
    coverage: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['concept', 'requiredDepth', 'lessonEvidence', 'visualEvidence', 'artifact', 'result'],
        properties: {
          concept: { type: 'string' },
          requiredDepth: { type: 'string' },
          lessonEvidence: { type: 'string', description: 'where/how it lands, with file:line' },
          visualEvidence: { type: 'string', description: 'visual/evidence present, or n/a' },
          artifact: { type: 'string', description: 'artifact present, or n/a' },
          result: { type: 'string', enum: ['PASS', 'PARTIAL', 'FAIL'] },
        },
      },
    },
    gateVerdicts: {
      type: 'object', additionalProperties: false,
      required: ['coachVoice', 'visualEvidence', 'artifact', 'math', 'coachLayer6pack'],
      properties: {
        coachVoice: { type: 'string', description: 'PASS/PARTIAL/FAIL + one-line reason' },
        visualEvidence: { type: 'string' },
        artifact: { type: 'string' },
        math: { type: 'string' },
        coachLayer6pack: { type: 'string', description: 'which of Jargon/MathLadder/silent-failure/trade-off/interview/quiz-Q are present' },
      },
    },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['id', 'gate', 'severity', 'summary', 'evidence', 'why', 'suggestedFix'],
        properties: {
          id: { type: 'string', description: 'e.g. D1-F1' },
          gate: { type: 'string', enum: ['coverage', 'coachVoice', 'visual', 'artifact', 'math', 'legacy', 'js-nav', 'defect-class', 'other'] },
          severity: { type: 'string', enum: ['P0', 'P1', 'P2'] },
          summary: { type: 'string' },
          evidence: { type: 'string', description: 'file path + line number + short verbatim quote' },
          why: { type: 'string' },
          suggestedFix: { type: 'string' },
        },
      },
    },
    legacyRefsInFile: { type: 'array', items: { type: 'string', description: 'legacy skill name + file:line' } },
  },
}

const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['day', 'verdicts', 'missed'],
  properties: {
    day: { type: 'string' },
    verdicts: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['id', 'verdict', 'severityAfter', 'note'],
        properties: {
          id: { type: 'string' },
          verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'ADJUSTED'] },
          severityAfter: { type: 'string', enum: ['P0', 'P1', 'P2', 'DROP'] },
          note: { type: 'string', description: 'what the re-read of the cited line(s) showed' },
        },
      },
    },
    missed: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['gate', 'severity', 'summary', 'evidence', 'why'],
        properties: {
          gate: { type: 'string' },
          severity: { type: 'string', enum: ['P0', 'P1', 'P2'] },
          summary: { type: 'string' },
          evidence: { type: 'string' },
          why: { type: 'string' },
        },
      },
    },
  },
}

const GENERIC_FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['scope', 'summary', 'findings'],
  properties: {
    scope: { type: 'string' },
    summary: { type: 'string', description: 'overall verdict for this cross-cutting check' },
    passes: { type: 'array', items: { type: 'string', description: 'things verified OK, with evidence' } },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['id', 'severity', 'summary', 'evidence', 'why', 'suggestedFix'],
        properties: {
          id: { type: 'string' },
          severity: { type: 'string', enum: ['P0', 'P1', 'P2'] },
          summary: { type: 'string' },
          evidence: { type: 'string', description: 'file path + line number + short verbatim quote' },
          why: { type: 'string' },
          suggestedFix: { type: 'string' },
        },
      },
    },
  },
}

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------
function qaPrompt(d) {
  return `You are a STRICT v7 QA reviewer for one Frontier Lab lesson. Report-only, no edits.

TARGET LESSON: ${MOD}/${d.folder}/lesson.html  (${d.id}, quest-id ${d.qid})
TARGET ARTIFACT STUB: ${MOD}/${d.folder}/experiment.py
Day goal: ${d.goal}

${READ_ONLY}

STEP 1 — Read the full lesson.html and the experiment.py stub for this day. Read the four contract files if you need the surrounding context, but the per-day spec below is authoritative for what to check.

STEP 2 — COVERAGE TRACEABILITY. For EACH must-cover concept below, locate it in the lesson text, judge whether it lands at the required depth, and whether the required visual/evidence and artifact are present. Return a coverage row (result PASS/PARTIAL/FAIL) for each. Also check the should-cover items (a missing should-cover is at most P1; a missing must-cover is P0).
MUST-COVER for ${d.id}:
- ${d.mustCover.join('\n- ')}
${d.shouldCover ? 'SHOULD-COVER for ' + d.id + ':\n- ' + d.shouldCover.join('\n- ') : ''}

STEP 3 — Apply every v7 gate below and produce findings.
${V7_GATES}

STEP 4 — Day-specific checks:
- ARTIFACT CONTRACT ROW (verify the Produce step + acceptance criteria match this EXACTLY, and that numbers agree with the playground/Math Ladder):
${d.artifact}
- VISUAL CONTRACT (P0): ${d.visual}
- REQUIRED ANALOGY (must be present, deep — more than one sentence — and must STATE where it breaks): ${d.analogy}
- MISCONCEPTIONS this day must reframe: ${d.misconceptions}
- STAFF LENS (one silent-failure callout + one trade-off callout required): ${d.staffLens}
- DEFECT-CLASS CHECKS (these are fixes the refactor claimed to make — verify each is actually done; a regression here is P0): ${d.defectChecks}
${d.isFoundation ? '- THIS IS A NEURAL-NETWORK FOUNDATION LESSON: the brain/human analogy + its caveat is REQUIRED. Missing it is P0.' : ''}

STEP 5 — Legacy skill references. Grep the lesson.html + experiment.py for these names and report each with file:line: ${LEGACY_NAMES.join(', ')}. Note: /frontier-experiment-lab appearing in the Produce "Option B" is a known curriculum-wide item — still report it with its exact line, but note it is the contract-sanctioned Option-B pattern.

Every finding: give a stable id (${d.id}-F1, ${d.id}-F2, ...), the gate, severity P0/P1/P2, a one-sentence summary, exact evidence (file + line + short verbatim quote), why it matters, and a concrete suggested fix. Be exhaustive but do not fabricate — if the lesson is genuinely good on a dimension, record it as a PASS in the relevant gateVerdict and move on. Do NOT edit any file.`
}

function verifyPrompt(qa, d) {
  return `You are an ADVERSARIAL verifier. A first-pass QA reviewer produced findings for lesson ${MOD}/${d.folder}/lesson.html (${d.id}). Your job: re-read the ACTUAL cited lines and decide whether each finding is real, plus catch anything the first pass MISSED.

${READ_ONLY}

For EACH finding below, open the cited file at the cited line, read the surrounding context, and rule:
- CONFIRMED — the defect is real as described (keep severity, or set severityAfter).
- ADJUSTED — real but mis-severity or mis-described (set the corrected severityAfter + explain in note).
- REFUTED — the cited evidence does not support the claim, or the thing is actually present/correct (set severityAfter=DROP, explain what the line actually says).
Default to skepticism: if the quote in the finding does not actually appear at/near that line, REFUTE it.

Then, independently, scan for MISSES — real P0/P1 defects the first pass did not report (especially: a must-cover concept that is actually absent, an anchor-spine number mismatch between playground and Produce, a failure mode with no evidence, an unlabeled simulated terminal, a formula-first opening, or a missing required analogy/brain-analogy). Report each miss with exact evidence.

Day-specific ground truth to check the numbers against (these are the contract-correct values):
- ${d.id} artifact/acceptance: ${d.artifact}
- ${d.id} defect-class expectations: ${d.defectChecks}

FIRST-PASS FINDINGS TO VERIFY (JSON):
${JSON.stringify(qa && qa.findings ? qa.findings : [], null, 1)}

FIRST-PASS COVERAGE RESULTS (JSON) — spot-check any you doubt:
${JSON.stringify(qa && qa.coverage ? qa.coverage : [], null, 1)}

Return verdicts (one per finding id) and a missed[] list. Do NOT edit any file.`
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------
phase('PerDayQA')
log(`v7 QA over ${DAYS.length} days of ${MOD} — report-only.`)

const perDay = await pipeline(
  DAYS,
  (d) => agent(qaPrompt(d), { label: `qa:${d.id}`, phase: 'PerDayQA', schema: QA_SCHEMA, effort: 'xhigh' }),
  (qa, d) => agent(verifyPrompt(qa, d), { label: `verify:${d.id}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high' })
    .then((verify) => ({ day: d.id, folder: d.folder, qa, verify })),
)

// ---------------------------------------------------------------------------
// Cross-cutting checks (independent of per-day; run concurrently)
// ---------------------------------------------------------------------------
phase('CrossCutting')

const reviewPagesPrompt = `You are a v7 QA reviewer for the two REVIEW/GATE pages of Module 2.

TARGETS:
- ${MOD}/review-part-a.html (quest-id wf2-review) — the mid-module gate AFTER Day 3. Recaps "what the network IS" (D1–D3).
- ${MOD}/review.html (quest-id wf3-review) — the FINAL gate AFTER Day 9. Recaps the full learning loop + generalization.

${READ_ONLY}

Check:
1. COACH VOICE appropriate for a gate (warm, recaps the arc, not a dry checklist); pain-point / victory-lap framing present.
2. COMPLETION CONTRACT: review gates complete via done.verdict (NOT done.produce). Confirm the completion JS writes/reads done.verdict and that the page can actually reach a "complete" state. (See the known review-gate completion contract: index pill only counted done.produce historically; the gate pages themselves must set done.verdict.)
3. MODULE-LABEL DEFECT CLASS (P0 if wrong). All learner-facing labels must say "Module 2" (kicker, <title>, brand-sub, nav-group, .fin banner). SPECIFICALLY verify these forward-copy fixes landed:
   - review.html: "Ready for Module 4?" should be "Ready for Module 3? (Attention)"; "(M1–M3)" should be "(M1–M2)"; "M4 · Attention" should be "M3 · Attention"; any "Module 3" heading should be "Module 2".
   - review-part-a.html: "Gate Before Module 3" should refer to the TRAINING half of Module 2 (Days 4–9), not "Module 3".
4. COVERAGE RECAP: does each gate actually recap the concepts it claims (D1–D3 for part-a; D1–D9 for review)? Any dangling reference to a day/concept not taught?
5. NAV: review-part-a sits Day3->review-part-a->Day4; review sits Day9->review->m03. Verify prev/next hrefs.
6. Legacy skill names: ${LEGACY_NAMES.join(', ')} — report each with file:line.

Report findings with exact evidence (file + line + short verbatim quote), severity P0/P1/P2, why, and suggested fix. List what PASSES too. Do NOT edit any file.`

const legacyPrompt = `You are doing the v7 LEGACY-SKILL-REFERENCE sweep for Module 2. Report-only.

${READ_ONLY}

Grep every lesson.html, experiment.py, review*.html under ${MOD}/ for these uninstalled/legacy skill names and report EVERY hit with file:line and the verbatim line:
${LEGACY_NAMES.join(', ')}

Use ripgrep/grep with line numbers. Produce a complete, deduplicated inventory (do not sample). For each distinct legacy name found, note: how many files, which files:lines, and the context (is it a Produce "Option B" prompt, a comment, a link?).

Classification guidance: the v7 gate lists "legacy skill reference to uninstalled skill" as P0. However there is a documented curriculum-wide decision that /frontier-experiment-lab appears in ~243 files and is a P1 tech-debt item to be swept curriculum-wide, NOT fixed per-module — and the Module 2 artifact contract itself PRESCRIBES the /frontier-experiment-lab Option-B prompt. So: report /frontier-experiment-lab hits as P1 with that context noted; report ANY OTHER legacy name as P0 (those are not contract-sanctioned). Do NOT edit any file.`

const jsNavPrompt = `You are the v7 JS / NAV / COMPLETION behavior reviewer for Module 2. Report-only.

${READ_ONLY}

Targets: all 9 ${MOD}/day-*/lesson.html plus review*.html.

You MAY run the repo's self-contained Python audit scripts if helpful (they need no npm):
  python3 sessions/nav_audit.py    (prev/next chain)
  python3 sessions/lesson_audit.py m02-the-neuron   (shell-aware lesson audit)
  python3 sessions/coverage_audit.py    (if it accepts a module arg)
Note from repo memory: jsdom is NOT resolvable from cwd; staff_lens_audit.js "render:BROKEN" is a benign .sec-vs-.module-section false positive — do not report it as a defect. Run 'node --check' only on extracted <script> bodies if you can; otherwise inspect statically.

Check, across all 9 lessons + 2 gates:
1. 7-section model present (data-sec: what/intuition/play/why/build/quiz/produce) and the "0/7" checklist count + .gotit gating are wired.
2. Playground DEMOS gates section completion after all buttons clicked; BUILD scroll-reveal gates after last step; QS quiz present; Produce copy-button present; .term tooltips wired.
3. NAV prev/next forms ONE continuous chain: Day1->Day2->Day3->review-part-a->Day4->...->Day9->review->(m03). Report any broken/mismatched href.
4. COMPLETION: lesson pages complete on done.produce; review gates on done.verdict. Confirm localStorage key is 'frontier-lesson:<QID>' with the frozen quest-ids (wf2-d01-neuron ... wf3-d06-split, wf2-review, wf3-review). Flag any wrong/missing quest-id (breaks hub progress + the "First Neuron" badge on wf2-d01-neuron).
5. viz-iframe autoresize: D5/D7/D8 embed viz/gradient-descent.html via .build-embed. Confirm a postMessage height sender exists (else the iframe sticks at 520px). Report any day whose embed would stick.
6. Any obviously broken JS (undefined refs, truncated <script>, syntax errors).

Report findings with exact evidence (file + line + short verbatim quote or audit output), severity P0/P1/P2, why, suggested fix. List PASSES too. Do NOT edit any file.`

const numericPrompt = `You are the v7 CROSS-DAY NUMERIC / ANCHOR-SPINE consistency reviewer for Module 2. Report-only.

${READ_ONLY}

The anchor-spine rule: within each day, the numbers in the Playground fake-terminal must equal the numbers in that day's Math Ladder and in the Produce acceptance criteria. Across days, the worked story must be self-consistent. Read the relevant sections of each day's lesson.html and experiment.py and VERIFY these specific contract-mandated numbers/fixes actually landed (a mismatch is P0 — anchor-spine break or incorrect math):

- D1: neuron anchor x=[2,3], w=[0.5,-1], b=1 -> weighted sum -2, +bias -1, output 0.0; positive case w=[1,1],b=0 -> 5.0. np.maximum used in BOTH playground and Produce.
- D2: ONE anchor matrix pair used identically across playground / math-ladder / produce (the audit said three drifted); (x@W1)@W2 == x@(W1@W2) collapse, relu inserted breaks it; sigmoid'(0)≈0.25.
- D4: CE for p_true=[0.3,0.6,0.9] -> ≈1.204 / 0.511 / 0.105, SAME order in playground and Produce; MSE==0 at exact match.
- D5: GD loop loss=w**2 from w=3 with step w=w-0.1*(2*w); finite-diff vs analytic <1e-6; forward-pass cost stated as ~3x (NOT an inconsistent "2-3x" vs "3x").
- D6: the loss value reads 0.746 (NOT 0.747); epoch/iteration worked count (e.g. 100/25=4 steps) is arithmetically correct; |w-3|<0.1 and final loss<~0.01.
- D7: first optimizer step both == w=1.8; momentum's real continuation is g=−9.6, v=−24.0 (NOT a hypothetical −12.8); lr=0.05, β=0.9.
- D8: three LRs 0.005/0.1/0.6; the |1−lr·c|<1 rule with c=8 -> tipping lr=2/c=0.25 is arithmetically correct; playground step count == Produce step count.
- D9: best-val checkpoint reported at step 200 (NOT step 100); split sizes are disjoint and sum to the dataset size.

For each item: state the value(s) you actually found (with file:line + quote) and whether it matches the contract. Report every mismatch as a finding (P0 for anchor-spine/number errors). List the ones that correctly match as passes. Do NOT edit any file.`

const cross = await parallel([
  () => agent(reviewPagesPrompt, { label: 'review-pages', phase: 'CrossCutting', schema: GENERIC_FINDINGS_SCHEMA, effort: 'xhigh' }),
  () => agent(legacyPrompt, { label: 'legacy-sweep', phase: 'CrossCutting', schema: GENERIC_FINDINGS_SCHEMA, effort: 'medium' }),
  () => agent(jsNavPrompt, { label: 'js-nav', phase: 'CrossCutting', schema: GENERIC_FINDINGS_SCHEMA, effort: 'high' }),
  () => agent(numericPrompt, { label: 'numeric-consistency', phase: 'CrossCutting', schema: GENERIC_FINDINGS_SCHEMA, effort: 'xhigh' }),
])

return {
  perDay,
  crossCutting: {
    reviewPages: cross[0],
    legacySweep: cross[1],
    jsNav: cross[2],
    numericConsistency: cross[3],
  },
}
