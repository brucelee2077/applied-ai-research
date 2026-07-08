# Module 2 — Artifact Contract

> Every lesson's Produce step ships one artifact: a tiny **NumPy-only, deterministic** `experiment.py`
> that prints evidence for that day's claim. `experiment.py` stays a **guided stub** (the learner writes
> it via Produce **Option A**, or **Option B** copies a prompt that triggers `/frontier-experiment-lab`).
> The refactor makes each Produce prompt + acceptance criteria **exact** and **numerically identical to
> that day's playground and Math Ladder** (fixing today's drift on D7/D8/D9). Plus a 5-minute research
> log to `log.md` answering the explain-back question.
>
> **Path convention (all days):** `sessions/m02-the-neuron/<day-folder>/experiment.py`
> **Run convention (all days):** `python3 sessions/m02-the-neuron/<day-folder>/experiment.py`
> **Rules:** NumPy only (no torch/jax/sklearn); fixed seed / fixed inputs → reproducible; small & fast;
> prints labeled steps; every acceptance criterion is a concrete number or behavior visible in the output.

---

| Lesson | Artifact (what the script computes & prints) | Path (folder) | Acceptance criteria | Explain-back (→ log.md) |
|---|---|---|---|---|
| **D1 Neuron** `wf2-d01-neuron` | `neuron(x,w,b)` = `relu(np.dot(w,x)+b)` using `np.maximum`. Runs the anchor `x=[2,3], w=[0.5,-1], b=1` printing weighted-sum `-2`, +bias `-1`, output `0.0`; then a positive case `w=[1,1], b=0` → `5.0`. Prints the same input twice to show determinism. | `day-01-single-neuron` | (1) prints 3 labeled steps: weighted sum, +bias, output; (2) anchor case outputs `0.0`, positive case outputs a positive number; (3) same input twice → identical output; (4) uses `np.maximum` (not built-in `max`) — consistent with the playground | Why is a neuron just a weighted sum + bias, and what does the bias let it do that weights alone cannot? |
| **D2 Activations** `wf2-d02-activations` | `relu`, `sigmoid`, `tanh` over a fixed grid (e.g. `np.linspace(-6,6,13)`), printed as a table. Then the **one anchor matrix pair** (same as the playground): show `(x@W1)@W2 == x@(W1@W2)` to 1e-10 (collapse), then `relu(x@W1)@W2` differs (cure). | `day-02-activations` | (1) table: ReLU 0 on negatives / `x` on positives, sigmoid in (0,1), tanh in (−1,1); (2) stacked-linear == single-combined-linear within 1e-10; (3) inserting ReLU changes the output (nonzero diff); (4) prints sigmoid'(0)≈`0.25` and notes ReLU grad is 0 for negatives | Why does stacking linear layers with no activation give no more power than one linear layer? |
| **D3 Forward pass** `wf2-d03-forward` | 2-layer MLP: `x → relu(x@W1+b1)=h → h@W2+b2 = ŷ` with fixed small matrices; prints shape **and values** at each stage. Recomputes one hidden unit by hand to match the vectorized entry. Runs a batch of 4 to show batching = an extra dim. | `day-03-layers-forward-pass` | (1) shape printed each stage and they chain (e.g. `(2,)→(3,)→(1,)`); (2) one hand-computed hidden unit matches `W1@x+b1` to 1e-10; (3) batched run prints `(4, out)` and row 0 == single-input result; (4) post-ReLU hidden values all ≥ 0 | How does a layer turn one matmul into many neurons at once, and why is the batch just an extra dimension? |
| **D4 Loss** `wf3-d01-loss` | `mse(pred,target)` and `cross_entropy(p_correct)=-np.log(p_correct)` (clipped to avoid log(0)). Prints MSE close vs far; CE for `p_true ∈ [0.3,0.6,0.9]` → `≈1.204, 0.511, 0.105` (same order as the playground). Shows confident-wrong ≫ hesitant-wrong. | `day-04-loss` | (1) MSE == 0 (within 1e-12) at match, larger for any mismatch; (2) CE for `p=[0.3,0.6,0.9]` prints `≈1.204/0.511/0.105` (matches playground order); (3) confident-wrong CE ≫ hesitant-wrong CE; (4) inputs clipped/offset so no `log(0)` NaN/inf | What does a loss turn mistakes into, and why must it be smallest when the prediction is exactly right? |
| **D5 Gradients/backprop** `wf3-d02-backprop` | Two parts. (a) The named GD baseline: `loss=w**2`, loop `w = w - 0.1*(2*w)` from `w=3`, print `w, loss` falling toward 0. (b) Chain-rule check on `loss=(sigmoid(w*x+b)-y)**2`: derive `dL/dw`, `dL/db` analytically, print the chain pieces, and verify against finite differences. | `day-05-gradients-backprop` | (1) GD loop: loss strictly decreasing, `w→0`; (2) analytic vs numerical gradient for `w` and `b` agree to < 1e-6; (3) prints the intermediate chain pieces (`dL/dpred`, `dpred/dz`, `dz/dw`) so the multiplication is visible; (4) fixed `eps` + fixed inputs → reproducible | Why is backprop just the chain rule backward, and what does the finite-difference check prove? |
| **D6 Training loop** `wf3-d03-training-loop` | The full loop wiring D1–D5: fit `pred=w*x` to `x=2, y=6` (ideal `w=3`) from `w=1.0`. Each step: forward → loss → grad → **(zero any accumulator) →** update `w -= lr*grad`. Prints `step, w, loss` for ~20 steps; confirms `w→3`, `loss→0`. | `day-06-training-loop` | (1) loss at step 0 ≫ final loss (final below a stated small threshold, e.g. < 0.01); (2) learned `w` within a stated tolerance of 3 (e.g. |w−3| < 0.1); (3) prints the labeled stages (forward, loss, backward, update) at least once; (4) deterministic: two runs print identical final loss | What are the four (with zeroing, five) repeating steps of a training loop, and which one actually changes the weights? |
| **D7 Optimizers** `wf3-d04-optimizers` | Plain SGD vs SGD+momentum (β=0.9, v=0) on the **same** problem (`pred=w*x`, `x=2, y=6`, `w=1.0`, `lr=0.05`), two 15-step loops printing `step, w, loss`. Momentum reaches `w≈3` in fewer steps. **Numbers must match the (corrected) playground**: first step both give `w=1.8`; momentum's real continuation uses `g=−9.6` at `w=1.8`. | `day-07-optimizers` | (1) both start from identical `w=1.0` / same seed, printed once; (2) first step of each == `w=1.8` (matches the lesson); (3) momentum reaches the target loss in fewer steps than plain SGD; (4) both converge (final loss below the stated target) | What does momentum fix that plain SGD struggles with, and what extra thing does Adam adapt per-parameter? |
| **D8 Learning rate** `wf3-d05-lr` | Same loop, same init (`w=1.0`), three learning rates `0.005 / 0.1 / 0.6`, each an **N-step** loop (N matches the lesson's playground — align to one count), printing `step, w, loss` and labeling each run crawl / converge / diverge. Comment the `|1 − lr·c| < 1` rule with `c=8` → tipping point `lr = 2/c = 0.25`. | `day-08-learning-rate` | (1) `lr=0.005` ends far from converged (still crawling); (2) `lr=0.1` reaches `w≈3` / low loss; (3) `lr=0.6` diverges (loss grows / oscillates) and is labeled as such; (4) same seed + init printed once so lr is the only variable; step count == the playground's | What happens when the LR is too small vs too large, and why tune it first? |
| **D9 Train/val/test** `wf3-d06-split` | **Match the playground's framing** (pick one and use it in both): synthetic noisy data, split train/val/test (disjoint), fit a model flexible enough to overfit; print train vs val loss at each checkpoint showing train↓ while val↓-then-↑; report the best-val checkpoint (not the last) and the test loss there. | `day-09-train-val-test` | (1) prints train & val loss per checkpoint; a point where train↓ continues while val↑ (overfitting visible); (2) best-val checkpoint is **not** the final one; (3) the three splits are disjoint, sizes printed and sum to the dataset size; (4) test loss evaluated only at the best-val point; deterministic under a fixed seed | Why does training loss alone lie about a model, and what is the validation set protecting you from? |

---

## experiment.py stub convention (what each file contains after the refactor)

Each `experiment.py` remains a short guided placeholder that points back to the lesson's Produce step —
it is intentionally NOT a filled reference solution (the learner produces it). Keep the existing stub
shape, but ensure the **run path in the comment matches the folder** and mentions both options:

```python
# <day-folder> — experiment
#
# Fill this from the lesson's PRODUCE step (open lesson.html, section 7):
#   Option A: write it yourself, or  Option B: paste the /frontier-experiment-lab prompt.
# Then run:  python3 sessions/m02-the-neuron/<day-folder>/experiment.py
```

## Artifact QA (checked in `/frontier-refactor-qa`)

- [ ] Every day's Produce Option A prompt + acceptance criteria == this contract (exact numbers)
- [ ] Every Produce run command path == the actual folder (no stale `Module 3 Day N` in prompts)
- [ ] Playground numbers == Produce acceptance numbers for every day (anchor-spine)
- [ ] D3 Produce Option-B prompt has NO authoring artifact ("W2 (4,4)... use W2 (4,2)" → cleaned)
- [ ] D1 uses `np.maximum` in both playground and Produce
- [ ] Each `experiment.py` stub's commented run path matches its folder
- [ ] Each lesson has the 📓 5-minute-log callout naming the explain-back question
