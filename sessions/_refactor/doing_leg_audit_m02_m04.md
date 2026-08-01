# Doing-leg adversarial audit — m02 / m03 / m04

Read-only audit, 2026-08-01. These 20 days predate the wave-2 discipline: m02 (9) was
written before `_experiment_check.py` existed and had never been adversarially planted;
m03 (5) and m04 (6) were planted in wave 1, before the five no-op traps and the cross-day
pass were discovered. All 20 pass the gate and are deterministic — this audit tests whether
their CHECKS mean anything, which no gate can do.

Method: 271 semantic defects planted one at a time on /tmp copies, full stdout diffed against
a baseline. CAUGHT = the ❌ branch fired or an assert tripped. SURVIVED = stdout changed but
the script still printed `✅ you got it` — a real gap. NO_OP = byte-identical, so the plant
was not semantically real (or the quantity is genuinely invariant) and proves nothing.

## Plant results

| Module | Days | Plants | Caught | **Survived** | No-op |
|---|---|---|---|---|---|
| m02-the-neuron | 9 | 103 | 56 | **28 (27%)** | 19 |
| m03-attention | 5 | 83 | 53 | **7 (8%)** | 23 |
| m04-first-model-mlp | 6 | 85 | 72 | **7 (8%)** | 6 |
| **total** | 20 | **271** | 181 | **42** | 48 |


## m02-the-neuron — per-day findings

### day-06-training-loop

- **[P1] zero-initialised / co-linear parameter makes its own code path untestable (b starts at 0.0, x = 2.0)** — The bias half of the four-step loop is completely unchecked. Deleting the bias update (`b = b - lr*grad_b`), or deleting `b` from the forward pass, or SWAPPING which gradient drives which parameter, all print a visibly different loss curve (0.1887 / 0.0356 / 0.0067 / 0.0013 instead of the documented 0.1216 / 0.0148 / 0.0018 / 0.00022) and still exit 0 with ✅. The check only bands the FINAL loss (`< 1e-3`) plus rough monotonicity, and on this 1-D convex problem any descent-shaped update converges inside 50 steps. Worse, `x = 2.0` makes the two gradients numerically confusable: 2x^2 = 8 = 2x + 4, so "swap grad_w and grad_b" is byte-identical to "delete the bias".
  - proven by: `experiment.py:36 `b = b - lr * grad_b` -> `pass` (plant 01); line 24 `pred = w*x + b` -> `pred = w*x` (plant 02); lines 35-36 swapped operands (plant 08)` (stdout changed: True)
  - fix: Return `w, b` and pin them (`abs(w - 0.4)` / `abs(b - 0.2)` at the shipped settings), and pin the printed curve itself: `abs(healthy[10] - 0.121577) < 1e-6`.
- **[P1] 3 fake/absent prediction — an acceptance-listed observable is never computed** — Both the produce spec and the acceptance criteria require the lr = 1.5 run to PRINT the prediction and show it flipping sign every lap: "the guess swings 0 -> 15 -> -195 -> 2745 -> ...". The artifact never computes or prints `pred` for that run at all — it prints only the final loss, then a prose line asserting the flip happened (line 57) and a prose line claiming that continuing would reach inf/NaN. Neither is computed nor asserted. So the sign-flip mechanism — the actual reason a too-high lr diverges, as opposed to merely growing — is claimed in text only.
  - proven by: `experiment.py:53 `train_one_neuron(lr=1.5)` -> `lr=1.2`: still ✅ (`> 1e6` has 106 decades of headroom), though the documented 0/15/-195/2745 sequence would now be 0/12/-132/1452` (stdout changed: True)
  - fix: Return the pred history too, print it for lr = 1.5, and assert `preds[:4] == [0.0, 15.0, -195.0, 2745.0]` plus alternating signs.
- **[P2] 2 too weak to see the bug (bands with 100+ decades / 4% of slack)** — `final_high > 1e6` is checked against an actual 2.09e+112, and `0.9 < final_low < 0.99` against an actual 0.9522. So neither of the two failure-mode runs can see its own learning rate: lr 1.5 -> 1.2 still passes, and lr 0.00005 with 25 iterations instead of 0.00005 with 50 also passes. The two documented numbers ("around 10^112", "about 0.952") are printed but not pinned.
  - proven by: `experiment.py:53 `lr=1.5` -> `lr=1.2`; line 60 `lr=0.00005` -> `lr=0.0001, num_iters=25`` (stdout changed: True)
  - fix: Tighten to `1e100 < final_high < 1e120` and `abs(final_low - 0.952169) < 1e-5`.

### day-07-optimizers

- **[P1] claim printed but never asserted (the loss@30 column)** — The `loss@30` column — the three numbers 5.49e-03 / 3.83e-07 / 7.97e-28 that make the race result quantitative — is printed but never asserted; only the three step counts are. Replacing `run_plain_sgd`'s returned loss with the loss at the STARTING point prints `loss@30 = 4.72e+00` for plain SGD (three orders too high, and above the 0.01 threshold the same line claims was crossed at step 26) and still prints ✅. Changing the adaptive optimizer's epsilon from 1e-8 to 1e-1 moves its printed loss by seven decades, also ✅.
  - proven by: `experiment.py:57 `return first_below, loss(wx, wy)` -> `return first_below, loss(1.5, 1.5)` (plant 12); line 97 `+ 1e-8` -> `+ 1e-1` (plant 10)` (stdout changed: True)
  - fix: Add `assert plain_loss < 0.01 < 1.0` style consistency checks plus absolute pins on the three losses (e.g. `abs(mom_loss - 3.83e-07) < 1e-9`).
- **[P2] 5 entailed clause / dead code** — `assert plain_step > mom_step > adap_step` sits INSIDE `if got == expected:` where `expected = (26, 13, 4)`. The tuple equality already implies 26 > 13 > 4, so this assert can never be the failing one — it is unreachable-as-a-failure. Deleting it is byte-identical.
  - proven by: `experiment.py:133 assert -> `pass` (probe E2): NO-OP` (stdout changed: False)
  - fix: Move the ordering assert into the else/knob-changed branch where the tuple is NOT pinned, or drop it.
- **[P2] untestable strictness (no value sits on the boundary)** — The convergence test `if loss(wx, wy) < 0.01` is the file's only threshold, and no run ever produces a loss of exactly 0.01, so `<` vs `<=` is byte-identical. The `first_below` step counts — the day's three headline numbers — therefore have an unverified boundary semantics.
  - proven by: `experiment.py:55/76/99 `< 0.01` -> `<= 0.01` (plant 09): NO-OP (byte-identical)` (stdout changed: False)
  - fix: Add one probe run whose loss is exactly 0.01 (or assert the boundary semantics on a synthetic list).

### day-08-learning-rate

- **[P1] claim printed but never asserted, in a DUPLICATED implementation of the update rule** — `loss_curve()` (lines 32-38) is a second, independent copy of the update `weight = weight - lr*(2*weight)` whose output is the day's headline evidence — the two printed per-step curves the produce spec explicitly asks for ("print the per-step loss for lr=1.0 and lr=0.1 so the too-big divergence and the just-right smooth drop are both visible"). Nothing in the self-check ever touches it. Flipping its sign prints `lr = 1.0 loss per step (too big -> stuck bouncing at 4): [36.0, 324.0, 2916.0, 26244.0, 236196.0, 2125764.0]` — a printed label flatly contradicted by the numbers beside it — and still exits 0 with ✅. Dropping the factor 2 prints `[0.0, 0.0, 0.0, ...]` under the same "bouncing at 4" label, also ✅. An off-by-one (record before the hop) also survives.
  - proven by: `experiment.py:36 `weight = weight - lr * (2 * weight)` -> `weight = weight + lr * (2 * weight)` (plant 01); and -> `weight - lr * weight` (plant 02); and swap the append above the hop (plant 03)` (stdout changed: True)
  - fix: Delete the duplicate: have `train()` return the curve and take `curve[-1]` for the sweep. Then pin `loss_curve(1.0)[:6] == [4.0]*6` and `abs(loss_curve(0.1)[0] - 2.56) < 1e-9`.
- **[P1] 2 too weak to see the bug (9 decades of slack) — only the floor of the schedule is pinned** — For the bonus schedule the only real clause is `abs(sched_lr - 0.1) < 1e-9`, which any decay factor < 1 satisfies once it reaches the floor. The companion clause is `sched_loss < 1.0` against an actual value of 1.4e-09 — nine orders of headroom. So the halving factor (0.5 -> 0.9), the starting rate (0.8 -> 0.3), the ORDER of hop-then-halve (halve first), and the step count (40 -> 8) all change the printed final loss and all still print ✅ — under a banner line that hardcodes the string "start 0.8, halve, floor 0.1" regardless of what the code did.
  - proven by: `experiment.py:49 `lr = max(floor, lr*0.5)` -> `lr*0.9` (plant 05); line 44 `start_lr=0.8` -> `0.3` (plant 06); lines 48-49 reordered (plant 04); `train_with_schedule(steps=8)` (probe E5)` (stdout changed: True)
  - fix: Pin `abs(sched_loss - 1.3976036e-09) < 1e-15`, and return the lr trace to pin `lrs[:4] == [0.8, 0.4, 0.2, 0.1]`.

### day-09-train-val-test

- **[P0] 9 identity/scale-invariant claim + 5 entailed clause (the whole verdict reduces to one scale-blind argmin)** — The self-check's only load-bearing clause is `best_degree == 5`, computed from `np.argmin(val_losses)` — which is invariant to ANY positive rescaling of the loss. The other three clauses (`train_monotone`, `best_in_middle`, `overfits_at_end`) are all entailed by it, so nothing in the file pins the SCALE of either loss, nor the train-vs-val comparison, nor the `gap` column that is the day's entire teaching claim. Swapping `np.mean` for `np.sum` in `mse()` rescales train losses by 32x and val losses by 8x (different divisors, so the two piles become incomparable) and the script still prints ✅; hardcoding the divisor to 32 for both piles makes val_loss look 4x BETTER than train at every single degree — the printed table then shows a NEGATIVE gap almost everywhere while the ✅ line still announces "the widening gap IS overfitting".
  - proven by: `experiment.py:49 `return float(np.mean((pred-target)**2))` -> `np.sum(...)` (plant 01), and separately -> `np.sum(...)/32` (plant 02)` (stdout changed: True)
  - fix: Pin at least one absolute loss value written down in the check (e.g. `abs(train_losses[0] - 0.2886) < 1e-3` and `abs(val_losses[4] - 0.0935) < 1e-3`), and assert the gap sign story directly (`val_losses[-1] - train_losses[-1] > val_losses[2] - train_losses[2]`), so a rescaled or wrongly-normalised loss fails.
- **[P2] 5 entailed clause + 8 self-derived tautology** — All three supporting clauses are entailed by `best_degree == 5`: `best_in_middle` is `1 < 5 < 15`; `overfits_at_end` is `val_losses[-1] > best_val` where `best_val = min(val_losses)` on the same list, so it is true for every history whose argmin is not the last element — which `best_degree == 5` already guarantees. Deleting all three from `ok` is byte-identical. `train_monotone`'s tolerance can be widened from 1e-9 to 1.0 with no change, so it is nowhere near firing either.
  - proven by: `experiment.py:106-107 `ok = (best_degree == expected_best_degree and train_monotone and best_in_middle and overfits_at_end)` -> `ok = (best_degree == expected_best_degree)` (probe E1): NO-OP; line 99 `+ 1e-9` -> `+ 1.0` (plant 11): NO-OP` (stdout changed: False)
  - fix: Replace `overfits_at_end` with an absolute comparison against a written-down value (`val_losses[13] > 0.30 > val_losses[4]`) so it is independent of the argmin.

### day-01-single-neuron

- **[P2] dead code — a defined, documented function that is never called** — `step(z)` (lines 14-17) is defined, documented as the day's 'hard on/off switch' with the boundary rule 'if the score z reaches the bar (zero), the neuron fires -> 1', and never called anywhere. `best_single_line_accuracy()` re-implements it inline as `guess = 1 if z >= 0 else 0`, so the two spellings of the boundary can drift apart silently.
  - proven by: `line 17: `return (z >= 0).astype(float)` -> `raise RuntimeError("step() is never called")`` (stdout changed: False)
  - fix: Either call it (`assert step(np.array([-0.1, 0.0, 0.1])).tolist() == [0.0, 1.0, 1.0]` pins the boundary) and use it in the XOR loop (`guess = step(z)`) so there is one spelling, or delete it.
- **[P2] untested strictness + formula exercised at effectively one setting** — `assert best == 0.75` is a CEILING claim, and many degenerate searches also reach that ceiling, so nothing about the search itself is tested: the `>=`/`>` boundary in the inline step, the resolution of the candidate grid, and whether w1 is searched at all are all invisible.
  - proven by: `line 63: `guess = 1 if z >= 0 else 0` -> `guess = 1 if z > 0 else 0`` (stdout changed: False)
  - fix: Also pin a FLOOR from the same search — assert that at least one line reaching 0.75 exists AND that the count of distinct (w1,w2,b) triples reaching 0.75 matches a literal — and add one point where z is exactly 0 (e.g. assert the neuron's behaviour at w=[1,1], b=-2 on x=[1,1]) so `>=` vs `>` is decidable.
- **[P2] no ❌ path at all** — day-01 contains no ❌ string anywhere. It asserts with messages and prints `✅ you got it` on success; on failure the learner gets an AssertionError traceback and no 'expected ...' guidance line.
  - proven by: `line 101: `assert best == 0.75` -> `assert best == 1.0`` (stdout changed: True)
  - fix: Wrap the four checks in the boolean-then-verdict shape used by day-02/day-03 and add the '❌ not yet — expected z = -1.0, sigmoid 0.269, best XOR 0.75' line.

### day-02-activations

- **[P1] printed but never asserted (pattern 3, fake prediction, for the two slope lines)** — Part 1 prints all five activation curves over a 13-point grid and Part 2 prints two slope facts, and the self-check asserts NOTHING about any of them. Only `relu` is pinned, and only indirectly via Part 4's `values_match`. `step`, `leaky_relu`, `tanh` and the whole `sigmoid` curve are unconstrained. The lines `ReLU slope for negatives = 0` and `leaky slope for negatives = 0.01` are hardcoded strings, never computed from the functions they describe.
  - proven by: `line 28: `def leaky_relu(z, alpha=0.01):` -> `def leaky_relu(z, alpha=0.1):`` (stdout changed: True)
  - fix: Add one pinned literal per curve in the self-check, e.g. `assert list(np.round(step(grid),3)) == [0,0,0,0,0,0,1,1,1,1,1,1,1]`, `assert leaky_relu(-1.0) == -0.01`, `assert round(float(sigmoid(-1.0)),3) == 0.269`, `assert round(float(tanh(1.0)),3) == 0.762`; and compute the two slope numbers from the functions (a central difference at z=-1) instead of typing them into a string.
- **[P1] circular (pattern 1) — expected value re-derived from the code path under test** — `sigmoid_slope_at_0 = s0 * (1 - s0)` where `s0 = sigmoid(0.0)`, then `sigmoid_ok = abs(sigmoid_slope_at_0 - 0.25) < 1e-9`. This is the analytic derivative formula applied to the implementation's own output: it equals 0.25 for ANY function whose value at 0 is 0.5, no matter what its real slope there is. The slope is never measured against `sigmoid`.
  - proven by: `line 35: `return 1.0 / (1.0 + np.exp(-z))` -> `return 0.5 * (1.0 + np.tanh(z))`` (stdout changed: True)
  - fix: Measure the slope instead of asserting the formula: `eps = 1e-5; measured = (sigmoid(eps) - sigmoid(-eps)) / (2*eps); assert abs(measured - 0.25) < 1e-6` and print `measured`, so the printed 0.25 comes from the implemented sigmoid rather than from an identity.
- **[P1] circular / identity claim (patterns 1 and 9)** — `collapse_holds = np.allclose((x_a @ W1) @ W2, x_a @ (W1 @ W2))` is matrix-multiplication associativity. It is an algebraic identity that holds for every x_a, W1, W2 — it cannot fail, whatever the code does. It is asserted on line 129 with the message 'two linear layers are one', i.e. the day's central claim, and the printed collapse numbers `[[-2. -1.]]` are pinned nowhere.
  - proven by: `lines 78+80 (one semantic change, applied consistently): use `W1.T` in both collapse paths — `two_layers = (x_a @ W1.T) @ W2` and `one_layer = x_a @ (W1.T @ W2)`` (stdout changed: True)
  - fix: Add a literal pin beside the identity, as day-03 does: `assert np.allclose(two_layers, [[-2.0, -1.0]])` and `assert np.allclose(one_layer, [[-2.0, -1.0]])`, so the assertion set can distinguish 'these two paths agree' from 'these two paths agree on the right answer'.
- **[P2] symmetric test data (x_b = -x_a) — vacuous clause** — `x_b = [[-1.0, 3.0]]` is exactly `-x_a`, so `x_sum = [[0., 0.]]`. Consequently `g(x_sum)` and `g(x_a) + g(x_b)` are both exactly zero for EVERY linear map g, which makes `lin_adds_up` unfalsifiable (asserted on line 131), and `np.allclose(f(x_sum), [[0.0, 0.0]])` inside `values_match` is likewise true for every relu-like f. The printed line `straight path g(x_a)+g(x_b) = [[0. 0.]]  g(x_a+x_b) = [[0. 0.]]` is independent of W1 and W2.
  - proven by: `line 93: `return x @ W1 @ W2` (inside `g`) -> `return x @ W2 @ W1`` (stdout changed: False)
  - fix: Pick x_b so that x_a + x_b is non-zero and lopsided (e.g. x_b = [[-1.0, 4.0]] — I verified this choice makes the numbers non-trivial: straight path [[3,1]], bent path [[7,2]] vs [[3,1]]), and pin g(x_sum) to a literal so a wrong W1/W2 order in g is visible.

### day-02-activations + day-03-layers-forward-pass

- **[P2] cross-day collision (no gate is cross-file)** — Both days demonstrate the linear collapse with the SAME two matrices — [[1,2],[0,1]] and [[1,0],[3,1]] — and print two different 'combined grid' answers. day-02 (row-vector convention, `x @ W1 @ W2`) prints and pins `W1 @ W2 = [[7,2],[3,1]]`; day-03 (column-vector convention, `B @ (A @ u)`) prints and pins `combined grid B @ A = [[1,2],[3,7]]`. Neither day mentions that the convention flipped between them.
  - proven by: `n/a — read both files' pinned literals side by side: day-02 line 112 `expected_W = np.array([[7, 2], [3, 1]])` vs day-03 line 129 `expected_grid = [[1, 2], [3, 7]]`` (stdout changed: False)
  - fix: Add one sentence to day-03's Part 2 comment naming the flip explicitly ('day-02 used row vectors x @ W1 @ W2, so its combined grid was W1 @ W2 = [[7,2],[3,1]]; here u is a column vector, so the combined grid is its transpose, B @ A = [[1,2],[3,7]]'), and run the cross-day consistency pass over m02 days 1-9 rather than fixing this one instance.

### day-03-layers-forward-pass

- **[P1] printed but never asserted + absorbed by a downstream clip** — `z1` — the raw pre-activation printed so the learner can 'compare z1 with a1 by eye' — is never asserted. Because ReLU clips both negative entries to 0, the MAGNITUDE of the negative biases in b1 is invisible to the only assertion that touches them (`np.allclose(a1, [1.5, 0.0, 0.6, 0.0])`).
  - proven by: `line 54: `b1 = np.array([0.0, -0.5, 0.0, -2.0])` -> `b1 = np.array([0.0, -1.5, 0.0, -9.0])`` (stdout changed: True)
  - fix: `assert np.allclose(z1, [1.5, -0.2, 0.6, -0.1]), "pre-activation z1 should match the lesson"` before the ReLU is applied.
- **[P1] printed but never asserted (the printed evidence is not derived from an asserted quantity)** — `layer()` prints the shape trace that is the day's stated headline ('watch the length change 3 -> 4 -> 2') via `print(f"... has shape {z.shape} ...")`. The shapes ARE asserted downstream (`a1.shape == (4,)`, `yhat.shape == (2,)`), but the printed line is a separate expression, so it can print anything.
  - proven by: `line 32: `print(f"    layer: W @ x + b has shape {z.shape}  (one number per neuron)")` -> `... {x.shape} ...`` (stdout changed: True)
  - fix: Capture the printed trace and assert it: collect each `z.shape` into a list inside `layer()` and `assert shape_trace == [(4,), (2,)], "the shape trace should read 3 -> 4 -> 2"`.
- **[P2] a parameter initialised to ZERO makes its own code path untestable** — `b2 = np.array([0.0, 0.0])` is commented 'one bias per output neuron', and the day teaches the chaining rule that governs these shapes. Because every entry is zero, numpy broadcasting makes the shape irrelevant to the result, and nothing asserts b2's shape.
  - proven by: `line 62: `b2 = np.array([0.0, 0.0])  # one bias per output neuron` -> `b2 = 0.0`` (stdout changed: False)
  - fix: Give b2 a non-zero, asymmetric value (e.g. `b2 = np.array([0.1, -0.2])`), update the pinned `yhat` literal accordingly, and add `assert b2.shape == (W2.shape[0],), "one bias per output neuron"`.

### day-04-loss

- **[P1] uniform-output trap + one-sided bound (printed-but-unasserted)** — The day's headline contrast — MSE amplifies the outlier ~1900x while MAE merely FEELS it ~10x — is only asserted on the MSE side. `wild_a` has a single one-sided bound (`assert wild_a < 6`) and no lower bound, `grow_a` is never bounded at all, and `close_mae` (0.333) is printed and never asserted. The crowd data is `np.full(20, 3.5)` vs `np.full(20, 3.0)`, i.e. every error is exactly 0.5, so mean, median and max all return 0.5 on it and `assert (clean_m, clean_a) == (0.25, 0.5)` cannot distinguish the reduction.
  - proven by: `line 33: `return np.mean(np.abs(pred - target))` -> `return np.median(np.abs(pred - target))`` (stdout changed: True)
  - fix: Pin the crowd MAE and its growth factor two-sided against literals written in the self-check (`assert 5.0 < wild_a < 5.5` and `assert 9 < grow_a < 12`), assert `close_mae == 0.333`, and make the 20 ordinary misses non-uniform (e.g. alternating +0.5/-0.5 plus a couple of 0.25s) so mean vs median vs max give three different answers.
- **[P2] entailed clause (pattern 5)** — `assert grow_m > 100 * grow_a` can never be the failing clause. The asserts above it already force `clean_m == 0.25` and `wild_m > 400`, hence `grow_m = round(wild_m/0.25) > 1600`; and `clean_a == 0.5` with `wild_a < 6` forces `grow_a = round(wild_a/0.5, 1) < 12`, hence `100 * grow_a < 1200 < 1600`.
  - proven by: `line 81: `grow_a = round(wild_a / clean_a, 1)` -> `grow_a = round(wild_a / clean_m, 1)`` (stdout changed: True)
  - fix: Replace the entailed inequality with a two-sided pin on the ratios themselves: `assert 1850 < grow_m < 1950` and `assert 10.0 < grow_a < 11.0`.
- **[P2] dead branch — half of the epsilon-clip is never exercised** — `p_safe = np.clip(p, 1e-7, 1 - 1e-7)` has two guards, and the day explains both as 'epsilon clipping ... so we never take log(0)'. Every p the script tests (0.99, 0.6, 0.1, 1e-7, 0.0) is below the upper bound, so the `1 - 1e-7` clamp never fires.
  - proven by: `line 47: `p_safe = np.clip(p, 1e-7, 1 - 1e-7)` -> `p_safe = np.clip(p, 1e-7, 1.0)`` (stdout changed: False)
  - fix: Add `assert round(float(cross_entropy(1.0)), 7) == 1e-7` (or a p = 1.0 row in the printed ce table) so the upper clamp has an observable effect.
- **[P2] untested strictness (no input sits on the boundary)** — `assert wild_a < 6` — the sole guard on the crowd MAE — has no input anywhere near its boundary (`wild_a` is 5.238), so the threshold's value and strictness are both unconstrained.
  - proven by: `line 116: `assert wild_a < 6, wild_a` -> `assert wild_a < 7, wild_a`` (stdout changed: False)
  - fix: Replace the one-sided inequality with the two-sided pin from finding 1 (`assert 5.0 < wild_a < 5.5`), which makes both the value and the strictness decidable.

### day-05-gradients-backprop

- **[P1] dead branch (pattern 7) + untested strictness** — The day states four backprop rules; Rule 1 is 'the activation charges a toll' — `relu_slope = 1.0 if z > 0 else 0.0` and `delta = incoming * relu_slope`. The only setting tested is x=2.0, w=0.5, b=0.1, so z = 1.1 and `relu_slope` is 1.0 on every run. The `else 0.0` branch never executes, so the multiplication by the local slope has no observable effect and the dead-ReLU case the file's own comments name (`if z is above 0 keep it; otherwise the neuron outputs a flat 0`) is never run.
  - proven by: `line 48: `delta = incoming * relu_slope` -> `delta = incoming`` (stdout changed: False)
  - fix: Run the same trace at a second setting where the ReLU is off — e.g. w = -1.0 giving z = -1.9 — and assert `w_grad == 0.0 and b_grad == 0.0` there, plus assert the boundary case z == 0 explicitly so `>` vs `>=` is decidable.
- **[P1] printed but never asserted** — Rule 4, `passed_back = delta * w` ('what gets handed one layer further back'), is computed, printed as `to prev layer: delta × w = 0.1`, and never appears in any assert or in the ✅/❌ condition.
  - proven by: `line 54: `passed_back = delta * w` -> `passed_back = delta * x`` (stdout changed: True)
  - fix: `assert math.isclose(passed_back, 0.1, abs_tol=1e-9), "delta x w should be 0.1"` and add it to the ✅ guard.
- **[P2] self-derived tautology (pattern 8) + the printed value is the recomputed side** — `assert math.isclose(w_grad, x * b_grad, abs_tol=1e-9)` is the day's stated 'headline'. But `w_grad = delta * x` and `b_grad = delta * 1.0` are written two lines apart, so `x * b_grad` is `x * delta`, i.e. identically `w_grad` by construction. The printed line `check ratio:   w_grad = 2.0 x b_grad = 0.4` displays `round(x * b_grad, 6)` — the recomputed product, not `w_grad` — so it agrees with itself even if `w_grad` is wrong.
  - proven by: `lines 50+52 (one semantic change applied consistently): `w_grad = delta * x * 3.0` and `b_grad = delta * 3.0`` (stdout changed: True)
  - fix: Print `w_grad` on the ratio line (not `x * b_grad`) so the two sides are visibly different quantities, and derive the ratio claim at a second (x, w, b) setting where x differs, so it says something about x rather than about the two lines above it.
- **[P2] dead branch (pattern 7) — the ❌ message is unreachable** — The `if ... else` at lines 82-85 re-checks two conditions that the asserts on lines 74-79 have already enforced, so the `❌ not yet — expected w_grad 0.4 ...` line can never print: any failure raises AssertionError first.
  - proven by: `line 76: `assert math.isclose(w_grad, 0.4, abs_tol=1e-9)` -> `assert math.isclose(w_grad, 0.5, abs_tol=1e-9)`` (stdout changed: True)
  - fix: Move the ✅/❌ branch above the asserts (compute the booleans, print the verdict, then assert), as day-02 and day-03 do.


## m03-attention — per-day findings

### day-01-embeddings

- **[P2] printed but unasserted + untested strictness (no-op trap #3)** — `shared_dog` and `shared_car` are printed ('shared columns: cat-dog 3  cat-car 1') but only their ordering is asserted, via `predict_ok = predicted == measured`. Inflating one count leaves the ordering intact.
  - proven by: ``shared_dog = int(np.sum((embed("cat") >= 0) & (embed("dog") >= 0)))` — one `>` to `>=` on one side` (stdout changed: True)
  - fix: Add `shared_dog == 3 and shared_car == 1` to `predict_ok` (or to a new `columns_ok`) so the counts, not just their order, are pinned.
- **[P2] entailed clause (pattern 5) + identity claim (pattern 9)** — In `order_ok`, `set(ids_a) == set(ids_b)` and `not np.array_equal(stack_a, stack_b)` are both entailed by the pinned `ids_a == [1,5,6] and ids_b == [6,5,1]` that sits in the same conjunction; in `bank_ok`, `bank_score == 1.0` is entailed by `np.array_equal(bank_from_money, bank_from_river)` because cosine(v, v) == 1 for every nonzero v.
  - proven by: `none needed — provable by inspection; the exhaustive plant `stack_b = E[ids_a]` (same-value plant on the second stack) IS caught by the pinned id lists, which is what shows the extra clauses are padding.` (stdout changed: False)
  - fix: Drop the entailed clauses, or replace them with something independent: pin the printed `same SET of rows? True / same ORDERED rows? False` pair against the multiset (`sorted(ids_a) == sorted(ids_b)`) computed from an unpinned source.

### day-02-qkv

- **[P2] symmetric input hides a transpose (no-op trap #4, applied to a weight matrix)** — `W_K` is a pure diagonal matrix, so `x @ W_K.T` equals `x @ W_K` exactly. The day's stated convention for all three grids — 'row = input slot, column = output slot' — is untestable for W_K.
  - proven by: ``Q, K, V = x @ Wq, x @ Wk.T, x @ Wv` — byte-identical (the same plant on Wq IS caught, and Wv is non-square so it errors)` (stdout changed: False)
  - fix: Break W_K's symmetry by one entry (e.g. give the 'noun' row a small off-diagonal share, as W_K already does on day-05) and re-pin exp_K.

### day-02-qkv / day-03-attention-scores / day-05-positional

- **[P1] formula exercised at exactly one dial setting (no-op trap #2, the floor-division family)** — Every day scales by `np.sqrt(d_k)` with d_k fixed at 4, where sqrt(d_k) == d_k/2 == d_v == 2. Three semantically different divisors are byte-identical, so the module's named mechanism ('scaled' dot-product) is asserted by nothing that can tell it from a wrong law.
  - proven by: ``scaled_scores = raw_scores / (d_k / 2)` and separately `/ d_v` — both byte-identical on all three days (dropping the divide entirely IS caught)` (stdout changed: False)
  - fix: Evaluate the divisor at a second width computed from one shared literal — e.g. print and pin `[float(np.sqrt(w)) for w in (4, 9, 64)] == [2.0, 3.0, 8.0]`, or rescale the same score row at d_k = 9 — so d_k/2 and d_v both disagree.

### day-03-attention-scores

- **[P1] permuted reduction axis + printed-but-unasserted (no-op trap: symmetric/coincidental argmin)** — `distances = np.linalg.norm(V - blended, axis=1)` is printed but never asserted, and the only claim built on it (`nearest_ok`) is an argmin that still lands on 'river' when the reduction runs down the wrong axis.
  - proven by: ``distances = np.linalg.norm(V - blended, axis=0)` (one character)` (stdout changed: True)
  - fix: Pin the distance vector: `np.allclose(distances, [8.2560, 2.5159, 3.8093])` and assert `distances.shape == (3,)` so a collapsed axis cannot pass.
- **[P1] printed but unasserted (only one of three rows pinned)** — Of the 3x3 weight matrix and the 3x2 output table both printed in full, only bank's row is pinned (`shares_ok`, `blend_ok`). Rows 0 and 1 are checked by `budget_ok` alone, which only requires each row to sum to 1.
  - proven by: ``weights = np.array([softmax_row(scaled[2]), softmax_row(scaled[1]), softmax_row(scaled[2])])` — row 0 now holds bank's shares` (stdout changed: True)
  - fix: Add `np.allclose(weights[0], [1/3, 1/3, 1/3])` and `np.allclose(weights[1], [0.2119, 0.5761, 0.2119])`, plus pin the other two rows of `weights @ V` ([4.6667, 4.0] and [6.6089, 3.2716]).

### day-04-multihead

- **[P1] formula exercised at exactly one dial setting (no-op trap #2)** — `d_k = d_model // h` runs only at (d_model, h) = (4, 2), where `d_model // h`, `d_model - h` and plain `h` all equal 2. The lesson's central rule 'each head works in d_k = d_model / h' therefore cannot be distinguished from two wrong rules.
  - proven by: ``d_k = d_model - h` (byte-identical); `d_k = h` (also byte-identical); `d_k = d_model % 3` IS caught` (stdout changed: False)
  - fix: Compute the rule at several settings from one shared literal and pin them: `[(dm, h2, dm // h2) for dm, h2 in ((4,2),(12,3),(64,8))] == [(4,2,2),(12,3,4),(64,8,8)]`. `d_model - h` and `h` then both disagree.
- **[P2] reduction over a block instead of the labelled slots (no-op trap #4 variant: reading the max/min instead of the labelled cell)** — The Part-4 mixing claim is checked as `close(moved_mixing[:, :a_cols].min(), 0.10008479)`. A block-min is insensitive to the array's layout: transposing the whole `moved_mixing` matrix leaves that min at 0.10008479.
  - proven by: ``moved_mixing = np.abs(joined_without_B @ W_O - final).T` — byte-identical (the sibling `moved_no_mixing`, which is pinned with `array_equal` against a zeros block of the right shape, IS caught by the same plant)` (stdout changed: False)
  - fix: Pin the block itself, as the no-mixing branch already does: `np.allclose(moved_mixing[:, :a_cols], [[0.26667,0.13333],[0.59915,0.10008],[0.26667,0.13333]], atol=1e-4)`.
- **[P2] narrative number recomputed beside the assertion (printed but unasserted)** — Several headline numbers are printed from a second, independent copy of the expression that the self-check asserts, so print and assertion can disagree.
  - proven by: `print-side only: `round(float(moved_mixing[:, :a_cols].max()), 4)` in the Part-4 print (prints 0.5992 instead of 0.1001), and `np.round(final.T, 4)` in the 'final values' print — both exit 0 with ✅` (stdout changed: True)
  - fix: Compute once into a named variable (`smallest_mixing_move = ...`, `final_rounded = np.round(final, 4)`) and use that same variable in both the print and the assertion.
- **[P2] constant fold (pattern 6) — guard compares two literals** — `heads_own_their_own_keys = not np.array_equal(W_K_A, W_K_B)` and `corners_both_zeroed` compare hand-typed literals to hand-typed literals. They guard the INPUT data, not any code path, so no defect in split/attend/join/mix can make them fail.
  - proven by: `tying the queries (`W_Q_B` set equal to `W_Q_A`) is byte-identical because cat's row is [0,0,1,1] — slots 2 and 3 both 1.0, so both heads receive the same Query. The author documents this and moved the check to the keys; the key check is then literal-vs-literal.` (stdout changed: False)
  - fix: State the claim as an observable instead: assert head A's and head B's Key tables differ (`not np.allclose(X @ W_K_A, X @ W_K_B)`) and that the two heads' argmax attention lands on different words (`argmax(head_weights[0][CAT]) == 0 and argmax(head_weights[1][CAT]) == 2`).

### day-05-positional

- **[P1] symmetric aggregation hides a swap (no-op trap #4) + printed but unasserted** — Part 4 prints two labelled tables (`with seats, 'dog bites man'` and `with seats, 'man bites dog'`) and asserts only the aggregate `fix_gap = |out_pos_shuffled - out_pos[new_seats]|.max() == 1.7269`. Because `new_seats = [2,1,0]` is an involution, that max is invariant when the two outputs swap roles, so the labelling of both tables is unfalsifiable.
  - proven by: ``out_pos = self_attention(x_shuffled)` / `out_pos_shuffled = self_attention(x_original)` (the two Part-4 assignments exchanged)` (stdout changed: True)
  - fix: Pin one labelled row from each table, e.g. `np.allclose(out_pos[0], [0.9017, 1.1412, 0.1203, 1.0575], atol=1e-4)` and `np.allclose(out_pos_shuffled[0], [1.8366, -0.3416, 0.0474, 0.3795], atol=1e-4)` — an asymmetric pin the swap cannot satisfy.


## m04-first-model-mlp — per-day findings

### day-01-mlp-mnist

- **[P1] too weak to see the bug (the clip is unasserted; min/max survive uint8 wraparound)** — make_stand_in_digits does `np.clip(images + wobble, 0.0, 1.0) * 255.0` then `.round().astype(np.uint8)`, and the comment sells the order ("clip first, then to bytes"). The only claim over it is `scaled_ok = x.min() == 0.0 and x.max() == 1.0 and raw_images.dtype == np.uint8`, which is still literally true after uint8 wraparound, and `data_ok = data_accuracy > 0.95`, which the nearest-average-image labeller still passes on the corrupted data.
  - proven by: `line 27: `pixels = np.clip(images + wobble, 0.0, 1.0) * 255.0` -> `pixels = (images + wobble) * 255.0`` (stdout changed: True)
  - fix: Assert the pre-cast values were in range, e.g. keep the float array and check `float(clipped.min()) == 0.0 and float(clipped.max()) == 1.0` plus a count claim such as `(raw_images == 0).mean() > 0.4` (a clipped stand-in has a large exactly-black background; a wrapped one does not).
- **[P2] nuisance data parameters with no claim over them** — Two more day-01 plants survived: the pen-wobble amplitude and the byte quantisation. `wobble = rng.normal(0.0, 0.18, ...)` set to std 0.00 makes all 50 samples of a class byte-identical (the uniform-data condition), and dropping `.round()` before `.astype(np.uint8)` silently truncates instead of rounding. Neither is covered by any claim.
  - proven by: `line 26: `rng.normal(0.0, 0.18, size=images.shape)` -> `rng.normal(0.0, 0.00, size=images.shape)`; separately line 28: `pixels.round().astype(np.uint8)` -> `pixels.astype(np.uint8)`` (stdout changed: True)
  - fix: Pin one data statistic that a degenerate generator would fail, e.g. `0.30 < float(x.std()) < 0.50` and `float(np.abs(x[0] - x[1]).max()) > 0.05` (two samples of one class must not be identical).

### day-02-backward-pass

- **[P1] 6 — constant fold: the printed number and the checked number are two independent copies of the same literal expression** — Line 51 prints `(2.0 - 5.0) ** 2` and `(4.0 - 5.0) ** 2`; line 116 checks `golf_ok = (2.0 - 5.0) ** 2 == 9.0 and (4.0 - 5.0) ** 2 == 1.0`. Both sides are literals and the check re-types the expression instead of reading the value that was printed, so the printed number is not covered by the claim it appears to back.
  - proven by: `line 51: `print("squared miss: guess 2 vs target 5 ->", (2.0 - 5.0) ** 2, "| guess 4 ->", (4.0 - 5.0) ** 2)` -> the same print with `abs(2.0 - 5.0)` and `abs(4.0 - 5.0)`` (stdout changed: True)
  - fix: Compute the printed values into named variables (`miss_2, miss_4 = (2.0 - 5.0) ** 2, (4.0 - 5.0) ** 2`), print those variables, and check `miss_2 == 9.0 and miss_4 == 1.0`. Same for `gate_ok`'s first two clauses, which re-assert the literals `z_fwd` and `upstream_delta` were just defined as.
- **[P2] untested strictness — no input sits on the boundary** — The ReLU gate is spelled `z_fwd > 0` in Part 2 and `(z1 > 0)` in backward(). Part 2's exhibit is `z_fwd = [2.0, -1.0, 3.0, -0.5]` and z1 is random float, so no value is ever exactly 0 and the strictness is unobservable.
  - proven by: `line 56: `gate = z_fwd > 0` -> `gate = z_fwd >= 0`` (stdout changed: False)
  - fix: Add a boundary element: `z_fwd = np.array([2.0, -1.0, 3.0, -0.5, 0.0])` with a matching `upstream_delta`, and pin the outgoing delta at that slot to exactly 0.0 — then `>` and `>=` are no longer the same program.

### day-04-training-loss-dropout

- **[P1] no-op trap #5 — a bias whose own code path is untestable (here: tolerances wider than the bias's whole effect)** — `b1 = np.zeros(HIDDEN)` and `b2 = np.zeros(CLASSES)`, and the loop updates them with `b1 -= lr * d_pre.sum(axis=0)` / `b2 -= lr * d_scores.sum(axis=0)`. Every numeric claim in the self-check is a ±0.05 tolerance (train_a 0.047, best_val_a 1.231, val_a 1.422, gap_a 1.375, gap_b 0.742), which is wider than either bias's total contribution over 40 epochs, so neither update line is covered by any claim.
  - proven by: `line 77: `W1 -= lr * (x.T @ d_pre);         b1 -= lr * d_pre.sum(axis=0)` -> `W1 -= lr * (x.T @ d_pre)` (deleting only the hidden-bias update). The same experiment on line 76's `b2 -= lr * d_scores.sum(axis=0)` also survives.` (stdout changed: True)
  - fix: Print and pin the learned biases after Run A, e.g. `abs(float(b1_final.std()) - <recorded>) < 1e-6` and `abs(float(b2_final.max()) - <recorded>) < 1e-6`, or tighten one loss pin to 1e-4 (the run is fully seeded, so an exact pin is available).
- **[P2] 9 — identity claim: three evaluations of one deterministic expression** — `steady_rows = [softmax_rows(hidden_eval @ W2 + b2)[0] for _ in range(3)]` has no random source, so `len({r.tobytes() for r in steady_rows}) == 1` ("eval mode must repeat") is true by construction for any weights and any model. The neighbouring clauses on flicker_rows are real; this one is padding.
  - proven by: `(no plant — static: the expression is re-evaluated unchanged three times, so no defect elsewhere in the file can make the three rows differ)` (stdout changed: False)
  - fix: Make eval mode's repeatability contingent on the code under test: build the three eval rows through the SAME code path the training loop's eval branch uses (a `predict(x, train_mode=False)` helper), so leaving dropout on in that helper makes the three rows differ.

### day-05-pytorch-version

- **[P1] claim printed but never asserted (the two runs' shared starting point)** — run_loop's docstring says zero_grad() "is the ONLY difference between the two runs", and Part 5 prints a 25-row loss/gradient table plus "loss rose in 10 of 24 steps, gradient size 1.46 -> 3.11". Nothing asserts the bug run starts from the same weights as the good run, and nothing pins `rises` or either `bug_grads` value — the only clauses are `rises >= 5` and `bug_grads[-1] > 2 * bug_grads[0]`. Day-06's equivalent section pins `rises == 5`, `bug_losses[-1] == 5.21676` and `bug_grads[-1] == 11.60718`, so day-05 is the weak sibling of a claim its own module already knows how to pin.
  - proven by: `line 184: `bug_losses, bug_grads = run_loop(fresh_model(), X, y, clear_gradients=False)` -> `run_loop(MLP(), X, y, clear_gradients=False)` (the bug run now starts from DIFFERENT weights, so zero_grad is no longer the only difference)` (stdout changed: True)
  - fix: Add to the claims dict: `bug_losses[0] == good_losses[0] and abs(bug_grads[0] - good_grads[0]) < 1e-9` (epoch 0 is identical by construction when only zero_grad differs), and pin the printed values: `rises == 10 and abs(bug_grads[-1] - 3.109) < 0.005`.
- **[P2] a claim that merely restates a library guarantee** — `"plain data with requires_grad False, and every weight with requires_grad True": X.requires_grad is False and weights_track_history` cannot fail for any model code: `torch.tensor(numpy_array)` never has requires_grad, and every `nn.Parameter` always does. There is no authoring mistake in this day that the clause protects against.
  - proven by: `(no plant — static: no single-line edit to the lesson's own code can falsify it; only editing the tensor constructor or the claim itself can)` (stdout changed: False)
  - fix: Either drop the clause or give it teeth the way day-06 does — e.g. assert that a tensor built inside `torch.no_grad()` from the weights has no `grad_fn` while the same expression outside does, which a wrongly placed `no_grad` in the loop would break.
- **[P2] 5 — entailed clause: a tight pin implies the loose clause beside it** — Four entailed clauses across the module, each of which can never be the failing one: day-05 `abs(untrained_accuracy - 1/CLASSES) < 0.05` (implied by the 0.07520±0.002 pin), `torch_lines * 20 < HAND_WRITTEN_LINES` (480 < 500, implied by `torch_lines == 24`), `abs(good_losses[0] - predicted_start) < 0.1` (implied by the 2.32391±0.002 pin); day-06 `LR < flip_lr < BIG_LR` (implied by `abs(flip_lr - 0.80486) < 2e-4`); day-03 `updates_done == planned_batches * N_EPOCHS` (implied by `updates_done == 192` with the module constants).
  - proven by: `(no plant — static: each loose clause is arithmetically entailed by the pin next to it, so no input can make the loose one false while the tight one holds)` (stdout changed: False)
  - fix: Delete the entailed clause, or replace the tight pin with the pedagogical inequality if the inequality is the point (e.g. keep `torch_lines * 20 < HAND_WRITTEN_LINES` and drop `torch_lines == 24`, so a refactor of run_loop does not spuriously fail).


## m02-the-neuron — cross-day collisions (10, 2 P0)

None of these fails any gate: every one lives in the seam between two files.

- **[P0] convention-split** (day-02-activations, day-03-layers-forward-pass) — The weight-layout convention flips between Day 2 and Day 3 for the SAME anchor pair of matrices, so the two days print two different answers for "the single combined layer". Day 2 uses the row-vector convention (x @ W1 @ W2, so W1's rows are INPUTS) and prints W1@W2 = [[7,2],[3,1]]. Day 3 uses the column-vector convention (W @ x + b, and declares the rule "rows = neurons, columns = inputs") and prints the combined grid B@A = [[1,2],[3,7]] from the numerically IDENTICAL pair. Neither file says a convention changed, and [[1,2],[3,7]] is not even the transpose of [[7,2],[3,1]] — it is the other multiplication order. The intermediates are mirrored too: Day 2 gets x_a@W1 = [1,-1], Day 3 gets A@u = [-1,1].
- **[P0] word-meaning-shift** (day-06-training-loop, day-08-learning-rate, day-07-optimizers) — The verdict words "healthy" / "crawls" attach to the same numeric learning rate on two different days with opposite meanings. Day 6 labels lr = 0.01 healthy and converging; Day 8 labels lr = 0.01 a partway crawl and puts 0.1 in the good slot. Both are correct for their own loss surface (Day 6's gradient carries an extra factor x=2, so its effective curvature is 8 vs the bowl's 2), but no day states that these labels are relative to the loss curvature — the learner is handed an absolute-sounding verdict on a number.
- **[P1] helper-lost-knobs** (day-05-gradients-backprop, day-06-training-loop) — Day 5 teaches the backward pass as four numbered rules whose Rule 1 IS the mechanism: the activation charges a toll (delta = incoming * relu_slope) before the weight's share is taken. Day 6 — the very next day, and the day that presents "the four steps" as THE training loop — computes the same gradient with the toll silently deleted, because its neuron silently has no activation at all. Day 1 called "decide" one of the neuron's three beats and Day 2 argued a bend must be there; Day 6 reintroduces a bend-less neuron as "one neuron" with no note.
- **[P1] word-meaning-shift** (day-04-loss, day-07-optimizers, day-08-learning-rate) — "Loss" changes referent mid-module. Days 4/5/6/9 define it as how wrong a guess is against a target, zero at a perfect prediction. Days 7 and 8 use a synthetic bowl in weight space with no prediction, no data and no target, and Day 8 states its bottom is at weight = 0. Neither day says the bowl is a stand-in for a loss surface rather than a loss.
- **[P1] word-meaning-shift** (day-06-training-loop, day-09-train-val-test) — Day 6 defines an iteration as one weight update and an epoch as one full pass over the training data (also glossed in Day 9's own glossary: "epoch | one full pass over the training split"). Day 9's artifact re-labels one increment of polynomial DEGREE — a capacity knob, not a pass over data — as an "epoch", and then reports the resulting degree as the early-stopping point.
- **[P1] convention-split** (day-02-activations, day-03-layers-forward-pass) — Day 2 explicitly rules out an argument, and Day 3 then uses exactly that argument as its proof. Day 2 states that showing one output changed is NOT evidence that no single matrix can reproduce the bent path, and switches to an additivity test for that reason. Day 3's collapse-break rests on a single inequality against one particular combined grid, yet its print and comment claim the general conclusion.
- **[P1] formula-under-many-names** (day-02-activations, day-03-layers-forward-pass) — One object — the single matrix that two bend-less layers collapse into — carries five names across two days: `one_layer` and `W1 @ W2` on Day 2, and `combined_grid`, `one_step` and `B @ A` on Day 3; the two weight matrices themselves are `W1`/`W2` on Day 2 and `A`/`B` on Day 3 despite being numerically identical, and the two-layer path is `two_layers` vs `no_bend`. Nothing in either file tells the learner these are the same demo.
- **[P2] misplaced-comment** (day-09-train-val-test) — The docstring numbers the shuffle fix as Trap 3. In the day's own lesson the shuffle is the cure for Trap 2 (the sorted slice); Trap 1 is peeking and the third trap is preprocessing leakage, which the artifact does not implement at all.
- **[P2] convention-split** (day-01-single-neuron, day-05-gradients-backprop) — Seven of nine days follow the same self-check shape: print a ❌ line naming the expected values, then fail hard. Day 1 has no ❌ branch at all (bare asserts, then an unconditional ✅). Day 5 has a ❌ branch that can never execute, because the asserts above it test the same two conditions with the same tolerances.
- **[P2] convention-split** (day-01-single-neuron, day-02-activations, day-03-layers-forward-pass, day-06-training-loop, day-07-optimizers, day-08-learning-rate) — Pure naming drift for identical objects and knobs: the prediction is `yhat` (Day 3) but `pred` (Days 4, 6, 9); the loss value is `L` (Day 5) but `loss` (Day 6); the iteration budget is `num_iters=50` (Day 6), `STEPS = 30` (Day 7), `steps=40` (Day 8); the learning rate is `lr` (Days 6, 8) but module-constant `LR` (Day 7); and `grid` names three unrelated objects — a parameter search space (Day 1), an input sweep axis (Day 2), and a weight matrix (Day 3's `combined_grid`, where "grid" is the day's core metaphor for W).


## m03-attention — cross-day collisions (12, 1 P0)

None of these fails any gate: every one lives in the seam between two files.

- **[P0] convention-split** (day-02-qkv, day-03-attention-scores, day-04-multihead, day-05-positional) — The score-matrix argument order is spelled KEY-first on Day 2 and QUERY-first on Days 3/4/5, and no day ever states the matrix form of Day 2's spelling. Day 2 only ever scores a single asker, so it can write the key on the left and stay numerically right; the natural generalisation of that spelling (K @ Q.T) is the TRANSPOSE of the grid Day 3 defines, and Day 5 separately proves the transposed read gives a different answer (asymmetry gap 3.0) without telling the learner which spelling produces which orientation.
- **[P1] convention-split** (day-03-attention-scores, day-04-multihead) — `slot N` is 1-BASED on Day 3 and 0-BASED on Day 4, for the same thing (a column of a word vector). Each day's code is internally consistent, so no gate can see it; only the comments collide.
- **[P1] identifier-two-objects** (day-02-qkv, day-03-attention-scores, day-04-multihead, day-05-positional) — `softmax` names two different functions. Day 2's takes no axis at all — it normalises the WHOLE array — and its docstring promises "shares that add up to 1" with no shape caveat. Day 4's and Day 5's normalise the LAST AXIS. Day 3 sidesteps the clash by renaming its own to `softmax_row`, which leaves the module with three spellings of one convention.
- **[P1] identifier-two-objects** (day-02-qkv, day-03-attention-scores, day-04-multihead, day-05-positional) — `scores` means the RAW (pre-sqrt(d_k)) grid on Days 2-3 and the ALREADY-SCALED grid on Days 4-5, and Day 4 goes further by calling a post-scaling array "raw scores". The raw/scaled pair also carries three different name pairs across the module.
- **[P1] misplaced-comment** (day-01-embeddings, day-02-qkv, day-05-positional) — Two comments claim an artifact came from an earlier day when it did not. Day 2 says its embeddings are Day 1's; Day 5 says its projection matrices are Day 2's. In both cases the numbers, the axis meanings and (for Day 5) even the shape differ. Day 2's claim additionally breaks the rule Day 1 exists to assert — one frozen row per word.
- **[P1] helper-lost-knobs** (day-02-qkv, day-03-attention-scores, day-04-multihead, day-05-positional) — Day 2 teaches `d_v` — the Value width — AS a mechanism, a free choice independent of d_k, and Day 3 keeps it (V is width 2 with d_k = 4). Day 4's reusable head then defines the Value as d_k-wide, asserts that in a self-check, and `d_v` never appears again on Day 4 or Day 5. No day says the free choice has been spent.
- **[P1] helper-lost-knobs** (day-03-attention-scores, day-04-multihead, day-05-positional) — Day 3 presents masking as beat 4 of the five-beat pipeline, then its own one-line formula omits it, and every later day's reusable `attention` helper has no mask argument and no note that one belongs there. Day 4 also labels the rewritten helper "reused unchanged" when it is neither unchanged nor mask-capable.
- **[P1] word-meaning-shift** (day-01-embeddings, day-02-qkv) — Day 1 teaches that a raw dot product is the misleading number — it grows with vector length — and that cosine is the fix. Day 2 then makes the raw dot product the match score, and blames the same big-score symptom on the WIDTH of the vector, fixing it with sqrt(d_k). Cosine is never mentioned again and no day says why it was dropped. The word "long" flips meaning between the two accounts.
- **[P1] word-meaning-shift** (day-01-embeddings, day-02-qkv, day-04-multihead, day-05-positional) — `slot` means a word's POSITION IN THE SENTENCE in Day 1's two headline limit statements, and a COLUMN OF A VECTOR everywhere else in the module — including three other places inside Day 1 itself. Day 5 then renames position to "seat".
- **[P2] formula-under-many-names** (day-01-embeddings, day-02-qkv, day-04-multihead, day-05-positional) — The projection matrices are spelled `Wq/Wk/Wv` in Day 2's code but `W_Q/W_K/W_V` in Day 2's own comments and in Days 4-5's code; and the same three objects are called "filter", "grid", "recipe" and "table" across days, while "table" already means the embedding matrix on Day 1 and "recipe" already means the attention formula on Day 5.
- **[P2] word-meaning-shift** (day-03-attention-scores, day-04-multihead) — `budget` is Day 3's word for the attention mass of 1.0 (nine uses, including the assert text), and Day 4's word for the parameter count.
- **[P2] convention-split** (day-02-qkv, day-03-attention-scores, day-04-multihead, day-05-positional) — Small spelling drifts for identical operations: the input matrix is `x` on Days 2 and 5 but `X` on Day 4; `d_k` is read as `Q.shape[1]` on Days 2/3/5 and `Q.shape[-1]` on Day 4; a row sum is `axis=1` on Day 3 and `axis=-1` on Days 4/5; the un-normalised exp inside softmax is called `slices`, `grown`, `exp_scores` and — on Day 5 only — `weights`, which is the module's word for the normalised shares.


## m04-first-model-mlp — cross-day collisions (15, 3 P0)

None of these fails any gate: every one lives in the seam between two files.

- **[P0] convention-split** (day-02-backward-pass, day-03-minibatch-loop, day-04-training-loss-dropout, day-06-why-pytorch) — The squared-error loss has two different reductions across the module and no day states the change. Days 02/04/06 average over every ELEMENT (denominator = err.size); day 03 SUMS over the 10 classes then averages over rows (denominator = batch) — a factor of exactly 10 on both the loss and the output-layer delta. Day 03 attributes its version to Day 2 ("Day 2's backward pass"), and Day 06 explicitly names day 03's spelling — dividing by rows instead of elements — "the classic wrong-denominator bug" and asserts it is "2x too big" on its own 2-column target.
- **[P0] word-meaning-shift** (day-04-training-loss-dropout, day-05-pytorch-version) — Day 04 defines cross-entropy as a function of PROBABILITIES and always feeds it softmax output. Day 05 hands the learner nn.CrossEntropyLoss and feeds it RAW scores, because torch applies log-softmax internally — but neither day-05's experiment.py nor its source.md ever states that rule. The word "cross-entropy" silently changes what it eats between the two days.
- **[P0] helper-lost-knobs** (day-01-mlp-mnist, day-03-minibatch-loop) — Day 03's init_params claims to be "He initialisation from Day 1: scale each weight by sqrt(2 / n_in)" and then multiplies W2 by an extra 0.5, so the output layer gets HALF the He spread day 01 taught as the mechanism and pinned in its own self-check. The 0.5 is load-bearing (removing it breaks day-03's pinned numbers), so it is a deliberate tuning presented as the rule it violates.
- **[P1] identifier-two-objects** (day-01-mlp-mnist, day-03-minibatch-loop) — `make_stand_in_digits` names a different function on each of the two days that define it: the first positional argument means examples PER CLASS on day 01 and TOTAL examples on day 03 (a 10x difference in dataset size), the return arity is 2 vs 3, and the pixel dtype/range is uint8 0-255 vs float 0.0-1.0.
- **[P1] convention-split** (day-01-mlp-mnist, day-02-backward-pass, day-03-minibatch-loop, day-05-pytorch-version, day-06-why-pytorch) — Days 01-03 build the hand-written weights as (in, out) — W1 is (784, 128), W2 is (128, 10) — and day 02 asserts that shape. Day 05's comments attribute the TRANSPOSED torch layout to those hand-written weights, and day 06 labels an (OUT, IN) layout "the day-2 way". The shape order for the module's central object is spelled both ways, and the wrong way is the one credited to the earlier days.
- **[P1] convention-split** (day-01-mlp-mnist, day-02-backward-pass, day-03-minibatch-loop) — `forward` is defined three times with three different signatures and three different return tuples, and day 02 inserts `loss` as the FIRST return value, shifting every other slot by one relative to days 01 and 03.
- **[P1] formula-under-many-names** (day-01-mlp-mnist, day-02-backward-pass, day-03-minibatch-loop, day-04-training-loss-dropout, day-05-pytorch-version) — The two objects the whole module is built on carry four names each. The layer-1 pre-activation is `hidden_pre` / `z1` / `pre` / `Z1`; the output-layer backprop delta is `delta2` / `d_out` / `d_scores` / `dZ2` — even on the days whose own comments say they are reusing the earlier day's pass verbatim.
- **[P1] convention-split** (day-03-minibatch-loop, day-04-training-loss-dropout, day-05-pytorch-version) — The same three-constant tuple assignment is written in two different orders. Days 03 and 05 use `PIXELS, HIDDEN, CLASSES`; day 04 swaps the last two to `PIXELS, CLASSES, HIDDEN` while keeping the values in the new order, so the line looks the same and means something different.
- **[P2] word-meaning-shift** (day-01-mlp-mnist, day-03-minibatch-loop, day-04-training-loss-dropout, day-05-pytorch-version) — `PIXELS` is 784 on days 01/02/03/05 and silently 64 on day 04, so the module's spine architecture changes for exactly one day and changes back. Day 04's source.md never states an input size at all (grep for 784|8x8|64 pixel returns nothing), and day 05 then asserts the network has been the same all along.
- **[P2] word-meaning-shift** (day-01-mlp-mnist, day-04-training-loss-dropout, day-05-pytorch-version) — Day 01 asserts as a checked claim that "pixels" are bytes scaled to span exactly 0.0-1.0. Day 04 calls unbounded `prototype + N(0,1)` values "pixels" with no note; day 05 at least flags its own deviation. The rule day 01 pinned is dropped without ever being retired.
- **[P2] formula-under-many-names** (day-01-mlp-mnist, day-04-training-loss-dropout, day-05-pytorch-version) — The model's size is counted under three names and two different definitions: "knobs"/"parameters" including biases (day 01), "weights" excluding biases (day 04), "numbers to learn" including biases (day 05).
- **[P2] formula-under-many-names** (day-01-mlp-mnist, day-02-backward-pass, day-03-minibatch-loop, day-04-training-loss-dropout, day-05-pytorch-version) — "logits" is taught as the name for the 10 raw output scores on day 01 (14 uses) and then never used again on any later day — the same object is `out`, `Z2` or `scores`. Day 03 even writes the word in a comment and then binds the value to `out`.
- **[P2] identifier-two-objects** (day-02-backward-pass) — Inside the one file that teaches `h` as the saved hidden activation (the breadcrumb the weight gradient multiplies), `h` is also the name of the finite-difference step size in the gradient checker.
- **[P2] convention-split** (day-01-mlp-mnist, day-02-backward-pass, day-03-minibatch-loop, day-04-training-loss-dropout, day-05-pytorch-version, day-06-why-pytorch) — Four cosmetic spellings drift across the six days: the He spelling (US vs UK), the ReLU zero literal, the sum-axis spelling, and the position of `rng` in the data-factory argument list.
- **[P2] misplaced-comment** (day-05-pytorch-version) — Day 05's line-count comparison attributes the lesson's ~500 hand-written lines to "days 2-4", but the lesson says the last FOUR days (01-04) — day 01 is where the hand-written forward pass and He init are built.


## What this means

- The 42 surviving plants are MISSING DEFENCES, not wrong output: these artifacts print
  correct numbers today. The risk is that a learner who edits one gets false reassurance,
  and any future edit can break them silently.
- The 6 cross-day P0s ARE live teaching defects in the current state.
- m05a / m05b / m06 (20 days, wave 2) received exactly this treatment when built and are
  not implicated.
- Every failure class here is now named in the `frontier-refactor-qa` skill, and
  `sessions/_compiler/workflows/doing_leg_build.js` runs the cross-day pass by default —
  the practices these 20 days lack are the ones that pass produced.
