# The gradient check: how a hand-wired backprop proves itself before it wastes a run

Day 1 left us with a correctly-wired 784 → 128 → 10 MLP whose 101,770 knobs were
still random, so it guessed at chance. Day 2 wires the corrector: a loss scalar,
its gradient for every weight, and one reverse sweep that computes them all by
reusing the forward pass. The interesting engineering claim is not "backprop
works" — it's that backprop is *easy to get subtly wrong and impossible to
notice*, so the only defensible way to trust a hand-derived gradient is to check
it against a slow-but-honest finite-difference gradient. This note produces that
check on a real network and then deliberately breaks it. Every number below comes
from `experiment.py`, seeded for reproducibility, in float64.

## The mechanism: one reverse sweep, four local slopes

The forward pass on a batch of 32 caches three arrays — `z1`, `h`, `probs`. The
backward pass reuses them and never recomputes a per-weight slope:

```
dlogits = (probs - y_true) / B      # softmax + cross-entropy collapses to this
dW2     = h.T @ dlogits             # (128,10) — local slope of the last linear step is h
dh      = dlogits @ W2.T            # hand the blame back through W2
dz1     = dh * (z1 > 0)             # through the ReLU gate: blame only where the unit was ON
dW1     = x.T @ dz1                 # (784,128) — local slope of the first linear step is x
```

The load-bearing property is the shape contract: **each gradient must have the
same shape as its weight**, because the update rule subtracts them elementwise.
The run confirms it exactly:

```
dW1 (784, 128) vs W1 (784, 128) -> MATCH
dW2 (128, 10)  vs W2 (128, 10)  -> MATCH
```

A transpose in the wrong place would flip one of these to `(128, 784)` and either
crash or — worse — silently broadcast. Shape agreement is necessary but *not
sufficient*: a gradient can have the right shape and the wrong value. That is what
the numerical check is for.

## The proof: analytic vs finite differences

The finite-difference (numerical) gradient is the pre-backprop ancestor. For one
weight entry `w`, nudge it by `ε` and re-run the forward pass:
`grad ≈ (loss(w+ε) − loss(w−ε)) / (2ε)`. It costs one forward pass per weight, so
it is useless for training 101,770 knobs — but it is almost impossible to derive
wrong, which makes it the perfect *test*. Comparing our fast analytic backprop
against it on 12 sampled entries per matrix (`ε = 1e-6`):

```
W2 (dW2 = h.T @ dlogits)   max|analytic - numerical| = 2.975e-10
W1 (through ReLU gate)     max|analytic - numerical| = 2.849e-10
```

A max absolute disagreement of `2.975e-10` across every sampled entry means the
analytic and numerical gradients **agree to ~10 decimal places** — proof the chain
rule was composed correctly through both the linear and ReLU steps. The residual
is not backprop error; it is the O(ε²) truncation of the central difference plus
float64 round-off, well below the `ε = 1e-6` floor. The reviewer's takeaway: a
central-difference check that lands near `1e-7`–`1e-10` is a *pass*; one that lands
at `1e-2` or higher is a bug, not noise.

## Failure mode: the silent gradient bug that never crashes

Now break exactly one term. Drop the ReLU gate — `dz1 = dh` instead of
`dz1 = dh * (z1 > 0)` — so blame leaks backward through hidden units that were
*off* during the forward pass. This is the single most common backprop bug: the
gate is trivial to forget because the code still runs, still produces
correctly-shaped gradients, and still lowers the loss *most* of the time. Re-run
the identical check on the identical batch:

```
W1 (gate dropped -> BUG)   max|analytic - numerical| = 2.437e-02
```

The disagreement jumps from `2.849e-10` to `2.437e-02` — the same check is now
**8.2e+07× worse**. Nothing threw. No stack trace, no NaN, no shape error. The
only signal that the gradient is wrong is the number the check prints. This is the
whole reason gradient checking is a discipline and not a formality: an incorrect
gradient produces a descent direction that is *plausible but wrong*, so the loss
often still trends down on easy batches and the bug survives into an overnight run
before anyone notices the model plateaus early. You catch it in seconds on a tiny
net, or you catch it in hours of wasted compute — the physics of the failure is
identical either way; only the cost differs.

## Design trade-off: analytic speed vs numerical trust — and the learning rate

Two trade-offs sit on top of this mechanism.

**Analytic vs numerical gradient.** The analytic backward sweep is fast — one
pass, all 101,770 gradients — but easy to get subtly wrong (a sign flip, a missing
gate, a transpose). The numerical gradient is trustworthy to ~10 decimals but
pays one forward pass *per weight*, which is why nobody trains with it. Standard
practice is exactly what this experiment does: use the numerical gradient once, on
a tiny net, as a *check*, then trust the fast analytic one for real training. You
buy correctness insurance for the price of ~24 extra forward passes.

**The learning rate is a knob, and its sign of effect is not free.** Backprop
gives the *direction*; the update `param = param − lr × gradient` needs a *size*.
The run shows both regimes on the same correct gradient and the same batch:

```
right-sized lr=0.05  loss 2.620411 -> 2.150728  (delta -0.469683)
too-large   lr=0.5   loss 2.620411 -> 9.245406  (delta +6.624995)
```

At `lr = 0.05` a single step cuts the loss by `0.47` — the corrector works. At
`lr = 0.5` the *same correct gradient* overshoots the valley and the loss *climbs*
by `6.62`. This is the subtle point for a staff-level reader: a rising loss does
**not** imply a wrong gradient. Here the gradient passed the check to ten decimals,
yet a 10× learning rate turned a descent into a divergence. Diagnosing training
requires separating the three silent failures — a wrong gradient (caught by the
check), a learning rate too large (loss climbs with a correct gradient), and
vanishing/exploding gradients (magnitudes drift to ~0 or blow up across layers).
None of them crashes. You separate them by watching gradient magnitudes and the
loss curve, not by waiting for an exception.

## What this evidence does and does not show

It shows the backward pass is wired correctly (shape contract, ~10-decimal
agreement with finite differences), that a one-term omission is a silent
`8.2e+07×` regression the check catches, and that a correct gradient with a bad
learning rate still diverges (`+6.62` at `lr=0.5` vs `−0.47` at `lr=0.05`). It does
*not* show convergence — this is one step on one batch of 32, not a training run.
It also does not exercise vanishing/exploding gradients, which need depth to
manifest; those are the failure the ReLU-vs-sigmoid choice and gradient clipping
address, and they belong to the training-loop days that follow. What you can carry
forward is the habit: derive the gradient, then prove it against the ancestor
before you spend a GPU-hour trusting it.
