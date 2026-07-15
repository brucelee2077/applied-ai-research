# The loss, the overfitting turn, and dropout: why the held-out number is the only one you can trust

Day 3 drove a training loss down and made the staff-level point that a falling
*training* loss is a fact about the optimizer, not the model. Day 4 is where that
warning becomes operational. It names the number the loop is minimizing (the
**loss**), picks the right one for classification (**cross-entropy**), and then
shows the failure that a training-loss-only view is structurally blind to:
**overfitting**. The cure — **dropout**, specifically inverted dropout with its
`1/(1−p)` train-time scaling — is built from scratch and, critically,
*measured*, because dropout is one of those techniques where the folklore ("it
regularizes") is easy to repeat and easy to get wrong (wrong rate, wrong data
regime, left on at eval).

Every number below comes from `experiment.py`: a seeded, float64, from-scratch
NumPy MLP (no autograd, no framework), `30 → 512 (ReLU) → 4`, trained on a
**nonlinear** 4-class task — a `(radius-band × angle-band)` decision boundary
hidden among 28 distractor-noise features, with 10% of labels randomly flipped.
That combination — genuine nonlinear signal, real distractor and label noise, a
wide net, and only `N_train = 600` — is deliberately chosen: it is a regime where
the net *can* memorize the training noise and where dropout *can* actually help.
A trivially separable task would collapse to loss 0 and hide every effect this
note is about.

## The mechanism: one scalar, and cross-entropy's shape

The loss is one number for "how wrong were these guesses," and it is one number
for a mechanical reason: the backward pass needs a single scalar to push down.
For classification the standard choice is cross-entropy, which for one example is
`−log(p)` where `p` is the probability the model assigned to the *correct* class.
The shape is the whole point — the penalty is near 0 when `p → 1`, modest
(`−log 0.5 ≈ 0.69`) when the model is unsure, and unbounded as `p → 0`. Confident
*and wrong* (tiny `p` on the true class) is punished hardest, which is exactly the
signal you want: the loop fixes its most arrogant mistakes first. Training then
minimizes the **average** cross-entropy over the training set — the mean, not the
sum, so the objective is batch-size-independent.

The subtlety a staff reader must hold: that objective only ever touches the
*training* data. Nothing in it references data the model has not seen. So the
loop can drive the training loss arbitrarily low without the model being any good
— which is precisely what the next section measures.

## The overfitting turn: the training loss lies, the held-out loss tells the truth

Train the over-capacity net with no dropout and watch both losses per epoch. The
training loss falls monotonically toward zero. The held-out loss does something
the training loss never reveals: it bottoms out, then turns and climbs.

```
epoch |  train loss  | held-out loss
    1 |      1.5395 |      1.6906
   34 |      0.7868 |      1.5187  <-- turn
  201 |      0.1448 |      1.8586
  401 |      0.0476 |      2.1633
  800 |      0.0169 |      2.4560
```

The **overfitting turn is at epoch 34**, where held-out loss bottoms at `1.5187`.
After that, the training loss keeps falling (`1.5395 → 0.0169`) while the held-out
loss *rises* `+0.9373` to `2.4560`. The final generalization gap
(held-out − train) is `+2.4391`: the model reached a near-perfect training loss
of `0.0169` and a held-out loss of `2.4560` — worse, at the end, than the
uniform-guess baseline of `ln(4) = 1.3863`. A practitioner watching only the
training curve would have declared victory somewhere around epoch 400 and shipped
a model that had been *degrading on unseen data for 366 epochs*.

The mechanistic cause is the standard one: capacity (512 hidden units, ~17.9k
weights) far exceeds what 600 noisy points can pin down, so the cheapest way to
shrink training loss past the turn is to memorize individual points — including
the 10% of labels that are pure noise. Memorizing noise cannot transfer, so
held-out loss rises.

## The cure, measured: dropout narrows the gap — but it is not free

Re-run the identical net and optimizer with inverted dropout at `p = 0.5` on the
hidden layer. Every effect the theory predicts shows up in the numbers:

```
best held-out loss  : no-dropout=1.5187   dropout=1.3258   (IMPROVED +0.1929)
best held-out acc   : no-dropout=0.3930   dropout=0.4753   (IMPROVED +0.0822)
FINAL held-out loss : no-dropout=2.4560   dropout=1.4237   (IMPROVED +1.0323)
final generalization gap : no-dropout=+2.4391  dropout=+1.2746  (NARROWED +1.1645)
```

Dropout improved best held-out loss by `0.1929`, lifted best held-out accuracy by
`+0.0822` (0.393 → 0.475), and cut the *final* held-out loss almost in half
(`2.4560 → 1.4237`). The final gap narrowed by `1.1645`. That is the regularizer
working: by randomly zeroing half the activations each step, no single neuron can
become a crutch, so the network is pushed toward distributed features that
transfer instead of a brittle memorized lookup.

The trade-off is visible in the same run and must be stated: **dropout makes the
training loss worse on purpose.** The final training loss with dropout is `0.1491`
versus `0.0169` without — an order of magnitude higher. Practice is deliberately
harder. This is the design tension: a larger `p` fights memorization harder but
handicaps more of the network at once, so it can *under*-fit if you overdo it.
`p` is the one knob, and you tune it by watching the train-vs-held-out gap, never
the training loss alone. `p = 0.5` is the standard hidden-layer starting point,
but "add dropout" is not automatically a win: too small a `p` barely regularizes
and too large a `p` under-fits, and whether it helps at all depends on the data
regime (a near-linearly-separable task with little noise gives it nothing to fix).
That is exactly why this experiment fixed a regime where the effect is real — a
nonlinear boundary with genuine label noise — and then *measured* the gain rather
than asserting it.

## The `1/(1−p)` scaling, and why eval needs no adjustment

Zeroing half a layer's outputs halves its total signal, so training and test
would see systematically different magnitudes unless something corrects it.
Inverted dropout puts the correction in *training*: divide the survivors by
`(1 − p)`.

```
full layer            = [4.0, 4.0, 4.0, 4.0]  sum = 16.0
after drop (no scale) = [4.0, 0.0, 4.0, 0.0]  sum = 8.0   (half of 16)
survivors / (1-0.5)   = [8.0, 0.0, 8.0, 0.0]  sum = 16.0  (back to full)
eval (all on, no scale) = [4.0, 4.0, 4.0, 4.0] sum = 16.0
```

Train-time sum (`16`) equals eval-time sum (`16`): the magnitudes agree, so **eval
is a plain full-strength forward pass with no scaling.** On the real 512-unit
layer this holds in expectation too — mean hidden activation `0.5789` (full)
versus `0.5788` (inverted-dropout), ratio `1.000`. This is why the scaling lives
in training: it buys a do-nothing-extra eval pass.

## The failure mode that ships silently: dropout left on at eval

The classic bug is not a crash — it is a model in `train` mode when you evaluate,
so dropout keeps randomly zeroing neurons while you predict. Nothing errors; the
numbers are just quietly, randomly wrong. The tell is that identical inputs give
different answers across runs. Evaluating the trained dropout model on the *same
fixed* held-out set five times, with dropout mistakenly left on versus in proper
eval mode:

```
dropout LEFT ON at eval (bug) : [0.403, 0.4158, 0.407, 0.4183, 0.4073]
    spread across runs        = 0.0152   (same input, different answers -- NOISY)
proper EVAL mode (dropout off): [0.4713, 0.4713, 0.4713, 0.4713, 0.4713]
    spread across runs        = 0.0000   (IDENTICAL every run -- STABLE)
```

Two things to notice. First, the buggy accuracy *jitters* by `0.0152` across runs
on data that never changed — that non-determinism is the diagnostic. Second, the
buggy accuracy (~0.41) is also *lower* than the correct eval accuracy (`0.4713`),
because a randomly crippled network is strictly weaker than the full one at test
time. The senior habit that catches this in one line: before reporting any
metric, confirm the model is in eval mode and that scoring the same fixed data
twice returns the *exact* same number. A non-zero spread means a train-only
behavior (dropout, or batchnorm's running stats) is still active.

## The ceiling: what dropout cannot do

Dropout regularizes; it cannot manufacture information the training set never
contained. In this run, even the best dropout model tops out at `0.4753` held-out
accuracy — well above the `0.25` random floor, but capped by the 10% irreducible
label noise and the finite `N_train = 600`. Dropout changed *how* the model leaned
on the data it had; it could not hand the model data it was never given. Closing
that kind of gap is a data problem (collect more, or more varied, examples), not
a regularization problem — which is why the honest summary of the whole day is:
the train-vs-held-out **gap**, not the training loss, is the number you trust, and
no regularizer excuses you from measuring it.

## What the run establishes

The mini-batch loop from Day 3 optimizes; Day 4 shows that optimization and
generalization are different things you must measure separately. On a from-scratch
NumPy MLP: the training loss fell to `0.0169` while held-out loss turned up at
epoch 34 and climbed to `2.4560` (gap `+2.4391`); inverted dropout at `p = 0.5`
cut that final gap to `+1.2746` and lifted best held-out accuracy `0.393 → 0.475`,
at the cost of a deliberately higher training loss (`0.0169 → 0.1491`); and
leaving dropout on at eval made accuracy jitter by `0.0152` on fixed data that a
proper eval pass scored identically every time. Not one of those facts was visible
in the training loss alone.
