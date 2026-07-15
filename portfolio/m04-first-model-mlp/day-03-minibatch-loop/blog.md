# The mini-batch loop: why a falling training loss is evidence of an optimizer, not a model

Day 2 produced a corrector — a forward pass that guesses, a backward pass that
hands back one gradient per weight. That was a single correction on a single
batch. The engineering content of Day 3 is small to state and large in
consequence: put that correction inside a loop over the data, and for the first
time the training loss *moves*. The interesting claim for a staff-level reader is
not "the loss falls" — it is that a falling **training** loss is a fact about the
*optimizer*, and says almost nothing about the *model*. This note builds a real
NumPy mini-batch loop (no autograd, no framework), drives the training loss down,
and then breaks it three ways to show what the loss curve does and does not
license you to conclude. Every number below comes from `experiment.py`, seeded,
float64, on a 3-class problem with heavily overlapping clouds (mean spread 1.0,
noise std 3.0) so the loss cannot trivially collapse to zero.

## The mechanism: four steps on repeat, over shuffled slices

The loop reuses the Day 1-2 forward and backward passes verbatim. The only new
code is the control flow around them:

```
for epoch in range(epochs):
    idx = rng.permutation(N)              # shuffle the INDICES every epoch
    for start in range(0, N, batch_size): # slice into consecutive chunks
        b  = idx[start:start+batch_size]   # this batch (the LAST one may be ragged)
        loss, cache = forward(p, X[b], Y[b])          # guess
        grads = backward(p, X[b], Y[b], cache, "mean") # which way (averaged)
        p = p - lr * grads                            # one step
```

The vocabulary is exact arithmetic, not hand-waving. For the lesson's canonical
case, `N = 60,000`, `batch_size = 32` gives `60,000 // 32 = 1,875` iterations per
epoch, and it divides evenly. This experiment deliberately uses a size that does
*not* divide evenly:

```
this run : N=1000, batch_size=32 -> 31 full batches + 1 ragged batch of 8 = 32 iterations/epoch
batch sizes sum back to N : 1000 == 1000  -> OK (leftovers still train)
```

The **ragged final batch** of 8 is the wrinkle worth naming: a loop that
hard-codes `batch_size` anywhere (a reshape to `(32, ...)`, a divide by a literal
`32`) either crashes or silently computes wrong numbers on that last short batch.
The `31 full + 1 ragged of 8 = 32` sizes summing back to exactly `1000` is the
check that the leftovers are not dropped.

Run the loop and the training loss falls hard. The per-batch loss inside a single
epoch is bumpy (that is the mini-batch gradient noise — `min=2.478, max=13.625`
in epoch 1 alone), but the per-epoch mean trends cleanly down:

```
epoch  1: 6.8972
epoch  5: 2.8106
epoch 10: 1.3641
epoch 20: 0.3901
epoch 40: 0.0025
```

From `6.8972` to `0.0025` — a drop of `6.8947`, and far below the uniform-guess
baseline of `ln 3 = 1.0986`. That is the first live proof the whole Day 1-2
machinery connects and learns. It is also the last honest thing the training loss
tells you on its own.

## Failure mode 1: forgetting to shuffle sorted data — silent, no crash

Sort the data by label (`000...111...222`) and run the identical optimizer with
`shuffle=False`. Nothing throws. But now every batch of 32 is almost entirely one
class, so consecutive batches yank the weights toward one class and then another.
The signal is real and measurable:

```
final training loss  shuffled=0.0025   un-shuffled(sorted)=0.4971
within-epoch-1 batch-loss swing (max-min)  shuffled=11.147   un-shuffled(sorted)=18.133
```

The sorted run's within-epoch batch loss swings wider (`18.133` vs `11.147` in
epoch 1 — single-class batches whipsaw the weights), and it settles at a final
training loss of `0.4971` against the shuffled run's `0.0025` — **196x higher**,
with no exception, no NaN, no shape error. This is the canonical *silent* bug: a
suspiciously jerky loss on data that happens to be sorted is a data-ordering
smell, not a learning-rate problem. The fix is to shuffle the indices every epoch
— not to fiddle with the step size. A staff engineer who reaches for `lr` first
here wastes hours; the tell is that the loss is jerky *and* the data was never
shuffled.

## Failure mode 2: summing the batch gradient makes batch_size a hidden lr

When you reduce the per-example loss and gradient over a batch, you take the
**mean**, not the sum. This is a one-word change in code (`/ B` versus `/ 1`) and
a large change in behavior, because it couples the effective step size to
`batch_size`. Summing makes the gradient roughly `batch_size` times larger, so at
the same nominal learning rate the *real* step is `~32x` too big:

```
MEAN gradient (correct): epoch losses = [6.897, 4.278, 4.423, 2.729, 2.811]
SUM  gradient (bug)    : epoch losses = [16.906, 17.711, 17.47, 18.08, 17.208]
```

With the mean, `lr = 0.1` converges (`6.897 -> 2.811` over five epochs). With the
sum — same `lr`, same data, same seed — the loss *diverges* and sticks around
`17`, because the effective step overshoots the valley every time. The
consequence for the engineer: if you sum, your learning rate silently becomes a
function of your batch size, so the moment you change `batch_size` (say to fit
memory) your previously-tuned `lr` breaks. Taking the mean keeps `lr` a clean,
orthogonal knob you can tune once and reuse across batch sizes. That decoupling is
the entire reason the convention exists.

## Design trade-off: the batch_size dial — noise versus frequency versus memory

`batch_size` is the dial that sets where you sit between the two ancestors of
mini-batch descent. Full-batch (all `N`) gives an exact, smooth gradient but only
*one* update per pass over the data — on 60,000 images that is one step per epoch,
and it must hold all `N` examples' activations in memory at once. Single-example
SGD (`batch_size = 1`) updates `N` times per epoch but each gradient is a noisy
one-example estimate. Mini-batch averages a handful: in this run, 32 examples of
evidence per step instead of 1, which is why the per-epoch mean trends down
smoothly even though the per-batch loss swings between `2.478` and `13.625`.

The trade-off has a sharp edge that the noise is not purely a cost: small batches
give a shakier gradient *and often generalize better*, because the jitter knocks
weights out of sharp minima. Large batches give a smoother, more accurate gradient
but fewer updates per epoch and higher memory. In practice `32`-`256` is the
sweet spot. The important staff-level point is that `batch_size` and `lr` are
coupled through the reduction: only because we take the *mean* can we treat them
as separable at all.

## The ceiling: a low training loss is not a good model

Here is the claim that matters most, and the one the loss curve cannot make on its
own. Train the same architecture longer (200 epochs, a wide 256-unit hidden layer,
only 1000 training points) and then measure the loss on 4000 held-out points drawn
from the *same distribution* the model never trained on:

```
TRAINING  loss (data it practiced on) = 0.0003
HELD-OUT  loss (data it never saw)    = 4.5263
gap (held-out - train)                = +4.5260
```

The loop did exactly what a loop does: it drove the **training** loss to `0.0003`,
essentially zero. And the model is **17,732x worse** on unseen data. The mini-batch
loop is an *optimization procedure* — a machine for driving the training loss down
— and that is *all* it is. A perfectly-wired loop with a near-zero training loss
can be a badly overfit model that memorized 1000 noisy points. The training loss
alone gives you no way to tell the difference; both a great model and a memorizing
one show a falling training loss. The `+4.5260` gap is invisible until you hold out
data and measure it. This is why "our training loss is going down" is not a claim
that survives a design review, and why the very next step is a train/validation
split.

## What this evidence shows and does not show

It shows a from-scratch mini-batch loop drives the training loss `6.8972 ->
0.0025` (it learns); that dropping the shuffle on sorted data silently costs
`196x` in final training loss with no crash; that summing instead of averaging the
gradient turns a converging run (`-> 2.811`) into a diverging one (`~17`) at the
same `lr`; and that a training loss of `0.0003` coexists with a held-out loss of
`4.5263` — the overfitting gap the loop is blind to. It does *not* show how to
close that gap: measuring generalization needs the held-out split, and shrinking
it needs regularization (dropout, weight decay) — both of which belong to the
training-loop days that follow. The habit to carry forward is the discipline of
distrust: a falling training loss earns you the right to believe your loop is
wired, and nothing more.
