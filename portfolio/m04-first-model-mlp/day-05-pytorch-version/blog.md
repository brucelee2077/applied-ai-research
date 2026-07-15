# The same model in PyTorch: proving a re-expression, not a new algorithm

Days 1–4 built an MLP by hand in NumPy: the forward pass, the hand-derived
backward pass, the mini-batch loop, cross-entropy, dropout. Day 5 rebuilds the
*exact same* `784 → 128 → 10` model in PyTorch and makes the claim that the whole
week has been building toward — the claim a staff engineer must be able to *prove*
rather than assume: **PyTorch is a re-expression of that NumPy model, not a new
algorithm.** `nn.Module` is a container, `nn.Linear` is a pre-built weight matrix,
`loss.backward()` is autograd running the chain rule, and `optimizer.step()` is
the update line. None of it changes what the model *can* learn. It changes the
labor, not the power.

That claim is only worth stating if it is *falsifiable*, so `experiment.py` puts
it on trial. It builds the MLP twice — once in PyTorch as an `nn.Module`, once in
from-scratch NumPy with a **hand-derived** backward pass (no autograd in the NumPy
path) — and checks equivalence at four levels: the forward output, the gradients,
the whole training trajectory, and the one failure mode the framework introduces.
Everything is `float64` and seeded, so the numbers below are reproducible.

## The one detail that trips everyone: the `(out, in)` transpose

Before the outputs can agree, the weights have to be copied across correctly, and
that copy is where the first subtlety lives. Your NumPy `W1` was shaped
`(784, 128)` — `(in, out)` — and you computed `x @ W1 + b1`. PyTorch's `nn.Linear`
stores the *same* weights **transposed**:

```
model.fc1.weight.shape = (128, 784)   (out,in) -- the TRANSPOSE of the numpy W1 (784, 128) (in,out)
```

`nn.Linear` holds `(out, in)` and computes `x @ W.T + b` under the hood. It is the
exact same linear map, laid out the other way around. Seeing `(128, 784)` when you
expected `(784, 128)` is correct, not a bug — and the equivalence check only works
if the copy respects it (`W_numpy = W_torch.T`). This is the kind of shape detail
that produces a "PyTorch predicts a different digit" bug when a port is done
carelessly; the fix is not "PyTorch is different," it is "I transposed wrong."

## Forward equivalence: same weights + same input → same output

With the weights copied across (transpose handled), feed the identical 64-example
batch through both models and compare the 10 output scores per example:

```
max |torch_logits - numpy_logits| over a 64x10 batch = 3.886e-16
predicted-class agreement = 64/64 examples
```

The logits agree to `3.886e-16` — that is machine epsilon for `float64`, i.e. they
are the same number down to floating-point round-off from a different order of
operations, not a different computation. All 64 examples pick the same class. This
is the calculator test from the lesson made literal: same input, same answer to
the last representable digit means the same machine inside. One matching example
is a hint; 64 matching to `1e-16` is proof.

## Gradient equivalence: `loss.backward()` *is* the Day-2 chain rule

Forward agreement is necessary but not sufficient — two models could agree on the
forward pass and still learn differently if their gradients differ. So the harder
test: compute the gradient of the same cross-entropy loss two ways. The PyTorch
path is one `loss.backward()` call. The NumPy path is the ~30 lines of hand-derived
chain rule from Day 2 (`dlogits = probs − onehot`, `dW2 = h.T @ dlogits`,
`dh = dlogits @ W2.T`, ReLU-gate, `dW1 = x.T @ dz1`, …). The loss and every weight
gradient:

```
loss:  numpy CE = 2.313130   torch CE = 2.313130   |diff| = 1.031e-11
grad W1: max |torch.grad - numpy hand-derived grad| = 1.041e-17
grad b1: max |torch.grad - numpy hand-derived grad| = 5.204e-18
grad W2: max |torch.grad - numpy hand-derived grad| = 4.163e-17
grad b2: max |torch.grad - numpy hand-derived grad| = 2.776e-17
```

Every gradient matches to `~1e-17`. This is the mechanistic core of the whole day:
`loss.backward()` did not do anything clever or different — it walked the recorded
autograd graph backward and applied the exact same chain rule you derived by hand,
producing the exact same numbers. The reason the forward pass returns *raw scores*
(no softmax) is that `nn.CrossEntropyLoss` folds softmax in — and it does so
*stably*, computing `log-softmax` in one fused, numerically-safe step rather than
`log(softmax(x))`, which is where a hand-rolled version overflows for large logits.
The NumPy path here mirrors that (subtract-max before `exp`), which is why the two
losses agree to `1e-11` and not just `1e-3`.

## Training equivalence: the same trajectory, the same accuracy

Static agreement on one batch still leaves room for the two loops to *diverge over
training* if the update rule differs. So train both from the **same initial
weights**, with the **same SGD rule** (`weight -= lr * grad`, `lr = 0.1`), fed the
**same shuffle order** each epoch, and compare the per-epoch loss curves and the
final held-out accuracy:

```
train loss  epoch 1  : torch=0.3961   numpy=0.3961
train loss  epoch 15 : torch=0.0009   numpy=0.0009
max per-epoch loss-curve difference over 15 epochs = 1.976e-12
final TEST accuracy : torch=1.0000   numpy=1.0000   |diff|=0.000e+00
```

The two loss curves stay within `1.976e-12` of each other over 15 epochs and land
on **identical** test accuracy. `torch.optim.SGD(model.parameters(), lr=0.1)` is
your Day-3 update line as an object; `optimizer.step()` sweeps every parameter and
applies `weight -= lr * weight.grad`. The five-line PyTorch loop
(`zero_grad → forward → loss → backward → step`) maps one-to-one onto the
hand-written loop, and the numbers confirm it produces not just a similar model
but — to floating-point precision — the *same* model. (The task here is a
self-contained synthetic 10-class problem at MNIST shape, chosen so the script
needs no dataset download; the equivalence claim is a statement about the two code
paths, independent of which data flows through them. The test accuracy of `1.0000`
reflects a cleanly separable synthetic task, not a claim about MNIST difficulty —
what matters is that both implementations reach it *identically*.)

## The failure mode the framework adds: forgetting `zero_grad()`

PyTorch buys the automation with one genuinely new obligation that has no analog in
the NumPy code: **you must call `optimizer.zero_grad()` before each backward pass**,
because PyTorch *accumulates* gradients — `backward()` **adds** the new gradient
onto whatever is already in `.grad` rather than replacing it. Drop that one line
and the failure is silent — no exception, no warning, just a run that quietly won't
learn. Running the same training loop with `zero_grad()` removed:

```
fc1 grad-norm step 1    = 2.2273
fc1 grad-norm PEAK      = 1013.0119   (grew 454.8x -- gradients ACCUMULATE across batches)
loss step 1  = 2.3656
loss PEAK    = 21085.1494   loss final = 21085.1494   (DIVERGED (spiked / NaN)) -- and NO error was thrown.
```

The `fc1` gradient norm ballooned from `2.2273` to `1013.0119` — a **454.8×** blow-up
— because each batch's gradient piled onto the last. The optimizer, reading those
ever-growing `.grad` values, took ever-larger steps, and the loss spiked from
`2.3656` to `21085.15`. Crucially, **nothing errored.** The code ran to completion
and returned a garbage model. This is the archetype of the bug that ships: a
green run, a broken result, and no stack trace to point at.

Two design points a staff reader should extract. First, why does PyTorch accumulate
by default at all, when it causes this? Because accumulation is a *feature* —
gradient accumulation across micro-batches is how you simulate a large batch size
that will not fit in memory, and how multi-loss objectives sum their gradients.
The `zero_grad()` call is the price of that flexibility, and it is paid one line at
a time. Second, this bug is structurally impossible in the NumPy path: there you
compute a *fresh* `dW` each batch and assign it, with nothing to accumulate onto.
The framework did not just automate the backward pass — it changed the *default*
around gradient state, and that changed default is exactly where the trap is.

The senior habit that catches it: when a freshly-ported loop diverges for no
obvious reason, check the loop *order and completeness* first —
`zero_grad → forward → loss → backward → step` — before you touch the learning
rate or the architecture. `zero_grad` must precede `backward` (or grads
accumulate); `backward` must precede `step` (or `step` spends stale/empty grads).
The `454.8×` grad-norm growth above is the diagnostic fingerprint of the missing
`zero_grad` specifically.

## The ceiling: automation is not capability

The honest bookend. The equivalence result is not only a correctness check — it is
a *capability* statement. Because the PyTorch model is provably the same
computation as the NumPy one (logits to `3.9e-16`, gradients to `1e-17`, identical
training trajectory), it has *exactly* the same expressiveness and *exactly* the
same failure modes. If the NumPy `784 → 128 → 10` MLP could not separate a class,
the PyTorch one cannot either. If it overfit, this one overfits the same way.
PyTorch is faster to write and runs on a GPU, but it gives the model **zero extra
ability** — a tool that only removes typing cannot, by construction, make the model
smarter. That is *why* the equivalence check belongs at the end of the week: it is
the proof that the abstraction is a productivity layer over understanding you
already own, not a substitute for it.

That division — framework as the productivity layer, hand-built understanding as
the debugging layer — is what a frontier lab actually hires for. Nobody
hand-derives backprop for a billion parameters; everybody who *debugs* a
billion-parameter run leans on knowing precisely what `loss.backward()` and
`optimizer.step()` do underneath, and on being able to recognize a `454×` grad-norm
blow-up as a missing `zero_grad()` rather than a mysterious instability. This
experiment is that knowledge made concrete: the same model, proven, plus the one
new failure mode named and measured.

## What the run establishes

On a from-scratch NumPy MLP (hand-derived backward) versus the same net as an
`nn.Module`: forward outputs matched to `3.886e-16` with `64/64` predictions
agreeing; `loss.backward()` reproduced the hand-derived gradients to `~1e-17` (loss
to `1.031e-11`); trained from the same init with the same SGD, the loss curves
stayed within `1.976e-12` over 15 epochs and reached identical test accuracy
(`1.0000` vs `1.0000`, diff `0.0`); and removing `optimizer.zero_grad()` grew the
`fc1` gradient norm `454.8×` (`2.2273 → 1013.0119`) and spiked the loss to
`21085.15` with no error thrown. PyTorch is a re-expression, not a new algorithm —
it changes the labor, not the power.
