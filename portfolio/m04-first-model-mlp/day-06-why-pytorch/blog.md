# Autograd is the chain rule replayed over a tape — not magic

Day 5 proved the PyTorch MLP is a re-expression of the from-scratch NumPy MLP:
same weights, same output, same training curve. That left one honest question
hanging — *where did the ~30 lines of hand-derived backward pass go?* The answer
is [autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html),
and the claim a staff engineer must be able to *prove* rather than recite is
narrow and sharp: **autograd is not symbolic math and not an optimizer. It is a
tape plus the chain rule.** As the forward pass runs, PyTorch records every
operation on a tracked tensor into a computation graph; `loss.backward()` walks
that graph in reverse and mechanically applies the same chain rule you derived by
hand. So the gradient it returns *equals the closed-form analytic gradient to
floating-point precision* — for arbitrary compositions, and even when the forward
took a data-dependent Python branch. That correctness is purely mechanical: it
tells you **nothing** about whether the graph is a good model. `experiment.py`
puts all of this on trial against a hand-written analytic reference (no autograd
in that path). Everything is seeded; the numbers below are the real run output.

## The canonical check: `y = x²` at `x = 3`

The whole mechanism in one line. Mark `x` tracked with `requires_grad=True`,
compute `y = x**2` (one op recorded on the tape), call `y.backward()`, and read
`x.grad`. The Day-2 analytic answer is `dy/dx = 2x = 6`:

```
autograd x.grad = 6.000000
analytic 2*x    = 6.000000
|autograd - analytic| = 0.000e+00   (exactly equal: True)
```

Not "close to 6" — **bit-for-bit `6.0`**, `exactly equal: True`. There is no
approximation, no finite-difference step, no symbolic simplification. Autograd
recorded the single `pow` node and, on the backward walk, applied its local
derivative rule (`d/dx x² = 2x`) evaluated at the stored input `3`. `x.grad` is
the attribute where that result lands — the exact `dW` you assigned by hand,
produced by one call.

## A composed graph: the chain rule replayed node-by-node

One op proves little; the interesting claim is *composition*. Take
`f(x) = sin(x²) + 3·log(x)`, whose hand-derived derivative is
`f'(x) = 2x·cos(x²) + 3/x`. Autograd sees a graph of four nodes (`square`, `sin`,
`log`, `add`) and, on the backward walk, multiplies the local derivative at each
node — that *is* the chain rule. Over five test points:

```
x               = [0.5, 1.0, 1.7, 2.3, 3.0]
autograd grad   = [6.968912, 4.080604, -1.528253, 3.816058, -4.466782]
analytic grad   = [6.968912, 4.080604, -1.528253, 3.816058, -4.466782]
max |autograd - analytic| over 5 points = 0.000e+00
```

Zero difference. A dense 300-point sweep of the same function agrees to
`9.5e-07` (float32 round-off from a different order of operations, not a
different computation) — the left panel of the figure overlays the two curves
and they are indistinguishable. This is the mechanistic core: `backward()` did
nothing clever. It replayed the *same* chain rule you would write by hand,
node by node, and produced the *same* numbers.

## Define-by-run: why plain `if` and `for` need no special syntax

Here is the property that makes PyTorch feel like ordinary Python rather than a
second, stranger language: the graph is built **dynamically, as the forward
actually runs**. Whatever operations execute — including inside an `if` branch or
a `for` loop of data-dependent length — are what get taped, and the backward walk
retraces exactly that. The experiment runs a forward pass with a real branch
(`x > 0`) and a loop that raises `x` to an integer power `k` that varies per
input:

```
x=  2.0, k=3  [x>0 branch, x**3  ]  autograd=+12.0000  analytic=+12.0000  |diff|=0.00e+00
x=  1.5, k=4  [x>0 branch, x**4  ]  autograd=+13.5000  analytic=+13.5000  |diff|=0.00e+00
x= -1.0, k=3  [x<=0 branch, -2x  ]  autograd=-2.0000  analytic=-2.0000  |diff|=0.00e+00
x=  0.7, k=5  [x>0 branch, x**5  ]  autograd=+1.2005  analytic=+1.2005  |diff|=1.14e-08
max |autograd - analytic| across branches/loops = 1.144e-08
```

Every input matched its *branch-specific* analytic gradient (`k·x^(k-1)` on the
positive branch, `-2` on the negative one) to `1.14e-08`. No `tf.cond`, no
graph-mode compilation, no shape declared in advance. This is why porting an MLP
to PyTorch feels like editing NumPy, and it is the practical payoff of
define-by-run: the framework differentiates *whatever code path ran*, so control
flow that depends on the data is free.

## Failure mode: the freed graph, and the accumulation trap underneath it

Define-by-run buys correctness cheaply, but it costs one thing that trips every
beginner. To save memory, PyTorch **frees the computation graph the instant the
first `backward()` finishes** — the tape it needs is gone. Call `backward()`
again on that same graph and it errors:

```
after 1st backward:  w.grad = 6.0000   (analytic 2*(w-1) = 6.0000)
2nd backward RAISED RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors aft...
```

Unlike most autograd bugs this one is *loud* — a `RuntimeError`, not a silent
wrong answer. The fix in a normal training loop is that you never hit it: you
recompute the forward each step, which builds a **fresh** tape every time. The
experiment confirms recomputing lets `backward()` run again — but it exposes the
genuinely dangerous default hiding underneath:

```
after recomputing forward + 2nd backward (NO zero_grad):  w.grad = 12.0000
  => it ACCUMULATED: 6.0000 + 6.0000 = 12.0000
```

`backward()` **adds** into `.grad`; it does not replace. Two backward passes with
no `zero_grad()` between them left `w.grad = 12`, not `6`. That is the root cause
of the most famous silent bug in PyTorch: forget `optimizer.zero_grad()` at the
top of the loop and each batch's gradient piles onto the last, step sizes balloon,
and the loss diverges *with no error thrown* (Day 5 measured a 454× grad-norm
blow-up from exactly this). The `12 = 6 + 6` above is that mechanism isolated to
two lines. Note the design *choice*: accumulation is a feature — it is how you
sum gradients across micro-batches to simulate a batch that will not fit in
memory, and how multi-loss objectives combine. `zero_grad()` is the one-line price
of that flexibility, and it has no NumPy ancestor, because in NumPy you compute a
fresh `dW` each batch with nothing to accumulate onto.

## The capability limit: a correct gradient is no evidence of a good model

The sharpest point of the whole module, made measurable. Give autograd a
*badly-scaled* graph — the same functional form `(scale·w − 1)²`, but with a
constant of `1e4` — and watch what it does:

```
well-scaled (scale=1):    autograd grad = -1.000000   analytic = -1.000000   |diff|=0.00e+00
badly-scaled (scale=1e4): autograd grad = +9.998000e+07   analytic = +9.998000e+07   |rel diff|=0.00e+00
the bad graph's gradient is 9.998e+07x larger -- CORRECT, but enormous.
```

The badly-scaled gradient is `~1e8×` larger — and it is **exactly correct**,
matching the analytic value to `rel diff = 0`. Autograd did not warn you, did not
clip it, did not suggest a better scale. It computed the *right* gradient of a
*bad* graph and returned it silently. This is the honest limit stated as a fact
rather than a slogan: **autograd is not symbolic math that simplifies your model,
and not an optimizer that fixes it.** A gradient being "correct" is a statement
about the calculus, not about the model. Feed autograd a poorly-conditioned,
badly-initialized, or wrong-loss architecture and it will differentiate it
faithfully and train it into a bad result without a single complaint. Every real
modeling decision — how many layers, which loss, what learning rate, why the loss
plateaued — stays with you. The right panel of the figure shows the two gradient
magnitudes on a log scale: same math, one bad constant, eight orders of magnitude
apart, both correct.

## Why a frontier lab hires for the layer below the abstraction

Nobody hand-derives backprop for a billion parameters — the framework is the
productivity layer, and this experiment is *why you can trust it*: the gradient it
produces is provably the chain rule you already know, replayed over a recorded
tape, to floating-point precision. But everybody who *debugs* a
billion-parameter run leans on the layer below. When a training run silently will
not learn and autograd throws no error, the staff move is to drop beneath the
abstraction: is the graph freed unexpectedly, are grads accumulating because a
`zero_grad()` is missing (the `6 → 12` fingerprint), or is the gradient simply
correct-but-huge because the model is badly scaled (the `1e8×` fingerprint)?
Those three failure signatures are all in this run. The framework changes the
labor, not the judgment — which is exactly why the module made you build backprop
by hand first.

## What the run establishes

Against a hand-written analytic reference: for `y = x²` at `x = 3`, autograd
returned `x.grad = 6.0`, `exactly equal: True`; on the composed graph
`sin(x²)+3·log(x)` its gradient matched the hand-derived derivative to `0.0` over
five points and `9.5e-07` over a 300-point sweep; through a Python `if`-branch and
a data-dependent `for`-loop it matched the branch-specific analytic gradient to
`1.14e-08` (define-by-run); a second `backward()` on the freed graph raised a
`RuntimeError`, and two backward passes without `zero_grad()` accumulated to
`6 + 6 = 12` (the `zero_grad` root cause); and on a badly-scaled graph autograd
returned a correct-but-`9.998e+07×`-larger gradient (`rel diff = 0`) with no
complaint. Autograd is the chain rule replayed over a tape, not magic — and a
correct gradient is no evidence of a good model.
