# Chance-level by construction: what an untrained MLP actually does

A 2-layer MLP for MNIST is 784 → 128 → 10: flatten the image to a 784-vector,
one hidden layer of 128 ReLU units, a 10-way output layer of logits, and
`argmax` for the guess. The whole model is one line —
`logits = ReLU(x @ W1 + b1) @ W2 + b2`. This note produces the evidence for two
claims that get hand-waved on day one: (1) a correctly-wired but untrained
network sits *at chance*, and (2) the ReLU between the two layers is load-bearing,
not decorative. Every number below comes from `experiment.py` on the real 70k
MNIST set, seeded for reproducibility.

## The mechanism, and why the number is 10% not "small"

The forward pass runs the exact shape flow the architecture promises:

```
input  x : (10000, 784)
hidden h : (10000, 128)   (after ReLU(x@W1+b1))
logits   : (10000, 10)    10 class scores per image
```

The model has **101,770 parameters** (`W1` 100,352 + `b1` 128 + `W2` 1,280 +
`b2` 10) — the "~100k knobs" intuition made exact. With He-initialized weights
(`W ~ N(0,1) * sqrt(2/n_in)`, biases zero) and no training, the test accuracy is:

```
untrained accuracy : 9.87%   (chance over 10 classes = 10.00%)
```

The precise value matters. 9.87% is not "a bit better than nothing" — it is
statistically indistinguishable from `1/10`. With random weights, the argmax over
10 logits is (to first order) uniform over the classes independent of the input,
so the expected accuracy is exactly the prior of the most-frequently-guessed
class, which for a balanced 10-way task is ~10%. The 0.13-point shortfall is
sampling noise on 10,000 examples (a binomial standard error of
`sqrt(0.1·0.9/10000) ≈ 0.3%`), plus the fact that the untrained argmax
concentrates on whichever slot's random `W2` column happens to run hot rather
than spreading perfectly uniformly — visible as the uneven bars in the plot. The
takeaway for a reviewer: if a from-scratch classifier reports ~10% before
training, the *wiring is correct and nothing has learned yet*. Reporting 0% or
100% at init is the bug signal, not 10%.

## Failure mode: linear collapse (the ReLU is not optional)

The most common silent defect in a hand-rolled MLP is dropping the activation:
`logits = (x @ W1) @ W2 + b2`. It runs, it trains, it looks like a "deep"
network — and it can never beat a single linear layer, because two linear maps
compose into one: `(x @ W1) @ W2 = x @ (W1 @ W2)`, and `W1 @ W2` is just another
matrix. The experiment measures this directly on 64 test images with random
784×128 and 128×10 matrices in float64:

```
linear collapse : max|(x@La)@Lb - x@(La@Lb)| = 6.253e-13   (two linear layers ARE one)
with a ReLU bend : max|ReLU(x@La)@Lb - x@(La@Lb)| = 1.920e+02   (the bend earns its keep)
```

`6.253e-13` is floating-point round-off on the associativity of matmul — the two
expressions are *the same function*. Insert `ReLU` between the matmuls and the
gap jumps to `1.920e+02`: the bend clips negatives before the second matmul,
producing values no single matrix can. That `1.9e2`-vs-`6e-13` contrast is the
whole argument for depth. A network stacked without nonlinearity has the
representational capacity of one layer regardless of how many layers you draw,
which is why it cannot separate non-linearly-separable data (the XOR failure the
lesson opens with). The failure is invisible in a stack trace and shows up only
as a suspiciously low ceiling on training accuracy — you catch it by *knowing the
architecture collapses*, not by reading an error.

## Design trade-off: hidden width vs. capacity and dead units

Two dials interact at initialization. **Width** is the obvious one: 128 hidden
units buys 100,352 of the model's 101,770 parameters. Widen to 256 or 1024 and
you gain capacity to represent finer stroke features (mitigating underfitting)
at a linear cost in the first weight matrix and the matmul. The subtler dial is
the **init scheme's variance**. He init exists precisely to keep ReLU units
alive: scaling by `sqrt(2/n_in)` keeps the pre-activation variance ~1 layer to
layer, so units are unlikely to be born permanently negative. The activation
statistic that proves it here:

```
dead ReLU fraction : 0.00% of hidden units never fire
```

Zero of the 128 hidden units are dead across all 10,000 test images. That is the
diagnostic a senior engineer logs — not final accuracy, but the fraction of ReLU
units stuck at zero. Had we initialized with a large fixed variance instead of
`2/n_in`, or later cranked the learning rate, a chunk of units would output 0 for
every input and stop learning entirely (the "dead ReLU" failure), and the model
would quietly underperform its parameter count with no crash to point at. The
trade-off in one sentence: He init trades a free lunch (any random weights work
to *run* the model) for the discipline of variance-matched weights that keep the
network's capacity actually usable once gradients start flowing.

## What this evidence does and does not show

It shows the architecture is wired correctly (shape flow, parameter count),
that the starting point is chance by construction (9.87%), that the ReLU is
mathematically necessary (`6e-13` vs `1.9e2`), and that He init keeps units live
(0.00% dead). It does *not* show learning — every knob is still random. The next
lesson wires the backward pass, which is where those 101,770 parameters start
moving off their random values and 9.87% starts climbing.
