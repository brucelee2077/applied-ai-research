#!/usr/bin/env python3
"""
Evidence for M5 Day 2 — "The Backward Pass, Wired by Hand".

ONE concrete claim, made falsifiable:

  A hand-wired analytic backprop gradient for a 784 -> 128 -> 10 MLP agrees
  with a slow finite-difference (numerical) gradient to many decimal places on
  every sampled weight. Drop a single term of the chain rule -- the ReLU gate
  in the dz1 step -- and the SAME check blows up by ~8 orders of magnitude.
  Nothing crashes; the only thing that catches the bug is the check itself.

We back it up three ways:

  (1) Gradient shapes: every gradient has the SAME shape as its weight
      (dW1 == W1.shape, dW2 == W2.shape). A mismatch is a transpose bug.
  (2) Correct backprop: max|analytic - numerical| over sampled W2/W1 entries
      is ~1e-10 (well below the finite-difference floor), i.e. AGREE.
  (3) Broken backprop (dz1 = dh, no ReLU gate): the SAME max-abs difference
      jumps to O(1e-2) -- a silent, uncrashing gradient bug the check flags.

Also demonstrates the update rule lowers the loss on this batch, and plots the
per-sampled-weight analytic-vs-numerical agreement (correct vs broken).

Self-contained script (NOT a notebook), so savefig() into assets/ is allowed.
Everything is seeded, so the printed numbers are reproducible.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend -- no display in the sandbox
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)  # seed everything so the run is reproducible
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)

# ---------------------------------------------------------------------------
# Model dims: the 784 -> 128 -> 10 MLP from Day 1, now with a backward pass.
# We work in float64 so the finite-difference check is not swamped by float32
# round-off (the check needs the *analytic* code to be the suspect, not noise).
# ---------------------------------------------------------------------------
N_IN, N_HID, N_OUT = 784, 128, 10
BATCH = 32

# He-initialized weights (same scheme as Day 1) keep ReLU units alive.
W1 = RNG.standard_normal((N_IN, N_HID)) * np.sqrt(2.0 / N_IN)
b1 = np.zeros(N_HID)
W2 = RNG.standard_normal((N_HID, N_OUT)) * np.sqrt(2.0 / N_HID)
b2 = np.zeros(N_OUT)

# A small batch of fake "images" (pixels in [0,1]) and their true labels.
x = RNG.random((BATCH, N_IN))
y_idx = RNG.integers(0, N_OUT, size=BATCH)           # integer class labels
y_true = np.zeros((BATCH, N_OUT))                    # one-hot targets
y_true[np.arange(BATCH), y_idx] = 1.0


# ---------------------------------------------------------------------------
# Forward pass. Returns the loss AND caches z1, h, probs -- backprop reuses
# these instead of recomputing a slope per weight (the whole point of backprop).
# ---------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)             # subtract max for stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(W1, b1, W2, b2, x, y_true):
    z1 = x @ W1 + b1                                 # (BATCH, N_HID) pre-activation
    h = np.maximum(0.0, z1)                          # ReLU bend -> hidden activations
    logits = h @ W2 + b2                             # (BATCH, N_OUT) class scores
    probs = softmax(logits)                          # (BATCH, N_OUT) probabilities
    # mean cross-entropy loss: one scalar that says "how wrong" over the batch
    loss = -np.mean(np.sum(y_true * np.log(probs + 1e-12), axis=1))
    cache = (z1, h, probs)
    return loss, cache


# ---------------------------------------------------------------------------
# Backward pass. gate=True is the correct chain rule; gate=False DROPS the ReLU
# gate in the dz1 step (dz1 = dh instead of dz1 = dh * (z1 > 0)) -- a silent
# gradient bug: blame leaks back through hidden units that were OFF.
# ---------------------------------------------------------------------------
def backward(W2, x, cache, y_true, gate=True):
    z1, h, probs = cache                             # reuse cached forward values
    B = x.shape[0]
    # d(loss)/d(logits) for softmax + cross-entropy = (probs - y_true) / B
    dlogits = (probs - y_true) / B                   # (BATCH, N_OUT)
    dW2 = h.T @ dlogits                              # (N_HID, N_OUT) -> matches W2
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ W2.T                              # (BATCH, N_HID) hand blame back
    if gate:
        dz1 = dh * (z1 > 0)                          # CORRECT: ReLU gate, blame only where unit was on
    else:
        dz1 = dh                                     # BUG: gate dropped -> blame leaks through off units
    dW1 = x.T @ dz1                                  # (N_IN, N_HID) -> matches W1
    db1 = dz1.sum(axis=0)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


# ---------------------------------------------------------------------------
# (1) Forward + backward once; confirm every gradient shape matches its weight.
# ---------------------------------------------------------------------------
loss0, cache = forward(W1, b1, W2, b2, x, y_true)
grads = backward(W2, x, cache, y_true, gate=True)

print(f"batch / dims       : batch={BATCH}, MLP {N_IN}->{N_HID}->{N_OUT} (float64)")
print(f"initial loss       : {loss0:.6f}  (uniform-guess baseline = ln 10 = {np.log(10):.6f})")
print()
print("gradient shapes match their weights (mismatch = transpose bug):")
for name, W in [("W1", W1), ("W2", W2)]:
    g = grads["d" + name]
    ok = g.shape == W.shape
    print(f"  d{name} {str(g.shape):<12} vs {name} {str(W.shape):<12} -> {'MATCH' if ok else 'MISMATCH'}")
print()


# ---------------------------------------------------------------------------
# (2)+(3) Gradient check: central finite differences on sampled weight entries.
# For a chosen (i,j) of a weight matrix: nudge by +/- eps, re-run the forward
# pass, and estimate the slope as (loss(w+eps) - loss(w-eps)) / (2 eps).
# Compare against the analytic gradient. Do this for CORRECT and BROKEN backprop.
# ---------------------------------------------------------------------------
EPS = 1e-6
N_SAMPLE = 12                                        # sampled entries per matrix


def numerical_grad_entry(param, i, j):
    """Central-difference d(loss)/d(param[i,j]) via two extra forward passes."""
    orig = param[i, j]
    param[i, j] = orig + EPS
    lp, _ = forward(W1, b1, W2, b2, x, y_true)
    param[i, j] = orig - EPS
    lm, _ = forward(W1, b1, W2, b2, x, y_true)
    param[i, j] = orig                               # restore -- leave no trace
    return (lp - lm) / (2 * EPS)


def check(param, analytic, label):
    """Max |analytic - numerical| over N_SAMPLE random entries of `param`."""
    ii = RNG.integers(0, param.shape[0], size=N_SAMPLE)
    jj = RNG.integers(0, param.shape[1], size=N_SAMPLE)
    diffs = []
    for i, j in zip(ii, jj):
        num = numerical_grad_entry(param, i, j)
        diffs.append(abs(analytic[i, j] - num))
    diffs = np.array(diffs)
    print(f"  {label:<34} max|analytic - numerical| = {diffs.max():.3e}")
    return diffs


print("CORRECT backprop -- analytic gradient vs numerical (finite-difference):")
good_grads = backward(W2, x, cache, y_true, gate=True)
good_W2 = check(W2, good_grads["dW2"], "W2 (dW2 = h.T @ dlogits)")
good_W1 = check(W1, good_grads["dW1"], "W1 (through ReLU gate)")
print()

print("BROKEN backprop -- SAME check after DROPPING the ReLU gate (dz1 = dh):")
bad_grads = backward(W2, x, cache, y_true, gate=False)
# dW2 does not depend on the gate, so W2 stays correct; the bug lives in dW1.
bad_W1 = check(W1, bad_grads["dW1"], "W1 (gate dropped -> BUG)")
print()

good_max = max(good_W2.max(), good_W1.max())
bad_max = bad_W1.max()
ratio = bad_max / good_max
print(f"gradient-check verdict:")
print(f"  correct dW1/dW2 agree to  ~{-int(np.floor(np.log10(good_max)))} decimal places "
      f"(max diff {good_max:.3e})")
print(f"  broken  dW1 disagrees by   {bad_max:.3e}  "
      f"({ratio:.1e}x worse -- silent bug the check caught)")
print()


# ---------------------------------------------------------------------------
# Update rule: one gradient-descent step against the (correct) gradient.
# param = param - learning_rate * gradient. A RIGHT-SIZED lr lowers the loss;
# a TOO-LARGE lr overshoots the valley and RAISES it -- exactly the c5 failure.
# We show both so the sign of delta is honest, not asserted.
# ---------------------------------------------------------------------------
def step_loss(lr):
    W1n = W1 - lr * good_grads["dW1"]
    b1n = b1 - lr * good_grads["db1"]
    W2n = W2 - lr * good_grads["dW2"]
    b2n = b2 - lr * good_grads["db2"]
    return forward(W1n, b1n, W2n, b2n, x, y_true)[0]


LR_GOOD, LR_BIG = 0.05, 0.5
loss_good = step_loss(LR_GOOD)
loss_big = step_loss(LR_BIG)
print("update rule (param = param - lr * grad):")
print(f"  right-sized lr={LR_GOOD:<5} loss {loss0:.6f} -> {loss_good:.6f}  "
      f"(delta {loss_good - loss0:+.6f}; downhill step lowers the loss)")
print(f"  too-large   lr={LR_BIG:<5} loss {loss0:.6f} -> {loss_big:.6f}  "
      f"(delta {loss_big - loss0:+.6f}; OVERSHOOTS -> loss climbs)")
print()


# ---------------------------------------------------------------------------
# Plot: per-sampled-entry analytic-vs-numerical agreement, correct vs broken.
# Log-scale so the ~6-decimal agreement and the blown-up bug both read clearly.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.0))
xs = np.arange(1, N_SAMPLE + 1)
ax.semilogy(xs, np.clip(good_W1, 1e-16, None), "o-", color="#2D8B55",
            label="correct backprop (dW1, ReLU gate)")
ax.semilogy(xs, np.clip(good_W2, 1e-16, None), "s-", color="#7C6DAA",
            label="correct backprop (dW2)")
ax.semilogy(xs, np.clip(bad_W1, 1e-16, None), "x--", color="#C93B3B",
            label="broken backprop (dW1, gate dropped)")
ax.axhline(EPS, color="#8A6D3B", ls=":", lw=1,
           label=f"finite-diff floor (eps={EPS:g})")
ax.set_xlabel("sampled weight entry")
ax.set_ylabel("|analytic - numerical|  (log scale)")
ax.set_title("Gradient check: hand-wired backprop vs finite differences")
ax.legend(fontsize=8, loc="center right")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
plot_path = os.path.join(ASSETS, "gradient_check.png")
fig.savefig(plot_path, dpi=110)
print(f"saved plot         : assets/{os.path.basename(plot_path)}")
print()

print("KEY RESULT: hand-wired analytic backprop agrees with a finite-difference "
      f"gradient to ~{-int(np.floor(np.log10(good_max)))} decimals "
      f"(max diff {good_max:.3e}); dropping the ReLU gate breaks the SAME check "
      f"({bad_max:.3e}, {ratio:.1e}x worse) with no crash.")
