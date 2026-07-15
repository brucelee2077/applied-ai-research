#!/usr/bin/env python3
"""
Evidence for M5 Day 1 — "Your First Model (MLP on MNIST)".

One concrete claim, two ways to see it:

  (1) An UNTRAINED 2-layer MLP (784 -> 128 -> 10) with He-initialized weights
      scores at chance (~10%) on real MNIST test digits. Right wiring, random
      knobs. We also print the shape flow (N,784) -> (N,128) -> (N,10).

  (2) The reason the hidden layer needs a ReLU: two LINEAR layers with no bend
      collapse into a single linear map, i.e. (x @ W1) @ W2 == x @ (W1 @ W2)
      to floating-point precision. That is why an activation is not optional.

Self-contained script (not a notebook) so savefig() into assets/ is allowed.
Everything is seeded, so the printed numbers are reproducible.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend — no display in the sandbox
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)  # seed everything so the run is reproducible
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)


# ---------------------------------------------------------------------------
# Data: real MNIST from the sklearn/OpenML cache if reachable; otherwise a
# synthetic 10-class stand-in with the same shape (784 features, 10 classes)
# so the claim about an untrained network at chance still holds honestly.
# ---------------------------------------------------------------------------
def load_digits():
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml(
            "mnist_784", version=1, return_X_y=True,
            as_frame=False, parser="liac-arff",
        )
        X = X.astype(np.float32) / 255.0          # scale pixels to [0, 1]
        y = y.astype(np.int64)                     # labels come back as strings
        return X, y, "MNIST (real, 70k handwritten digits)"
    except Exception as exc:                       # offline / no cache -> fallback
        print(f"[warn] real MNIST unavailable ({type(exc).__name__}); "
              f"using a synthetic 784-dim, 10-class stand-in.")
        n = 14000
        y = RNG.integers(0, 10, size=n).astype(np.int64)
        # give each class a faint mean bump so it is a plausible dataset, not noise
        X = (RNG.standard_normal((n, 784)).astype(np.float32) * 0.1)
        for c in range(10):
            X[y == c, (c * 70):(c * 70 + 70)] += 0.3
        X = np.clip(X, 0.0, 1.0)
        return X, y, "synthetic 784-dim / 10-class fallback"


X, y, source = load_digits()

# fixed 10k test split (last 10k of MNIST is the canonical test set)
n_test = 10000 if X.shape[0] >= 20000 else X.shape[0] // 5
X_test, y_test = X[-n_test:], y[-n_test:]
print(f"data source        : {source}")
print(f"full dataset shape : X={X.shape}  y={y.shape}")
print(f"test split shape   : X_test={X_test.shape}  y_test={y_test.shape}")


# ---------------------------------------------------------------------------
# He initialization: weights ~ N(0,1) * sqrt(2 / n_in); biases start at zero.
# This is the variance-preserving scheme built for ReLU (lesson concept c8).
# ---------------------------------------------------------------------------
def he(n_in, n_out):
    return (RNG.standard_normal((n_in, n_out)).astype(np.float32)
            * np.sqrt(2.0 / n_in))


W1, b1 = he(784, 128), np.zeros(128, dtype=np.float32)
W2, b2 = he(128, 10),  np.zeros(10,  dtype=np.float32)
n_params = W1.size + b1.size + W2.size + b2.size
print(f"\nparameter count    : {n_params:,} "
      f"(W1 {W1.size:,} + b1 {b1.size} + W2 {W2.size:,} + b2 {b2.size})")


def relu(z):
    return np.maximum(0.0, z)


# ---------------------------------------------------------------------------
# Forward pass — print the shape at every step, exactly as the lesson promises:
# (N, 784) -> (N, 128) -> (N, 10).
# ---------------------------------------------------------------------------
def forward(x, verbose=False):
    if verbose:
        print(f"  input  x : {x.shape}")
    h = relu(x @ W1 + b1)                          # hidden layer + bend
    if verbose:
        print(f"  hidden h : {h.shape}   (after ReLU(x@W1+b1))")
    logits = h @ W2 + b2                            # output layer -> 10 logits
    if verbose:
        print(f"  logits   : {logits.shape}  (10 class scores per image)")
    return logits


print("\nforward-pass shape flow (one batch):")
logits = forward(X_test, verbose=True)

# argmax over the 10 logits = the guessed digit
preds = logits.argmax(axis=1)
acc = float((preds == y_test).mean())
print(f"\nuntrained accuracy : {acc*100:.2f}%  (chance over 10 classes = 10.00%)")

# fraction of ReLU units that are dead (zero for every test image) — the
# activation statistic a senior engineer logs to catch a silent failure.
hidden = relu(X_test @ W1 + b1)
dead_frac = float((hidden.max(axis=0) == 0).mean())
print(f"dead ReLU fraction : {dead_frac*100:.2f}% of hidden units never fire")


# ---------------------------------------------------------------------------
# Claim (2): linear collapse. Two linear layers with no ReLU == one matmul.
# ---------------------------------------------------------------------------
La = RNG.standard_normal((784, 128)).astype(np.float64)
Lb = RNG.standard_normal((128, 10)).astype(np.float64)
x_probe = X_test[:64].astype(np.float64)

two_layers = (x_probe @ La) @ Lb                   # stack, no bend
one_layer = x_probe @ (La @ Lb)                    # fused single matrix
max_abs_diff = float(np.abs(two_layers - one_layer).max())
print(f"\nlinear collapse    : max|(x@La)@Lb - x@(La@Lb)| = {max_abs_diff:.3e} "
      f"(≈0 -> two linear layers ARE one)")

# contrast: with a ReLU between them the two are NOT equal
with_bend = relu(x_probe @ La) @ Lb
bend_diff = float(np.abs(with_bend - one_layer).max())
print(f"with a ReLU bend   : max|ReLU(x@La)@Lb - x@(La@Lb)| = {bend_diff:.3e} "
      f"(large -> the bend earns its keep)")


# ---------------------------------------------------------------------------
# Plot: distribution of the untrained model's guesses across the 10 digit slots.
# A trained model would concentrate on the true label; an untrained one is
# roughly flat with a bias toward whichever random slot happens to run hot.
# ---------------------------------------------------------------------------
counts = np.bincount(preds, minlength=10)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(range(10), counts / counts.sum(), color="#7C6DAA")
ax.axhline(0.10, color="#C93B3B", ls="--", lw=1.5, label="uniform chance (10%)")
ax.set_xticks(range(10))
ax.set_xlabel("guessed digit (argmax of logits)")
ax.set_ylabel("fraction of test images")
ax.set_title(f"Untrained MLP guesses — {acc*100:.1f}% accuracy (≈ chance)")
ax.legend()
fig.tight_layout()
plot_path = os.path.join(ASSETS, "untrained_guess_distribution.png")
fig.savefig(plot_path, dpi=110)
print(f"\nsaved plot         : assets/{os.path.basename(plot_path)}")

print("\nKEY RESULT: correct wiring (784->128->10) + random He weights = chance-"
      f"level accuracy ({acc*100:.2f}%); linear layers with no bend collapse "
      f"into one (diff {max_abs_diff:.1e}).")
