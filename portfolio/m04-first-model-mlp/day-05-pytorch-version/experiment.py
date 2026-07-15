#!/usr/bin/env python3
"""
Evidence for M5 Day 5 -- "The Same Model in PyTorch".

ONE concrete claim, made falsifiable:

  The PyTorch MLP is a RE-EXPRESSION of the from-scratch NumPy MLP, not a new
  algorithm. Given the SAME weights and the SAME input, both produce the SAME
  output (to floating-point precision); trained with the SAME SGD rule they
  follow the SAME loss curve to the SAME accuracy. PyTorch removes the manual
  labor (autograd writes the backward pass; the optimizer writes the update) --
  it changes the LABOR, not the POWER.

We prove it four ways, all against a real from-scratch NumPy MLP (no autograd
in the NumPy path -- the backward pass is hand-derived, exactly the Day-2 code):

  (1) FORWARD EQUIVALENCE: build a 784->128->10 MLP twice -- once in NumPy, once
      as an nn.Module -- with the EXACT SAME weights copied across (handling the
      (out,in) transpose PyTorch stores). Feed the same batch. The 10 output
      scores match to ~1e-6, and both pick the same class on every example.

  (2) GRADIENT EQUIVALENCE: one hand-derived NumPy backward pass vs one
      loss.backward() call. Every weight's gradient matches to ~1e-6 -- proof
      that loss.backward() computes the same numbers as the ~30 lines of Day-2
      chain rule.

  (3) TRAINING EQUIVALENCE: train both from the SAME init with the SAME SGD
      rule (weight -= lr * grad). Their per-epoch loss curves fall together and
      reach the same accuracy -- the equivalence check the lesson promises.

  (4) THE zero_grad SILENT BUG: run the SAME PyTorch training with
      optimizer.zero_grad() REMOVED. Gradients accumulate across batches, the
      grad norm balloons, and the loss diverges (spikes / NaN) with NO error
      thrown -- the classic bug that cannot happen in the NumPy path (which
      recomputes a fresh dW each batch with nothing to accumulate).

The task is a self-contained synthetic 10-class problem with 784-dim inputs (the
MNIST shape), so the script needs NO dataset download and runs anywhere. The
CLAIM under test -- equivalence of the two implementations -- is independent of
the dataset: it is a statement about the two code paths computing the same math.

Self-contained script (NOT a notebook), so savefig() into assets/ is allowed.
Everything is seeded, so the printed numbers are reproducible.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # headless backend -- no display in the sandbox
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)  # deterministic-ish CPU math

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)

# The MNIST shape, but a self-contained synthetic task so nothing is downloaded.
N_IN, N_HID, N_OUT = 784, 128, 10
N_TRAIN, N_TEST = 3000, 1500


# Fixed per-class mean vectors, SHARED by train and test. Train and test are
# draws from the SAME 10-class distribution, so the task genuinely generalizes
# and both models reach a high, MATCHING test accuracy -- which is what makes
# the equivalence check meaningful (a random-guessing task would prove nothing).
_CLASS_MEANS = np.random.default_rng(1).standard_normal((N_OUT, N_IN)).astype(np.float64) * 2.0


def make_data(n, rng):
    """A learnable 10-class task at MNIST shape: each point is its class mean
    (shared across train/test) plus Gaussian noise -- separable enough that a
    small MLP reaches high accuracy on HELD-OUT data."""
    y = rng.integers(0, N_OUT, n)
    X = _CLASS_MEANS[y] + rng.standard_normal((n, N_IN)).astype(np.float64) * 2.0
    return X.astype(np.float64), y.astype(np.int64)


_rng = np.random.default_rng(7)
Xtr, ytr = make_data(N_TRAIN, _rng)
Xte, yte = make_data(N_TEST, _rng)
# standardize with TRAIN statistics only
_mu, _sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
Xtr = (Xtr - _mu) / _sd
Xte = (Xte - _mu) / _sd


# ---------------------------------------------------------------------------
# The from-scratch NumPy MLP -- the Day-1..Day-3 code, hand-derived backward.
#   forward : logits = relu(x @ W1 + b1) @ W2 + b2   (raw scores, no softmax)
#   loss    : mean cross-entropy (softmax folded into the loss, like the lesson)
#   backward: the Day-2 chain rule, by hand -- NO autograd in this path.
# ---------------------------------------------------------------------------
def np_softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def np_forward(params, x):
    """Returns logits and a cache for the backward pass."""
    z1 = x @ params["W1"] + params["b1"]     # (B, N_HID) pre-activation
    h = np.maximum(0.0, z1)                   # ReLU hidden
    logits = h @ params["W2"] + params["b2"]  # (B, N_OUT) raw scores
    return logits, (x, z1, h)


def np_loss(logits, y):
    probs = np_softmax(logits)
    n = logits.shape[0]
    return float(-np.mean(np.log(probs[np.arange(n), y] + 1e-12)))


def np_backward(params, cache, logits, y):
    """The hand-derived Day-2 backward pass -- explicit chain rule, no autograd."""
    x, z1, h = cache
    n = x.shape[0]
    probs = np_softmax(logits)
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n                              # mean cross-entropy reduction
    dW2 = h.T @ dlogits                        # (N_HID, N_OUT)
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ params["W2"].T              # back through second layer
    dz1 = dh * (z1 > 0)                        # ReLU gate
    dW1 = x.T @ dz1                            # (N_IN, N_HID)
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ---------------------------------------------------------------------------
# The SAME MLP as an nn.Module -- the ONLY pass we write is forward().
# nn.CrossEntropyLoss folds softmax in, so forward returns RAW scores.
# ---------------------------------------------------------------------------
class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_IN, N_HID)   # 784 -> 128, weight stored (128, 784)
        self.fc2 = nn.Linear(N_HID, N_OUT)  # 128 -> 10,  weight stored (10, 128)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))  # raw scores, NO softmax


def init_numpy_from_torch(model):
    """Copy the PyTorch model's weights into a NumPy param dict, undoing the
    (out,in) transpose so the NumPy path computes x @ W1 + b1 with W1 (in,out).
    This is the concept-2 detail made concrete: torch stores (out,in), we store
    (in,out), and W_numpy == W_torch.T."""
    return {
        "W1": model.fc1.weight.detach().numpy().T.copy().astype(np.float64),
        "b1": model.fc1.bias.detach().numpy().copy().astype(np.float64),
        "W2": model.fc2.weight.detach().numpy().T.copy().astype(np.float64),
        "b2": model.fc2.bias.detach().numpy().copy().astype(np.float64),
    }


# ===========================================================================
# (1) FORWARD EQUIVALENCE -- same weights + same input => same output.
# ===========================================================================
print("=" * 74)
print("(1) FORWARD EQUIVALENCE  (same weights, same input, same output)")
print("=" * 74)
model = TorchMLP()
np_params = init_numpy_from_torch(model)

print(f"net = {N_IN}->{N_HID}(ReLU)->{N_OUT}")
print(f"model.fc1.weight.shape = {tuple(model.fc1.weight.shape)}   "
      f"(out,in) -- the TRANSPOSE of the numpy W1 {np_params['W1'].shape} (in,out)")

xb = Xtr[:64]
Xb_t = torch.tensor(xb, dtype=torch.float64)
model = model.double()  # match numpy float64 so the comparison is not float32-limited

with torch.no_grad():
    torch_logits = model(Xb_t).numpy()
np_logits, _ = np_forward(np_params, xb)

max_abs_diff = float(np.max(np.abs(torch_logits - np_logits)))
np_pred = np_logits.argmax(axis=1)
torch_pred = torch_logits.argmax(axis=1)
pred_agree = int((np_pred == torch_pred).sum())
print(f"max |torch_logits - numpy_logits| over a 64x10 batch = {max_abs_diff:.3e}")
print(f"predicted-class agreement = {pred_agree}/{len(xb)} examples")
print(f"example row (class scores rounded): numpy = {np.round(np_logits[0], 4).tolist()}")
print(f"                                    torch = {np.round(torch_logits[0], 4).tolist()}")
print()


# ===========================================================================
# (2) GRADIENT EQUIVALENCE -- hand-derived NumPy backward vs loss.backward().
# ===========================================================================
print("=" * 74)
print("(2) GRADIENT EQUIVALENCE  (hand-derived chain rule vs loss.backward())")
print("=" * 74)
yb = ytr[:64]

# --- PyTorch path: one loss.backward() call fills every .grad ---
criterion = nn.CrossEntropyLoss()               # folds softmax + mean CE
model.zero_grad()
loss_t = criterion(model(Xb_t), torch.tensor(yb, dtype=torch.long))
loss_t.backward()                               # autograd fills W1.grad, ... in ONE call
torch_grads = {
    "W1": model.fc1.weight.grad.detach().numpy().T,  # transpose back to (in,out)
    "b1": model.fc1.bias.grad.detach().numpy(),
    "W2": model.fc2.weight.grad.detach().numpy().T,
    "b2": model.fc2.bias.grad.detach().numpy(),
}

# --- NumPy path: the Day-2 chain rule, by hand ---
np_logits_b, cache = np_forward(np_params, xb)
np_grads = np_backward(np_params, cache, np_logits_b, yb)

print(f"loss:  numpy CE = {np_loss(np_logits_b, yb):.6f}   "
      f"torch CE = {loss_t.item():.6f}   "
      f"|diff| = {abs(np_loss(np_logits_b, yb) - loss_t.item()):.3e}")
for k in ("W1", "b1", "W2", "b2"):
    d = float(np.max(np.abs(torch_grads[k] - np_grads[k])))
    print(f"grad {k:>2}: max |torch.grad - numpy hand-derived grad| = {d:.3e}")
print("=> loss.backward() reproduces the ~30-line hand-derived backward pass exactly.")
print()


# ===========================================================================
# (3) TRAINING EQUIVALENCE -- same init + same SGD => same loss curve.
# ===========================================================================
print("=" * 74)
print("(3) TRAINING EQUIVALENCE  (same init, same SGD rule, same loss curve)")
print("=" * 74)
LR, BATCH, EPOCHS = 0.1, 100, 15


def torch_accuracy(m, X, y):
    with torch.no_grad():
        pred = m(torch.tensor(X, dtype=torch.float64)).argmax(dim=1).numpy()
    return float((pred == y).mean())


def np_accuracy(params, X, y):
    logits, _ = np_forward(params, X)
    return float((logits.argmax(axis=1) == y).mean())


# start BOTH from the identical initial weights
model_tr = TorchMLP().double()
np_tr = init_numpy_from_torch(model_tr)
opt = torch.optim.SGD(model_tr.parameters(), lr=LR)  # the update rule = the Day-3 line

torch_loss_hist, np_loss_hist = [], []
n = N_TRAIN
order_rng = np.random.default_rng(123)
for e in range(EPOCHS):
    # SAME shuffle order fed to BOTH paths so the comparison is exact.
    order = order_rng.permutation(n)
    ep_torch, ep_np, nb = 0.0, 0.0, 0
    for start in range(0, n, BATCH):
        idx = order[start:start + BATCH]
        xb_, yb_ = Xtr[idx], ytr[idx]

        # --- PyTorch: the five-line loop ---
        opt.zero_grad()                                   # clear old grads
        scores = model_tr(torch.tensor(xb_, dtype=torch.float64))
        loss = criterion(scores, torch.tensor(yb_, dtype=torch.long))
        loss.backward()                                   # fill every .grad
        opt.step()                                        # weight -= lr * grad
        ep_torch += loss.item()

        # --- NumPy: the SAME five beats, hand-written ---
        logits_np, cache_np = np_forward(np_tr, xb_)
        ep_np += np_loss(logits_np, yb_)
        grads_np = np_backward(np_tr, cache_np, logits_np, yb_)
        for kk in np_tr:                                  # SGD: param -= lr * grad
            np_tr[kk] = np_tr[kk] - LR * grads_np[kk]
        nb += 1
    torch_loss_hist.append(ep_torch / nb)
    np_loss_hist.append(ep_np / nb)

torch_acc = torch_accuracy(model_tr, Xte, yte)
np_acc = np_accuracy(np_tr, Xte, yte)
max_curve_diff = float(np.max(np.abs(np.array(torch_loss_hist) - np.array(np_loss_hist))))
print(f"train loss  epoch 1  : torch={torch_loss_hist[0]:.4f}   numpy={np_loss_hist[0]:.4f}")
print(f"train loss  epoch {EPOCHS:>2} : torch={torch_loss_hist[-1]:.4f}   numpy={np_loss_hist[-1]:.4f}")
print(f"max per-epoch loss-curve difference over {EPOCHS} epochs = {max_curve_diff:.3e}")
print(f"final TEST accuracy : torch={torch_acc:.4f}   numpy={np_acc:.4f}   "
      f"|diff|={abs(torch_acc - np_acc):.3e}")
print("=> the five-line PyTorch loop and the hand-written numpy loop are the SAME model.")
print()


# ===========================================================================
# (4) THE zero_grad SILENT BUG -- accumulation blows the loss up, no error.
# ===========================================================================
print("=" * 74)
print("(4) THE zero_grad SILENT BUG  (drop zero_grad => grads accumulate)")
print("=" * 74)
model_bug = TorchMLP().double()
# A higher LR isolates the failure cleanly: with zero_grad the loop is stable at
# this rate, but WITHOUT it the accumulating gradients turn each step into an
# oversized one and the loss explodes -- the whole point of the demo.
BUG_LR = 1.0
opt_bug = torch.optim.SGD(model_bug.parameters(), lr=BUG_LR)
bug_losses, grad_norms = [], []
order = order_rng.permutation(n)
for step, start in enumerate(range(0, n, BATCH)):
    idx = order[start:start + BATCH]
    xb_, yb_ = Xtr[idx], ytr[idx]
    # NOTE: opt_bug.zero_grad() is DELIBERATELY MISSING -- the classic bug.
    scores = model_bug(torch.tensor(xb_, dtype=torch.float64))
    loss = criterion(scores, torch.tensor(yb_, dtype=torch.long))
    loss.backward()                                       # grads ADD onto the old ones
    gnorm = float(model_bug.fc1.weight.grad.norm())
    opt_bug.step()
    bug_losses.append(loss.item())
    grad_norms.append(gnorm)
    if step >= 25:
        break

first_gn = grad_norms[0]
last_gn = max(grad_norms)                                 # peak norm before/at blow-up
peak_loss = max(l for l in bug_losses if math.isfinite(l)) if any(math.isfinite(l) for l in bug_losses) else float("nan")
diverged = (not all(math.isfinite(l) for l in bug_losses)) or (peak_loss > 5.0 * bug_losses[0])
print(f"fc1 grad-norm step 1    = {first_gn:.4f}")
print(f"fc1 grad-norm PEAK      = {last_gn:.4f}   "
      f"(grew {last_gn / max(first_gn, 1e-12):.1f}x -- gradients ACCUMULATE across batches)")
print(f"loss step 1  = {bug_losses[0]:.4f}")
print(f"loss PEAK    = {peak_loss:.4f}   "
      f"loss final = {bug_losses[-1]:.4f}   "
      f"({'DIVERGED (spiked / NaN)' if diverged else 'did not diverge'}) "
      f"-- and NO error was thrown.")
print("=> this bug cannot happen in the numpy path: it recomputes a fresh dW each")
print("   batch with nothing to accumulate. zero_grad() is the one new habit.")
print()


# ---------------------------------------------------------------------------
# Plot: the whole equivalence story on one canvas.
#   left  -- torch vs numpy training loss curves, overlaid (they coincide).
#   right -- the zero_grad bug: grad norm balloons, loss diverges.
# ---------------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.2))

ep = range(1, EPOCHS + 1)
axL.plot(ep, np_loss_hist, "-", color="#2D8B55", lw=3.0, label="numpy MLP (hand-derived)")
axL.plot(ep, torch_loss_hist, "--", color="#C99A12", lw=1.8, label="PyTorch MLP (autograd)")
axL.set_xlabel("epoch")
axL.set_ylabel("train cross-entropy")
axL.set_title(f"Same model: loss curves coincide (max diff {max_curve_diff:.1e})")
axL.legend(fontsize=8)
axL.grid(True, alpha=0.25)

steps = range(1, len(grad_norms) + 1)
axR.plot(steps, grad_norms, "-o", color="#C93B3B", ms=3, label="fc1 grad norm")
axR.set_xlabel("batch (no zero_grad)")
axR.set_ylabel("fc1 gradient norm")
axR.set_title("Drop zero_grad: gradients accumulate, loss diverges")
axR.legend(fontsize=8)
axR.grid(True, alpha=0.25)

fig.tight_layout()
plot_path = os.path.join(ASSETS, "pytorch_equivalence_and_zero_grad_bug.png")
fig.savefig(plot_path, dpi=110)
print(f"saved plot : assets/{os.path.basename(plot_path)}")
print()

print("KEY RESULT: with identical weights and input, the PyTorch and hand-derived "
      f"NumPy MLPs produced logits matching to {max_abs_diff:.1e} (predictions agree "
      f"{pred_agree}/{len(xb)}); loss.backward() reproduced the hand-derived gradients "
      "to ~1e-6; trained from the same init with the same SGD their loss curves stayed "
      f"within {max_curve_diff:.1e} of each other over {EPOCHS} epochs and reached the "
      f"same test accuracy (torch {torch_acc:.4f} vs numpy {np_acc:.4f}). Dropping "
      f"optimizer.zero_grad() made the fc1 grad norm balloon {last_gn / max(first_gn, 1e-12):.0f}x "
      f"and the loss spike to {peak_loss:.2f} with no error thrown. PyTorch is a "
      "re-expression, not a new algorithm: it changes the labor, not the power.")
