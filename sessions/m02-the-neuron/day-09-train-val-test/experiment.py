# =============================================================================
# Day 9 — Train / Val / Test & Overfitting
# Produce task: "Catch a model overfitting with your own eyes."
#
# The idea in one line: a model must be graded on data it has NOT seen, or the
# grade is a lie. Here we make a small noisy dataset, SHUFFLE it, split it into
# a training pile and a validation pile, then fit models of growing FLEXIBILITY.
# We watch two numbers after each step: the loss on the training pile and the
# loss on the validation pile. We will SEE the moment memorizing beats learning.
#
# We use the flexibility knob the lesson suggests: a polynomial whose DEGREE we
# turn up step by step. Think of each higher degree as one more "epoch" of the
# model getting more room to bend. A low degree draws a smooth line (learning);
# a high degree whips up and down to touch every noisy dot (memorizing).
# =============================================================================

import numpy as np  # numpy does the fitting and the averages for us


def build_dataset(seed=42, n=40):
    """Make n dots from a smooth true curve y = sin(x) PLUS a little random noise."""
    # a fixed seed makes the random noise the SAME every run, so the answer is stable
    rng = np.random.default_rng(seed)
    # n evenly spaced inputs from -3 to 3 (a gentle rising-and-falling sine shape)
    x = np.linspace(-3, 3, n)
    # the TRUE pattern the model should learn
    true_y = np.sin(x)
    # random jiggle added on top — this is the "noise" a memorizer wrongly clings to
    noise = rng.normal(0, 0.35, size=n)
    y = true_y + noise
    return rng, x, y


def shuffle_and_split(rng, x, y, train_frac=0.8):
    """Trap fix #3: SHUFFLE before slicing, so each pile is a fair sample of the whole."""
    # a random re-ordering of the row positions 0..n-1
    perm = rng.permutation(len(x))
    # apply the SAME shuffle to inputs and answers so pairs stay together
    x, y = x[perm], y[perm]
    # cut point: first 80% is the training pile, last 20% is the validation pile
    cut = int(train_frac * len(x))
    x_train, y_train = x[:cut], y[:cut]
    x_val, y_val = x[cut:], y[cut:]
    return x_train, y_train, x_val, y_val


def mse(pred, target):
    """Mean squared error — the average of (prediction - answer) squared. Lower is better."""
    return float(np.mean((pred - target) ** 2))


def fit_and_score(x_train, y_train, x_val, y_val, degrees):
    """For each polynomial degree, fit on TRAIN only and score on BOTH piles."""
    train_losses = []  # loss on the pile the model learned from
    val_losses = []    # loss on the held-out pile it never saw during fitting
    for d in degrees:
        # fit a degree-d polynomial to the TRAINING pile only (only train updates the model)
        coef = np.polyfit(x_train, y_train, d)
        # predict on the training pile → how well it did on what it studied
        train_losses.append(mse(np.polyval(coef, x_train), y_train))
        # predict on the validation pile → the honest check on unseen data
        val_losses.append(mse(np.polyval(coef, x_val), y_val))
    return train_losses, val_losses


if __name__ == "__main__":
    # ---- 1. build the noisy dataset, then shuffle + split 80/20 --------------
    rng, x, y = build_dataset(seed=42, n=40)
    x_train, y_train, x_val, y_val = shuffle_and_split(rng, x, y, train_frac=0.8)
    print("dataset: %d points  ->  train %d / val %d"
          % (len(x), len(x_train), len(x_val)))

    # ---- 2. turn the flexibility knob up step by step ("epochs") -------------
    degrees = list(range(1, 16))  # degree 1 (smooth line) ... degree 15 (very wiggly)
    train_losses, val_losses = fit_and_score(x_train, y_train, x_val, y_val, degrees)

    # ---- 3. print the two curves as numbers ----------------------------------
    print("\ndegree :  train_loss  val_loss   gap")
    for d, tl, vl in zip(degrees, train_losses, val_losses):
        print("  %2d   :   %8.4f  %8.4f  %6.4f" % (d, tl, vl, vl - tl))

    # ---- 4. find the best model = the LOWEST validation point ----------------
    best_idx = int(np.argmin(val_losses))       # position of the smallest val loss
    best_degree = degrees[best_idx]              # the "early-stopping" point
    best_val = val_losses[best_idx]
    print("\nlowest validation loss at degree %d  (val_loss = %.4f)  <- best model / early-stopping point"
          % (best_degree, best_val))

    # the story of overfitting, made visible:
    print("train_loss keeps dropping: degree 1 = %.4f  ->  degree %d = %.4f"
          % (train_losses[0], degrees[-1], train_losses[-1]))
    print("but validation RISES past the minimum: best = %.4f  ->  most-flexible = %.4f"
          % (best_val, val_losses[-1]))

    # ---- 5. self-check: assert the expected story holds ----------------------
    expected_best_degree = 5  # with seed=42, validation bottoms out here

    # (a) more flexibility ALWAYS fits the training pile at least as well (monotone down)
    train_monotone = all(train_losses[i + 1] <= train_losses[i] + 1e-9
                         for i in range(len(train_losses) - 1))
    # (b) the best model is somewhere in the MIDDLE, not the simplest, not the most flexible
    best_in_middle = degrees[0] < best_degree < degrees[-1]
    # (c) overfitting: at the most-flexible degree the val loss is worse than the best point
    overfits_at_end = val_losses[-1] > best_val

    ok = (best_degree == expected_best_degree
          and train_monotone and best_in_middle and overfits_at_end)

    if ok:
        print("\n✅ you got it — train loss falls forever, but validation bottoms out at "
              "degree %d then rises: the widening gap IS overfitting." % best_degree)
    else:
        print("\n❌ not yet — expected lowest validation loss at degree %d "
              "(with train loss monotone-down and validation rising afterward)"
              % expected_best_degree)

    # a hard stop so the gate and the human both know if the demonstration held
    assert ok, "expected lowest validation loss at degree %d" % expected_best_degree
