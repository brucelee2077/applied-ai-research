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
# turn up step by step. Each higher degree is one more FIT STEP on the CAPACITY axis —
# more room to bend — and NOT an epoch: an epoch is one full pass over the training
# split (Day 6's word, and this day's glossary), which is a training-TIME axis. Both
# axes produce the same U-shaped validation curve, which is why the lesson can teach
# early stopping on either, but they are two different knobs: stopping early caps
# training time, while stopping at a degree caps model capacity. A low degree draws a
# smooth line (learning); a high degree whips up and down to touch every noisy dot
# (memorizing).
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
    """Trap 2's fix: SHUFFLE before slicing, so each pile is a fair sample of the whole.

    (The lesson's Trap 1 is peeking at the test pile and Trap 3 is preprocessing
    leakage; this file implements the cure for the sorted-slice trap, Trap 2.)
    """
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
    TRAIN_FRAC = 0.8  # the promised split: 80% to train, 20% held out for validation
    rng, x, y = build_dataset(seed=42, n=40)
    x_train, y_train, x_val, y_val = shuffle_and_split(rng, x, y, train_frac=TRAIN_FRAC)
    print("dataset: %d points  ->  train %d / val %d"
          % (len(x), len(x_train), len(x_val)))

    # ---- 2. turn the flexibility knob up step by step (the CAPACITY axis) -----
    degrees = list(range(1, 16))  # degree 1 (smooth line) ... degree 15 (very wiggly)
    train_losses, val_losses = fit_and_score(x_train, y_train, x_val, y_val, degrees)

    # ---- 3. print the two curves as numbers ----------------------------------
    # Compute each gap ONCE, into a list. The table below prints these numbers and
    # the self-check at the bottom asserts these same numbers — one value, one
    # expression. (Printing `vl - tl` while the check recomputed the gap separately
    # would let a flipped subtraction show a wrong column and still pass.)
    gaps = [vl - tl for tl, vl in zip(train_losses, val_losses)]
    # Build each table row ONCE too, so the self-check can pin the exact text that
    # reached the screen — a stray transformation applied at the print statement
    # would otherwise show a wrong column that no number below could see.
    rows = ["  %2d   :   %8.4f  %8.4f  %6.4f" % (d, tl, vl, gap)
            for d, tl, vl, gap in zip(degrees, train_losses, val_losses, gaps)]
    print("\ndegree :  train_loss  val_loss   gap")
    for row in rows:
        print(row)

    # ---- 4. find the best model = the LOWEST validation point ----------------
    best_idx = int(np.argmin(val_losses))       # position of the smallest val loss
    # The best point on the axis we swept. On the training-TIME axis the same shape
    # appears epoch by epoch and stopping there is "early stopping"; here the axis is
    # capacity, so this is the same rule applied to model size, not to training length.
    best_degree = degrees[best_idx]
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

    # (a) more flexibility ALWAYS fits the training pile better — every single step must
    #     be a real DROP. No slack term: a rise of any size, however tiny, fails here.
    train_monotone = all(train_losses[i + 1] < train_losses[i]
                         for i in range(len(train_losses) - 1))

    # (b) both piles must be measured on the SAME scale, so the gap column means something.
    #     argmin alone cannot see scale: multiply every loss by 100 and it does not move.
    #     So pin the actual printed numbers from BOTH piles. This is what catches the
    #     classic reduction bug — summing instead of averaging, or dividing both piles by
    #     the train count — which leaves the argmin at degree 5 but breaks the comparison.
    train_scale_pinned = abs(train_losses[0] - 0.2886) < 1e-4   # printed above as 0.2886
    train_end_pinned = abs(train_losses[-1] - 0.0364) < 1e-4    # printed above as 0.0364
    val_scale_pinned = abs(val_losses[4] - 0.0935) < 1e-4       # printed above as 0.0935

    # (c) overfitting in ABSOLUTE terms, not "worse than the minimum" (which is true of any
    #     list whose minimum is not last). Both ends are pinned two-sided to the numbers
    #     printed above: a one-sided "> 0.30" threshold sat far from both 0.3742 and
    #     0.0935, so it could be moved to 0.20 (or 0.15) with no visible effect at all.
    late_val_pinned = abs(val_losses[13] - 0.3742) < 1e-4    # printed above as 0.3742
    overfits_absolutely = (late_val_pinned
                           and val_losses[13] > 3.0 * val_losses[4])

    # (d) the day's actual claim: the GAP (val - train) WIDENS. Tiny and positive at the
    #     best degree, large late. A loss whose two piles are on different scales makes
    #     this gap shrink or go negative, so this clause fails independently of (b).
    #     These are the SAME numbers the gap column printed, not a second subtraction.
    gap_at_best = gaps[4]
    gap_late = gaps[13]
    gap_widens = (0.0 < gap_at_best < 0.05 < 0.25 < gap_late
                  and abs(gap_at_best - 0.0154) < 1e-4     # printed above as 0.0154
                  and abs(gap_late - 0.3277) < 1e-4)       # printed above as 0.3277

    # (e) the split FRACTION must really be the promised 80/20. At n = 40 the `int()`
    #     absorbs it — train_frac 0.80 and 0.82 both cut at 32 — so the printed
    #     "train 32 / val 8" line proves nothing about the fraction on its own. Re-run
    #     the SAME splitter at sizes where the fraction IS visible in the counts, with
    #     its own throwaway generator so the main run's randomness is untouched.
    probe_counts = []
    for n_probe in (10, 100, 1000):
        probe_x = np.arange(n_probe, dtype=float)
        p_xt, _, p_xv, _ = shuffle_and_split(
            np.random.default_rng(0), probe_x, probe_x.copy(), train_frac=TRAIN_FRAC)
        probe_counts.append((len(p_xt), len(p_xv)))
    split_frac_pinned = (probe_counts == [(8, 2), (80, 20), (800, 200)]
                         # and this run's own printed counts
                         and (len(x_train), len(x_val)) == (32, 8))

    # (f) the two rows the story quotes, pinned as the exact TEXT that was printed.
    #     This is what makes the gap COLUMN honest and not just the gap variable.
    rows_pinned = (rows[4] == "   5   :     0.0781    0.0935  0.0154"
                   and rows[13] == "  14   :     0.0466    0.3742  0.3277")

    ok = (best_degree == expected_best_degree
          and train_monotone
          and train_scale_pinned and train_end_pinned and val_scale_pinned
          and overfits_absolutely and gap_widens and split_frac_pinned
          and rows_pinned)

    if ok:
        print("\n✅ you got it — train loss falls forever, but validation bottoms out at "
              "degree %d then rises: the widening gap IS overfitting." % best_degree)
    else:
        print("\n❌ not yet — expected lowest validation loss at degree %d, train loss "
              "monotone-down, train_loss[deg 1] = 0.2886 / val_loss[deg 5] = 0.0935 "
              "(same scale on both piles), an 80/20 split (%s from the probe sizes "
              "10/100/1000, and %d/%d here), val_loss[deg 14] = 0.3742, and the gap "
              "widening from %.4f at the best degree to 0.3277 by degree 14 (got %.4f)"
              % (expected_best_degree, probe_counts, len(x_train), len(x_val),
                 gap_at_best, gap_late))

    # a hard stop so the gate and the human both know if the demonstration held
    assert ok, ("expected lowest validation loss at degree %d with both piles on the same "
                "scale and a widening train-val gap" % expected_best_degree)
