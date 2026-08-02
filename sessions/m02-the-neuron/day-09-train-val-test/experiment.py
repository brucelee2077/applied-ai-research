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
    # The three counts are read ONCE into names, so the line the learner reads and the
    # counts the self-check pins are the same three values, not two separate lookups.
    n_all, n_train, n_val = len(x), len(x_train), len(x_val)
    # The whole LINE is rendered once and pinned as text below, because pinning the three
    # counts leaves their ORDER on the line free: "32 points -> train 40 / val 8" is a
    # different, wrong story told with the same three correctly-pinned numbers.
    dataset_line = "dataset: %d points  ->  train %d / val %d" % (n_all, n_train, n_val)
    print(dataset_line)

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
    # Every number the story below quotes is rendered ONCE, here, and the self-check pins
    # these same strings. Formatting a value again inside the print while the check looked
    # at the raw float would be two expressions for one number: corrupt the printed one and
    # the learner reads a wrong loss under a passing ✅.
    best_val_cell = "%.4f" % best_val
    # …and the headline LINE rendered once too, so the degree and the loss cannot trade
    # places on screen while both stay correctly pinned as values.
    best_line = ("lowest validation loss at degree %d  (val_loss = %s)"
                 "  <- best model / early-stopping point" % (best_degree, best_val_cell))
    print()
    print(best_line)

    # the story of overfitting, made visible:
    first_degree, last_degree = degrees[0], degrees[-1]
    train_start_cell = "%.4f" % train_losses[0]
    train_end_cell = "%.4f" % train_losses[-1]
    most_flexible_val = val_losses[-1]                 # the 'most-flexible' figure…
    most_flexible_cell = "%.4f" % most_flexible_val    # …rendered once, pinned below
    # Both of these lines are DIRECTION claims, and a direction lives in the ORDER of two
    # cells, not in either cell. Swapping the two operands of the second line prints
    # "best = 0.1077  ->  most-flexible = 0.0935" — validation FALLING, directly under a
    # label that says it RISES — with both cells still individually correct. So render each
    # line once, pin the rendered TEXT below, and print the pinned string.
    train_drop_line = ("train_loss keeps dropping: degree %d = %s  ->  degree %d = %s"
                       % (first_degree, train_start_cell, last_degree, train_end_cell))
    val_rise_line = ("but validation RISES past the minimum: best = %s  ->  most-flexible = %s"
                     % (best_val_cell, most_flexible_cell))
    print(train_drop_line)
    print(val_rise_line)

    # ---- 4b. the day's headline claim, measured: does the GAP really WIDEN? ---
    # One row is not a trend, and this validation pile holds only 8 points — so a single
    # degree can get lucky. Degree 15 does: its gap falls back to 0.0713, well under
    # degree 14's 0.3277. Quoting only the last row would therefore make the printed
    # summary argue AGAINST the day's claim. So state the claim the way the evidence
    # supports it: the gap across the flexible END of the sweep against the gap at the
    # best degree, plus where the gap is widest, plus the bumps themselves — printed, so
    # the noise is something the learner is told about instead of something we trimmed.
    gap_at_best = gaps[best_idx]
    late_degrees = degrees[-5:]                    # the flexible end: degrees 11-15
    late_gap_mean = sum(gaps[-5:]) / len(gaps[-5:])
    gap_ratio = late_gap_mean / gap_at_best
    widest_idx = int(np.argmax(gaps))              # found, not hand-picked
    widest_degree = degrees[widest_idx]
    gap_late = gaps[widest_idx]
    gap_best_cell = "%.4f" % gap_at_best
    gap_late_mean_cell = "%.4f" % late_gap_mean
    gap_ratio_cell = "%.1fx" % gap_ratio
    gap_widest_cell = "%.4f" % gap_late
    # Eight cells on one line, every one of them pinned as a value and every one of them
    # free to move: read in the wrong order this line reports the gap SHRINKING with
    # flexibility. Render once, pin the text, print the pinned string.
    gap_line = ("the GAP widens with flexibility: %s at degree %d  ->  %s averaged over degrees"
                " %d-%d (%s wider), widest %s at degree %d"
                % (gap_best_cell, best_degree, gap_late_mean_cell, late_degrees[0],
                   late_degrees[-1], gap_ratio_cell, gap_widest_cell, widest_degree))
    print(gap_line)

    # and the bumps, measured rather than glossed over: every degree whose gap came out
    # NARROWER than the degree before it. Small val pile -> a jumpy val curve, which is
    # the same "unlucky slice" effect the lesson warns about, seen from the other side.
    dip_degrees = [degrees[i] for i in range(1, len(gaps)) if gaps[i] < gaps[i - 1]]
    dip_cell = ", ".join(str(d) for d in dip_degrees)
    print("the rise is bumpy, not smooth: on a validation pile of only %d points the gap"
          " dips at degrees %s — read the trend across the degrees, never one row"
          % (n_val, dip_cell))

    # the acceptance step's last ask: how far past the best point would you have gone if
    # you had only ever looked at the training curve? It never turns around, so: all of it.
    overshoot = last_degree - best_degree
    # "past the best" is a direction too: printed the other way round it reads
    # "(degree 15 -> 5)", which is turning the knob DOWN, the opposite of the warning.
    overshoot_line = ("had you watched only the training curve, you would have kept turning"
                      " the knob %d more degrees past the best (degree %d -> %d)"
                      % (overshoot, best_degree, last_degree))
    print(overshoot_line)

    # The victory line quotes four cells that are all pinned already, so the only thing left
    # loose in it is the order — and "the gap widens from 0.2020 there to 0.0154" is the
    # day's claim backwards. Rendered here, pinned as text below, printed only when ok.
    victory_line = ("✅ you got it — train loss falls forever, but validation bottoms out at "
                    "degree %d then rises: the gap widens from %s there to %s across degrees "
                    "%d-%d. That widening IS overfitting."
                    % (best_degree, gap_best_cell, gap_late_mean_cell,
                       late_degrees[0], late_degrees[-1]))

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
    train_scale_pinned = (abs(train_losses[0] - 0.2886) < 1e-4  # printed above as 0.2886
                          and train_start_cell == "0.2886")     # …and as that exact text
    train_end_pinned = (abs(train_losses[-1] - 0.0364) < 1e-4   # printed above as 0.0364
                        and train_end_cell == "0.0364")
    val_scale_pinned = (abs(val_losses[best_idx] - 0.0935) < 1e-4  # printed as 0.0935
                        and best_val_cell == "0.0935")
    # the two ends of the sweep as the story labelled them, so a printed degree label
    # cannot drift away from the list that was actually swept
    sweep_ends_pinned = (first_degree, last_degree) == (1, 15)

    # (b2) the 'most-flexible' figure the summary quotes. Before this clause existed the
    #      number on that line was asserted NOWHERE — every late-stage claim pinned the
    #      widest degree instead — so the one value the summary ends on could drift freely.
    #      It is also the honest form of "validation RISES past the minimum": even at the
    #      luckiest flexible degree, validation still sits ABOVE the degree-5 minimum.
    most_flexible_pinned = (abs(most_flexible_val - 0.1077) < 1e-4  # printed as 0.1077
                           and most_flexible_cell == "0.1077"
                           and most_flexible_val > best_val)

    # (c) overfitting in ABSOLUTE terms, not "worse than the minimum" (which is true of any
    #     list whose minimum is not last). Both ends are pinned two-sided to the numbers
    #     printed above: a one-sided "> 0.30" threshold sat far from both 0.3742 and
    #     0.0935, so it could be moved to 0.20 (or 0.15) with no visible effect at all.
    late_val_pinned = abs(val_losses[widest_idx] - 0.3742) < 1e-4   # printed as 0.3742
    overfits_absolutely = (late_val_pinned
                           and val_losses[widest_idx] > 3.0 * val_losses[best_idx])

    # (d) the day's actual claim: the GAP (val - train) WIDENS. Tiny and positive at the
    #     best degree, large late. A loss whose two piles are on different scales makes
    #     this gap shrink or go negative, so this clause fails independently of (b).
    #     These are the SAME numbers the gap column printed, not a second subtraction.
    gap_widens = (0.0 < gap_at_best < 0.05 < 0.25 < gap_late
                  and abs(gap_at_best - 0.0154) < 1e-4     # printed above as 0.0154
                  and abs(gap_late - 0.3277) < 1e-4        # printed above as 0.3277
                  and widest_degree == 14)                 # and that IS the widest gap

    # (d2) the same claim stated across the flexible END rather than at one degree — the
    #      form the printed summary now uses, because degree 15 alone dips. The trend has
    #      to survive that dip: the mean gap over degrees 11-15 must still be many times
    #      the gap at the best degree, and the bumps are pinned as the bumps that exist
    #      (so quietly smoothing the curve, or losing a dip, fails here).
    gap_trend_widens = (abs(late_gap_mean - 0.2020) < 1e-4     # printed as 0.2020
                        and gap_late_mean_cell == "0.2020"
                        and gap_best_cell == "0.0154"
                        and gap_widest_cell == "0.3277"
                        and late_gap_mean > 5.0 * gap_at_best  # widened, not nudged
                        and gap_ratio_cell == "13.1x"
                        and late_degrees == [11, 12, 13, 14, 15]
                        and dip_degrees == [2, 5, 8, 12, 15]   # printed as that list
                        and dip_cell == "2, 5, 8, 12, 15"
                        and overshoot == 10)                   # 5 -> 15 on the knob

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
                         # and this run's own printed counts — the same three values the
                         # first line printed, not a second pass over the arrays
                         and (n_all, n_train, n_val) == (40, 32, 8))

    # (f) the WHOLE table, pinned as the exact TEXT that reached the screen. This is what
    #     makes the gap COLUMN honest and not just the gap variable — and pinning every
    #     row, not only the two the story quotes, means no row can be quietly rewritten.
    rows_pinned = rows == [
        "   1   :     0.2886    0.2416  -0.0470",
        "   2   :     0.2885    0.2392  -0.0493",
        "   3   :     0.0782    0.0941  0.0159",
        "   4   :     0.0782    0.0948  0.0166",
        "   5   :     0.0781    0.0935  0.0154",
        "   6   :     0.0717    0.0950  0.0233",
        "   7   :     0.0602    0.1782  0.1180",
        "   8   :     0.0595    0.1765  0.1170",
        "   9   :     0.0567    0.1744  0.1177",
        "  10   :     0.0553    0.2232  0.1680",
        "  11   :     0.0518    0.2690  0.2172",
        "  12   :     0.0491    0.1858  0.1367",
        "  13   :     0.0470    0.3041  0.2571",
        "  14   :     0.0466    0.3742  0.3277",
        "  15   :     0.0364    0.1077  0.0713",
    ]

    # (g) the six STORY lines, pinned as the exact text that reached the screen. Every
    #     number on them is already pinned above as a value — what is pinned here is the
    #     ORDER those numbers appear in, which is where each claim actually lives. Without
    #     these, swapping two operands at a print call prints "best = 0.1077 -> most-
    #     flexible = 0.0935" (validation falling, under a label saying it rises), or
    #     "32 points -> train 40", or a GAP line that reports shrinking — every cell still
    #     correct, every existing assert still green.
    lines_pinned = (
        dataset_line == "dataset: 40 points  ->  train 32 / val 8"
        and best_line == ("lowest validation loss at degree 5  (val_loss = 0.0935)"
                          "  <- best model / early-stopping point")
        and train_drop_line == ("train_loss keeps dropping: degree 1 = 0.2886"
                                "  ->  degree 15 = 0.0364")
        and val_rise_line == ("but validation RISES past the minimum: best = 0.0935"
                              "  ->  most-flexible = 0.1077")
        and gap_line == ("the GAP widens with flexibility: 0.0154 at degree 5  ->  0.2020"
                         " averaged over degrees 11-15 (13.1x wider), widest 0.3277 at"
                         " degree 14")
        and overshoot_line == ("had you watched only the training curve, you would have kept"
                               " turning the knob 10 more degrees past the best"
                               " (degree 5 -> 15)")
        and victory_line == ("✅ you got it — train loss falls forever, but validation bottoms"
                             " out at degree 5 then rises: the gap widens from 0.0154 there to"
                             " 0.2020 across degrees 11-15. That widening IS overfitting.")
    )

    ok = (best_degree == expected_best_degree
          and train_monotone
          and train_scale_pinned and train_end_pinned and val_scale_pinned
          and sweep_ends_pinned and most_flexible_pinned
          and overfits_absolutely and gap_widens and gap_trend_widens
          and split_frac_pinned
          and rows_pinned
          and lines_pinned)

    if ok:
        # the victory line was rendered — and pinned — above; here it is only printed
        print()
        print(victory_line)
    else:
        print("\n❌ not yet — expected lowest validation loss at degree %d, train loss "
              "monotone-down, train_loss[deg 1] = 0.2886 / val_loss[deg 5] = 0.0935 "
              "(same scale on both piles), an 80/20 split (%s from the probe sizes "
              "10/100/1000, and %d/%d here), val_loss[deg 14] = 0.3742, the most-flexible "
              "val_loss = 0.1077 (still above the minimum), and the gap widening from "
              "%.4f at the best degree to a mean of 0.2020 over degrees 11-15 "
              "(got %.4f, widest %.4f at degree %d)"
              % (expected_best_degree, probe_counts, n_train, n_val,
                 gap_at_best, late_gap_mean, gap_late, widest_degree))

    # a hard stop so the gate and the human both know if the demonstration held
    assert ok, ("expected lowest validation loss at degree %d with both piles on the same "
                "scale and a train-val gap that widens across the flexible end"
                % expected_best_degree)
