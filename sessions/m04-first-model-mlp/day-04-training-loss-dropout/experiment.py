# day-04-training-loss-dropout — experiment
#
# Today's big idea in two lines of output:
#   With NO dropout the training loss dives to 0.05 while the held-out loss bottoms out at
#   epoch 9 and then turns back UP — the two curves split, and that split is overfitting.
#   Switch dropout on (p = 0.5, survivors scaled by 1/(1-p)) and the gap shrinks by half.
#
# Run it:  python3 sessions/m04-first-model-mlp/day-04-training-loss-dropout/experiment.py

import numpy as np  # arrays, matrix multiply (@), and seeded random numbers

# DEVIATION, on purpose and only for today: days 1, 2, 3 and 5 all run PIXELS = 784 (a 28x28
# image). Today needs a model that is clearly over-powered for its 300 training examples, so the
# input shrinks to an 8x8 = 64-pixel stamp while the hidden layer stays at 128. The tuple is
# written in the module's order — PIXELS, HIDDEN, CLASSES — so the line can be read at a glance.
PIXELS, HIDDEN, CLASSES = 64, 128, 10   # 8x8 "pixels" -> 128 hidden units -> 10 digits
DROP_P = 0.5                            # the lesson's dropout rate: bench half the units
EPOCHS = 40                             # full passes through the training data, both runs

def mse(gaps):
    # Mean squared error: square every gap, then average over every ELEMENT (day 2's
    # denominator). Big misses count for much more. Day 3 uses a different, named reduction.
    return float(np.mean(np.square(gaps)))

def cross_entropy(probs, labels):
    # Cross-entropy: -log(the probability the model gave the TRUE class), averaged. NOTE what
    # this eats: PROBABILITIES, already through softmax_rows. Day 5's nn.CrossEntropyLoss eats
    # raw scores instead, because it runs the softmax itself — day 5 prints what happens if you
    # carry today's habit over and hand it probabilities.
    return float(np.mean(-np.log(probs[np.arange(len(labels)), labels])))

def softmax_rows(scores):
    # Turn each row of raw scores into probabilities that are positive and sum to 1.
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))   # shift: keeps exp() safe
    return weights / weights.sum(axis=1, keepdims=True)

def keep_scale(p):
    # The 1/(1-p) rule: with a fraction p switched off, scale the survivors UP by this much,
    # so the average train-time signal matches what eval time (everyone kept) will see.
    return 1.0 / (1.0 - p)

def relu_gate(z):
    # The gate the backward pass multiplies by: STRICTLY greater than zero. A unit sitting exactly
    # ON the bend (z = 0) is OFF and must receive no blame. Writing (z >= 0) here changes nothing
    # in any random run — no seeded value lands on 0.0 — so Part 6 builds a net whose z IS 0 and
    # runs the gate both ways to make the one-character difference visible.
    return z > 0

def dropout_mask(dice, shape, p):
    # Bench each unit with probability p — so KEEP it with probability (1-p) — and give every
    # survivor the 1/(1-p) boost. A benched unit gets 0, a survivor gets 1/(1-p).
    return (dice.random(shape) > p) * keep_scale(p)

def forward_eval(x, W1, b1, W2, b2, dice=None, drop_p=0.0):
    # THE eval path, used in three places: the per-epoch validation score inside the loop, the
    # signal ladder in Part 5, and Part 7's "steady" predictions. Dropout stays off unless a
    # caller hands over dice, so if dropout ever leaked into this path Part 7's three repeats
    # would stop matching — the one code change that makes eval mode stop repeating.
    hidden = np.maximum(0, x @ W1 + b1)
    if drop_p:
        hidden = hidden * dropout_mask(dice, hidden.shape, drop_p)
    return hidden, softmax_rows(hidden @ W2 + b2)   # raw scores (day 1's logits) -> probabilities

# No dataset is cached on this machine, so we synthesize 8x8 = 64-pixel "digit stamps": each
# class gets one fixed prototype, and every example is that prototype plus noise. A fraction
# of the TRAINING labels is then made wrong on purpose — that wrong-label noise is what the
# lesson says an over-powered model memorizes. The validation labels are all left correct.
# SECOND DEVIATION: day 1 pinned its pixels to bytes scaled into exactly [0.0, 1.0]. These
# stamp values are prototype + N(0, 1), so they are centred near 0 and run negative — the other
# standard input scaling, not day 1's. Part 2 prints the real range and pins it, so the change
# is on the page rather than hidden behind the word "pixels".
def make_stamps(prototypes, per_class, wrong_fraction, rng):   # rng last, as on days 1 and 3
    labels = np.repeat(np.arange(CLASSES), per_class)
    images = prototypes[labels] + rng.normal(0, 1.0, (len(labels), PIXELS))
    spoiled = rng.permutation(len(labels))[:int(round(wrong_fraction * len(labels)))]
    labels = labels.copy()
    labels[spoiled] = (labels[spoiled] + rng.integers(1, CLASSES, size=len(spoiled))) % CLASSES
    return images, labels

def train_run(train_x, train_y, val_x, val_y, drop_p, epochs=EPOCHS, lr=0.05, batch=16):
    """Day 3's mini-batch loop, now recording BOTH average losses every epoch."""
    start = np.random.default_rng(1)                 # both runs get the same starting weights
    # Day 1's He spread and Day 1's (in, out) layout: W1 is (64, 128), W2 is (128, 10).
    W1 = start.normal(0, np.sqrt(2 / PIXELS), (PIXELS, HIDDEN)); b1 = np.zeros(HIDDEN)
    W2 = start.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, CLASSES)); b2 = np.zeros(CLASSES)
    dice = np.random.default_rng(2)                  # the dice that bench neurons
    order = np.random.default_rng(3)                 # the shuffler
    history = []
    for _ in range(epochs):
        shuffled = order.permutation(len(train_y))
        batch_losses = []
        for first in range(0, len(shuffled), batch):          # --- train mode: dropout ON ---
            rows = shuffled[first:first + batch]
            x, y = train_x[rows], train_y[rows]
            z1 = x @ W1 + b1                                  # days 1-3's pre-activation
            hidden = np.maximum(0, z1)                        # ReLU
            # mask 1.0 means "keep everything, scale nothing" — that is the no-dropout run.
            mask = dropout_mask(dice, hidden.shape, drop_p) if drop_p else 1.0
            hidden = hidden * mask
            probs = softmax_rows(hidden @ W2 + b2)
            batch_losses.append(cross_entropy(probs, y))
            d_out = probs.copy()                       # day 2's seed delta, for softmax+x-entropy
            d_out[np.arange(len(y)), y] -= 1
            d_out /= len(y)
            d_hidden = (d_out @ W2.T) * mask * relu_gate(z1)   # benched units get no gradient
            W2 -= lr * (hidden.T @ d_out); b2 -= lr * d_out.sum(axis=0)
            W1 -= lr * (x.T @ d_hidden);   b1 -= lr * d_hidden.sum(axis=0)
        # --- eval mode: dropout OFF, every unit kept, no scaling ---
        _, val_probs = forward_eval(val_x, W1, b1, W2, b2)
        history.append((float(np.mean(batch_losses)), cross_entropy(val_probs, val_y)))
    return history, (W1, b1, W2, b2)

def validation_bottom(val_losses):
    # The early-stopping point: the 1-based epoch whose held-out loss is the lowest. TIES GO TO THE
    # EARLIER epoch (np.argmin returns the first minimum) — the cheaper model wins a draw. Every
    # real curve here has a unique minimum, so the tie rule never shows up in a run; the self-check
    # feeds it two drawn curves on purpose so the policy is tested rather than assumed.
    return int(np.argmin(val_losses)) + 1

def one_backward_step(x, W1, b1, W2, b2, labels, gate=relu_gate):
    """The training loop's own forward + backward, small enough to run on hand-built weights so a
    net whose z1 is exactly 0 can be checked. `gate` is the loop's strict rule by default; Part 6
    passes the loose (z >= 0) spelling to show what that one character would cost."""
    z1 = x @ W1 + b1
    hidden = np.maximum(0, z1)
    probs = softmax_rows(hidden @ W2 + b2)
    d_out = probs.copy()                       # day 2's seed delta, for softmax + cross-entropy
    d_out[np.arange(len(labels)), labels] -= 1
    d_out /= len(labels)
    d_hidden = (d_out @ W2.T) * gate(z1)       # the gate under test
    return z1, cross_entropy(probs, labels), x.T @ d_hidden

def show_curve(history, every=4):
    # Print epoch / train loss / validation loss, and mark the validation bottom. Every number
    # handed back is a number this function PRINTED: each row is rounded ONCE into shown_rows and
    # that same value is both printed and returned, so the summary lines and the self-check below
    # read the table the learner reads. Corrupt a printed loss and the claims move with it.
    # Rounding once is still not enough for two things that are decided AT the print: which row
    # wears the "best model" marker, and which order the three columns come out in. So each row
    # is built as one line of TEXT, that text is what gets printed, and the exact list of lines
    # is handed back for the self-check to pin. An inverted marker (`i != bottom`), a deleted
    # marker, and a train/validation column swap all change the text and so all fail there.
    bottom = validation_bottom([v for _, v in history]) - 1
    shown_rows = []                       # (epoch, train loss, validation loss) exactly as printed
    shown_lines = []                      # the same rows as the exact text that reached the screen
    for i, (t, v) in enumerate(history):
        if i % every == 0 or i == bottom or i == len(history) - 1:
            epoch, shown_t, shown_v = i + 1, round(t, 4), round(v, 4)
            shown_rows.append((epoch, shown_t, shown_v))
            marker = "  <-- validation bottom (best model)" if i == bottom else ""
            line = "  epoch %2d   train %.4f   validation %.4f%s" % (epoch, shown_t, shown_v,
                                                                     marker)
            shown_lines.append(line)
            print(line)
    last_train, last_val = shown_rows[-1][1], shown_rows[-1][2]
    # the lowest held-out loss is read off the row the printout MARKED, not recomputed here
    bottom_val = [v for epoch, _, v in shown_rows if epoch == bottom + 1][0]
    return bottom + 1, last_train, last_val, bottom_val, shown_lines


if __name__ == "__main__":
    # --- Part 1: the two loss flavors, on the lesson's own numbers ---------
    gaps = [0.5, 1.0, 2.0, 4.0]
    # Bind what gets printed, then print THAT — the self-check below reads these same two names,
    # so the squares and the MSE on the page and the checked numbers cannot drift apart.
    shown_squares = [g * g for g in gaps]
    shown_mse = round(mse(gaps), 4)
    # Built as one line, pinned as that line, then printed: "gaps -> squared -> MSE" only teaches
    # squaring if those three columns come out in that order.
    mse_line = "MSE  gaps %s -> squared %s -> MSE %s" % (gaps, shown_squares, shown_mse)
    print(mse_line)
    demo_probs = (0.90, 0.50, 0.01)          # confident-and-right, hedging, confident-and-wrong
    # Score each row through the SAME cross_entropy the training loop uses: two columns, the
    # true class is column 0, so the loss is -log(the probability given to the true class).
    demo_losses = [round(cross_entropy(np.array([[p, 1 - p]]), np.array([0])), 2)
                   for p in demo_probs]
    # The three rows are the probability -> loss MAP, so the pairing is the claim. Build the rows
    # as text and pin the whole list below: a reversed zip would print 0.90 next to 4.61 while
    # both lists stayed individually correct.
    ce_lines = [f"cross-entropy, {p_true:.2f} on the true class -> loss {loss:.2f}"
                for p_true, loss in zip(demo_probs, demo_losses)]
    for line in ce_lines:
        print(line)
    pain_ratio = demo_losses[2] / demo_losses[0]        # 4.61 / 0.11 — how much worse it hurts
    shown_pain_ratio = round(pain_ratio, 1)             # the 41.9 the next line prints
    pain_shown = round(pain_ratio / 10) * 10            # to the nearest ten, as the lesson quotes
    # The two losses in the bracket are read out of the list AT the print, by index. Swap those
    # two indices and the page says confident-and-WRONG is the cheap one — the exact opposite of
    # the day's point — with every value still pinned. So the rendered line is pinned instead.
    pain_line = (f"-> confident-and-wrong (0.01) hurts about {pain_shown}x confident-and-right"
                 f" (0.90)  ({demo_losses[2]:.2f} / {demo_losses[0]:.2f}"
                 f" = {shown_pain_ratio:.1f})")
    print(pain_line)

    # --- Part 2: split the data into train and held-out validation ---------
    rng = np.random.default_rng(0)
    prototypes = 0.5 * rng.normal(0, 1, (CLASSES, PIXELS))     # one stamp per digit
    train_x, train_y = make_stamps(prototypes, 30, 0.4, rng)   # small slice, 40% wrong labels
    val_x, val_y = make_stamps(prototypes, 50, 0.0, rng)       # never trained on, clean labels
    # Day 1 counted 101770 "knobs" = weights PLUS both bias vectors. This count is WEIGHTS only
    # (the biases would add HIDDEN + CLASSES = 138); it is the number the over-powered argument
    # rests on, so the two days are quoting two different quantities under similar words.
    weights = PIXELS * HIDDEN + HIDDEN * CLASSES
    # The four shapes and the example count are bound here and printed below, so the self-check
    # reads the very numbers on the page instead of asking the arrays a second time.
    shown_train_shape, shown_train_labels = train_x.shape, train_y.shape
    shown_val_shape, shown_val_labels = val_x.shape, val_y.shape
    shown_train_count = len(train_y)
    print(f"\ntrain images {shown_train_shape} labels {shown_train_labels}"
          f" | held-out val {shown_val_shape} labels {shown_val_labels}")
    print(f"model {PIXELS} -> {HIDDEN} -> {CLASSES} = {weights} weights for {shown_train_count}"
          " training examples (over-powered on purpose)")
    # The pixel-range deviation, measured. Day 1's rule was bytes scaled into exactly [0.0, 1.0];
    # these stamp values are prototype + N(0, 1), so they run negative and are not bounded by 1.
    stamp_lo, stamp_hi = float(train_x.min()), float(train_x.max())
    shown_stamp_lo, shown_stamp_hi = round(stamp_lo, 2), round(stamp_hi, 2)   # printed AND checked
    print(f"stamp values run ({shown_stamp_lo:.2f}, {shown_stamp_hi:.2f}) — NOT day 1's [0.0, 1.0]"
          " byte pixels: these are centred near 0 with spread ~1, the other standard input scaling")
    # Three predictions, written down BEFORE the runs and tested in the self-check below.
    print("predict: (1) train loss -> near 0 while validation turns UP; (2) the rising curve is"
          " VALIDATION; (3) dropout makes the gap SMALLER")

    # --- Part 3: Run A — no dropout, watch the two curves split ------------
    # Warm-up: hand the lesson's own six-epoch table to the same bottom-finder that is about to
    # mark the real run, and see that it lands where the lesson read the bottom by eye.
    lesson_val = [0.62, 0.44, 0.35, 0.36, 0.42, 0.51]             # the lesson's 6-epoch table
    shown_lesson_bottom = validation_bottom(lesson_val)   # printed here, checked below, once each
    shown_lesson_min = min(lesson_val)
    print(f"\nlesson's table {lesson_val} -> the bottom-finder says epoch"
          f" {shown_lesson_bottom} (the lesson read epoch 3, loss {shown_lesson_min})")
    print("\n--- Run A: NO dropout ---")
    hist_a, params_a = train_run(train_x, train_y, val_x, val_y, drop_p=0.0)
    best_a, train_a, val_a, best_val_a, lines_a = show_curve(hist_a)
    rise_a, gap_a = val_a - best_val_a, val_a - train_a
    shown_rise_a, shown_gap_a = round(rise_a, 3), round(gap_a, 3)   # the two summary numbers
    b1_a, b2_a = params_a[1], params_a[3]
    shown_b1_spread, shown_b2_top = round(float(b1_a.std()), 4), round(float(b2_a.max()), 4)
    # Run A's verdict, in the same shape Run B's gets below: the three lines are built once, pinned
    # as text in the self-check, and printed from that text. The gap on the second line is the day's
    # headline number and Run B re-quotes it, so a value doubled at THIS print would put two
    # different "Run A gaps" on one page. The two bias rows started as exact zeros, so whatever they
    # hold now was learned by the loop's two bias update lines.
    summary_lines_a = [
        f"  validation was lowest at epoch {best_a}, then rose by {shown_rise_a:.3f}"
        " while train dived",
        f"  final gap (validation - train) = {shown_gap_a:.3f} <- this growing gap IS overfitting",
        f"  biases started at 0 and learned: b1 spread {shown_b1_spread:.4f},"
        f" b2 top {shown_b2_top:.4f}",
    ]
    for line in summary_lines_a:
        print(line)

    # --- Part 4: Run B — the same run, now with dropout p = 0.5 ------------
    shown_keep = keep_scale(DROP_P)          # the survivor boost this run really used
    print(f"\n--- Run B: dropout p = {DROP_P} (train mode only, survivors x{shown_keep}) ---")
    hist_b, params_b = train_run(train_x, train_y, val_x, val_y, drop_p=DROP_P)
    best_b, train_b, val_b, best_val_b, lines_b = show_curve(hist_b)
    rise_b, gap_b = val_b - best_val_b, val_b - train_b
    shown_rise_b, shown_gap_b = round(rise_b, 3), round(gap_b, 3)
    shown_best_val_b, shown_best_val_a = round(best_val_b, 3), round(best_val_a, 3)
    shown_train_b, shown_train_a = round(train_b, 3), round(train_a, 3)
    # These three lines are the day's verdict, and each one is a Run B vs Run A CONTRAST: swap the
    # two operands in any of them and the page says dropout made the gap wider, reached a worse
    # best, or cost nothing — while every value in it stays correctly pinned. So the three
    # rendered lines are pinned as text below and printed from that same text.
    compare_lines = [
        f"  best validation {shown_best_val_b:.3f} at epoch {best_b}"
        f" (Run A reached {shown_best_val_a:.3f})",
        f"  final gap {shown_gap_b:.3f} vs Run A's {shown_gap_a:.3f};"
        f" late rise {shown_rise_b:.3f} vs {shown_rise_a:.3f}",
        f"  not free: after the same {EPOCHS} epochs its train loss is {shown_train_b:.3f},"
        f" not {shown_train_a:.3f}",
    ]
    for line in compare_lines:
        print(line)

    # --- Part 5: why the 1/(1-p) scaling is there --------------------------
    kept_total = 2 * 1.0 * keep_scale(DROP_P)    # lesson's ladder: 4 units worth 1, keep 2, x2
    all_kept_total = 4 * 1.0                     # the same four units with nobody benched
    print(f"\n4 units kept as-is -> total {all_kept_total} | 2 benched, survivors"
          f" x{shown_keep} -> total {kept_total}")
    # The same ladder at a SECOND rate, so the line above cannot be a constant in disguise: at
    # p = 0.75 the surviving count and the boost both change (1 unit, x4) while the total still
    # comes back to 4 — no single hardcoded number is right for both rungs. The boost is then
    # measured on 4000 real maskings, so the printed factor is the one dropout_mask really applies.
    shown_keep_075 = keep_scale(0.75)
    kept_total_075 = 1 * 1.0 * shown_keep_075
    probe = dropout_mask(np.random.default_rng(5), (4000,), 0.75)
    probe_kept, probe_boost = round(float((probe > 0).mean()), 3), float(probe.max())
    print(f"same ladder at p = 0.75 -> 1 kept, survivors x{shown_keep_075} -> total"
          f" {kept_total_075} | 4000 real maskings: {probe_kept:.3f} survived, each x{probe_boost}")
    W1, b1, W2, b2 = params_b
    hidden_eval, _ = forward_eval(val_x[:1], W1, b1, W2, b2)   # eval mode: all units kept
    dice = np.random.default_rng(11)
    masks = dropout_mask(dice, (500,) + hidden_eval.shape, DROP_P)   # 500 random benchings
    eval_signal = float(hidden_eval.sum())
    scaled_mean = float((hidden_eval * masks).sum(axis=(1, 2)).mean())
    unscaled_mean = float((hidden_eval * (masks > 0)).sum(axis=(1, 2)).mean())
    # The three signal totals: bound once, printed, and compared to each other in the self-check.
    shown_eval_signal = round(eval_signal, 3)
    shown_scaled_mean, shown_unscaled_mean = round(scaled_mean, 3), round(unscaled_mean, 3)
    # Which of these two averages is the SCALED one is the whole 1/(1-p) lesson. Swapped columns
    # would teach that scaling halves the signal, so the line is pinned as the text it prints.
    signal_line = (f"eval signal {shown_eval_signal:.3f} | dropout averages"
                   f" {shown_scaled_mean:.3f} (scaled)"
                   f" vs {shown_unscaled_mean:.3f} (if you forget to scale)")
    print(signal_line)

    # --- Part 6: the gate is STRICT — a unit sitting exactly on the bend ----
    # Every z1 in the two runs above came from random weights, so none of them was ever exactly
    # 0.0 and the difference between (z > 0) and (z >= 0) never showed. Here is a hand-built net
    # whose second hidden unit lands exactly ON the bend: 3*1 + 1*(-3) = 0. Distinct numbers from
    # the earlier days' hinge nets, run through the loop's own backward step.
    hinge_x = np.array([[3.0, 1.0]])                        # one row, two features
    hinge_W1 = np.array([[2.0, 1.0], [-1.0, -3.0]])         # column 2 cancels to exactly 0
    hinge_W2 = np.array([[1.0, 0.0], [0.0, 1.0]])           # each hidden unit feeds one class
    hinge_y = np.array([1])                                 # the true class is the dead unit's
    hinge_z1, hinge_loss, hinge_dW1 = one_backward_step(
        hinge_x, hinge_W1, np.zeros(2), hinge_W2, np.zeros(2), hinge_y)
    loose_z1, loose_loss, loose_dW1 = one_backward_step(
        hinge_x, hinge_W1, np.zeros(2), hinge_W2, np.zeros(2), hinge_y, gate=lambda z: z >= 0)
    shown_hinge_z1 = np.round(hinge_z1[0], 4)
    shown_hinge_loss, shown_loose_loss = round(hinge_loss, 4), round(loose_loss, 4)
    shown_live_col, shown_dead_col = np.round(hinge_dW1[:, 0], 4), np.round(hinge_dW1[:, 1], 4)
    shown_loose_dead_col = np.round(loose_dW1[:, 1], 4)
    # The point is WHICH unit gets the blame: live unit -> a real gradient, z = 0 unit -> exactly
    # zero, and the loose gate handing that same dead unit a gradient instead. Exchange the two
    # columns and the page teaches the bug as the fix, so both lines are pinned as rendered text.
    hinge_line = (f"hinge net: z1 = {shown_hinge_z1} (loss {shown_hinge_loss}) -> dW1 for the live"
                  f" unit {shown_live_col}, for the z = 0 unit {shown_dead_col}")
    loose_line = (f"the same step with a (z >= 0) gate (loss {shown_loose_loss}, unchanged) hands"
                  f" that dead unit {shown_loose_dead_col} instead — same forward, different"
                  " weights: the one-character bug a random run never shows")
    print("\n" + hinge_line)
    print(loose_line)

    # --- Part 7: the silent bug — dropout left ON at prediction time -------
    # Both sets of rows go through the SAME forward_eval the loop scored validation with; the
    # ONLY difference is whether dice and a drop rate are handed in. So "eval mode repeats" is a
    # claim about that code path, not about re-reading one expression three times.
    steady_rows = [forward_eval(val_x[:1], W1, b1, W2, b2)[1][0] for _ in range(3)]   # eval mode
    dice = np.random.default_rng(7)
    flicker_rows = [forward_eval(val_x[:1], W1, b1, W2, b2, dice, DROP_P)[1][0]       # oops: ON
                    for _ in range(3)]
    # The four printed lists, bound once each: the self-check reads THESE, so a corrupted digit or
    # confidence in the printout can no longer sit under a passing tick.
    flicker_digits = [int(r.argmax()) for r in flicker_rows]
    flicker_conf = [round(float(r.max()), 3) for r in flicker_rows]
    steady_digits = [int(r.argmax()) for r in steady_rows]
    steady_conf = [round(float(r.max()), 3) for r in steady_rows]
    # Two lines whose MEANING is which label sits with which list: dropout-ON must be the row that
    # disagrees with itself, eval mode the row that repeats. Swap the payloads and the page blames
    # eval mode for the flicker. Pinned as the exact two lines that reached the screen.
    predict_lines = [
        "same picture, dropout still ON -> digit %s confidence %s" % (flicker_digits, flicker_conf),
        "same picture, eval mode        -> digit %s confidence %s" % (steady_digits, steady_conf),
    ]
    print("\n" + predict_lines[0])
    print(predict_lines[1])
    print("\nremember: a low TRAINING loss proves nothing about new data — the held-out loss is")
    print("the honest number. And dropout is not free: it learns slower, and it can hurt a model")
    print("that already generalizes. Reach for it when you SEE the gap grow.")

    # --- Self-check: one boolean per claim ---------------------------------
    # Every expected value below is written down here — the lesson's demo numbers, and the
    # numbers this script printed when it was written — never re-derived from the code above.
    # The shape claims read the recorded history of the two runs, so a model that learned
    # nothing (a flat curve) fails them instead of agreeing with itself.
    lesson_numbers_ok = (
        shown_squares == [0.25, 1.0, 4.0, 16.0]                   # the squares Part 1 printed
        and shown_mse == 5.3125                                   # lesson demo: MSE 5.31
        and demo_losses == [0.11, 0.69, 4.61]                     # lesson: 0.11 / 0.69 / 4.61
        and pain_shown == 40                                      # the "about 40x" Part 1 printed
        and shown_pain_ratio == 41.9                              # and the exact ratio beside it
        # The bottom-finder that marked BOTH runs above, on three curves whose answer is known
        # by hand: the lesson's table (bottom at epoch 3), a curve still falling at the end
        # (bottom = the last epoch), and one rising from the start (bottom = the first epoch).
        and shown_lesson_bottom == 3 and shown_lesson_min == 0.35
        and validation_bottom([0.9, 0.8, 0.7]) == 3
        and validation_bottom([0.7, 0.8, 0.9]) == 1
        # And the tie rule, which is otherwise invisible: two epochs equally low, EARLIEST wins.
        # Switching argmin for a last-minimum policy would flip these two from 2 to 3 and from 1
        # to 3 — with a plain min() there would be nothing here to notice.
        and validation_bottom([0.5, 0.4, 0.4, 0.6]) == 2
        and validation_bottom([0.4, 0.6, 0.4]) == 1)
    # The two places this day leaves the module's spine, pinned so neither can pass as an
    # accident: a 64-pixel input (days 1/2/3/5 use 784) with the tuple still in PIXELS, HIDDEN,
    # CLASSES order, and stamp values that really do leave day 1's [0.0, 1.0] band on BOTH sides.
    # The shapes and the weight count are the ones Part 2 printed, read back here.
    inputs_ok = ((PIXELS, HIDDEN, CLASSES) == (64, 128, 10)
                 and shown_stamp_lo == -4.76 and shown_stamp_hi == 4.36
                 and stamp_lo < 0.0 < 1.0 < stamp_hi
                 and shown_train_shape == (300, 64) and shown_train_labels == (300,)
                 and shown_val_shape == (500, 64) and shown_val_labels == (500,)
                 and shown_train_count == 300 and weights == 9472)
    # A wrong scale factor shows up here: scaled must match eval, unscaled must be half of it.
    # A mask that KEPT with probability p would look fine at p = 0.5, so the second rung Part 5
    # printed is pinned too: at p = 0.75 only about a quarter of 4000 units may survive, each
    # boosted by 4 — numbers no constant folded into the p = 0.5 rung could satisfy.
    scale_ok = (shown_keep == 2.0 and abs(kept_total - 4.0) < 1e-12 and all_kept_total == 4.0
                and abs(shown_scaled_mean - shown_eval_signal) / shown_eval_signal < 0.02
                and abs(shown_unscaled_mean / shown_eval_signal - 0.5) < 0.02
                and shown_keep_075 == 4.0 and abs(kept_total_075 - 4.0) < 1e-12
                and abs(probe_kept - 0.25) < 0.02
                and abs(probe_boost - 4.0) < 1e-12)
    # Predictions 1 and 2, read off Run A's own recorded history. TRAIN is the curve that keeps
    # falling — every one of the last ten epochs, and it ends far below where it started. The
    # VALIDATION curve must make a real U: it drops a long way into the epoch the printout
    # marked, that epoch sits well before the end, and all ten final epochs are back above the
    # bottom. A model that learned nothing would print a flat curve and fail every one of these.
    train_keeps_falling = all(hist_a[i][0] < hist_a[i - 1][0] for i in range(EPOCHS - 10, EPOCHS))
    train_dived = hist_a[0][0] - train_a > 1.0                 # 2.822 -> 0.047, a fall of 2.77
    val_fell_into_bottom = hist_a[0][1] - best_val_a > 0.50    # 2.067 -> 1.231, a fall of 0.84
    val_climbed_back = all(v > best_val_a + 0.05 for _, v in hist_a[-10:])   # margins 0.158 up
    split_ok = (train_keeps_falling and train_dived and val_fell_into_bottom and val_climbed_back
                and 2 <= best_a <= 30 and shown_rise_a > 0.10
                and abs(shown_train_a - 0.047) < 0.05
                and abs(best_val_a - 1.231) < 0.05 and abs(val_a - 1.422) < 0.05)
    # Prediction 3: dropout halves the gap (1.375 -> 0.742), reaches a better best, rises less,
    # and costs something — its train loss is still high after the same EPOCHS epochs. The gap is
    # narrower at every one of the last ten epochs, not only on the epoch that gets printed.
    gap_smaller_all_along = all(hist_b[i][1] - hist_b[i][0] < hist_a[i][1] - hist_a[i][0]
                                for i in range(EPOCHS - 10, EPOCHS))
    dropout_ok = (abs(shown_gap_a - 1.375) < 0.05 and abs(shown_gap_b - 0.742) < 0.05
                  and gap_smaller_all_along
                  and shown_best_val_b == 1.075 and shown_best_val_a == 1.231
                  and best_val_b < best_val_a and shown_rise_b < shown_rise_a / 2
                  and shown_train_b > 0.20)
    # The two bias update lines: b1 and b2 start as exact zeros, so these two pins fail flat (at
    # 0.0000) if either line stops running, and shift far past the printed 4 decimals if either
    # one uses the wrong gradient or step. The run is fully seeded, so both numbers are pinned to
    # exactly what the line above printed.
    biases_learned_ok = (shown_b1_spread == 0.0268 and shown_b2_top == 0.0689)
    # The eval-mode bug: three dropout-ON passes all disagree, three eval passes are identical,
    # and leaving dropout on flipped the predicted digit on at least one pass. Both sets came out
    # of forward_eval, so dropout leaking into that path breaks the "identical" clause. The digit
    # and confidence clauses read the four lists that were printed — and they read them in ORDER,
    # pinned to the exact printed lists. "at least one digit differs" and "three distinct
    # confidences" are both blind to a reversed list, so each is now backed by an exact pin.
    steady, top = steady_rows[0], steady_digits[0]
    eval_bug_ok = (len({r.tobytes() for r in flicker_rows}) == 3
                   and len({r.tobytes() for r in steady_rows}) == 1
                   and steady_digits == [top, top, top] and len(set(steady_conf)) == 1
                   and any(d != top for d in flicker_digits)
                   and flicker_digits == [7, 0, 0]
                   and len(set(flicker_conf)) == 3
                   and flicker_conf == [0.481, 0.954, 0.753]
                   and steady_digits == [0, 0, 0] and steady_conf == [0.919, 0.919, 0.919]
                   and max(float(steady.max() - r[top]) for r in flicker_rows) > 0.30)
    # The gate's strictness, the one thing no random run can show: this net's second hidden unit
    # sits exactly ON the bend, so the strict (z > 0) gate must hand it EXACTLY zero blame while
    # the loose (z >= 0) spelling hands it a real gradient. The forward pass and the loss are the
    # same either way — only the weights that would be learned differ, which is what makes the
    # change silent. Every number here is one the two lines above printed.
    hinge_ok = (np.array_equal(shown_hinge_z1, [5.0, 0.0])
                and np.array_equal(loose_z1, hinge_z1)
                and shown_hinge_loss == 5.0067 and shown_loose_loss == shown_hinge_loss
                and bool(relu_gate(0.0)) is False and bool(relu_gate(1e-12)) is True
                and np.array_equal(shown_dead_col, [0.0, 0.0])
                and np.array_equal(shown_live_col, [2.9799, 0.9933])
                and np.array_equal(shown_loose_dead_col, [-2.9799, -0.9933])
                and np.array_equal(np.round(loose_dW1[:, 0], 4), shown_live_col))
    # The two curve TABLES, pinned as the exact text that reached the screen — the rows that carry a
    # claim, not every row. Each table's FIRST row (where the two curves start), its LAST row (where
    # they end up) and the row wearing the "<-- validation bottom (best model)" marker are what the
    # story is told with, and the marked row is read out by FILTERING on the marker, so the list also
    # says it is the ONLY marked row: inverting the test to `i != bottom` fills that list with the
    # other rows, deleting the marker empties it. Those three rows per table also make the epoch /
    # train / validation COLUMN order part of the claim. The rows in between are a monotone stretch
    # of the same curve — "validation bottoms out at epoch 9 then rises" is already asserted, epoch
    # by epoch, by val_climbed_back and gap_smaller_all_along over the recorded history.
    table_lines_ok = (
        len(lines_a) == 11
        and lines_a[0] == "  epoch  1   train 2.8215   validation 2.0670"
        and [line for line in lines_a if "validation bottom" in line] == [
            "  epoch  9   train 0.6088   validation 1.2307  <-- validation bottom (best model)",
        ]
        and lines_a[-1] == "  epoch 40   train 0.0471   validation 1.4218"
        and len(lines_b) == 12
        and lines_b[0] == "  epoch  1   train 3.5068   validation 2.1104"
        and [line for line in lines_b if "validation bottom" in line] == [
            "  epoch 14   train 1.1009   validation 1.0750  <-- validation bottom (best model)",
        ]
        and lines_b[-1] == "  epoch 40   train 0.3796   validation 1.1219")
    # The other lines whose LAYOUT carries the claim, pinned the same way: the loss demos in Part
    # 1, the three Run B vs Run A contrasts, the scaled-vs-unscaled signal, the strict-gate pair,
    # and the flicker-vs-repeat pair. Every value in them is already pinned above; what is pinned
    # HERE is the sentence the learner actually reads, which no value pin can reach.
    rendered_lines_ok = (
        mse_line == "MSE  gaps [0.5, 1.0, 2.0, 4.0] -> squared [0.25, 1.0, 4.0, 16.0] -> MSE 5.3125"
        and ce_lines == [
            "cross-entropy, 0.90 on the true class -> loss 0.11",
            "cross-entropy, 0.50 on the true class -> loss 0.69",
            "cross-entropy, 0.01 on the true class -> loss 4.61",
        ]
        and pain_line == ("-> confident-and-wrong (0.01) hurts about 40x confident-and-right "
                          "(0.90)  (4.61 / 0.11 = 41.9)")
        and summary_lines_a == [
            "  validation was lowest at epoch 9, then rose by 0.191 while train dived",
            "  final gap (validation - train) = 1.375 <- this growing gap IS overfitting",
            "  biases started at 0 and learned: b1 spread 0.0268, b2 top 0.0689",
        ]
        and compare_lines == [
            "  best validation 1.075 at epoch 14 (Run A reached 1.231)",
            "  final gap 0.742 vs Run A's 1.375; late rise 0.047 vs 0.191",
            "  not free: after the same 40 epochs its train loss is 0.380, not 0.047",
        ]
        and signal_line == ("eval signal 60.939 | dropout averages 60.811 (scaled) vs 30.405 "
                            "(if you forget to scale)")
        and hinge_line == ("hinge net: z1 = [5. 0.] (loss 5.0067) -> dW1 for the live unit "
                           "[2.9799 0.9933], for the z = 0 unit [0. 0.]")
        and loose_line == ("the same step with a (z >= 0) gate (loss 5.0067, unchanged) hands "
                           "that dead unit [-2.9799 -0.9933] instead — same forward, different "
                           "weights: the one-character bug a random run never shows")
        and predict_lines == [
            "same picture, dropout still ON -> digit [7, 0, 0] confidence [0.481, 0.954, 0.753]",
            "same picture, eval mode        -> digit [0, 0, 0] confidence [0.919, 0.919, 0.919]",
        ])

    if (lesson_numbers_ok and scale_ok and split_ok and dropout_ok and eval_bug_ok
            and biases_learned_ok and inputs_ok and hinge_ok
            and table_lines_ok and rendered_lines_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected MSE 5.3125, cross-entropy 0.11/0.69/4.61 (about 40x the "
              "pain), the bottom-finder to say epoch 3 on the lesson's table and the EARLIER "
              "epoch on a tie, 1/(1-0.5) == 2 with the p = 0.75 rung at x4, "
              "a 64 -> 128 -> 10 model on stamp values spanning (-4.76, 4.36) rather than day "
              "1's [0, 1], "
              "Run A to dive to train 0.047 while validation dips to 1.231 and climbs back to "
              "1.422 (gap 1.375), Run B's gap near 0.742 and narrower at every late epoch, "
              "Run A's learned biases at b1 spread 0.0268 / b2 top 0.0689, "
              "the hinge net's z = 0 unit to receive exactly [0, 0] where a (z >= 0) gate would "
              "hand it [-2.9799, -0.9933], and "
              "dropout-ON predictions to differ while eval-mode ones repeat, and every one of "
              "those lines to read on the page exactly as it reads here — same columns, same "
              "order, and the bottom marker on the epoch that really is the bottom")

    assert lesson_numbers_ok, "MSE 5.3125, cross-entropy 0.11/0.69/4.61, bottom-finder epoch 3"
    assert inputs_ok, ("today's deliberate deviations: a 64-pixel input in PIXELS, HIDDEN, "
                       "CLASSES order, and stamp values outside day 1's [0.0, 1.0]")
    assert scale_ok, "p = 0.5 must scale survivors by 2 so the average signal matches eval mode"
    assert split_ok, "Run A must dive on train while the held-out loss dips early, then climbs"
    assert dropout_ok, "dropout must shrink the gap all along and the late rise, and cost train"
    assert biases_learned_ok, "both bias update lines must run: b1 spread 0.0268, b2 top 0.0689"
    assert eval_bug_ok, "dropout left ON must give different answers; eval mode must repeat"
    assert hinge_ok, ("the ReLU gate must be STRICT: at z = 0 exactly, the dead unit gets [0, 0], "
                      "where a (z >= 0) gate would hand it [-2.9799, -0.9933]")
    assert table_lines_ok, ("both curve tables must print epoch / train / validation in that "
                            "order on their first and last rows, with the bottom marker on exactly "
                            "one row — Run A's epoch 9 and Run B's epoch 14")
    assert rendered_lines_ok, ("every claim-carrying line must reach the screen as written: the "
                               "loss demos, Run A's gap 1.375 and learned biases, the three Run B "
                               "vs Run A contrasts, scaled 60.811 vs "
                               "unscaled 30.405, the live/dead gradient pair, and dropout-ON "
                               "flickering where eval mode repeats")
