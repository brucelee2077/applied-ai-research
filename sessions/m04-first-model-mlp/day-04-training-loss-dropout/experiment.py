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

def dropout_mask(dice, shape, p):
    # Bench each unit with probability p — so KEEP it with probability (1-p) — and give every
    # survivor the 1/(1-p) boost. A benched unit gets 0, a survivor gets 1/(1-p).
    return (dice.random(shape) > p) * keep_scale(p)

def forward_eval(x, W1, b1, W2, b2, dice=None, drop_p=0.0):
    # THE eval path, used in three places: the per-epoch validation score inside the loop, the
    # signal ladder in Part 5, and Part 6's "steady" predictions. Dropout stays off unless a
    # caller hands over dice, so if dropout ever leaked into this path Part 6's three repeats
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
            d_hidden = (d_out @ W2.T) * mask * (z1 > 0)        # benched units get no gradient
            W2 -= lr * (hidden.T @ d_out); b2 -= lr * d_out.sum(axis=0)
            W1 -= lr * (x.T @ d_hidden);   b1 -= lr * d_hidden.sum(axis=0)
        # --- eval mode: dropout OFF, every unit kept, no scaling ---
        _, val_probs = forward_eval(val_x, W1, b1, W2, b2)
        history.append((float(np.mean(batch_losses)), cross_entropy(val_probs, val_y)))
    return history, (W1, b1, W2, b2)

def validation_bottom(val_losses):
    # The early-stopping point: the 1-based epoch whose held-out loss is the lowest.
    return int(np.argmin(val_losses)) + 1

def show_curve(history, every=4):
    # Print epoch / train loss / validation loss, and mark the validation bottom.
    bottom = validation_bottom([v for _, v in history]) - 1
    for i, (t, v) in enumerate(history):
        if i % every == 0 or i == bottom or i == len(history) - 1:
            print("  epoch %2d   train %.4f   validation %.4f%s" % (i + 1, t, v,
                  "  <-- validation bottom (best model)" if i == bottom else ""))
    return bottom + 1, history[-1][0], history[-1][1], min(v for _, v in history)


if __name__ == "__main__":
    # --- Part 1: the two loss flavors, on the lesson's own numbers ---------
    gaps = [0.5, 1.0, 2.0, 4.0]
    print("MSE  gaps", gaps, "-> squared", [g * g for g in gaps], "-> MSE", round(mse(gaps), 4))
    demo_probs = (0.90, 0.50, 0.01)          # confident-and-right, hedging, confident-and-wrong
    # Score each row through the SAME cross_entropy the training loop uses: two columns, the
    # true class is column 0, so the loss is -log(the probability given to the true class).
    demo_losses = [round(cross_entropy(np.array([[p, 1 - p]]), np.array([0])), 2)
                   for p in demo_probs]
    for p_true, loss in zip(demo_probs, demo_losses):
        print(f"cross-entropy, {p_true:.2f} on the true class -> loss {loss:.2f}")
    pain_ratio = demo_losses[2] / demo_losses[0]        # 4.61 / 0.11 — how much worse it hurts
    pain_shown = round(pain_ratio / 10) * 10            # to the nearest ten, as the lesson quotes
    print(f"-> confident-and-wrong (0.01) hurts about {pain_shown}x confident-and-right (0.90)"
          f"  ({demo_losses[2]:.2f} / {demo_losses[0]:.2f} = {pain_ratio:.1f})")

    # --- Part 2: split the data into train and held-out validation ---------
    rng = np.random.default_rng(0)
    prototypes = 0.5 * rng.normal(0, 1, (CLASSES, PIXELS))     # one stamp per digit
    train_x, train_y = make_stamps(prototypes, 30, 0.4, rng)   # small slice, 40% wrong labels
    val_x, val_y = make_stamps(prototypes, 50, 0.0, rng)       # never trained on, clean labels
    # Day 1 counted 101770 "knobs" = weights PLUS both bias vectors. This count is WEIGHTS only
    # (the biases would add HIDDEN + CLASSES = 138); it is the number the over-powered argument
    # rests on, so the two days are quoting two different quantities under similar words.
    weights = PIXELS * HIDDEN + HIDDEN * CLASSES
    print(f"\ntrain images {train_x.shape} labels {train_y.shape}"
          f" | held-out val {val_x.shape} labels {val_y.shape}")
    print(f"model {PIXELS} -> {HIDDEN} -> {CLASSES} = {weights} weights for {len(train_y)}"
          " training examples (over-powered on purpose)")
    # The pixel-range deviation, measured. Day 1's rule was bytes scaled into exactly [0.0, 1.0];
    # these stamp values are prototype + N(0, 1), so they run negative and are not bounded by 1.
    stamp_lo, stamp_hi = float(train_x.min()), float(train_x.max())
    print(f"stamp values run ({stamp_lo:.2f}, {stamp_hi:.2f}) — NOT day 1's [0.0, 1.0] byte"
          " pixels: these are centred near 0 with spread ~1, the other standard input scaling")
    # Three predictions, written down BEFORE the runs and tested in the self-check below.
    print("predict: (1) train loss -> near 0 while validation turns UP; (2) the rising curve is"
          " VALIDATION; (3) dropout makes the gap SMALLER")

    # --- Part 3: Run A — no dropout, watch the two curves split ------------
    # Warm-up: hand the lesson's own six-epoch table to the same bottom-finder that is about to
    # mark the real run, and see that it lands where the lesson read the bottom by eye.
    lesson_val = [0.62, 0.44, 0.35, 0.36, 0.42, 0.51]             # the lesson's 6-epoch table
    print(f"\nlesson's table {lesson_val} -> the bottom-finder says epoch"
          f" {validation_bottom(lesson_val)} (the lesson read epoch 3, loss {min(lesson_val)})")
    print("\n--- Run A: NO dropout ---")
    hist_a, params_a = train_run(train_x, train_y, val_x, val_y, drop_p=0.0)
    best_a, train_a, val_a, best_val_a = show_curve(hist_a)
    rise_a, gap_a = val_a - best_val_a, val_a - train_a
    print(f"  validation was lowest at epoch {best_a}, then rose by {rise_a:.3f} while train dived")
    print(f"  final gap (validation - train) = {gap_a:.3f} <- this growing gap IS overfitting")
    # The two bias rows started as exact zeros, so whatever they hold now was learned by the two
    # bias update lines in the loop. Both numbers are pinned tightly in the self-check.
    b1_a, b2_a = params_a[1], params_a[3]
    print(f"  biases started at 0 and learned: b1 spread {b1_a.std():.4f}, b2 top {b2_a.max():.4f}")

    # --- Part 4: Run B — the same run, now with dropout p = 0.5 ------------
    print(f"\n--- Run B: dropout p = {DROP_P} (train mode only, survivors x{keep_scale(DROP_P)}) ---")
    hist_b, params_b = train_run(train_x, train_y, val_x, val_y, drop_p=DROP_P)
    best_b, train_b, val_b, best_val_b = show_curve(hist_b)
    rise_b, gap_b = val_b - best_val_b, val_b - train_b
    print(f"  best validation {best_val_b:.3f} at epoch {best_b} (Run A reached {best_val_a:.3f})")
    print(f"  final gap {gap_b:.3f} vs Run A's {gap_a:.3f}; late rise {rise_b:.3f} vs {rise_a:.3f}")
    print(f"  not free: after the same {EPOCHS} epochs its train loss is {train_b:.3f},"
          f" not {train_a:.3f}")

    # --- Part 5: why the 1/(1-p) scaling is there --------------------------
    kept_total = 2 * 1.0 * keep_scale(DROP_P)    # lesson's ladder: 4 units worth 1, keep 2, x2
    print(f"\n4 units kept as-is -> total {4 * 1.0} | 2 benched, survivors"
          f" x{keep_scale(DROP_P)} -> total {kept_total}")
    W1, b1, W2, b2 = params_b
    hidden_eval, _ = forward_eval(val_x[:1], W1, b1, W2, b2)   # eval mode: all units kept
    dice = np.random.default_rng(11)
    masks = dropout_mask(dice, (500,) + hidden_eval.shape, DROP_P)   # 500 random benchings
    eval_signal = float(hidden_eval.sum())
    scaled_mean = float((hidden_eval * masks).sum(axis=(1, 2)).mean())
    unscaled_mean = float((hidden_eval * (masks > 0)).sum(axis=(1, 2)).mean())
    print(f"eval signal {eval_signal:.3f} | dropout averages {scaled_mean:.3f} (scaled)"
          f" vs {unscaled_mean:.3f} (if you forget to scale)")

    # --- Part 6: the silent bug — dropout left ON at prediction time -------
    # Both sets of rows go through the SAME forward_eval the loop scored validation with; the
    # ONLY difference is whether dice and a drop rate are handed in. So "eval mode repeats" is a
    # claim about that code path, not about re-reading one expression three times.
    steady_rows = [forward_eval(val_x[:1], W1, b1, W2, b2)[1][0] for _ in range(3)]   # eval mode
    dice = np.random.default_rng(7)
    flicker_rows = [forward_eval(val_x[:1], W1, b1, W2, b2, dice, DROP_P)[1][0]       # oops: ON
                    for _ in range(3)]
    print("\nsame picture, dropout still ON -> digit", [int(r.argmax()) for r in flicker_rows],
          "confidence", [round(float(r.max()), 3) for r in flicker_rows])
    print("same picture, eval mode        -> digit", [int(r.argmax()) for r in steady_rows],
          "confidence", [round(float(r.max()), 3) for r in steady_rows])
    print("\nremember: a low TRAINING loss proves nothing about new data — the held-out loss is")
    print("the honest number. And dropout is not free: it learns slower, and it can hurt a model")
    print("that already generalizes. Reach for it when you SEE the gap grow.")

    # --- Self-check: one boolean per claim ---------------------------------
    # Every expected value below is written down here — the lesson's demo numbers, and the
    # numbers this script printed when it was written — never re-derived from the code above.
    # The shape claims read the recorded history of the two runs, so a model that learned
    # nothing (a flat curve) fails them instead of agreeing with itself.
    lesson_numbers_ok = (
        abs(mse(gaps) - 5.3125) < 1e-9                            # lesson demo: MSE 5.31
        and demo_losses == [0.11, 0.69, 4.61]                     # lesson: 0.11 / 0.69 / 4.61
        and pain_shown == 40                                      # the "about 40x" Part 1 printed
        # The bottom-finder that marked BOTH runs above, on three curves whose answer is known
        # by hand: the lesson's table (bottom at epoch 3), a curve still falling at the end
        # (bottom = the last epoch), and one rising from the start (bottom = the first epoch).
        and validation_bottom(lesson_val) == 3
        and validation_bottom([0.9, 0.8, 0.7]) == 3
        and validation_bottom([0.7, 0.8, 0.9]) == 1)
    # The two places this day leaves the module's spine, pinned so neither can pass as an
    # accident: a 64-pixel input (days 1/2/3/5 use 784) with the tuple still in PIXELS, HIDDEN,
    # CLASSES order, and stamp values that really do leave day 1's [0.0, 1.0] band on BOTH sides.
    inputs_ok = ((PIXELS, HIDDEN, CLASSES) == (64, 128, 10)
                 and abs(stamp_lo + 4.76) < 0.02 and abs(stamp_hi - 4.36) < 0.02
                 and stamp_lo < 0.0 < 1.0 < stamp_hi)
    # A wrong scale factor shows up here: scaled must match eval, unscaled must be half of it.
    # A mask that KEPT with probability p would look fine at p = 0.5, so probe an uneven rate
    # too: at p = 0.75 only about a quarter of the units may survive, each boosted by 4.
    probe = dropout_mask(np.random.default_rng(5), (4000,), 0.75)
    scale_ok = (abs(keep_scale(0.5) - 2.0) < 1e-12 and abs(kept_total - 4.0) < 1e-12
                and abs(scaled_mean - eval_signal) / eval_signal < 0.02
                and abs(unscaled_mean / eval_signal - 0.5) < 0.02
                and abs(float((probe > 0).mean()) - 0.25) < 0.02
                and abs(float(probe.max()) - 4.0) < 1e-12)
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
                and 2 <= best_a <= 30 and rise_a > 0.10
                and abs(train_a - 0.047) < 0.05
                and abs(best_val_a - 1.231) < 0.05 and abs(val_a - 1.422) < 0.05)
    # Prediction 3: dropout halves the gap (1.375 -> 0.742), reaches a better best, rises less,
    # and costs something — its train loss is still high after the same EPOCHS epochs. The gap is
    # narrower at every one of the last ten epochs, not only on the epoch that gets printed.
    gap_smaller_all_along = all(hist_b[i][1] - hist_b[i][0] < hist_a[i][1] - hist_a[i][0]
                                for i in range(EPOCHS - 10, EPOCHS))
    dropout_ok = (abs(gap_a - 1.375) < 0.05 and abs(gap_b - 0.742) < 0.05
                  and gap_smaller_all_along
                  and best_val_b < best_val_a and rise_b < rise_a / 2 and train_b > 0.20)
    # The two bias update lines: b1 and b2 start as exact zeros, so these two pins fail flat (at
    # 0.0000) if either line stops running, and shift far past 1e-4 if either one uses the wrong
    # gradient or step. The run is fully seeded, so both numbers are pinned to what it printed.
    biases_learned_ok = (abs(float(b1_a.std()) - 0.026786) < 1e-4
                         and abs(float(b2_a.max()) - 0.068894) < 1e-4)
    # The eval-mode bug: three dropout-ON passes all disagree, three eval passes are identical,
    # and leaving dropout on flipped the predicted digit on at least one pass. Both sets came out
    # of forward_eval, so dropout leaking into that path breaks the "identical" clause.
    steady, top = steady_rows[0], int(steady_rows[0].argmax())
    eval_bug_ok = (len({r.tobytes() for r in flicker_rows}) == 3
                   and len({r.tobytes() for r in steady_rows}) == 1
                   and any(int(r.argmax()) != top for r in flicker_rows)
                   and max(float(steady.max() - r[top]) for r in flicker_rows) > 0.30)

    if (lesson_numbers_ok and scale_ok and split_ok and dropout_ok and eval_bug_ok
            and biases_learned_ok and inputs_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected MSE 5.3125, cross-entropy 0.11/0.69/4.61 (about 40x the "
              "pain), the bottom-finder to say epoch 3 on the lesson's table, 1/(1-0.5) == 2, "
              "a 64 -> 128 -> 10 model on stamp values spanning (-4.76, 4.36) rather than day "
              "1's [0, 1], "
              "Run A to dive to train 0.047 while validation dips to 1.231 and climbs back to "
              "1.422 (gap 1.375), Run B's gap near 0.742 and narrower at every late epoch, "
              "Run A's learned biases at b1 spread 0.0268 / b2 top 0.0689, and "
              "dropout-ON predictions to differ while eval-mode ones repeat")

    assert lesson_numbers_ok, "MSE 5.3125, cross-entropy 0.11/0.69/4.61, bottom-finder epoch 3"
    assert inputs_ok, ("today's deliberate deviations: a 64-pixel input in PIXELS, HIDDEN, "
                       "CLASSES order, and stamp values outside day 1's [0.0, 1.0]")
    assert scale_ok, "p = 0.5 must scale survivors by 2 so the average signal matches eval mode"
    assert split_ok, "Run A must dive on train while the held-out loss dips early, then climbs"
    assert dropout_ok, "dropout must shrink the gap all along and the late rise, and cost train"
    assert biases_learned_ok, "both bias update lines must run: b1 spread 0.0268, b2 top 0.0689"
    assert eval_bug_ok, "dropout left ON must give different answers; eval mode must repeat"
