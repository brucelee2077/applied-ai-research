# day-04-training-loss-dropout — experiment
#
# Today's big idea in two lines of output:
#   With NO dropout the training loss dives to 0.05 while the held-out loss bottoms out at
#   epoch 9 and then turns back UP — the two curves split, and that split is overfitting.
#   Switch dropout on (p = 0.5, survivors scaled by 1/(1-p)) and the gap shrinks by half.
#
# Run it:  python3 sessions/m04-first-model-mlp/day-04-training-loss-dropout/experiment.py

import numpy as np  # arrays, matrix multiply (@), and seeded random numbers

PIXELS, CLASSES, HIDDEN = 64, 10, 128   # 8x8 "pixels" -> 128 hidden units -> 10 digits
DROP_P = 0.5                            # the lesson's dropout rate: bench half the units
EPOCHS = 40                             # full passes through the training data, both runs

def mse(gaps):
    # Mean squared error: square every gap, then average. Big misses count for much more.
    return float(np.mean(np.square(gaps)))

def cross_entropy(probs, labels):
    # Cross-entropy: -log(the probability the model gave the TRUE class), averaged.
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

# No dataset is cached on this machine, so we synthesize 8x8 = 64-pixel "digit stamps": each
# class gets one fixed prototype, and every example is that prototype plus noise. A fraction
# of the TRAINING labels is then made wrong on purpose — that wrong-label noise is what the
# lesson says an over-powered model memorizes. The validation labels are all left correct.
def make_stamps(rng, prototypes, per_class, wrong_fraction):
    labels = np.repeat(np.arange(CLASSES), per_class)
    images = prototypes[labels] + rng.normal(0, 1.0, (len(labels), PIXELS))
    spoiled = rng.permutation(len(labels))[:int(round(wrong_fraction * len(labels)))]
    labels = labels.copy()
    labels[spoiled] = (labels[spoiled] + rng.integers(1, CLASSES, size=len(spoiled))) % CLASSES
    return images, labels

def train_run(train_x, train_y, val_x, val_y, drop_p, epochs=EPOCHS, lr=0.05, batch=16):
    """Day 3's mini-batch loop, now recording BOTH average losses every epoch."""
    start = np.random.default_rng(1)                 # both runs get the same starting weights
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
            pre = x @ W1 + b1
            hidden = np.maximum(0, pre)                       # ReLU
            # mask 1.0 means "keep everything, scale nothing" — that is the no-dropout run.
            mask = dropout_mask(dice, hidden.shape, drop_p) if drop_p else 1.0
            hidden = hidden * mask
            probs = softmax_rows(hidden @ W2 + b2)
            batch_losses.append(cross_entropy(probs, y))
            d_scores = probs.copy()                           # gradient of softmax + x-entropy
            d_scores[np.arange(len(y)), y] -= 1
            d_scores /= len(y)
            d_pre = (d_scores @ W2.T) * mask * (pre > 0)       # benched units get no gradient
            W2 -= lr * (hidden.T @ d_scores); b2 -= lr * d_scores.sum(axis=0)
            W1 -= lr * (x.T @ d_pre);         b1 -= lr * d_pre.sum(axis=0)
        # --- eval mode: dropout OFF, every unit kept, no scaling ---
        val_probs = softmax_rows(np.maximum(0, val_x @ W1 + b1) @ W2 + b2)
        history.append((float(np.mean(batch_losses)), cross_entropy(val_probs, val_y)))
    return history, (W1, b1, W2, b2)

def show_curve(history, every=4):
    # Print epoch / train loss / validation loss, and mark the validation bottom.
    bottom = int(np.argmin([v for _, v in history]))
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
    demo_losses = [round(-float(np.log(p)), 2) for p in demo_probs]
    for p_true, loss in zip(demo_probs, demo_losses):
        print(f"cross-entropy, {p_true:.2f} on the true class -> loss {loss:.2f}")
    hurt = round(np.log(demo_probs[2]) / np.log(demo_probs[0]))
    print(f"-> confident-and-wrong (0.01) hurts about {hurt}x confident-and-right (0.90)")

    # --- Part 2: split the data into train and held-out validation ---------
    rng = np.random.default_rng(0)
    prototypes = 0.5 * rng.normal(0, 1, (CLASSES, PIXELS))     # one stamp per digit
    train_x, train_y = make_stamps(rng, prototypes, 30, 0.4)   # small slice, 40% wrong labels
    val_x, val_y = make_stamps(rng, prototypes, 50, 0.0)       # never trained on, clean labels
    weights = PIXELS * HIDDEN + HIDDEN * CLASSES
    print(f"\ntrain images {train_x.shape} labels {train_y.shape}"
          f" | held-out val {val_x.shape} labels {val_y.shape}")
    print(f"model {PIXELS} -> {HIDDEN} -> {CLASSES} = {weights} weights for {len(train_y)}"
          " training examples (over-powered on purpose)")
    # Three predictions, written down BEFORE the runs and tested in the self-check below.
    print("predict: (1) train loss -> near 0 while validation turns UP; (2) the rising curve is"
          " VALIDATION; (3) dropout makes the gap SMALLER")

    # --- Part 3: Run A — no dropout, watch the two curves split ------------
    print("\n--- Run A: NO dropout ---")
    hist_a, _ = train_run(train_x, train_y, val_x, val_y, drop_p=0.0)
    best_a, train_a, val_a, best_val_a = show_curve(hist_a)
    rise_a, gap_a = val_a - best_val_a, val_a - train_a
    print(f"  validation was lowest at epoch {best_a}, then rose by {rise_a:.3f} while train dived")
    print(f"  final gap (validation - train) = {gap_a:.3f} <- this growing gap IS overfitting")

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
    hidden_eval = np.maximum(0, val_x[:1] @ W1 + b1)           # eval mode: all units kept
    dice = np.random.default_rng(11)
    masks = dropout_mask(dice, (500,) + hidden_eval.shape, DROP_P)   # 500 random benchings
    eval_signal = float(hidden_eval.sum())
    scaled_mean = float((hidden_eval * masks).sum(axis=(1, 2)).mean())
    unscaled_mean = float((hidden_eval * (masks > 0)).sum(axis=(1, 2)).mean())
    print(f"eval signal {eval_signal:.3f} | dropout averages {scaled_mean:.3f} (scaled)"
          f" vs {unscaled_mean:.3f} (if you forget to scale)")

    # --- Part 6: the silent bug — dropout left ON at prediction time -------
    steady_rows = [softmax_rows(hidden_eval @ W2 + b2)[0] for _ in range(3)]   # eval mode
    dice = np.random.default_rng(7)
    flicker_rows = [softmax_rows((hidden_eval * m) @ W2 + b2)[0]               # train mode, oops
                    for m in dropout_mask(dice, (3,) + hidden_eval.shape, DROP_P)]
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
    lesson_val = [0.62, 0.44, 0.35, 0.36, 0.42, 0.51]             # the lesson's 6-epoch table
    lesson_numbers_ok = (
        abs(mse(gaps) - 5.3125) < 1e-9                            # lesson demo: MSE 5.31
        and demo_losses == [0.11, 0.69, 4.61]                     # lesson: 0.11 / 0.69 / 4.61
        and hurt >= 40                                            # lesson: "about 40x the pain"
        and int(np.argmin(lesson_val)) + 1 == 3                   # lesson: bottom at epoch 3
        and abs(min(lesson_val) - 0.35) < 1e-9                    # lesson: the bottom is 0.35
        and abs(lesson_val[-1] - min(lesson_val) - 0.16) < 1e-9   # lesson: 0.35 climbs to 0.51
        and all(lesson_val[i] > lesson_val[i - 1] for i in (3, 4, 5)))   # rises after the bottom
    # A wrong scale factor shows up here: scaled must match eval, unscaled must be half of it.
    # A mask that KEPT with probability p would look fine at p = 0.5, so probe an uneven rate
    # too: at p = 0.75 only about a quarter of the units may survive, each boosted by 4.
    probe = dropout_mask(np.random.default_rng(5), (4000,), 0.75)
    scale_ok = (abs(keep_scale(0.5) - 2.0) < 1e-12 and abs(kept_total - 4.0) < 1e-12
                and abs(scaled_mean - eval_signal) / eval_signal < 0.02
                and abs(unscaled_mean / eval_signal - 0.5) < 0.02
                and abs(float((probe > 0).mean()) - 0.25) < 0.02
                and abs(float(probe.max()) - 4.0) < 1e-12)
    # Predictions 1 and 2, read off Run A's own recorded history: TRAIN is the curve that keeps
    # falling (every one of the last ten epochs), VALIDATION is the one that ends above its
    # bottom — and the reported bottom epoch must really be where that lowest value sits.
    train_keeps_falling = all(hist_a[i][0] < hist_a[i - 1][0] for i in range(EPOCHS - 10, EPOCHS))
    bottom_is_the_bottom = hist_a[best_a - 1][1] == best_val_a and 2 <= best_a <= 30
    split_ok = (train_keeps_falling and bottom_is_the_bottom
                and train_a < 0.10 and abs(train_a - 0.047) < 0.05 and rise_a > 0.10
                and abs(best_val_a - 1.231) < 0.05 and abs(val_a - 1.422) < 0.05)
    # Prediction 3: dropout halves the gap (1.375 -> 0.742), reaches a better best, rises less,
    # and costs something — its train loss is still high after the same EPOCHS epochs.
    dropout_ok = (abs(gap_a - 1.375) < 0.05 and abs(gap_b - 0.742) < 0.05 and gap_b < gap_a
                  and best_val_b < best_val_a and rise_b < rise_a / 2 and train_b > 0.20)
    # The eval-mode bug: three dropout-ON passes all disagree, three eval passes are identical,
    # and leaving dropout on flipped the predicted digit on at least one pass.
    steady, top = steady_rows[0], int(steady_rows[0].argmax())
    eval_bug_ok = (len({r.tobytes() for r in flicker_rows}) == 3
                   and len({r.tobytes() for r in steady_rows}) == 1
                   and any(int(r.argmax()) != top for r in flicker_rows)
                   and max(float(steady.max() - r[top]) for r in flicker_rows) > 0.30)

    if lesson_numbers_ok and scale_ok and split_ok and dropout_ok and eval_bug_ok:
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected MSE 5.3125, cross-entropy 0.11/0.69/4.61, the lesson's "
              "table to bottom at epoch 3, 1/(1-0.5) == 2, Run A to end near train 0.047 and "
              "validation 1.422 (gap 1.375) with its bottom before the last epoch, Run B's gap "
              "near 0.742, and dropout-ON predictions to differ while eval-mode ones repeat")

    assert lesson_numbers_ok, "MSE 5.3125, cross-entropy 0.11/0.69/4.61, lesson bottom epoch 3"
    assert scale_ok, "p = 0.5 must scale survivors by 2 so the average signal matches eval mode"
    assert split_ok, "Run A must dive on train while the held-out loss bottoms early, then rises"
    assert dropout_ok, "dropout must shrink the gap and the late rise, and cost some train loss"
    assert eval_bug_ok, "dropout left ON must give different answers; eval mode must repeat"
