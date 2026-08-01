# day-03-minibatch-loop — experiment
#
# Today's big idea in two lines of output:
#   One epoch is not one update. On full MNIST (N = 60,000, batch_size = 32) it is 1,875 of
#   them; on this run's 1,000-example stand-in it is 32. Both counts are printed, kept apart.
#   Shuffle every epoch, average (never sum) each batch, and the training loss falls.
#
# It counts epochs / iterations / batches, slices one epoch (keeping the short last batch),
# checks the batch gradient really IS the average of the per-example ones, trains and watches
# the loss fall, then repeats one epoch on label-sorted data with NO shuffle.
# Run it:  python3 sessions/m04-first-model-mlp/day-03-minibatch-loop/experiment.py

import numpy as np  # arrays, matrix multiply (@) and a seeded random generator

PIXELS, HIDDEN, CLASSES = 784, 128, 10  # the Day 1 MLP: 784 pixels -> 128 hidden -> 10 scores
BATCH_SIZE = 32         # batch_size: how many examples share one weight update
LEARNING_RATE = 0.02    # the step size, applied once per batch update
N_TRAIN, N_EPOCHS = 1000, 6   # a small stand-in set (see below); an epoch is one full pass
# NOT part of He: an extra half-scale on the OUTPUT layer only. He is sqrt(2 / n_in) and nothing
# else; this factor is a deliberate tuning of this demo, and Part 2 prints and pins both the
# spread it produces and the untrained loss it buys, so the deviation is measured, not implied.
W2_EXTRA_SCALE = 0.5


def make_ink_digits(n_total, rng):
    # A DIFFERENT generator from Day 1's make_stand_in_digits, so it gets its own name: n_total
    # is a TOTAL example count (Day 1's argument was per class), the pixels come back as floats
    # already inside [0, 1] (Day 1 returned uint8 bytes 0-255), and this one also returns the
    # one-hot target. MNIST is not on this machine and this script may not use the network, so
    # it is seeded: the shapes the lesson uses, 784 pixels in and 10 classes out. Each class
    # gets a sparse "ink" pattern plus noise, so the task is learnable but not trivial.
    labels = rng.integers(0, CLASSES, size=n_total)
    ink = rng.random((CLASSES, PIXELS)) < 0.10        # about 10% of the pixels lit per class
    images = np.clip(ink[labels] * 0.9 + rng.normal(0.0, 0.40, (n_total, PIXELS)), 0.0, 1.0)
    return images, labels, np.eye(CLASSES)[labels]    # one-hot target: 1 in the true slot

def init_params(rng, w2_scale=W2_EXTRA_SCALE):
    # He initialization from Day 1 on the hidden layer: scale by sqrt(2 / n_in), biases at 0.
    # The output layer gets He AND w2_scale on top — see the W2_EXTRA_SCALE note above. Layout
    # is Day 1's (in, out): W1 is (784, 128) and W2 is (128, 10).
    return [rng.normal(size=(PIXELS, HIDDEN)) * np.sqrt(2 / PIXELS), np.zeros(HIDDEN),
            rng.normal(size=(HIDDEN, CLASSES)) * np.sqrt(2 / HIDDEN) * w2_scale,
            np.zeros(CLASSES)]

def forward(x, params):
    W1, b1, W2, b2 = params                    # Day 1: logits = relu(x @ W1 + b1) @ W2 + b2
    z1 = x @ W1 + b1
    hidden = np.maximum(0, z1)                                        # the ReLU bend
    return z1, hidden, hidden @ W2 + b2        # days 1-2's tuple: (z1, hidden, out)

def loss_and_grads(x, one_hot, params):
    # Day 2's chain rule, over a whole batch — but a DIFFERENT REDUCTION, named here so the two
    # days cannot be mistaken for one. Day 2 used MSE: a mean over every ELEMENT, dividing by
    # out.size. This day uses the lesson's rule instead — the squared error SUMMED over the
    # CLASSES columns to give one loss per example, then MEANED over the batch ROWS. Both are
    # "average, don't add" over the batch; they differ by exactly CLASSES = 10, and Part 3
    # prints and asserts that factor on a real batch.
    W1, b1, W2, b2 = params
    z1, hidden, out = forward(x, params)
    gap = out - one_hot
    loss = float((gap ** 2).sum(axis=1).mean())   # sum over CLASSES, then mean over rows
    batch = x.shape[0]              # read the real size — the ragged last batch is smaller
    d_out = 2.0 * gap / batch       # dividing by ROWS is what turns that sum into a row mean
    d_hidden = (d_out @ W2.T) * (z1 > 0)                # push back through the ReLU gate
    return loss, [x.T @ d_hidden, d_hidden.sum(axis=0), hidden.T @ d_out, d_out.sum(axis=0)]

def one_example_dW1(x_row, one_hot_row, params):
    # The same dW1 for ONE example, written a different way (an outer product instead of a
    # matrix multiply). Part 3 uses it as an independent second opinion on the batch version.
    W1, b1, W2, b2 = params
    z1 = x_row @ W1 + b1
    d_out = 2.0 * ((np.maximum(0, z1) @ W2 + b2) - one_hot_row)
    return np.outer(x_row, (d_out @ W2.T) * (z1 > 0))

def batch_slices(order, batch_size):
    # Cut the index list into chunks of batch_size. When N does not divide evenly the last
    # chunk comes out short: that is the ragged final batch, and we keep it.
    return [order[start:start + batch_size] for start in range(0, len(order), batch_size)]

def run_epochs(x, one_hot, params, n_epochs, shuffle, seed):
    # One iteration = one update on one batch. One epoch = many iterations, one full pass.
    rng = np.random.default_rng(seed)
    params = [p.copy() for p in params]                # never train the caller's weights
    epoch_means, epoch1_losses, orders, batch_log = [], [], [], []
    for _ in range(n_epochs):
        order = rng.permutation(len(x)) if shuffle else np.arange(len(x))   # fresh each epoch
        orders.append(order)
        batch_losses, sizes_used = [], []
        for index in batch_slices(order, BATCH_SIZE):
            loss, grads = loss_and_grads(x[index], one_hot[index], params)
            batch_losses.append(loss)
            sizes_used.append(len(index))    # record the real size of every update we make
            for p, g in zip(params, grads):
                p -= LEARNING_RATE * g                  # forward, loss, backward, update
        epoch_means.append(float(np.mean(batch_losses)))
        batch_log.append(sizes_used)         # one row per epoch, one number per update
        if not epoch1_losses:
            epoch1_losses = batch_losses
    return params, epoch_means, epoch1_losses, orders, batch_log

def full_loss(x, one_hot, params):   # loss over the WHOLE set: one fair number per run
    # Same reduction as loss_and_grads: sum over CLASSES, then mean over rows.
    return float(((forward(x, params)[2] - one_hot) ** 2).sum(axis=1).mean())

def jerkiness(batch_losses):         # how far the loss jumps from one batch to the next
    return float(np.abs(np.diff(batch_losses)).mean())


if __name__ == "__main__":
    # --- Part 1: the vocabulary, counted ----------------------------------
    # Two sets of numbers, kept apart. First the full-MNIST ones the lesson quotes. This
    # machine has no MNIST, so those are a quote, not a measurement of anything here. Then
    # the numbers for THIS run, which trains on the small stand-in set built in Part 2.
    # Both are COUNTED with batch_slices, the same helper the training loop slices with.
    MNIST_N, MNIST_EPOCHS = 60_000, 10        # the lesson's example, not the data below
    mnist_iters = len(batch_slices(np.arange(MNIST_N), BATCH_SIZE))
    mnist_steps = MNIST_EPOCHS * mnist_iters
    print(f"the lesson's figures, quoted from full MNIST (not this run): N = {MNIST_N}"
          f"  batch_size = {BATCH_SIZE}")
    print(f"   -> {mnist_iters} iterations per epoch, so {MNIST_EPOCHS} epochs"
          f" = {mnist_steps} updates")
    planned_batches = len(batch_slices(np.arange(N_TRAIN), BATCH_SIZE))
    print(f"THIS run's own figures, counted the same way: N = {N_TRAIN}"
          f"  batch_size = {BATCH_SIZE}")
    print(f"   -> {planned_batches} iterations per epoch, so {N_EPOCHS} epochs"
          f" = {planned_batches * N_EPOCHS} updates — an epoch is many iterations, not one")
    ragged = [len(s) for s in batch_slices(np.arange(100), BATCH_SIZE)]
    print(f"N = 100 sliced into batches of 32 -> {ragged}  sum = {sum(ragged)}"
          "  (the last batch is ragged: 4, not 32)")

    # --- Part 2: a seeded stand-in for MNIST, and one epoch's batches ------
    rng = np.random.default_rng(0)
    x_train, labels, y_train = make_ink_digits(N_TRAIN, rng)   # N_TRAIN is a TOTAL, not per class
    params0 = init_params(rng)
    pix_lo, pix_hi = float(x_train.min()), float(x_train.max())   # the two edges of the clip
    print(f"\nx_train {x_train.shape}  y_train (one-hot) {y_train.shape}"
          f"  pixel range ({pix_lo}, {pix_hi})")
    # The two spreads, side by side, so the He deviation is a printed number and not a claim in a
    # comment. W1 is plain He. W2 is He x W2_EXTRA_SCALE, which Day 1's own He self-check would
    # reject. What the extra factor buys: a quieter untrained output, so this 6-epoch demo starts
    # near 2.2 instead of 5.7. Undo it (divide W2 back out) and the loop still learns — it just
    # starts from a much louder place, which is why the factor is tuning and not a rule.
    he_std, w2_std = float(params0[0].std()), float(params0[2].std())
    pure_he_params = [params0[0], params0[1], params0[2] / W2_EXTRA_SCALE, params0[3]]
    loss_half_he, loss_pure_he = (full_loss(x_train, y_train, params0),
                                  full_loss(x_train, y_train, pure_he_params))
    print(f"He spread: W1 std {he_std:.4f} = sqrt(2/{PIXELS}) as Day 1 taught  ·  W2 std"
          f" {w2_std:.4f} = {W2_EXTRA_SCALE} x sqrt(2/{HIDDEN}), NOT He")
    print(f"untrained whole-set loss with that half-scale {loss_half_he:.4f}"
          f"  ·  at full He spread {loss_pure_he:.4f}  -> the factor is tuning, not the rule")
    order_demo = np.random.default_rng(1).permutation(N_TRAIN)   # a shuffled index list
    sizes = [len(s) for s in batch_slices(order_demo, BATCH_SIZE)]
    print(f"one epoch over {N_TRAIN} examples -> {len(sizes)} batches: "
          f"{sizes.count(BATCH_SIZE)} of size 32 plus a ragged last one of {sizes[-1]}")

    # --- Part 3: the batch gradient is the AVERAGE, not the sum ------------
    first_batch = batch_slices(np.arange(N_TRAIN), BATCH_SIZE)[0]
    batch_loss, batch_grads = loss_and_grads(x_train[first_batch], y_train[first_batch], params0)
    per_example_sum = np.zeros((PIXELS, HIDDEN))
    for i in first_batch:                              # 32 separate one-example gradients
        per_example_sum += one_example_dW1(x_train[i], y_train[i], params0)
    per_example_mean = per_example_sum / len(first_batch)
    biggest_mean = float(np.abs(batch_grads[0]).max())
    dW1_shape = batch_grads[0].shape                 # named, so the printed shape is checkable
    biggest_single = float(np.abs(per_example_mean).max())   # the second opinion's own number
    biggest_sum = float(np.abs(per_example_sum).max())       # the summed (wrong) step's number
    print(f"\nbatch of {len(first_batch)} -> loss {batch_loss:.4f}"
          f"  dW1 {dW1_shape} (same shape as W1)")
    print(f"largest |dW1|  batch step {biggest_mean:.6f}  ·  32 single steps averaged "
          f"{biggest_single:.6f}  -> the same number")
    print(f"largest |dW1| if we SUMMED the 32 instead: {biggest_sum:.6f}"
          "  -> 32x bigger, so the step would secretly grow with batch_size")
    # Which average, though? Day 2 divided by ELEMENTS (rows x classes); this day sums the
    # CLASSES columns into one loss per example and divides by ROWS. Same squared errors, two
    # named conventions, exactly CLASSES apart — on the loss and on every gradient entry. Both
    # are printed so neither day has to be taken on trust.
    gap0 = forward(x_train[first_batch], params0)[2] - y_train[first_batch]
    elements_loss = float((gap0 ** 2).mean())            # day 2's spelling on the SAME batch
    elements_dW1 = per_example_mean / CLASSES            # and the gradient it would have seeded
    print(f"reduction: sum over {CLASSES} classes then mean over {len(first_batch)} rows"
          f" -> {batch_loss:.4f}; day 2's mean over all {gap0.size} elements -> "
          f"{elements_loss:.4f}  ({CLASSES}x apart, largest |dW1| "
          f"{float(np.abs(elements_dW1).max()):.6f} vs {biggest_mean:.6f})")

    # --- Part 4: train, and watch the loss fall ---------------------------
    trained, epoch_means, epoch1_losses, orders, batch_log = run_epochs(
        x_train, y_train, params0, N_EPOCHS, True, 1)
    updates_done = sum(len(sizes) for sizes in batch_log)   # counted inside the real loop
    epoch1_last = batch_log[0][-1]        # the ragged batch the loop really stepped on last
    print(f"\nthe loop itself counted {len(batch_log)} epochs x {len(batch_log[0])} batches"
          f" = {updates_done} weight updates (epoch 1's last batch held {epoch1_last})")
    for epoch, mean_loss in enumerate(epoch_means, start=1):
        print(f"epoch {epoch}  average training loss {mean_loss:.4f}")
    went_up = int(np.sum(np.diff(epoch1_losses) > 0))
    print(f"inside epoch 1 the per-batch loss went UP {went_up} of {len(epoch1_losses) - 1} times"
          f" (bumpy), first batch {epoch1_losses[0]:.4f} -> last {epoch1_losses[-1]:.4f}")
    print(f"epoch 1 and 2 use different orders: {not np.array_equal(orders[0], orders[1])}"
          " — a fresh shuffle every epoch")

    # --- Part 5: one epoch on label-sorted data, with NO shuffle ----------
    sorted_index = np.argsort(labels, kind="stable")    # all the 0s, then all the 1s, ...
    p_sorted, _, sorted_losses, _, _ = run_epochs(
        x_train[sorted_index], y_train[sorted_index], params0, 1, False, 1)
    p_shuffled, _, shuffled_losses, _, _ = run_epochs(x_train, y_train, params0, 1, True, 1)
    jerk_sorted, jerk_shuffled = jerkiness(sorted_losses), jerkiness(shuffled_losses)
    loss_sorted, loss_shuffled = (full_loss(x_train, y_train, p_sorted),
                                  full_loss(x_train, y_train, p_shuffled))
    loss_before = full_loss(x_train, y_train, params0)   # the untrained control both are read against
    print(f"\naverage jump between batches  shuffled {jerk_shuffled:.4f}"
          f"   label-sorted, no shuffle {jerk_sorted:.4f}")
    print(f"whole-set loss after 1 epoch  before training "
          f"{loss_before:.4f}  shuffled {loss_shuffled:.4f}"
          f"  sorted {loss_sorted:.4f}")
    print("-> same weights, same 32 updates: only the ORDER changed, and sorted learns worse")

    # --- Part 6: the honest reminder --------------------------------------
    print("\nreminder: this is TRAINING loss, so a falling curve only proves the LOOP works."
          "\nwhether the model is any good needs held-out data it never trained on (Day 4).")

    # --- Self-check: every claim pinned to a number written down here -----
    # Everything here is seeded, so the numbers below are the ones this file really printed:
    # each one is compared exactly, after rounding, instead of "close enough". The update
    # counts come from the loop's own record of every batch it stepped on, not from arithmetic
    # on 60,000. The batch layout is checked as an exact list of sizes and positions, not only
    # as "the same examples somewhere", and the batch gradient against Part 3's 32 separate
    # gradients.
    checks = [
        ("this run really made 6 epochs x 32 batches = 192 updates, as Part 1 said",
         [len(sizes) for sizes in batch_log] == [32] * 6 and updates_done == 192
         and updates_done == planned_batches * N_EPOCHS and len(x_train) == N_TRAIN
         and batch_log[0] == [32] * 31 + [8] and epoch1_last == 8),
        ("slicing full MNIST the same way gives 1875 iterations per epoch, 18750 over 10",
         mnist_iters == 1875 and mnist_steps == 18750),
        ("N=100 slices into [32, 32, 32, 4]", ragged == [32, 32, 32, 4] and sum(ragged) == 100),
        ("the stand-in pixels stay inside the clip, 0.0 to 1.0, in 784 columns and 10 classes",
         pix_lo == 0.0 and pix_hi == 1.0 and x_train.shape == (N_TRAIN, PIXELS)
         and y_train.shape == (N_TRAIN, CLASSES)),
        ("one epoch = 31 full batches then a ragged 8, each index exactly once",
         sizes == [32] * 31 + [8] and sorted(order_demo.tolist()) == list(range(N_TRAIN))
         and np.array_equal(np.concatenate(batch_slices(order_demo, BATCH_SIZE)), order_demo)),
        ("shuffling changes the order, and each epoch reshuffles",
         not np.array_equal(orders[0], np.arange(N_TRAIN))
         and not np.array_equal(orders[0], orders[1])),
        ("batch dW1 == mean of the 32 one-example dW1, largest 0.148989, batch loss 2.359",
         np.array_equal(np.round(batch_grads[0], 12), np.round(per_example_mean, 12))
         and round(biggest_mean, 6) == 0.148989 and round(batch_loss, 4) == 2.359
         and dW1_shape == (PIXELS, HIDDEN) and round(biggest_single, 6) == 0.148989),
        ("SUMMING the 32 instead of averaging is exactly 32x bigger, 4.767646",
         round(biggest_sum, 6) == 4.767646 and biggest_sum / biggest_single == 32.0),
        (f"this day's reduction is a per-example SUM over {CLASSES} classes then a mean over "
         "rows, which is exactly 10x day 2's mean over elements: 2.3590 vs 0.2359",
         abs(batch_loss - CLASSES * elements_loss) < 1e-12
         and round(batch_loss, 4) == 2.3590 and round(elements_loss, 4) == 0.2359
         and np.allclose(batch_grads[0], CLASSES * elements_dW1, rtol=1e-9, atol=1e-12)),
        ("W1 is plain He (std 0.0506) while W2 is deliberately HALF He (0.0637, not 0.1274), "
         "which costs an untrained loss of 2.2460 instead of 5.6774",
         round(he_std, 4) == 0.0506 and round(w2_std, 4) == 0.0637
         and abs(w2_std - W2_EXTRA_SCALE * float(pure_he_params[2].std())) < 1e-12
         and abs(float(np.sqrt(2 / HIDDEN)) - 0.125) < 1e-3
         and round(loss_half_he, 4) == 2.2460 and round(loss_pure_he, 4) == 5.6774),
        ("the average loss falls every epoch, 0.7296 -> 0.1187",
         all(epoch_means[i + 1] < epoch_means[i] for i in range(N_EPOCHS - 1))
         and round(epoch_means[0], 4) == 0.7296 and round(epoch_means[-1], 4) == 0.1187),
        ("inside epoch 1 the per-batch loss rises 10 of 31 times, 2.3596 -> 0.3884",
         went_up == 10 and round(epoch1_losses[0], 4) == 2.3596
         and round(epoch1_losses[-1], 4) == 0.3884),
        ("label-sorted with no shuffle is jerkier, 0.7219 vs 0.0907",
         jerk_sorted > jerk_shuffled and round(jerk_sorted, 4) == 0.7219
         and round(jerk_shuffled, 4) == 0.0907),
        ("untrained, the whole set costs 2.2460, and BOTH 1-epoch runs come in under it",
         round(loss_before, 4) == 2.2460 and loss_before > loss_sorted > loss_shuffled),
        ("label-sorted with no shuffle ends worse, 1.0552 vs 0.3609",
         loss_sorted > loss_shuffled and round(loss_sorted, 4) == 1.0552
         and round(loss_shuffled, 4) == 0.3609),
    ]
    missed = [claim for claim, ok in checks if not ok]
    print("\n✅ you got it" if not missed else "\n❌ not yet — expected " + "; ".join(missed))
    for claim, ok in checks:      # these asserts stop the program if a claim is wrong
        assert ok, claim
