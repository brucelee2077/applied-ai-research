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


def make_stand_in_digits(n, rng):
    # MNIST is not on this machine and this script may not use the network, so we build a
    # seeded stand-in with the shapes the lesson uses: 784 pixels in, 10 classes out. Each
    # class gets a sparse "ink" pattern plus noise, so the task is learnable but not trivial.
    labels = rng.integers(0, CLASSES, size=n)
    ink = rng.random((CLASSES, PIXELS)) < 0.10        # about 10% of the pixels lit per class
    images = np.clip(ink[labels] * 0.9 + rng.normal(0.0, 0.40, (n, PIXELS)), 0.0, 1.0)
    return images, labels, np.eye(CLASSES)[labels]    # one-hot target: 1 in the true slot

def init_params(rng):
    # He initialisation from Day 1: scale each weight by sqrt(2 / n_in), biases start at 0.
    return [rng.normal(size=(PIXELS, HIDDEN)) * np.sqrt(2 / PIXELS), np.zeros(HIDDEN),
            rng.normal(size=(HIDDEN, CLASSES)) * np.sqrt(2 / HIDDEN) * 0.5, np.zeros(CLASSES)]

def forward(x, params):
    W1, b1, W2, b2 = params                    # Day 1: logits = relu(x @ W1 + b1) @ W2 + b2
    z1 = x @ W1 + b1
    hidden = np.maximum(0.0, z1)                                      # the ReLU bend
    return z1, hidden, hidden @ W2 + b2

def loss_and_grads(x, one_hot, params):
    # Day 2's backward pass, over a whole batch. The loss is the squared error per example,
    # AVERAGED (mean, not sum) over the batch, so the step does not grow with batch_size.
    W1, b1, W2, b2 = params
    z1, hidden, out = forward(x, params)
    gap = out - one_hot
    loss = float((gap ** 2).sum(axis=1).mean())
    batch = x.shape[0]              # read the real size — the ragged last batch is smaller
    d_out = 2.0 * gap / batch       # dividing by batch is what turns the sum into a mean
    d_hidden = (d_out @ W2.T) * (z1 > 0)                # push back through the ReLU gate
    return loss, [x.T @ d_hidden, d_hidden.sum(0), hidden.T @ d_out, d_out.sum(0)]

def one_example_dW1(x_row, one_hot_row, params):
    # The same dW1 for ONE example, written a different way (an outer product instead of a
    # matrix multiply). Part 3 uses it as an independent second opinion on the batch version.
    W1, b1, W2, b2 = params
    z1 = x_row @ W1 + b1
    d_out = 2.0 * ((np.maximum(0.0, z1) @ W2 + b2) - one_hot_row)
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
    x_train, labels, y_train = make_stand_in_digits(N_TRAIN, rng)
    params0 = init_params(rng)
    print(f"\nx_train {x_train.shape}  y_train (one-hot) {y_train.shape}"
          f"  pixel range ({x_train.min()}, {x_train.max()})")
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
    print(f"\nbatch of {len(first_batch)} -> loss {batch_loss:.4f}"
          f"  dW1 {batch_grads[0].shape} (same shape as W1)")
    print(f"largest |dW1|  batch step {biggest_mean:.6f}  ·  32 single steps averaged "
          f"{np.abs(per_example_mean).max():.6f}  -> the same number")
    print(f"largest |dW1| if we SUMMED the 32 instead: {np.abs(per_example_sum).max():.6f}"
          "  -> 32x bigger, so the step would secretly grow with batch_size")

    # --- Part 4: train, and watch the loss fall ---------------------------
    trained, epoch_means, epoch1_losses, orders, batch_log = run_epochs(
        x_train, y_train, params0, N_EPOCHS, True, 1)
    updates_done = sum(len(sizes) for sizes in batch_log)   # counted inside the real loop
    print(f"\nthe loop itself counted {len(batch_log)} epochs x {len(batch_log[0])} batches"
          f" = {updates_done} weight updates (epoch 1's last batch held {batch_log[0][-1]})")
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
    print(f"\naverage jump between batches  shuffled {jerk_shuffled:.4f}"
          f"   label-sorted, no shuffle {jerk_sorted:.4f}")
    print(f"whole-set loss after 1 epoch  before training "
          f"{full_loss(x_train, y_train, params0):.4f}  shuffled {loss_shuffled:.4f}"
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
         and batch_log[0] == [32] * 31 + [8]),
        ("slicing full MNIST the same way gives 1875 iterations per epoch, 18750 over 10",
         mnist_iters == 1875 and mnist_steps == 18750),
        ("N=100 slices into [32, 32, 32, 4]", ragged == [32, 32, 32, 4] and sum(ragged) == 100),
        ("one epoch = 31 full batches then a ragged 8, each index exactly once",
         sizes == [32] * 31 + [8] and sorted(order_demo.tolist()) == list(range(N_TRAIN))
         and np.array_equal(np.concatenate(batch_slices(order_demo, BATCH_SIZE)), order_demo)),
        ("shuffling changes the order, and each epoch reshuffles",
         not np.array_equal(orders[0], np.arange(N_TRAIN))
         and not np.array_equal(orders[0], orders[1])),
        ("batch dW1 == mean of the 32 one-example dW1, largest 0.148989, batch loss 2.359",
         np.array_equal(np.round(batch_grads[0], 12), np.round(per_example_mean, 12))
         and round(biggest_mean, 6) == 0.148989 and round(batch_loss, 4) == 2.359),
        ("the average loss falls every epoch, 0.7296 -> 0.1187",
         all(epoch_means[i + 1] < epoch_means[i] for i in range(N_EPOCHS - 1))
         and round(epoch_means[0], 4) == 0.7296 and round(epoch_means[-1], 4) == 0.1187),
        ("inside epoch 1 the per-batch loss rises 10 of 31 times, 2.3596 -> 0.3884",
         went_up == 10 and round(epoch1_losses[0], 4) == 2.3596
         and round(epoch1_losses[-1], 4) == 0.3884),
        ("label-sorted with no shuffle is jerkier, 0.7219 vs 0.0907",
         jerk_sorted > jerk_shuffled and round(jerk_sorted, 4) == 0.7219
         and round(jerk_shuffled, 4) == 0.0907),
        ("label-sorted with no shuffle ends worse, 1.0552 vs 0.3609",
         loss_sorted > loss_shuffled and round(loss_sorted, 4) == 1.0552
         and round(loss_shuffled, 4) == 0.3609),
    ]
    missed = [claim for claim, ok in checks if not ok]
    print("\n✅ you got it" if not missed else "\n❌ not yet — expected " + "; ".join(missed))
    for claim, ok in checks:      # these asserts stop the program if a claim is wrong
        assert ok, claim
