# day-04-batch-normalization — experiment
#
# Today's big idea in two lines of output:
#   Subtract the batch's average, divide by its spread, and a drifting channel
#   snaps to average 0 and spread 1 — then gamma and beta hand any range back.
#
# Also here: the epsilon that keeps the divide safe on a flat channel, one thermostat
# per CHANNEL (read DOWN the batch), and why eval mode reads a frozen running memory.
# Run it:  python3 sessions/m06-cnns-vision-encoders/day-04-batch-normalization/experiment.py

import numpy as np  # numpy gives us arrays, averages (.mean) and spreads (.std)

eps = 1e-5        # the tiny safety number inside the divide, so it is never / 0
momentum = 0.1    # how much of each fresh batch is blended into the running memory


def pinned(actual, expected, digits=4):
    """True when `actual`, rounded, equals numbers that were written down by hand."""
    return np.array_equal(np.round(actual, digits), np.array(expected))


def batch_norm_one(value, batch, training, memory):
    """Push ONE value through the layer: train reads the live batch, eval the memory."""
    if training:
        live = np.append(batch, value)      # the value arrives inside this batch
        mean, var = live.mean(), live.var()
    else:
        mean, var = memory                  # frozen numbers; the batch is not read
    return float((value - mean) / np.sqrt(var + eps))


if __name__ == "__main__":
    # --- Part 1: the "before" — one channel that drifted hot --------------
    # This channel's value for each of the 5 pictures in the batch.
    x = np.array([10., 20., 30., 40., 50.])
    mu = x.mean()      # the batch average — what the thermostat reads
    sigma = x.std()    # the batch spread; numpy's .std divides by N, as batch norm does
    print("Part 1 · one channel across a batch of 5, shape", x.shape, ":", x)
    print("  average mu =", round(float(mu), 4), " spread sigma =", round(float(sigma), 4))
    # Predict without calling .mean(): an evenly spaced list averages to its midpoint.
    predicted_mu = (x[0] + x[-1]) / 2
    print("  predicted from the two ends =", round(float(predicted_mu), 4),
          "-> equals the measured average:", bool(predicted_mu == mu))

    # --- Part 2: the thermostat move — steady it to 0 and 1 --------------
    # Subtract the average (kills the off-centre drift), divide by the spread (kills
    # the too-big / too-tiny drift), and eps keeps that divide safe.
    x_hat = (x - mu) / np.sqrt(sigma ** 2 + eps)
    print("\nPart 2 · steady it: (x - mu) / sqrt(sigma^2 + eps), shape", x_hat.shape)
    print("  x_hat       :", np.round(x_hat, 4), " <- 30, which WAS the average, is now 0")
    print("  average now :", round(float(x_hat.mean()), 12),
          " spread now:", round(float(x_hat.std()), 6))

    # --- Part 3: the two learnable knobs, gamma and beta -----------------
    gamma, beta = 2.0, 5.0                  # scale (volume knob), shift (slider)
    y = gamma * x_hat + beta
    # Predicted from the algebra, not from y: because x_hat sits at average 0 and spread
    # 1, mean(gamma*x_hat + beta) = beta and spread(gamma*x_hat) = gamma.
    print("\nPart 3 · scale by gamma =", gamma, "then shift by beta =", beta)
    print("  y           :", np.round(y, 4), " shape", y.shape)
    print("  predicted center", beta, "spread", gamma, "-> measured center",
          round(float(y.mean()), 6), "spread", round(float(y.std()), 6))
    # Set gamma = the old spread and beta = the old average: the layer undoes itself.
    y_undo = sigma * x_hat + mu
    print("  gamma=sigma, beta=mu ->", np.round(y_undo, 4), " the original x is back")

    # --- Part 4: why epsilon is there — a flat channel -------------------
    # Every value the same, so the spread is exactly 0, and a plain divide breaks.
    flat = np.array([3., 3., 3., 3.])
    flat_var = flat.var()
    flat_hat = (flat - flat.mean()) / np.sqrt(flat_var + eps)
    with np.errstate(invalid="ignore", divide="ignore"):
        naive = (flat - flat.mean()) / np.sqrt(flat_var)    # no eps: 0 / 0
    print("\nPart 4 · a flat channel", flat, "has spread", round(float(flat_var), 6))
    print("  with eps   :", flat_hat, " all finite:", bool(np.isfinite(flat_hat).all()))
    print("  without eps:", naive, " all not-a-number:", bool(np.isnan(naive).all()))
    print("  eps sets the biggest stretch: 3/sqrt(0+eps) =",
          round(float(3.0 / np.sqrt(0.0 + eps)), 1))

    # --- Part 5: one thermostat per channel ------------------------------
    # 5 pictures (rows) x 3 channels (columns). Each channel drifts on its own, so each
    # gets its OWN average and spread, read DOWN the batch (axis=0), not across a row.
    # Note the layout: this is (pictures, channels), so channels are axis 1. Day 3's conv
    # activations are (channels, height, width) — channels there are axis 0. Same idea,
    # different axis number: batch norm always reduces over everything EXCEPT the channel.
    x2 = np.array([[10., 1., -6.], [20., 3., -2.], [30., 2., 9.],
                   [40., 6., 1.], [50., 3., -7.]])
    chan_mu = x2.mean(axis=0, keepdims=True)      # one average per channel -> (1, 3)
    chan_var = x2.var(axis=0, keepdims=True)      # one spread^2 per channel -> (1, 3)
    x2_hat = (x2 - chan_mu) / np.sqrt(chan_var + eps)
    print("\nPart 5 · per channel: x2 shape", x2.shape, " chan_mu shape", chan_mu.shape)
    print("  channel averages:", np.round(chan_mu.ravel(), 4),
          " channel spreads:", np.round(np.sqrt(chan_var).ravel(), 4))
    # The middle column (index 1) has the smallest spread of the three, so eps leaves a
    # visible fingerprint there: its spread comes back 0.999998 rather than exactly 1.
    print("  x2_hat column averages:", np.round(x2_hat.mean(axis=0), 6),
          " column spreads:", np.round(x2_hat.std(axis=0), 6))
    print("  x2_hat ROW averages:", np.round(x2_hat.mean(axis=1), 4), "<- rows NOT centred")

    # --- Part 6: the running memory, then train mode vs eval mode --------
    running_mean, running_var = 0.0, 1.0     # the memory starts cold
    n_batches = 30
    rng = np.random.default_rng(0)           # seeded, so the printed memory never moves
    mean_history = []
    for step in range(1, n_batches + 1):
        # A fresh batch of 8 values for this channel, around the same typical numbers.
        batch = rng.normal(loc=30.0, scale=14.0, size=8)
        running_mean = (1 - momentum) * running_mean + momentum * batch.mean()
        running_var = (1 - momentum) * running_var + momentum * batch.var()
        mean_history.append(running_mean)
    print("\nPart 6 · running memory over", n_batches, "batches, momentum", momentum)
    print("  memory after batches 1-4:", np.round(mean_history[:4], 4), "<- creeps up")
    print("  final running_mean:", round(running_mean, 4),
          " running_var:", round(running_var, 4))
    memory = (running_mean, running_var)
    value = 25.0                             # one test value
    batch_a = np.array([0., 5., 10.])        # it arrives among small numbers...
    batch_b = np.array([80., 90., 100.])     # ...or among large ones
    train_a = batch_norm_one(value, batch_a, True, memory)
    train_b = batch_norm_one(value, batch_b, True, memory)
    eval_a = batch_norm_one(value, batch_a, False, memory)
    eval_b = batch_norm_one(value, batch_b, False, memory)
    print("  train mode (live batch):", round(train_a, 4), "vs", round(train_b, 4),
          "-> the SAME value wobbles with its batch-mates")
    print("  eval mode  (memory)    :", round(eval_a, 4), "vs", round(eval_b, 4),
          "-> identical:", bool(eval_a == eval_b))
    print("\nTakeaway: batch norm = subtract batch mean, divide by batch spread (+eps)"
          " -> avg 0 spread 1; then scale by gamma and shift by beta;"
          " train uses the live batch, eval uses the frozen running memory.")

    # --- Self-check: one boolean per claim -------------------------------
    # Every number below was copied from a real run and written down here, so a broken
    # change above cannot quietly agree with itself.
    raw_stats_ok = round(float(mu), 4) == 30.0 and round(float(sigma), 4) == 14.1421
    prediction_ok = bool(predicted_mu == mu)
    steady_ok = (pinned(x_hat, [-1.4142, -0.7071, 0.0, 0.7071, 1.4142])
                 and round(float(x_hat.mean()), 12) == 0.0
                 and round(float(x_hat.std()), 6) == 1.0)
    knobs_ok = (pinned(y, [2.1716, 3.5858, 5.0, 6.4142, 7.8284])
                and round(float(y.mean()), 6) == 5.0        # beta set the centre
                and round(float(y.std()), 6) == 2.0)        # gamma set the spread
    undo_ok = pinned(y_undo, x)
    eps_finite_ok = bool(np.isfinite(flat_hat).all()) and pinned(flat_hat, [0., 0., 0., 0.])
    no_eps_breaks_ok = bool(np.isnan(naive).all())          # negative test: 0 / 0 is nan
    eps_scale_ok = round(float(3.0 / np.sqrt(0.0 + eps)), 1) == 948.7
    chan_stats_ok = (pinned(chan_mu.ravel(), [30., 3., -1.])
                     and pinned(np.sqrt(chan_var).ravel(), [14.1421, 1.6733, 5.7619]))
    chan_centered_ok = (pinned(x2_hat.mean(axis=0), [0., 0., 0.], 6)
                        and pinned(x2_hat.std(axis=0), [1.0, 0.999998, 1.0], 6))
    rows_not_centered_ok = pinned(x2_hat.mean(axis=1),
                                  [-1.1591, -0.2936, 0.3793, 0.949, 0.1243])
    memory_ok = round(running_mean, 4) == 27.0654 and round(running_var, 4) == 190.7659
    eval_ok = eval_a == eval_b and round(eval_a, 4) == -0.1495
    train_wobbles_ok = round(train_a, 4) == 1.6036 and round(train_b, 4) == -1.6798

    if (raw_stats_ok and prediction_ok and steady_ok and knobs_ok and undo_ok
            and eps_finite_ok and no_eps_breaks_ok and eps_scale_ok and chan_stats_ok
            and chan_centered_ok and rows_not_centered_ok and memory_ok and eval_ok
            and train_wobbles_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected mu 30.0, sigma 14.1421; x_hat "
              "[-1.4142 -0.7071 0. 0.7071 1.4142] at average 0, spread 1; y centred on 5.0 "
              "with spread 2.0; gamma=sigma, beta=mu giving x back; a flat channel of zeros "
              "(nan without eps) and 3/sqrt(eps) = 948.7; channel averages [30. 3. -1.] with "
              "only COLUMNS centred; memory 27.0654 / 190.7659; and 25.0 -> -0.1495 in eval "
              "mode for either batch, but 1.6036 vs -1.6798 in train mode")

    assert raw_stats_ok, "the raw batch should sit at average 30.0 with spread 14.1421"
    assert prediction_ok, "(first + last) / 2 must equal the measured average"
    assert steady_ok, "x_hat should be [-1.4142 -0.7071 0. 0.7071 1.4142], average 0, spread 1"
    assert knobs_ok, "y should be [2.1716 3.5858 5. 6.4142 7.8284], centre 5.0, spread 2.0"
    assert undo_ok, "gamma=sigma and beta=mu must undo the normalization and give x back"
    assert eps_finite_ok, "a flat channel must normalize to finite zeros, not nan"
    assert no_eps_breaks_ok, "without eps the same divide gives nan — that is why eps exists"
    assert eps_scale_ok, "3 / sqrt(0 + eps) should be 948.7"
    assert chan_stats_ok, "per-channel averages/spreads must be read DOWN the batch (axis=0)"
    assert chan_centered_ok, "every CHANNEL (column) must end at average 0, spread ~1"
    assert rows_not_centered_ok, "rows must NOT be centred — a sample is never normalized"
    assert memory_ok, "the running memory should land on 27.0654 / 190.7659"
    assert eval_ok, "eval mode must give 25.0 the same answer (-0.1495) in either batch"
    assert train_wobbles_ok, "train mode's answer must change with the batch-mates"
