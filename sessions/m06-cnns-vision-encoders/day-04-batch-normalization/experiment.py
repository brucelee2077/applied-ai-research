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
    # Every number the learner reads is bound to a name here and then checked below, so
    # the printed line and the checked line can never be two different computations.
    shown_x_shape = x.shape
    shown_mu = round(float(mu), 4)
    shown_sigma = round(float(sigma), 4)
    print("Part 1 · one channel across a batch of 5, shape", shown_x_shape, ":", x)
    print("  average mu =", shown_mu, " spread sigma =", shown_sigma)
    # Predict without calling .mean(): an evenly spaced list averages to its midpoint.
    predicted_mu = (x[0] + x[-1]) / 2
    shown_predicted_mu = round(float(predicted_mu), 4)
    shown_prediction_matches = bool(predicted_mu == mu)
    print("  predicted from the two ends =", shown_predicted_mu,
          "-> equals the measured average:", shown_prediction_matches)

    # --- Part 2: the thermostat move — steady it to 0 and 1 --------------
    # Subtract the average (kills the off-centre drift), divide by the spread (kills
    # the too-big / too-tiny drift), and eps keeps that divide safe.
    #
    # One word, two operations — worth naming once, here, where the word is defined.
    # "Normalize" in this file (and on days 5 and 6) means THIS recipe: subtract a mean,
    # divide by a spread. Day 8 uses the same word for a different operation — divide
    # each row by its own length, which changes only length and never re-centres
    # anything — and its helper is the one actually called `normalize`.
    # And eps is the mechanism, not decoration: Part 4 below proves the same divide
    # returns nan without it. Later days drop it (day 5 divides by a frozen spread of
    # 4.6098, day 8 by a row length) and are safe only because those divisors are
    # provably far from 0 — not because the crumb stopped mattering.
    x_hat = (x - mu) / np.sqrt(sigma ** 2 + eps)
    shown_x_hat_shape = x_hat.shape
    shown_x_hat = np.round(x_hat, 4)
    shown_x_hat_mean = round(float(x_hat.mean()), 12)
    shown_x_hat_std = round(float(x_hat.std()), 6)
    print("\nPart 2 · steady it: (x - mu) / sqrt(sigma^2 + eps), shape", shown_x_hat_shape)
    print("  x_hat       :", shown_x_hat, " <- 30, which WAS the average, is now 0")
    print("  average now :", shown_x_hat_mean,
          " spread now:", shown_x_hat_std)

    # --- Part 3: the two learnable knobs, gamma and beta -----------------
    gamma, beta = 2.0, 5.0                  # scale (volume knob), shift (slider)
    y = gamma * x_hat + beta
    shown_y = np.round(y, 4)
    shown_y_shape = y.shape
    shown_y_mean = round(float(y.mean()), 6)
    shown_y_std = round(float(y.std()), 6)
    # Predicted from the algebra, not from y: because x_hat sits at average 0 and spread
    # 1, mean(gamma*x_hat + beta) = beta and spread(gamma*x_hat) = gamma.
    print("\nPart 3 · scale by gamma =", gamma, "then shift by beta =", beta)
    print("  y           :", shown_y, " shape", shown_y_shape)
    print("  predicted center", beta, "spread", gamma, "-> measured center",
          shown_y_mean, "spread", shown_y_std)
    # Set gamma = the old spread and beta = the old average: the layer undoes itself.
    y_undo = sigma * x_hat + mu
    shown_y_undo = np.round(y_undo, 4)
    print("  gamma=sigma, beta=mu ->", shown_y_undo, " the original x is back")

    # --- Part 4: why epsilon is there — a flat channel -------------------
    # Every value the same, so the spread is exactly 0, and a plain divide breaks.
    flat = np.array([3., 3., 3., 3.])
    flat_var = flat.var()
    flat_hat = (flat - flat.mean()) / np.sqrt(flat_var + eps)
    with np.errstate(invalid="ignore", divide="ignore"):
        naive = (flat - flat.mean()) / np.sqrt(flat_var)    # no eps: 0 / 0
    shown_flat_var = round(float(flat_var), 6)
    shown_flat_all_finite = bool(np.isfinite(flat_hat).all())
    shown_naive_all_nan = bool(np.isnan(naive).all())
    # The stretch line quotes the channel's own value, so read it OFF the channel rather
    # than typing "3" into the text: a channel of 6s would otherwise still print the
    # sentence "3/sqrt(0+eps) = 948.7" beside a printed [6. 6. 6. 6.]. Rendered once,
    # pinned below, then printed.
    shown_eps_max_stretch = round(float(flat[0] / np.sqrt(0.0 + eps)), 1)
    eps_stretch_line = ("  eps sets the biggest stretch: %g/sqrt(0+eps) = %s"
                        % (flat[0], shown_eps_max_stretch))
    print("\nPart 4 · a flat channel", flat, "has spread", shown_flat_var)
    print("  with eps   :", flat_hat, " all finite:", shown_flat_all_finite)
    print("  without eps:", naive, " all not-a-number:", shown_naive_all_nan)
    print(eps_stretch_line)

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
    shown_x2_shape = x2.shape
    shown_chan_mu_shape = chan_mu.shape
    shown_chan_mu = np.round(chan_mu.ravel(), 4)
    shown_chan_sigma = np.round(np.sqrt(chan_var).ravel(), 4)
    shown_col_means = np.round(x2_hat.mean(axis=0), 6)
    shown_col_stds = np.round(x2_hat.std(axis=0), 6)
    shown_row_means = np.round(x2_hat.mean(axis=1), 4)
    print("\nPart 5 · per channel: x2 shape", shown_x2_shape,
          " chan_mu shape", shown_chan_mu_shape)
    print("  channel averages:", shown_chan_mu,
          " channel spreads:", shown_chan_sigma)
    # The middle column (index 1) has the smallest spread of the three, so eps leaves a
    # visible fingerprint there: its spread comes back 0.999998 rather than exactly 1.
    print("  x2_hat column averages:", shown_col_means,
          " column spreads:", shown_col_stds)
    print("  x2_hat ROW averages:", shown_row_means, "<- rows NOT centred")

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
    shown_mem_first4 = np.round(mean_history[:4], 4)
    shown_running_mean = round(running_mean, 4)
    shown_running_var = round(running_var, 4)
    print("\nPart 6 · running memory over", n_batches, "batches, momentum", momentum)
    print("  memory after batches 1-4:", shown_mem_first4, "<- creeps up")
    print("  final running_mean:", shown_running_mean,
          " running_var:", shown_running_var)
    memory = (running_mean, running_var)
    value = 25.0                             # one test value
    batch_a = np.array([0., 5., 10.])        # it arrives among small numbers...
    batch_b = np.array([80., 90., 100.])     # ...or among large ones
    train_a = batch_norm_one(value, batch_a, True, memory)
    train_b = batch_norm_one(value, batch_b, True, memory)
    eval_a = batch_norm_one(value, batch_a, False, memory)
    eval_b = batch_norm_one(value, batch_b, False, memory)
    shown_train_a = round(train_a, 4)
    shown_train_b = round(train_b, 4)
    shown_eval_a = round(eval_a, 4)
    shown_eval_b = round(eval_b, 4)
    shown_eval_identical = bool(eval_a == eval_b)
    print("  train mode (live batch):", shown_train_a, "vs", shown_train_b,
          "-> the SAME value wobbles with its batch-mates")
    print("  eval mode  (memory)    :", shown_eval_a, "vs", shown_eval_b,
          "-> identical:", shown_eval_identical)
    print("\nTakeaway: batch norm = subtract batch mean, divide by batch spread (+eps)"
          " -> avg 0 spread 1; then scale by gamma and shift by beta;"
          " train uses the live batch, eval uses the frozen running memory.")

    # --- Self-check: one boolean per claim -------------------------------
    # Every number below was copied from a real run and written down here, so a broken
    # change above cannot quietly agree with itself. Each claim reads the SAME `shown_*`
    # value that was printed, so corrupting a printed number also breaks its claim.
    raw_stats_ok = (shown_mu == 30.0 and shown_sigma == 14.1421
                    and shown_x_shape == (5,))
    prediction_ok = shown_prediction_matches and shown_predicted_mu == 30.0
    steady_ok = (pinned(shown_x_hat, [-1.4142, -0.7071, 0.0, 0.7071, 1.4142])
                 and shown_x_hat_shape == (5,)
                 and shown_x_hat_mean == 0.0
                 and shown_x_hat_std == 1.0)
    knobs_ok = (pinned(shown_y, [2.1716, 3.5858, 5.0, 6.4142, 7.8284])
                and shown_y_shape == (5,)
                and shown_y_mean == 5.0        # beta set the centre
                and shown_y_std == 2.0)        # gamma set the spread
    undo_ok = pinned(shown_y_undo, x)
    eps_finite_ok = shown_flat_all_finite and pinned(flat_hat, [0., 0., 0., 0.])
    no_eps_breaks_ok = shown_naive_all_nan                  # negative test: 0 / 0 is nan
    # The flat channel is the printed evidence for "spread exactly 0", and the narrative
    # two lines up calls it a channel of 3s, so pin the values and not only the spread:
    # a channel of 6s has spread 0.0 too and would slip past a spread-only check.
    flat_spread_ok = shown_flat_var == 0.0 and pinned(flat, [3., 3., 3., 3.])
    eps_scale_ok = (shown_eps_max_stretch == 948.7
                    and eps_stretch_line == "  eps sets the biggest stretch: 3/sqrt(0+eps) = 948.7")
    chan_stats_ok = (pinned(shown_chan_mu, [30., 3., -1.])
                     and pinned(shown_chan_sigma, [14.1421, 1.6733, 5.7619])
                     and shown_x2_shape == (5, 3) and shown_chan_mu_shape == (1, 3))
    chan_centered_ok = (pinned(shown_col_means, [0., 0., 0.], 6)
                        and pinned(shown_col_stds, [1.0, 0.999998, 1.0], 6))
    rows_not_centered_ok = pinned(shown_row_means,
                                  [-1.1591, -0.2936, 0.3793, 0.949, 0.1243])
    memory_ok = shown_running_mean == 27.0654 and shown_running_var == 190.7659
    memory_creep_ok = pinned(shown_mem_first4, [3.4928, 4.9057, 7.6808, 9.541])
    eval_ok = (shown_eval_identical and shown_eval_a == -0.1495
               and shown_eval_b == -0.1495)
    train_wobbles_ok = shown_train_a == 1.6036 and shown_train_b == -1.6798

    if (raw_stats_ok and prediction_ok and steady_ok and knobs_ok and undo_ok
            and eps_finite_ok and no_eps_breaks_ok and flat_spread_ok and eps_scale_ok
            and chan_stats_ok and chan_centered_ok and rows_not_centered_ok
            and memory_ok and memory_creep_ok and eval_ok
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
    assert flat_spread_ok, "the flat channel must print as four 3s with spread exactly 0.0"
    assert eps_scale_ok, "the stretch line must read 3/sqrt(0 + eps) = 948.7"
    assert chan_stats_ok, "per-channel averages/spreads must be read DOWN the batch (axis=0)"
    assert chan_centered_ok, "every CHANNEL (column) must end at average 0, spread ~1"
    assert rows_not_centered_ok, "rows must NOT be centred — a sample is never normalized"
    assert memory_ok, "the running memory should land on 27.0654 / 190.7659"
    assert memory_creep_ok, "the first four running means should be 3.4928 4.9057 7.6808 9.541"
    assert eval_ok, "eval mode must give 25.0 the same answer (-0.1495) in either batch"
    assert train_wobbles_ok, "train mode's answer must change with the batch-mates"
