# day-03-cnn-classifier — experiment
#
# Today's big idea in two lines of output:
#   The funnel squeezes a 3×32×32 picture down to 64×4×4 — the grid halves in every
#   block while the channels grow — then flatten → head → 10 logits → softmax → loss.
#
# Every shape is PREDICTED first with Day 2's formula, then the layers run and the two
# paths are compared — so a padding or stride slip cannot hide.
# Run it:  python3 sessions/m06-cnns-vision-encoders/day-03-cnn-classifier/experiment.py

import numpy as np

# One seeded generator for every random weight, so the numbers pinned below never move.
rng = np.random.default_rng(0)


def out_size(n, k_size, p, s):
    # Day 2's tape measure: grid size after a k_size-wide window, p padding each side,
    # moving s pixels at a time. The PREDICTION path, kept separate from the layer code
    # below. `k_size` is the window's SIDE LENGTH — one number — never the weight grid;
    # day 2 spells the pair the same way (`k_size` scalar, `kernel` array).
    return (n - k_size + 2 * p) // s + 1


def same_pad(k_size):
    """"Same" padding is P = (K−1)/2 — the rule that keeps H and W unchanged.

    Named, because at K=3 it gives 1 and so do several WRONG rules (K−2, K//3, or the
    constant 1). Part 3b points this same function at K=5, where P has to be 2.
    """
    return (k_size - 1) // 2


def conv_same(x, kernels):
    # "same"-padding convolution the way the lesson slides a stencil: line each kernel cell
    # up with the padded picture and add its contribution. Shapes: x is (in_channels, H, W)
    # and kernels is (out_channels, in_channels, k_size, k_size) — a whole BANK of filters,
    # one per output channel. Day 1's `convolve(img, k)` takes a single 2-D grid and day 2's
    # `conv2d(img, k_size, pad, stride, kernel=None)` takes the weights last and optional;
    # this is the same operation with the weights in a third shape.
    out_ch, _, k_size, _ = kernels.shape
    pad = same_pad(k_size)                          # p = 1 for a 3×3 kernel keeps H and W
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    H, Wd = x.shape[1], x.shape[2]
    out = np.zeros((out_ch, H, Wd))
    for i in range(k_size):
        for j in range(k_size):
            patch = xp[:, i:i + H, j:j + Wd]   # what sits under kernel cell (i, j)
            # each filter's cell (i, j) against that slice, summed over the input channels
            out += np.tensordot(kernels[:, :, i, j], patch, axes=([1], [0]))
    return out


def max_pool_channels(x, size=2):
    # size x size windows, stride size: keep the BIGGEST value in each square, as the
    # lesson says. Day 1's `max_pool(fmap, size=2)` does this to ONE 2-D map; this works
    # through a stack of maps, (channels, height, width). The `size` dial day 1 taught as
    # the mechanism stays a dial here — it is not baked in as a literal 2.
    C, H, Wd = x.shape
    windows = x.reshape(C, H // size, size, Wd // size, size)  # axes 2 and 4 sit inside one window
    return windows.max(axis=(2, 4))


def block(x, kernels, name):
    # One LEGO brick of the funnel: conv → ReLU → pool, printing the shape each time.
    # Every number in the three printed lines is BOUND to a name first and handed back in
    # `stats`, so the self-check can read exactly what was printed — for every block, not
    # just the first one.
    pre = conv_same(x, kernels)
    n_filters = int(kernels.shape[0])
    pre_shape, pre_count = pre.shape, int(pre.size)
    n_negative = int((pre < 0).sum())
    print("  %s conv (%d filters, 3×3, same) -> %-12s %d of %d values came out negative"
          % (name, n_filters, str(pre_shape), n_negative, pre_count))
    bent = np.maximum(0, pre)                       # ReLU: keep positives, zero negatives
    bent_shape = bent.shape
    bent_min = float(bent.min())
    n_zero = int((bent == 0).sum())
    print("  %s relu (keep positives)        -> %-12s min is now %.4f, %d values sit at 0"
          % (name, str(bent_shape), bent_min, n_zero))
    small = max_pool_channels(bent, size=2)
    pooled_shape = small.shape
    print("  %s pool (2×2, stride 2)         -> %-12s grid halved, channels unchanged"
          % (name, str(pooled_shape)))
    stats = (n_filters, pre_shape, n_negative, pre_count,
             bent_shape, bent_min, n_zero, pooled_shape)
    return pre, bent, small, stats


if __name__ == "__main__":
    # --- Part 1: the picture — 3 colour channels, 32×32 pixels --------------
    # No image dataset is cached here, so the picture is painted by hand: the spec's blank
    # canvas plus a deliberately LOPSIDED pattern, because symmetric pixels would let a
    # flipped kernel or a mirrored window pass unnoticed.
    x = np.zeros((3, 32, 32))
    x[0, 4:14, 6:12] = 1.0                          # channel 0: tall bright block, left
    x[1, 20:24, 8:28] = 0.6                         # channel 1: wide dim bar, low right
    rows = np.arange(32)
    x[2, rows, (rows + 5) % 32] = 0.8               # channel 2: a diagonal stripe
    x_shape = x.shape
    bright_per_channel = [int((x[c] > 0).sum()) for c in range(3)]
    print("picture shape:", x_shape, "(channels, height, width)")
    print("bright pixels per channel:", bright_per_channel)

    # --- Part 2: predict every shape BEFORE running the layers -------------
    filters = [16, 32, 64]                          # how many stencils each block holds
    # "channels" here is the same ruler day 2 called DEPTH: axis 0 of a stack of feature
    # maps, set only by the filter count and never by K, P or S. Day 2's 12-filter stack
    # printing (12, 14, 14) is this number; here it climbs 16 -> 32 -> 64.
    side, pred_conv, pred_pool = x.shape[1], [], []
    for out_ch in filters:
        conv_side = out_size(side, k_size=3, p=1, s=1)  # conv with same padding: side unchanged
        side = out_size(conv_side, k_size=2, p=0, s=2)  # pool 2×2 stride 2: side halved
        pred_conv.append((out_ch, conv_side, conv_side))
        pred_pool.append((out_ch, side, side))
    predicted_flat = pred_pool[-1][0] * side * side
    # Both prediction lines are rendered strings — bind them, print them, check them.
    pred_conv_line = " -> ".join(str(s) for s in pred_conv)
    pred_pool_line = " -> ".join(str(s) for s in pred_pool)
    print("\nformula predicts after each conv:", pred_conv_line)
    print("        and after each pool:", pred_pool_line,
          "-> flatten", predicted_flat, "numbers")

    # --- Part 3: run the three blocks and watch the funnel squeeze ----------
    print("\nrunning the funnel:")
    feature, actual, actual_conv, stages, funnel_stats = x, [], [], [], []
    for n, out_ch in enumerate(filters, start=1):
        # Each block learns its own stencils; sqrt(fan-in) keeps the numbers readable.
        kernels = rng.standard_normal((out_ch, feature.shape[0], 3, 3)) / np.sqrt(feature.shape[0] * 9)
        pre, bent, feature, stats = block(feature, kernels, "block %d" % n)
        stages.append((pre, bent, feature))
        funnel_stats.append(stats)
        actual_conv.append(pre.shape)
        actual.append(feature.shape)
    march_line = " -> ".join(str(s) for s in actual)
    print("shape march:", x_shape, "->", march_line)

    # Block 1, channel 0, a 2×2 window whose four values all differ (a flat window would
    # let a mean/max mix-up hide).
    pre1, bent1, pooled1 = stages[0]
    window = bent1[0, 2:4, 8:10]
    pool_val = float(pooled1[0, 1, 4])              # the pooled cell that window feeds
    edge_val = float(pre1[0, 0, 5])                 # top row, where the diagonal starts
    pre1_min = round(float(pre1.min()), 4)
    bent1_min = round(float(bent1.min()), 4)
    window_shown = np.round(window.ravel(), 4)
    pool_shown = round(pool_val, 4)
    window_mean = round(float(window.mean()), 4)
    edge_shown = round(edge_val, 4)
    print("\nblock 1 detail — smallest conv value before the bend: %.4f, after it: %.4f"
          % (pre1_min, bent1_min))
    print("  channel 0, rows 2-3 × cols 8-9:", window_shown)
    print("  pool kept %.4f (the max) — the MEAN of that window would be %.4f"
          % (pool_shown, window_mean))
    print("  value on the top edge under the diagonal: %.4f — the zero padding "
          "supplied the missing row above" % edge_shown)

    # --- Part 3b: does "same" padding hold at a SECOND kernel size? ---------
    # Every conv above uses K=3, where P = (K−1)/2 = 1 — and so do several WRONG rules
    # (K−2, K//3, or just the constant 1). One shared list of kernel sizes drives both the
    # padding rule and the prediction, so K=5 has to work too, where P must come out 2.
    # The stencils are all ones (no random draw here), so these numbers stay put.
    small_x = x[:, :8, :8]                          # an 8×8 corner of the same picture
    same_pad_rows = []
    print("\n\"same\" padding at two kernel sizes, on an 8×8 corner", small_x.shape)
    for k_size_try in (3, 5):
        p_try = same_pad(k_size_try)                # the ONE rule, asked twice
        kernels_try = np.ones((2, small_x.shape[0], k_size_try, k_size_try))
        out_try = conv_same(small_x, kernels_try)
        pred_try = out_size(small_x.shape[1], k_size_try, p_try, 1)
        row = (k_size_try, p_try, pred_try, out_try.shape,
               round(float(out_try[0, 5, 6]), 4), round(float(out_try.sum()), 4))
        same_pad_rows.append(row)
        print("  K=%d -> P=(K−1)/2 = %d, formula says side %d, actual %-10s "
              "cell(5,6) %.4f, total %.4f" % row)

    # --- Part 4: flatten the last stack into one long list -----------------
    flat = feature.reshape(-1)                      # 3-D stack -> one 1-D list
    feature_shape, flat_shape = feature.shape, flat.shape
    flat_size = int(flat.size)
    print("\nflatten:", feature_shape, "->", flat_shape,
          "= %d × %d × %d = %d numbers" % (feature_shape + (flat_size,)))
    hot = int(flat.argmax())
    hot_val = round(float(flat[hot]), 4)
    print("  biggest feature value: %.4f, at position %d in the list" % (hot_val, hot))

    # --- Part 5: the dense head -> 10 raw scores (logits) ------------------
    # A trained head has bigger weights than a fresh initializer, so its scores spread
    # out. The 12.0 stands in for that training; without it all ten slices look equal.
    W_head = rng.standard_normal((10, flat.size)) * 12.0 / np.sqrt(flat.size)
    logits = W_head @ flat                          # (10, 1024) @ (1024,) -> (10,)
    head_shape, logits_shape = W_head.shape, logits.shape
    logits_shown = np.round(logits, 4)
    top_class = int(logits.argmax())
    print("\nhead:", head_shape, "@", flat_shape, "->", logits_shape)
    print("  logits:", "  ".join("%.4f" % v for v in logits_shown))
    print("  raw scores, NOT probabilities — the biggest is class", top_class)

    # --- Part 6: softmax -> ten probabilities that add to 1 ----------------
    e = np.exp(logits - logits.max())               # subtract the max so exp cannot blow up
    probs = e / e.sum()                             # divide by the total -> slices of one pizza
    probs_shown = np.round(probs, 4)
    probs_sum = round(float(probs.sum()), 12)
    probs_top = round(float(probs.max()), 4)
    top_prob_class = int(probs.argmax())
    print("\nprobs: ", "  ".join("%.4f" % p for p in probs_shown))
    print("  they add to %.12f, and the biggest slice is %.4f for class %d"
          % (probs_sum, probs_top, top_prob_class))

    # --- Part 7: cross-entropy — how wrong was the guess? ------------------
    true_class = 0
    loss = -np.log(probs[true_class])
    p_true = round(float(probs[true_class]), 4)
    loss_shown = round(float(loss), 4)
    print("\ncross-entropy with true_class = 0: -log(%.4f) = %.4f" % (p_true, loss_shown))
    # The lesson's small-loss / big-loss contrast, from THIS model's own probabilities
    # rather than two typed-in ones: its favourite class vs. its least liked.
    best, worst = int(probs.argmax()), int(probs.argmin())
    loss_best, loss_worst = -np.log(probs[best]), -np.log(probs[worst])
    p_best, p_worst = round(float(probs[best]), 4), round(float(probs[worst]), 4)
    loss_best_shown = round(float(loss_best), 4)
    loss_worst_shown = round(float(loss_worst), 4)
    print("  if the true class were %d (p = %.4f, its favourite):   loss %.4f — small"
          % (best, p_best, loss_best_shown))
    print("  if the true class were %d (p = %.4f, its least liked): loss %.4f — large"
          % (worst, p_worst, loss_worst_shown))
    # The ladder does arithmetic at the print site too, so bind each pair first.
    ladder = [(p, round(float(-np.log(p)), 2)) for p in (0.01, 0.30, 0.91, 0.999)]
    print("  surprise ladder:", "  ".join("p=%.3f -> loss %.2f" % pair for pair in ladder))
    print("\nimage -> blocks (grid shrinks, channels grow) -> flatten -> dense head ->"
          " 10 logits -> softmax (probs add to 1) -> cross-entropy (how wrong);"
          " that is one forward pass and the loss training minimizes.")

    # --- Self-check: one boolean per claim ---------------------------------
    # Every expected number was read off a real run and typed in, so the code above
    # cannot quietly agree with itself. Each claim reads the SAME name that was printed,
    # so corrupting a printed number breaks a claim instead of slipping past.
    expected_logits = np.array([0.9524, 2.3005, -3.9897, -0.5339, 3.1951,
                                1.7214, -2.4551, -1.8453, 1.4541, -0.9565])
    shapes_ok = (x_shape == (3, 32, 32)
                 and bright_per_channel == [60, 80, 32]
                 and actual == [(16, 16, 16), (32, 8, 8), (64, 4, 4)]
                 and march_line == "(16, 16, 16) -> (32, 8, 8) -> (64, 4, 4)")
    # Two independent paths: the tape-measure formula vs. the padding-and-reshape code.
    prediction_ok = (pred_conv == actual_conv and pred_pool == actual
                     and predicted_flat == 1024
                     and pred_conv_line == "(16, 32, 32) -> (32, 16, 16) -> (64, 8, 8)"
                     and pred_pool_line == "(16, 16, 16) -> (32, 8, 8) -> (64, 4, 4)")
    # The whole funnel table, block by block — before this, only block 1's numbers were
    # pinned and blocks 2 and 3 could print anything at all.
    funnel_ok = funnel_stats == [
        (16, (16, 32, 32), 2901, 16384, (16, 32, 32), 0.0, 13925, (16, 16, 16)),
        (32, (32, 16, 16), 2750, 8192, (32, 16, 16), 0.0, 5278, (32, 8, 8)),
        (64, (64, 8, 8), 1871, 4096, (64, 8, 8), 0.0, 2063, (64, 4, 4)),
    ]
    relu_ok = (funnel_stats[0][2] == 2901              # the conv really made negatives
               and pre1_min == -1.0496
               and funnel_stats[0][5] == 0.0           # and the bend flattened every one
               and bent1_min == 0.0
               and funnel_stats[0][6] == 13925)        # 11024 were already 0, plus the 2901
    pad_ok = edge_shown == -0.2169
    pool_ok = (pool_shown == 0.5226                    # the window's max...
               and window_mean == 0.2577               # ...not its mean
               and window_shown.tolist() == [0.2249, 0.1391, 0.1443, 0.5226])
    flat_ok = (flat_size == 1024 and hot == 308 and hot_val == 0.7778
               and flat_shape == (1024,) and feature_shape == (64, 4, 4)
               and feature_shape[0] * feature_shape[1] * feature_shape[2] == flat_size)
    logits_ok = (logits_shape == (10,) and head_shape == (10, 1024)
                 and np.array_equal(logits_shown, expected_logits))
    probs_ok = (probs_sum == 1.0
                and top_class == 4 and top_prob_class == 4 and probs_top == 0.5076
                and int(probs.argmin()) == 2 and round(float(probs.min()), 4) == 0.0004
                and probs_shown.tolist() == [0.0539, 0.2075, 0.0004, 0.0122, 0.5076,
                                             0.1163, 0.0018, 0.0033, 0.089, 0.008])
    loss_ok = (loss_shown == 2.9207 and p_true == 0.0539
               and loss_best_shown == 0.678 and p_best == 0.5076
               and loss_worst_shown == 7.8628 and p_worst == 0.0004)
    # The surprise ladder is arithmetic done for the printed line, so pin the pairs.
    ladder_ok = ladder == [(0.01, 4.61), (0.30, 1.2), (0.91, 0.09), (0.999, 0.0)]
    # "Same" padding must hold at a SECOND kernel size. At K=3, P=1 is also what K−2,
    # K//3 and the constant 1 give; at K=5 only (K−1)/2 = 2 keeps the grid at 8×8.
    same_pad_ok = same_pad_rows == [(3, 1, 8, (2, 8, 8), 6.0, 143.6),
                                    (5, 2, 8, (2, 8, 8), 8.0, 311.6)]

    if (shapes_ok and prediction_ok and funnel_ok and relu_ok and pad_ok and pool_ok
            and flat_ok and logits_ok and probs_ok and loss_ok and ladder_ok
            and same_pad_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected (3,32,32) -> (16,16,16) -> (32,8,8) -> (64,4,4) with "
              "the formula predicting the same, flatten 1024 (biggest 0.7778 at 308), block 1 "
              "making 2901 negatives (13925 zeros afterwards) with minimum -1.0496 that "
              "ReLU lifts to 0.0, blocks 2 and 3 making 2750 and 1871, top-edge value -0.2169, "
              "pooled 0.5226 (max) not 0.2577 "
              "(mean), logits [0.9524 2.3005 "
              "-3.9897 -0.5339 3.1951 1.7214 -2.4551 -1.8453 1.4541 -0.9565], probs summing "
              "to 1.0 with 0.5076 on class 4 / 0.0004 on class 2, losses 2.9207, 0.678, 7.8628, "
              "the surprise ladder 4.61 / 1.20 / 0.09 / 0.00, and \"same\" padding holding at "
              "K=5 with P=2 (8×8 in, 8×8 out)")

    # These asserts stop the program if any single fact is wrong.
    assert shapes_ok, "the funnel must go (3,32,32) -> (16,16,16) -> (32,8,8) -> (64,4,4)"
    assert prediction_ok, "Day 2's formula must predict the shapes the layers actually give"
    assert funnel_ok, "every printed number in the three-block funnel table must match"
    assert relu_ok, "block 1: 2901 negatives, min -1.0496, and ReLU must flatten them (13925 zeros)"
    assert pad_ok, "the top-edge conv value should be -0.2169 with zero padding above"
    assert pool_ok, "the pool must keep 0.5226 (the window's max), not 0.2577 (its mean)"
    assert flat_ok, "flatten should give 1024 numbers, biggest 0.7778 at position 308"
    assert logits_ok, "the head should give the 10 pinned logits, biggest 3.1951 on class 4"
    assert probs_ok, "probs must add to 1 with 0.5076 on class 4 and 0.0004 on class 2"
    assert loss_ok, "losses should be 2.9207 for class 0, 0.678 (best) and 7.8628 (worst)"
    assert ladder_ok, "the surprise ladder should read 4.61 / 1.20 / 0.09 / 0.00"
    assert same_pad_ok, "\"same\" padding must keep 8×8 at K=3 (P=1) and at K=5 (P=2)"
