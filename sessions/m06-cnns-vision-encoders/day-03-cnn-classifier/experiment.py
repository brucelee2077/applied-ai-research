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


def out_size(n, k, p, s):
    # Day 2's tape measure: grid size after a k-wide window, p padding each side, moving
    # s pixels at a time. The PREDICTION path, kept separate from the layer code below.
    return (n - k + 2 * p) // s + 1


def conv_same(x, W):
    # "same"-padding convolution the way the lesson slides a stencil: line each kernel cell
    # up with the padded picture and add its contribution. Shapes: x is (in_channels, H, W)
    # and W is (out_channels, in_channels, k, k).
    out_ch, _, k, _ = W.shape
    pad = (k - 1) // 2                              # p = 1 for a 3×3 kernel keeps H and W
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    H, Wd = x.shape[1], x.shape[2]
    out = np.zeros((out_ch, H, Wd))
    for i in range(k):
        for j in range(k):
            patch = xp[:, i:i + H, j:j + Wd]   # what sits under kernel cell (i, j)
            # each filter's cell (i, j) against that slice, summed over the input channels
            out += np.tensordot(W[:, :, i, j], patch, axes=([1], [0]))
    return out


def maxpool2(x):
    # 2×2 windows, stride 2: keep the BIGGEST value in each square, as the lesson says.
    C, H, Wd = x.shape
    windows = x.reshape(C, H // 2, 2, Wd // 2, 2)   # axes 2 and 4 sit inside one window
    return windows.max(axis=(2, 4))


def block(x, W, name):
    # One LEGO brick of the funnel: conv → ReLU → pool, printing the shape each time.
    pre = conv_same(x, W)
    print("  %s conv (%d filters, 3×3, same) -> %-12s %d of %d values came out negative"
          % (name, W.shape[0], str(pre.shape), int((pre < 0).sum()), pre.size))
    bent = np.maximum(0, pre)                       # ReLU: keep positives, zero negatives
    print("  %s relu (keep positives)        -> %-12s min is now %.4f, %d values sit at 0"
          % (name, str(bent.shape), bent.min(), int((bent == 0).sum())))
    small = maxpool2(bent)
    print("  %s pool (2×2, stride 2)         -> %-12s grid halved, channels unchanged"
          % (name, str(small.shape)))
    return pre, bent, small


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
    print("picture shape:", x.shape, "(channels, height, width)")
    print("bright pixels per channel:", [int((x[c] > 0).sum()) for c in range(3)])

    # --- Part 2: predict every shape BEFORE running the layers -------------
    filters = [16, 32, 64]                          # how many stencils each block holds
    side, pred_conv, pred_pool = x.shape[1], [], []
    for out_ch in filters:
        conv_side = out_size(side, k=3, p=1, s=1)    # conv with same padding: side unchanged
        side = out_size(conv_side, k=2, p=0, s=2)    # pool 2×2 stride 2: side halved
        pred_conv.append((out_ch, conv_side, conv_side))
        pred_pool.append((out_ch, side, side))
    predicted_flat = pred_pool[-1][0] * side * side
    print("\nformula predicts after each conv:", " -> ".join(str(s) for s in pred_conv))
    print("        and after each pool:", " -> ".join(str(s) for s in pred_pool),
          "-> flatten", predicted_flat, "numbers")

    # --- Part 3: run the three blocks and watch the funnel squeeze ----------
    print("\nrunning the funnel:")
    feature, actual, actual_conv, stages = x, [], [], []
    for n, out_ch in enumerate(filters, start=1):
        # Each block learns its own stencils; sqrt(fan-in) keeps the numbers readable.
        W = rng.standard_normal((out_ch, feature.shape[0], 3, 3)) / np.sqrt(feature.shape[0] * 9)
        pre, bent, feature = block(feature, W, "block %d" % n)
        stages.append((pre, bent, feature))
        actual_conv.append(pre.shape)
        actual.append(feature.shape)
    print("shape march:", x.shape, "->", " -> ".join(str(s) for s in actual))

    # Block 1, channel 0, a 2×2 window whose four values all differ (a flat window would
    # let a mean/max mix-up hide).
    pre1, bent1, pooled1 = stages[0]
    window = bent1[0, 2:4, 8:10]
    pool_val = float(pooled1[0, 1, 4])              # the pooled cell that window feeds
    edge_val = float(pre1[0, 0, 5])                 # top row, where the diagonal starts
    print("\nblock 1 detail — smallest conv value before the bend: %.4f, after it: %.4f"
          % (pre1.min(), bent1.min()))
    print("  channel 0, rows 2-3 × cols 8-9:", np.round(window.ravel(), 4))
    print("  pool kept %.4f (the max) — the MEAN of that window would be %.4f"
          % (pool_val, window.mean()))
    print("  value on the top edge under the diagonal: %.4f — the zero padding "
          "supplied the missing row above" % edge_val)

    # --- Part 4: flatten the last stack into one long list -----------------
    flat = feature.reshape(-1)                      # 3-D stack -> one 1-D list
    print("\nflatten:", feature.shape, "->", flat.shape,
          "= %d × %d × %d = %d numbers" % (feature.shape + (flat.size,)))
    hot = int(flat.argmax())
    print("  biggest feature value: %.4f, at position %d in the list" % (flat[hot], hot))

    # --- Part 5: the dense head -> 10 raw scores (logits) ------------------
    # A trained head has bigger weights than a fresh initializer, so its scores spread
    # out. The 12.0 stands in for that training; without it all ten slices look equal.
    W_head = rng.standard_normal((10, flat.size)) * 12.0 / np.sqrt(flat.size)
    logits = W_head @ flat                          # (10, 1024) @ (1024,) -> (10,)
    print("\nhead:", W_head.shape, "@", flat.shape, "->", logits.shape)
    print("  logits:", "  ".join("%.4f" % v for v in logits))
    print("  raw scores, NOT probabilities — the biggest is class", int(logits.argmax()))

    # --- Part 6: softmax -> ten probabilities that add to 1 ----------------
    e = np.exp(logits - logits.max())               # subtract the max so exp cannot blow up
    probs = e / e.sum()                             # divide by the total -> slices of one pizza
    print("\nprobs: ", "  ".join("%.4f" % p for p in probs))
    print("  they add to %.12f, and the biggest slice is %.4f for class %d"
          % (probs.sum(), probs.max(), int(probs.argmax())))

    # --- Part 7: cross-entropy — how wrong was the guess? ------------------
    true_class = 0
    loss = -np.log(probs[true_class])
    print("\ncross-entropy with true_class = 0: -log(%.4f) = %.4f" % (probs[0], loss))
    # The lesson's small-loss / big-loss contrast, from THIS model's own probabilities
    # rather than two typed-in ones: its favourite class vs. its least liked.
    best, worst = int(probs.argmax()), int(probs.argmin())
    loss_best, loss_worst = -np.log(probs[best]), -np.log(probs[worst])
    print("  if the true class were %d (p = %.4f, its favourite):   loss %.4f — small"
          % (best, probs[best], loss_best))
    print("  if the true class were %d (p = %.4f, its least liked): loss %.4f — large"
          % (worst, probs[worst], loss_worst))
    print("  surprise ladder:", "  ".join("p=%.3f -> loss %.2f" % (p, -np.log(p))
                                          for p in (0.01, 0.30, 0.91, 0.999)))
    print("\nimage -> blocks (grid shrinks, channels grow) -> flatten -> dense head ->"
          " 10 logits -> softmax (probs add to 1) -> cross-entropy (how wrong);"
          " that is one forward pass and the loss training minimizes.")

    # --- Self-check: one boolean per claim ---------------------------------
    # Every expected number was read off a real run and typed in, so the code above
    # cannot quietly agree with itself.
    expected_logits = np.array([0.9524, 2.3005, -3.9897, -0.5339, 3.1951,
                                1.7214, -2.4551, -1.8453, 1.4541, -0.9565])
    shapes_ok = (x.shape == (3, 32, 32)
                 and actual == [(16, 16, 16), (32, 8, 8), (64, 4, 4)])
    # Two independent paths: the tape-measure formula vs. the padding-and-reshape code.
    prediction_ok = (pred_conv == actual_conv and pred_pool == actual
                     and predicted_flat == 1024)
    relu_ok = (int((pre1 < 0).sum()) == 2901          # the conv really made negatives
               and round(float(pre1.min()), 4) == -1.0496
               and float(bent1.min()) == 0.0          # and the bend flattened every one
               and int((bent1 == 0).sum()) == 13925)  # 11024 were already 0, plus the 2901
    pad_ok = round(edge_val, 4) == -0.2169
    pool_ok = (round(pool_val, 4) == 0.5226                   # the window's max...
               and round(float(window.mean()), 4) == 0.2577)  # ...not its mean
    flat_ok = (flat.size == 1024 and hot == 308
               and round(float(flat[hot]), 4) == 0.7778)
    logits_ok = (logits.shape == (10,)
                 and np.array_equal(np.round(logits, 4), expected_logits))
    probs_ok = (round(float(probs.sum()), 12) == 1.0
                and int(logits.argmax()) == 4 and round(float(probs.max()), 4) == 0.5076
                and int(probs.argmin()) == 2 and round(float(probs.min()), 4) == 0.0004)
    loss_ok = (round(float(loss), 4) == 2.9207
               and round(float(loss_best), 4) == 0.678
               and round(float(loss_worst), 4) == 7.8628)

    if (shapes_ok and prediction_ok and relu_ok and pad_ok and pool_ok and flat_ok
            and logits_ok and probs_ok and loss_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected (3,32,32) -> (16,16,16) -> (32,8,8) -> (64,4,4) with "
              "the formula predicting the same, flatten 1024 (biggest 0.7778 at 308), block 1 "
              "making 2901 negatives (13925 zeros afterwards) with minimum -1.0496 that "
              "ReLU lifts to 0.0, top-edge value -0.2169, pooled 0.5226 (max) not 0.2577 "
              "(mean), logits [0.9524 2.3005 "
              "-3.9897 -0.5339 3.1951 1.7214 -2.4551 -1.8453 1.4541 -0.9565], probs summing "
              "to 1.0 with 0.5076 on class 4 / 0.0004 on class 2, losses 2.9207, 0.678, 7.8628")

    # These asserts stop the program if any single fact is wrong.
    assert shapes_ok, "the funnel must go (3,32,32) -> (16,16,16) -> (32,8,8) -> (64,4,4)"
    assert prediction_ok, "Day 2's formula must predict the shapes the layers actually give"
    assert relu_ok, "block 1: 2901 negatives, min -1.0496, and ReLU must flatten them (13925 zeros)"
    assert pad_ok, "the top-edge conv value should be -0.2169 with zero padding above"
    assert pool_ok, "the pool must keep 0.5226 (the window's max), not 0.2577 (its mean)"
    assert flat_ok, "flatten should give 1024 numbers, biggest 0.7778 at position 308"
    assert logits_ok, "the head should give the 10 pinned logits, biggest 3.1951 on class 4"
    assert probs_ok, "probs must add to 1 with 0.5076 on class 4 and 0.0004 on class 2"
    assert loss_ok, "losses should be 2.9207 for class 0, 0.678 (best) and 7.8628 (worst)"
