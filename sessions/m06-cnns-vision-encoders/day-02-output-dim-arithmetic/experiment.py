# day-02-output-dim-arithmetic — experiment
#
# Today's big idea in two lines of output:
#   O = ⌊(N − K + 2P)/S⌋ + 1 tells you a layer's output side before you run it,
#   and a real sliding window lands on exactly that many spots — every time.
#   Depth ignores that formula: it is simply the number of filters in the layer.
#
# Run it:  python3 sessions/m06-cnns-vision-encoders/day-02-output-dim-arithmetic/experiment.py

import numpy as np  # numpy gives us arrays, np.pad for the border, np.stack for depth


def out_size(N, K, P, S):
    """The tape-measure:  O = ⌊(N − K + 2P) / S⌋ + 1.

    N : input side — the height or width of the grid coming in (one dimension at a time)
    K : kernel size, the side of the stencil · P : padding pixels glued on EACH side
    S : stride — how many pixels the stencil jumps between stops
    Returns O, the output side. `//` is Python's floor divide, which is the ⌊ ⌋.
    """
    return (N - K + 2 * P) // S + 1


def stop_positions(padded_side, K, S):
    # Count the stops the slow, honest way, with no formula at all: start at 0 and keep
    # jumping S pixels while the whole K-wide stencil still fits inside. A SECOND,
    # independent route to the output side, so comparing it against out_size() is a real
    # test and not the formula quietly agreeing with itself.
    return list(range(0, padded_side - K + 1, S))


def conv2d(img, k_size, pad, stride, kernel=None):
    """Slide a k_size x k_size window over img and return the feature map.

    img / k_size / pad / stride : the picture, then K, P and S from the formula.
    kernel : optional k_size x k_size weights. None means "just add the window up",
             the plain sum-window version the lesson's produce step asks for.
    """
    padded = np.pad(img, pad)                                # a ring of P zeros: the "paper mat"
    rows = stop_positions(padded.shape[0], k_size, stride)   # stops going down the height
    cols = stop_positions(padded.shape[1], k_size, stride)   # stops going across the width
    fmap = np.zeros((len(rows), len(cols)))
    for i, top in enumerate(rows):
        for j, left in enumerate(cols):
            window = padded[top:top + k_size, left:left + k_size]   # this stop's patch
            # Cross-correlation, the operation the lesson teaches: lay the stencil on
            # the window and multiply position-by-position, with NO flip.
            fmap[i, j] = window.sum() if kernel is None else float((window * kernel).sum())
    return fmap


if __name__ == "__main__":
    # --- Part 1: the tape-measure on the four settings the lesson names ----
    settings = [(28, 3, 0, 1, "valid"),          # no border, single steps
                (28, 3, 1, 1, "same"),           # border holds the size
                (28, 3, 1, 2, "same+stride2"),   # border, but bigger steps
                (32, 5, 2, 1, "same 5x5")]       # P = (5−1)/2 is "same" for a 5x5
    predicted = []
    print("tape-measure  O = ⌊(N − K + 2P)/S⌋ + 1")
    for N, K, P, S, label in settings:
        predicted.append(out_size(N, K, P, S))
        print("  %-13s N=%2d K=%d P=%d S=%d  ->  ⌊(%d − %d + %d)/%d⌋ + 1 = %2d"
              % (label, N, K, P, S, N, K, 2 * P, S, predicted[-1]))
    # stop_positions() never divides by S — it steps by S — so when the two agree, the
    # formula really is counting the stencil's stops.
    stops = [len(stop_positions(N + 2 * P, K, S)) for N, K, P, S, _ in settings]
    print("counted stops (no formula used):", stops, "  formula says:", predicted)
    # The labels have to earn their names: "valid" means no border at all, and "same"
    # means P = (K−1)/2, which is only a whole number when K is odd.
    labels_honest = all((P == 0) if label == "valid" else (K % 2 == 1 and P == (K - 1) // 2)
                        for N, K, P, S, label in settings)
    print("labels honest (valid -> P=0; same -> odd K with P=(K−1)/2):", labels_honest)
    fN, fK, fP, fS = settings[2][:4]     # the stride-2 row, read back off the table
    exact_steps = (fN - fK + 2 * fP) / fS   # 13.5 — this division is NOT clean
    print("floor check: (%d − %d + %d)/%d =" % (fN, fK, 2 * fP, fS), exact_steps,
          "-> rounds down to", int(exact_steps), "so O =", int(exact_steps) + 1)
    # All four rows above happen to have (N − K + 2P) one short of a multiple of S, where
    # ⌊a/S⌋ + 1 and ⌊(a+1)/S⌋ agree by accident. Two stride-3 rows separate them, checked
    # against the stop counter again — so a floor in the wrong place cannot slip through.
    odd_stride = [(28, 3, 1, 3), (32, 5, 0, 3)]
    odd_pred = [out_size(N, K, P, S) for N, K, P, S in odd_stride]
    odd_stops = [len(stop_positions(N + 2 * P, K, S)) for N, K, P, S in odd_stride]
    print("stride-3 cross-check  formula:", odd_pred, " counted stops:", odd_stops)

    # --- Part 2: slide a real window over the lesson's 28x28 array ---------
    img_zeros = np.zeros((28, 28))
    fmap_zeros = conv2d(img_zeros, 3, 1, 2)   # K=3, P=1, S=2, plain sum window
    print("\nimg = np.zeros((28,28)) -> shape", img_zeros.shape,
          " padded with P=1 -> shape", np.pad(img_zeros, 1).shape, "(28 + 2P = 30)")
    print("ACTUAL feature-map shape:", fmap_zeros.shape,
          " formula predicted:", (predicted[2], predicted[2]))
    print("but every value in it is", float(fmap_zeros.max()), "- an all-zero picture proves"
          " the SHAPE and nothing else, so the value pins below use a real picture.")

    # --- Part 3: a lopsided picture through a lopsided stencil -------------
    # No image dataset is cached on this machine, so the picture is a stand-in. The
    # generator is SEEDED, so it is the same picture, and the same numbers, every run.
    rng = np.random.default_rng(0)
    img = np.round(rng.random((28, 28)), 3)
    img[3:7, 18:24] += 5.0            # one bright patch, off to the upper right
    kernel = np.array([[1., 2., 3.],  # deliberately lopsided: not its own mirror, so doing
                       [0., 1., 0.],  # a true convolution (flipping the stencil) instead of
                       [0., 0., -1.]])  # cross-correlation changes the numbers below
    # Guards on the test data: a left-right symmetric picture, or a stencil equal to its
    # own 180° flip, would make a flip bug or an axis-swap bug change nothing at all.
    img_lopsided = (not np.array_equal(img, img[:, ::-1])) and (not np.array_equal(img, img.T))
    kernel_lopsided = not np.array_equal(kernel, kernel[::-1, ::-1])
    print("\npicture differs from its mirror and its transpose:", img_lopsided,
          " stencil differs from its 180° flip:", kernel_lopsided)
    fmap = conv2d(img, 3, 1, 2, kernel=kernel)
    print("weighted feature map, K=3 P=1 S=2 -> shape", fmap.shape)
    print("  fmap[0,0]  =", round(float(fmap[0, 0]), 4), "(top-left: most of this window is padding zeros)")
    print("  fmap[3,11] =", round(float(fmap[3, 11]), 4), "(sits on the bright patch)")
    print("  fmap[0,11] =", round(float(fmap[0, 11]), 4), "(same column, top row: no bright patch)")
    print("  fmap.sum() =", round(float(fmap.sum()), 4), "(whole map, one number)")

    # --- Part 4: depth is the OTHER ruler ---------------------------------
    filters = np.round(rng.standard_normal((12, 3, 3)), 3)   # a "bank" of 12 stencils
    # The same 12 filters at two (P, S) settings. np.stack puts the filters on a NEW
    # first axis, so the result reads (depth, height, width).
    bank_a = np.stack([conv2d(img, 3, 1, 2, kernel=f) for f in filters])
    bank_b = np.stack([conv2d(img, 3, 1, 1, kernel=f) for f in filters])
    print("\n12 filters, K=3 P=1 S=2 -> stacked shape", bank_a.shape)
    print("12 filters, K=3 P=1 S=1 -> stacked shape", bank_b.shape)
    print("depth :", bank_a.shape[0], "and", bank_b.shape[0], "-> the same 12, from the filter count")
    print("H x W :", bank_a.shape[1:], "and", bank_b.shape[1:], "-> these DID move when S moved")
    print("  bank_a[0,0,0]    =", round(float(bank_a[0, 0, 0]), 4), "(filter 0, top-left stop)")
    print("  bank_a[11,13,13] =", round(float(bank_a[11, 13, 13]), 4), "(filter 11, last stop)")
    grid_totals = {round(float(g.sum()), 6) for g in bank_a}
    print("distinct grid totals in the stack:", len(grid_totals), "-> 12 different filter reports")
    print("\ntakeaway: predicted size == actual size, every time; depth = number of filters.")

    # --- Self-check: one boolean per claim --------------------------------
    # Every number below was read off a real run and written down here, so the code
    # above cannot quietly agree with itself.
    formula_pins = predicted == [26, 28, 14, 32]        # the four values the lesson states
    stops_agree = stops == predicted                    # counted stops == formula, all four
    # "13.5 is not whole" is true of those literals whatever the code does, so the second
    # half is the part that bites: out_size must land on the ROUNDED-DOWN count.
    floor_bites = (exact_steps != int(exact_steps)
                   and out_size(fN, fK, fP, fS) == int(exact_steps) + 1)
    odd_stride_ok = odd_pred == odd_stops == [10, 10]   # stride 3, both routes agree
    real_shape_ok = fmap_zeros.shape == (14, 14) and fmap.shape == (14, 14)
    zeros_are_zero = float(fmap_zeros.max()) == 0.0 and float(fmap_zeros.min()) == 0.0
    data_lopsided = img_lopsided and kernel_lopsided
    corner_ok = round(float(fmap[0, 0]), 4) == -0.013     # window is mostly padding zeros
    patch_ok = round(float(fmap[3, 11]), 4) == 38.764     # the bright patch lifts this one
    off_patch_ok = round(float(fmap[0, 11]), 4) == 0.432  # same column, away from the patch
    total_ok = round(float(fmap.sum()), 4) == 735.151     # whole weighted map
    bank_shapes_ok = bank_a.shape == (12, 14, 14) and bank_b.shape == (12, 28, 28)
    bank_values_ok = (round(float(bank_a[0, 0, 0]), 4) == -0.4269
                      and round(float(bank_a[11, 13, 13]), 4) == 0.4472)
    grids_distinct = len(grid_totals) == 12              # 12 filters -> 12 different maps

    if (formula_pins and stops_agree and labels_honest and floor_bites and odd_stride_ok
            and real_shape_ok
            and zeros_are_zero and data_lopsided and corner_ok and patch_ok and off_patch_ok
            and total_ok and bank_shapes_ok and bank_values_ok and grids_distinct):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected predictions [26, 28, 14, 32] matching the counted stops, "
              "every named row earning its label, the real map at (14, 14), fmap[0,0] == -0.013, "
              "fmap[3,11] == 38.764, fmap[0,11] == 0.432, fmap.sum() == 735.151, the 12 filters "
              "stacking to (12, 14, 14) and (12, 28, 28) with bank_a[0,0,0] == -0.4269 and "
              "bank_a[11,13,13] == 0.4472, and all 12 grid totals different")

    # These asserts make the check hard: a wrong fact stops the program.
    assert formula_pins, "out_size should give [26, 28, 14, 32] for the four settings"
    assert stops_agree, "the counted stops must equal the formula for all four settings"
    assert labels_honest, "a \"same\" row needs an odd K with P = (K−1)/2; \"valid\" needs P = 0"
    assert floor_bites, "(28−3+2)/2 must be 13.5 and out_size must round it DOWN to 14"
    assert odd_stride_ok, "at stride 3 the formula and the counted stops must both give 10"
    assert real_shape_ok, "the real feature map must come out (14, 14) for K=3, P=1, S=2"
    assert zeros_are_zero, "an all-zero picture must give an all-zero map"
    assert data_lopsided, "picture and stencil must both be lopsided, or a flip proves nothing"
    assert corner_ok, "fmap[0,0] should be -0.013"
    assert patch_ok, "fmap[3,11] should be 38.764"
    assert off_patch_ok, "fmap[0,11] should be 0.432"
    assert total_ok, "fmap.sum() should be 735.151"
    assert bank_shapes_ok, "12 filters must stack to (12, 14, 14) and (12, 28, 28)"
    assert bank_values_ok, "bank_a[0,0,0] should be -0.4269 and bank_a[11,13,13] should be 0.4472"
    assert grids_distinct, "the 12 stacked grids must all be different maps"
