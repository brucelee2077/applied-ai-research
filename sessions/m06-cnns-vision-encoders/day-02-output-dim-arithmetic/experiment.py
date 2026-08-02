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


def label_honest(N, K, P, S, label):
    """Does this row earn the name it is printed under?

    "valid" means no border at all, so P must be 0. "same" means P = (K−1)/2, which is
    only a whole number when K is odd — an even K can never give "same" padding. Kept as
    a named function so the SAME test can be pointed at rows that should fail it.
    """
    if label == "valid":
        return P == 0
    return K % 2 == 1 and P == (K - 1) // 2


def pair(v):
    """One number means "both axes the same"; a (rows, cols) pair means they differ.

    Height and width are two independent tape-measures. Letting K, P and S differ per
    axis is what makes an accidental height/width swap show up as a wrong shape.
    """
    return (v, v) if isinstance(v, int) else (v[0], v[1])


def slide_window(img, k_size, pad, stride, kernel):
    """Slide a k_size x k_size window over img and return the feature map.

    img / k_size / pad / stride : the picture, then K, P and S from the formula. Each of
             K, P and S may be one number (same for both axes) or a (rows, cols) pair.
    kernel : weights the size of the window, or None for "just add the window up" — the
             plain sum-window version the lesson's produce step asks for.

    This is the shared engine, and it is deliberately NOT the name any call site uses.
    Two wrappers below split its two behaviours, because they are two different things:
    `conv2d(...)` REQUIRES a stencil (day 1 defined convolution as multiplying each pixel
    by the weight sitting on top of it), and `sum_window(...)` says out loud that there
    are no weights at all. One optional argument doing both jobs would let a plain
    unweighted sum print itself as "a convolution" whenever the argument was left off.
    Naming note for reading across the module: `k_size` is the window's SIDE LENGTH (the K
    of the formula) and `kernel` is the WEIGHT GRID — day 1 keeps the same split, calling
    the grid `k`, and day 3 calls a whole bank of them `kernels`. The weights do sit in a
    different POSITION on each day (second and required on day 1's `convolve(img, k)`,
    last here, second again as a 4-D bank in day 3's `conv_same(x, kernels)`).
    """
    kh, kw = pair(k_size)
    ph, pw = pair(pad)
    sh, sw = pair(stride)
    padded = np.pad(img, ((ph, ph), (pw, pw)))               # a ring of P zeros: the "paper mat"
    rows = stop_positions(padded.shape[0], kh, sh)           # stops going down the height
    cols = stop_positions(padded.shape[1], kw, sw)           # stops going across the width
    fmap = np.zeros((len(rows), len(cols)))
    for i, top in enumerate(rows):
        for j, left in enumerate(cols):
            window = padded[top:top + kh, left:left + kw]    # this stop's patch
            # Cross-correlation, the operation the lesson teaches: lay the stencil on
            # the window and multiply position-by-position, with NO flip.
            fmap[i, j] = window.sum() if kernel is None else float((window * kernel).sum())
    return fmap


def conv2d(img, k_size, pad, stride, kernel):
    """A real convolution: every pixel multiplied by the weight sitting on top of it.

    The stencil is REQUIRED — leaving it out is a TypeError, not a silent sum.
    """
    return slide_window(img, k_size, pad, stride, kernel)


def sum_window(img, k_size, pad, stride):
    """The produce step's step 3: add each window up, with no weights anywhere.

    Its own name, so nothing in this file calls an unweighted total "a convolution".
    Part 2b then shows the two meet: summing IS weighting by a stencil of ones.
    """
    return slide_window(img, k_size, pad, stride, None)


if __name__ == "__main__":
    # --- Part 1: the tape-measure on the four settings the lesson names ----
    settings = [(28, 3, 0, 1, "valid"),          # no border, single steps
                (28, 3, 1, 1, "same"),           # border holds the size
                (28, 3, 1, 2, "same+stride2"),   # border, but bigger steps
                (32, 5, 2, 1, "same 5x5")]       # P = (5−1)/2 is "same" for a 5x5
    predicted = []
    table_lines = []
    print("tape-measure  O = ⌊(N − K + 2P)/S⌋ + 1")
    for N, K, P, S, label in settings:
        predicted.append(out_size(N, K, P, S))
        # Render the row ONCE into a string, then print that string. The "+ 2P" term is
        # arithmetic done at the print, so it has to be pinned like any other number.
        table_lines.append("  %-13s N=%2d K=%d P=%d S=%d  ->  ⌊(%d − %d + %d)/%d⌋ + 1 = %2d"
                           % (label, N, K, P, S, N, K, 2 * P, S, predicted[-1]))
        print(table_lines[-1])
    # stop_positions() never divides by S — it steps by S — so when the two agree, the
    # formula really is counting the stencil's stops.
    stops = [len(stop_positions(N + 2 * P, K, S)) for N, K, P, S, _ in settings]
    print("counted stops (no formula used):", stops, "  formula says:", predicted)
    # The labels have to earn their names: "valid" means no border at all, and "same"
    # means P = (K−1)/2, which is only a whole number when K is odd.
    labels_honest = all(label_honest(*row) for row in settings)
    print("labels honest (valid -> P=0; same -> odd K with P=(K−1)/2):", labels_honest)
    # A test that only ever says True has proved nothing. Point the SAME test at rows
    # that are deliberately mislabeled. Row 2 is the sharp one: with K=4, P=2 equals
    # K//2, so a version that forgot the odd-K half would wave it through.
    mislabels = [(28, 3, 1, 1, "valid"),      # a padded row calling itself "valid"
                 (28, 4, 2, 1, "same"),       # even K: "same" padding is not a whole number
                 (28, 5, 1, 1, "same")]       # K=5 needs P=2, not P=1
    mislabel_verdicts = [label_honest(*row) for row in mislabels]
    print("  same test on 3 deliberately mislabeled rows:", mislabel_verdicts, "-> all False")
    fN, fK, fP, fS = settings[2][:4]     # the stride-2 row, read back off the table
    exact_steps = (fN - fK + 2 * fP) / fS   # 13.5 — this division is NOT clean
    floor_steps = int(exact_steps)          # the ⌊ ⌋ itself: 13
    floor_out = floor_steps + 1             # ... + 1 = the output side the formula must give
    floor_head = "floor check: (%d − %d + %d)/%d =" % (fN, fK, 2 * fP, fS)
    print(floor_head, exact_steps, "-> rounds down to", floor_steps, "so O =", floor_out)
    # All four rows above happen to have (N − K + 2P) one short of a multiple of S, where
    # ⌊a/S⌋ + 1 and ⌊(a+1)/S⌋ agree by accident. Two stride-3 rows separate them, checked
    # against the stop counter again — so a floor in the wrong place cannot slip through.
    odd_stride = [(28, 3, 1, 3), (32, 5, 0, 3)]
    odd_pred = [out_size(N, K, P, S) for N, K, P, S in odd_stride]
    odd_stops = [len(stop_positions(N + 2 * P, K, S)) for N, K, P, S in odd_stride]
    print("stride-3 cross-check  formula:", odd_pred, " counted stops:", odd_stops)

    # --- Part 2: slide a real window over the lesson's 28x28 array ---------
    img_zeros = np.zeros((28, 28))
    fmap_zeros = sum_window(img_zeros, 3, 1, 2)   # K=3, P=1, S=2, plain sum window
    img_zeros_shape = img_zeros.shape
    padded_zeros_shape = np.pad(img_zeros, 1).shape
    fmap_zeros_shape = fmap_zeros.shape
    predicted_shape_2 = (predicted[2], predicted[2])
    zeros_max = float(fmap_zeros.max())
    zeros_min = float(fmap_zeros.min())
    print("\nimg = np.zeros((28,28)) -> shape", img_zeros_shape,
          " padded with P=1 -> shape", padded_zeros_shape, "(28 + 2P = 30)")
    print("ACTUAL feature-map shape:", fmap_zeros_shape,
          " formula predicted:", predicted_shape_2)
    print("but every value in it is", zeros_max, "- an all-zero picture proves"
          " the SHAPE and nothing else, so the value pins below use a real picture.")

    # --- Part 2b: the plain sum window on a picture that is NOT blank -------
    # Step 3 of today's brief is a plain sum window (`sum_window`). On the all-zero
    # picture above every total is 0, so adding up the WRONG pixels would look exactly
    # the same. A 4x4 counting picture makes each window total unmistakable.
    counting = np.arange(1, 17, dtype=float).reshape(4, 4)
    sum_map = sum_window(counting, 2, 0, 2)  # K=2, P=0, S=2: four separate 2x2 windows
    sum_shape = sum_map.shape
    sum_shown = sum_map.astype(int).tolist()
    # Adding up IS weighting every pixel by 1, so a stencil of ones must land on the
    # same numbers — a second, independent route to the same map.
    ones_map = conv2d(counting, 2, 0, 2, kernel=np.ones((2, 2)))
    sum_matches_ones = np.array_equal(sum_map, ones_map)
    print("\n4x4 counting picture, PLAIN sum window, K=2 P=0 S=2 -> shape", sum_shape)
    print("  window totals:", sum_shown, "-> the top-left one is 1+2+5+6 = 14")
    print("  same map from a stencil of ones:", sum_matches_ones, "-> summing = weighting by 1")

    # --- Part 2c: two axes, two tape-measures ------------------------------
    # Every picture so far is square with the same K, P, S on both axes, which hides
    # height/width mix-ups: rows and cols come out as the same list. This picture is
    # 28 x 40 and gets a DIFFERENT K, P and S on each axis, so the two must be measured
    # separately — one call to out_size per axis, exactly as the formula intends.
    wide = np.arange(28 * 40, dtype=float).reshape(28, 40) / 1000.0
    wide[5:9, 30:36] += 5.0                  # a bright patch, off to the upper right again
    wide_shape = wide.shape
    wide_K, wide_P, wide_S = (3, 5), (1, 0), (2, 3)      # (rows, cols) for each dial
    wide_sides = [out_size(wide_shape[0], wide_K[0], wide_P[0], wide_S[0]),
                  out_size(wide_shape[1], wide_K[1], wide_P[1], wide_S[1])]
    wide_fmap = sum_window(wide, wide_K, wide_P, wide_S)  # sum window again, no stencil
    wide_fmap_shape = wide_fmap.shape
    wide_corner = round(float(wide_fmap[0, 0]), 4)
    wide_total = round(float(wide_fmap.sum()), 4)
    # The axis line does three unguarded things at the print site: it picks rows from
    # wide_sides[0] and cols from wide_sides[1], and it works out "+ 2P". wide_sides is
    # pinned as a LIST, which says nothing about which end of it reached the screen — a
    # shifted index would print "⌊(28 − 3 + 2)/2⌋ + 1 = 12" one line above an ACTUAL
    # shape of (14, 12). So render the whole line ONCE from the dials, pin that string,
    # and print the string: every number on it now has a check behind it.
    wide_axis_line = ("  rows: ⌊(%d − %d + %d)/%d⌋ + 1 = %d"
                      "  cols: ⌊(%d − %d + %d)/%d⌋ + 1 = %d"
                      % (wide_shape[0], wide_K[0], 2 * wide_P[0], wide_S[0], wide_sides[0],
                         wide_shape[1], wide_K[1], 2 * wide_P[1], wide_S[1], wide_sides[1]))
    print("\nnon-square picture", wide_shape, " K=(rows 3, cols 5) P=(1, 0) S=(2, 3)")
    print(wide_axis_line)
    print("  ACTUAL shape:", wide_fmap_shape, " formula predicted:", tuple(wide_sides))
    print("  wide_fmap[0,0] =", wide_corner, " whole map sum =", wide_total)

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
    fmap_shape = fmap.shape
    corner = round(float(fmap[0, 0]), 4)
    patch_val = round(float(fmap[3, 11]), 4)
    off_patch = round(float(fmap[0, 11]), 4)
    fmap_total = round(float(fmap.sum()), 4)
    print("weighted feature map, K=3 P=1 S=2 -> shape", fmap_shape)
    print("  fmap[0,0]  =", corner, "(top-left: most of this window is padding zeros)")
    print("  fmap[3,11] =", patch_val, "(sits on the bright patch)")
    print("  fmap[0,11] =", off_patch, "(same column, top row: no bright patch)")
    print("  fmap.sum() =", fmap_total, "(whole map, one number)")

    # --- Part 4: depth is the OTHER ruler ---------------------------------
    # One word for two: "depth" here is the same number days 3 and 4 call the CHANNEL
    # count — axis 0 of a stack of feature maps. Day 3's channels climbing 16 -> 32 -> 64
    # is exactly this ruler being turned up block by block, and day 4's one-thermostat-
    # per-channel reads down that same axis. Depth, channels, filters: one count.
    filters = np.round(rng.standard_normal((12, 3, 3)), 3)   # a "bank" of 12 stencils
    # The same 12 filters at two (P, S) settings. np.stack puts the filters on a NEW
    # first axis, so the result reads (depth, height, width).
    bank_a = np.stack([conv2d(img, 3, 1, 2, kernel=f) for f in filters])
    bank_b = np.stack([conv2d(img, 3, 1, 1, kernel=f) for f in filters])
    bank_a_shape, bank_b_shape = bank_a.shape, bank_b.shape
    depth_a, depth_b = bank_a_shape[0], bank_b_shape[0]
    hw_a, hw_b = bank_a_shape[1:], bank_b_shape[1:]
    bank_first = round(float(bank_a[0, 0, 0]), 4)
    bank_last = round(float(bank_a[11, 13, 13]), 4)
    print("\n12 filters, K=3 P=1 S=2 -> stacked shape", bank_a_shape)
    print("12 filters, K=3 P=1 S=1 -> stacked shape", bank_b_shape)
    print("depth :", depth_a, "and", depth_b, "-> the same 12, from the filter count")
    print("H x W :", hw_a, "and", hw_b, "-> these DID move when S moved")
    print("  bank_a[0,0,0]    =", bank_first, "(filter 0, top-left stop)")
    print("  bank_a[11,13,13] =", bank_last, "(filter 11, last stop)")
    grid_totals = {round(float(g.sum()), 6) for g in bank_a}
    n_distinct = len(grid_totals)
    print("distinct grid totals in the stack:", n_distinct, "-> 12 different filter reports")
    print("\ntakeaway: predicted size == actual size, every time; depth = number of filters.")

    # --- Self-check: one boolean per claim --------------------------------
    # Every number below was read off a real run and written down here, so the code
    # above cannot quietly agree with itself. Each claim reads the SAME name that was
    # printed, so a corrupted printed number breaks a claim instead of slipping past.
    formula_pins = predicted == [26, 28, 14, 32]        # the four values the lesson states
    stops_agree = stops == predicted                    # counted stops == formula, all four
    # The printed table is a rendered string, so pin the string itself — the "+ 2P" term
    # and the label column are arithmetic and formatting done at the print site.
    table_ok = table_lines == [
        "  valid         N=28 K=3 P=0 S=1  ->  ⌊(28 − 3 + 0)/1⌋ + 1 = 26",
        "  same          N=28 K=3 P=1 S=1  ->  ⌊(28 − 3 + 2)/1⌋ + 1 = 28",
        "  same+stride2  N=28 K=3 P=1 S=2  ->  ⌊(28 − 3 + 2)/2⌋ + 1 = 14",
        "  same 5x5      N=32 K=5 P=2 S=1  ->  ⌊(32 − 5 + 4)/1⌋ + 1 = 32",
    ]
    # "13.5 is not whole" is true of those literals whatever the code does, so the second
    # half is the part that bites: out_size must land on the ROUNDED-DOWN count.
    floor_bites = (exact_steps == 13.5 and exact_steps != floor_steps
                   and floor_steps == 13 and floor_out == 14
                   and out_size(fN, fK, fP, fS) == floor_out
                   and floor_head == "floor check: (28 − 3 + 2)/2 =")
    odd_stride_ok = odd_pred == odd_stops == [10, 10]   # stride 3, both routes agree
    # The label test is only worth printing if it can say False. These three rows are
    # deliberately mislabeled, and the SECOND one is the one a weakened test would pass:
    # with K=4, P=2 equals K//2 but not (K−1)//2, so dropping the odd-K half lets it in.
    guard_bites = labels_honest and mislabel_verdicts == [False, False, False]
    real_shape_ok = (fmap_zeros_shape == (14, 14) and fmap_shape == (14, 14)
                     and img_zeros_shape == (28, 28) and padded_zeros_shape == (30, 30)
                     and predicted_shape_2 == (14, 14)
                     and fmap_zeros_shape == predicted_shape_2)
    zeros_are_zero = zeros_max == 0.0 and zeros_min == 0.0
    data_lopsided = img_lopsided and kernel_lopsided
    corner_ok = corner == -0.013                          # window is mostly padding zeros
    patch_ok = patch_val == 38.764                        # the bright patch lifts this one
    off_patch_ok = off_patch == 0.432                     # same column, away from the patch
    total_ok = fmap_total == 735.151                      # whole weighted map
    bank_shapes_ok = (bank_a_shape == (12, 14, 14) and bank_b_shape == (12, 28, 28)
                      and depth_a == depth_b == 12
                      and hw_a == (14, 14) and hw_b == (28, 28))
    bank_values_ok = bank_first == -0.4269 and bank_last == 0.4472
    grids_distinct = n_distinct == 12                     # 12 filters -> 12 different maps
    # The plain sum window (`sum_window`) — the produce step's own step 3 — on a picture
    # that is NOT all zeros, so the summing itself is pinned and not just the shape.
    sum_window_ok = (sum_shape == (2, 2) and sum_shown == [[14, 22], [46, 54]]
                     and sum_matches_ones)
    # A non-square picture with a different K, P and S on each axis: rows and cols are
    # now different lengths, so a height/width mix-up changes the answer.
    wide_ok = (wide_shape == (28, 40) and wide_sides == [14, 12]
               and wide_fmap_shape == (14, 12)
               and wide_corner == 0.22 and wide_total == 1658.31)
    # ...and the line the learner actually READS for those two sides, character for
    # character. wide_ok pins the list; this pins which end of it went where.
    wide_axis_line_ok = wide_axis_line == ("  rows: ⌊(28 − 3 + 2)/2⌋ + 1 = 14"
                                           "  cols: ⌊(40 − 5 + 0)/3⌋ + 1 = 12")

    if (formula_pins and stops_agree and table_ok and labels_honest and floor_bites
            and odd_stride_ok and guard_bites
            and real_shape_ok
            and zeros_are_zero and data_lopsided and corner_ok and patch_ok and off_patch_ok
            and total_ok and bank_shapes_ok and bank_values_ok and grids_distinct
            and sum_window_ok and wide_ok and wide_axis_line_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected predictions [26, 28, 14, 32] matching the counted stops, "
              "every named row earning its label while three mislabeled rows are rejected, the "
              "real map at (14, 14), fmap[0,0] == -0.013, "
              "fmap[3,11] == 38.764, fmap[0,11] == 0.432, fmap.sum() == 735.151, the 12 filters "
              "stacking to (12, 14, 14) and (12, 28, 28) with bank_a[0,0,0] == -0.4269 and "
              "bank_a[11,13,13] == 0.4472, all 12 grid totals different, the sum window on the "
              "4x4 counting picture giving [[14, 22], [46, 54]], and the 28x40 picture at "
              "K=(3,5) P=(1,0) S=(2,3) giving (14, 12) with corner 0.22 and sum 1658.31")

    # These asserts make the check hard: a wrong fact stops the program.
    assert formula_pins, "out_size should give [26, 28, 14, 32] for the four settings"
    assert stops_agree, "the counted stops must equal the formula for all four settings"
    assert table_ok, "the printed tape-measure table must read exactly as written down"
    assert labels_honest, "a \"same\" row needs an odd K with P = (K−1)/2; \"valid\" needs P = 0"
    assert floor_bites, "(28−3+2)/2 must be 13.5 and out_size must round it DOWN to 14"
    assert odd_stride_ok, "at stride 3 the formula and the counted stops must both give 10"
    assert guard_bites, "the label test must reject all three deliberately mislabeled rows"
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
    assert sum_window_ok, "the sum window on the 4x4 counting picture gives [[14, 22], [46, 54]]"
    assert wide_ok, "28x40 at K=(3,5) P=(1,0) S=(2,3) must give a (14, 12) map, corner 0.22"
    assert wide_axis_line_ok, ("the two axis lines must read rows 14 and cols 12 — in that "
                              "order, next to their own dials")
