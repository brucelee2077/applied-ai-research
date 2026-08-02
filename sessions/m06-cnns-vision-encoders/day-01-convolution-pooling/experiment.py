# day-01-convolution-pooling — experiment
#
# Today's big idea in two lines of output:
#   A 9-weight stencil slides over a 6x6 image and turns it into a 4x4 feature map:
#   0 over the flat halves, 27 along the edge — then max pooling shrinks it to 2x2.
#
# Nothing here is random: every pixel and weight is hand-written, so runs match.
# Run it:  python3 sessions/m06-cnns-vision-encoders/day-01-convolution-pooling/experiment.py

import numpy as np  # numpy gives us grids of numbers and elementwise multiply (*)

def out_size(n, k_size, p, s):
    """The lesson's formula: output side = (N - K + 2P)/S + 1, one side at a time."""
    # `k_size` is the stencil's SIDE LENGTH — one number, e.g. 3 for a 3x3 stencil. It is
    # NOT the stencil itself; the weight grid is `k` below (day 2 spells the same pair
    # `k_size` and `kernel`). Keeping the two apart matters because K means the side
    # length in the formula and the whole 3x3 array in convolve().
    # Floor division, because the stencil can only stop on whole pixels.
    return (n - k_size + 2 * p) // s + 1

def convolve(img, k):
    """Slide kernel k over img with stride 1 and no padding: one dot product per stop."""
    # Here `k` is the WEIGHT GRID (a 3x3 array), the name the lesson's produce step uses.
    # Reading across the module: this same operation is `conv2d(img, k_size, pad, stride,
    # kernel=None)` on day 2 — where the weights are the LAST argument and optional,
    # because day 2's brief asks for a plain sum window — and `conv_same(x, kernels)` on
    # day 3, where the weights are a whole bank of filters. Same idea, three signatures.
    kh, kw = k.shape
    # Stride 1, padding 0 is the "valid" sweep — the stencil never hangs off the edge.
    out_h, out_w = out_size(img.shape[0], kh, 0, 1), out_size(img.shape[1], kw, 0, 1)
    fmap = np.zeros((out_h, out_w))
    for i in range(out_h):                     # walk down the rows of stops
        for j in range(out_w):                 # walk across the columns of stops
            patch = img[i:i + kh, j:j + kw]    # the receptive field at this one stop
            # Multiply each pixel by the weight sitting on top of it, add all 9.
            fmap[i, j] = np.sum(patch * k)
    return fmap

def max_pool(fmap, size=2):
    """Keep the BIGGEST value in each non-overlapping size x size window."""
    # `size` is a DIAL, not a constant: the lesson calls the pool size "how big each
    # group is". Day 3 pools stacks of feature maps with `max_pool_channels(x, size=2)`,
    # which keeps this same dial and adds a channel axis — the window is never baked in.
    out_h, out_w = fmap.shape[0] // size, fmap.shape[1] // size
    pooled = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            # Window (i, j) takes rows i*size..i*size+size-1 and those columns.
            window = fmap[i * size:(i + 1) * size, j * size:(j + 1) * size]
            pooled[i, j] = np.max(window)
    return pooled

def avg_pool(fmap, size=2):
    """The lesson's other summary: the MEAN of each window instead of the max."""
    out_h, out_w = fmap.shape[0] // size, fmap.shape[1] // size
    # Split each axis into (which window, where inside it), average inside a window.
    return fmap.reshape(out_h, size, out_w, size).mean(axis=(1, 3))

if __name__ == "__main__":
    # --- Part 1: a 6x6 fake image with one clean vertical edge -------------
    img = np.zeros((6, 6))        # start with an all-dark picture
    img[:, 3:] = 9                # make columns 3, 4, 5 bright
    # Bind every number BEFORE printing it, so the self-check reads the same value the
    # reader sees. A printed number that is computed twice can disagree with itself.
    img_shape = img.shape
    bright_pixels = int((img == 9).sum())
    img_shown = img.astype(int)
    print("Part 1 · the image — shape:", img_shape, " bright pixels:", bright_pixels)
    print(img_shown)
    print("  columns 0-2 dark (0), columns 3-5 bright (9) -> one vertical edge between")

    # --- Part 2: the hand-made edge kernel, plus the size formula ----------
    k = np.array([[-1, 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1]])    # right column minus left column
    k_shape = k.shape
    kernel_weights = int(k.size)  # 3x3 = 9, shared at every stop — printed twice, bound once
    print("\nPart 2 · the kernel (the stencil) and the output-size formula")
    print("  k shape:", k_shape, " weights:", kernel_weights)
    print(k)
    # Work the formula BEFORE sliding, so Part 3's size is a real prediction.
    predicted_side = out_size(6, 3, 0, 1)
    print("  output side = (N - K + 2P)/S + 1;  ours N=6 K=3 P=0 S=1 ->", predicted_side)
    # The lesson's three dial settings, same 32x32 image and same 3x3 kernel each time.
    dials = [out_size(32, 3, p, s) for p, s in ((0, 1), (1, 1), (1, 2))]
    print("  32x32 dials [valid S1, same S1, same S2]:", dials)

    # --- Part 3: slide the stencil and read the feature map ----------------
    print("\nPart 3 · slide it: one dot product per stop")
    patch = img[0:3, 1:4]         # one stop worked by hand, to show the dot product
    patch_shown = patch.astype(int).tolist()
    right_col = int(patch[:, 2].sum())
    left_col = int(patch[:, 0].sum())
    stop_dot = int(np.sum(patch * k))
    print("  stop (row 0, col 1) patch:", patch_shown, "-> right col",
          right_col, "- left col", left_col, "=",
          stop_dot)
    fmap = convolve(img, k)
    fmap_shape = fmap.shape
    predicted_shape = (predicted_side, predicted_side)
    fmap_shown = fmap.astype(int)
    fmap_row0 = fmap[0].astype(int).tolist()
    print("  fmap shape:", fmap_shape, " (predicted", predicted_shape, ")")
    print(fmap_shown)
    print("  every row:", fmap_row0, "-> 0 over flat halves, 27 at the edge")

    # --- Part 4: pool the map to shrink it ---------------------------------
    print("\nPart 4 · pool it: one number per 2x2 window (size=2 is the default)")
    pooled = max_pool(fmap)
    pooled_shape = pooled.shape
    pooled_count = pooled.size
    fmap_count = fmap.size
    pooled_shown = pooled.astype(int)
    print("  pooled shape:", pooled_shape, " values:", pooled_count, "— a quarter of", fmap_count)
    print(pooled_shown)
    averaged = avg_pool(fmap)
    averaged_row0 = averaged[0].tolist()
    print("  average pooling, same windows:", averaged_row0, "-> the mean, not the max")

    # --- Part 5: count the weights ----------------------------------------
    n_neurons = 10                          # the "say 10 neurons" from the lesson
    n_pixels = int(img.size)                # 36 pixels in the 6x6 image
    fc_weights = n_pixels * n_neurons       # one weight per pixel per neuron
    print("\nPart 5 · weight sharing in numbers")
    print("  conv:", kernel_weights, "weights · fully-connected:", n_pixels,
          "pixels x", n_neurons, "neurons =", fc_weights)
    print("  and the kernel stays", kernel_weights, "weights on ANY image size — weight sharing")

    # --- Part 6: shift the edge, and watch the map shift with it -----------
    # Translation equivariance: the bright half starts one column further left and
    # nothing else changes, so the feature map should move one column left too.
    img_shift = np.zeros((6, 6))
    img_shift[:, 2:] = 9
    fmap_shift = convolve(img_shift, k)
    fmap_shift_shape = fmap_shift.shape
    fmap_shift_row0 = fmap_shift[0].astype(int).tolist()
    print("\nPart 6 · shift the edge one column left — fmap_shift shape:", fmap_shift_shape)
    print("  rows now", fmap_shift_row0, "was", fmap_row0)
    pooled_shift = max_pool(fmap_shift)
    pooled_shift_shown = pooled_shift.astype(int).tolist()
    print("  pooled_shift:", pooled_shift_shown, "-> edge in the left window")
    # Average pooling the lopsided map too. On the map from Part 3 each window holds the
    # same pair of values whichever way you group it, so only this one pins the axes.
    averaged_shift = avg_pool(fmap_shift)
    averaged_shift_shown = averaged_shift.tolist()
    print("  averaged_shift:", averaged_shift_shown, "-> same windows, the mean of each")

    # --- Part 7: do the windows really line up? ----------------------------
    # Every map above is SQUARE and only holds two values, so pooling cannot show you
    # WHICH pixels ended up in a window: a 4x4 map at size 2 makes both window counts
    # and both window sides equal to 2, and swapping them changes nothing. Here is a
    # 4x6 map where every number is different, so any mis-grouped window shows up.
    grid = np.arange(1, 25, dtype=float).reshape(4, 6)
    grid_shape = grid.shape
    grid_shown = grid.astype(int)
    print("\nPart 7 · a 4x6 counting map — shape:", grid_shape, "(rows != cols on purpose)")
    print(grid_shown)
    grid_max = max_pool(grid)
    grid_avg = avg_pool(grid)
    grid_pooled_shape = grid_max.shape
    grid_max_shown = grid_max.astype(int).tolist()
    grid_avg_shown = grid_avg.tolist()
    print("  pooled shape:", grid_pooled_shape, "-> 4/2 rows x 6/2 cols, not a square")
    print("  max of each window: ", grid_max_shown, "-> the bottom-right of each 2x2")
    print("  mean of each window:", grid_avg_shown, "-> e.g. (1+2+7+8)/4 = 4.5")

    # --- Self-check: one boolean per claim --------------------------------
    # Values written down from the output above, never recomputed from the code they
    # check. array_equal pins the shape as well as the numbers. Each claim reads the
    # SAME bound name that was printed, so corrupting a printed number fails a claim.
    img_rows = np.tile([0, 0, 0, 9, 9, 9], (6, 1))
    img_ok = (np.array_equal(img, img_rows) and np.array_equal(img_shown, img_rows)
              and img_shape == (6, 6) and bright_pixels == 18)
    kernel_ok = (np.array_equal(k, np.tile([-1, 0, 1], (3, 1)))
                 and k_shape == (3, 3) and kernel_weights == 9)
    formula_ok = predicted_side == 4 and dials == [30, 32, 16]
    # The worked stop must agree with itself: right column minus left column IS the sum.
    stop_ok = (patch_shown == [[0, 0, 9], [0, 0, 9], [0, 0, 9]]
               and right_col == 27 and left_col == 0
               and stop_dot == 27 and stop_dot == right_col - left_col)
    fmap_rows = np.tile([0, 27, 27, 0], (4, 1))
    fmap_ok = (np.array_equal(fmap, fmap_rows) and np.array_equal(fmap_shown, fmap_rows)
               and fmap_shape == predicted_shape == (4, 4)
               and fmap_row0 == [0, 27, 27, 0])
    pooled_rows = np.tile([27, 27], (2, 1))
    pool_max_ok = (np.array_equal(pooled, pooled_rows)
                   and np.array_equal(pooled_shown, pooled_rows)
                   and pooled_shape == (2, 2) and pooled_count == 4 and fmap_count == 16)
    pool_avg_ok = (np.array_equal(averaged, np.tile([13.5, 13.5], (2, 1)))
                   and averaged_row0 == [13.5, 13.5])
    # Two halves of one claim: the shift must really change the map, and by one column.
    shift_equivariant = (not np.array_equal(fmap_shift, fmap)
                         and np.array_equal(fmap_shift[:, :-1], fmap[:, 1:])
                         and fmap_shift_shape == (4, 4)
                         and fmap_shift_row0 == [27, 27, 0, 0])
    # This pooled map is lopsided (27 left, 0 right), so a window in the wrong place or
    # on the wrong axis shows up here. Pooling the map above could not show that.
    pool_shift_ok = (np.array_equal(pooled_shift, np.array([[27, 0], [27, 0]]))
                     and pooled_shift_shown == [[27, 0], [27, 0]])
    # Same reason for average pooling: the Part-3 map holds {0, 27} in EVERY window, so
    # grouping the wrong pixels together still averages to 13.5. This one does not.
    pool_avg_shift_ok = (np.array_equal(averaged_shift, np.array([[27.0, 0.0], [27.0, 0.0]]))
                         and averaged_shift_shown == [[27.0, 0.0], [27.0, 0.0]])
    weights_ok = (kernel_weights == 9 and n_pixels == 36 and n_neurons == 10
                  and fc_weights == 360)
    # Part 7 pins the window GROUPING itself, which no square map above can do — and the
    # printed counting map too: it is what the learner hand-checks the pooled numbers
    # against ("the bottom-right of each 2x2", "(1+2+7+8)/4 = 4.5"), so a wrong cell on
    # screen would make those two lines argue with each other.
    pool_grid_ok = (grid_shape == (4, 6)
                    and grid_shown.tolist() == [[1, 2, 3, 4, 5, 6],
                                                [7, 8, 9, 10, 11, 12],
                                                [13, 14, 15, 16, 17, 18],
                                                [19, 20, 21, 22, 23, 24]]
                    and grid_max_shown == [[8, 10, 12], [20, 22, 24]]
                    and grid_avg_shown == [[4.5, 6.5, 8.5], [16.5, 18.5, 20.5]]
                    and grid_pooled_shape == (2, 3))

    if (img_ok and kernel_ok and formula_ok and stop_ok and fmap_ok and pool_max_ok
            and pool_avg_ok and shift_equivariant and pool_shift_ok
            and pool_avg_shift_ok and weights_ok and pool_grid_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected img rows [0 0 0 9 9 9]; the kernel rows [-1 0 1]; the "
              "formula to give 4 here and 30 / 32 / 16 on the lesson's 32x32 dials; the worked "
              "stop to read 27 - 0 = 27; a 4x4 feature map of rows [0 27 27 0]; "
              "max pooling 2x2 of 27 where average pooling gives 13.5; the shifted map to read "
              "[27 27 0 0] and pool to [[27 0] [27 0]] by max and by mean; 9 weights "
              "against 360; and the 4x6 counting map to pool to [[8 10 12] [20 22 24]] by max "
              "and [[4.5 6.5 8.5] [16.5 18.5 20.5]] by mean")

    assert img_ok, "img should be 6x6 with every row [0 0 0 9 9 9], 18 bright pixels"
    assert kernel_ok, "the kernel should be 3x3 with every row [-1 0 1] — 9 weights"
    assert formula_ok, "the formula should give 4, and 30 / 32 / 16 on the 32x32 dials"
    assert stop_ok, "the worked stop should be right col 27 - left col 0 = 27"
    assert fmap_ok, "the feature map should be 4x4 with every row [0 27 27 0]"
    assert pool_max_ok, "2x2 max pooling should give [[27 27] [27 27]], 4 values from 16"
    assert pool_avg_ok, "2x2 average pooling should give [[13.5 13.5] [13.5 13.5]]"
    assert shift_equivariant, "the shifted map must differ, and equal the original moved 1 left"
    assert pool_shift_ok, "pooling the shifted map should give [[27 0] [27 0]]"
    assert pool_avg_shift_ok, "average pooling the shifted map should also give [[27 0] [27 0]]"
    assert weights_ok, "9 kernel weights against 36 pixels x 10 neurons = 360"
    assert pool_grid_ok, ("the 4x6 counting map must PRINT 1..24 in order and pool to 2x3: "
                          "max [[8 10 12] [20 22 24]]")
