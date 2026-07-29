# day-01-convolution-pooling — experiment
#
# Today's big idea in two lines of output:
#   A 9-weight stencil slides over a 6x6 image and turns it into a 4x4 feature map:
#   0 over the flat halves, 27 along the edge — then max pooling shrinks it to 2x2.
#
# Nothing here is random: every pixel and weight is hand-written, so runs match.
# Run it:  python3 sessions/m06-cnns-vision-encoders/day-01-convolution-pooling/experiment.py

import numpy as np  # numpy gives us grids of numbers and elementwise multiply (*)

def out_size(n, k, p, s):
    """The lesson's formula: output side = (N - K + 2P)/S + 1, one side at a time."""
    # Floor division, because the stencil can only stop on whole pixels.
    return (n - k + 2 * p) // s + 1

def convolve(img, k):
    """Slide kernel k over img with stride 1 and no padding: one dot product per stop."""
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
    print("Part 1 · the image — shape:", img.shape, " bright pixels:", int((img == 9).sum()))
    print(img.astype(int))
    print("  columns 0-2 dark (0), columns 3-5 bright (9) -> one vertical edge between")

    # --- Part 2: the hand-made edge kernel, plus the size formula ----------
    k = np.array([[-1, 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1]])    # right column minus left column
    print("\nPart 2 · the kernel (the stencil) and the output-size formula")
    print("  k shape:", k.shape, " weights:", k.size)
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
    print("  stop (row 0, col 1) patch:", patch.astype(int).tolist(), "-> right col",
          int(patch[:, 2].sum()), "- left col", int(patch[:, 0].sum()), "=",
          int(np.sum(patch * k)))
    fmap = convolve(img, k)
    print("  fmap shape:", fmap.shape, " (predicted", (predicted_side, predicted_side), ")")
    print(fmap.astype(int))
    print("  every row:", fmap[0].astype(int).tolist(), "-> 0 over flat halves, 27 at the edge")

    # --- Part 4: pool the map to shrink it ---------------------------------
    print("\nPart 4 · pool it: one number per 2x2 window (size=2 is the default)")
    pooled = max_pool(fmap)
    print("  pooled shape:", pooled.shape, " values:", pooled.size, "— a quarter of", fmap.size)
    print(pooled.astype(int))
    averaged = avg_pool(fmap)
    print("  average pooling, same windows:", averaged[0].tolist(), "-> the mean, not the max")

    # --- Part 5: count the weights ----------------------------------------
    n_neurons = 10                          # the "say 10 neurons" from the lesson
    kernel_weights = int(k.size)            # 3x3 = 9, shared at every stop
    fc_weights = int(img.size) * n_neurons  # one weight per pixel per neuron
    print("\nPart 5 · weight sharing in numbers")
    print("  conv:", kernel_weights, "weights · fully-connected:", int(img.size),
          "pixels x", n_neurons, "neurons =", fc_weights)
    print("  and the kernel stays", kernel_weights, "weights on ANY image size — weight sharing")

    # --- Part 6: shift the edge, and watch the map shift with it -----------
    # Translation equivariance: the bright half starts one column further left and
    # nothing else changes, so the feature map should move one column left too.
    img_shift = np.zeros((6, 6))
    img_shift[:, 2:] = 9
    fmap_shift = convolve(img_shift, k)
    print("\nPart 6 · shift the edge one column left — fmap_shift shape:", fmap_shift.shape)
    print("  rows now", fmap_shift[0].astype(int).tolist(), "was", fmap[0].astype(int).tolist())
    pooled_shift = max_pool(fmap_shift)
    print("  pooled_shift:", pooled_shift.astype(int).tolist(), "-> edge in the left window")
    # Average pooling the lopsided map too. On the map from Part 3 each window holds the
    # same pair of values whichever way you group it, so only this one pins the axes.
    averaged_shift = avg_pool(fmap_shift)
    print("  averaged_shift:", averaged_shift.tolist(), "-> same windows, the mean of each")

    # --- Self-check: one boolean per claim --------------------------------
    # Values written down from the output above, never recomputed from the code they
    # check. array_equal pins the shape as well as the numbers.
    img_ok = np.array_equal(img, np.tile([0, 0, 0, 9, 9, 9], (6, 1)))
    formula_ok = predicted_side == 4 and dials == [30, 32, 16]
    fmap_ok = np.array_equal(fmap, np.tile([0, 27, 27, 0], (4, 1)))
    pool_max_ok = np.array_equal(pooled, np.tile([27, 27], (2, 1)))
    pool_avg_ok = np.array_equal(averaged, np.tile([13.5, 13.5], (2, 1)))
    # Two halves of one claim: the shift must really change the map, and by one column.
    shift_equivariant = (not np.array_equal(fmap_shift, fmap)
                         and np.array_equal(fmap_shift[:, :-1], fmap[:, 1:]))
    # This pooled map is lopsided (27 left, 0 right), so a window in the wrong place or
    # on the wrong axis shows up here. Pooling the map above could not show that.
    pool_shift_ok = np.array_equal(pooled_shift, np.array([[27, 0], [27, 0]]))
    # Same reason for average pooling: the Part-3 map holds {0, 27} in EVERY window, so
    # grouping the wrong pixels together still averages to 13.5. This one does not.
    pool_avg_shift_ok = np.array_equal(averaged_shift, np.array([[27.0, 0.0], [27.0, 0.0]]))
    weights_ok = kernel_weights == 9 and fc_weights == 360

    if (img_ok and formula_ok and fmap_ok and pool_max_ok and pool_avg_ok
            and shift_equivariant and pool_shift_ok and pool_avg_shift_ok and weights_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected img rows [0 0 0 9 9 9]; the formula to give 4 here and "
              "30 / 32 / 16 on the lesson's 32x32 dials; a 4x4 feature map of rows [0 27 27 0]; "
              "max pooling 2x2 of 27 where average pooling gives 13.5; the shifted map to read "
              "[27 27 0 0] and pool to [[27 0] [27 0]] by max and by mean; and 9 weights "
              "against 360")

    assert img_ok, "img should be 6x6 with every row [0 0 0 9 9 9]"
    assert formula_ok, "the formula should give 4, and 30 / 32 / 16 on the 32x32 dials"
    assert fmap_ok, "the feature map should be 4x4 with every row [0 27 27 0]"
    assert pool_max_ok, "2x2 max pooling should give [[27 27] [27 27]]"
    assert pool_avg_ok, "2x2 average pooling should give [[13.5 13.5] [13.5 13.5]]"
    assert shift_equivariant, "the shifted map must differ, and equal the original moved 1 left"
    assert pool_shift_ok, "pooling the shifted map should give [[27 0] [27 0]]"
    assert pool_avg_shift_ok, "average pooling the shifted map should also give [[27 0] [27 0]]"
    assert weights_ok, "9 kernel weights against 360 fully-connected ones"
