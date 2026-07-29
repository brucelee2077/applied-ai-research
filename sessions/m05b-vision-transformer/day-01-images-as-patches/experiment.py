# day-01-images-as-patches — experiment
#
# Today's big idea in two lines of output:
#   "Patchify" snaps a picture's grid of numbers into equal square tiles, and each
#   tile flattens into one word-vector a Transformer can read. Halve the patch size
#   P and the tile count does not double — it quadruples.
#
# It cuts an 8x8 fake image with P=4, re-cuts it with P=2, and flattens a color tile.
# Run it:  python3 sessions/m05b-vision-transformer/day-01-images-as-patches/experiment.py

import numpy as np  # numpy gives us grids (arrays), slicing, and reshape

def patchify(img, P):
    """Cut an H x W image into non-overlapping P x P tiles, in reading order."""
    H, W = img.shape                       # H = height in pixels, W = width
    tiles = []
    # Walk DOWN the image P pixels at a time, then ACROSS P at a time — the order you
    # read a page. The count comes from this walk, never from the formula.
    for top in range(0, H, P):
        for left in range(0, W, P):
            # One square block: rows top..top+P, columns left..left+P. Slicing
            # (not reshape) is what makes this a real 2-D cut of the grid.
            tiles.append(img[top:top + P, left:left + P])
    return np.stack(tiles)                 # (num_patches, P, P)

def n_patches_formula(H, W, P):
    # The lesson's formula: how many tiles fit down, times how many fit across.
    return (H // P) * (W // P)

if __name__ == "__main__":
    # --- Part 1: the picture is a grid of numbers -------------------------
    # No image dataset is cached here, so the "image" is a grid of counting numbers.
    # Every pixel value differs on purpose: a wrong cut cannot hide behind repeats.
    img = np.arange(8 * 8).reshape(8, 8)
    H, W = img.shape
    print("Part 1 — the fake image")
    print("  img shape:", img.shape, "-> H =", H, "rows, W =", W, "columns")
    print("  img =\n", img)
    print("  top row:", img[0].tolist(), " left column:", img[:, 0].tolist())

    # --- Part 2: predict the count, THEN cut with P = 4 -------------------
    P = 4
    print("\nPart 2 — predict with the formula, then patchify (P = 4)")
    predicted_4 = n_patches_formula(H, W, P)   # the prediction is COMPUTED here
    print("  formula (H//P) x (W//P) = (%d//%d) x (%d//%d) = %d x %d = %d tiles"
          % (H, P, W, P, H // P, W // P, predicted_4))
    patches4 = patchify(img, P)
    print("  patchify returned shape:", patches4.shape, "-> tiles from the walk:",
          patches4.shape[0], " agrees with the formula?", predicted_4 == patches4.shape[0])
    # Each tile's top-left pixel says WHERE in the image that tile was cut from.
    print("  each tile's top-left pixel:", patches4[:, 0, 0].tolist(),
          "-> across the top strip (0, 4), then the strip below (32, 36)")
    # A NON-SQUARE picture is the only thing that can catch an (H, W) mix-up: on a square
    # image, reading height where you meant width changes absolutely nothing.
    wide = np.arange(4 * 6).reshape(4, 6)          # 4 rows DOWN, 6 columns ACROSS
    wide_patches = patchify(wide, 2)
    wide_predicted = n_patches_formula(4, 6, 2)
    print("  a 4x6 picture at P = 2 -> formula", wide_predicted, "tiles ; patchify",
          wide_patches.shape, " top-left pixels:", wide_patches[:, 0, 0].tolist(),
          "\n  -> (2 down) x (3 across), never (3 down) x (2 across)")

    # --- Part 3: one tile becomes one word-vector -------------------------
    print("\nPart 3 — flatten a tile into its word-vector")
    tile = patches4[0]
    print("  tile 0 shape:", tile.shape, "-> its %dx%d grid:\n" % (P, P), tile)
    flat0 = tile.reshape(-1)               # unroll row by row into one long strip
    print("  tile.reshape(-1) -> shape", flat0.shape, " length", flat0.shape[0],
          "\n  flat tile 0:", flat0.tolist(), "(row 1, row 2, row 3, row 4 in a line)")
    seq4 = patches4.reshape(patches4.shape[0], -1)   # every tile, flattened
    print("  all tiles flattened:", seq4.shape, "= (num_patches, P*P) -> a sequence")
    print("  sequence row 1:", seq4[1].tolist(), "-> the tile RIGHT of tile 0")
    tile_sums = patches4.sum(axis=(1, 2))  # add up the pixels inside each tile
    print("  total inside each tile:", tile_sums.tolist(), " image total:", int(img.sum()))
    # Sorting proves no pixel was dropped or used twice. It can NOT prove the layout is
    # right (a reshape passes it too) — the exact tile contents are pinned below.
    seen = np.sort(patches4.reshape(-1))
    print("  every pixel used exactly once?", np.array_equal(seen, np.arange(H * W)))

    # --- Part 4: shrink P to 2 and watch the count jump -------------------
    P_small = 2
    print("\nPart 4 — same image, smaller tiles (P = 2)")
    predicted_2 = n_patches_formula(H, W, P_small)
    patches2 = patchify(img, P_small)
    print("  formula: (%d//%d) x (%d//%d) = %d x %d = %d tiles ; patchify returned %s"
          % (H, P_small, W, P_small, H // P_small, W // P_small, predicted_2, patches2.shape))
    # The wrong guess, COMPUTED from the shapes and not typed in: count the extra
    # tiles ACROSS but forget the extra tiles DOWN -> (H//P) x (W//P_small) = 2 x 4.
    across_only = (H // P) * (W // P_small)
    print("  counting only the extra tiles ACROSS would give", across_only, "- reality:",
          patches2.shape[0], " ratio:", patches2.shape[0] / patches4.shape[0],
          "\n  -> smaller tiles = far more tiles = longer sequence = more compute")
    print("  first P=2 tile:", patches2[0].tolist(),
          " top-left pixel of each:", patches2[:, 0, 0].tolist())

    # --- Part 5: a color tile carries 3 numbers per pixel -----------------
    print("\nPart 5 — flattening a color tile (P x P x 3)")
    P_c = 16                               # the real ViT patch size from the lesson
    k = np.arange(P_c * P_c).reshape(P_c, P_c)   # one channel plane: pixel k holds k
    # The three planes hold different numbers (k, 100+k, 200+k) so the flatten order
    # is visible, and axis=-1 stacks them side by side AT each pixel, giving (P,P,3).
    gray_tile, color_tile = k, np.stack([k, k + 100, k + 200], axis=-1)
    flat_gray, flat_color = gray_tile.reshape(-1), color_tile.reshape(-1)
    print("  GRAY tile", gray_tile.shape, "-> flattens to", flat_gray.shape[0], "numbers")
    print("  COLOR tile", color_tile.shape, "-> flattens to", flat_color.shape[0],
          "numbers = %d*%d*3 -> 3x longer than gray, because the channels stack up"
          % (P_c, P_c))
    first6 = flat_color[:6].tolist()       # the first two pixels' worth of numbers
    print("  first 6 flat color numbers:", first6, "-> pixel 0's R,G,B, then pixel 1's")

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected value below was READ OFF a real run, so a broken cut cannot agree.
    img_layout_ok = (img.shape == (8, 8) and img[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
                     and img[:, 0].tolist() == [0, 8, 16, 24, 32, 40, 48, 56])
    counts_ok = ((predicted_4, patches4.shape[0], predicted_2, patches2.shape[0],
                  across_only) == (4, 4, 16, 16, 8))
    shapes_ok = (patches4.shape == (4, 4, 4) and patches2.shape == (16, 2, 2)
                 and seq4.shape == (4, 16) and flat0.shape == (16,))
    # The load-bearing layout claims: tile 0 read row by row, exact numbers, and
    # tile 1 sitting to the RIGHT of tile 0 rather than below it.
    tile0_ok = flat0.tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    order4_ok = (patches4[:, 0, 0].tolist() == [0, 4, 32, 36] and seq4[1].tolist()
                 == [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31])
    order2_ok = patches2[:, 0, 0].tolist() == [0, 2, 4, 6, 16, 18, 20, 22, 32, 34, 36, 38, 48, 50, 52, 54]
    # The non-square guard: 4x6 at P=2 is 2 tiles down by 3 across = 6. A patchify that mixed
    # H up with W cannot even stack its tiles on this picture, let alone match these numbers.
    wide_ok = (wide_predicted == 6 and wide_patches.shape == (6, 2, 2)
               and wide_patches[:, 0, 0].tolist() == [0, 2, 4, 12, 14, 16])
    sums_ok = (tile_sums.shape == (4,) and tile_sums.tolist() == [216, 280, 728, 792])
    tiling_ok = np.array_equal(seen, np.arange(H * W))    # no gaps, no overlaps
    color_len_ok = color_tile.shape == (16, 16, 3) and (flat_gray.shape[0], flat_color.shape[0]) == (256, 768)
    color_order_ok = first6 == [0, 100, 200, 1, 101, 201]

    if (img_layout_ok and counts_ok and shapes_ok and tile0_ok and order4_ok
            and order2_ok and wide_ok and sums_ok and tiling_ok and color_len_ok
            and color_order_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected 4 tiles at P=4 and 16 at P=2, shapes (4,4,4)/(16,2,2)/"
              "(4,16), tile 0 = [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27], top-left pixels "
              "[0,4,32,36] and [0,2,4,6,...,54], 6 tiles of shape (2,2) on the 4x6 picture "
              "starting [0,2,4,12,14,16], 8 tiles if only W shrank, tile totals "
              "[216,280,728,792], every pixel used once, and 16x16 -> 256 gray / 768 color "
              "starting [0,100,200,1,101,201]")

    assert img_layout_ok, "img should be a row-major 8x8 grid with top row [0..7]"
    assert counts_ok, "formula and walk: 4 tiles at P=4, 16 at P=2, 8 if only W shrank"
    assert shapes_ok, "expected shapes (4,4,4), (16,2,2), (4,16) and (16,)"
    assert tile0_ok, "tile 0 must flatten to [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]"
    assert order4_ok, "tile top-left pixels should be [0,4,32,36] — across, then down"
    assert order2_ok, "at P=2 the tile top-left pixels should be [0,2,4,6,16,...,52,54]"
    assert wide_ok, "a 4x6 picture at P=2 gives 6 tiles of shape (2,2), top-lefts [0,2,4,12,14,16]"
    assert sums_ok, "the four per-tile totals should be [216, 280, 728, 792]"
    assert tiling_ok, "the tiles together must use every pixel exactly once"
    assert color_len_ok, "a 16x16 tile is 256 numbers gray and 768 numbers in color"
    assert color_order_ok, "a (P,P,3) tile flattens R,G,B per pixel: [0,100,200,1,101,201]"
