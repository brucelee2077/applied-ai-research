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
    """Cut an H x W image into non-overlapping P x P tiles, in reading order.

    WHOLE tiles only: a leftover strip thinner than P is not a tile and is dropped,
    which is exactly what (H//P) * (W//P) counts. Day 2 defines a patchify with the
    same name; it obeys this same contract, so the two days never disagree — not
    even on a picture P does not divide evenly.

    One honest limit on that agreement: P must be no bigger than H and W. If no whole
    tile fits at all the formula says 0, but there is nothing to stack, so both days
    raise the same error instead of returning an empty array.
    """
    H, W = img.shape                       # H = height in pixels, W = width
    tiles = []
    # The OUTER loop steps DOWN the image P pixels at a time, the INNER loop runs ACROSS,
    # so tiles come out in reading order: across a strip, then down to the next strip.
    # Day 2's loop nest is the same and is described with those same words.
    # The count comes from this walk, never from the formula.
    for top in range(0, H, P):
        for left in range(0, W, P):
            # One square block: rows top..top+P, columns left..left+P. Slicing
            # (not reshape) is what makes this a real 2-D cut of the grid.
            tile = img[top:top + P, left:left + P]
            if tile.shape != (P, P):       # a short edge strip: not a full tile, skip it
                continue
            tiles.append(tile)
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
    # Bind each number BEFORE printing it, then have the self-check read the SAME name.
    # The line you read and the line that is checked must be ONE value, not two
    # separately written expressions that happen to agree today.
    shown_img_shape = img.shape
    shown_top_row = img[0].tolist()
    shown_left_col = img[:, 0].tolist()
    print("Part 1 — the fake image")
    print("  img shape:", shown_img_shape, "-> H =", H, "rows, W =", W, "columns")
    print("  img =\n", img)
    print("  top row:", shown_top_row, " left column:", shown_left_col)

    # --- Part 2: predict the count, THEN cut with P = 4 -------------------
    P = 4
    print("\nPart 2 — predict with the formula, then patchify (P = 4)")
    predicted_4 = n_patches_formula(H, W, P)   # the prediction is COMPUTED here
    rows_4, cols_4 = H // P, W // P            # tiles DOWN and tiles ACROSS, both shown
    print("  formula (H//P) x (W//P) = (%d//%d) x (%d//%d) = %d x %d = %d tiles"
          % (H, P, W, P, rows_4, cols_4, predicted_4))
    patches4 = patchify(img, P)
    shown_patches4_shape = patches4.shape
    n_tiles_4 = patches4.shape[0]
    formula_agrees_4 = predicted_4 == n_tiles_4   # the verdict word, bound before it prints
    print("  patchify returned shape:", shown_patches4_shape, "-> tiles from the walk:",
          n_tiles_4, " agrees with the formula?", formula_agrees_4)
    # Each tile's top-left pixel says WHERE in the image that tile was cut from.
    shown_topleft_4 = patches4[:, 0, 0].tolist()
    print("  each tile's top-left pixel:", shown_topleft_4,
          "-> across the top strip (0, 4), then the strip below (32, 36)")
    # A NON-SQUARE picture is the only thing that can catch an (H, W) mix-up: on a square
    # image, reading height where you meant width changes absolutely nothing.
    wide = np.arange(4 * 6).reshape(4, 6)          # 4 rows DOWN, 6 columns ACROSS
    wide_patches = patchify(wide, 2)
    wide_predicted = n_patches_formula(4, 6, 2)
    shown_wide_shape = wide_patches.shape
    shown_wide_topleft = wide_patches[:, 0, 0].tolist()
    # The (down, across) factorisation is COMPUTED here in the same order as rows_4/cols_4
    # above instead of being typed into the sentence as prose. On THIS picture the two counts
    # differ, so swapping them prints "(3 down) x (2 across)" and the check below fails; on
    # the square 8x8 image the same swap changes nothing, which is why it is done here.
    wide_rows, wide_cols = 4 // 2, 6 // 2
    print("  a 4x6 picture at P = 2 -> formula", wide_predicted, "tiles ; patchify",
          shown_wide_shape, " top-left pixels:", shown_wide_topleft,
          "\n  -> (%d down) x (%d across), never (%d down) x (%d across)"
          % (wide_rows, wide_cols, wide_cols, wide_rows))
    # The leftover-strip rule, actually RUN and not just claimed in a comment: on a 5x5
    # picture P = 2 divides neither side, so the bottom pixel row AND the right pixel column
    # belong to no whole tile. The formula floors for exactly that reason, so the two still
    # agree — and Day 2's patchify, same contract, agrees with both.
    odd = np.arange(5 * 5).reshape(5, 5)
    odd_patches = patchify(odd, 2)
    odd_predicted = n_patches_formula(5, 5, 2)
    shown_odd_n = odd_patches.shape[0]
    odd_dropped = odd.size - odd_patches.size      # pixels inside no whole tile
    shown_odd_last = odd_patches[-1].tolist()
    print("  a 5x5 picture at P = 2 (P divides neither side) -> formula", odd_predicted,
          "tiles ; patchify cut", shown_odd_n, "-> last tile", shown_odd_last, ";",
          odd_dropped, "pixels of", odd.size,
          "left out\n  -> the thin bottom strip and thin right strip are not tiles")

    # --- Part 3: one tile becomes one word-vector -------------------------
    print("\nPart 3 — flatten a tile into its word-vector")
    tile = patches4[0]
    shown_tile_shape = tile.shape
    print("  tile 0 shape:", shown_tile_shape, "-> its %dx%d grid:\n" % (P, P), tile)
    flat0 = tile.reshape(-1)               # unroll row by row into one long strip
    shown_flat0_shape, shown_flat0_len = flat0.shape, flat0.shape[0]
    shown_flat0 = flat0.tolist()
    print("  tile.reshape(-1) -> shape", shown_flat0_shape, " length", shown_flat0_len,
          "\n  flat tile 0:", shown_flat0, "(row 1, row 2, row 3, row 4 in a line)")
    seq4 = patches4.reshape(patches4.shape[0], -1)   # every tile, flattened
    shown_seq4_shape = seq4.shape
    shown_seq4_row1 = seq4[1].tolist()
    print("  all tiles flattened:", shown_seq4_shape, "= (num_patches, P*P) -> a sequence")
    print("  sequence row 1:", shown_seq4_row1, "-> the tile RIGHT of tile 0")
    tile_sums = patches4.sum(axis=(1, 2))  # add up the pixels inside each tile
    shown_tile_sums = tile_sums.tolist()
    img_total = int(img.sum())             # the tile totals must add back up to this
    print("  total inside each tile:", shown_tile_sums, " image total:", img_total)
    # Sorting proves no pixel was dropped or used twice. It can NOT prove the layout is
    # right (a reshape passes it too) — the exact tile contents are pinned below.
    seen = np.sort(patches4.reshape(-1))
    tiling_ok = np.array_equal(seen, np.arange(H * W))    # no gaps, no overlaps
    print("  every pixel used exactly once?", tiling_ok)

    # --- Part 4: shrink P to 2 and watch the count jump -------------------
    P_small = 2
    print("\nPart 4 — same image, smaller tiles (P = 2)")
    predicted_2 = n_patches_formula(H, W, P_small)
    patches2 = patchify(img, P_small)
    rows_2, cols_2 = H // P_small, W // P_small
    shown_patches2_shape = patches2.shape
    print("  formula: (%d//%d) x (%d//%d) = %d x %d = %d tiles ; patchify returned %s"
          % (H, P_small, W, P_small, rows_2, cols_2, predicted_2, shown_patches2_shape))
    # The wrong guess, COMPUTED from the shapes and not typed in: count the extra
    # tiles ACROSS but forget the extra tiles DOWN -> (H//P) x (W//P_small) = 2 x 4.
    across_only = (H // P) * (W // P_small)
    n_tiles_2 = patches2.shape[0]
    tile_count_ratio = n_tiles_2 / n_tiles_4        # halve P -> 4x the tiles, not 2x
    print("  counting only the extra tiles ACROSS would give", across_only, "- reality:",
          n_tiles_2, " ratio:", tile_count_ratio,
          "\n  -> smaller tiles = far more tiles = longer sequence = more compute")
    shown_first_p2_tile = patches2[0].tolist()
    shown_topleft_2 = patches2[:, 0, 0].tolist()
    print("  first P=2 tile:", shown_first_p2_tile,
          " top-left pixel of each:", shown_topleft_2)

    # --- Part 5: a color tile carries 3 numbers per pixel -----------------
    print("\nPart 5 — flattening a color tile (P x P x 3)")
    P_c = 16                               # the real ViT patch size from the lesson
    k = np.arange(P_c * P_c).reshape(P_c, P_c)   # one channel plane: pixel k holds k
    # The three planes hold different numbers (k, 100+k, 200+k) so the flatten order
    # is visible, and axis=-1 stacks them side by side AT each pixel, giving (P,P,3).
    gray_tile, color_tile = k, np.stack([k, k + 100, k + 200], axis=-1)
    flat_gray, flat_color = gray_tile.reshape(-1), color_tile.reshape(-1)
    shown_gray_shape, shown_gray_len = gray_tile.shape, flat_gray.shape[0]
    shown_color_shape, shown_color_len = color_tile.shape, flat_color.shape[0]
    print("  GRAY tile", shown_gray_shape, "-> flattens to", shown_gray_len, "numbers")
    print("  COLOR tile", shown_color_shape, "-> flattens to", shown_color_len,
          "numbers = %d*%d*3 -> 3x longer than gray, because the channels stack up"
          % (P_c, P_c))
    first6 = flat_color[:6].tolist()       # the first two pixels' worth of numbers
    print("  first 6 flat color numbers:", first6, "-> pixel 0's R,G,B, then pixel 1's")

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected value below was READ OFF a real run, so a broken cut cannot agree.
    # Each claim reads the SAME `shown_*` name that was printed above, so corrupting a
    # printed line breaks the claim it belongs to instead of sliding past a green tick.
    img_layout_ok = (shown_img_shape == (8, 8) and shown_top_row == [0, 1, 2, 3, 4, 5, 6, 7]
                     and shown_left_col == [0, 8, 16, 24, 32, 40, 48, 56])
    counts_ok = ((predicted_4, n_tiles_4, predicted_2, n_tiles_2,
                  across_only) == (4, 4, 16, 16, 8))
    # The printed factorisation (2 x 2, 4 x 4) and the printed verdict word are claims too.
    factors_ok = ((rows_4, cols_4, rows_2, cols_2) == (2, 2, 4, 4)
                  and formula_agrees_4 is True)
    ratio_ok = tile_count_ratio == 4.0        # halving P quadruples the tile count
    shapes_ok = (shown_patches4_shape == (4, 4, 4) and shown_patches2_shape == (16, 2, 2)
                 and shown_seq4_shape == (4, 16) and shown_flat0_shape == (16,)
                 and shown_tile_shape == (4, 4) and shown_flat0_len == 16)
    # The load-bearing layout claims: tile 0 read row by row, exact numbers, and
    # tile 1 sitting to the RIGHT of tile 0 rather than below it.
    tile0_ok = shown_flat0 == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    order4_ok = (shown_topleft_4 == [0, 4, 32, 36] and shown_seq4_row1
                 == [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31])
    order2_ok = (shown_topleft_2 == [0, 2, 4, 6, 16, 18, 20, 22, 32, 34, 36, 38, 48, 50, 52, 54]
                 and shown_first_p2_tile == [[0, 1], [8, 9]])
    # The non-square guard: 4x6 at P=2 is 2 tiles down by 3 across = 6. A patchify that mixed
    # H up with W cannot even stack its tiles on this picture, let alone match these numbers,
    # and the printed (down, across) factorisation is pinned in that order too.
    wide_ok = (wide_predicted == 6 and shown_wide_shape == (6, 2, 2)
               and (wide_rows, wide_cols) == (2, 3)
               and shown_wide_topleft == [0, 2, 4, 12, 14, 16])
    # Off the divisible path, formula and real cut must STILL agree: 4 whole tiles, 9 pixels
    # left out. A guard that only watched the row strip would keep the thin right column.
    odd_ok = (odd_predicted == shown_odd_n == 4 and odd_dropped == 9
              and shown_odd_last == [[12, 13], [17, 18]])
    # The per-tile totals must also add back up to the whole-image total that was printed.
    sums_ok = (tile_sums.shape == (4,) and shown_tile_sums == [216, 280, 728, 792]
               and img_total == 2016 and sum(shown_tile_sums) == img_total)
    color_len_ok = (shown_color_shape == (16, 16, 3) and shown_gray_shape == (16, 16)
                    and (shown_gray_len, shown_color_len) == (256, 768))
    color_order_ok = first6 == [0, 100, 200, 1, 101, 201]

    if (img_layout_ok and counts_ok and factors_ok and ratio_ok and shapes_ok
            and tile0_ok and order4_ok
            and order2_ok and wide_ok and odd_ok and sums_ok and tiling_ok and color_len_ok
            and color_order_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected 4 tiles at P=4 and 16 at P=2, shapes (4,4,4)/(16,2,2)/"
              "(4,16), tile 0 = [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27], top-left pixels "
              "[0,4,32,36] and [0,2,4,6,...,54], 6 tiles of shape (2,2) on the 4x6 picture "
              "starting [0,2,4,12,14,16], 4 whole tiles and 9 pixels left out on the 5x5 "
              "picture, 8 tiles if only W shrank, tile totals "
              "[216,280,728,792] summing to 2016, a 4.0x tile-count ratio, every pixel used "
              "once, and 16x16 -> 256 gray / 768 color "
              "starting [0,100,200,1,101,201]")

    assert img_layout_ok, "img should be a row-major 8x8 grid with top row [0..7]"
    assert counts_ok, "formula and walk: 4 tiles at P=4, 16 at P=2, 8 if only W shrank"
    assert factors_ok, "the printed factorisation is 2 x 2 at P=4 and 4 x 4 at P=2, and they agree"
    assert ratio_ok, "halving P must quadruple the tile count: 16 / 4 = 4.0"
    assert shapes_ok, "expected shapes (4,4,4), (16,2,2), (4,16) and (16,)"
    assert tile0_ok, "tile 0 must flatten to [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]"
    assert order4_ok, "tile top-left pixels should be [0,4,32,36] — across, then down"
    assert order2_ok, "at P=2 the tile top-left pixels should be [0,2,4,6,16,...,52,54]"
    assert wide_ok, "a 4x6 picture at P=2 gives 6 tiles of shape (2,2), 2 down x 3 across, top-lefts [0,2,4,12,14,16]"
    assert odd_ok, "a 5x5 picture at P=2 gives 4 whole tiles (last [[12,13],[17,18]]) and leaves 9 pixels out"
    assert sums_ok, "the four per-tile totals should be [216, 280, 728, 792], summing to 2016"
    assert tiling_ok, "the tiles together must use every pixel exactly once"
    assert color_len_ok, "a 16x16 tile is 256 numbers gray and 768 numbers in color"
    assert color_order_ok, "a (P,P,3) tile flattens R,G,B per pixel: [0,100,200,1,101,201]"
