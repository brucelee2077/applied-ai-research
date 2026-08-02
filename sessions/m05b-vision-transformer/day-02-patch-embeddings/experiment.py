# day-02-patch-embeddings — experiment
#
# Today's big idea in two lines of output:
#   One shared matrix W turns every flattened tile into a length-D "passport", and
#   those passports line up as a sequence: prepend [CLS] (N -> N+1), then ADD a
#   position stamp so two identical tiles in different slots stop looking the same.
# Run it:  python3 sessions/m05b-vision-transformer/day-02-patch-embeddings/experiment.py

import numpy as np  # numpy gives us arrays and matrix multiply (@)

# W, [CLS] and the position stamps start random in the lesson; one SEEDED generator
# for all three keeps the printed numbers identical on every run.
rng = np.random.default_rng(0)


def patchify(img, P):
    # Cut an image into non-overlapping P x P tiles, each flattened to length P*P; return
    # the (N, P*P) rows and the (rows, cols) grid. Same name as Day 1's patchify and the
    # SAME contract: whole tiles only, in reading order, a leftover strip thinner than P
    # dropped — so both days agree with (H//P)*(W//P) even when P does not divide evenly.
    # (One shared limit: P must be no bigger than H and W. With no whole tile at all the
    # formula says 0 but there is nothing to stack, so both days raise the same error.)
    # Day 1 returned (N, P, P) tiles you could still see as squares; here they arrive
    # already flattened into rows, which is the shape the projection needs.
    # W here is the patch matrix, so the image width is called W_img.
    H, W_img = img.shape
    flat_rows, rows, cols = [], 0, 0          # rows/cols are COUNTED by the walk below,
    for top in range(0, H, P):                # never taken from the formula, so the
        strip = []                            # "formula vs actually cut" check is real
        for left in range(0, W_img, P):
            # Reads rows top..top+P and columns left..left+P, stepping by exactly P.
            tile = img[top:top + P, left:left + P]
            if tile.shape != (P, P):           # a short edge strip is not a tile
                continue
            strip.append(tile.reshape(-1))     # flatten this tile into one row
        if not strip:                          # a short bottom strip: no tiles at all
            continue
        rows, cols = rows + 1, len(strip)      # one more row of tiles, this many across
        # The OUTER loop steps DOWN, the INNER loop runs ACROSS, exactly as on Day 1, so the
        # tiles come out in reading order: across a strip, then down to the next strip.
        flat_rows.extend(strip)
    return np.stack(flat_rows), (rows, cols)


def show(labels, matrix):
    # One labelled row per line, so the story reads straight off stdout. It RETURNS the
    # rounded rows it printed, so a check below can read the very numbers the reader saw
    # instead of rounding the matrix a second time by itself.
    rendered = np.round(matrix, 3)
    for label, row in zip(labels, rendered):
        print("  %-8s %s" % (label, row))
    return rendered


if __name__ == "__main__":
    P, D = 2, 8                              # patch size, and passport length D

    # --- Part 1: a fake grayscale picture ---------------------------------
    img = np.arange(4 * 4).reshape(4, 4)     # 16 numbers standing in for pixels
    H, W_img = img.shape
    # Every number that carries a claim is BOUND here and printed from the binding, so
    # the line you read and the line the self-check reads are one value, not two.
    shown_img_shape = img.shape
    print("Part 1 · the picture — img shape:", shown_img_shape, "-> H =", H, "down, W =", W_img, "across")
    print(img)

    # --- Part 2: cut into P x P tiles and flatten each --------------------
    # NAMING, kept straight for the rest of the module: on this day a "*_tiles" array
    # holds flattened PIXEL rows, width P*P. From Day 3 on, `tiles` means already-EMBEDDED
    # token vectors, width D — same word, different thing, so this one is `flat_tiles`.
    # And "grid" here is the (rows, cols) COUNT of tiles, not Day 1's grid of pixels and not
    # Day 3's grid of attention shares — so it is always printed as the "tile grid".
    flat_tiles, grid = patchify(img, P)
    n_from_formula = (H // P) * (W_img // P)  # the lesson's N = (H/P) x (W/P)
    n_from_tiles = flat_tiles.shape[0]        # how many tiles we actually cut
    shown_tiles_shape = flat_tiles.shape
    patch_len = P * P                         # one tile flattens to this many numbers
    tile_labels = ["tile %d" % i for i in range(n_from_tiles)]
    print("\nPart 2 · cut into P x P tiles, P =", P, " tile grid (rows, cols):", grid,
          " N from the formula (H//P)*(W//P):", n_from_formula, " N actually cut:", n_from_tiles,
          " tiles shape:", shown_tiles_shape, "-> each tile flattened to P*P =", patch_len)
    printed_tiles = show(tile_labels, flat_tiles)
    # No gaps and no overlaps: every pixel lands in exactly one tile. On its own this can NOT
    # prove the LAYOUT is right — a plain reshape of the whole image passes it too — so the
    # exact tile contents are pinned in the self-check below.
    every_pixel_once = np.array_equal(np.sort(flat_tiles.reshape(-1)), np.sort(img.reshape(-1)))
    print("every pixel used exactly once:", every_pixel_once)
    # The formula is two separate counts, so a non-square picture shows the order.
    wide_tiles, wide_grid = patchify(np.arange(4 * 6).reshape(4, 6), P)
    wide_n = wide_tiles.shape[0]
    shown_wide_tile1 = wide_tiles[1]
    print("a 4x6 picture -> tile grid:", wide_grid, " N =", wide_n, " tile 1 =", shown_wide_tile1)
    # Off the divisible path, RUN and not just claimed: on a 5x5 picture P = 2 divides
    # neither side, so the thin bottom strip and the thin right strip are not tiles. The
    # formula floors for the same reason, so it still matches the real cut — and Day 1's
    # patchify, same contract, gives the same 4 whole tiles on the same picture.
    odd_tiles, odd_grid = patchify(np.arange(5 * 5).reshape(5, 5), P)
    odd_formula = (5 // P) * (5 // P)
    odd_n = odd_tiles.shape[0]
    odd_dropped = 5 * 5 - odd_tiles.size          # pixels inside no whole tile
    shown_odd_last = odd_tiles[-1]
    print("a 5x5 picture (P divides neither side) -> tile grid:", odd_grid, " N from the formula:",
          odd_formula, " N actually cut:", odd_n, " last tile =", shown_odd_last,
          " pixels left out:", odd_dropped, "of 25")

    # --- Part 3: the passport machine — ONE shared linear projection ------
    W = rng.standard_normal((P * P, D))       # the same matrix for every tile
    patch_emb = flat_tiles @ W                # every flattened tile times that same W
    shown_W_shape, shown_emb_shape = W.shape, patch_emb.shape
    print("\nPart 3 · the passport machine — W shape:", shown_W_shape, "-> P*P =", patch_len,
          "in, D =", D, "out")
    print("patch_emb shape:", shown_emb_shape, "-> (N, D); one W for all = weight sharing")
    printed_emb = show(tile_labels, patch_emb)

    # --- Part 4: prepend the [CLS] leader token ---------------------------
    # This `cls` is the vector that goes IN, before any block runs. Day 3 calls the same
    # object `cls_token`; Day 4's `cls` is that token's vector coming OUT of the last block.
    # Same three letters, two ends of the pipeline — check which one a day means.
    cls = rng.standard_normal(D)              # a length-D vector, not from the picture
    seq = np.vstack([cls, patch_emb])         # PREPEND: leader first, then the tiles
    shown_cls_shape, shown_seq_shape = cls.shape, seq.shape
    n_tokens = n_from_tiles + 1               # N + 1: the tiles plus their leader
    print("\nPart 4 · pin [CLS] at the front — cls shape:", shown_cls_shape, " seq shape:",
          shown_seq_shape, "-> N+1 =", n_tokens, "tokens")
    show(["[CLS]"] + tile_labels, seq)

    # --- Part 5: ADD one learned position stamp per slot ------------------
    pos_emb = rng.standard_normal((n_from_tiles + 1, D))  # one stamp per slot
    seq_pos = seq + pos_emb                   # elementwise ADD, so the shape is unchanged
    shown_pos_shape, shown_seq_pos_shape = pos_emb.shape, seq_pos.shape
    print("\nPart 5 · ADD a position stamp — pos_emb shape:", shown_pos_shape, " seq+pos_emb shape:",
          shown_seq_pos_shape, "-> ADDED, not glued on, so the width stays", D)
    printed_seq_pos = show(["[CLS]"] + tile_labels, seq_pos)

    # --- Part 6: twin tiles — permutation-invariance, then the cure -------
    # This picture holds the SAME four pixels in slot 0 and in slot 3.
    twin_img = np.array([[1, 2, 0, 0], [3, 4, 0, 0],
                         [0, 0, 1, 2], [0, 0, 3, 4]])
    twin_tiles, _ = patchify(twin_img, P)              # flattened PIXEL rows again
    twins_are_twins = np.array_equal(twin_tiles[0], twin_tiles[3])
    twin_emb = twin_tiles @ W                          # the same shared W as before
    twin_seq = np.vstack([cls, twin_emb]) + pos_emb    # same [CLS], same stamps
    # Slot 0 sits in token row 1 and slot 3 in row 4, because [CLS] took row 0.
    gap_before = float(np.abs(twin_emb[0] - twin_emb[3]).max())
    gap_after = float(np.abs(twin_seq[1] - twin_seq[4]).max())
    shown_gap_before, shown_gap_after = round(gap_before, 4), round(gap_after, 4)
    twin_pair = np.vstack([twin_seq[1], twin_seq[4]])  # the two twin tokens, side by side
    print("\nPart 6 · two tiles, identical pixels, different slots — same pixels:", twins_are_twins)
    show(["slot %d" % i for i in range(twin_tiles.shape[0])], twin_tiles)
    print("biggest gap BEFORE positions:", shown_gap_before, "-> identical passports")
    print("biggest gap AFTER  positions:", shown_gap_after, "-> different rows now: permutation-invariance cured")
    printed_twin_pair = show(["slot 0", "slot 3"], twin_pair)

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected number below was read off a real run and written down here.
    # Each claim reads the SAME name that was printed, so corrupting a printed value
    # breaks its claim instead of slipping past a green tick.
    expected_tiles = np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
    expected_emb = np.array(
        [[1.6366, -2.0606, -2.6942, -0.3972, -5.1278, 6.348, -8.9548, -0.3721],
         [1.199, -5.3002, -3.3237, 0.1368, -12.0217, 9.807, -12.1882, 0.3422],
         [-0.1138, -15.0191, -5.212, 1.7389, -32.7035, 20.1837, -21.8886, 2.4851],
         [-0.5514, -18.2587, -5.8415, 2.2729, -39.5974, 23.6427, -25.122, 3.1994]])
    expected_twin_pair = np.array(
        [[2.501, -1.9207, -1.9879, -1.5801, -7.4067, 5.5607, -8.5102, 0.0955],
         [1.7033, -3.8537, -0.5232, -1.6922, -8.0638, 5.8392, -7.1728, 1.7028]])
    grid_ok = grid == (2, 2) and n_from_formula == n_from_tiles == 4  # both routes to N
    tiles_ok = np.array_equal(flat_tiles, expected_tiles) and shown_tiles_shape == (4, 4)
    # A non-square picture: the grid is (down, across) = (2, 3), never (3, 2).
    wide_ok = (wide_grid == (2, 3) and wide_n == 6
               and np.array_equal(shown_wide_tile1, np.array([2, 3, 8, 9])))
    # The non-divisible picture: formula and real cut still agree at 4, with 9 pixels left
    # out. A guard that watched only the row strip would keep the thin right column.
    odd_ok = (odd_grid == (2, 2) and odd_formula == odd_n == 4 and odd_dropped == 9
              and np.array_equal(shown_odd_last, np.array([12, 13, 17, 18])))
    proj_ok = (shown_W_shape == (4, 8) and shown_emb_shape == (4, 8) and patch_len == 4
               and round(float(np.abs(patch_emb).sum()), 4) == 289.7406)
    # ALL FOUR passports pinned, IN ORDER, because the claim is "tile i's passport is row i".
    # A wrong cut, a transposed operand, a per-tile W, or a shuffle that hands tile 1 the
    # passport belonging to tile 2 all change these numbers. Pinning only the first and last
    # row would let that middle shuffle through.
    passports_ok = np.array_equal(np.round(patch_emb, 4), expected_emb)
    cls_first_ok = (shown_seq_shape == (5, 8) and shown_cls_shape == (8,) and n_tokens == 5
                    and np.array_equal(seq[0], cls) and np.array_equal(seq[1:], patch_emb))
    pos_shape_ok = shown_pos_shape == (5, 8) and shown_seq_pos_shape == (5, 8)
    token1_ok = np.array_equal(np.round(seq_pos[1], 4), np.array(
        [3.4382, -0.7455, -2.3369, -1.6056, -5.1323, 7.0045, -10.2431, 0.023]))
    twins_ok = twins_are_twins and np.array_equal(twin_tiles[0], np.array([1, 2, 3, 4]))
    before_ok = gap_before == 0.0 and shown_gap_before == 0.0   # one shared W, same passport
    after_ok = shown_gap_after == 1.933
    # The twin tokens themselves — the day's payoff — pinned, not just the gap between them.
    twin_pair_ok = np.array_equal(np.round(twin_pair, 4), expected_twin_pair)
    # And the rounded rows the reader actually SAW, straight from show()'s return value, so
    # a corrupted printer cannot show one thing while the checks read another.
    printed_ok = (np.array_equal(printed_tiles, expected_tiles)
                  and round(float(np.abs(printed_emb).sum()), 3) == 289.741
                  and round(float(np.abs(printed_twin_pair).sum()), 3) == 60.114
                  and np.array_equal(printed_seq_pos[1], np.array(
                      [3.438, -0.745, -2.337, -1.606, -5.132, 7.005, -10.243, 0.023])))
    # The picture the whole day rests on.
    img_ok = shown_img_shape == (4, 4)

    if (grid_ok and tiles_ok and wide_ok and odd_ok and every_pixel_once and proj_ok
            and passports_ok and cls_first_ok and pos_shape_ok and token1_ok
            and twins_ok and before_ok and after_ok and twin_pair_ok
            and printed_ok and img_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected a (2,2) grid with N = 4, tile 0 = [0 1 4 5], tile 3 = "
              "[10 11 14 15], a (2,3) grid with tile 1 = [2 3 8 9] on the 4x6 picture, a (2,2) "
              "grid with 4 tiles and 9 pixels left out on the 5x5 picture, patch_emb "
              "(4, 8) with row 0 = [1.6366 -2.0606 -2.6942 -0.3972 -5.1278 6.348 -8.9548 -0.3721], "
              "[CLS] in row 0 of a (5, 8) sequence, tile 0's token after positions = [3.4382 "
              "-0.7455 -2.3369 -1.6056 -5.1323 7.0045 -10.2431 0.023], and a twin gap of 0.0 "
              "before positions and 1.933 after, with the twin tokens", expected_twin_pair)

    assert img_ok, "the picture should be the 4x4 grid the whole day is cut from"
    assert grid_ok, "the formula (H//P)*(W//P) and the real cut should both give N = 4, grid (2,2)"
    assert tiles_ok, "the tiles are (4, 4): tile 0 = [0 1 4 5], tile 2 = [8 9 12 13] — a tile, not an image row"
    assert wide_ok, "a 4x6 picture gives a (2, 3) tile grid with N = 6, and its tile 1 is [2 3 8 9]"
    assert odd_ok, "a 5x5 picture gives a (2, 2) tile grid, N = 4 from formula and cut alike, and leaves 9 pixels out"
    assert every_pixel_once, "the tiles must cover every pixel exactly once — no gaps, no overlaps"
    assert proj_ok, "W and patch_emb should be (4, 8), P*P = 4, and |patch_emb| should sum to 289.7406"
    assert passports_ok, "every tile's passport is pinned IN ORDER: tile 0 starts 1.6366 -2.0606, tile 1 1.199 -5.3002, tile 2 -0.1138 -15.0191, tile 3 -0.5514 -18.2587"
    assert cls_first_ok, "[CLS] must be prepended: row 0 is the leader, rows 1..N are the tiles, N+1 = 5"
    assert pos_shape_ok, "adding pos_emb must keep the shape (5, 8) — added, not concatenated"
    assert token1_ok, "tile 0's token after positions should start 3.4382 -0.7455 -2.3369"
    assert twins_ok, "slots 0 and 3 of twin_img must both flatten to [1 2 3 4]"
    assert before_ok, "identical pixels through one shared W give identical passports (gap 0.0)"
    assert after_ok, "after adding positions the twin tokens should differ by 1.933"
    assert twin_pair_ok, "the two twin tokens should be %s" % expected_twin_pair
    assert printed_ok, "the rows printed above must be the rows the checks read"
