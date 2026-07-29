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
    # the (N, P*P) tiles and the (rows, cols) grid. Same name as Day 1's patchify, but
    # Day 1 returned (N, P, P) tiles you could still see as squares; here they arrive
    # already flattened into rows, which is the shape the projection needs.
    # W here is the patch matrix, so the image width is called W_img.
    H, W_img = img.shape
    rows, cols = H // P, W_img // P          # how many tiles fit down, and across
    tiles = []
    for r in range(rows):                    # reading order: across, then down
        for c in range(cols):
            # Reads rows r*P..r*P+P and columns c*P..c*P+P, stepping by exactly P.
            tiles.append(img[r * P:r * P + P, c * P:c * P + P].reshape(-1))
    return np.stack(tiles), (rows, cols)


def show(labels, matrix):
    # One labelled row per line, so the story reads straight off stdout.
    for label, row in zip(labels, matrix):
        print("  %-8s %s" % (label, np.round(row, 3)))


if __name__ == "__main__":
    P, D = 2, 8                              # patch size, and passport length D

    # --- Part 1: a fake grayscale picture ---------------------------------
    img = np.arange(4 * 4).reshape(4, 4)     # 16 numbers standing in for pixels
    H, W_img = img.shape
    print("Part 1 · the picture — img shape:", img.shape, "-> H =", H, "down, W =", W_img, "across")
    print(img)

    # --- Part 2: cut into P x P tiles and flatten each --------------------
    tiles, grid = patchify(img, P)
    n_from_formula = (H // P) * (W_img // P)  # the lesson's N = (H/P) x (W/P)
    n_from_tiles = tiles.shape[0]             # how many tiles we actually cut
    tile_labels = ["tile %d" % i for i in range(n_from_tiles)]
    print("\nPart 2 · cut into P x P tiles, P =", P, " tile grid (rows, cols):", grid,
          " N from the formula (H//P)*(W//P):", n_from_formula, " N actually cut:", n_from_tiles,
          " tiles shape:", tiles.shape, "-> each tile flattened to P*P =", P * P)
    show(tile_labels, tiles)
    # No gaps and no overlaps: every pixel lands in exactly one tile. On its own this can NOT
    # prove the LAYOUT is right — a plain reshape of the whole image passes it too — so the
    # exact tile contents are pinned in the self-check below.
    every_pixel_once = np.array_equal(np.sort(tiles.reshape(-1)), np.sort(img.reshape(-1)))
    print("every pixel used exactly once:", every_pixel_once)
    # The formula is two separate counts, so a non-square picture shows the order.
    wide_tiles, wide_grid = patchify(np.arange(4 * 6).reshape(4, 6), P)
    print("a 4x6 picture -> grid:", wide_grid, " N =", wide_tiles.shape[0], " tile 1 =", wide_tiles[1])

    # --- Part 3: the passport machine — ONE shared linear projection ------
    W = rng.standard_normal((P * P, D))       # the same matrix for every tile
    patch_emb = tiles @ W                     # every flattened tile times that same W
    print("\nPart 3 · the passport machine — W shape:", W.shape, "-> P*P =", P * P, "in, D =", D, "out")
    print("patch_emb shape:", patch_emb.shape, "-> (N, D); one W for all = weight sharing")
    show(tile_labels, patch_emb)

    # --- Part 4: prepend the [CLS] leader token ---------------------------
    cls = rng.standard_normal(D)              # a length-D vector, not from the picture
    seq = np.vstack([cls, patch_emb])         # PREPEND: leader first, then the tiles
    print("\nPart 4 · pin [CLS] at the front — cls shape:", cls.shape, " seq shape:", seq.shape,
          "-> N+1 =", n_from_tiles + 1, "tokens")
    show(["[CLS]"] + tile_labels, seq)

    # --- Part 5: ADD one learned position stamp per slot ------------------
    pos_emb = rng.standard_normal((n_from_tiles + 1, D))  # one stamp per slot
    seq_pos = seq + pos_emb                   # elementwise ADD, so the shape is unchanged
    print("\nPart 5 · ADD a position stamp — pos_emb shape:", pos_emb.shape, " seq+pos_emb shape:",
          seq_pos.shape, "-> ADDED, not glued on, so the width stays", D)
    show(["[CLS]"] + tile_labels, seq_pos)

    # --- Part 6: twin tiles — permutation-invariance, then the cure -------
    # This picture holds the SAME four pixels in slot 0 and in slot 3.
    twin_img = np.array([[1, 2, 0, 0], [3, 4, 0, 0],
                         [0, 0, 1, 2], [0, 0, 3, 4]])
    twin_tiles, _ = patchify(twin_img, P)
    twins_are_twins = np.array_equal(twin_tiles[0], twin_tiles[3])
    twin_emb = twin_tiles @ W                          # the same shared W as before
    twin_seq = np.vstack([cls, twin_emb]) + pos_emb    # same [CLS], same stamps
    # Slot 0 sits in token row 1 and slot 3 in row 4, because [CLS] took row 0.
    gap_before = float(np.abs(twin_emb[0] - twin_emb[3]).max())
    gap_after = float(np.abs(twin_seq[1] - twin_seq[4]).max())
    print("\nPart 6 · two tiles, identical pixels, different slots — same pixels:", twins_are_twins)
    show(["slot %d" % i for i in range(twin_tiles.shape[0])], twin_tiles)
    print("biggest gap BEFORE positions:", round(gap_before, 4), "-> identical passports")
    print("biggest gap AFTER  positions:", round(gap_after, 4), "-> different rows now: permutation-invariance cured")
    show(["slot 0", "slot 3"], np.vstack([twin_seq[1], twin_seq[4]]))

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected number below was read off a real run and written down here.
    grid_ok = grid == (2, 2) and n_from_formula == n_from_tiles == 4  # both routes to N
    tiles_ok = np.array_equal(tiles, np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]))
    # A non-square picture: the grid is (down, across) = (2, 3), never (3, 2).
    wide_ok = wide_grid == (2, 3) and np.array_equal(wide_tiles[1], np.array([2, 3, 8, 9]))
    proj_ok = W.shape == (4, 8) and patch_emb.shape == (4, 8) and round(float(np.abs(patch_emb).sum()), 4) == 289.7406
    # ALL FOUR passports pinned, IN ORDER, because the claim is "tile i's passport is row i".
    # A wrong cut, a transposed operand, a per-tile W, or a shuffle that hands tile 1 the
    # passport belonging to tile 2 all change these numbers. Pinning only the first and last
    # row would let that middle shuffle through.
    passports_ok = np.array_equal(np.round(patch_emb, 4), np.array(
        [[1.6366, -2.0606, -2.6942, -0.3972, -5.1278, 6.348, -8.9548, -0.3721],
         [1.199, -5.3002, -3.3237, 0.1368, -12.0217, 9.807, -12.1882, 0.3422],
         [-0.1138, -15.0191, -5.212, 1.7389, -32.7035, 20.1837, -21.8886, 2.4851],
         [-0.5514, -18.2587, -5.8415, 2.2729, -39.5974, 23.6427, -25.122, 3.1994]]))
    cls_first_ok = seq.shape == (5, 8) and np.array_equal(seq[0], cls) and np.array_equal(seq[1:], patch_emb)
    pos_shape_ok = pos_emb.shape == (5, 8) and seq_pos.shape == (5, 8)
    token1_ok = np.array_equal(np.round(seq_pos[1], 4), np.array(
        [3.4382, -0.7455, -2.3369, -1.6056, -5.1323, 7.0045, -10.2431, 0.023]))
    twins_ok = twins_are_twins and np.array_equal(twin_tiles[0], np.array([1, 2, 3, 4]))
    before_ok = gap_before == 0.0     # one shared W: same pixels -> the same passport
    after_ok = round(gap_after, 4) == 1.933

    if (grid_ok and tiles_ok and wide_ok and every_pixel_once and proj_ok
            and passports_ok and cls_first_ok and pos_shape_ok and token1_ok
            and twins_ok and before_ok and after_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected a (2,2) grid with N = 4, tile 0 = [0 1 4 5], tile 3 = "
              "[10 11 14 15], a (2,3) grid with tile 1 = [2 3 8 9] on the 4x6 picture, patch_emb "
              "(4, 8) with row 0 = [1.6366 -2.0606 -2.6942 -0.3972 -5.1278 6.348 -8.9548 -0.3721], "
              "[CLS] in row 0 of a (5, 8) sequence, tile 0's token after positions = [3.4382 "
              "-0.7455 -2.3369 -1.6056 -5.1323 7.0045 -10.2431 0.023], and a twin gap of 0.0 "
              "before positions and 1.933 after")

    assert grid_ok, "the formula (H//P)*(W//P) and the real cut should both give N = 4, grid (2,2)"
    assert tiles_ok, "tile 0 should be [0 1 4 5] and tile 2 [8 9 12 13] — a tile, not an image row"
    assert wide_ok, "a 4x6 picture gives a (2, 3) tile grid, and its tile 1 is [2 3 8 9]"
    assert every_pixel_once, "the tiles must cover every pixel exactly once — no gaps, no overlaps"
    assert proj_ok, "W and patch_emb should be (4, 8), and |patch_emb| should sum to 289.7406"
    assert passports_ok, "every tile's passport is pinned IN ORDER: tile 0 starts 1.6366 -2.0606, tile 1 1.199 -5.3002, tile 2 -0.1138 -15.0191, tile 3 -0.5514 -18.2587"
    assert cls_first_ok, "[CLS] must be prepended: row 0 is the leader, rows 1..N are the tiles"
    assert pos_shape_ok, "adding pos_emb must keep the shape (5, 8) — added, not concatenated"
    assert token1_ok, "tile 0's token after positions should start 3.4382 -0.7455 -2.3369"
    assert twins_ok, "slots 0 and 3 of twin_img must both flatten to [1 2 3 4]"
    assert before_ok, "identical pixels through one shared W give identical passports (gap 0.0)"
    assert after_ok, "after adding positions the twin tokens should differ by 1.933"
