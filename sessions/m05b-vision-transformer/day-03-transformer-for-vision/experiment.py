# day-03-transformer-for-vision — experiment
#
# Today's big idea in two lines of output:
#   The SAME Transformer block that reads a sentence reads a line of picture
#   tiles: it hands back the exact same shape, so you can stack it in a relay —
#   and the whole picture's answer is read off ONE row, the [CLS] token.
#
# It builds a fake line of 4 tile tokens, prepends [CLS], ADDS Day 2's position
# stamp (the mechanism that keeps two look-alike tiles apart), runs the reused
# block (attention + feed-forward, each with a norm and a shortcut), stacks it,
# then reads 3 class scores off row 0 only.
# Run it:  python3 sessions/m05b-vision-transformer/day-03-transformer-for-vision/experiment.py

import numpy as np  # numpy gives us arrays and matrix multiply (@)

# The lesson writes np.random.randn(...); we SEED the generator so the numbers
# pinned in the self-check are the same on every run.
rng = np.random.default_rng(0)
N, D, C = 4, 8, 3        # 4 tile tokens, D numbers per token, 3 possible classes

# One block's learned numbers. Q, K, V use three DIFFERENT matrices, so a tile's
# question is never its own answer.
W_Q = rng.standard_normal((D, D)) * 0.5
W_K = rng.standard_normal((D, D)) * 0.5
W_V = rng.standard_normal((D, D)) * 0.5
W_1 = rng.standard_normal((D, 4 * D)) * 0.2      # feed-forward: widen D -> 4D
W_2 = rng.standard_normal((4 * D, D)) * 0.2      # ...then narrow 4D -> D again

def layer_norm(x):
    # Tidy each token's OWN D numbers: axis=-1 = one mean per ROW (axis=0 would
    # mix different tokens together).
    return (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)

def softmax(scores):
    # Subtract each row's largest score first so exp() never blows up.
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)   # each ROW divided by its own total -> 1

def block(x):
    """The text Transformer's block, nothing changed for vision: job 1 attention
    (every token looks at every other), job 2 a feed-forward layer (think per
    token), each wrapped in a norm and a shortcut. Output shape == input shape,
    which is why blocks stack. Returns (output, attention_weights)."""
    xn = layer_norm(x)
    Q, K, V = xn @ W_Q, xn @ W_K, xn @ W_V       # three recipes -> three tables
    scores = (Q @ K.T) / np.sqrt(D)              # row i = how token i rates each token
    attn = softmax(scores)                       # each row of shares adds up to 1
    looked = x + attn @ V                        # job 1 + its shortcut (add x back)
    thought = looked + np.maximum(0, layer_norm(looked) @ W_1) @ W_2   # job 2 + shortcut
    return thought, attn

if __name__ == "__main__":
    # --- Part 1: a fake line of tile tokens -------------------------------
    # Yesterday's patch embeddings would hand us exactly this: one vector per tile.
    # NAMING, so the module stays straight: from today on `tiles` holds already-EMBEDDED
    # token vectors, width D = 8 — what came OUT of Day 2's shared W. Day 2's `flat_tiles`
    # were raw flattened PIXEL rows, width P*P. Same English word, different thing.
    # Every printed number is BOUND to a name first and the self-check reads that same
    # name, so a corrupted printed line breaks its own claim.
    tiles = rng.standard_normal((N, D))
    shown_tiles_shape = tiles.shape
    shown_tile0 = np.round(tiles[0], 4)
    print("tiles shape:", shown_tiles_shape, "->", N, "tile tokens,", D, "numbers each")
    print("tile 0:", shown_tile0)

    # --- Part 2: prepend [CLS], then ADD Day 2's position stamp -----------
    # Predict first, but COMPUTE the prediction from the input instead of typing it.
    predicted_len = tiles.shape[0] + 1
    # Day 2 called this same object `cls` — the [CLS] vector that goes IN, before block 1.
    # (Day 4's `cls` is this token's vector coming OUT of the last block, a different end
    # of the same pipeline.)
    cls_token = rng.standard_normal(D)                   # the captain's own vector
    seq_nopos = np.vstack([cls_token[None, :], tiles])   # [CLS] leads the line
    # Day 2's payoff, CARRIED here instead of quietly dropped: the ADDED stamp is what
    # stops two identical tiles in different slots from looking the same, and a real ViT
    # adds it before block 1. Real stamps start SMALL (std ~ 0.02) so they nudge rather
    # than swamp; 0.5 here is big enough that you can watch it change who attends to whom.
    pos_emb = rng.standard_normal((N + 1, D)) * 0.5
    seq = seq_nopos + pos_emb                            # elementwise ADD: same shape out
    shown_nopos_shape, shown_pos_shape = seq_nopos.shape, pos_emb.shape
    shown_seq_shape = seq.shape
    cls_leads = np.array_equal(seq_nopos[0], cls_token)
    tiles_follow = np.array_equal(seq_nopos[1:], tiles)
    stamp_reach = round(float(np.abs(pos_emb).max()), 4)
    print("\npredicted sequence length:", predicted_len, "= N + 1  ->  seq shape:", shown_seq_shape)
    print("before the stamp — row 0 is the [CLS] vector?", cls_leads,
          " rows 1..%d are the tiles?" % N, tiles_follow)
    print("position stamp:", shown_pos_shape, "ADDED, not glued on -> shape stays",
          shown_seq_shape, "· biggest stamp number:", stamp_reach)
    # Proof the stamp is working and not just riding along: copy tile 0 into slot 3, so two
    # tokens hold identical numbers. WITHOUT the stamp the block hands back identical rows
    # for them (it cannot tell the slots apart); WITH it, the rows come out different.
    twin = tiles.copy()
    twin[3] = twin[0]
    twin_nopos = np.vstack([cls_token[None, :], twin])
    twin_out_nopos, _ = block(twin_nopos)
    twin_out_pos, _ = block(twin_nopos + pos_emb)
    twin_gap_nopos = round(float(np.abs(twin_out_nopos[1] - twin_out_nopos[4]).max()), 4)
    twin_gap_pos = round(float(np.abs(twin_out_pos[1] - twin_out_pos[4]).max()), 4)
    print("two IDENTICAL tile tokens in slots 0 and 3, after the block — gap without the stamp:",
          twin_gap_nopos, " with the stamp:", twin_gap_pos,
          "\n  -> Day 2's cure, still curing: the stamp keeps look-alike tiles apart")

    # --- Part 3: run the reused block, then stack it ----------------------
    out, attn = block(seq)
    shown_out_shape = out.shape
    printed_attn = np.round(attn, 4)             # the shares, rounded ONCE, printed below
    # Fixed 4-decimal columns: a tiny share like 0.0004 would otherwise print as 4.0e-04.
    attn_lines = ["  %-7s [%s]" % (name, " ".join("%.4f" % v for v in printed_attn[i]))
                  for i, name in enumerate(["[CLS]"] + ["tile %d" % t for t in range(N)])]
    print("\nblock in:", shown_seq_shape, "-> block out:", shown_out_shape,
          "(the SAME shape -> stackable)")
    print("attention shares, row i = how much token i took from each token:")
    for line in attn_lines:
        print(line)
    rows_sum_one = np.allclose(attn.sum(axis=1), 1.0)
    cols_sum_one = np.allclose(attn.sum(axis=0), 1.0)
    print("every ROW adds up to 1?", rows_sum_one, " every COLUMN?", cols_sum_one,
          "-> rows: the softmax runs along axis -1")
    # The day's headline, in numbers: no share is zero, so EVERY token took something from
    # EVERY other token — one global look, from block 1. Day 4 leans on this.
    min_share = round(float(attn.min()), 4)
    all_nonzero = bool((attn > 0).all())
    print("smallest share anywhere:", min_share, " is every share above zero?", all_nonzero,
          "-> every token sees every other token already in block 1")
    lopsided = round(float(np.abs(attn - attn.T).max()), 4)
    moved = round(float(np.abs(out - seq).max()), 4)
    # "SHARE grid" on purpose: on this day a grid is the square table of attention shares,
    # not Day 1's grid of pixels and not Day 2's (rows, cols) count of tiles.
    print("biggest gap between share[i][j] and share[j][i]:", lopsided,
          "· furthest any number moved inside the block:", moved,
          "-> lopsided share grid (reading it the wrong way round changes the answer), real mixing")
    relayed = seq                                 # a relay: the same block, again and again
    relay_shapes = []
    for depth in range(3):
        relayed, _ = block(relayed)
        relay_shapes.append(relayed.shape)
        print("after block", depth + 1, "-> shape", relay_shapes[-1])
    # The shape ALONE proves nothing about the relay: `relayed = seq` already has shape
    # (5, 8), so a loop that never called block() would pass a shape-only check. These two
    # distances are the evidence that the block really ran three times and each round moved
    # the numbers on: away from the input, and past the 1-block output.
    relay_moved = round(float(np.abs(relayed - seq).max()), 4)
    relay_past_one = round(float(np.abs(relayed - out).max()), 4)
    shown_relayed_shape = relayed.shape
    print("3 blocks moved a number by", relay_moved, "from the input, and sit", relay_past_one,
          "from the 1-block output -> the relay advanced, it did not just keep the shape")

    # --- Part 4: read the answer off row 0 only ---------------------------
    W_head = rng.standard_normal((D, C))          # the classification head
    # NAMING across the last two days: Day 4 writes this same head as a bare `W`, and calls
    # its output `logits` where this day says "class scores". One object, two names each —
    # the numbers below are what Day 4 means by logits.
    # The readout path Day 4 states and enforces, used here too so the two days agree:
    # [CLS] -> LayerNorm -> linear head. The block's raw output does NOT go straight into
    # the head. layer_norm tidies each ROW on its own (axis=-1), so norming the whole
    # (5, 8) block and taking row 0 gives exactly row 0's own norm — checked below.
    cls_out = out[0]                              # ONLY the [CLS] token's output
    cls_tidied = layer_norm(cls_out)              # the same tidy step Day 4 runs
    class_scores = cls_tidied @ W_head
    top_class = int(np.argmax(class_scores))
    shown_cls_out_shape, shown_head_shape = cls_out.shape, W_head.shape
    shown_scores_shape = class_scores.shape
    printed_scores = np.round(class_scores, 4)
    n_tokens = seq.shape[0]
    print("\n[CLS] output:", shown_cls_out_shape, "(1 row of", n_tokens, ") · head:",
          shown_head_shape, "· path: [CLS] -> LayerNorm -> linear head")
    print("class scores", shown_scores_shape, ":", printed_scores,
          "-> highest is class index", top_class)
    # What if we had read a different row? Compute it instead of guessing — same path,
    # every row tidied on its own, so row 0 here must reproduce the scores above.
    all_scores = layer_norm(out) @ W_head
    other_tops = [int(t) for t in all_scores[1:].argmax(axis=1)]
    printed_all_scores = np.round(all_scores, 4)
    shown_all_shape = all_scores.shape
    print("scores for every row (shape %s):\n" % (shown_all_shape,), printed_all_scores)
    print("argmax per row -> [CLS]:", top_class, " tiles:", other_tops,
          "-> they do not all agree, so WHICH row you read decides the answer")
    closest_other_row = round(float(np.abs(all_scores[1:] - class_scores).max(axis=1).min()), 4)
    print("closest any OTHER row's scores get to row 0's:", closest_other_row)
    print("\nanswer from 1 row of", n_tokens, "· stacking changed no shape:", shown_relayed_shape)

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected value below was read off a real run and written down here.
    expected_tile0 = np.array([0.1437, 0.4812, 0.1522, -0.6357, -0.1158, 0.2948, -0.268, -0.3719])
    expected_attn_cls = np.array([0.0269, 0.0032, 0.0112, 0.0246, 0.9341])
    expected_attn_line0 = "  [CLS]   [0.0269 0.0032 0.0112 0.0246 0.9341]"
    # One number per row of the grid, so a corruption in rows 1..4 cannot hide behind row 0.
    expected_row_max = np.array([0.9341, 0.9622, 0.9515, 0.8485, 0.4069])
    expected_row_argmax = [4, 4, 4, 1, 1]
    expected_tidied = np.array([-1.0455, -1.0476, -0.1295, -0.1889, 0.0878, 2.3428, 0.3513, -0.3704])
    expected_scores = np.array([1.0913, -0.8062, 2.3422])   # class 2 wins, from row 0
    expected_other_tops = [0, 2, 0, 1]     # the tile rows disagree with [CLS] and each other
    # Two of those four rows share the winner 0, so the argmax list alone cannot tell row 1
    # from row 3. The scores themselves can, so they are pinned too.
    expected_other_scores = np.array([[1.906, 0.7619, 0.9495], [-0.1166, -2.1554, 1.1415],
                                      [1.1105, 0.894, 0.3594], [0.152, 1.2058, 0.9495]])
    input_ok = shown_tiles_shape == (4, 8) and np.array_equal(shown_tile0, expected_tile0)
    prepend_ok = (shown_nopos_shape == (5, 8) and shown_seq_shape == (5, 8)
                  and predicted_len == seq.shape[0]   # the computed prediction held
                  and cls_leads and tiles_follow)
    # The stamp is carried AND load-bearing: identical tokens in different slots come out
    # identical without it (gap 0.0) and different with it.
    stamp_ok = (shown_pos_shape == (5, 8) and stamp_reach == 1.2362
                and twin_gap_nopos == 0.0 and twin_gap_pos == 1.9889)
    shape_preserved = (shown_out_shape == shown_seq_shape == (5, 8)
                       and shown_relayed_shape == (5, 8)
                       and relay_shapes == [(5, 8), (5, 8), (5, 8)])
    relay_ran = relay_moved == 5.6908 and relay_past_one == 4.8109   # the relay really composed
    shares_ok = rows_sum_one and not cols_sum_one   # rows are shares, columns are not
    global_look = all_nonzero and min_share == 0.0004   # nothing is zero: everyone sees everyone
    cls_row_ok = (np.array_equal(printed_attn[0], expected_attn_cls)
                  and attn_lines[0] == expected_attn_line0
                  and np.array_equal(printed_attn.max(axis=1), expected_row_max)
                  and printed_attn.argmax(axis=1).tolist() == expected_row_argmax)
    grid_lopsided = lopsided == 0.8472
    mixed_ok = moved == 2.0217
    head_ok = (shown_head_shape == (8, 3) and shown_cls_out_shape == (8,)
               and shown_scores_shape == (3,) and shown_all_shape == (5, 3) and n_tokens == 5)
    tidy_ok = np.array_equal(np.round(cls_tidied, 4), expected_tidied)
    # Row 0 of the batched readout IS the single-row readout: one path, not two.
    readout_path_ok = np.array_equal(all_scores[0], class_scores)
    scores_ok = np.array_equal(printed_scores, expected_scores) and top_class == 2
    rows_differ = (other_tops == expected_other_tops
                   and np.array_equal(printed_all_scores[1:], expected_other_scores)
                   # row 0 of the printed table must be the [CLS] scores printed above it
                   and np.array_equal(printed_all_scores[0], printed_scores))
    row0_unique = closest_other_row == 1.3492

    if (input_ok and prepend_ok and stamp_ok and shape_preserved and relay_ran and shares_ok
            and global_look and cls_row_ok and grid_lopsided and mixed_ok and head_ok
            and tidy_ok and readout_path_ok and scores_ok and rows_differ and row0_unique):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected seq (5, 8) with [CLS] in row 0 and the position stamp"
              "\n   ADDED (twin gap 0.0 without it, 1.9889 with it), block output (5, 8),"
              "\n   [CLS] attention row", expected_attn_cls, "every row summing to 1 with the"
              "\n   smallest share 0.0004 (never zero), lopsidedness 0.8472, biggest move 2.0217,"
              "\n   a 3-block relay 5.6908 from the input and 4.8109 past the 1-block output,"
              "\n   closest other row 1.3492, scores", expected_scores, "-> class 2, tile rows",
              expected_other_tops, "with scores", expected_other_scores)

    assert input_ok, "the seeded tile tokens are (4, 8) and tile 0 starts 0.1437 0.4812 0.1522"
    assert prepend_ok, "prepending [CLS] must give (5, 8) with the CLS vector in row 0"
    assert stamp_ok, "the ADDED (5, 8) stamp must split two identical tokens: gap 0.0 -> 1.9889"
    assert shape_preserved, "the block and a stack of blocks must both return (5, 8)"
    assert relay_ran, "3 stacked blocks must land 5.6908 from the input and 4.8109 past 1 block"
    assert shares_ok, "each ROW of attention must add up to 1, and the columns must not"
    assert global_look, "no share may be zero: every token sees every other, smallest 0.0004"
    assert cls_row_ok, "the [CLS] attention row should be %s" % expected_attn_cls
    assert grid_lopsided, "share[i][j] and share[j][i] should differ by up to 0.8472"
    assert mixed_ok, "the block should move some number by 2.0217, not copy its input"
    assert head_ok, "the head is (8, 3) and it reads one row of 8 numbers"
    assert tidy_ok, "the readout's LayerNorm should give %s" % expected_tidied
    assert readout_path_ok, "row 0 of the tidied readout must reproduce the [CLS] class scores"
    assert scores_ok, "the [CLS] row's class scores should be %s" % expected_scores
    assert rows_differ, "the tile rows' argmax should be %s, with scores %s" % (
        expected_other_tops, expected_other_scores)
    assert row0_unique, "the nearest other row's scores should sit 1.3492 away from row 0's"
