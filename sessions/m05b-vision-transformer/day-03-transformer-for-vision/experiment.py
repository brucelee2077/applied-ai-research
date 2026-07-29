# day-03-transformer-for-vision — experiment
#
# Today's big idea in two lines of output:
#   The SAME Transformer block that reads a sentence reads a line of picture
#   tiles: it hands back the exact same shape, so you can stack it in a relay —
#   and the whole picture's answer is read off ONE row, the [CLS] token.
#
# It builds a fake line of 4 tile tokens, prepends [CLS], runs the reused block
# (attention + feed-forward, each with a norm and a shortcut), stacks it, then
# reads 3 class scores off row 0 only.
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
    tiles = rng.standard_normal((N, D))
    print("tiles shape:", tiles.shape, "->", N, "tile tokens,", D, "numbers each")
    print("tile 0:", np.round(tiles[0], 4))

    # --- Part 2: prepend the [CLS] token ----------------------------------
    # Predict first, but COMPUTE the prediction from the input instead of typing it.
    predicted_len = tiles.shape[0] + 1
    cls_token = rng.standard_normal(D)                # the captain's own vector
    seq = np.vstack([cls_token[None, :], tiles])      # [CLS] leads the line
    print("\npredicted sequence length:", predicted_len, "= N + 1  ->  seq shape:", seq.shape)
    print("row 0 is the [CLS] vector?", np.array_equal(seq[0], cls_token),
          " rows 1..%d are the tiles?" % N, np.array_equal(seq[1:], tiles))

    # --- Part 3: run the reused block, then stack it ----------------------
    out, attn = block(seq)
    print("\nblock in:", seq.shape, "-> block out:", out.shape, "(the SAME shape -> stackable)")
    print("attention shares, row i = how much token i took from each token:")
    for i, name in enumerate(["[CLS]"] + ["tile %d" % t for t in range(N)]):
        print("  %-7s %s" % (name, np.round(attn[i], 4)))
    print("every ROW adds up to 1?", np.allclose(attn.sum(axis=1), 1.0), " every COLUMN?",
          np.allclose(attn.sum(axis=0), 1.0), "-> rows: the softmax runs along axis -1")
    lopsided = round(float(np.abs(attn - attn.T).max()), 4)
    moved = round(float(np.abs(out - seq).max()), 4)
    print("biggest gap between share[i][j] and share[j][i]:", lopsided,
          "· furthest any number moved inside the block:", moved,
          "-> lopsided grid (reading it the wrong way round changes the answer), real mixing")
    relayed = seq                                 # a relay: the same block, again and again
    for depth in range(3):
        relayed, _ = block(relayed)
        print("after block", depth + 1, "-> shape", relayed.shape)
    # The shape ALONE proves nothing about the relay: `relayed = seq` already has shape
    # (5, 8), so a loop that never called block() would pass a shape-only check. These two
    # distances are the evidence that the block really ran three times and each round moved
    # the numbers on: away from the input, and past the 1-block output.
    relay_moved = round(float(np.abs(relayed - seq).max()), 4)
    relay_past_one = round(float(np.abs(relayed - out).max()), 4)
    print("3 blocks moved a number by", relay_moved, "from the input, and sit", relay_past_one,
          "from the 1-block output -> the relay advanced, it did not just keep the shape")

    # --- Part 4: read the answer off row 0 only ---------------------------
    W_head = rng.standard_normal((D, C))          # the classification head
    cls_out = out[0]                              # ONLY the [CLS] token's output
    class_scores = cls_out @ W_head
    top_class = int(np.argmax(class_scores))
    print("\n[CLS] output:", cls_out.shape, "(1 row of", seq.shape[0], ") · head:", W_head.shape)
    print("class scores", class_scores.shape, ":", np.round(class_scores, 4),
          "-> highest is class index", top_class)
    # What if we had read a different row? Compute it instead of guessing.
    all_scores = out @ W_head
    other_tops = [int(t) for t in all_scores[1:].argmax(axis=1)]
    print("scores for every row (shape %s):\n" % (all_scores.shape,), np.round(all_scores, 4))
    print("argmax per row -> [CLS]:", top_class, " tiles:", other_tops,
          "-> they do not all agree, so WHICH row you read decides the answer")
    closest_other_row = round(float(np.abs(all_scores[1:] - class_scores).max(axis=1).min()), 4)
    print("closest any OTHER row's scores get to row 0's:", closest_other_row)
    print("\nanswer from 1 row of", seq.shape[0], "· stacking changed no shape:", relayed.shape)

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected value below was read off a real run and written down here.
    expected_attn_cls = np.array([0.2396, 0.1725, 0.1397, 0.1362, 0.312])
    expected_scores = np.array([2.2043, -4.9609, 4.7966])   # class 2 wins, from row 0
    expected_other_tops = [2, 2, 1, 2]     # tile 2 picks a different class than [CLS]
    # Two of those four rows share the winner 2, so the argmax list alone cannot tell row 1
    # from row 2. The scores themselves can, so they are pinned too.
    expected_other_scores = np.array([[-1.9303, 1.1152, 1.7769], [2.6227, -1.201, 3.1094],
                                      [0.2454, 5.9853, 0.855], [-3.3239, -1.7974, -0.6052]])
    prepend_ok = (tiles.shape == (4, 8) and seq.shape == (5, 8)
                  and predicted_len == seq.shape[0]   # the computed prediction held
                  and np.array_equal(seq[0], cls_token) and np.array_equal(seq[1:], tiles))
    shape_preserved = (out.shape == seq.shape == (5, 8) and relayed.shape == (5, 8))
    relay_ran = relay_moved == 6.518 and relay_past_one == 5.4549   # the relay really composed
    shares_ok = (np.allclose(attn.sum(axis=1), 1.0)           # rows are shares
                 and not np.allclose(attn.sum(axis=0), 1.0))  # columns are not
    cls_row_ok = np.array_equal(np.round(attn[0], 4), expected_attn_cls)
    grid_lopsided = lopsided == 0.509
    mixed_ok = moved == 2.0073
    head_ok = W_head.shape == (8, 3) and cls_out.shape == (8,)
    scores_ok = np.array_equal(np.round(class_scores, 4), expected_scores)
    rows_differ = (other_tops == expected_other_tops
                   and np.array_equal(np.round(all_scores[1:], 4), expected_other_scores))
    row0_unique = closest_other_row == 3.7599

    if (prepend_ok and shape_preserved and relay_ran and shares_ok and cls_row_ok
            and grid_lopsided and mixed_ok and head_ok and scores_ok and rows_differ
            and row0_unique):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected seq (5, 8) with [CLS] in row 0, block output (5, 8),"
              "\n   [CLS] attention row", expected_attn_cls, "every row summing to 1,"
              "\n   lopsidedness 0.509, biggest move 2.0073, a 3-block relay 6.518 from the"
              "\n   input and 5.4549 past the 1-block output, closest other row 3.7599, scores",
              expected_scores, "-> class 2, tile rows", expected_other_tops, "with scores",
              expected_other_scores)

    assert prepend_ok, "prepending [CLS] must give (5, 8) with the CLS vector in row 0"
    assert shape_preserved, "the block and a stack of blocks must both return (5, 8)"
    assert relay_ran, "3 stacked blocks must land 6.518 from the input and 5.4549 past 1 block"
    assert shares_ok, "each ROW of attention must add up to 1, and the columns must not"
    assert cls_row_ok, "the [CLS] attention row should be %s" % expected_attn_cls
    assert grid_lopsided, "share[i][j] and share[j][i] should differ by up to 0.509"
    assert mixed_ok, "the block should move some number by 2.0073, not copy its input"
    assert head_ok, "the head is (8, 3) and it reads one row of 8 numbers"
    assert scores_ok, "the [CLS] row's class scores should be %s" % expected_scores
    assert rows_differ, "the tile rows' argmax should be %s, with scores %s" % (
        expected_other_tops, expected_other_scores)
    assert row0_unique, "the nearest other row's scores should sit 3.7599 away from row 0's"
