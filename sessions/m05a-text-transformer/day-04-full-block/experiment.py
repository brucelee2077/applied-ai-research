# day-04-full-block — experiment
#
# Today's big idea in two lines of output:
#   3 words, 8 numbers each, go into one transformer block and come back (3, 8) —
#   the very shape they arrived in, which is WHY you can stack the same block.
#
# It builds layer norm, a toy mixing attention and a position-wise ffn, wraps each
# part in the Pre-LN order (norm -> part -> add the input back), stacks the SAME
# block 4 times, then shuffles the words and catches the block being blind to order.
# Run it:  python3 sessions/m05a-text-transformer/day-04-full-block/experiment.py

import numpy as np  # numpy gives us arrays and matrix multiply (@)

# The produce step says np.random.randn(...). We use a SEEDED generator instead,
# so every number printed below is the same on every run — the self-check pins them.
rng = np.random.default_rng(0)

SEQ = 3                    # three words in our little sentence
D_MODEL = 8                # each word is 8 numbers wide (the residual lane)
HIDDEN = 4 * D_MODEL       # the feed-forward widens 8 -> 32: the classic 4x
EPS = 1e-5                 # stops layer norm from ever dividing by zero
# The ffn's two matrices, divided by the square root of their fan-in (the usual
# initialisation) so one block's change is about the size of the lane it joins.
W1 = rng.standard_normal((D_MODEL, HIDDEN)) / np.sqrt(D_MODEL)   # widen  [8, 32]
W2 = rng.standard_normal((HIDDEN, D_MODEL)) / np.sqrt(HIDDEN)    # shrink [32, 8]


def layer_norm(x):
    # Day 2's layer_norm took two learnable knobs, gamma and beta. Here they are left at
    # their starting values (gamma = 1, beta = 0), which changes nothing, so they are
    # dropped from the signature to keep the block's wiring the thing you read.
    # Re-level each WORD on its own: take away that word's own average, divide by
    # its own spread, along its 8 numbers (axis=-1 = across one word's row).
    mu = x.mean(axis=-1, keepdims=True)
    sd = np.sqrt(x.var(axis=-1, keepdims=True) + EPS)
    return (x - mu) / sd

def attention(x):
    # A toy stand-in for self-attention: each word keeps half of itself and takes
    # half of the sequence average. Real attention learns its weights; what is
    # kept here is the part that matters — this MIXES across the 3 words.
    seq_mean = x.mean(axis=0, keepdims=True)   # one average word, over the words
    return 0.5 * x + 0.5 * seq_mean

def ffn(x, show=False):
    # The position-wise workshop: widen to 32, bend, shrink back to 8. The same
    # two matrices run on each word's row alone — no word ever sees another.
    h = x @ W1                    # step 1: 8 -> 32
    a = np.maximum(0, h)          # step 2: the ReLU bend, out in the wide space
    out = a @ W2                  # step 3: 32 -> 8, back to the lane's width
    if show:
        print("  in :", x.shape, "-> h = x@W1 :", h.shape, "-> relu(h) :", a.shape,
              "-> out = a@W2 :", out.shape)
    return out

def sublayer(x, fn):
    # Pre-LN wrapping: re-level FIRST, run the part, then add the input back. The
    # raw x sits outside the norm, so the residual lane is never rescaled.
    return x + fn(layer_norm(x))

def transformer_block(x, show=False):
    a_out = sublayer(x, attention)     # sub-layer 1: the words SHARE
    f_out = sublayer(a_out, ffn)       # sub-layer 2: each word THINKS on its own
    if show:
        print("  input:", x.shape, " after attention sublayer:", a_out.shape,
              " after ffn sublayer:", f_out.shape, "<- the shape it came in with")
    return f_out

def show(labels, matrix):
    # One labelled row per line, rounded so the table stays readable.
    for label, row in zip(labels, matrix):
        print("  %-6s %s" % (label, np.round(row, 4)))


if __name__ == "__main__":
    words = ["the", "cat", "sat"]

    # --- Part 1: the workbench, and layer norm re-levelling each word ------
    x = rng.standard_normal((SEQ, D_MODEL))
    print("seq =", SEQ, " d_model =", D_MODEL, " hidden =", HIDDEN, "(a 4x widen)")
    print("W1:", W1.shape, " W2:", W2.shape, " x:", x.shape, "(3 words, 8 wide)")
    show(words, x)
    xn = layer_norm(x)
    print("layer_norm(x) shape:", xn.shape)
    show(words, xn)
    print("each word's own average:", np.round(x.mean(axis=-1), 4), "->",
          np.round(xn.mean(axis=-1), 4), " own spread:", np.round(x.std(axis=-1), 4),
          "->", np.round(xn.std(axis=-1), 4), "(re-levelled per word)")

    # --- Part 2: one part mixes across words, the other never does ---------
    print("\nffn widths, all 3 words at once:")
    ffn(xn, show=True)
    # Poke ONE word and watch whose answers move. The prediction is computed, not
    # typed: a part that mixes across words must move word 1 when word 0 changes.
    x_poke = x.copy()
    x_poke[0] = x[0] + 1.0                      # change word 0, nothing else
    attn_leak = float(np.abs(attention(x_poke)[1] - attention(x)[1]).max())
    ffn_leak = float(np.abs(ffn(x_poke)[1] - ffn(x)[1]).max())
    print("poke word 0 -> word 1 moves by", round(attn_leak, 4), "under attention (MIXES)")
    print("poke word 0 -> word 1 moves by", round(ffn_leak, 4), "under the ffn (per word)")

    # --- Part 3: wrap both parts, run one whole block ----------------------
    print("\none block, Pre-LN order (norm -> part -> add the input back):")
    one_block = transformer_block(x, show=True)
    show(words, one_block)
    lane_move = float(np.abs(one_block - x).max())
    print("furthest any number moved from its input:", round(lane_move, 4),
          "-> the block edits the lane, it does not replace it")

    # --- Part 4: the same block, stacked 4 times ---------------------------
    print("\nstacking the SAME block 4 times:")
    stacked, shapes, lengths = x, [], []
    for depth in range(4):
        stacked = transformer_block(stacked)
        shapes.append(stacked.shape)
        lengths.append(round(float(np.linalg.norm(stacked, axis=-1).mean()), 4))
        print("  after block %d: shape %s   average word length %.4f"
              % (depth + 1, stacked.shape, lengths[-1]))
    print("every shape identical:", shapes.count((SEQ, D_MODEL)) == len(shapes),
          "-> that is why a tower of identical blocks fits together")

    # --- Part 5: shuffle the words — a bare block cannot tell --------------
    seats = [2, 0, 1]                            # "sat the cat": a 3-way rotation
    predicted = one_block[seats]                 # what order-blindness predicts
    out_shuffled = transformer_block(x[seats])   # what we actually get
    print("\nshuffled to", [words[k] for k in seats], "-> shape", out_shuffled.shape)
    show([words[k] for k in seats], out_shuffled)
    order_gap = float(np.abs(out_shuffled - predicted).max())
    print("gap from the shuffled-rows prediction:", round(order_gap, 12),
          "-> rows moved, values did not: permutation-equivariant")
    # The fix from the positional-encoding day (Module 4, Day 5): stamp each SEAT. A flat shift would be
    # invisible (layer norm takes each word's own average away), so the stamp has
    # to vary across the 8 slots — like the sine stamps you built.
    seat_tags = 0.5 * np.sin(np.outer(np.arange(1, SEQ + 1), np.arange(1, D_MODEL + 1)))
    tagged = transformer_block(x + seat_tags)                   # words in seats 1,2,3
    tagged_shuffled = transformer_block(x[seats] + seat_tags)   # the same three seats
    fixed_gap = float(np.abs(tagged_shuffled - tagged[seats]).max())
    print("with seat stamps added, the same prediction misses by", round(fixed_gap, 4),
          "-> order now changes the answer")

    # --- Part 6: Pre-LN left the lane alone; Post-LN would re-level it -----
    # Post-LN is the 2017 original: add the shortcut, THEN norm. Its lane runs
    # through the norm, so every exit row leaves at a spread of exactly 1.
    post_attn = layer_norm(x + attention(x))
    post_out = layer_norm(post_attn + ffn(post_attn))
    pre_spreads = np.round(one_block.std(axis=-1), 4).tolist()
    post_spreads = np.round(post_out.std(axis=-1), 4).tolist()
    print("\nword spreads leaving Pre-LN :", pre_spreads, "(lane untouched)")
    print("word spreads leaving Post-LN:", post_spreads, "(every exit row re-levelled)")

    # --- Self-check: one boolean per claim --------------------------------
    # Every number below was read off a real run and typed in here, so a broken
    # change above cannot quietly agree with itself.
    norm_centred = round(float(np.abs(xn.mean(axis=-1)).max()), 12) == 0.0
    norm_scaled = np.round(xn.std(axis=-1), 4).tolist() == [1.0, 1.0, 1.0]
    attention_mixes = round(attn_leak, 4) == 0.1667
    ffn_is_position_wise = ffn_leak == 0.0
    shape_kept = one_block.shape == (SEQ, D_MODEL) == out_shuffled.shape
    block_row_ok = np.round(one_block[0], 4).tolist() == [-1.0298, 0.177, 1.7421, -2.5462,
                                                         2.6288, 2.6941, -2.4096, -0.6692]
    lane_move_ok = round(lane_move, 4) == 1.5784
    stack_shapes_ok = shapes == [(SEQ, D_MODEL)] * 4
    stack_lengths_ok = lengths == [4.4954, 6.8004, 9.2372, 11.7813]
    order_blind = round(order_gap, 12) == 0.0
    seat_stamps_fix_it = round(fixed_gap, 4) == 1.6095
    pre_ln_left_lane_alone = pre_spreads == [1.9682, 1.1003, 1.6142]
    post_ln_relevels = post_spreads == [1.0, 1.0, 1.0]

    if (norm_centred and norm_scaled and attention_mixes and ffn_is_position_wise
            and shape_kept and block_row_ok and lane_move_ok and stack_shapes_ok
            and stack_lengths_ok and order_blind and seat_stamps_fix_it
            and pre_ln_left_lane_alone and post_ln_relevels):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected each word centred at 0 with spread 1 after layer norm;"
              " word 1 moving 0.1667 under attention and 0.0 under the ffn when word 0 is"
              " poked; one block (3, 8) in and out, row 0 = [-1.0298 0.177 1.7421 -2.5462"
              " 2.6288 2.6941 -2.4096 -0.6692], nothing further than 1.5784 from its input;"
              " 4 stacked blocks all (3, 8), word length [4.4954, 6.8004, 9.2372, 11.7813];"
              " the shuffled run matching the reordered rows exactly; seat stamps breaking"
              " that by 1.6095; Pre-LN spreads [1.9682, 1.1003, 1.6142] vs Post-LN [1, 1, 1]")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert norm_centred, "after layer norm each word's own average must be 0"
    assert norm_scaled, "after layer norm each word's own spread must be 1"
    assert attention_mixes, "poking word 0 should move word 1's attention answer by 0.1667"
    assert ffn_is_position_wise, "the ffn must not move word 1 when only word 0 changed"
    assert shape_kept, "a block must hand back the (3, 8) it received"
    assert block_row_ok, "one block's row 0 should be [-1.0298 0.177 1.7421 -2.5462 " \
                         "2.6288 2.6941 -2.4096 -0.6692]"
    assert lane_move_ok, "the furthest number should move 1.5784 from its input"
    assert stack_shapes_ok, "all 4 stacked blocks must report (3, 8)"
    assert stack_lengths_ok, "average word length should grow 4.4954 -> 11.7813 over 4 blocks"
    assert order_blind, "a shuffled input must give the unshuffled output with rows reordered"
    assert seat_stamps_fix_it, "with seat stamps the order-blind prediction should miss by 1.6095"
    assert pre_ln_left_lane_alone, "Pre-LN spreads should be [1.9682, 1.1003, 1.6142]"
    assert post_ln_relevels, "Post-LN must re-level every exit row to a spread of 1.0"
