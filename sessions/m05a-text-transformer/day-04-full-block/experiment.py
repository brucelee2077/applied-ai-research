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
# "the residual lane" = the pass-through x that each part's change is ADDED to. Day 1 used
# the bare word "the residual" for the CHANGE, F(x); in this file the change is called
# "one block's change" and the lane is the thing it joins. Two objects, kept apart.
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

def relu(h):
    # The bend. It lives in its own function so its EDGE can be probed: inside the
    # block h never lands exactly on 0, so nothing there tests what happens at 0.
    return np.maximum(0, h)

def ffn(x, *, trace=None):
    # The position-wise workshop: widen to 32, bend, shrink back to 8. The same
    # two matrices run on each word's row alone — no word ever sees another.
    # This is Day 3's `feed_forward(x, W1, W2)` under a shorter name, with the two matrices
    # taken from module level so the BLOCK's wiring is what you read here. Same three steps
    # and the same 4x width, and still no biases (Day 8 adds those). `trace` is
    # keyword-only on purpose: it is a printing hook, so passing Day 3's weights in
    # positionally has to raise instead of quietly binding them to it.
    h = x @ W1                    # step 1: 8 -> 32
    a = relu(h)                   # step 2: the ReLU bend, out in the wide space
    out = a @ W2                  # step 3: 32 -> 8, back to the lane's width
    if trace is not None:
        # One tuple of widths, and the LINE built from it exactly once. Both are handed
        # back, so the self-check pins the widths AND the rendered text. That matters:
        # pinning the tuple alone would leave the four subscripts at the print site free
        # to shift, printing "in : (3, 32)" — the 4x widen inverted — under a passing ✅.
        widths = (x.shape, h.shape, a.shape, out.shape)
        trace_line = ("  in : %s -> h = x@W1 : %s -> relu(h) : %s -> out = a@W2 : %s"
                      % widths)
        trace.append(widths)
        trace.append(trace_line)
        print(trace_line)
    return out

def sublayer(x, fn):
    # Pre-LN wrapping: re-level FIRST, run the part, then add the input back. The
    # raw x sits outside the norm, so the residual lane is never rescaled.
    return x + fn(layer_norm(x))

def transformer_block(x, trace=None):
    a_out = sublayer(x, attention)     # sub-layer 1: the words SHARE
    f_out = sublayer(a_out, ffn)       # sub-layer 2: each word THINKS on its own
    if trace is not None:
        stages = (x.shape, a_out.shape, f_out.shape)   # printed AND checked
        trace.append(stages)
        print("  input:", stages[0], " after attention sublayer:", stages[1],
              " after ffn sublayer:", stages[2], "<- the shape it came in with")
    return f_out

def show(labels, matrix):
    # One labelled row per line, rounded so the table stays readable. The rounded rows
    # are handed BACK as well as printed, so the self-check can pin the exact rows the
    # learner reads — this renderer is not a second, unguarded computation.
    rendered = []
    for label, row in zip(labels, matrix):
        shown_row = np.round(row, 4)
        print("  %-6s %s" % (label, shown_row))
        rendered.append(shown_row.tolist())
    return rendered


if __name__ == "__main__":
    words = ["the", "cat", "sat"]

    # --- Part 1: the workbench, and layer norm re-levelling each word ------
    x = rng.standard_normal((SEQ, D_MODEL))
    # Every printed number below is bound to a name FIRST and then checked by that same
    # name, so a wrong number cannot be printed under a passing ✅.
    shown_widths = (SEQ, D_MODEL, HIDDEN)
    shown_w1, shown_w2, shown_x_shape = W1.shape, W2.shape, x.shape
    print("seq = %d  d_model = %d  hidden = %d (a 4x widen)" % shown_widths)
    print("W1:", shown_w1, " W2:", shown_w2, " x:", shown_x_shape, "(3 words, 8 wide)")
    shown_x = show(words, x)
    xn = layer_norm(x)
    shown_xn_shape = xn.shape
    print("layer_norm(x) shape:", shown_xn_shape)
    shown_xn = show(words, xn)
    shown_x_means = np.round(x.mean(axis=-1), 4)
    shown_xn_means = np.round(xn.mean(axis=-1), 4)
    shown_x_spreads = np.round(x.std(axis=-1), 4)
    shown_xn_spreads = np.round(xn.std(axis=-1), 4)
    print("each word's own average:", shown_x_means, "->",
          shown_xn_means, " own spread:", shown_x_spreads,
          "->", shown_xn_spreads, "(re-levelled per word)")

    # --- Part 2: one part mixes across words, the other never does ---------
    print("\nffn widths, all 3 words at once:")
    ffn_widths = []
    ffn(xn, trace=ffn_widths)
    # The bend's edge, probed on the SAME relu the ffn calls: 0 itself and a hair either
    # side of it, which the block's own numbers never land on.
    relu_probe_in = np.array([-2.5, -1e-12, -0.0, 0.0, 1e-12, 3.5])
    relu_probe_out = relu(relu_probe_in).tolist()
    # Rendered once and pinned as text below, so the INPUT of the probe is held too: the
    # pinned output alone would still print if 0 or -0.0 quietly left the probe.
    relu_probe_line = ("relu boundary probe: %s -> %s"
                       % (relu_probe_in.tolist(), relu_probe_out))
    print(relu_probe_line)
    # Poke ONE word and watch whose answers move. The prediction is computed, not
    # typed: a part that mixes across words must move word 1 when word 0 changes.
    x_poke = x.copy()
    x_poke[0] = x[0] + 1.0                      # change word 0, nothing else
    attn_leak = float(np.abs(attention(x_poke)[1] - attention(x)[1]).max())
    ffn_leak = float(np.abs(ffn(x_poke)[1] - ffn(x)[1]).max())
    shown_attn_leak = round(attn_leak, 4)
    shown_ffn_leak = round(ffn_leak, 4)
    print("poke word 0 -> word 1 moves by", shown_attn_leak, "under attention (MIXES)")
    print("poke word 0 -> word 1 moves by", shown_ffn_leak, "under the ffn (per word)")

    # --- Part 3: wrap both parts, run one whole block ----------------------
    print("\none block, Pre-LN order (norm -> part -> add the input back):")
    block_stages = []
    one_block = transformer_block(x, trace=block_stages)
    shown_one_block = show(words, one_block)
    lane_move = float(np.abs(one_block - x).max())
    shown_lane_move = round(lane_move, 4)
    print("furthest any number moved from its input:", shown_lane_move,
          "-> the block edits the lane, it does not replace it")

    # --- Part 4: the same block, stacked 4 times ---------------------------
    print("\nstacking the SAME block 4 times:")
    stacked, shapes, lengths = x, [], []
    for depth in range(4):
        stacked = transformer_block(stacked)
        shapes.append(stacked.shape)
        lengths.append(round(float(np.linalg.norm(stacked, axis=-1).mean()), 4))
        print("  after block %d: shape %s   average word length %.4f"
              % (depth + 1, shapes[-1], lengths[-1]))
    every_shape_identical = shapes.count((SEQ, D_MODEL)) == len(shapes)
    print("every shape identical:", every_shape_identical,
          "-> that is why a tower of identical blocks fits together")

    # --- Part 5: shuffle the words — a bare block cannot tell --------------
    seats = [2, 0, 1]                            # "sat the cat": a 3-way rotation
    predicted = one_block[seats]                 # what order-blindness predicts
    out_shuffled = transformer_block(x[seats])   # what we actually get
    shuffled_words = [words[k] for k in seats]
    shown_shuffled_shape = out_shuffled.shape
    print("\nshuffled to", shuffled_words, "-> shape", shown_shuffled_shape)
    shown_shuffled = show(shuffled_words, out_shuffled)
    order_gap = float(np.abs(out_shuffled - predicted).max())
    shown_order_gap = round(order_gap, 12)
    print("gap from the shuffled-rows prediction:", shown_order_gap,
          "-> rows moved, values did not: permutation-equivariant")
    # The fix from the positional-encoding day (Module 4, Day 5): stamp each SEAT. A flat shift would be
    # invisible (layer norm takes each word's own average away), so the stamp has
    # to vary across the 8 slots — like the sine stamps you built.
    seat_tags = 0.5 * np.sin(np.outer(np.arange(1, SEQ + 1), np.arange(1, D_MODEL + 1)))
    tagged = transformer_block(x + seat_tags)                   # words in seats 1,2,3
    tagged_shuffled = transformer_block(x[seats] + seat_tags)   # the same three seats
    fixed_gap = float(np.abs(tagged_shuffled - tagged[seats]).max())
    shown_fixed_gap = round(fixed_gap, 4)
    print("with seat stamps added, the same prediction misses by", shown_fixed_gap,
          "-> order now changes the answer")

    # --- Part 6: Pre-LN left the lane alone; Post-LN would re-level it -----
    # Post-LN is the 2017 original: add the shortcut, THEN norm. Its lane runs
    # through the norm, so every exit row leaves at a spread of 1 (to the 4 decimals
    # printed — Day 2 pinned the exact hair below 1 that eps leaves behind).
    post_attn = layer_norm(x + attention(x))
    post_out = layer_norm(post_attn + ffn(post_attn))
    pre_spreads = np.round(one_block.std(axis=-1), 4).tolist()
    post_spreads = np.round(post_out.std(axis=-1), 4).tolist()
    print("\nword spreads leaving Pre-LN :", pre_spreads, "(lane untouched)")
    print("word spreads leaving Post-LN:", post_spreads, "(every exit row re-levelled)")
    # Those three 1.0s are true of ANY layer_norm output, so on their own they would
    # still print if the Post-LN wiring were wrong (shortcut dropped, inputs rescaled).
    # Print the rows Post-LN actually produces, so the WIRING is what gets checked.
    print("the rows Post-LN hands out (spread 1 cannot tell you these are right):")
    shown_post_out = show(words, post_out)

    # --- Self-check: one boolean per claim --------------------------------
    # Every number below was read off a real run and typed in here, so a broken
    # change above cannot quietly agree with itself. Each claim reads the SAME name
    # the print statement read — there is no second computation to disagree with.
    widths_ok = shown_widths == (3, 8, 32) and HIDDEN == 4 * D_MODEL
    matrix_shapes_ok = ((shown_w1, shown_w2) == ((8, 32), (32, 8))
                        and shown_x_shape == shown_xn_shape == (SEQ, D_MODEL))
    input_rows_ok = shown_x[0] == [-0.544, -0.1335, 1.2979, -0.9678,
                                   1.9267, 1.8793, -1.7134, -0.141]
    raw_stats_ok = (shown_x_means.tolist() == [0.2005, 0.1185, 0.5542]
                    and shown_x_spreads.tolist() == [1.2655, 0.823, 0.7317])
    norm_centred = ([abs(v) for v in shown_xn_means.tolist()] == [0.0, 0.0, 0.0]
                    and round(float(np.abs(xn.mean(axis=-1)).max()), 12) == 0.0)
    # "spread 1" here means 1 TO THE 4 DECIMALS PRINTED. Day 2 showed the exact spread is a
    # hair under 1 (0.999999925) because eps sits inside the root; nothing on this day
    # contradicts that, and a learner testing exact equality should expect Day 2's number.
    norm_scaled = shown_xn_spreads.tolist() == [1.0, 1.0, 1.0]
    norm_rows_ok = shown_xn[0] == [-0.5883, -0.2639, 0.8671, -0.9232,
                                   1.3639, 1.3266, -1.5124, -0.2699]
    ffn_widths_ok = ffn_widths[:1] == [((3, 8), (3, 32), (3, 32), (3, 8))]
    # The same widths as the learner actually READS them: the whole line, character for
    # character. A shifted subscript or a swapped operand at the print site changes this
    # string, so the 4x widen cannot come out inverted under a passing ✅.
    ffn_trace_line_ok = ffn_widths[1:] == [
        "  in : (3, 8) -> h = x@W1 : (3, 32) -> relu(h) : (3, 32) -> out = a@W2 : (3, 8)"]
    # A relu that shifted its threshold, leaked negatives, or clipped tiny positives
    # would change one of these six numbers; the block's own numbers never test it.
    relu_boundary_ok = relu_probe_out == [0.0, 0.0, 0.0, 0.0, 1e-12, 3.5]
    # ...and the probe's INPUT, pinned through the line it was printed on, so the six
    # values being probed stay the six the comment above promises.
    relu_probe_line_ok = relu_probe_line == ("relu boundary probe: "
                                             "[-2.5, -1e-12, -0.0, 0.0, 1e-12, 3.5] -> "
                                             "[0.0, 0.0, 0.0, 0.0, 1e-12, 3.5]")
    attention_mixes = shown_attn_leak == 0.1667
    ffn_is_position_wise = shown_ffn_leak == 0.0 and ffn_leak == 0.0
    shape_kept = one_block.shape == (SEQ, D_MODEL) == shown_shuffled_shape
    block_stages_ok = block_stages == [((SEQ, D_MODEL),) * 3]
    block_row_ok = shown_one_block[0] == [-1.0298, 0.177, 1.7421, -2.5462,
                                         2.6288, 2.6941, -2.4096, -0.6692]
    block_rest_ok = shown_one_block[1:] == [
        [0.1629, -1.2829, -1.1827, 0.3945, 0.8198, -2.0859, 1.2686, 0.3556],
        [-1.4456, 2.4196, 2.0283, -0.2104, -1.354, 2.9022, -0.4974, -0.0835]]
    lane_move_ok = shown_lane_move == 1.5784
    stack_shapes_ok = every_shape_identical and shapes == [(SEQ, D_MODEL)] * 4
    stack_lengths_ok = lengths == [4.4954, 6.8004, 9.2372, 11.7813]
    order_blind = shown_order_gap == 0.0
    # The printed shuffled table must be the printed unshuffled table, rows reordered —
    # checked on the RENDERED rows, which are pinned to literals just above.
    shuffled_table_ok = (shuffled_words == ["sat", "the", "cat"]
                         and shown_shuffled == [shown_one_block[k] for k in seats])
    seat_stamps_fix_it = shown_fixed_gap == 1.6095
    pre_ln_left_lane_alone = pre_spreads == [1.9682, 1.1003, 1.6142]
    post_ln_relevels = post_spreads == [1.0, 1.0, 1.0]
    # The rows themselves, pinned. Dropping Post-LN's shortcut or rescaling its inputs
    # leaves the three spreads at 1.0 but moves these numbers, so this is the claim
    # that actually holds the Post-LN wiring in place.
    post_ln_rows_ok = shown_post_out == [
        [-0.4309, 0.3036, 0.8539, -1.538, 1.2635, 1.1643, -1.2378, -0.3786],
        [0.3817, -0.9016, -0.7037, 0.8619, 0.9346, -1.9797, 0.8058, 0.6009],
        [-1.1721, 1.1566, 1.1795, -0.4723, -1.044, 1.3469, -0.8686, -0.1261]]

    if (widths_ok and matrix_shapes_ok and input_rows_ok and raw_stats_ok
            and norm_centred and norm_scaled and norm_rows_ok and ffn_widths_ok
            and ffn_trace_line_ok and relu_boundary_ok and relu_probe_line_ok
            and attention_mixes and ffn_is_position_wise
            and shape_kept and block_stages_ok and block_row_ok and block_rest_ok
            and lane_move_ok and stack_shapes_ok and stack_lengths_ok and order_blind
            and shuffled_table_ok and seat_stamps_fix_it and pre_ln_left_lane_alone
            and post_ln_relevels and post_ln_rows_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected each word centred at 0 with spread 1 after layer norm;"
              " the ffn widening (3, 8) -> (3, 32) -> (3, 8) and its relu leaving 0 at 0 while"
              " a tiny positive passes through; word 1 moving 0.1667 under attention and 0.0"
              " under the ffn when word 0 is"
              " poked; one block (3, 8) in and out, row 0 = [-1.0298 0.177 1.7421 -2.5462"
              " 2.6288 2.6941 -2.4096 -0.6692], nothing further than 1.5784 from its input;"
              " 4 stacked blocks all (3, 8), word length [4.4954, 6.8004, 9.2372, 11.7813];"
              " the shuffled run matching the reordered rows exactly; seat stamps breaking"
              " that by 1.6095; Pre-LN spreads [1.9682, 1.1003, 1.6142] vs Post-LN [1, 1, 1]"
              " with Post-LN's first exit row = [-0.4309 0.3036 0.8539 -1.538 1.2635 1.1643"
              " -1.2378 -0.3786]")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert widths_ok, "the printed widths should be seq 3, d_model 8, hidden 32 (a 4x widen)"
    assert matrix_shapes_ok, "W1 must print (8, 32), W2 (32, 8), x and layer_norm(x) (3, 8)"
    assert input_rows_ok, "the input table's row 0 should be [-0.544 -0.1335 1.2979 " \
                          "-0.9678 1.9267 1.8793 -1.7134 -0.141]"
    assert raw_stats_ok, "before layer norm the words' averages/spreads should print " \
                         "[0.2005 0.1185 0.5542] and [1.2655 0.823 0.7317]"
    assert norm_centred, "after layer norm each word's own average must be 0"
    assert norm_scaled, "after layer norm each word's own spread must be 1 to 4 decimals"
    assert norm_rows_ok, "layer_norm(x)'s row 0 should be [-0.5883 -0.2639 0.8671 " \
                         "-0.9232 1.3639 1.3266 -1.5124 -0.2699]"
    assert ffn_widths_ok, "the ffn should print (3, 8) -> (3, 32) -> (3, 32) -> (3, 8)"
    assert ffn_trace_line_ok, "the ffn's printed line must read '  in : (3, 8) -> h = x@W1 : " \
                              "(3, 32) -> relu(h) : (3, 32) -> out = a@W2 : (3, 8)' exactly"
    assert relu_boundary_ok, "relu must send 0 and every negative to 0.0 and pass 1e-12 " \
                             "through unchanged"
    assert relu_probe_line_ok, "the probe line must show the six probed values " \
                               "[-2.5, -1e-12, -0.0, 0.0, 1e-12, 3.5] and their relu"
    assert attention_mixes, "poking word 0 should move word 1's attention answer by 0.1667"
    assert ffn_is_position_wise, "the ffn must not move word 1 when only word 0 changed"
    assert shape_kept, "a block must hand back the (3, 8) it received"
    assert block_stages_ok, "the block should report (3, 8) at its input and after both sub-layers"
    assert block_row_ok, "one block's row 0 should be [-1.0298 0.177 1.7421 -2.5462 " \
                         "2.6288 2.6941 -2.4096 -0.6692]"
    assert block_rest_ok, "one block's rows 1 and 2 should be [0.1629 -1.2829 -1.1827 " \
                          "0.3945 0.8198 -2.0859 1.2686 0.3556] and [-1.4456 2.4196 2.0283 " \
                          "-0.2104 -1.354 2.9022 -0.4974 -0.0835]"
    assert lane_move_ok, "the furthest number should move 1.5784 from its input"
    assert stack_shapes_ok, "all 4 stacked blocks must report (3, 8)"
    assert stack_lengths_ok, "average word length should grow 4.4954 -> 11.7813 over 4 blocks"
    assert order_blind, "a shuffled input must give the unshuffled output with rows reordered"
    assert shuffled_table_ok, "the shuffled table must be the same printed rows in seat order " \
                              "sat, the, cat"
    assert seat_stamps_fix_it, "with seat stamps the order-blind prediction should miss by 1.6095"
    assert pre_ln_left_lane_alone, "Pre-LN spreads should be [1.9682, 1.1003, 1.6142]"
    assert post_ln_relevels, "Post-LN must re-level every exit row to a spread of 1.0 (4 dp)"
    assert post_ln_rows_ok, "Post-LN's exit rows should be [-0.4309 0.3036 0.8539 -1.538 " \
                            "1.2635 1.1643 -1.2378 -0.3786], [0.3817 -0.9016 -0.7037 0.8619 " \
                            "0.9346 -1.9797 0.8058 0.6009], [-1.1721 1.1566 1.1795 -0.4723 " \
                            "-1.044 1.3469 -0.8686 -0.1261]"
