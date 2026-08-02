# day-03-feed-forward — experiment
#
# Today's big idea in two lines of output:
#   A word walks into the workshop 4 numbers wide, spreads out to 16, and comes
#   back 4 wide. Take the bend out of the middle and the two matrices fold into
#   ONE matrix, so all that widening bought nothing at all.
# Run it:  python3 sessions/m05a-text-transformer/day-03-feed-forward/experiment.py

import numpy as np  # numpy gives us arrays and matrix multiply (@)

# The produce step says np.random.randn(...). We use a SEEDED generator instead,
# so every number printed below is the same on every run and on every machine —
# the self-check at the bottom pins these exact values.
rng = np.random.default_rng(0)

D_MODEL = 4               # width of a word's vector as it flows up the transformer
HIDDEN = 4 * D_MODEL      # the classic 4x expansion: 4 -> 16


def relu(z):
    # The bend: keep positive numbers as they are, turn every negative into a flat 0.
    return np.maximum(0, z)

def gelu(z):
    # A smooth bend (the tanh form of GELU, the one GPT uses). Positives behave
    # much like ReLU, but small negatives leak through as tiny non-zero numbers.
    return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))

def count_negative(values):
    # STRICT "<": a value sitting at exactly 0 is NOT negative. The boundary check in
    # Part 4 is what makes that strictness testable.
    return int(np.count_nonzero(values < 0))

def dark_positions(values):
    # A bench is dark only at EXACTLY 0, not merely at-or-below 0. The gelu check in
    # Part 4 is what makes THAT strictness testable: gelu's outputs do go negative.
    return np.flatnonzero(values == 0.0)

def count_dark(values):
    return int(dark_positions(values).size)

def feed_forward(x, W1, W2):
    # Position-wise: the SAME two matrices run on each word's vector on its own.
    # ORIENTATION CHANGE, announced because Day 1 made orientation load-bearing: Day 1 put
    # the weight on the LEFT of a column vector (W @ x, W shaped (out, in)). From here on a
    # word is a ROW and the weight sits on the RIGHT (x @ W, W shaped (in, out)) — the
    # convention module m03 used and every later day in this module uses.
    h = x @ W1       # step 1 up-projection: d_model -> hidden (spread out wide)
    a = relu(h)      # step 2 the bend, applied in the wide space (the thinking)
    out = a @ W2     # step 3 down-projection: hidden -> d_model (carry it home)
    return h, a, out

def show_row(name, arr):
    # Render the shape and the numbers ONCE and hand them back, so the line the reader
    # sees and the line the self-check pins are the very same values.
    shape, shown = arr.shape, np.round(arr, 4)
    print("  %-9s" % name, shape, "->", shown)
    return shape, shown


if __name__ == "__main__":
    # Every number that carries a claim is BOUND first, printed from that binding, and
    # checked through the same binding — one value read twice, never two expressions.

    # --- Part 1: the two matrices of the workshop -------------------------
    x = rng.standard_normal(D_MODEL)              # one word, D_MODEL numbers wide
    W1 = rng.standard_normal((D_MODEL, HIDDEN))   # up-projection   [4, 16]
    W2 = rng.standard_normal((HIDDEN, D_MODEL))   # down-projection [16, 4]
    shown_w1_shape, shown_w2_shape = W1.shape, W2.shape
    print("d_model =", D_MODEL, " hidden =", HIDDEN, " W1", shown_w1_shape,
          " W2", shown_w2_shape)

    # --- Part 2: push one word through, watching the width ----------------
    print("\none word through the workshop — watch the width go 4 -> 16 -> 4:")
    h, a, out = feed_forward(x, W1, W2)
    shown_in_shape, shown_in = show_row("in", x)
    shown_h_shape, shown_h = show_row("h=x@W1", h)
    shown_a_shape, shown_a = show_row("relu(h)", a)
    shown_out_shape, shown_out = show_row("out=a@W2", out)
    widths = (x.size, h.size, out.size)
    # "residual stream" = the pass-through x that a block's change gets ADDED to (the "+ x"
    # of Day 1). Careful: Day 1 used the bare word "the residual" for the CHANGE itself.
    # From here on, residual stream / residual lane always means the pass-through.
    print("  widths   ", widths, "-> as wide out as in, so it fits the residual stream")

    # --- Part 3: with no bend, the two matrices collapse into one ---------
    straight = lambda vec: (vec @ W1) @ W2       # no bend at all, just both matrices
    bent = lambda vec: relu(vec @ W1) @ W2       # the real workshop, ReLU in the middle
    W_collapsed = W1 @ W2                        # ONE matrix doing the no-bend path
    shown_collapsed_shape = W_collapsed.shape
    shown_collapsed = np.round(W_collapsed, 4)
    print("\nW1 @ W2 = one matrix,", shown_collapsed_shape, ":\n", shown_collapsed)
    one_matrix = x @ W_collapsed
    collapse_holds = np.allclose(straight(x), one_matrix)
    shown_two_step = np.round(straight(x), 4)
    shown_one_step = np.round(one_matrix, 4)
    shown_bent = np.round(bent(x), 4)
    print("(x@W1)@W2     =", shown_two_step)
    print("x@(W1@W2)     =", shown_one_step, "-> same numbers:", collapse_holds)
    print("relu(x@W1)@W2 =", shown_bent, "-> the bend changed the answer")

    # One matched answer could still be luck, so test the property that DEFINES a
    # single matrix: it adds up, g(u+v) == g(u) + g(v). Fail that even once and NO
    # single matrix can reproduce the bent path.
    u, v = x, rng.standard_normal(D_MODEL)        # two different words
    straight_adds_up = np.allclose(straight(u + v), straight(u) + straight(v))
    bent_add_gap = float(np.abs(bent(u + v) - (bent(u) + bent(v))).max())
    shown_bent_add_gap = round(bent_add_gap, 4)
    print("straight path adds up:", straight_adds_up, " bent path off by:",
          shown_bent_add_gap, "-> the bend cannot be folded into one matrix")

    # --- Part 4: dark benches, and the gentler bend that saves them -------
    # Predict the dark benches by counting negatives in h, then see if relu agrees.
    predicted_dark = count_negative(h)
    dark_benches = count_dark(a)
    shown_negatives = np.round(h[h < 0], 4)
    print("\nnegative values in the wide space:", shown_negatives)
    print("dark benches predicted from those signs:", predicted_dark,
          " benches relu set to exactly 0:", dark_benches, "of", HIDDEN)
    smooth = gelu(h)                              # the same h, bent gently instead
    dark_idx = dark_positions(a)                  # which benches relu switched off
    hard_zeros_left = count_dark(smooth)
    shown_smooth_dark = np.round(smooth[dark_idx], 4)
    print("gelu on those same benches:", shown_smooth_dark)
    print("benches gelu leaves at a hard 0:", hard_zeros_left, "of", HIDDEN,
          "-> tiny non-zero numbers, so a whisper of slope stays")
    # h is continuous noise, so it never lands exactly ON the boundary — which means the
    # counts above cannot tell "< 0" from "<= 0". This hand-made vector sits exactly on it:
    # relu(0) is 0, so that bench IS dark, yet 0 is NOT negative. So the two counts must
    # disagree here (1 vs 2), and predicting darkness from signs alone would undercount.
    edge_h = np.array([-0.5, 0.0, 0.25])
    edge_a = relu(edge_h)
    edge_negatives = count_negative(edge_h)       # 1: only the -0.5
    edge_dark = count_dark(edge_a)                # 2: the -0.5 AND the exact 0
    # The two counts are pinned below, but they are DERIVED from these two arrays, and the
    # arrays are what the reader reads. So the line is rendered once and pinned as text:
    # otherwise edge_a could print a wrong relu (relu(0.25) shown as 0.5) with the counts
    # still agreeing, which is the one thing this boundary case exists to rule out.
    edge_line = ("boundary case %s -> relu %s : negatives %d but dark benches %d"
                 " -> exactly 0 is dark without being negative"
                 % (edge_h, edge_a, edge_negatives, edge_dark))
    print(edge_line)

    # --- Part 5: the honest limit — one word never touches another --------
    stack = np.vstack([x, v, rng.standard_normal(D_MODEL)])   # three words at once
    together = feed_forward(stack, W1, W2)[2]
    edited = stack.copy()
    edited[1] = rng.standard_normal(D_MODEL)      # swap ONLY the middle word
    after = feed_forward(edited, W1, W2)[2]
    own_move = float(np.abs(after[1] - together[1]).max())
    # A mixing workshop would move the neighbours too, so rows 0 and 2 must match.
    neighbour_move = float(np.abs(after[[0, 2]] - together[[0, 2]]).max())
    shown_stack_shape, shown_together_shape = stack.shape, together.shape
    shown_together = np.round(together, 4)
    shown_own_move = round(own_move, 4)
    shown_neighbour_move = round(neighbour_move, 12)
    print("\nthree words at once ->", shown_stack_shape, "in,", shown_together_shape,
          "out\n", shown_together)
    print("swapped word 1 -> its own answer moved", shown_own_move, "and its two",
          "neighbours moved", shown_neighbour_move, "-> no word touched another")

    # --- Self-check: one boolean per claim --------------------------------
    # Every number below was read off a real run and typed in here by hand.
    shapes_ok = (widths == (4, 16, 4) and shown_collapsed_shape == (4, 4)
                 and shown_w1_shape == (4, 16) and shown_w2_shape == (16, 4)
                 and shown_in_shape == (4,) and shown_h_shape == (16,)
                 and shown_a_shape == (16,) and shown_out_shape == (4,))
    x_ok = shown_in.tolist() == [0.1257, -0.1321, 0.6404, 0.1049]
    h_ok = shown_h.tolist() == [-0.4696, -0.1492, 0.6188, 1.0705, -0.9691,
                                0.871, 0.7576, 0.5579, -0.1083, -0.3804,
                                1.0929, 1.1389, 1.1409, 0.7039, 0.4184, -0.5512]
    a_ok = shown_a.tolist() == [0.0, 0.0, 0.6188, 1.0705, 0.0, 0.871, 0.7576, 0.5579,
                                0.0, 0.0, 1.0929, 1.1389, 1.1409, 0.7039, 0.4184, 0.0]
    out_ok = shown_out.tolist() == [-1.5292, 2.2412, -0.6077, 1.1303]
    # The 4x4 collapsed matrix is SQUARE, so pinning only its shape would not notice a
    # transpose or a wrong pair of matrices. Pin what it actually contains.
    collapsed_ok = shown_collapsed.tolist() == [[1.2984, -5.2732, -1.3208, 0.1147],
                                                [2.9633, -0.9484, 3.4569, -1.8946],
                                                [-0.1074, 6.6351, 0.3903, -1.1048],
                                                [-4.9032, -0.0656, -2.1971, -0.6981]]
    no_bend_ok = (shown_one_step.tolist() == [-0.8114, 3.7047, -0.6033, -0.5161]
                  and shown_two_step.tolist() == [-0.8114, 3.7047, -0.6033, -0.5161]
                  and shown_bent.tolist() == [-1.5292, 2.2412, -0.6077, 1.1303])
    bend_breaks_linearity = straight_adds_up and shown_bent_add_gap == 2.0059
    dark_ok = (dark_benches == 6 and predicted_dark == dark_benches
               and shown_negatives.tolist() == [-0.4696, -0.1492, -0.9691,
                                                -0.1083, -0.3804, -0.5512])
    # The boundary case: strict "<" counts 1 where "exactly 0" counts 2. Loosen either
    # comparison and this pair stops holding, which is what the real h cannot check.
    boundary_ok = (edge_negatives == 1 and edge_dark == 2
                   and edge_negatives != edge_dark)
    # The same boundary case as it was RENDERED: both arrays and both counts on one line,
    # so a wrong relu output cannot reach the screen under the two counts still agreeing.
    boundary_line_ok = edge_line == ("boundary case [-0.5   0.    0.25] -> "
                                     "relu [0.   0.   0.25] : negatives 1 but dark "
                                     "benches 2 -> exactly 0 is dark without being negative")
    gelu_leaks = hard_zeros_left == 0
    gelu_values_ok = shown_smooth_dark.tolist() == [-0.15, -0.0658, -0.1613,
                                                    -0.0495, -0.1338, -0.1603]
    swap_did_something = shown_own_move == 3.6047
    neighbours_untouched = neighbour_move == 0.0 and shown_neighbour_move == 0.0
    together_ok = (shown_stack_shape == (3, 4) and shown_together_shape == (3, 4)
                   and shown_together.tolist() == [[-1.5292, 2.2412, -0.6077, 1.1303],
                                                   [-0.5495, -4.489, -0.231, -1.6653],
                                                   [-4.8062, -9.1648, -1.4448, 3.0331]])

    if (shapes_ok and x_ok and h_ok and a_ok and out_ok and collapsed_ok and no_bend_ok
            and collapse_holds and bend_breaks_linearity and dark_ok and boundary_ok
            and boundary_line_ok
            and gelu_leaks and gelu_values_ok and swap_did_something
            and neighbours_untouched and together_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected widths (4, 16, 4); out = [-1.5292 2.2412 -0.6077 "
              "1.1303]; W1@W2 row 0 = [1.2984 -5.2732 -1.3208 0.1147]; the no-bend answer "
              "[-0.8114 3.7047 -0.6033 -0.5161] from either "
              "path; the straight path to add up while the bent path misses by 2.0059; 6 "
              "dark benches of 16; on [-0.5, 0, 0.25] one negative but TWO dark benches; "
              "gelu leaving 0 hard zeros, those 6 at [-0.15 -0.0658 "
              "-0.1613 -0.0495 -0.1338 -0.1603]; and swapping word 1 to move its own "
              "answer 3.6047 while its neighbours move exactly 0")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert shapes_ok, "the widths must go 4 -> 16 -> 4 and W1@W2 must be [4, 4]"
    assert x_ok, "the input word should be [0.1257 -0.1321 0.6404 0.1049]"
    assert h_ok, "the up-projected h does not match the run these numbers are pinned to"
    assert a_ok, "relu(h) should keep the positives and flatten the 6 negatives to 0"
    assert out_ok, "out should be [-1.5292 2.2412 -0.6077 1.1303]"
    assert collapsed_ok, "W1@W2 should start [1.2984 -5.2732 -1.3208 0.1147] — contents, not just shape"
    assert no_bend_ok, "the no-bend answer should be [-0.8114 3.7047 -0.6033 -0.5161]"
    assert collapse_holds, "(x@W1)@W2 must equal x@(W1@W2) — two linear layers are one"
    assert bend_breaks_linearity, "the straight path must add up and the bent path must not"
    assert dark_ok, "relu should switch off 6 of the 16 benches — one per negative in h"
    assert boundary_ok, ("on [-0.5, 0, 0.25] exactly 0 must count as dark but NOT as negative: "
                         "1 negative, 2 dark benches")
    assert boundary_line_ok, ("the boundary line must read 'boundary case [-0.5   0.    0.25] "
                              "-> relu [0.   0.   0.25] : negatives 1 but dark benches 2' — "
                              "the printed relu itself, not just the counts")
    assert gelu_leaks, "gelu must leave no bench at a hard 0"
    assert gelu_values_ok, "gelu on the 6 dark benches should be [-0.15 -0.0658 -0.1613 -0.0495 -0.1338 -0.1603]"
    assert swap_did_something, "swapping word 1 should move its own answer by 3.6047"
    assert neighbours_untouched, "words 0 and 2 must not move when word 1 is swapped"
    assert together_ok, "the three words together should come out [[-1.5292 ...], [-0.5495 ...], [-4.8062 ...]]"
