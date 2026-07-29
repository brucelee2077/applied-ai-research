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

def feed_forward(x, W1, W2, show=False):
    # Position-wise: the SAME two matrices run on each word's vector on its own.
    h = x @ W1       # step 1 up-projection: d_model -> hidden (spread out wide)
    a = relu(h)      # step 2 the bend, applied in the wide space (the thinking)
    out = a @ W2     # step 3 down-projection: hidden -> d_model (carry it home)
    if show:
        for name, arr in (("in", x), ("h=x@W1", h), ("relu(h)", a), ("out=a@W2", out)):
            print("  %-9s" % name, arr.shape, "->", np.round(arr, 4))
    return h, a, out


if __name__ == "__main__":
    # --- Part 1: the two matrices of the workshop -------------------------
    x = rng.standard_normal(D_MODEL)              # one word, D_MODEL numbers wide
    W1 = rng.standard_normal((D_MODEL, HIDDEN))   # up-projection   [4, 16]
    W2 = rng.standard_normal((HIDDEN, D_MODEL))   # down-projection [16, 4]
    print("d_model =", D_MODEL, " hidden =", HIDDEN, " W1", W1.shape, " W2", W2.shape)

    # --- Part 2: push one word through, watching the width ----------------
    print("\none word through the workshop — watch the width go 4 -> 16 -> 4:")
    h, a, out = feed_forward(x, W1, W2, show=True)
    widths = (x.size, h.size, out.size)
    print("  widths   ", widths, "-> as wide out as in, so it fits the residual stream")

    # --- Part 3: with no bend, the two matrices collapse into one ---------
    straight = lambda vec: (vec @ W1) @ W2       # no bend at all, just both matrices
    bent = lambda vec: relu(vec @ W1) @ W2       # the real workshop, ReLU in the middle
    W_collapsed = W1 @ W2                        # ONE matrix doing the no-bend path
    print("\nW1 @ W2 = one matrix,", W_collapsed.shape, ":\n", np.round(W_collapsed, 4))
    one_matrix = x @ W_collapsed
    collapse_holds = np.allclose(straight(x), one_matrix)
    print("(x@W1)@W2     =", np.round(straight(x), 4))
    print("x@(W1@W2)     =", np.round(one_matrix, 4), "-> same numbers:", collapse_holds)
    print("relu(x@W1)@W2 =", np.round(bent(x), 4), "-> the bend changed the answer")

    # One matched answer could still be luck, so test the property that DEFINES a
    # single matrix: it adds up, g(u+v) == g(u) + g(v). Fail that even once and NO
    # single matrix can reproduce the bent path.
    u, v = x, rng.standard_normal(D_MODEL)        # two different words
    straight_adds_up = np.allclose(straight(u + v), straight(u) + straight(v))
    bent_add_gap = float(np.abs(bent(u + v) - (bent(u) + bent(v))).max())
    print("straight path adds up:", straight_adds_up, " bent path off by:",
          round(bent_add_gap, 4), "-> the bend cannot be folded into one matrix")

    # --- Part 4: dark benches, and the gentler bend that saves them -------
    # Predict the dark benches by counting negatives in h, then see if relu agrees.
    predicted_dark = int(np.count_nonzero(h < 0))
    dark_benches = int(np.count_nonzero(a == 0.0))
    print("\nnegative values in the wide space:", np.round(h[h < 0], 4))
    print("dark benches predicted from those signs:", predicted_dark,
          " benches relu set to exactly 0:", dark_benches, "of", HIDDEN)
    smooth = gelu(h)                              # the same h, bent gently instead
    dark_idx = np.flatnonzero(a == 0.0)           # which benches relu switched off
    hard_zeros_left = int(np.count_nonzero(smooth == 0.0))
    print("gelu on those same benches:", np.round(smooth[dark_idx], 4))
    print("benches gelu leaves at a hard 0:", hard_zeros_left, "of", HIDDEN,
          "-> tiny non-zero numbers, so a whisper of slope stays")

    # --- Part 5: the honest limit — one word never touches another --------
    stack = np.vstack([x, v, rng.standard_normal(D_MODEL)])   # three words at once
    together = feed_forward(stack, W1, W2)[2]
    edited = stack.copy()
    edited[1] = rng.standard_normal(D_MODEL)      # swap ONLY the middle word
    after = feed_forward(edited, W1, W2)[2]
    own_move = float(np.abs(after[1] - together[1]).max())
    # A mixing workshop would move the neighbours too, so rows 0 and 2 must match.
    neighbour_move = float(np.abs(after[[0, 2]] - together[[0, 2]]).max())
    print("\nthree words at once ->", stack.shape, "in,", together.shape, "out\n",
          np.round(together, 4))
    print("swapped word 1 -> its own answer moved", round(own_move, 4), "and its two",
          "neighbours moved", round(neighbour_move, 12), "-> no word touched another")

    # --- Self-check: one boolean per claim --------------------------------
    # Every number below was read off a real run and typed in here by hand.
    shapes_ok = widths == (4, 16, 4) and W_collapsed.shape == (4, 4)
    h_ok = np.round(h, 4).tolist() == [-0.4696, -0.1492, 0.6188, 1.0705, -0.9691,
                                       0.871, 0.7576, 0.5579, -0.1083, -0.3804,
                                       1.0929, 1.1389, 1.1409, 0.7039, 0.4184, -0.5512]
    out_ok = np.round(out, 4).tolist() == [-1.5292, 2.2412, -0.6077, 1.1303]
    no_bend_ok = np.round(one_matrix, 4).tolist() == [-0.8114, 3.7047, -0.6033, -0.5161]
    bend_breaks_linearity = straight_adds_up and round(bent_add_gap, 4) == 2.0059
    dark_ok = dark_benches == 6 and predicted_dark == dark_benches
    gelu_leaks = hard_zeros_left == 0
    gelu_values_ok = np.round(smooth[dark_idx], 4).tolist() == [-0.15, -0.0658, -0.1613,
                                                               -0.0495, -0.1338, -0.1603]
    swap_did_something = round(own_move, 4) == 3.6047
    neighbours_untouched = neighbour_move == 0.0

    if (shapes_ok and h_ok and out_ok and no_bend_ok and collapse_holds
            and bend_breaks_linearity and dark_ok and gelu_leaks and gelu_values_ok
            and swap_did_something and neighbours_untouched):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected widths (4, 16, 4); out = [-1.5292 2.2412 -0.6077 "
              "1.1303]; the no-bend answer [-0.8114 3.7047 -0.6033 -0.5161] from either "
              "path; the straight path to add up while the bent path misses by 2.0059; 6 "
              "dark benches of 16; gelu leaving 0 hard zeros, those 6 at [-0.15 -0.0658 "
              "-0.1613 -0.0495 -0.1338 -0.1603]; and swapping word 1 to move its own "
              "answer 3.6047 while its neighbours move exactly 0")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert shapes_ok, "the widths must go 4 -> 16 -> 4 and W1@W2 must be [4, 4]"
    assert h_ok, "the up-projected h does not match the run these numbers are pinned to"
    assert out_ok, "out should be [-1.5292 2.2412 -0.6077 1.1303]"
    assert no_bend_ok, "the no-bend answer should be [-0.8114 3.7047 -0.6033 -0.5161]"
    assert collapse_holds, "(x@W1)@W2 must equal x@(W1@W2) — two linear layers are one"
    assert bend_breaks_linearity, "the straight path must add up and the bent path must not"
    assert dark_ok, "relu should switch off 6 of the 16 benches — one per negative in h"
    assert gelu_leaks, "gelu must leave no bench at a hard 0"
    assert gelu_values_ok, "gelu on the 6 dark benches should be [-0.15 -0.0658 -0.1613 -0.0495 -0.1338 -0.1603]"
    assert swap_did_something, "swapping word 1 should move its own answer by 3.6047"
    assert neighbours_untouched, "words 0 and 2 must not move when word 1 is swapped"
