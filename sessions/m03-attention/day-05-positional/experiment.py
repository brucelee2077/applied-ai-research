# day-05-positional — experiment
#
# Today's big idea in two lines of output:
#   Plain self-attention cannot see word order — shuffle the words and every
#   word's answer comes back unchanged, only moved to a different row.
#   Add a "seat stamp" (positional encoding) and the order changes the answer.
#
# This script runs a tiny self-attention on "dog bites man", catches it ignoring
# the word order, builds the sine/cosine seat stamps, and adds them in.
# Run it:  python3 sessions/m03-attention/day-05-positional/experiment.py

import numpy as np  # numpy gives us arrays, matrix multiply (@), and sin/cos


# ---- The Day-3 recipe, unchanged ------------------------------------------
def softmax(scores):
    # Subtract each row's largest score first so exp() never blows up.
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    # Divide each row by its own total, so every row of weights adds up to 1.
    return weights / weights.sum(axis=-1, keepdims=True)

def self_attention(x):
    # x has shape (seq, d_model): one row per word, d_model numbers per word.
    # Q, K and V are all x itself here, so only the words are in play.
    d_k = x.shape[1]
    scores = (x @ x.T) / np.sqrt(d_k)   # word i rates word j, scaled by sqrt(d_k)
    return softmax(scores) @ x          # each word's answer = a blend of all words


# ---- The fix: sinusoidal positional encoding ("the seat stamp") ------------
def sinusoidal_encoding(max_len, d_model):
    # One row per seat. Even slots get a sine wave, odd slots get a cosine wave.
    pe = np.zeros((max_len, d_model))
    positions = np.arange(max_len)[:, None]      # 0, 1, 2, ... as a column
    pair_index = np.arange(d_model // 2)         # which pair of slots we are in
    # The 10000 knob spreads the wave speeds: front pairs fast, back pairs slow.
    div = np.power(10000.0, (2 * pair_index) / d_model)
    pe[:, 0::2] = np.sin(positions / div)
    pe[:, 1::2] = np.cos(positions / div)
    return pe

def smallest_row_gap(matrix):
    # How far apart are the two most similar rows? 0 means two rows are equal.
    return min(np.abs(matrix[i] - matrix[j]).max()
               for i in range(len(matrix)) for j in range(i + 1, len(matrix)))

def show(labels, matrix):
    # Print one labelled row per line, rounded so the table stays readable.
    for label, row in zip(labels, matrix):
        print("  %-6s %s" % (label, np.round(row, 4)))


if __name__ == "__main__":
    # --- Part 1: a 3-word sentence with stand-in word vectors --------------
    # No trained model is cached on this machine, so these word vectors are made
    # up with a fixed seed. Order-blindness does not depend on the values.
    words = ["dog", "bites", "man"]
    seq, d_model = 3, 4
    x = np.round(np.random.default_rng(0).random((seq, d_model)), 2)
    print("x shape:", x.shape, "(3 words, 4 numbers each)")
    show(words, x)

    # --- Part 2: run attention, then predict what a shuffle will do -------
    out_original = self_attention(x)
    print("\nattention on 'dog bites man' -> shape", out_original.shape)
    show(words, out_original)
    # A prediction we COMPUTE, not one we type: attention reads its input as an
    # unordered set, so moving a word should move its answer to the same new row.
    new_seats = [2, 1, 0]                        # "man bites dog": swap seat 0 and 2
    predicted = out_original[new_seats]          # what that idea says we will get
    out_shuffled = self_attention(x[new_seats])  # what we actually get
    print("\nattention on 'man bites dog' -> shape", out_shuffled.shape)
    show([words[k] for k in new_seats], out_shuffled)
    layout_error = float(np.abs(out_shuffled - predicted).max())
    print("gap between prediction and reality:", round(layout_error, 12),
          "-> rows moved, values did not: attention is order-blind")

    # --- Part 3: build the seat stamps ------------------------------------
    pe_wide = sinusoidal_encoding(max_len=6, d_model=8)
    print("\nseat stamps, 6 seats x 8 slots -> shape", pe_wide.shape)
    show(["seat %d" % s for s in range(len(pe_wide))], pe_wide)
    print("slot 0 is the FAST wave:", np.round(pe_wide[:4, 0], 2),
          " slot 4 is a SLOW wave:", np.round(pe_wide[:4, 4], 2))
    seat_gap = smallest_row_gap(pe_wide)
    print("closest two seats differ by", round(seat_gap, 4), "-> each seat is unique")

    # --- Part 4: add the stamp, and order starts to matter ----------------
    pe = sinusoidal_encoding(max_len=6, d_model=d_model)   # 4 slots wide, to match x
    x_original = x + pe[:seq]              # "dog bites man" sitting in seats 0,1,2
    x_shuffled = x[new_seats] + pe[:seq]   # "man bites dog" in the SAME three seats
    print("\nx + pe[:seq] shape:", x_original.shape, "-> ADDED, so the width stays", d_model)
    out_pos = self_attention(x_original)
    out_pos_shuffled = self_attention(x_shuffled)
    print("with seats, 'dog bites man':")
    show(words, out_pos)
    print("with seats, 'man bites dog':")
    show([words[k] for k in new_seats], out_pos_shuffled)
    # The same prediction rule as Part 2 — this time it should FAIL.
    fix_gap = float(np.abs(out_pos_shuffled - out_pos[new_seats]).max())
    print("gap between prediction and reality now:", round(fix_gap, 4),
          "-> the order-blind prediction broke: the seats are being read")

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected value below is written down here, so a broken change in the
    # code above cannot quietly agree with itself.
    expected_dog_out = np.array([0.6682, 0.7230, 0.5042, 0.2696])  # printed in Part 2
    expected_seat0 = np.array([0., 1., 0., 1., 0., 1., 0., 1.])    # sin(0)=0, cos(0)=1
    expected_fast = np.array([0.00, 0.84, 0.91, 0.14])   # the lesson's fast wave
    expected_slow = np.array([0.00, 0.01, 0.02, 0.03])   # the lesson's slow wave
    dog_out_ok = np.allclose(out_original[0], expected_dog_out, atol=1e-4)
    # dog sat in row 0 and now sits in row 2 with the SAME four numbers.
    dog_moved_ok = np.allclose(out_shuffled[2], expected_dog_out, atol=1e-4)
    layout_exact = layout_error < 1e-12      # whole table = original, rows reordered
    # The three words must get different answers, or moving them proves nothing.
    answers_differ = abs(smallest_row_gap(out_original) - 0.0503) < 1e-4
    seat0_ok = np.array_equal(pe_wide[0], expected_seat0)
    shapes_ok = (pe_wide.shape == (6, 8) and pe.shape == (6, 4)
                 and x_original.shape == x.shape == (seq, d_model))
    waves_ok = (np.allclose(np.round(pe_wide[:4, 0], 2), expected_fast)
                and np.allclose(np.round(pe_wide[:4, 4], 2), expected_slow))
    seats_unique = abs(seat_gap - 0.7682) < 1e-4   # closest pair of seats, printed above
    fix_ok = abs(fix_gap - 0.4230) < 1e-4          # the number printed in Part 4

    if (dog_out_ok and dog_moved_ok and layout_exact and answers_differ
            and seat0_ok and shapes_ok and waves_ok and seats_unique and fix_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected dog's answer [0.6682 0.723 0.5042 0.2696] in row 0 and "
              "again in row 2 after the shuffle, seat 0 = [0 1 0 1 0 1 0 1], fast wave "
              "[0. 0.84 0.91 0.14], slow wave [0. 0.01 0.02 0.03], gap after seats = 0.4230")

    assert dog_out_ok, "dog's answer should be [0.6682 0.723 0.5042 0.2696]"
    assert answers_differ, "the three words' answers must differ by 0.0503 at the closest"
    assert dog_moved_ok, "after the shuffle dog's answer should reappear in row 2, unchanged"
    assert layout_exact, "shuffled output must equal the original reordered by [2,1,0]"
    assert shapes_ok, "stamps should be 6x8 and 6x4, and x + pe[:seq] must stay (3, 4)"
    assert seat0_ok, "seat 0 should be exactly [0 1 0 1 0 1 0 1]"
    assert waves_ok, "slot 0 should be [0 .84 .91 .14] and slot 4 [0 .01 .02 .03]"
    assert seats_unique, "the closest two seat stamps should differ by 0.7682"
    assert fix_ok, "with seats, the order-blind prediction should miss by 0.4230"
