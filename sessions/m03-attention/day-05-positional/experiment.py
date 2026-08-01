# day-05-positional — experiment
#
# Today's big idea in two lines of output:
#   Plain self-attention cannot see word order — shuffle the words and every
#   word's answer comes back unchanged, only moved to a different row.
#   Add a "seat stamp" (positional encoding) and the order changes the answer.
#
# This script runs a tiny self-attention on "dog bites man", catches it ignoring
# the word order, builds the sine/cosine seat stamps, and adds them in.
# The attention is Day 3's formula with Day 2's rule kept: Q, K and V come from
# three DIFFERENT recipe matrices, so a word's question is never its own answer.
# Run it:  python3 sessions/m03-attention/day-05-positional/experiment.py

import numpy as np  # numpy gives us arrays, matrix multiply (@), and sin/cos


# ---- Day 3's attention, with three separate recipes in Day 2's style -------
# UNMASKED, like every day in this module: the short form softmax(Q @ Kᵀ / √d_k) @ V.
# Day 3's fifth beat, the mask, is not here — a 3-word sentence has nothing to hide.
# The full form is softmax(Q @ Kᵀ / √d_k + mask) @ V; the mask returns in the decoder module.
def softmax(scores):
    # Subtract each row's largest score first so exp() never blows up.
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    # Divide each row by its own total, so every row of shares adds up to 1.
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)

# Each word here is 4 numbers, and the 4 slots read as [animal, action, person, noun].
# The three matrices below play the same three roles as Day 2's W_Q, W_K and W_V, but the
# NUMBERS are hand-written fresh for this 4-slot world — Day 2's slots meant something else
# (water, place, money, action) and its W_V was 4x2 while this one is 4x4. One turns a word
# into its Query ("what I am looking for"), one into its Key ("what I advertise"), one into
# its Value ("what I hand over"). They are different from each other on purpose:
# Day 2's rule is that Q must not be the same table as K.
W_Q = np.array([[0.0, 2.0, 0.0, 0.0],   # an animal looks for the action
                [0.8, 0.0, 2.0, 0.0],   # an action looks for the person, a bit for the animal
                [2.0, 0.4, 0.0, 0.0],   # a person looks for the animal
                [0.0, 0.0, 0.0, 0.0]])  # being a noun asks for nothing extra
W_K = np.array([[2.5, 0.0, 0.0, 0.0],   # an animal advertises "animal"
                [0.0, 2.5, 0.0, 0.0],   # an action advertises "action"
                [0.0, 0.0, 2.5, 0.0],   # a person advertises "person"
                [0.5, 0.0, 0.5, 0.0]])  # a noun advertises a little of both nouns
W_V = np.array([[1.0, 0.0, 0.0, 0.2],   # what an animal hands over
                [0.0, 1.0, 0.0, 0.6],   # what an action hands over
                [0.0, 0.0, 1.0, 0.4],   # what a person hands over
                [0.0, 0.0, 0.0, 0.1]])  # a noun adds a little to the last slot
# d_v (a Value's width) is 4 here, the same as d_k. Day 2 established that it need not be,
# and Day 3 ran it at 2 against a d_k of 4; matching them just keeps this day's tables square.

def attention_parts(x, wq=None, wk=None, wv=None):
    # x has shape (seq, d_model): one row per word, d_model numbers per word.
    # The recipes default to the three above. Part 5 hands in WIDER ones instead,
    # so the sqrt(d_k) scaling is exercised at more than one width.
    wq = W_Q if wq is None else wq
    wk = W_K if wk is None else wk
    wv = W_V if wv is None else wv
    Q, K, V = x @ wq, x @ wk, x @ wv      # three recipes -> three different tables
    d_k = Q.shape[-1]
    # QUERY-FIRST, the module's one layout: Q @ K.T, so ROW i = word i ASKING and column j =
    # word j being offered. And this grid is the SCALED one — Day 3's `scaled_scores`, not
    # its raw `raw_scores` — because the sqrt(d_k) division happens right here.
    scaled_scores = (Q @ K.T) / np.sqrt(d_k)
    weights = softmax(scaled_scores)      # every row of shares adds up to 1
    return Q, K, V, scaled_scores, weights, weights @ V

def self_attention(x):
    # Each word's answer = a blend of all the Values, using that word's own shares.
    return attention_parts(x)[-1]


# ---- The fix: sinusoidal positional encoding ("the seat stamp") ------------
def sinusoidal_encoding(max_len, d_model):
    # One row per seat ("seat" = a word's position in the sentence; "slot" = a column).
    # Counting columns from 0, the EVEN columns (0, 2, 4 …) get a sine wave and the ODD
    # ones a cosine — that is the lesson's PE(pos, 2i) / PE(pos, 2i+1) pair, so this one
    # place counts from 0 while the module's prose counts slots from 1.
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

def ruler_recipe(width, tilt):
    # A width x width recipe with no randomness in it: every entry is a function
    # of its own position, so these numbers are identical on every run.
    grid = np.arange(width * width).reshape(width, width)
    return ((grid + tilt) % 7 + tilt) / 10.0

def ruler_rows(rows, width):
    # Stand-in word vectors of any width, again built from positions only.
    return ((np.arange(rows * width).reshape(rows, width) % 5) - 2) / 4.0


if __name__ == "__main__":
    # --- Part 1: a 3-word sentence and the three recipes -------------------
    # No trained model is cached on this machine, so these word vectors are
    # hand-written and small enough to read: dog is an animal and a noun, bites
    # is an action, man is a person and a noun. Order-blindness does not depend
    # on the values — any numbers show it.
    words = ["dog", "bites", "man"]
    seq, d_model = 3, 4
    x = np.array([[1.0, 0.0, 0.0, 1.0],    # dog:   animal + noun
                  [0.0, 1.0, 0.0, 0.0],    # bites: action
                  [0.0, 0.0, 1.0, 1.0]])   # man:   person + noun
    print("x shape:", x.shape, "(3 words, 4 numbers each)")
    show(words, x)
    Q, K, V, scaled_scores, weights, out_original = attention_parts(x)
    print("Q, K, V shapes:", Q.shape, K.shape, V.shape, "(one row per word)")
    print("is Q the same table as K?", np.array_equal(Q, K),
          "-> must be False, that is Day 2's rule")
    print("SCALED score grid, row i = how word i rates every word:")
    show(words, scaled_scores)
    # If Q were K, the grid would be a mirror across its diagonal and asking
    # "who rates whom" would be one question instead of two. Measure it.
    lopsided = float(np.abs(scaled_scores - scaled_scores.T).max())
    print("biggest gap between score[i][j] and score[j][i]:", round(lopsided, 4),
          "-> the grid is lopsided, so reading it the wrong way round changes the answer")
    # And WHICH way round is ours: query-first, so the ROW is the asker. That is not a detail
    # a 3.0 gap can settle on its own, so name the two cells the gap is made of.
    dog_rates_man = float(scaled_scores[0, 2])
    man_rates_dog = float(scaled_scores[2, 0])
    print("dog rates man", round(dog_rates_man, 4), "but man rates dog", round(man_rates_dog, 4),
          "-> row = the asker, column = the word offered. That 3.0 gap IS these two cells.")

    # --- Part 2: run attention, then predict what a shuffle will do -------
    print("\nshares each word handed out (each row adds up to 1):")
    show(words, weights)
    share_row_sums = weights.sum(axis=-1)
    print("row sums:", share_row_sums, "(one whole budget per asking word — the last axis)")
    print("attention on 'dog bites man' -> shape", out_original.shape)
    show(words, out_original)
    # A blend is only worth looking at if it is not the plain average of the
    # Values. So compare the two, and print how far apart they get.
    plain_average = V.mean(axis=0)
    blend_spread = float(np.abs(out_original - plain_average).max())
    print("plain average of the three Values:", np.round(plain_average, 4))
    print("furthest any answer sits from that average:", round(blend_spread, 4),
          "-> each word picked a favourite instead of averaging")
    # A prediction we COMPUTE, not one we type: attention reads its input as an
    # unordered set, so moving a word should move its answer to the same new row.
    # The three recipes do not save us — they hit every row the same way.
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
    print("column 0 is the FAST wave:", np.round(pe_wide[:4, 0], 2),
          " column 4 is a SLOW wave:", np.round(pe_wide[:4, 4], 2))
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

    # --- Part 5: is the divisor really sqrt(d_k)? --------------------------
    # At d_k = 4 the lesson's sqrt(4) = 2.0 is ALSO d_k/2 and ALSO d_v, so this
    # one sentence cannot tell the scaling law apart from two wrong laws. Run the
    # same attention at two more widths and MEASURE the divisor it used: the
    # unscaled grid divided by the scaled grid that came back.
    print("\ndivisor attention actually used, measured at three widths:")
    measured_divisors = []
    for width in (4, 9, 64):
        wq, wk, wv = (ruler_recipe(width, 1), ruler_recipe(width, 2),
                      ruler_recipe(width, 3))
        rows = ruler_rows(seq, width)
        wide_scores = attention_parts(rows, wq, wk, wv)[3]
        raw = (rows @ wq) @ (rows @ wk).T          # the same grid, never scaled
        measured = float(np.abs(raw).max() / np.abs(wide_scores).max())
        measured_divisors.append(round(measured, 4))
        print("  d_k=%2d  measured %6.3f   sqrt(d_k) %6.3f   d_k/2 %6.3f   d_v %6.3f"
              % (width, measured, np.sqrt(width), width / 2, width))
    print("  -> only sqrt(d_k) matches at all three widths;"
          " d_k/2 and d_v happen to tie with it at d_k=4")

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected value below is written down here, so a broken change in the
    # code above cannot quietly agree with itself.
    expected_dog_out = np.array([0.0705, 0.8590, 0.0705, 0.5718])  # printed in Part 2
    expected_seat0 = np.array([0., 1., 0., 1., 0., 1., 0., 1.])    # sin(0)=0, cos(0)=1
    expected_fast = np.array([0.00, 0.84, 0.91, 0.14])   # the lesson's fast wave
    expected_slow = np.array([0.00, 0.01, 0.02, 0.03])   # the lesson's slow wave
    # Day 2's rule, restated: the two tables must not be the same numbers, and the
    # score grid must not be its own mirror. Both are what makes reading the grid
    # the wrong way round (scores.T) give a different, wrong answer.
    qk_differ = not np.array_equal(Q, K)
    grid_lopsided = float(np.round(lopsided, 6)) == 3.0      # printed in Part 1
    # Which reading is ours, pinned as two named cells rather than left to the gap: dog
    # asking man scores 0.0 while man asking dog scores 3.0. Swap the row and column
    # meanings and those two numbers trade places, so this claim fails on a transpose.
    query_first_ok = (dog_rates_man == 0.0 and man_rates_dog == 3.0
                      and dog_rates_man != man_rates_dog)
    # Every asking word must hand out one whole budget of 1 — the softmax axis, pinned.
    budget_ok = np.allclose(share_row_sums, 1.0) and share_row_sums.shape == (seq,)
    dog_out_ok = np.allclose(out_original[0], expected_dog_out, atol=1e-4)
    # dog sat in row 0 and now sits in row 2 with the SAME four numbers.
    dog_moved_ok = np.allclose(out_shuffled[2], expected_dog_out, atol=1e-4)
    layout_exact = layout_error < 1e-12      # whole table = original, rows reordered
    # The three words must get different answers, or moving them proves nothing.
    answers_differ = abs(smallest_row_gap(out_original) - 0.7207) < 1e-4
    # And each answer must sit far from the plain average, or "attention" did nothing.
    blend_selective = float(np.round(blend_spread, 4)) == 0.5256    # printed in Part 2
    seat0_ok = np.array_equal(pe_wide[0], expected_seat0)
    shapes_ok = (pe_wide.shape == (6, 8) and pe.shape == (6, 4)
                 and x_original.shape == x.shape == (seq, d_model))
    waves_ok = (np.allclose(np.round(pe_wide[:4, 0], 2), expected_fast)
                and np.allclose(np.round(pe_wide[:4, 4], 2), expected_slow))
    seats_unique = abs(seat_gap - 0.7682) < 1e-4   # closest pair of seats, printed above
    fix_ok = abs(fix_gap - 1.7269) < 1e-4          # the number printed in Part 4
    # fix_gap alone cannot say WHICH table is which: [2,1,0] is its own undo, so
    # swapping the two Part-4 runs leaves that gap at 1.7269 while both tables
    # print under the wrong sentence. Pin one row of each — they are not equal.
    expected_dog_seated = np.array([0.9017, 1.1412, 0.1203, 1.0575])   # 'dog bites man', row 0
    expected_man_seated = np.array([1.8366, -0.3416, 0.0474, 0.3795])  # 'man bites dog', row 0
    labels_ok = (np.allclose(out_pos[0], expected_dog_seated, atol=1e-4)
                 and np.allclose(out_pos_shuffled[0], expected_man_seated, atol=1e-4))
    # And the scaling law, measured in Part 5 at three widths from one sqrt.
    scaling_is_sqrt = measured_divisors == [2.0, 3.0, 8.0]

    if (qk_differ and grid_lopsided and query_first_ok and budget_ok and dog_out_ok
            and dog_moved_ok and layout_exact
            and answers_differ and blend_selective and seat0_ok and shapes_ok
            and waves_ok and seats_unique and fix_ok and labels_ok and scaling_is_sqrt):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected Q and K to be different tables with a lopsided score "
              "grid (gap 3.0), dog's answer [0.0705 0.859 0.0705 0.5718] in row 0 and again "
              "in row 2 after the shuffle, every answer up to 0.5256 away from the plain "
              "average, seat 0 = [0 1 0 1 0 1 0 1], fast wave [0. 0.84 0.91 0.14], slow wave "
              "[0. 0.01 0.02 0.03], gap after seats = 1.7269, seated 'dog bites man' row 0 = "
              "[0.9017 1.1412 0.1203 1.0575] while seated 'man bites dog' row 0 = "
              "[1.8366 -0.3416 0.0474 0.3795], and divisors [2.0, 3.0, 8.0] at d_k = 4, 9, 64")


    assert qk_differ, "Q and K must be different tables, or a word only rates itself"
    assert grid_lopsided, "score[i][j] and score[j][i] should differ by up to 3.0"
    assert query_first_ok, ("the grid is QUERY-first: dog asking man scores 0.0 while man asking "
                            "dog scores 3.0, so row = the asker and column = the word offered")
    assert budget_ok, "each of the 3 asking words must hand out one whole budget of 1"
    assert dog_out_ok, "dog's answer should be [0.0705 0.859 0.0705 0.5718]"
    assert answers_differ, "the three words' answers must differ by 0.7207 at the closest"
    assert blend_selective, "the furthest answer should sit 0.5256 from the plain average"
    assert dog_moved_ok, "after the shuffle dog's answer should reappear in row 2, unchanged"
    assert layout_exact, "shuffled output must equal the original reordered by [2,1,0]"
    assert shapes_ok, "stamps should be 6x8 and 6x4, and x + pe[:seq] must stay (3, 4)"
    assert seat0_ok, "seat 0 should be exactly [0 1 0 1 0 1 0 1]"
    assert waves_ok, "column 0 should be [0 .84 .91 .14] and column 4 [0 .01 .02 .03]"
    assert seats_unique, "the closest two seat stamps should differ by 0.7682"
    assert fix_ok, "with seats, the order-blind prediction should miss by 1.7269"
    assert labels_ok, ("seated 'dog bites man' row 0 should be [0.9017 1.1412 0.1203 1.0575] "
                       "and seated 'man bites dog' row 0 [1.8366 -0.3416 0.0474 0.3795]")
    assert scaling_is_sqrt, "the measured divisor should be sqrt(d_k): 2.0, 3.0, 8.0"
