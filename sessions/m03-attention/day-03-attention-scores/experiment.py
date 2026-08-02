# day-03-attention-scores — experiment
#
# Today's big idea in two lines of output:
#   Nine raw match numbers turn into one budget of shares that adds up to 1.
#   Spend that budget on the Values and "bank" comes out river-flavoured: [7.63, 2.85].
# We walk the lesson's own 3-word sentence "the river bank" through all five beats: score
# every pair -> divide by sqrt(d_k) -> softmax -> mask -> blend the Values. Four of those
# five fit on one line of maths:  softmax(Q @ K.T / sqrt(d_k)) @ V
# The mask is the fifth, and it is NOT in that line — a 3-word sentence with nothing to
# hide needs no mask, so beat 4 runs on its own 4-score row in Part 5. Written in full the
# line is softmax(Q @ K.T / sqrt(d_k) + mask) @ V, and Days 4 and 5 keep using the short
# unmasked form; the mask comes back for real in the decoder module.
# Run it:  python3 sessions/m03-attention/day-03-attention-scores/experiment.py

import numpy as np  # numpy holds the lists as arrays and does every dot product at once with @


def hand_dot(list_a, list_b):
    # The dot product done the paper way: multiply slot by slot, then add the products.
    total = 0.0
    for slot_a, slot_b in zip(list_a, list_b):
        total += slot_a * slot_b
    return total


def hand_score_matrix(queries, keys):
    # One hand-made dot product per PAIR, in the module's QUERY-FIRST layout (the same one
    # Days 2, 4 and 5 use): row = the word asking, column = the word offered.
    return np.array([[hand_dot(query, key) for key in keys] for query in queries])


def ranking_with_values(names, row):
    # Who beats whom in ONE row, weakest first: each word's name paired with the number it
    # actually holds in THAT row. Carrying the number matters. Two different rows can share
    # the same order, so a list of names alone cannot tell you which row it came from.
    return [(names[i], float(row[i])) for i in np.argsort(row, kind="stable")]


def scale_by_sqrt(score_block, width):
    # The ONE place the "scaled" in scaled dot-product attention is spelled out: divide by
    # sqrt(width). Not by width, and not by half of width — those only look the same at
    # width 4, which is the single width the lesson's tiny world uses.
    return score_block / np.sqrt(width)


def softmax(scores):
    # Turn EVERY ROW of scores into positive shares that add up to 1. The axis is the last
    # one — the same spelling Days 2, 4 and 5 use — so a grid keeps one whole budget PER
    # ROW instead of one budget shared across the whole grid.
    # Taking each row's biggest score away first does not change the answer (softmax
    # reads only the gaps between scores) and it stops exp() from overflowing.
    grown = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return grown / np.sum(grown, axis=-1, keepdims=True)


if __name__ == "__main__":
    # --- Part 1: the tiny world, exactly the numbers from the lesson ---------
    words = ["the", "river", "bank"]
    # Slot numbering follows the lesson's sliders and is 1-BASED: slot 1 is the FIRST
    # column of a row, so "a lot of slot 1" puts the 2.0 at index 0. (A "slot" is always
    # a column here; a word's position in the sentence is a "seat". Day 4 counts the same
    # way; Day 5's sine/cosine formula indexes columns from 0 instead, and says so.)
    Q = np.array([[0.0, 0.0, 1.0, 1.0],     # "the"   asks for nothing in particular
                  [1.0, 0.0, 1.0, 1.0],     # "river" asks a little for slot-1 things
                  [2.0, 1.0, 1.0, 1.0]])    # "bank"  asks hard for slot 1, a bit for slot 2
    K = np.array([[0.0, 0.0, 1.0, 1.0],     # "the"   offers only the two shared slots
                  [2.0, 0.0, 1.0, 1.0],     # "river" offers a lot of slot 1 (column index 0)
                  [0.0, 2.0, 1.0, 1.0]])    # "bank"  offers a lot of slot 2 (column index 1)
    V = np.array([[0.0, 6.0], [10.0, 2.0], [4.0, 4.0]])   # what each word hands over
    d_k = Q.shape[-1]                       # d_k = how many slots each Query and Key has
    # Day 2's free choice, still in force: a Value is never compared to anything, so its
    # width d_v does not have to equal d_k. Here it does not — 2 against 4.
    d_v = V.shape[-1]
    # Bind every printed number to a name and check THAT name in the self-check, so the
    # line the reader sees and the line the check reads are one value, not two guesses.
    q_shape, k_shape, v_shape = Q.shape, K.shape, V.shape
    sqrt_d_k = np.sqrt(d_k)
    print("Q", q_shape, " K", k_shape, " V", v_shape, " d_k =", d_k, " sqrt(d_k) =", sqrt_d_k)
    print("d_v =", d_v, "and d_k =", d_k, "-> a Value's width is still a free choice")

    # --- Part 2: score every pair (the score matrix) -------------------------
    raw_scores = Q @ K.T                    # every Query row meets every Key column
    by_hand = hand_score_matrix(Q, K)       # the same nine numbers, one paper dot product at a time
    hand_agrees = np.array_equal(raw_scores, by_hand)
    bank_leans_on = words[int(np.argmax(raw_scores[2]))]
    print("\n--- Part 2: the score matrix (row = who asks, column = who is offered) ---")
    print(raw_scores)
    print("the @ shortcut and the by-hand loop agree:", hand_agrees)
    # A row and the word it leans on only mean something TOGETHER, and that pairing exists
    # nowhere but the rendered line: pinning the row and the word separately still lets the
    # print call wrap either one. So build the line, pin the text, then print it.
    bank_row_line = "bank's row %s -> bank leans on %s" % (raw_scores[2], bank_leans_on)
    print(bank_row_line)
    # Which way round matters. Spell it KEY-first — K @ Q.T — and you get the TRANSPOSE:
    # a grid that looks just as reasonable and answers a different question.
    key_first = K @ Q.T
    key_first_leans_on = words[int(np.argmax(key_first[2]))]
    key_first_line = ("key-first K @ Q.T gives bank's row %s -> bank would lean on %s "
                      "instead. Query first, always."
                      % (key_first[2], key_first_leans_on))
    print(key_first_line)

    # --- Part 3: turn the volume down by dividing by sqrt(d_k) ---------------
    scaled_scores = scale_by_sqrt(raw_scores, d_k)
    print("\n--- Part 3: scaled scores (every cell divided by sqrt(d_k)) ---")
    print(scaled_scores)
    # The LAW, not the value: at d_k = 4 the right divisor sqrt(4) = 2 is byte-identical to
    # d_k/2 and to d_v, so this one width cannot tell a right divisor from a wrong one. Run
    # the SAME scaling helper on the SAME literal row at two more widths, where they split.
    probe_row = np.array([2.0, 6.0, 4.0])       # bank's raw row again, one shared literal
    probe_widths = (4, 9, 64)
    sqrt_divisors = [float(np.sqrt(w)) for w in probe_widths]
    half_divisors = [w / 2 for w in probe_widths]
    width_list = list(probe_widths)
    # The row's label is rendered from the row itself, so the caption cannot advertise a
    # row the scaling never saw.
    probe_label = "[" + ",".join(str(int(cell)) for cell in probe_row) + "]"
    probe_rows = [np.round(scale_by_sqrt(probe_row, w), 4) for w in probe_widths]
    print("widths          :", width_list)
    # Two laws on one line: which list sits under which heading IS the teaching, so swapping
    # them would advertise width/2 as the real divisor while both lists stay correct.
    divisor_line = ("sqrt(width)     : %s  width/2 (the wrong law): %s"
                    % (sqrt_divisors, half_divisors))
    print(divisor_line)
    # The width and the row it produced, paired on the page. Build the three lines, pin the
    # list, then print them — that one claim covers a mismatched width and a dropped row.
    probe_lines = ["   {} / sqrt({:<2}) -> {}".format(probe_label, width, got)
                   for width, got in zip(probe_widths, probe_rows)]
    for probe_line in probe_lines:
        print(probe_line)
    # Who beats whom, weakest first — twice, from two different rows of numbers.
    raw_row = raw_scores[2]                  # bank's row BEFORE the division: [2, 6, 4]
    scaled_row = scaled_scores[2]            # bank's row AFTER  the division: [1, 3, 2]
    rank_before = ranking_with_values(words, raw_row)
    rank_after = ranking_with_values(words, scaled_row)
    names_before = [pair[0] for pair in rank_before]
    names_after = [pair[0] for pair in rank_after]
    same_order = names_before == names_after
    same_numbers = rank_before == rank_after
    print("ranking before:", rank_before)
    print("ranking after :", rank_after)
    print("same order:", same_order,
          " same numbers:", same_numbers,
          "(division moved every number and left the order alone)")

    # --- Part 4: softmax splits one budget into shares -----------------------
    weights = softmax(scaled_scores)             # the whole grid at once, row by row
    row_by_row = np.array([softmax(row) for row in scaled_scores])   # one row at a time
    weights_shown = np.round(weights, 4)
    row_sums = weights.sum(axis=-1)
    grid_equals_rowwise = np.array_equal(weights, row_by_row)
    print("\n--- Part 4: softmax over each row (the attention weights) ---")
    print(weights_shown)
    print("row sums:", row_sums, "(one whole budget per asking word)")
    print("whole grid at once == one row at a time:", grid_equals_rowwise)
    # A softmax with NO axis divides by the total of all nine cells instead, so no row
    # gets a whole budget. Same exp, one wrong denominator — measure what it costs.
    no_axis = np.exp(scaled_scores - np.max(scaled_scores))
    no_axis = no_axis / no_axis.sum()
    no_axis_row_sums = np.round(no_axis.sum(axis=-1), 4)
    no_axis_bank_row = np.round(no_axis[2], 4)
    print("with no axis at all, the row sums would be:", no_axis_row_sums,
          "and bank's row", no_axis_bank_row, "-> not one budget per word")
    # PREDICT from the RAW scores, before reading any share: the biggest score must win
    # the biggest share, because softmax re-scales a row and never re-orders it.
    predicted_winner = bank_leans_on          # the raw-row argmax, named once in Part 2
    actual_winner = words[int(np.argmax(weights[2]))]
    print("predicted biggest share:", predicted_winner, " actual:", actual_winner)

    # --- Part 5: mask two words so nobody can listen to them -----------------
    # Beat 4 of the pipeline, and the one beat the short one-line formula leaves out.
    # Today's 3-word sentence has nothing to hide, so the mask runs on its own row here.
    mask_labels = ["the", "river", "FUTURE", "pad"]
    mask_scores = np.array([1.0, 3.0, 2.0, 0.0])            # the lesson's 4-score row
    forbidden = np.array([False, False, True, True])         # FUTURE and pad are off limits
    open_shares = softmax(mask_scores)                       # what happens with no mask
    masked_scores = np.where(forbidden, -1e9, mask_scores)   # huge negative score BEFORE softmax
    masked_shares = softmax(masked_scores)
    biggest_masked_share = float(np.abs(masked_shares[forbidden]).max())
    # Why "huge" and not merely "low": run the same mask again at only -12 and measure the leak.
    mild_shares = softmax(np.where(forbidden, -12.0, mask_scores))
    mild_leak = float(np.abs(mild_shares[forbidden]).max())
    # The leak is SHOWN in scientific notation, so bind that rendered text and check it —
    # "bigger than zero" would let any leak size print under a passing tick.
    mild_leak_text = "{:.2e}".format(mild_leak)
    open_shown = np.round(open_shares, 4)
    open_sum = round(float(open_shares.sum()), 6)
    masked_shown = np.round(masked_shares, 4)
    masked_sum = round(float(masked_shares.sum()), 6)
    change_shown = np.round(masked_shares - open_shares, 4)
    print("\n--- Part 5: masking (a forbidden word gets a huge negative score) ---")
    print("labels  :", mask_labels)
    # Three rows of one small table, and the whole point is WHICH row is the masked one.
    # Swap the first two and the page teaches that masking leaves a share on a forbidden
    # word, with both arrays still individually correct — so pin the rendered rows.
    mask_lines = ["no mask : %s  sum %s" % (open_shown, open_sum),
                  "masked  : %s  sum %s" % (masked_shown, masked_sum),
                  "change  : %s (freed budget moves to allowed words)" % (change_shown,)]
    for mask_line in mask_lines:
        print(mask_line)
    # The same for the two leak lines: their order is the claim that -1e9 is clean while a
    # merely-low -12 still leaks. Swapped, they say exactly the opposite.
    leak_lines = ["biggest share left on a forbidden word, mask -1e9 : %s"
                  % (biggest_masked_share,),
                  "biggest share left on a forbidden word, mask -12   : %s (a mask that is"
                  " only a bit negative still leaks budget)" % (mild_leak_text,)]
    for leak_line in leak_lines:
        print(leak_line)

    # --- Part 6: spend the budget on the Values -----------------------------
    bank_shares = weights[2]
    blended = bank_shares @ V               # each share times its Value, added slot by slot
    # PREDICT from the shares alone: the answer must land nearest the Value of the word
    # holding most of the budget ("nearest" = shortest straight-line distance).
    predicted_nearest = words[int(np.argmax(bank_shares))]
    distances = np.linalg.norm(V - blended, axis=-1)
    actual_nearest = words[int(np.argmin(distances))]
    bank_shares_shown = np.round(bank_shares, 4)
    blended_shown = np.round(blended, 4)
    distances_shown = np.round(distances, 3)
    print("\n--- Part 6: blend the Values by bank's shares ---")
    # Today's headline, so pin it as the text it prints: shares in, blend out.
    blend_line = "shares %s @ V -> %s" % (bank_shares_shown, blended_shown)
    print(blend_line)
    print("predicted nearest Value:", predicted_nearest, " actual:", actual_nearest,
          " distances:", distances_shown)
    print("all three rows of softmax(Q @ K.T / sqrt(d_k)) @ V:")
    all_outputs = weights @ V                # one blended list per asking word, not just bank's
    all_outputs_shown = np.round(all_outputs, 4)
    # One row per asking word: the NAME beside the numbers is what makes the table readable,
    # so pin the rendered rows and a reordered or dropped row cannot slip through.
    output_lines = ["   {:<6} -> {}".format(name, row)
                    for name, row in zip(words, all_outputs_shown)]
    for output_line in output_lines:
        print(output_line)

    # --- Self-check: one boolean per claim ----------------------------------
    # Every expected value is written down here, so nothing is checked against a re-derived number.
    lesson_grid = np.array([[2.0, 2.0, 2.0], [2.0, 4.0, 2.0], [2.0, 6.0, 4.0]])
    grid_ok = (np.array_equal(raw_scores, lesson_grid) and np.array_equal(by_hand, lesson_grid)
               and hand_agrees is True and bank_leans_on == "river"
               and q_shape == (3, 4) and k_shape == (3, 4) and v_shape == (3, 2))
    # Query-first is the whole grid's meaning, so pin the wrong spelling too: key-first is
    # the transpose, and on bank's row it hands the win to bank itself instead of river.
    orientation_ok = (np.array_equal(key_first, lesson_grid.T)
                      and np.array_equal(key_first[2], np.array([2.0, 2.0, 4.0]))
                      and key_first_leans_on == "bank"
                      and not np.array_equal(raw_scores, key_first))
    value_width_ok = d_v == 2 and d_k == 4 and d_v != d_k
    scale_ok = (sqrt_d_k == 2.0 and np.array_equal(scaled_scores[2], np.array([1.0, 3.0, 2.0]))
                and np.array_equal(scaled_scores,
                                   np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 3.0, 2.0]])))
    # sqrt(width) and width/2 agree ONLY at width 4, so pin the divisor at three widths.
    law_ok = (sqrt_divisors == [2.0, 3.0, 8.0]
              and half_divisors == [2.0, 4.5, 32.0]
              and width_list == [4, 9, 64] and probe_label == "[2,6,4]"
              # the same three rows the loop above printed, pinned as printed
              and np.array_equal(probe_rows[0], np.array([1.0, 3.0, 2.0]))
              and np.array_equal(probe_rows[1], np.array([0.6667, 2.0, 1.3333]))
              and np.array_equal(probe_rows[2], np.array([0.25, 0.75, 0.5]))
              and np.allclose(scale_by_sqrt(probe_row, 4), [1.0, 3.0, 2.0])
              and np.allclose(scale_by_sqrt(probe_row, 9), [0.66666667, 2.0, 1.33333333])
              and np.allclose(scale_by_sqrt(probe_row, 64), [0.25, 0.75, 0.5]))
    # The order must survive the division, and the two rankings must really be the before
    # and the after: the names match each other, while the numbers pin each list to its own
    # row ([2,4,6] only exists before the division, [1,2,3] only after).
    order_ok = (names_before == ["the", "bank", "river"]
                and names_after == names_before
                and same_order is True and same_numbers is False
                and rank_before == [("the", 2.0), ("bank", 4.0), ("river", 6.0)]
                and rank_after == [("the", 1.0), ("bank", 2.0), ("river", 3.0)])
    shares_ok = (np.allclose(weights[2], [0.09003057, 0.66524096, 0.24472847])
                 and np.array_equal(weights_shown[2], np.array([0.09, 0.6652, 0.2447])))
    # Every asking word gets its OWN budget, so pin all three rows, not just bank's: a row
    # that quietly copies bank's shares still sums to 1 and would slip past budget_ok.
    rows_ok = (np.allclose(weights[0], [0.33333333, 0.33333333, 0.33333333])
               and np.allclose(weights[1], [0.21194156, 0.57611688, 0.21194156])
               and np.array_equal(weights_shown[0], np.array([0.3333, 0.3333, 0.3333]))
               and np.array_equal(weights_shown[1], np.array([0.2119, 0.5761, 0.2119]))
               and np.allclose(all_outputs[0], [4.66666667, 4.0])
               and np.allclose(all_outputs[1], [6.60893508, 3.27164935])
               and np.array_equal(all_outputs_shown[0], np.array([4.6667, 4.0]))
               and np.array_equal(all_outputs_shown[1], np.array([6.6089, 3.2716]))
               and np.array_equal(all_outputs_shown[2], blended_shown)
               and np.allclose(all_outputs[2], blended))
    # The printed row sums read "[1. 1. 1.]", so pin them at the precision that line shows —
    # exact float equality would fail on the last bit and teach nothing.
    budget_ok = (np.allclose(row_sums, 1.0)
                 and np.array_equal(np.round(row_sums, 6), np.array([1.0, 1.0, 1.0])))
    # The AXIS itself: the whole-grid call must equal the row-by-row one, and the no-axis
    # version must NOT — otherwise "one budget per asking word" is an untested promise.
    axis_ok = (grid_equals_rowwise is True
               and np.array_equal(no_axis_row_sums, np.array([0.1594, 0.2506, 0.59]))
               and not np.allclose(no_axis.sum(axis=-1), 1.0)
               and np.array_equal(no_axis_bank_row, np.array([0.0531, 0.3925, 0.1444])))
    winner_ok = predicted_winner == "river" and actual_winner == "river"
    open_ok = (np.allclose(open_shares, [0.08714432, 0.64391426, 0.23688282, 0.03205860])
               and np.array_equal(open_shown, np.array([0.0871, 0.6439, 0.2369, 0.0321]))
               and open_sum == 1.0 and mask_labels == ["the", "river", "FUTURE", "pad"])
    # A forbidden word must keep NOTHING, not merely something that rounds to nothing.
    mask_ok = (np.allclose(masked_shares[:2], [0.11920292, 0.88079708])
               and biggest_masked_share == 0.0
               and np.array_equal(masked_shown, np.array([0.1192, 0.8808, 0.0, 0.0]))
               and masked_sum == 1.0
               and np.allclose(masked_shares.sum(), 1.0))
    # And the size of the negative number is the reason it is nothing: -12 leaks 2.69e-07.
    # The band is two-sided: "bigger than 1e-9" alone would let a leak of 0.9 print here.
    magnitude_ok = (1e-9 < mild_leak < 1e-5 and mild_leak_text == "2.69e-07"
                    and biggest_masked_share == 0.0)
    freed_ok = (masked_shares[0] > open_shares[0] and masked_shares[1] > open_shares[1]
                and np.array_equal(change_shown,
                                   np.array([0.0321, 0.2369, -0.2369, -0.0321])))
    blend_ok = (np.allclose(blended, [7.63132344, 2.84957923])
                and np.array_equal(blended_shown, np.array([7.6313, 2.8496]))
                and np.array_equal(bank_shares_shown, np.array([0.09, 0.6652, 0.2447])))
    nearest_ok = predicted_nearest == "river" and actual_nearest == "river"
    # "Nearest" is only evidence if the whole distance table is right: one distance PER word,
    # each pinned. A distance list that collapsed to the wrong length still has an argmin.
    distance_ok = (distances.shape == (3,)
                   and np.array_equal(distances_shown, np.array([8.256, 2.516, 3.809]))
                   and np.allclose(distances, [8.25604315, 2.51642872, 3.80919649]))
    # --- the RENDERED TEXT of the lines whose meaning lives in their layout -------------
    # A value bound to a name is still free at the print CALL: two swapped operands, or one
    # of them wrapped in arithmetic, leave every name untouched. So the lines that carry a
    # pairing or a direction are pinned as the exact strings that reach the screen.
    printed_orientation_ok = (
        bank_row_line == "bank's row [2. 6. 4.] -> bank leans on river"
        and key_first_line == ("key-first K @ Q.T gives bank's row [2. 2. 4.] -> bank would "
                               "lean on bank instead. Query first, always."))
    printed_law_ok = (
        divisor_line == ("sqrt(width)     : [2.0, 3.0, 8.0]  width/2 (the wrong law): "
                         "[2.0, 4.5, 32.0]")
        and probe_lines == ["   [2,6,4] / sqrt(4 ) -> [1. 3. 2.]",
                            "   [2,6,4] / sqrt(9 ) -> [0.6667 2.     1.3333]",
                            "   [2,6,4] / sqrt(64) -> [0.25 0.75 0.5 ]"])
    printed_mask_ok = (
        mask_lines == ["no mask : [0.0871 0.6439 0.2369 0.0321]  sum 1.0",
                       "masked  : [0.1192 0.8808 0.     0.    ]  sum 1.0",
                       "change  : [ 0.0321  0.2369 -0.2369 -0.0321] (freed budget moves to "
                       "allowed words)"]
        and leak_lines == ["biggest share left on a forbidden word, mask -1e9 : 0.0",
                           "biggest share left on a forbidden word, mask -12   : 2.69e-07 "
                           "(a mask that is only a bit negative still leaks budget)"])
    printed_blend_ok = (
        blend_line == "shares [0.09   0.6652 0.2447] @ V -> [7.6313 2.8496]"
        and output_lines == ["   the    -> [4.6667 4.    ]",
                             "   river  -> [6.6089 3.2716]",
                             "   bank   -> [7.6313 2.8496]"])

    if (grid_ok and orientation_ok and value_width_ok and scale_ok and law_ok and order_ok
            and shares_ok and rows_ok and budget_ok and axis_ok
            and winner_ok and open_ok and mask_ok and magnitude_ok and freed_ok and blend_ok
            and nearest_ok and distance_ok
            and printed_orientation_ok and printed_law_ok and printed_mask_ok
            and printed_blend_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected the grid [[2,2,2],[2,4,2],[2,6,4]], bank's scaled row [1,3,2] "
              "ranked the(1) -> bank(2) -> river(3), the same order as the raw the(2) -> bank(4) -> river(6), "
              "the divisor sqrt(width) = 2/3/8 at widths 4/9/64 (never width/2 = 2/4.5/32), "
              "shares [0.0900,0.6652,0.2447] summing to 1, its own budget for each row "
              "([0.3333,0.3333,0.3333] for the, [0.2119,0.5761,0.2119] for river), masked shares "
              "[0.1192,0.8808] with an "
              "exact 0.0 on both forbidden words (a mild -12 mask leaks 2.69e-07 instead), "
              "and the blend [7.6313,2.8496] nearest river's Value at distances [8.256,2.516,3.809]")

    assert grid_ok, "Q @ K.T must give [[2,2,2],[2,4,2],[2,6,4]], and so must the by-hand loop"
    assert orientation_ok, ("the grid is QUERY-first: key-first K @ Q.T is its transpose, giving "
                            "bank the row [2,2,4] and handing the win to 'bank' instead of 'river'")
    assert value_width_ok, ("a Value's width is still free — d_v is 2 here while d_k is 4, "
                            "exactly the choice Day 2 established")
    assert scale_ok, "dividing by sqrt(4)=2 must turn bank's row [2,6,4] into [1,3,2]"
    assert law_ok, ("the divisor must be sqrt(width) at every width — 2, 3, 8 at widths 4, 9, 64 — "
                    "so width/2 (2, 4.5, 32) is only right by accident at width 4")
    assert order_ok, ("dividing by sqrt(d_k) must keep the ranking the -> bank -> river, holding "
                      "[2, 4, 6] before the division and [1, 2, 3] after it")
    assert shares_ok, "softmax([1,3,2]) must give [0.0900, 0.6652, 0.2447]"
    assert rows_ok, ("each asking word must get its own budget — [0.3333,0.3333,0.3333] for the and "
                     "[0.2119,0.5761,0.2119] for river — blending to [4.6667,4.0] and [6.6089,3.2716]")
    assert budget_ok, "every row of shares must add up to 1 — one whole budget"
    assert axis_ok, ("softmax must normalise the LAST AXIS: the whole-grid call must equal the "
                     "row-by-row one, and a no-axis softmax must fail it with row sums "
                     "[0.1594, 0.2506, 0.59] and bank's row [0.0531, 0.3925, 0.1444]")
    assert winner_ok, "the biggest score must also win the biggest share"
    assert open_ok, "unmasked softmax([1,3,2,0]) must give [0.0871, 0.6439, 0.2369, 0.0321]"
    assert mask_ok, ("masking before softmax must give [0.1192, 0.8808] to the allowed words, "
                     "exactly 0.0 to every forbidden word, and still sum to 1")
    assert magnitude_ok, ("the size of the negative matters: a -12 mask must still leak budget "
                          "while -1e9 leaves a clean 0.0")
    assert freed_ok, "the budget freed by masking must flow to the allowed words"
    assert blend_ok, "bank's shares times the Values must give [7.6313, 2.8496]"
    assert nearest_ok, "the blend must land nearest river's Value, the word holding most budget"
    assert distance_ok, ("there must be one distance per word — [8.2560, 2.5164, 3.8092] — so a "
                         "collapsed or wrong-axis distance list cannot supply the argmin")
    assert printed_orientation_ok, ("the page must read \"bank's row [2. 6. 4.] -> bank leans on "
                                    "river\", and the key-first line must hand the win to 'bank'")
    assert printed_law_ok, ("the printed law lines must keep sqrt(width) and width/2 under their "
                           "own headings, and pair each width with the row it produced")
    assert printed_mask_ok, ("the masked row on the page must be the one with 0. on both "
                            "forbidden words, and the -1e9 line the one reading 0.0")
    assert printed_blend_ok, ("the page must read 'shares [0.09   0.6652 0.2447] @ V -> "
                             "[7.6313 2.8496]' and name each of the three output rows")
