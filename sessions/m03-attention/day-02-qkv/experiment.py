# day-02-qkv — experiment  (Module 4 · Day 2 — Query, Key, Value)
#
# Today's big idea in two lines of output:
#   The word "bank" holds one question, scores it against every word's label,
#   and blends the winners' content -> the new "bank" leans river, not money.
#
# The sentence is the lesson's own: "the animal crossed the river bank". We walk
# the six stations of self-attention on it: embed, project into Query/Key/Value,
# score, scale by sqrt(d_k), softmax into shares, blend the Values. Every number
# printed here also sits on the lesson page, so each one can check the other.
# Run it:  python3 sessions/m03-attention/day-02-qkv/experiment.py

import numpy as np  # numpy gives us arrays and matrix multiply (@)


def softmax(scores):
    """Turn EVERY ROW of scores into positive shares that add up to 1 — "one pizza".

    The axis is the last one, so a grid keeps one whole budget per asking ROW. A
    plain 1-D row is just a grid with one row, which is all today's demo needs;
    Days 3, 4 and 5 hand it a whole grid, and the same axis keeps them right.
    We slide every row down by its own largest score first, so exp() cannot
    overflow. The shift does not change the answer: softmax reads only the gaps.
    """
    slices = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return slices / slices.sum(axis=-1, keepdims=True)   # each row adds to 1


def scale_by_sqrt_width(scores, width):
    """The "scaled" in scaled dot-product: divide by sqrt(width).

    Every caller below goes through this one function, so the law itself is under
    test — and it is tested at three widths, because at our d_k = 4 the right
    divisor sqrt(4) = 2.0 is byte-identical to half the width and to d_v.
    """
    return scores / np.sqrt(width)


def matches(computed, expected):
    """True when a computed number or grid equals a written-down one (6 decimals)."""
    return np.array_equal(np.round(np.asarray(computed, dtype=float), 6),
                          np.asarray(expected, dtype=float))


if __name__ == "__main__":
    WORDS = ["the", "animal", "crossed", "the", "river", "bank"]
    bank, river = 5, 4          # the two words we follow all day

    # --- Part 1: the sentence as six embedding cards ------------------------
    # Each ROW is one word's embedding — its profile card, exactly the kind of row
    # Day 1 looked up. The NUMBERS are hand-written fresh for today: Day 1's table
    # had a different vocabulary and different made-up axes (is-animal, is-furry,
    # is-vehicle, is-fast), so nothing here is Day 1's row for the same word. What
    # carries over is the idea — one row per word. A card is four numbers wide, and
    # that width has a name: d_model. The four slots (slot = COLUMN of a row) are
    # made-up meaning axes:  water  place  money  action
    x = np.array([[0.0, 0.2, 0.0, 0.0],    # 0 "the"     — filler, points almost nowhere
                  [0.0, 0.2, 0.0, 1.0],    # 1 "animal"  — a doer
                  [0.0, 0.4, 0.0, 1.0],    # 2 "crossed" — an action
                  [0.0, 0.2, 0.0, 0.0],    # 3 "the"     — filler again
                  [1.0, 1.0, 0.0, 0.0],    # 4 "river"   — watery AND a place
                  [0.0, 1.0, 1.0, 0.0]])   # 5 "bank"    — a place, unsure about money
    # Bind every printed number to a name and check THAT name below, so the line the
    # reader sees and the line the self-check reads can never be two different sums.
    x_shape = x.shape
    d_model = x_shape[-1]
    print("Part 1 — embeddings x, shape", x_shape, " d_model =", d_model)

    # --- Part 2: one card, three DIFFERENT recipes -> Q, K, V ---------------
    # A recipe is a grid you multiply the card by: row = input slot, column =
    # output slot. Days 4 and 5 call these three grids recipes too. A real model
    # learns them; ours are hand-picked.
    # W_Q makes the QUERY (the question a word asks), W_K the KEY (the label it
    # advertises), W_V the VALUE (the content it shares if someone listens).
    W_Q = np.array([[0.1, 0.1, 0.0, 0.0],  # W_Q water slot -> mild interest in water+place
                    [0.5, 0.5, 0.0, 0.0],  #     place slot  -> asks about water and place
                    [0.5, 0.0, 0.0, 0.0],  #     money slot  -> asks harder about water
                    [0.0, 0.0, 0.5, 0.5]]) #     action slot -> asks about money and action
    W_K = np.array([[4.4, 0.0, 0.0, 0.0],  # W_K water slot -> "I am watery"
                    [0.0, 2.0, 0.0, 0.0],  #     place slot  -> "I am a place"
                    [0.0, 0.0, 4.0, 0.0],  #     money slot  -> "I am about money"
                    [0.0, 0.0, 0.0, 4.0]]) #     action slot -> "I am an action"
    # W_V has only two columns: a Value is never compared to anything, so its
    # width d_v is a free choice and does not have to equal d_k.
    W_V = np.array([[0.9, 0.0],            # W_V water slot -> watery content
                    [0.1, 0.1],            #     place slot  -> a little of both
                    [0.0, 0.9],            #     money slot  -> money content
                    [0.2, 0.0]])           #     action slot -> a little watery content
    Q, K, V = x @ W_Q, x @ W_K, x @ W_V
    d_k, d_v = Q.shape[-1], V.shape[-1]
    q_shape, k_shape, v_shape = Q.shape, K.shape, V.shape
    print("Part 2 — Q", q_shape, " K", k_shape, " V", v_shape,
          " (d_k =", d_k, ", d_v =", d_v, ")")
    # Three DIFFERENT recipes is today's point, and it lives in the COLUMN ORDER of this
    # table: read Q under the K label and a word's question becomes its label. So build the
    # six rows as text FIRST, pin the whole list of rows below, and print them from that
    # list. One claim then covers a swapped column, a shifted word and a dropped row.
    qkv_rows = ["         %-8s Q=%s K=%s V=%s" % (word, Q[i], K[i], V[i])
                for i, word in enumerate(WORDS)]
    for qkv_row in qkv_rows:
        print(qkv_row)
    # Two different grids, so a word's question and its label are different lists.
    q_equals_k = np.array_equal(Q, K)
    print("         is Q the same numbers as K?", q_equals_k, "-> must be False")
    # W_K above is a pure diagonal, so x @ W_K and x @ W_K.T print the same numbers —
    # today's Key grid cannot show that "row = input slot, column = output slot" is
    # the rule. Pin the rule on a lopsided stand-in (day-05's W_K looks like this):
    # let the money row also feed the water column, and the two orders disagree.
    W_K_lopsided = W_K.copy()
    W_K_lopsided[2, 0] = 1.0               # money row -> water column
    bank_key_lopsided = x[bank] @ W_K_lopsided
    bank_key_column_first = x[bank] @ W_K_lopsided.T   # the wrong reading, bound once
    print("         lopsided W_K: bank's Key =", bank_key_lopsided,
          " but read column-first it would be", bank_key_column_first)

    # --- Part 3: "bank" scores its Query against every Key ------------------
    # A dot product multiplies two lists slot by slot and adds the products up.
    # A big total means "your label answers my question".
    # THE CONVENTION, spelled once for the whole module and used again on Days 3, 4
    # and 5: the score grid is QUERY-FIRST, scores = Q @ K.T, so ROW = the word
    # ASKING and COLUMN = the word being offered. Today only "bank" asks, so what we
    # want is bank's ROW of that grid: Q[bank] @ K.T.
    raw_scores = Q[bank] @ K.T             # bank's row of the grid: one score per word
    print("\nPart 3 — bank's raw match scores:", raw_scores)
    # WHICH word holds the 5.4 is the whole finding, and that pairing only exists in the
    # rendered row. Build the six rows, pin them as text, then print them.
    score_rows = ["         bank -> %-8s = %.1f" % (word, raw_scores[i])
                  for i, word in enumerate(WORDS)]
    for score_row in score_rows:
        print(score_row)
    # Looking is one-way traffic: W_Q and W_K are different grids, so
    # "bank -> river" and "river -> bank" come out as two different numbers.
    river_to_bank = Q[river] @ K[bank]     # river asks, bank answers -> Query first again
    # The DIRECTION is the claim, and a direction lives in the order of the two numbers on
    # the line — swap them and the page says looking back is five times stronger than
    # looking forward. So render the line once, pin the text, then print it.
    traffic_line = ("         bank->river = %.1f  but river->bank = %.1f"
                    % (raw_scores[river], river_to_bank))
    print(traffic_line)
    # The whole grid, so the convention has a LAYOUT to point at, not just a habit.
    raw_grid = Q @ K.T
    raw_grid_shape = raw_grid.shape
    bank_to_river_cell = raw_grid[bank, river]     # the two promised cells, read from the grid
    river_to_bank_cell = raw_grid[river, bank]
    bank_row_matches = np.array_equal(raw_grid[bank], raw_scores)
    print("         full grid Q @ K.T:", raw_grid_shape,
          "row = the asker, column = the word offered")
    # The ORIENTATION claim is nothing but the pairing of a number with an address, so the
    # rendered line is the evidence: swap the two addresses and the page teaches the
    # transpose while every value stays individually correct.
    cells_line = ("         so bank->river %.1f sits at [%d, %d] and river->bank %.1f at "
                  "[%d, %d]; row %d IS bank's row: %s"
                  % (bank_to_river_cell, bank, river, river_to_bank_cell, river, bank,
                     bank, bank_row_matches))
    print(cells_line)
    # Spelling it KEY-first instead gives K @ Q.T — the TRANSPOSE, a different table.
    # It looks just as plausible and it moves bank's biggest match off "river".
    key_first_grid = K @ Q.T
    key_first_best = int(np.argmax(key_first_grid[bank]))
    key_first_best_word = WORDS[key_first_best]
    key_first_line = ("         key-first K @ Q.T row %d = %s -> biggest match '%s', "
                      "not 'river'" % (bank, key_first_grid[bank], key_first_best_word))
    print(key_first_line)
    # Reconciling the two spellings, because the lesson's own Produce step writes the
    # ONE-ASKER form K @ Q[5]: for a single asker the two agree exactly, since a dot
    # product does not care which list comes first. It is only when a whole GRID is built
    # that the choice matters — and then the module's choice is Q @ K.T, every time.
    one_asker_key_first = K @ Q[bank]
    one_asker_agrees = np.array_equal(one_asker_key_first, raw_scores)
    # A verdict word: print `not one_asker_agrees` and the page says the two spellings
    # disagree for one asker, which is the opposite of what this paragraph teaches.
    produce_line = ("         the Produce step's K @ Q[%d] gives the same row: %s -> one "
                    "asker, no difference; a whole grid, the transpose"
                    % (bank, one_asker_agrees))
    print(produce_line)

    # --- Part 4: scale the raw scores by sqrt(d_k) --------------------------
    # d_k is the WIDTH of a Key — how many slots it has. (Careful: Day 1 used the
    # word "length" for how long an arrow is. That is a different quantity, and it
    # had a different cure, cosine. Attention keeps the RAW dot product and cures
    # the width instead; cosine attention is a real alternative, not an oversight.)
    # A dot product adds one product per slot, so WIDE cards give big scores, and
    # big scores make softmax hand everything to one word. Dividing by sqrt(d_k)
    # cancels that growth — the "scaled" in the name.
    scaled_scores = scale_by_sqrt_width(raw_scores, d_k)
    sqrt_d_k = float(np.sqrt(d_k))
    # Swap the two numbers on this line and the page reads "sqrt(2) = 4.0" — an arithmetic
    # absurdity that no value check can see, because both values are right on their own.
    sqrt_line = ("\nPart 4 — divide every score by sqrt(d_k) = sqrt(%d) = %.1f"
                 % (d_k, sqrt_d_k))
    print(sqrt_line)
    print("         scaled scores:", scaled_scores, " (same ranking, calmer numbers)")
    # Careful: at d_k = 4 the right divisor sqrt(4) = 2.0 is the same number as half
    # the width (4/2) and as d_v (2), so this width alone proves nothing about the
    # law. Push river's same 5.4 through the SAME function at two wider Keys, where
    # the three candidate laws split apart.
    WIDTHS = (4, 9, 64)
    river_raw = raw_scores[river]
    river_scaled = [float(scale_by_sqrt_width(river_raw, w)) for w in WIDTHS]
    # The two LOSING laws are printed too, so bind them as well — an unchecked contrast
    # number is exactly as misleading as an unchecked headline number.
    sqrt_law_divisors = [float(np.sqrt(w)) for w in WIDTHS]
    half_width_law = [float(river_raw / (w / 2)) for w in WIDTHS]
    d_v_law = [float(river_raw / d_v) for _ in WIDTHS]
    # Three laws sit side by side on each row, so the COLUMN a number lands in is what says
    # which law it belongs to. Build the rows, pin the list as text, then print them: a
    # swapped column would hand the sqrt law the half-width number and still print three
    # individually correct figures.
    law_rows = ["         d_k = %2d -> sqrt(d_k) = %.1f, river's %.1f becomes %.4f"
                "  (half-width law: %.4f, d_v law: %.4f)"
                % (w, divisor, river_raw, got, half, dv)
                for w, divisor, got, half, dv
                in zip(WIDTHS, sqrt_law_divisors, river_scaled, half_width_law, d_v_law)]
    for law_row in law_rows:
        print(law_row)

    # --- Part 5: softmax the scaled scores into shares ----------------------
    weights = softmax(scaled_scores)
    winner = int(np.argmax(weights))
    weights_shown = np.round(weights, 3)
    weights_total = round(float(weights.sum()), 6)
    winner_word = WORDS[winner]
    river_share_pct = round(float(weights[river] * 100), 1)   # the victory line's headline
    print("\nPart 5 — attention weights (shares):", weights_shown)
    print("         they add up to:", weights_total)
    print("         biggest slice: word %d = '%s'" % (winner, winner_word))
    # Today only one word asks, so a softmax with no axis at all would look right
    # here — and would be wrong the moment a whole grid arrives on Day 3. Hand it
    # a TWO-row block and check each row separately: with the last axis, both rows
    # spend a whole budget; sharing one budget across the block gives 0.5 each.
    two_rows = np.array([scaled_scores, scaled_scores[::-1]])
    two_row_sums = softmax(two_rows).sum(axis=-1)
    whole_block = np.exp(two_rows - two_rows.max())
    whole_block_sums = (whole_block / whole_block.sum()).sum(axis=-1)
    print("         two rows in -> row sums", two_row_sums,
          " (one shared budget would give", whole_block_sums, "instead)")

    # --- Part 6: blend the Values by those shares ---------------------------
    # Weighted sum: pour in each word's Value scaled by its share, then add up.
    output = weights @ V
    output_shown = np.round(output, 3)
    print("\nPart 6 — the new context-aware 'bank':", output_shown)
    # The blend table is evidence too: bind each printed row, then check that the six of
    # them really do add up to the output printed above.
    contributions = [np.round(weights[i] * V[i], 3) for i in range(len(WORDS))]
    # Pin the rendered rows, not only the numbers behind them: which word a share is paired
    # with is the table's meaning, and a shifted label prints river's 0.706 next to "the".
    blend_rows = ["         %.3f x V(%-7s) = %s" % (share, word, contribution)
                  for word, share, contribution
                  in zip(WORDS, weights_shown, contributions)]
    for blend_row in blend_rows:
        print(blend_row)
    contributions_total = np.round(np.sum(contributions, axis=0), 3)
    dist_to_river = round(float(np.linalg.norm(output - V[river])), 2)
    dist_to_old_bank = round(float(np.linalg.norm(output - V[bank])), 2)
    # The conclusion of the whole day is which of these two distances is the SMALLER one, so
    # swapping them would say the new "bank" stayed a money-bank. Pin the rendered line.
    distance_line = ("         distance to river's Value %.2f   to bank's old Value %.2f"
                     % (dist_to_river, dist_to_old_bank))
    print(distance_line)

    # --- Part 7: change the question, and the winner changes ----------------
    # The lesson's slider, in code. Keep every Key exactly as it is, and let
    # "bank" ask about money instead of water (slots: water, place, money, action).
    money_query = np.array([0.0, 0.5, 1.0, 0.0])
    money_scores = money_query @ K.T       # Query first again: one row of the grid
    money_winner = int(np.argmax(money_scores))
    money_winner_word = WORDS[money_winner]
    print("\nPart 7 — same Keys, a money-flavoured question:", money_scores)
    print("         winner is now word %d = '%s' — a money-bank, not a river-bank"
          % (money_winner, money_winner_word))

    # --- Self-check: the numbers the lesson says you should see -------------
    # Every expected value below is WRITTEN DOWN here (most are quoted on the lesson
    # page), never re-derived from the code above — so a wrong filter or a skipped
    # step cannot quietly agree with it.
    # These three grids are the lesson's own printed rows: bank asks [1.0, 0.5, 0, 0],
    # river shouts [4.4, 2.0, 0, 0], and their Values are [1.0, 0.1] and [0.1, 1.0].
    exp_Q = [[0.1, 0.1, 0, 0], [0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.5, 0.5],
             [0.1, 0.1, 0, 0], [0.6, 0.6, 0, 0], [1.0, 0.5, 0, 0]]
    exp_K = [[0, 0.4, 0, 0], [0, 0.4, 0, 4.0], [0, 0.8, 0, 4.0],
             [0, 0.4, 0, 0], [4.4, 2.0, 0, 0], [0, 2.0, 4.0, 0]]
    exp_V = [[0.02, 0.02], [0.22, 0.02], [0.24, 0.04],
             [0.02, 0.02], [1.0, 0.1], [0.1, 1.0]]
    # The blend table as the reader sees it, written down: share x Value, per word.
    exp_contributions = [[0.001, 0.001], [0.012, 0.001], [0.014, 0.002],
                         [0.001, 0.001], [0.706, 0.071], [0.008, 0.078]]
    # The summary a learner remembers, rendered ONCE here so the claims below can pin the
    # text itself — not just the two numbers it happens to quote.
    victory_line = ("\n✅ you got it — 'bank' put %.1f%% of its attention on 'river', and its"
                    " new vector %s leans watery, not money. That is self-attention."
                    % (river_share_pct, output_shown))
    claims = [
        (x_shape == (6, 4) and d_model == 4 and q_shape == (6, 4) and k_shape == (6, 4)
         and v_shape == (6, 2) and (d_k, d_v) == (4, 2),
         "expected x (6,4) with d_model 4, Q (6,4), K (6,4), V (6,2), d_k 4, d_v 2"),
        (matches(Q, exp_Q) and matches(K, exp_K) and matches(V, exp_V),
         "one card through three different grids should give the lesson's Q, K and V rows"),
        (q_equals_k is False, "Q and K must differ, or the score grid turns symmetric"),
        (matches(bank_key_lopsided, [1.0, 2.0, 4.0, 0.0])
         and matches(bank_key_column_first, [0.0, 2.0, 4.0, 0.0])
         and not np.array_equal(bank_key_lopsided, bank_key_column_first),
         "row = input slot: through a lopsided W_K bank's Key is [1. 2. 4. 0.], "
         "and reading the grid column-first gives a different list, [0. 2. 4. 0.]"),
        (matches(raw_scores, [0.2, 0.2, 0.4, 0.2, 5.4, 1.0]),
         "raw scores should be [0.2 0.2 0.4 0.2 5.4 1.0], with river the 5.4"),
        (matches(river_to_bank, 1.2), "river->bank should score 1.2, not bank->river's 5.4"),
        # The ORIENTATION, pinned on the day's own two promised numbers: 5.4 must sit at
        # [bank, river] and 1.2 at [river, bank]. The key-first spelling is the transpose,
        # so it fails both and moves bank's biggest match to 'crossed'.
        (raw_grid_shape == (6, 6) and bank_row_matches is True
         and matches(bank_to_river_cell, 5.4) and matches(river_to_bank_cell, 1.2)
         and np.allclose(key_first_grid, raw_grid.T)
         and matches(key_first_grid[bank], [0.2, 2.2, 2.4, 0.2, 1.2, 1.0])
         and key_first_best == 2 and key_first_best_word == "crossed"
         # one asker: both spellings agree; a whole grid: they are transposes
         and one_asker_agrees is True
         and not np.array_equal(key_first_grid, raw_grid),
         "the grid is Q @ K.T with row = the asker, so bank->river 5.4 sits at [5, 4] "
         "and river->bank 1.2 at [4, 5]; the Produce step's one-asker K @ Q[5] gives the "
         "same row, but the key-first GRID K @ Q.T is its transpose and makes bank's "
         "biggest match 'crossed'"),
        (matches(scaled_scores, [0.1, 0.1, 0.2, 0.1, 2.7, 0.5]) and sqrt_d_k == 2.0,
         "scaled scores should be [0.1 0.1 0.2 0.1 2.7 0.5] — raw divided by 2"),
        (matches(np.round(river_scaled, 4), [2.7, 1.8, 0.675])
         and matches(river_raw, 5.4) and matches(sqrt_law_divisors, [2.0, 3.0, 8.0])
         # the two losing laws are printed as the contrast, so pin their numbers too
         and matches(half_width_law, [2.7, 1.2, 0.16875])
         and matches(d_v_law, [2.7, 2.7, 2.7]),
         "the divisor is sqrt(d_k), so river's 5.4 becomes 2.7 / 1.8 / 0.675 at "
         "d_k = 4 / 9 / 64 — half the width would say 2.7 / 1.2 / 0.169, d_v 2.7 every time"),
        (bool(np.all(weights >= 0)) and weights_total == 1.0,
         "every share must be positive and the six must add up to 1"),
        # The AXIS, pinned: two rows in must come back as two whole budgets of 1, not
        # one budget of 1 split across the block (which would print 0.5 and 0.5).
        (matches(two_row_sums, [1.0, 1.0]) and matches(whole_block_sums, [0.5, 0.5]),
         "softmax must normalise the LAST AXIS: a two-row block gives row sums "
         "[1. 1.], where one shared budget would give [0.5 0.5]"),
        (matches(weights_shown, [0.052, 0.052, 0.058, 0.052, 0.706, 0.078]),
         "shares should be [0.052 0.052 0.058 0.052 0.706 0.078]"),
        (winner == 4 and winner_word == "river", "the biggest share should land on 'river'"),
        (matches(output_shown, [0.742, 0.154]), "output should be [0.742, 0.154]"),
        # The printed blend table must be the printed output, item by item and in total.
        (matches(contributions, exp_contributions)
         and matches(contributions_total, [0.742, 0.154])
         and np.array_equal(contributions_total, output_shown),
         "each printed share x Value row should match the lesson's blend table, and the "
         "six rows should add up to the printed output [0.742 0.154]"),
        (matches(dist_to_river, 0.26) and matches(dist_to_old_bank, 1.06),
         "output should sit 0.26 from river's Value and 1.06 from bank's old Value"),
        (money_winner == 5 and money_winner_word == "bank"
         and matches(money_scores, [0.2, 0.2, 0.4, 0.2, 1.0, 5.0]),
         "a money question should move the win to 'bank', scores [.. 1.0 5.0]"),
        (river_share_pct == 70.6,
         "the victory line's headline share should be river's 70.6%"),
        # --- the RENDERED TEXT of every line whose meaning lives in its layout ---------
        # A value pinned to a name is still free at the print CALL: `print(shown_x * 2)` and
        # a pair of swapped operands both leave the names untouched. These claims pin the
        # exact strings that reach the screen, so a swap, a re-wrap or a shifted label on
        # the lines that carry today's teaching fails instead of printing.
        (qkv_rows == [
            "         the      Q=[0.1 0.1 0.  0. ] K=[0.  0.4 0.  0. ] V=[0.02 0.02]",
            "         animal   Q=[0.1 0.1 0.5 0.5] K=[0.  0.4 0.  4. ] V=[0.22 0.02]",
            "         crossed  Q=[0.2 0.2 0.5 0.5] K=[0.  0.8 0.  4. ] V=[0.24 0.04]",
            "         the      Q=[0.1 0.1 0.  0. ] K=[0.  0.4 0.  0. ] V=[0.02 0.02]",
            "         river    Q=[0.6 0.6 0.  0. ] K=[4.4 2.  0.  0. ] V=[1.  0.1]",
            "         bank     Q=[1.  0.5 0.  0. ] K=[0. 2. 4. 0.] V=[0.1 1. ]",
         ],
         "the Q/K/V table must reach the page with each word's question under Q, its label "
         "under K and its content under V — the column order IS the lesson here"),
        (score_rows == [
            "         bank -> the      = 0.2",
            "         bank -> animal   = 0.2",
            "         bank -> crossed  = 0.4",
            "         bank -> the      = 0.2",
            "         bank -> river    = 5.4",
            "         bank -> bank     = 1.0",
         ],
         "the printed score rows must pair 5.4 with 'river' and 1.0 with 'bank'"),
        (traffic_line == "         bank->river = 5.4  but river->bank = 1.2",
         "the one-way-traffic line must read bank->river 5.4 BUT river->bank 1.2 — swap the "
         "two and the page claims looking back is the stronger direction"),
        (cells_line == ("         so bank->river 5.4 sits at [5, 4] and river->bank 1.2 at "
                        "[4, 5]; row 5 IS bank's row: True"),
         "the orientation line must put 5.4 at [5, 4] and 1.2 at [4, 5]"),
        (key_first_line == ("         key-first K @ Q.T row 5 = [0.2 2.2 2.4 0.2 1.2 1. ] "
                           "-> biggest match 'crossed', not 'river'"),
         "the key-first line must show the transposed row and name 'crossed' as its winner"),
        (produce_line == ("         the Produce step's K @ Q[5] gives the same row: True -> "
                          "one asker, no difference; a whole grid, the transpose"),
         "the Produce-step line must print True — for one asker the two spellings agree"),
        (sqrt_line == "\nPart 4 — divide every score by sqrt(d_k) = sqrt(4) = 2.0",
         "the scaling line must read sqrt(4) = 2.0, not sqrt(2) = 4.0"),
        (law_rows == [
            "         d_k =  4 -> sqrt(d_k) = 2.0, river's 5.4 becomes 2.7000"
            "  (half-width law: 2.7000, d_v law: 2.7000)",
            "         d_k =  9 -> sqrt(d_k) = 3.0, river's 5.4 becomes 1.8000"
            "  (half-width law: 1.2000, d_v law: 2.7000)",
            "         d_k = 64 -> sqrt(d_k) = 8.0, river's 5.4 becomes 0.6750"
            "  (half-width law: 0.1688, d_v law: 2.7000)",
         ],
         "the scaling-law table must reach the page with each law under its own heading — "
         "1.8000 for sqrt at d_k 9, 1.2000 for half-width, 2.7000 for d_v"),
        (blend_rows == [
            "         0.052 x V(the    ) = [0.001 0.001]",
            "         0.052 x V(animal ) = [0.012 0.001]",
            "         0.058 x V(crossed) = [0.014 0.002]",
            "         0.052 x V(the    ) = [0.001 0.001]",
            "         0.706 x V(river  ) = [0.706 0.071]",
            "         0.078 x V(bank   ) = [0.008 0.078]",
         ],
         "the blend table must pair the 0.706 share and the [0.706 0.071] contribution "
         "with 'river'"),
        (distance_line == ("         distance to river's Value 0.26   to bank's old Value "
                           "1.06"),
         "the distance line must read 0.26 to river and 1.06 to bank's old Value — swapped, "
         "it says the new 'bank' never moved"),
        (victory_line == ("\n✅ you got it — 'bank' put 70.6% of its attention on 'river', "
                          "and its new vector [0.742 0.154] leans watery, not money. That is "
                          "self-attention."),
         "the ✅ sentence must reach the page with river's 70.6% and the vector "
         "[0.742 0.154]"),
    ]
    missed = [why for ok, why in claims if not ok]
    if not missed:
        # Both numbers here are read back from checked names, and the whole sentence was
        # pinned as text above, so the victory line cannot quote a share or a vector this
        # run did not actually produce — nor put them in the wrong places.
        print(victory_line)
    else:
        print("\n❌ not yet — " + " | ".join(missed))
    for ok, why in claims:
        assert ok, why
