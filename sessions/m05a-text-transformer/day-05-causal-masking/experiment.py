# day-05-causal-masking — experiment
#
# Today's big idea in two lines of output:
#   Add -inf to every FUTURE cell of the attention-score grid before softmax, and
#   those words win exactly 0 attention — the past words still split all of it.
#
# The script shows the lesson's 4x4 score grid, builds the lower-triangular causal_mask,
# adds it and watches the future collapse to zero, runs the same softmax with NO mask as a
# control, and replays the lesson's one-row demo where the future word would have won 94%.
# Run it:  python3 sessions/m05a-text-transformer/day-05-causal-masking/experiment.py

import numpy as np  # numpy gives us arrays, exp(), and the triangle helper triu


def softmax(scores):
    # Row-wise softmax: every ROW of scores becomes weights that add up to 1.
    # Subtract each row's biggest score first so exp() never blows up.
    # axis=-1 means "the last axis" = along each row. Days 6, 7 and 8 use the same spelling.
    shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)                             # e^(-inf) is exactly 0.0
    return weights / weights.sum(axis=-1, keepdims=True)  # divide by that row's total


def show(labels, matrix):
    # Print one labelled row per line, 4 decimals, so a grid reads like the lesson's picture.
    # This is the ONLY renderer for every grid this day teaches from, so it also hands the
    # rows BACK — read straight out of the text it just printed. The self-check pins those,
    # which means a transform hidden in here cannot reprint every table wrong unnoticed.
    rendered = []
    for label, row in zip(labels, matrix):
        cells = ["    -inf" if np.isneginf(v) else "%8.4f" % v for v in row]
        print("   %-5s%s" % (label, "".join(cells)))
        rendered.append([float(c) for c in cells])   # the numbers exactly as printed
    return rendered


if __name__ == "__main__":
    # --- Part 1: the attention-score grid (row looks at column) -----------
    # The lesson's own 4-word sentence and its 4x4 grid of made-up scores. Fixed numbers
    # instead of np.random.randn, so this is the grid you read in the lesson and two runs
    # print byte-identical output.
    # Two conventions, stated once. (1) Rows and columns are counted FROM 0 all through this
    # module, so row 0 is the first word. (2) These scores are written down directly, so
    # they carry no 1/sqrt(d_k) division: real attention divides Q @ K.T by the square root
    # of the head width before the softmax (module m03 built that, and Day 8 applies it).
    # Scaling every score by the same constant would move the weights below, so the grid
    # here is best read as scores that have ALREADY been scaled.
    words = ["the", "cat", "sat", "mat"]
    N = len(words)
    scores = np.array([[2.1, 0.3, 1.0, 0.4],    # row "the" scores every word
                       [0.5, 3.2, 0.9, 0.2],    # row "cat" scores every word
                       [1.1, 2.4, 2.0, 0.6],    # row "sat" scores every word
                       [0.7, 1.3, 2.8, 1.9]])   # row "mat" scores every word
    # Every printed number and rendered line below is bound to a name FIRST and checked by
    # that same name, so a wrong value cannot appear on screen under a passing ✅.
    sentence = " ".join(words)
    shown_scores_shape = scores.shape
    print("sentence:", sentence)
    print("scores shape:", shown_scores_shape, "(row = word looking, column = word looked at)")
    shown_scores = show(words, scores)
    # A lopsided grid on purpose: read the wrong way round (a transpose) it gives other numbers.
    shown_cat_mat, shown_mat_cat = scores[1, 3], scores[3, 1]
    print("scores[cat][mat] =", shown_cat_mat, "vs scores[mat][cat] =", shown_mat_cat,
          "-> not a mirror.  The danger: row 'cat' scores 'mat', and 'mat' comes LATER")

    # --- Part 2: build the causal mask (the blindfold) --------------------
    # np.triu(..., k=1) picks the cells strictly ABOVE the diagonal — exactly the
    # cells where the column word comes later than the row word: the future.
    causal_mask = np.triu(np.full((N, N), -np.inf), k=1)
    shown_mask_shape = causal_mask.shape
    print("\ncausal_mask shape:", shown_mask_shape, "(0 = allowed, -inf = blocked future)")
    shown_mask = show(words, causal_mask)
    blocked_per_row = np.isneginf(causal_mask).sum(axis=1)   # how many cells each row loses
    shown_allowed_per_row = N - blocked_per_row
    print("blocked (future) cells per row:", blocked_per_row, " allowed per row:",
          shown_allowed_per_row, "-> row 0 may see 1 word, row 3 may see all 4")

    # --- Part 3: add the mask, then softmax each row ----------------------
    masked_scores = scores + causal_mask       # adding -inf pushes a future score to -inf
    shown_masked_scores_shape = masked_scores.shape
    print("\nmasked_scores shape:", shown_masked_scores_shape, "(future scores pushed to -inf)")
    shown_masked_scores = show(words, masked_scores)
    masked_weights = softmax(masked_scores)
    print("attention weights AFTER the mask:")
    shown_masked_weights = show(words, masked_weights)
    shown_row_sums = np.round(masked_weights.sum(axis=1), 6)
    print("each row's weights add up to:", shown_row_sums)
    # Predict first, then look. For row "sat" the future words are the ones that come after
    # it in the sentence, so those columns should end up at zero. The prediction below is
    # COMPUTED from the sentence, not typed in as an answer.
    row = words.index("sat")
    future_words = words[row + 1:]                                    # ['mat']
    predicted_zero_cols = [words.index(w) for w in future_words]      # -> [3]
    observed_zero_cols = [int(c) for c in np.flatnonzero(masked_weights[row] == 0.0)]
    print("row 'sat': future words", future_words, "-> predicted zero columns",
          predicted_zero_cols, "| observed", observed_zero_cols)
    # The six future cells of a 4x4 grid: read off the -inf cells of the mask you just
    # PRINTED, rather than typed in from a triangle no reader can see. The self-check
    # still holds them against the lesson's own six, so the ground truth is unchanged.
    future_cells = [(i, j) for i in range(N) for j in range(N)
                    if np.isneginf(causal_mask[i, j])]
    masked_future_total = float(sum(masked_weights[i, j] for i, j in future_cells))

    # --- Part 4: the control — the same softmax with NO mask --------------
    unmasked_weights = softmax(scores)
    print("\nattention weights with NO mask (the cheating version):")
    shown_unmasked_weights = show(words, unmasked_weights)
    unmasked_leak_per_row = np.array(
        [sum(unmasked_weights[i, j] for i, j in future_cells if i == r) for r in range(N)])
    shown_leak_per_row = np.round(unmasked_leak_per_row, 4)
    print("weight each row spends on its own FUTURE, no mask:", shown_leak_per_row)
    shown_masked_total = masked_future_total
    shown_unmasked_total = round(float(unmasked_leak_per_row.sum()), 4)
    print("total future weight: masked", shown_masked_total, "vs unmasked",
          shown_unmasked_total, "-> the leakage the mask removes")
    # The last word has no future at all, so the blindfold changes nothing for its row.
    last_row_unchanged = np.array_equal(masked_weights[3], unmasked_weights[3])
    print("row 'mat' identical with and without the mask?", last_row_unchanged)

    # --- Part 5: the lesson's one-row demo -------------------------------
    # Concept 4's row [2.0, 1.0, 5.0]: the third word is the FUTURE one, and its 5.0 is
    # the biggest score, so unmasked it would dominate the whole row.
    demo = np.array([[2.0, 1.0, 5.0]])
    demo_masked = softmax(demo + np.array([[0.0, 0.0, -np.inf]]))    # block column 2 only
    demo_open = softmax(demo)                                        # no blindfold
    shown_demo_scores = demo[0]
    shown_demo_open = np.round(demo_open[0], 4)
    shown_demo_masked = np.round(demo_masked[0], 4)
    shown_open_winner = int(demo_open[0].argmax())
    shown_masked_winner = int(demo_masked[0].argmax())
    print("\ndemo row scores:", shown_demo_scores, "(column 2 is the future word)")
    print("no mask:  ", shown_demo_open, "-> winner is column",
          shown_open_winner, "(the future word takes almost everything)")
    print("with mask:", shown_demo_masked, "-> winner is column",
          shown_masked_winner, "(a past word; the future got 0)")

    # --- Self-check: one boolean per claim -------------------------------
    # Every expected number below was read off a real run and typed in as a literal, so
    # the code above cannot quietly agree with itself. Each claim reads the SAME name the
    # print statement read — including the grids, which are checked as they were RENDERED.
    NEG = -np.inf
    expected_masked = [[1.0, 0.0, 0.0, 0.0], [0.063, 0.937, 0.0, 0.0],
                       [0.1403, 0.5147, 0.345, 0.0], [0.0699, 0.1273, 0.5707, 0.232]]
    expected_leak = [0.4051, 0.1233, 0.0784, 0.0]     # future weight per row, no mask
    header_ok = sentence == "the cat sat mat" and shown_scores_shape == (N, N) == (4, 4)
    score_grid_ok = shown_scores == [[2.1, 0.3, 1.0, 0.4], [0.5, 3.2, 0.9, 0.2],
                                     [1.1, 2.4, 2.0, 0.6], [0.7, 1.3, 2.8, 1.9]]
    not_a_mirror = (shown_cat_mat, shown_mat_cat) == (0.2, 1.3) and shown_cat_mat != shown_mat_cat
    # The printed triangle itself: 0 on and below the diagonal, -inf strictly above it.
    mask_grid_ok = shown_mask == [[0.0, NEG, NEG, NEG], [0.0, 0.0, NEG, NEG],
                                  [0.0, 0.0, 0.0, NEG], [0.0, 0.0, 0.0, 0.0]]
    mask_shape_ok = (shown_mask_shape == shown_masked_scores_shape == (4, 4)
                     and [int(b) for b in blocked_per_row] == [3, 2, 1, 0]
                     and [int(a) for a in shown_allowed_per_row] == [1, 2, 3, 4]
                     and np.array_equal(causal_mask[np.isfinite(causal_mask)], np.zeros(10)))
    future_cells_ok = future_cells == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    masked_scores_ok = shown_masked_scores == [[2.1, NEG, NEG, NEG], [0.5, 3.2, NEG, NEG],
                                               [1.1, 2.4, 2.0, NEG], [0.7, 1.3, 2.8, 1.9]]
    future_is_zero = all(masked_weights[i, j] == 0.0 for i, j in future_cells)
    rows_sum_to_one = shown_row_sums.tolist() == [1.0] * 4
    masked_grid_matches = shown_masked_weights == expected_masked
    prediction_held = predicted_zero_cols == observed_zero_cols == [3]
    unmasked_grid_ok = shown_unmasked_weights == [[0.5949, 0.0983, 0.198, 0.1087],
                                                  [0.0552, 0.8215, 0.0824, 0.0409],
                                                  [0.1293, 0.4743, 0.318, 0.0784],
                                                  [0.0699, 0.1273, 0.5707, 0.232]]
    leak_matches = shown_leak_per_row.tolist() == expected_leak
    totals_match = shown_masked_total == 0.0 and shown_unmasked_total == 0.6067
    demo_row_ok = shown_demo_scores.tolist() == [2.0, 1.0, 5.0]
    demo_matches = (shown_demo_masked.tolist() == [0.7311, 0.2689, 0.0]
                    and shown_demo_open.tolist() == [0.0466, 0.0171, 0.9362])
    demo_winner_moves = shown_open_winner == 2 and shown_masked_winner == 0
    if (header_ok and score_grid_ok and not_a_mirror and mask_grid_ok and mask_shape_ok
            and future_cells_ok and masked_scores_ok and future_is_zero and rows_sum_to_one
            and masked_grid_matches and prediction_held and unmasked_grid_ok and leak_matches
            and totals_match and last_row_unchanged and demo_row_ok and demo_matches
            and demo_winner_moves):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected 3,2,1,0 blocked cells per row; masked weights [[1,0,0,0],"
              "[.063,.937,0,0],[.1403,.5147,.345,0],[.0699,.1273,.5707,.232]] with every future "
              "cell exactly 0 and every row summing to 1; column 3 the only zero in row 'sat'; "
              "unmasked leak per row [.4051,.1233,.0784,0] adding up to 0.6067 against a masked "
              "total of 0.0; row 'mat' unchanged; demo row "
              "[.0466,.0171,.9362] (winner 2) -> [.7311,.2689,0] (winner 0)")

    assert header_ok, "the header should read 'the cat sat mat' with a (4, 4) score grid"
    assert score_grid_ok, "the printed score grid should be the lesson's [[2.1,.3,1,.4],[.5,3.2,"\
                          ".9,.2],[1.1,2.4,2,.6],[.7,1.3,2.8,1.9]]"
    assert not_a_mirror, "scores[cat][mat] is 0.2 and scores[mat][cat] is 1.3 — not a mirror"
    assert mask_grid_ok, "the printed mask must be 0 on and below the diagonal, -inf above it"
    assert mask_shape_ok, "the mask should block 3,2,1,0 future cells per row and hold 10 zeros"
    assert future_cells_ok, "the mask's -inf cells must be exactly the 6 above-diagonal cells"
    assert masked_scores_ok, "the printed masked scores should keep the past and show -inf ahead"
    assert future_is_zero, "every above-diagonal (future) weight must be exactly 0.0"
    assert rows_sum_to_one, "each row of masked weights must still add up to 1.0"
    assert masked_grid_matches, "masked weights should be [[1,0,0,0],[.063,.937,0,0]," \
                                "[.1403,.5147,.345,0],[.0699,.1273,.5707,.232]]"
    assert prediction_held, "row 'sat' should have exactly one zero column, column 3 ('mat')"
    assert unmasked_grid_ok, "the no-mask grid should be [[.5949,.0983,.198,.1087],[.0552,.8215," \
                             ".0824,.0409],[.1293,.4743,.318,.0784],[.0699,.1273,.5707,.232]]"
    assert leak_matches, "with no mask the rows should leak [.4051,.1233,.0784,0] onto the future"
    assert totals_match, "future weight should total 0.0 masked and 0.6067 unmasked"
    assert last_row_unchanged, "the last word has no future, so the mask must not change its row"
    assert demo_row_ok, "the demo row's scores should print [2. 1. 5.]"
    assert demo_matches, "the demo row should give [.0466,.0171,.9362] open and [.7311,.2689,0] masked"
    assert demo_winner_moves, "unmasked the demo winner is column 2 (future); masked it is column 0"
