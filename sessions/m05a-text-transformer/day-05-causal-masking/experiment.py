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
    # axis=-1 means "the last axis" = along each row. Days 6 and 8 use the same spelling.
    shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)                             # e^(-inf) is exactly 0.0
    return weights / weights.sum(axis=-1, keepdims=True)  # divide by that row's total


def show(labels, matrix):
    # Print one labelled row per line, 4 decimals, so a grid reads like the lesson's picture.
    for label, row in zip(labels, matrix):
        cells = ["    -inf" if np.isneginf(v) else "%8.4f" % v for v in row]
        print("   %-5s%s" % (label, "".join(cells)))


if __name__ == "__main__":
    # --- Part 1: the attention-score grid (row looks at column) -----------
    # The lesson's own 4-word sentence and its 4x4 grid of made-up scores. Fixed numbers
    # instead of np.random.randn, so this is the grid you read in the lesson and two runs
    # print byte-identical output.
    words = ["the", "cat", "sat", "mat"]
    N = len(words)
    scores = np.array([[2.1, 0.3, 1.0, 0.4],    # row "the" scores every word
                       [0.5, 3.2, 0.9, 0.2],    # row "cat" scores every word
                       [1.1, 2.4, 2.0, 0.6],    # row "sat" scores every word
                       [0.7, 1.3, 2.8, 1.9]])   # row "mat" scores every word
    print("sentence:", " ".join(words))
    print("scores shape:", scores.shape, "(row = word looking, column = word looked at)")
    show(words, scores)
    # A lopsided grid on purpose: read the wrong way round (a transpose) it gives other numbers.
    print("scores[cat][mat] =", scores[1, 3], "vs scores[mat][cat] =", scores[3, 1],
          "-> not a mirror.  The danger: row 'cat' scores 'mat', and 'mat' comes LATER")

    # --- Part 2: build the causal mask (the blindfold) --------------------
    # np.triu(..., k=1) picks the cells strictly ABOVE the diagonal — exactly the
    # cells where the column word comes later than the row word: the future.
    causal_mask = np.triu(np.full((N, N), -np.inf), k=1)
    print("\ncausal_mask shape:", causal_mask.shape, "(0 = allowed, -inf = blocked future)")
    show(words, causal_mask)
    blocked_per_row = np.isneginf(causal_mask).sum(axis=1)   # how many cells each row loses
    print("blocked (future) cells per row:", blocked_per_row, " allowed per row:",
          N - blocked_per_row, "-> word 1 may see 1 word, word 4 may see all 4")

    # --- Part 3: add the mask, then softmax each row ----------------------
    masked_scores = scores + causal_mask       # adding -inf pushes a future score to -inf
    print("\nmasked_scores shape:", masked_scores.shape, "(future scores pushed to -inf)")
    show(words, masked_scores)
    masked_weights = softmax(masked_scores)
    print("attention weights AFTER the mask:")
    show(words, masked_weights)
    print("each row's weights add up to:", np.round(masked_weights.sum(axis=1), 6))
    # Predict first, then look. For row "sat" the future words are the ones that come after
    # it in the sentence, so those columns should end up at zero. The prediction below is
    # COMPUTED from the sentence, not typed in as an answer.
    row = words.index("sat")
    future_words = words[row + 1:]                                    # ['mat']
    predicted_zero_cols = [words.index(w) for w in future_words]      # -> [3]
    observed_zero_cols = [int(c) for c in np.flatnonzero(masked_weights[row] == 0.0)]
    print("row 'sat': future words", future_words, "-> predicted zero columns",
          predicted_zero_cols, "| observed", observed_zero_cols)
    # The six future cells of a 4x4 grid, read straight off the lesson's triangle.
    future_cells = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    masked_future_total = float(sum(masked_weights[i, j] for i, j in future_cells))

    # --- Part 4: the control — the same softmax with NO mask --------------
    unmasked_weights = softmax(scores)
    print("\nattention weights with NO mask (the cheating version):")
    show(words, unmasked_weights)
    unmasked_leak_per_row = np.array(
        [sum(unmasked_weights[i, j] for i, j in future_cells if i == r) for r in range(N)])
    print("weight each row spends on its own FUTURE, no mask:", np.round(unmasked_leak_per_row, 4))
    print("total future weight: masked", masked_future_total, "vs unmasked",
          round(float(unmasked_leak_per_row.sum()), 4), "-> the leakage the mask removes")
    # The last word has no future at all, so the blindfold changes nothing for its row.
    last_row_unchanged = np.array_equal(masked_weights[3], unmasked_weights[3])
    print("row 'mat' identical with and without the mask?", last_row_unchanged)

    # --- Part 5: the lesson's one-row demo -------------------------------
    # Concept 4's row [2.0, 1.0, 5.0]: the third word is the FUTURE one, and its 5.0 is
    # the biggest score, so unmasked it would dominate the whole row.
    demo = np.array([[2.0, 1.0, 5.0]])
    demo_masked = softmax(demo + np.array([[0.0, 0.0, -np.inf]]))    # block column 2 only
    demo_open = softmax(demo)                                        # no blindfold
    print("\ndemo row scores:", demo[0], "(column 2 is the future word)")
    print("no mask:  ", np.round(demo_open[0], 4), "-> winner is column",
          int(demo_open[0].argmax()), "(the future word takes almost everything)")
    print("with mask:", np.round(demo_masked[0], 4), "-> winner is column",
          int(demo_masked[0].argmax()), "(a past word; the future got 0)")

    # --- Self-check: one boolean per claim -------------------------------
    # Every expected number below was read off a real run and typed in as a literal, so
    # the code above cannot quietly agree with itself.
    expected_masked = [[1.0, 0.0, 0.0, 0.0], [0.063, 0.937, 0.0, 0.0],
                       [0.1403, 0.5147, 0.345, 0.0], [0.0699, 0.1273, 0.5707, 0.232]]
    expected_leak = [0.4051, 0.1233, 0.0784, 0.0]     # future weight per row, no mask
    mask_shape_ok = ([int(b) for b in blocked_per_row] == [3, 2, 1, 0]
                     and np.array_equal(causal_mask[np.isfinite(causal_mask)], np.zeros(10)))
    future_is_zero = all(masked_weights[i, j] == 0.0 for i, j in future_cells)
    rows_sum_to_one = [round(float(s), 6) for s in masked_weights.sum(axis=1)] == [1.0] * 4
    masked_grid_matches = [[round(float(v), 4) for v in r] for r in masked_weights] == expected_masked
    prediction_held = predicted_zero_cols == observed_zero_cols == [3]
    leak_matches = [round(float(v), 4) for v in unmasked_leak_per_row] == expected_leak
    demo_matches = ([round(float(v), 4) for v in demo_masked[0]] == [0.7311, 0.2689, 0.0]
                    and [round(float(v), 4) for v in demo_open[0]] == [0.0466, 0.0171, 0.9362])
    demo_winner_moves = int(demo_open[0].argmax()) == 2 and int(demo_masked[0].argmax()) == 0
    if (mask_shape_ok and future_is_zero and rows_sum_to_one and masked_grid_matches
            and prediction_held and leak_matches and last_row_unchanged and demo_matches
            and demo_winner_moves):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected 3,2,1,0 blocked cells per row; masked weights [[1,0,0,0],"
              "[.063,.937,0,0],[.1403,.5147,.345,0],[.0699,.1273,.5707,.232]] with every future "
              "cell exactly 0 and every row summing to 1; column 3 the only zero in row 'sat'; "
              "unmasked leak per row [.4051,.1233,.0784,0]; row 'mat' unchanged; demo row "
              "[.0466,.0171,.9362] (winner 2) -> [.7311,.2689,0] (winner 0)")

    assert mask_shape_ok, "the mask should block 3,2,1,0 future cells per row and hold 10 zeros"
    assert future_is_zero, "every above-diagonal (future) weight must be exactly 0.0"
    assert rows_sum_to_one, "each row of masked weights must still add up to 1.0"
    assert masked_grid_matches, "masked weights should be [[1,0,0,0],[.063,.937,0,0]," \
                                "[.1403,.5147,.345,0],[.0699,.1273,.5707,.232]]"
    assert prediction_held, "row 'sat' should have exactly one zero column, column 3 ('mat')"
    assert leak_matches, "with no mask the rows should leak [.4051,.1233,.0784,0] onto the future"
    assert last_row_unchanged, "the last word has no future, so the mask must not change its row"
    assert demo_matches, "the demo row should give [.0466,.0171,.9362] open and [.7311,.2689,0] masked"
    assert demo_winner_moves, "unmasked the demo winner is column 2 (future); masked it is column 0"
