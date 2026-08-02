# day-06-encoder-vs-decoder — experiment
#
# Today's big idea in two lines of output:
#   With NO mask, every word keeps real weight on the future — an ENCODER, reading
#   both ways to understand. Add the causal mask to the SAME scores and every
#   future weight becomes 0.00 — a DECODER, looking back only to generate.
#
# Run it:  python3 sessions/m05a-text-transformer/day-06-encoder-vs-decoder/experiment.py

import numpy as np  # numpy gives us arrays, exp(), and -np.inf


def softmax(scores):
    # Row-wise softmax — the same step you built on Day 5, used unchanged for BOTH views.
    # Subtract each row's biggest score first so exp() never overflows.
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    # Divide each row by its own total, so every row of weights adds up to 1.
    return weights / weights.sum(axis=-1, keepdims=True)

def show(labels, matrix):
    # One labelled row per line. The caller rounds FIRST and keeps the rounded grid in a
    # name, so the numbers you read here are the exact same values the self-check tests.
    # The printed lines are also RETURNED, so the self-check can test the rendered text
    # itself — this helper prints five different grids, and a bug in here would quietly
    # change all five.
    lines = ["  %-5s %s" % (label, row) for label, row in zip(labels, matrix)]
    for line in lines:
        print(line)
    return lines

def compute_future_cells(n):
    # "Future" means the column sits to the RIGHT of the row: j > i. Written out by
    # hand on purpose — it is the PREDICTION, so it must not come from the mask.
    return sorted((i, j) for i in range(n) for j in range(n) if j > i)

def zero_cells(matrix):
    # Which cells hold exactly 0.0? An allowed cell can be tiny, but never 0.
    return sorted((int(i), int(j)) for i, j in np.argwhere(matrix == 0.0))


if __name__ == "__main__":
    # --- Part 1: one grid of made-up attention scores ---------------------
    words = ["the", "cat", "sat", "down"]
    n = len(words)
    # Nothing is random and no model is loaded, so these hand-written scores are the
    # same on every machine. Row i = how much word i rates each word; row 0 reuses the
    # lesson's demo row [2.0, 1.0, 5.0], where a FUTURE word wins. The grid is lopsided
    # on purpose (scores[0][2] = 5.0 but scores[2][0] = 1.5), so a transpose can't hide.
    # Rows and columns are counted from 0, as on every day of this module. And like Day 5's
    # grid, these scores are written down directly: they carry no 1/sqrt(d_k) division, so
    # read them as scores that have ALREADY been scaled (Day 8 does the dividing).
    scores = np.array([[2.0, 1.0, 5.0, 0.5],
                       [0.4, 2.5, 1.2, 4.0],
                       [1.5, 0.8, 3.0, 2.2],
                       [0.6, 1.8, 0.9, 2.4]])
    original_scores = scores.copy()   # kept aside to prove the mask never edits it
    # Every printed number below is bound to a name FIRST and then reused by the
    # self-check, so the line you read and the line that is tested are one value.
    shown_scores_shape = scores.shape
    shown_scores = np.round(scores, 4)
    print("scores shape:", shown_scores_shape, "(4 words rating 4 words)")
    show(words, shown_scores)
    lopsided = float(np.abs(scores - scores.T).max())
    shown_lopsided = round(lopsided, 4)
    print("biggest gap between scores[i][j] and scores[j][i]:", shown_lopsided,
          "-> not a mirror, so a transposed grid would give different weights")
    # Predict first, and COMPUTE it instead of typing it: row 1's future is j > 1.
    # Rows and columns are counted from 0 all through this module, so ROW = 1 is the
    # SECOND word. The whole line is rendered once and pinned as text below, because the
    # row number, the word and the column list have to agree: a shifted words[] subscript
    # would otherwise attach the causal-mask claim to the wrong word under a passing ✅.
    ROW = 1
    future = compute_future_cells(n)
    shown_future_row1 = [j for (i, j) in future if i == ROW]
    prediction_line = ("prediction for row %d ('%s'): columns %s are its future, "
                       "so only those should lose all weight in the decoder view"
                       % (ROW, words[ROW], shown_future_row1))
    print(prediction_line)

    # --- Part 2: the ENCODER view — same scores, no mask ------------------
    encoder_view = softmax(scores)
    shown_encoder_shape = encoder_view.shape
    shown_encoder = np.round(encoder_view, 4)
    print("\nENCODER view (no mask) -> shape", shown_encoder_shape)
    encoder_lines = show(words, shown_encoder)
    future_weights = np.array([encoder_view[i, j] for i, j in future])
    shown_future_weights = np.round(future_weights, 4)
    shown_future_min = round(float(future_weights.min()), 4)
    shown_future_max = round(float(future_weights.max()), 4)
    print("the 6 future (above-diagonal) weights:", shown_future_weights,
          "\n  smallest", shown_future_min, ", biggest", shown_future_max,
          "-> the future is ALIVE: it looks both ways")

    # --- Part 3: build the causal mask — the one switch -------------------
    # "The one switch" is precise about what it covers: inside ONE self-attention layer,
    # the mask is the whole difference between reading both ways and looking back only,
    # and this file proves it by feeding the SAME scores through the SAME softmax. Day 8's
    # full decoder is not just a masked encoder — it carries an extra sub-layer, the
    # cross-attention bridge to the encoder's output, which the encoder has no use for.
    # 0.0 on and below the diagonal (allowed), -inf above it (the future).
    causal_mask = np.triu(np.full((n, n), -np.inf), k=1)
    shown_mask_shape = causal_mask.shape
    shown_mask = np.round(causal_mask, 4)
    print("\ncausal mask -> shape", shown_mask_shape)
    show(words, shown_mask)
    masked_scores = scores + causal_mask   # the SAME scores, plus the blindfold
    shown_masked = np.round(masked_scores, 4)
    print("scores + mask (every future score is now -inf):")
    show(words, shown_masked)

    # --- Part 4: the DECODER view — same scores, same softmax, one mask ---
    decoder_view = softmax(masked_scores)
    shown_decoder_shape = decoder_view.shape
    shown_decoder = np.round(decoder_view, 4)
    print("\nDECODER view (causal mask) -> shape", shown_decoder_shape)
    decoder_lines = show(words, shown_decoder)
    dead = zero_cells(decoder_view)
    row_sums = decoder_view.sum(axis=1)
    shown_row_sums = np.round(row_sums, 4)
    print("cells that are exactly 0.00:", dead, "\n  cells we predicted would die:", future)
    print("row sums:", shown_row_sums, "-> each row still adds up to 1.0")

    # --- Part 5: side by side — what changed, and what did not ------------
    shown_encoder_2dp = np.round(encoder_view, 2)
    shown_decoder_2dp = np.round(decoder_view, 2)
    print("\nword   encoder (both ways)              decoder (look back only)")
    for i in range(n):
        print("  %-5s %-32s %s" % (words[i], shown_encoder_2dp[i], shown_decoder_2dp[i]))
    # Future cells go to 0. Past cells keep their RATIOS, only rescaled by
    # 1 / (the weight the row had left) so the row still sums to 1.
    rescales = np.array([1.0 / encoder_view[i, :i + 1].sum() for i in range(n)])
    shown_rescales = np.round(rescales, 4)
    print("past-cell rescale per row:", shown_rescales,
          "-> ratios kept, sizes grown; the last row needs no rescale at all")

    # --- Self-check: one boolean per claim --------------------------------
    # Every number below was READ OFF a real run, so the code cannot agree with itself.
    # The checks read the SHOWN values, so corrupting a printed number fails the run.
    expected_encoder = np.array([[0.0461, 0.0170, 0.9266, 0.0103], [0.0208, 0.1702, 0.0464, 0.7626],
                                 [0.1251, 0.0621, 0.5608, 0.2520], [0.0853, 0.2833, 0.1152, 0.5162]])
    expected_decoder = np.array([[1.0000, 0.0000, 0.0000, 0.0000], [0.1091, 0.8909, 0.0000, 0.0000],
                                 [0.1673, 0.0831, 0.7497, 0.0000], [0.0853, 0.2833, 0.1152, 0.5162]])
    # One shared input: four 4x4 grids, scores still unedited, and lopsided by 3.5.
    one_grid_ok = (all(s == (4, 4) for s in (shown_scores_shape, shown_encoder_shape,
                                            shown_mask_shape, shown_decoder_shape))
                   and np.array_equal(scores, original_scores)
                   and np.array_equal(shown_scores, original_scores) and shown_lopsided == 3.5)
    # The mask must block exactly the future cells — not the past, not the diagonal.
    blocked = sorted((int(i), int(j)) for i, j in np.argwhere(np.isinf(shown_mask)))
    mask_pattern_ok = (blocked == future and zero_cells(shown_mask) ==
                       sorted((i, j) for i in range(n) for j in range(n) if j <= i)
                       and shown_future_row1 == [2, 3]
                       # the printed "scores + mask" grid: -inf on the future, scores elsewhere
                       and sorted((int(i), int(j)) for i, j in np.argwhere(np.isinf(shown_masked))) == future
                       and all(shown_masked[i][j] == scores[i][j]
                               for i in range(n) for j in range(n) if j <= i))
    # The prediction line as it was RENDERED: row number, word and column list on one
    # line, so a shifted words[] subscript cannot pin the claim to the wrong word.
    prediction_line_ok = prediction_line == ("prediction for row 1 ('cat'): columns [2, 3] "
                                             "are its future, so only those should lose all "
                                             "weight in the decoder view")
    encoder_ok = (np.array_equal(shown_encoder, expected_encoder)
                  # the line as it was RENDERED, not just the numbers behind it
                  and encoder_lines[0] == "  the   [0.0461 0.017  0.9266 0.0103]")
    decoder_ok = (np.array_equal(shown_decoder, expected_decoder)
                  and decoder_lines[1] == "  cat   [0.1091 0.8909 0.     0.    ]")
    # The printed future weights are the encoder's own above-diagonal cells, all above 0.
    future_alive_ok = (np.array_equal(shown_future_weights,
                                      np.array([expected_encoder[i][j] for i, j in future]))
                       and shown_future_min == 0.0103 and shown_future_max == 0.9266
                       and shown_future_min > 0.0)
    # Dead means EXACTLY 0.0, not "small" — and dead in exactly the predicted cells.
    future_dead = dead == future and len(dead) == 6
    rows_sum_to_one = (np.array_equal(np.round(row_sums, 12), np.ones(n))
                       and np.array_equal(shown_row_sums, np.ones(n)))
    # Masking renormalizes the past cells: same ratios, rescaled to sum to 1.
    ratios_kept = (all(np.array_equal(np.round(encoder_view[i, :i + 1] * rescales[i], 12),
                                      np.round(decoder_view[i, :i + 1], 12)) for i in range(n))
                   and np.array_equal(shown_rescales, np.array([21.6765, 5.2356, 1.3368, 1.0])))
    # The side-by-side table at 2 decimals: future cells alive on the left, exactly 0 on the right.
    side_by_side_ok = (np.array_equal(shown_encoder_2dp, np.array([[0.05, 0.02, 0.93, 0.01],
                                                                  [0.02, 0.17, 0.05, 0.76],
                                                                  [0.13, 0.06, 0.56, 0.25],
                                                                  [0.09, 0.28, 0.12, 0.52]]))
                       and np.array_equal(shown_decoder_2dp, np.array([[1.00, 0.00, 0.00, 0.00],
                                                                       [0.11, 0.89, 0.00, 0.00],
                                                                       [0.17, 0.08, 0.75, 0.00],
                                                                       [0.09, 0.28, 0.12, 0.52]]))
                       and all(shown_decoder_2dp[i][j] == 0.0 and shown_encoder_2dp[i][j] > 0.0
                               for i, j in future))
    # The blindfold sits TOP-RIGHT: the last word has no future so its row is untouched,
    # while the first word loses everything except itself.
    direction_ok = (np.array_equal(decoder_view[n - 1], encoder_view[n - 1])
                    and not np.array_equal(decoder_view[0], encoder_view[0]))

    if (one_grid_ok and mask_pattern_ok and prediction_line_ok and encoder_ok and decoder_ok
            and future_alive_ok
            and future_dead and rows_sum_to_one and ratios_kept and side_by_side_ok and direction_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected four 4x4 grids from one unedited score grid with a gap of "
              "3.5, -inf on exactly the 6 future cells, encoder row 0 = [0.0461 0.017 0.9266 "
              "0.0103], decoder row 1 = [0.1091 0.8909 0. 0.] with exactly 0.0 on those 6 cells, "
              "every row summing to 1.0, rescales [21.6765 5.2356 1.3368 1.0], last row equal")

    # These asserts make the check hard: a wrong run stops the program.
    assert one_grid_ok, "four 4x4 grids, scores unedited, and a score gap of 3.5"
    assert mask_pattern_ok, "the mask must be -inf on exactly the 6 future cells, 0.0 elsewhere"
    assert prediction_line_ok, "the prediction must read row 1 ('cat'), columns [2, 3]"
    assert encoder_ok, "encoder view should start [0.0461 0.017 0.9266 0.0103]"
    assert decoder_ok, "decoder row 1 should be [0.1091 0.8909 0. 0.]"
    assert future_alive_ok, "the 6 printed future weights run 0.0103 to 0.9266, none of them 0"
    assert future_dead, "exactly the 6 predicted future cells should be exactly 0.0"
    assert rows_sum_to_one, "every masked row must still add up to 1.0"
    assert ratios_kept, "masking must keep the past ratios, rescaled by 21.6765/5.2356/1.3368/1"
    assert side_by_side_ok, "the 2-decimal table must show live future weights only on the left"
    assert direction_ok, "the last row must be untouched and the first row must change"
