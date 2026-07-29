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
    # One labelled row per line, rounded so the grid stays readable.
    for label, row in zip(labels, matrix):
        print("  %-5s %s" % (label, np.round(row, 4)))

def future_cells(n):
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
    scores = np.array([[2.0, 1.0, 5.0, 0.5],
                       [0.4, 2.5, 1.2, 4.0],
                       [1.5, 0.8, 3.0, 2.2],
                       [0.6, 1.8, 0.9, 2.4]])
    original_scores = scores.copy()   # kept aside to prove the mask never edits it
    print("scores shape:", scores.shape, "(4 words rating 4 words)")
    show(words, scores)
    lopsided = float(np.abs(scores - scores.T).max())
    print("biggest gap between scores[i][j] and scores[j][i]:", round(lopsided, 4),
          "-> not a mirror, so a transposed grid would give different weights")
    # Predict first, and COMPUTE it instead of typing it: row 1's future is j > 1.
    future = future_cells(n)
    print("prediction for row 1 ('%s'): columns %s are its future," %
          (words[1], [j for (i, j) in future if i == 1]),
          "so only those should lose all weight in the decoder view")

    # --- Part 2: the ENCODER view — same scores, no mask ------------------
    encoder_view = softmax(scores)
    print("\nENCODER view (no mask) -> shape", encoder_view.shape)
    show(words, encoder_view)
    future_weights = np.array([encoder_view[i, j] for i, j in future])
    print("the 6 future (above-diagonal) weights:", np.round(future_weights, 4),
          "\n  smallest", round(float(future_weights.min()), 4), ", biggest",
          round(float(future_weights.max()), 4), "-> the future is ALIVE: it looks both ways")

    # --- Part 3: build the causal mask — the one switch -------------------
    # 0.0 on and below the diagonal (allowed), -inf above it (the future).
    causal_mask = np.triu(np.full((n, n), -np.inf), k=1)
    print("\ncausal mask -> shape", causal_mask.shape)
    show(words, causal_mask)
    masked_scores = scores + causal_mask   # the SAME scores, plus the blindfold
    print("scores + mask (every future score is now -inf):")
    show(words, masked_scores)

    # --- Part 4: the DECODER view — same scores, same softmax, one mask ---
    decoder_view = softmax(masked_scores)
    print("\nDECODER view (causal mask) -> shape", decoder_view.shape)
    show(words, decoder_view)
    dead = zero_cells(decoder_view)
    row_sums = decoder_view.sum(axis=1)
    print("cells that are exactly 0.00:", dead, "\n  cells we predicted would die:", future)
    print("row sums:", np.round(row_sums, 4), "-> each row still adds up to 1.0")

    # --- Part 5: side by side — what changed, and what did not ------------
    print("\nword   encoder (both ways)              decoder (look back only)")
    for i in range(n):
        print("  %-5s %-32s %s" % (words[i], np.round(encoder_view[i], 2), np.round(decoder_view[i], 2)))
    # Future cells go to 0. Past cells keep their RATIOS, only rescaled by
    # 1 / (the weight the row had left) so the row still sums to 1.
    rescales = np.array([1.0 / encoder_view[i, :i + 1].sum() for i in range(n)])
    print("past-cell rescale per row:", np.round(rescales, 4),
          "-> ratios kept, sizes grown; the last row needs no rescale at all")

    # --- Self-check: one boolean per claim --------------------------------
    # Every number below was READ OFF a real run, so the code cannot agree with itself.
    expected_encoder = np.array([[0.0461, 0.0170, 0.9266, 0.0103], [0.0208, 0.1702, 0.0464, 0.7626],
                                 [0.1251, 0.0621, 0.5608, 0.2520], [0.0853, 0.2833, 0.1152, 0.5162]])
    expected_decoder = np.array([[1.0000, 0.0000, 0.0000, 0.0000], [0.1091, 0.8909, 0.0000, 0.0000],
                                 [0.1673, 0.0831, 0.7497, 0.0000], [0.0853, 0.2833, 0.1152, 0.5162]])
    # One shared input: four 4x4 grids, scores still unedited, and lopsided by 3.5.
    one_grid_ok = (all(g.shape == (4, 4) for g in (scores, encoder_view, causal_mask, decoder_view))
                   and np.array_equal(scores, original_scores) and round(lopsided, 4) == 3.5)
    # The mask must block exactly the future cells — not the past, not the diagonal.
    blocked = sorted((int(i), int(j)) for i, j in np.argwhere(np.isinf(causal_mask)))
    mask_pattern_ok = (blocked == future and zero_cells(causal_mask) ==
                       sorted((i, j) for i in range(n) for j in range(n) if j <= i))
    encoder_ok = np.array_equal(np.round(encoder_view, 4), expected_encoder)
    decoder_ok = np.array_equal(np.round(decoder_view, 4), expected_decoder)
    # Dead means EXACTLY 0.0, not "small" — and dead in exactly the predicted cells.
    future_dead = dead == future and len(dead) == 6
    rows_sum_to_one = np.array_equal(np.round(row_sums, 12), np.ones(n))
    # Masking renormalizes the past cells: same ratios, rescaled to sum to 1.
    ratios_kept = (all(np.array_equal(np.round(encoder_view[i, :i + 1] * rescales[i], 12),
                                      np.round(decoder_view[i, :i + 1], 12)) for i in range(n))
                   and np.array_equal(np.round(rescales, 4), np.array([21.6765, 5.2356, 1.3368, 1.0])))
    # The blindfold sits TOP-RIGHT: the last word has no future so its row is untouched,
    # while the first word loses everything except itself.
    direction_ok = (np.array_equal(decoder_view[n - 1], encoder_view[n - 1])
                    and not np.array_equal(decoder_view[0], encoder_view[0]))

    if (one_grid_ok and mask_pattern_ok and encoder_ok and decoder_ok
            and future_dead and rows_sum_to_one and ratios_kept and direction_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected four 4x4 grids from one unedited score grid with a gap of "
              "3.5, -inf on exactly the 6 future cells, encoder row 0 = [0.0461 0.017 0.9266 "
              "0.0103], decoder row 1 = [0.1091 0.8909 0. 0.] with exactly 0.0 on those 6 cells, "
              "every row summing to 1.0, rescales [21.6765 5.2356 1.3368 1.0], last row equal")

    # These asserts make the check hard: a wrong run stops the program.
    assert one_grid_ok, "four 4x4 grids, scores unedited, and a score gap of 3.5"
    assert mask_pattern_ok, "the mask must be -inf on exactly the 6 future cells, 0.0 elsewhere"
    assert encoder_ok, "encoder view should start [0.0461 0.017 0.9266 0.0103]"
    assert decoder_ok, "decoder row 1 should be [0.1091 0.8909 0. 0.]"
    assert future_dead, "exactly the 6 predicted future cells should be exactly 0.0"
    assert rows_sum_to_one, "every masked row must still add up to 1.0"
    assert ratios_kept, "masking must keep the past ratios, rescaled by 21.6765/5.2356/1.3368/1"
    assert direction_ok, "the last row must be untouched and the first row must change"
