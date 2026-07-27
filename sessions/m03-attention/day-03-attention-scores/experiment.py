# day-03-attention-scores — experiment
#
# GOAL (from the lesson's PRODUCE step): build the attention pipeline by hand on
# the same tiny 3-word sentence the lesson carries all day — "the river bank" —
# score -> scale -> softmax -> mask -> blend the Values. Watch a row of raw match
# numbers turn into a clean "budget" of shares that add up to exactly 1, then get
# spent on the Values. This is the exact move at the heart of every transformer:
# softmax(Q @ K.T / sqrt(d_k)) @ V
#
# Every number asserted below is a number printed on the lesson page.
#
# Run it:  python3 sessions/m03-attention/day-03-attention-scores/experiment.py

import numpy as np  # numpy does the slot-by-slot multiply-and-add (the dot product) for us


# ---------------------------------------------------------------------------
# softmax: turn one ROW of scores into positive shares that add up to exactly 1.
# The biggest score gets the biggest slice — like splitting one pizza by hunger.
# ---------------------------------------------------------------------------
def softmax_row(scores):
    # Subtract the row's biggest score first. This does NOT change the answer
    # (softmax only cares about the GAPS between scores — the lesson's fourth
    # promise), but it keeps exp() from overflowing on a big score. Libraries
    # call this the "stable softmax".
    shifted = scores - np.max(scores)
    # The stretch: raise e to the power of each score -> everything turns positive.
    grown = np.exp(shifted)
    # The pot + the share: add them all up, then divide each one by that total.
    # Now the row sums to 1 — one whole "budget" split into slices.
    return grown / np.sum(grown)


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # STEP 0 — our tiny world: the 3 words of "the river bank". Each word has a
    # Query list (what am I looking for?), a Key list (what do I offer?) and a
    # Value list (what do I hand over when someone listens?). Small whole numbers
    # on purpose, so every result can be checked by eye.
    # d_k = how many slots each Query/Key list has. Here d_k = 4.
    # These are the EXACT lists printed in the lesson's PRODUCE step.
    # -----------------------------------------------------------------------
    words = ["the", "river", "bank"]

    # One Query row per word (the asking side).
    Q = np.array([
        [0.0, 0.0, 1.0, 1.0],   # "the"   asks only for the two shared slots (a boring word)
        [1.0, 0.0, 1.0, 1.0],   # "river" asks a little for slot-1 things
        [2.0, 1.0, 1.0, 1.0],   # "bank"  asks hard for slot-1, a bit for slot-2
    ])

    # One Key row per word (the offering side).
    K = np.array([
        [0.0, 0.0, 1.0, 1.0],   # "the"   offers only the two shared slots
        [2.0, 0.0, 1.0, 1.0],   # "river" offers lots of slot-1   (bank's dream)
        [0.0, 2.0, 1.0, 1.0],   # "bank"  offers lots of slot-2
    ])

    d_k = Q.shape[1]  # number of slots in each list = 4
    print("d_k =", d_k, " ->  sqrt(d_k) =", np.sqrt(d_k), "\n")

    # -----------------------------------------------------------------------
    # STEP 1 — the score matrix: score EVERY Query against EVERY Key at once.
    # Q @ K.T lays every Query as a row and every Key as a column, then does the
    # dot product (multiply slot-by-slot, add) for every pair in one move.
    # PREDICT: "bank" asks hard for slot 1, and "river" offers lots of slot 1 —
    # so bank should score highest against river.
    # -----------------------------------------------------------------------
    scores = Q @ K.T
    print("STEP 1 — score matrix (row = asking word, column = offered word):")
    print(scores)
    # The whole grid should match the lesson's picture, cell for cell.
    lesson_grid = np.array([
        [2.0, 2.0, 2.0],   # "the"   shrugs: every word gets a 2 (a flat row)
        [2.0, 4.0, 2.0],   # "river" likes itself most (4)
        [2.0, 6.0, 4.0],   # "bank"  leans hard on "river" (6)
    ])
    # Hand-check the star cell: bank's Query [2,1,1,1] . river's Key [2,0,1,1]
    #                           = 2*2 + 1*0 + 1*1 + 1*1 = 6.
    assert np.array_equal(scores, lesson_grid), \
        "not yet — expected the lesson's grid [[2,2,2],[2,4,2],[2,6,4]]"
    print("  bank's row (the / river / bank):", scores[2])
    print("  bank listens most to 'river' (score 6) — the whole grid matches the lesson.\n")

    # -----------------------------------------------------------------------
    # STEP 2 — scale: divide the whole grid by sqrt(d_k) to turn the volume down.
    # NOTICE: the numbers get smaller, but WHO-BEATS-WHOM never changes. Scaling
    # changes the loudness, never the winner. Here sqrt(4) = 2, so bank's row
    # [2, 6, 4] becomes [1, 3, 2].
    # -----------------------------------------------------------------------
    scaled = scores / np.sqrt(d_k)
    print("STEP 2 — scaled by sqrt(d_k) = 2:")
    print(scaled)
    bank_row = scaled[2]
    assert np.array_equal(bank_row, [1.0, 3.0, 2.0]), \
        "not yet — expected bank's scaled row [1, 3, 2]"
    # The order (argsort) of bank's row must be identical before and after scaling.
    assert np.array_equal(np.argsort(scores[2]), np.argsort(scaled[2])), \
        "not yet — expected scaling to keep the same order (who beats whom)"
    print("  bank's row is now [1, 3, 2] — order unchanged, river still wins.\n")

    # -----------------------------------------------------------------------
    # STEP 3 — softmax each row: raw scores become a budget of shares summing to 1.
    # OBSERVE the row sums. Bank's row [1, 3, 2] should split into the lesson's
    # shares 0.09 / 0.67 / 0.24 — river takes two-thirds from a gap of only 1.
    # -----------------------------------------------------------------------
    weights = np.array([softmax_row(row) for row in scaled])
    print("STEP 3 — softmax over each row (the attention weights):")
    print(np.round(weights, 4))
    print("         row sums =", np.round(weights.sum(axis=1), 6), "(one full budget each)")
    bank_shares = weights[2]
    assert np.allclose(np.round(bank_shares, 2), [0.09, 0.67, 0.24]), \
        "not yet — expected bank's shares [0.09, 0.67, 0.24]"
    assert np.allclose(weights.sum(axis=1), 1.0), \
        "not yet — every row of shares must add up to exactly 1"
    print("  every share is positive and every row adds to exactly 1.\n")

    # -----------------------------------------------------------------------
    # STEP 4 — masking: some words are FORBIDDEN (a future word, or blank padding).
    # The lesson's row is [the 1, river 3, FUTURE 2, pad 0] — the same scaled
    # numbers, plus one future word and one padding slot.
    # WATCH: with no mask, FUTURE steals a real 0.24 slice of the budget.
    # Set forbidden scores to a huge negative number BEFORE softmax and their
    # shares collapse to ~0, while the freed budget flows to the allowed words.
    # -----------------------------------------------------------------------
    mask_labels = ["the", "river", "FUTURE", "pad"]
    mask_scores = np.array([1.0, 3.0, 2.0, 0.0])
    unmasked = softmax_row(mask_scores)
    print("STEP 4 — no mask ", mask_labels, ":", np.round(unmasked, 2), "(FUTURE steals attention!)")
    assert np.allclose(np.round(unmasked, 2), [0.09, 0.64, 0.24, 0.03]), \
        "not yet — expected unmasked shares [0.09, 0.64, 0.24, 0.03]"

    masked_scores = mask_scores.copy()
    masked_scores[2] = -1e9   # mute FUTURE (score = -infinity, in practice a big negative)
    masked_scores[3] = -1e9   # mute pad
    masked = softmax_row(masked_scores)
    print("STEP 4 — masked  ", mask_labels, ":", np.round(masked, 2), "(forbidden shares -> 0)")
    assert np.allclose(np.round(masked, 2), [0.12, 0.88, 0.00, 0.00]), \
        "not yet — expected masked shares [0.12, 0.88, 0.00, 0.00]"
    # The timing test from the lesson: mask BEFORE softmax, and the row still sums to 1.
    assert np.isclose(masked.sum(), 1.0), \
        "not yet — a correctly masked row must STILL add up to 1"
    print("  FUTURE and pad went to ~0; 'the' rose 0.09 -> 0.12 and river 0.64 -> 0.88.")
    print("  the row still sums to", round(float(masked.sum()), 6), "— masked before softmax, not after.\n")

    # -----------------------------------------------------------------------
    # STEP 5 — spend the budget: blend the Values by bank's shares.
    # Each word carries a 2-slot Value — what it hands over when listened to.
    # PREDICT: river holds two-thirds of the budget and its Value is [10, 2],
    # so slot 1 of the answer should land near 10, not near 0.
    # -----------------------------------------------------------------------
    V = np.array([
        [0.0, 6.0],    # "the"   Value
        [10.0, 2.0],   # "river" Value
        [4.0, 4.0],    # "bank"  Value
    ])
    blended = bank_shares @ V   # shares x Values, added up slot by slot
    print("STEP 5 — blend: bank's shares", np.round(bank_shares, 4), "@ Values ->", np.round(blended, 2))
    assert np.allclose(np.round(blended, 2), [7.63, 2.85]), \
        "not yet — expected the blended output [7.63, 2.85]"
    # Rounding note from the lesson: rounding the shares to 0.09/0.67/0.24 FIRST
    # and then blending gives [7.66, 2.84] — the same answer, rounded earlier.
    rounded_recipe = np.array([0.09, 0.67, 0.24])
    assert np.allclose(np.round(rounded_recipe @ V, 2), [7.66, 2.84]), \
        "not yet — expected the rounded-share blend [7.66, 2.84]"
    # The blend can only ever land BETWEEN the Values (a convex mix), never outside.
    assert np.all(blended >= V.min(axis=0)) and np.all(blended <= V.max(axis=0)), \
        "a blend must land inside the range of the Values"
    print("  full-precision shares -> [7.63, 2.85];  shares rounded first -> [7.66, 2.84].")
    print("  the answer sits inside the Values, pulled toward river — attention blends, never invents.\n")

    # -----------------------------------------------------------------------
    # BONUS — the whole thing in one line, exactly as the lesson writes it:
    # softmax(Q @ K.T / sqrt(d_k)) @ V   (one output row per asking word)
    # -----------------------------------------------------------------------
    out = weights @ V
    print("ALL AT ONCE — softmax(Q @ K.T / sqrt(d_k)) @ V:")
    for name, row in zip(words, np.round(out, 2)):
        print("  {:<6} -> {}".format(name, row))
    assert np.allclose(np.round(out[2], 2), [7.63, 2.85]), \
        "not yet — bank's output row should match the by-hand blend [7.63, 2.85]"

    # -----------------------------------------------------------------------
    # All five steps checked out. You just wrote the heart of the transformer:
    # score every pair -> scale -> mask -> split one attention budget with
    # softmax -> spend it on the Values.
    # -----------------------------------------------------------------------
    print("\nyou got it")
