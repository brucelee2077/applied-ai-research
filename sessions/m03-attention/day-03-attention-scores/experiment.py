# day-03-attention-scores — experiment
#
# GOAL (from the lesson's PRODUCE step): build the attention-scoring pipeline
# by hand on a tiny 3-word example — score -> scale -> softmax -> mask — and
# watch a row of raw match numbers turn into a clean "budget" of shares that
# add up to exactly 1. This is the exact move at the heart of every transformer.
#
# Run it:  python3 sessions/m03-attention/day-03-attention-scores/experiment.py

import numpy as np  # numpy does the slot-by-slot multiply-and-add (the dot product) for us


# ---------------------------------------------------------------------------
# softmax: turn one ROW of scores into positive shares that add up to exactly 1.
# The biggest score gets the biggest slice — like splitting one pizza by hunger.
# ---------------------------------------------------------------------------
def softmax_row(scores):
    # Subtract the max first. This does NOT change the answer (a math fact),
    # but it keeps exp() from overflowing when a score is a big number.
    shifted = scores - np.max(scores)
    # Step 1: raise e to the power of each score -> everything becomes positive.
    grown = np.exp(shifted)
    # Step 2 + 3: add them all up, then divide each one by that total.
    # Now the row sums to 1 — a "budget" split into shares.
    return grown / np.sum(grown)


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # STEP 0 — our tiny world: 3 words, each with a Query list and a Key list.
    # We pick small whole numbers on purpose so we can check the math by eye.
    # The Query is the "what am I looking for?" side; the Key is "what I offer".
    # d_k = how many slots each list has (here d_k = 3).
    # -----------------------------------------------------------------------
    words = ["the", "river", "bank"]

    # One Query row per word (the asking side).
    Q = np.array([
        [1.0, 0.0, 0.0],   # "the"   is looking for slot-1 things
        [0.0, 1.0, 0.0],   # "river" is looking for slot-2 things
        [2.0, 0.0, 1.0],   # "bank"  is looking for slot-1 + slot-3 things
    ])

    # One Key row per word (the offering side).
    K = np.array([
        [0.0, 0.0, 1.0],   # "the"   offers slot-3 only  (weak match for bank -> 1)
        [3.0, 0.0, 1.0],   # "river" offers slot-1 + slot-3  (matches bank well! -> 7)
        [1.0, 2.0, 1.0],   # "bank"  offers a bit of everything (-> 3)
    ])

    d_k = Q.shape[1]  # number of slots in each list = 3

    # -----------------------------------------------------------------------
    # STEP 1 — the score matrix: score EVERY Query against EVERY Key at once.
    # Q @ K.T lays every Query as a row and every Key as a column, then does
    # the dot product (multiply slot-by-slot, add) for every pair in one move.
    # PREDICT: "bank" is looking for slots 1 and 3, and "river" offers exactly
    # slots 1 and 3 — so bank should score highest against river.
    # -----------------------------------------------------------------------
    scores = Q @ K.T
    print("STEP 1 — score matrix (row = asking word, column = offered word):")
    print(scores)
    # bank's row is the last row: [the, river, bank]. Check it matches the lesson grid [1, 7, 3].
    bank_row = scores[2]
    print("bank's row (the / river / bank):", bank_row)
    # Hand-check the star cell: bank's Query [2,0,1] . river's Key [3,0,1] = 2*3 + 0*0 + 1*1 = 7.
    assert np.array_equal(bank_row, np.array([1.0, 7.0, 3.0])), \
        "❌ not yet — expected bank's row to be [1, 7, 3]"
    print("  bank listens most to 'river' (score 7) — just like the lesson's grid.\n")

    # -----------------------------------------------------------------------
    # STEP 2 — scale: divide the whole grid by sqrt(d_k) to turn the volume down.
    # NOTICE: numbers get smaller, but WHO-BEATS-WHOM never changes. Scaling
    # changes the loudness, never the winner.
    # -----------------------------------------------------------------------
    scaled = scores / np.sqrt(d_k)
    print("STEP 2 — scaled by sqrt(d_k) = sqrt(3):")
    print(scaled)
    # The order (argsort) of bank's row must be identical before and after scaling.
    assert np.array_equal(np.argsort(scores[2]), np.argsort(scaled[2])), \
        "❌ not yet — expected scaling to keep the same order (who beats whom)"
    print("  order unchanged: river still wins for 'bank'.\n")

    # -----------------------------------------------------------------------
    # STEP 3 — softmax each row: raw scores become a budget of shares summing to 1.
    # We use the lesson's exact demo row [2.0, 1.0, 0.1] so the numbers match:
    # it should split into shares [0.66, 0.24, 0.10].
    # -----------------------------------------------------------------------
    demo_scores = np.array([2.0, 1.0, 0.1])
    demo_shares = softmax_row(demo_scores)
    print("STEP 3 — softmax([2.0, 1.0, 0.1]):", np.round(demo_shares, 2))
    print("         row sum =", round(float(demo_shares.sum()), 4), "(a full budget of 1)")
    # Check the shares match the lesson (rounded to 2 decimals) and that they sum to 1.
    assert np.allclose(np.round(demo_shares, 2), [0.66, 0.24, 0.10]), \
        "❌ not yet — expected shares [0.66, 0.24, 0.10]"
    assert np.isclose(demo_shares.sum(), 1.0), "❌ not yet — expected the shares to sum to 1"
    print("  every share is positive and the row adds to exactly 1.\n")

    # -----------------------------------------------------------------------
    # STEP 4 — masking: some words are FORBIDDEN (a future word, or empty padding).
    # We score [river 3.0, FUTURE 2.0, pad 1.0]. FUTURE and pad are not allowed.
    # WATCH: without masking, FUTURE steals a real 0.24 slice of the budget.
    # Set forbidden scores to a huge negative number BEFORE softmax, and their
    # shares collapse to ~0 while river keeps the whole pizza.
    # -----------------------------------------------------------------------
    mask_scores = np.array([3.0, 2.0, 1.0])          # river, FUTURE, pad
    unmasked = softmax_row(mask_scores)
    print("STEP 4 — no mask   [river, FUTURE, pad]:", np.round(unmasked, 2), "(FUTURE steals attention!)")
    # Without a mask, FUTURE grabs about 0.24 — a quarter of the budget it should not have.
    # (river's exact share is 0.665; the lesson prose rounds it down to 0.66, we keep the true 0.67.)
    assert np.allclose(np.round(unmasked, 2), [0.67, 0.24, 0.09]), \
        "❌ not yet — expected unmasked shares [0.67, 0.24, 0.09]"

    masked_scores = mask_scores.copy()
    masked_scores[1] = -1e9  # mute FUTURE (set its score to ~ -infinity before softmax)
    masked_scores[2] = -1e9  # mute pad
    masked = softmax_row(masked_scores)
    print("STEP 4 — masked    [river, FUTURE, pad]:", np.round(masked, 2), "(forbidden shares -> 0)")
    # After masking, river should get essentially the whole budget (1.00) and the others ~0.
    assert np.allclose(np.round(masked, 2), [1.00, 0.00, 0.00]), \
        "❌ not yet — expected masked shares [1.00, 0.00, 0.00]"
    assert np.isclose(masked.sum(), 1.0), "❌ not yet — expected masked shares to still sum to 1"
    print("  masking drove FUTURE and pad to ~0 and handed the freed budget to river.\n")

    # -----------------------------------------------------------------------
    # All four steps checked out. You just wrote the heart of the transformer:
    # score every pair -> scale -> mask -> split one attention budget with softmax.
    # -----------------------------------------------------------------------
    print("✅ you got it")
