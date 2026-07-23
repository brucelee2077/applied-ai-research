# =============================================================================
# Module 4 · Day 2 — Query, Key, Value  (self-attention, by hand)
# =============================================================================
# Today's whole lesson in one runnable file. We take a tiny 4-word sentence,
# give every word three roles (Query = its question, Key = its label, Value =
# its content), let one word SCORE its Query against every Key, SCALE those
# scores, SOFTMAX them into shares that add to 1, and BLEND the Values by those
# shares into one fresh, context-aware word.
#
# The sentence (4 words):   the   river   money   bank
# The searcher:             word 3, "bank" — asking "what kind of place am I?"
# The hoped-for ending:     "bank" listens most to "river", so its new vector
#                           leans toward river's Value (river-bank, not money).
#
# Run it:  python3 sessions/m03-attention/day-02-qkv/experiment.py
# A curious 12-year-old can read this top to bottom — one comment per step.
# =============================================================================

import numpy as np  # numpy just gives us fast, tidy list-and-grid math


def softmax(scores):
    """Turn a list of scores into positive shares that add up to exactly 1.

    This is the "one pizza of attention" step. We slide the scores down by
    their max first (a standard trick) so that exp() never overflows — it does
    not change the answer, because softmax only cares about differences.
    """
    shifted = scores - np.max(scores)      # subtract the biggest score (safe exp)
    ex = np.exp(shifted)                   # exponentiate: bigger score -> much bigger slice
    return ex / ex.sum()                   # divide by the total so the slices sum to 1


# --- Step 1: a tiny sentence as embeddings ----------------------------------
# Each ROW is one word's Day-1 embedding (its profile card). 4 words, length 2.
# We hand-pick simple 0/1 cards so the numbers stay easy to follow by hand.
x = np.array([
    [0.0, 0.0],   # word 0: "the"    (a bland filler word — points nowhere)
    [2.0, 0.0],   # word 1: "river"  (points strongly along the first axis)
    [0.0, 1.0],   # word 2: "money"  (points along the second axis)
    [1.0, 0.2],   # word 3: "bank"   (confused: mostly the river-axis, a little money)
], dtype=float)
print("Step 1 — embeddings x, shape", x.shape)
print(x, "\n")

# --- Step 2: three DIFFERENT filters make Q, K, V ---------------------------
# One embedding, three grids (Wq, Wk, Wv). Same input, three different outputs.
# In a real model these grids are LEARNED; here we just pick them so the story
# works and the math is checkable. Each grid is 2x2 (card length in -> out).
Wq = np.array([[1.0, 0.0], [0.0, 1.0]])   # Query filter (here: identity — keep the card as-is)
Wk = np.array([[1.0, 0.0], [0.0, 1.0]])   # Key filter   (here: identity too)
Wv = np.array([[2.0, 0.0], [0.0, 2.0]])   # Value filter (here: double the card — distinct content)

Q = x @ Wq   # every word's Query  (its question)
K = x @ Wk   # every word's Key    (its advertised label)
V = x @ Wv   # every word's Value  (the content it will share)
print("Step 2 — Q, K, V are three different views of the SAME words")
print("Q shape", Q.shape, " K shape", K.shape, " V shape", V.shape)
print("Q =\n", Q)
print("V =\n", V, "\n")

# --- Step 3: word 3 ("bank") scores its Query against every Key -------------
# The dot product = multiply slot-by-slot and add. Big number = strong match.
q_bank = Q[3]                    # "bank"'s question
raw_scores = K @ q_bank          # one score per word: how well each Key matches
print("Step 3 — raw match scores (bank's Query · every Key):", raw_scores, "\n")

# --- Step 4: SCALE the scores by sqrt(d_k) ----------------------------------
# d_k = the length of each Key card. Long cards make the dot-product sum swell,
# which would make softmax "clip" to one winner. Dividing by sqrt(d_k) cancels
# that growth so the scores stay a sensible size. This is the "scaled" in
# "scaled dot-product attention".
d_k = K.shape[1]                          # card length = 2 here
scaled_scores = raw_scores / np.sqrt(d_k) # divide, THEN softmax
print("Step 4 — d_k =", d_k, " scaled scores:", scaled_scores, "\n")

# --- Step 5: SOFTMAX the scaled scores into attention weights ---------------
# Positive shares that add up to 1 — the recipe for how much to listen to each.
weights = softmax(scaled_scores)
print("Step 5 — attention weights (shares):", weights)
print("         they sum to:", weights.sum())
winner = int(np.argmax(weights))
words = ["the", "river", "money", "bank"]
print("         biggest slice goes to word %d = '%s'\n" % (winner, words[winner]))

# --- Step 6: BLEND the Values by those shares -> one fresh word --------------
# Weighted sum: pour in each word's Value scaled by its share, add them up.
output = weights @ V
print("Step 6 — new context-aware 'bank' = weighted sum of Values:", output)
# Distance check: is the new vector closer to river's Value or money's Value?
dist_to_river = np.linalg.norm(output - V[1])
dist_to_money = np.linalg.norm(output - V[2])
print("         distance to river's Value:", round(dist_to_river, 3))
print("         distance to money's Value:", round(dist_to_money, 3), "\n")


if __name__ == "__main__":
    # ---- SELF-CHECK: the four things the lesson says you should see ----------
    # These expected numbers were produced by running this very script, so they
    # are the ground truth for "did I build attention correctly?".

    # (a) Q, K, V are three DIFFERENT views (Value must differ from the Key).
    assert not np.allclose(K, V), "Q/K/V should be three DIFFERENT filters, not the same grid"

    # (b) The attention weights are all positive and sum to exactly 1 (one pizza).
    assert np.all(weights >= 0), "attention weights must all be positive"
    assert abs(weights.sum() - 1.0) < 1e-9, "attention weights must sum to 1.0"

    # (c) "bank" listens most to "river" (word 1 wins the biggest slice).
    expected_winner = 1  # "river"
    if winner != expected_winner:
        print("❌ not yet — expected the winner to be word %d ('river'), got word %d ('%s')"
              % (expected_winner, winner, words[winner]))
        raise SystemExit(1)

    # (d) The blended output leans toward river's Value, not money's Value.
    assert dist_to_river < dist_to_money, \
        "the new 'bank' should sit closer to river's Value than money's Value"

    # (e) The exact output vector matches the hand-checkable expected value.
    expected_output = np.array([2.470, 0.376])
    if not np.allclose(output, expected_output, atol=1e-2):
        print("❌ not yet — expected output ≈", expected_output, "but got", np.round(output, 3))
        raise SystemExit(1)

    print("✅ you got it — 'bank' put %.0f%% of its attention on 'river', and its new "
          "vector %s leans river-flavored. That's self-attention."
          % (weights[1] * 100, np.round(output, 3)))
