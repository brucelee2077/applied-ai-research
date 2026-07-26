# =============================================================================
# Module 4 · Day 2 — Query, Key, Value  (self-attention, by hand)
# =============================================================================
# Today's whole lesson in one runnable file. We take a tiny 6-word sentence,
# give every word three roles (Query = its question, Key = its label, Value =
# its content), let one word SCORE its Query against every Key, SCALE those
# scores by sqrt(d_k), SOFTMAX them into shares that add to 1, and BLEND the
# Values by those shares into one fresh, context-aware word.
#
# The sentence:   the  animal  crossed  the  river  bank
# The searcher:   word 5, "bank" — asking "what kind of place am I?"
# The ending:     "bank" listens most to "river", so its new vector leans
#                 toward river's Value (river-bank, not money-bank).
#
# Run it:  python3 sessions/m03-attention/day-02-qkv/experiment.py
# Every number printed here also appears in the lesson page, so you can check
# the lesson against the code and the code against the lesson.
# =============================================================================

import numpy as np  # numpy just gives us fast, tidy list-and-grid math


def softmax(scores):
    """Turn a list of scores into positive shares that add up to exactly 1.

    This is the "one pizza of attention" step. We slide the scores down by
    their max first (a standard trick) so that exp() never overflows — it does
    not change the answer, because softmax only cares about differences.
    """
    shifted = scores - np.max(scores)      # subtract the biggest score (safe exp)
    ex = np.exp(shifted)                   # bigger score -> much bigger slice
    return ex / ex.sum()                   # divide by the total -> slices sum to 1


WORDS = ["the", "animal", "crossed", "the", "river", "bank"]

# --- Step 1: a tiny sentence as embeddings -----------------------------------
# Each ROW is one word's Day-1 embedding (its profile card), d_model = 4.
# To keep the arithmetic readable the 4 slots stand for made-up meaning axes:
#            [ water ,  place ,  money ,  action ]
x = np.array([
    [0.0, 0.2, 0.0, 0.0],   # 0 "the"     — bland filler, barely points anywhere
    [0.0, 0.2, 0.0, 1.0],   # 1 "animal"  — an actor / doer
    [0.0, 0.4, 0.0, 1.0],   # 2 "crossed" — an action
    [0.0, 0.2, 0.0, 0.0],   # 3 "the"     — bland filler again
    [1.0, 1.0, 0.0, 0.0],   # 4 "river"   — water AND place
    [0.0, 1.0, 1.0, 0.0],   # 5 "bank"    — place, and ambiguous about money
], dtype=float)
d_model = x.shape[1]
print("Step 1 — embeddings x, shape", x.shape, "(6 words x d_model =", d_model, ")")
print(x, "\n")

# --- Step 2: three DIFFERENT filters make Q, K, V ---------------------------
# One embedding, three grids. In a real model these grids are LEARNED; here we
# hand-pick them so the story is checkable. NOTE they are all DIFFERENT grids —
# that is the whole point. Reusing one grid for Q and K would make the score
# matrix symmetric and delete every directional relation.
#
# W_Q: turns a card into a QUESTION  (d_model x d_k)
#      row = input axis, column = "am I asking about water / place / ... ?"
Wq = np.array([
    [0.1, 0.1, 0.0, 0.0],   # water-ness   -> mild curiosity about water+place
    [0.5, 0.5, 0.0, 0.0],   # place-ness   -> asks about water and place
    [0.5, 0.0, 0.0, 0.0],   # money-ness   -> asks harder about water
    [0.0, 0.0, 0.5, 0.5],   # action-ness  -> asks about money/action instead
])
# W_K: turns a card into an ADVERTISED LABEL  (d_model x d_k) — a different grid
Wk = np.array([
    [4.4, 0.0, 0.0, 0.0],   # water-ness   -> shouts "I am watery"
    [0.0, 2.0, 0.0, 0.0],   # place-ness   -> "I am a place"
    [0.0, 0.0, 4.0, 0.0],   # money-ness   -> "I am about money"
    [0.0, 0.0, 0.0, 4.0],   # action-ness  -> "I am an action"
])
# W_V: turns a card into CONTENT  (d_model x d_v) — note d_v = 2, NOT d_model.
# d_v is a free choice: the Value only ever gets poured, never compared.
Wv = np.array([
    [0.9, 0.0],             # water-ness   -> watery content
    [0.1, 0.1],             # place-ness   -> a little of both
    [0.0, 0.9],             # money-ness   -> money content
    [0.2, 0.0],             # action-ness  -> a little watery content
])

Q = x @ Wq   # every word's Query  (its question)
K = x @ Wk   # every word's Key    (its advertised label)
V = x @ Wv   # every word's Value  (the content it will share)
d_k, d_v = Q.shape[1], V.shape[1]
print("Step 2 — one card, three DIFFERENT filters, three different outputs")
print("         Q shape", Q.shape, " K shape", K.shape, " V shape", V.shape,
      " (d_k =", d_k, ", d_v =", d_v, ")")
for i, w in enumerate(WORDS):
    print("         %-8s Q=%s  K=%s  V=%s" % (w, Q[i], K[i], V[i]))
# The forbidden thing, checked out loud: Q and K must NOT be the same numbers.
print("         is Q identical to K?", np.array_equal(Q, K), "  <- must be False")
print("         is K identical to V?", K.shape == V.shape and np.array_equal(K, V),
      "  <- must be False\n")

# --- Step 3: word 5 ("bank") scores its Query against every Key -------------
# The dot product = multiply slot-by-slot and add. Big number = strong match.
searcher = 5                     # "bank"
q_bank = Q[searcher]             # "bank"'s question
raw_scores = K @ q_bank          # one score per word
print("Step 3 — raw match scores (bank's Query · every Key):", raw_scores)
print("         %-8s -> %s" % ("river", raw_scores[4]), " (the big one)")
# Asymmetry: because Wq != Wk, "A looks at B" is NOT "B looks at A".
print("         asymmetry check: bank->river =", raw_scores[4],
      " but river->bank =", K[searcher] @ Q[4], "\n")

# --- Step 4: SCALE the scores by sqrt(d_k) ----------------------------------
# d_k = the length of each Key card. Long cards make the dot-product sum swell
# like sqrt(d_k), which would make softmax "clip" onto one winner. Dividing by
# sqrt(d_k) cancels exactly that growth. This is the "scaled" in
# "scaled dot-product attention".
scaled_scores = raw_scores / np.sqrt(d_k)   # divide, THEN softmax
print("Step 4 — d_k =", d_k, " so divide by sqrt(d_k) =", np.sqrt(d_k))
print("         scaled scores:", scaled_scores, "\n")

# --- Step 5: SOFTMAX the scaled scores into attention weights ---------------
# Positive shares that add up to 1 — the recipe for how much to listen to each.
weights = softmax(scaled_scores)
print("Step 5 — attention weights (shares):", np.round(weights, 3))
print("         they sum to:", weights.sum())
winner = int(np.argmax(weights))
print("         biggest slice goes to word %d = '%s'\n" % (winner, WORDS[winner]))

# --- Step 6: BLEND the Values by those shares -> one fresh word --------------
# Weighted sum: pour in each word's Value scaled by its share, add them up.
output = weights @ V
print("Step 6 — new context-aware 'bank' = weighted sum of Values:",
      np.round(output, 3))
print("         every term of that sum, so you can check it by hand:")
for i, w in enumerate(WORDS):
    print("           %.3f x V(%-7s) = %s" % (weights[i], w, np.round(weights[i] * V[i], 3)))
dist_to_river = np.linalg.norm(output - V[4])
dist_to_money = np.linalg.norm(output - V[5])
print("         distance to river's Value:", round(dist_to_river, 3))
print("         distance to old bank's Value:", round(dist_to_money, 3), "\n")


if __name__ == "__main__":
    # ---- SELF-CHECK: the things the lesson says you should see --------------
    # These expected numbers were produced by running this very script, so they
    # are the ground truth for "did I build attention correctly?".

    # (a) Q, K, V are three DIFFERENT views — three different filters.
    assert not np.allclose(Q, K), \
        "Q and K must come from DIFFERENT grids (tying W_Q = W_K makes the scores symmetric)"
    assert V.shape != K.shape or not np.allclose(K, V), \
        "the Value must not be a copy of the Key — the address and the payload are separate"

    # (b) The attention weights are all positive and sum to exactly 1.
    assert np.all(weights >= 0), "attention weights must all be positive"
    assert abs(weights.sum() - 1.0) < 1e-9, "attention weights must sum to 1.0"

    # (c) "bank" listens most to "river" (word 4 wins the biggest slice).
    expected_winner = 4  # "river"
    if winner != expected_winner:
        print("❌ not yet — expected the winner to be word %d ('river'), got word %d ('%s')"
              % (expected_winner, winner, WORDS[winner]))
        raise SystemExit(1)

    # (d) The blended output leans toward river's Value, not old bank's Value.
    assert dist_to_river < dist_to_money, \
        "the new 'bank' should sit closer to river's Value than to its own old Value"

    # (e) The exact numbers the lesson page quotes.
    assert np.allclose(raw_scores, [0.2, 0.2, 0.4, 0.2, 5.4, 1.0], atol=1e-9), \
        "raw scores should be [0.2, 0.2, 0.4, 0.2, 5.4, 1.0]"
    assert np.allclose(scaled_scores, [0.1, 0.1, 0.2, 0.1, 2.7, 0.5], atol=1e-9), \
        "scaled scores should be [0.1, 0.1, 0.2, 0.1, 2.7, 0.5]"
    expected_output = np.array([0.742, 0.154])
    if not np.allclose(output, expected_output, atol=1e-3):
        print("❌ not yet — expected output ≈", expected_output, "but got", np.round(output, 3))
        raise SystemExit(1)

    print("✅ you got it — 'bank' put %.0f%% of its attention on 'river', and its new "
          "vector %s leans river-flavored (watery %.2f vs money %.2f). "
          "That's self-attention."
          % (weights[4] * 100, np.round(output, 3), output[0], output[1]))
