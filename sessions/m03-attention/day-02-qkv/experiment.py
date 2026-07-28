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
    """Turn scores into positive shares that add up to 1 — the "one pizza" step.

    We slide every score down by the largest one first, so exp() cannot overflow.
    The shift does not change the answer: softmax reads only the gaps.
    """
    slices = np.exp(scores - np.max(scores))   # bigger score -> much bigger slice
    return slices / slices.sum()               # divide by the total -> shares add to 1


def matches(computed, expected):
    """True when a computed number or grid equals a written-down one (6 decimals)."""
    return np.array_equal(np.round(np.asarray(computed, dtype=float), 6),
                          np.asarray(expected, dtype=float))


if __name__ == "__main__":
    WORDS = ["the", "animal", "crossed", "the", "river", "bank"]
    bank, river = 5, 4          # the two words we follow all day

    # --- Part 1: the sentence as six embedding cards ------------------------
    # Each ROW is one word's Day-1 embedding — its profile card. A card is four
    # numbers wide, and that width has a name: d_model. The four slots are
    # made-up meaning axes:  water  place  money  action
    x = np.array([[0.0, 0.2, 0.0, 0.0],    # 0 "the"     — filler, points almost nowhere
                  [0.0, 0.2, 0.0, 1.0],    # 1 "animal"  — a doer
                  [0.0, 0.4, 0.0, 1.0],    # 2 "crossed" — an action
                  [0.0, 0.2, 0.0, 0.0],    # 3 "the"     — filler again
                  [1.0, 1.0, 0.0, 0.0],    # 4 "river"   — watery AND a place
                  [0.0, 1.0, 1.0, 0.0]])   # 5 "bank"    — a place, unsure about money
    print("Part 1 — embeddings x, shape", x.shape, " d_model =", x.shape[1])

    # --- Part 2: one card, three DIFFERENT filters -> Q, K, V ---------------
    # A filter is a grid you multiply the card by: row = input slot, column =
    # output slot. A real model learns these three grids; ours are hand-picked.
    # W_Q makes the QUERY (the question a word asks), W_K the KEY (the label it
    # advertises), W_V the VALUE (the content it shares if someone listens).
    Wq = np.array([[0.1, 0.1, 0.0, 0.0],   # W_Q water slot -> mild interest in water+place
                   [0.5, 0.5, 0.0, 0.0],   #     place slot  -> asks about water and place
                   [0.5, 0.0, 0.0, 0.0],   #     money slot  -> asks harder about water
                   [0.0, 0.0, 0.5, 0.5]])  #     action slot -> asks about money and action
    Wk = np.array([[4.4, 0.0, 0.0, 0.0],   # W_K water slot -> "I am watery"
                   [0.0, 2.0, 0.0, 0.0],   #     place slot  -> "I am a place"
                   [0.0, 0.0, 4.0, 0.0],   #     money slot  -> "I am about money"
                   [0.0, 0.0, 0.0, 4.0]])  #     action slot -> "I am an action"
    # W_V has only two columns: a Value is never compared to anything, so its
    # length d_v is a free choice and does not have to equal d_k.
    Wv = np.array([[0.9, 0.0],             # W_V water slot -> watery content
                   [0.1, 0.1],             #     place slot  -> a little of both
                   [0.0, 0.9],             #     money slot  -> money content
                   [0.2, 0.0]])            #     action slot -> a little watery content
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    d_k, d_v = Q.shape[1], V.shape[1]
    print("Part 2 — Q", Q.shape, " K", K.shape, " V", V.shape,
          " (d_k =", d_k, ", d_v =", d_v, ")")
    for i, word in enumerate(WORDS):
        print("         %-8s Q=%s K=%s V=%s" % (word, Q[i], K[i], V[i]))
    # Two different grids, so a word's question and its label are different lists.
    print("         is Q the same numbers as K?", np.array_equal(Q, K), "-> must be False")

    # --- Part 3: "bank" scores its Query against every Key ------------------
    # A dot product multiplies two lists slot by slot and adds the products up.
    # A big total means "your label answers my question".
    raw_scores = K @ Q[bank]               # one score per word
    print("\nPart 3 — bank's raw match scores:", raw_scores)
    for i, word in enumerate(WORDS):
        print("         bank -> %-8s = %.1f" % (word, raw_scores[i]))
    # Looking is one-way traffic: W_Q and W_K are different grids, so
    # "bank -> river" and "river -> bank" come out as two different numbers.
    river_to_bank = K[bank] @ Q[river]
    print("         bank->river = %.1f  but river->bank = %.1f"
          % (raw_scores[river], river_to_bank))

    # --- Part 4: scale the scores by sqrt(d_k) ------------------------------
    # d_k is the length of a Key. A dot product adds one product per slot, so long
    # cards give big scores, and big scores make softmax hand everything to one
    # word. Dividing by sqrt(d_k) cancels that growth — the "scaled" in the name.
    scaled_scores = raw_scores / np.sqrt(d_k)
    print("\nPart 4 — divide every score by sqrt(d_k) = sqrt(%d) = %.1f"
          % (d_k, np.sqrt(d_k)))
    print("         scaled scores:", scaled_scores, " (same ranking, calmer numbers)")

    # --- Part 5: softmax the scaled scores into shares ----------------------
    weights = softmax(scaled_scores)
    winner = int(np.argmax(weights))
    print("\nPart 5 — attention weights (shares):", np.round(weights, 3))
    print("         they add up to:", round(float(weights.sum()), 6))
    print("         biggest slice: word %d = '%s'" % (winner, WORDS[winner]))

    # --- Part 6: blend the Values by those shares ---------------------------
    # Weighted sum: pour in each word's Value scaled by its share, then add up.
    output = weights @ V
    print("\nPart 6 — the new context-aware 'bank':", np.round(output, 3))
    for i, word in enumerate(WORDS):
        print("         %.3f x V(%-7s) = %s"
              % (weights[i], word, np.round(weights[i] * V[i], 3)))
    dist_to_river = np.linalg.norm(output - V[river])
    dist_to_old_bank = np.linalg.norm(output - V[bank])
    print("         distance to river's Value %.2f   to bank's old Value %.2f"
          % (dist_to_river, dist_to_old_bank))

    # --- Part 7: change the question, and the winner changes ----------------
    # The lesson's slider, in code. Keep every Key exactly as it is, and let
    # "bank" ask about money instead of water (slots: water, place, money, action).
    money_query = np.array([0.0, 0.5, 1.0, 0.0])
    money_scores = K @ money_query
    money_winner = int(np.argmax(money_scores))
    print("\nPart 7 — same Keys, a money-flavoured question:", money_scores)
    print("         winner is now word %d = '%s' — a money-bank, not a river-bank"
          % (money_winner, WORDS[money_winner]))

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
    claims = [
        (Q.shape == (6, 4) and K.shape == (6, 4) and V.shape == (6, 2)
         and (d_k, d_v) == (4, 2), "expected Q (6,4), K (6,4), V (6,2), d_k 4, d_v 2"),
        (matches(Q, exp_Q) and matches(K, exp_K) and matches(V, exp_V),
         "one card through three different grids should give the lesson's Q, K and V rows"),
        (not np.array_equal(Q, K), "Q and K must differ, or the score grid turns symmetric"),
        (matches(raw_scores, [0.2, 0.2, 0.4, 0.2, 5.4, 1.0]),
         "raw scores should be [0.2 0.2 0.4 0.2 5.4 1.0], with river the 5.4"),
        (matches(river_to_bank, 1.2), "river->bank should score 1.2, not bank->river's 5.4"),
        (matches(scaled_scores, [0.1, 0.1, 0.2, 0.1, 2.7, 0.5]),
         "scaled scores should be [0.1 0.1 0.2 0.1 2.7 0.5] — raw divided by 2"),
        (bool(np.all(weights >= 0)) and matches(weights.sum(), 1.0),
         "every share must be positive and the six must add up to 1"),
        (matches(np.round(weights, 3), [0.052, 0.052, 0.058, 0.052, 0.706, 0.078]),
         "shares should be [0.052 0.052 0.058 0.052 0.706 0.078]"),
        (winner == 4 and WORDS[4] == "river", "the biggest share should land on 'river'"),
        (matches(np.round(output, 3), [0.742, 0.154]), "output should be [0.742, 0.154]"),
        (matches(np.round(dist_to_river, 2), 0.26)
         and matches(np.round(dist_to_old_bank, 2), 1.06),
         "output should sit 0.26 from river's Value and 1.06 from bank's old Value"),
        (money_winner == 5 and WORDS[5] == "bank"
         and matches(money_scores, [0.2, 0.2, 0.4, 0.2, 1.0, 5.0]),
         "a money question should move the win to 'bank', scores [.. 1.0 5.0]"),
    ]
    missed = [why for ok, why in claims if not ok]
    if not missed:
        print("\n✅ you got it — 'bank' put %.1f%% of its attention on 'river', and its new"
              " vector %s leans watery, not money. That is self-attention."
              % (weights[river] * 100, np.round(output, 3)))
    else:
        print("\n❌ not yet — " + " | ".join(missed))
    for ok, why in claims:
        assert ok, why
