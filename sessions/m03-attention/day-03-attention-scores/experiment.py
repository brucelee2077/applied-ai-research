# day-03-attention-scores — experiment
#
# Today's big idea in two lines of output:
#   Nine raw match numbers turn into one budget of shares that adds up to 1.
#   Spend that budget on the Values and "bank" comes out river-flavoured: [7.63, 2.85].
# We walk the lesson's own 3-word sentence "the river bank" through all five beats: score
# every pair -> divide by sqrt(d_k) -> softmax -> mask -> blend the Values. All of it is
# one line of maths:  softmax(Q @ K.T / sqrt(d_k)) @ V
# Run it:  python3 sessions/m03-attention/day-03-attention-scores/experiment.py

import numpy as np  # numpy holds the lists as arrays and does every dot product at once with @


def hand_dot(list_a, list_b):
    # The dot product done the paper way: multiply slot by slot, then add the products.
    total = 0.0
    for slot_a, slot_b in zip(list_a, list_b):
        total += slot_a * slot_b
    return total


def hand_score_matrix(queries, keys):
    # One hand-made dot product per PAIR. Row = the word asking, column = the word offered.
    return np.array([[hand_dot(query, key) for key in keys] for query in queries])


def softmax_row(scores):
    # Turn one row of scores into positive shares that add up to 1.
    # Taking the row's biggest score away first does not change the answer (softmax
    # reads only the gaps between scores) and it stops exp() from overflowing.
    grown = np.exp(scores - np.max(scores))
    return grown / np.sum(grown)


if __name__ == "__main__":
    # --- Part 1: the tiny world, exactly the numbers from the lesson ---------
    words = ["the", "river", "bank"]
    Q = np.array([[0.0, 0.0, 1.0, 1.0],     # "the"   asks for nothing in particular
                  [1.0, 0.0, 1.0, 1.0],     # "river" asks a little for slot-1 things
                  [2.0, 1.0, 1.0, 1.0]])    # "bank"  asks hard for slot-1 things
    K = np.array([[0.0, 0.0, 1.0, 1.0],     # "the"   offers only the two shared slots
                  [2.0, 0.0, 1.0, 1.0],     # "river" offers a lot of slot 1
                  [0.0, 2.0, 1.0, 1.0]])    # "bank"  offers a lot of slot 2
    V = np.array([[0.0, 6.0], [10.0, 2.0], [4.0, 4.0]])   # what each word hands over
    d_k = Q.shape[1]                        # d_k = how many slots each Query and Key has
    print("Q", Q.shape, " K", K.shape, " V", V.shape, " d_k =", d_k, " sqrt(d_k) =", np.sqrt(d_k))

    # --- Part 2: score every pair (the score matrix) -------------------------
    scores = Q @ K.T                        # every Query row meets every Key column
    by_hand = hand_score_matrix(Q, K)       # the same nine numbers, one paper dot product at a time
    print("\n--- Part 2: the score matrix (row = who asks, column = who is offered) ---")
    print(scores)
    print("the @ shortcut and the by-hand loop agree:", np.array_equal(scores, by_hand))
    print("bank's row", scores[2], "-> bank leans on", words[int(np.argmax(scores[2]))])

    # --- Part 3: turn the volume down by dividing by sqrt(d_k) ---------------
    scaled = scores / np.sqrt(d_k)
    print("\n--- Part 3: scaled scores (every cell divided by sqrt(d_k)) ---")
    print(scaled)
    # Who beats whom: sort bank's row and read the word names, weakest first.
    rank_before = [words[i] for i in np.argsort(scores[2], kind="stable")]
    rank_after = [words[i] for i in np.argsort(scaled[2], kind="stable")]
    print("bank's ranking before:", rank_before, " after:", rank_after)

    # --- Part 4: softmax splits one budget into shares -----------------------
    weights = np.array([softmax_row(row) for row in scaled])
    print("\n--- Part 4: softmax over each row (the attention weights) ---")
    print(np.round(weights, 4))
    print("row sums:", weights.sum(axis=1), "(one whole budget per asking word)")
    # PREDICT from the RAW scores, before reading any share: the biggest score must win
    # the biggest share, because softmax re-scales a row and never re-orders it.
    predicted_winner = words[int(np.argmax(scores[2]))]
    actual_winner = words[int(np.argmax(weights[2]))]
    print("predicted biggest share:", predicted_winner, " actual:", actual_winner)

    # --- Part 5: mask two words so nobody can listen to them -----------------
    mask_labels = ["the", "river", "FUTURE", "pad"]
    mask_scores = np.array([1.0, 3.0, 2.0, 0.0])            # the lesson's 4-score row
    forbidden = np.array([False, False, True, True])         # FUTURE and pad are off limits
    open_shares = softmax_row(mask_scores)                   # what happens with no mask
    masked_scores = np.where(forbidden, -1e9, mask_scores)   # huge negative score BEFORE softmax
    masked_shares = softmax_row(masked_scores)
    print("\n--- Part 5: masking (a forbidden word gets a huge negative score) ---")
    print("labels  :", mask_labels)
    print("no mask :", np.round(open_shares, 4), " sum", round(float(open_shares.sum()), 6))
    print("masked  :", np.round(masked_shares, 4), " sum", round(float(masked_shares.sum()), 6))
    print("change  :", np.round(masked_shares - open_shares, 4), "(freed budget moves to allowed words)")

    # --- Part 6: spend the budget on the Values -----------------------------
    bank_shares = weights[2]
    blended = bank_shares @ V               # each share times its Value, added slot by slot
    # PREDICT from the shares alone: the answer must land nearest the Value of the word
    # holding most of the budget ("nearest" = shortest straight-line distance).
    predicted_nearest = words[int(np.argmax(bank_shares))]
    distances = np.linalg.norm(V - blended, axis=1)
    actual_nearest = words[int(np.argmin(distances))]
    print("\n--- Part 6: blend the Values by bank's shares ---")
    print("shares", np.round(bank_shares, 4), "@ V ->", np.round(blended, 4))
    print("predicted nearest Value:", predicted_nearest, " actual:", actual_nearest,
          " distances:", np.round(distances, 3))
    print("all three rows of softmax(Q @ K.T / sqrt(d_k)) @ V:")
    for name, row in zip(words, np.round(weights @ V, 4)):
        print("   {:<6} -> {}".format(name, row))

    # --- Self-check: one boolean per claim ----------------------------------
    # Every expected value is written down here, so nothing is checked against a re-derived number.
    lesson_grid = np.array([[2.0, 2.0, 2.0], [2.0, 4.0, 2.0], [2.0, 6.0, 4.0]])
    grid_ok = np.array_equal(scores, lesson_grid) and np.array_equal(by_hand, lesson_grid)
    scale_ok = np.sqrt(d_k) == 2.0 and np.array_equal(scaled[2], np.array([1.0, 3.0, 2.0]))
    order_ok = rank_before == ["the", "bank", "river"] and rank_after == ["the", "bank", "river"]
    shares_ok = np.allclose(weights[2], [0.09003057, 0.66524096, 0.24472847])
    budget_ok = np.allclose(weights.sum(axis=1), 1.0)
    winner_ok = predicted_winner == "river" and actual_winner == "river"
    open_ok = np.allclose(open_shares, [0.08714432, 0.64391426, 0.23688282, 0.03205860])
    mask_ok = (np.allclose(masked_shares[:2], [0.11920292, 0.88079708])
               and round(float(np.abs(masked_shares[forbidden]).max()), 6) == 0.0
               and np.allclose(masked_shares.sum(), 1.0))
    freed_ok = masked_shares[0] > open_shares[0] and masked_shares[1] > open_shares[1]
    blend_ok = np.allclose(blended, [7.63132344, 2.84957923])
    nearest_ok = predicted_nearest == "river" and actual_nearest == "river"

    if (grid_ok and scale_ok and order_ok and shares_ok and budget_ok and winner_ok
            and open_ok and mask_ok and freed_ok and blend_ok and nearest_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected the grid [[2,2,2],[2,4,2],[2,6,4]], bank's scaled row [1,3,2], "
              "shares [0.0900,0.6652,0.2447] summing to 1, masked shares [0.1192,0.8808,0,0], "
              "and the blend [7.6313,2.8496] nearest river's Value")

    assert grid_ok, "Q @ K.T must give [[2,2,2],[2,4,2],[2,6,4]], and so must the by-hand loop"
    assert scale_ok, "dividing by sqrt(4)=2 must turn bank's row [2,6,4] into [1,3,2]"
    assert order_ok, "scaling must keep the ranking the -> bank -> river"
    assert shares_ok, "softmax([1,3,2]) must give [0.0900, 0.6652, 0.2447]"
    assert budget_ok, "every row of shares must add up to 1 — one whole budget"
    assert winner_ok, "the biggest score must also win the biggest share"
    assert open_ok, "unmasked softmax([1,3,2,0]) must give [0.0871, 0.6439, 0.2369, 0.0321]"
    assert mask_ok, "masking before softmax must give [0.1192, 0.8808, 0, 0] and still sum to 1"
    assert freed_ok, "the budget freed by masking must flow to the allowed words"
    assert blend_ok, "bank's shares times the Values must give [7.6313, 2.8496]"
    assert nearest_ok, "the blend must land nearest river's Value, the word holding most budget"
