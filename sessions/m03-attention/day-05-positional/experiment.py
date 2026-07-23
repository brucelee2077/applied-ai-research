# day-05-positional — experiment
#
# GOAL: catch attention being "order-blind" (a shuffle changes nothing),
#       then fix it with a positional encoding — a "seat number" added to
#       each word so order finally carries meaning.
#
# Run me:  python3 sessions/m03-attention/day-05-positional/experiment.py
#
# Everything below is heavily commented so you can read it top to bottom.

import numpy as np  # numpy gives us fast arrays and matrix math


# -----------------------------------------------------------------------------
# STEP 1 — a tiny self-attention, exactly the Day-3 recipe:
#          softmax(Q @ Kᵀ / sqrt(d_k)) @ V.
#          Here (to keep it simple) Q, K, V are just the input x itself.
# -----------------------------------------------------------------------------
def softmax(scores):
    # subtract the row max first so exp() never blows up (a stability trick)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    # turn each score into a positive weight
    exp = np.exp(shifted)
    # divide each row by its own total so every row of weights adds up to 1
    return exp / exp.sum(axis=-1, keepdims=True)


def self_attention(x):
    # x has shape (seq, d_model): one row per word, d_model numbers per word
    d_k = x.shape[1]  # the width of each word — used to scale the scores
    # every word rates every other word: scores[i][j] = word i · word j
    scores = x @ x.T
    # scale by sqrt(d_k) so the numbers stay a sensible size before softmax
    scores = scores / np.sqrt(d_k)
    # turn the scores into attention weights (each row sums to 1)
    weights = softmax(scores)
    # each word's answer is a weighted blend of ALL the words
    return weights @ x


# -----------------------------------------------------------------------------
# STEP 2 — the fix: sinusoidal positional encoding (the "seat stamp").
#          Even slots use sine, odd slots use cosine, and the waves run at
#          spreading speeds (fast in front slots, slow in back slots).
#          KEY FACT from the lesson: seat 0 comes out as [0, 1, 0, 1, ...]
#          because sin(0) = 0 and cos(0) = 1.
# -----------------------------------------------------------------------------
def sinusoidal_encoding(max_len, d_model):
    # start with an empty table: one row per seat, d_model numbers per seat
    pe = np.zeros((max_len, d_model))
    # which seat each row is (0, 1, 2, ...), shaped as a column
    positions = np.arange(max_len)[:, None]
    # which slot-pair we are in (0, 1, 2, ...) — one per pair of slots
    pair_index = np.arange(d_model // 2)
    # the "10000" knob spreads the wave speeds: front pairs fast, back pairs slow
    div = np.power(10000.0, (2 * pair_index) / d_model)
    # even slots (0, 2, 4, ...) get the sine wave
    pe[:, 0::2] = np.sin(positions / div)
    # odd slots (1, 3, 5, ...) get the cosine wave
    pe[:, 1::2] = np.cos(positions / div)
    return pe


# -----------------------------------------------------------------------------
# Helper: a "set of per-word outputs" is order-blind if, after we sort both
# result tables, they contain the same rows. We use this to check the bug.
# -----------------------------------------------------------------------------
def sorted_rows(matrix):
    # round to kill tiny floating-point noise, then sort rows so order drops out
    rounded = np.round(matrix, 6)
    # sort by the string form of each row — a stable way to compare row-sets
    order = np.argsort([str(r) for r in rounded])
    return rounded[order]


if __name__ == "__main__":
    # fix the random seed so the numbers are the same every run
    np.random.seed(0)

    # a tiny sentence: 3 words, each word is 4 numbers wide (seq=3, d_model=4)
    seq, d_model = 3, 4
    x = np.random.rand(seq, d_model)

    # --- (a) run attention on the original word order ---
    out_original = self_attention(x)
    print("attention on original order:")
    print(np.round(out_original, 4))

    # --- (b) shuffle the word rows and run attention again ---
    perm = [2, 0, 1]  # a new seating order for the same three words
    out_shuffled = self_attention(x[perm])
    print("\nattention on shuffled order:")
    print(np.round(out_shuffled, 4))

    # THE BUG: the SET of per-word outputs is identical — order changed nothing.
    order_blind = np.allclose(sorted_rows(out_original), sorted_rows(out_shuffled))
    print("\nsame set of outputs after shuffle? ", order_blind, " <- attention is order-blind")

    # --- (c) build the seat stamps and inspect seat 0 ---
    pe = sinusoidal_encoding(max_len=6, d_model=d_model)
    print("\nsinusoidal encoding, seat 0:", np.round(pe[0], 4))
    # every seat's stamp must be unique (no two rows the same)
    unique_seats = len({tuple(np.round(row, 6)) for row in pe}) == pe.shape[0]
    print("all 6 seats unique fingerprints? ", unique_seats)

    # --- (d) add the seat stamp, then order finally matters ---
    x_with_pos = x + pe[:seq]           # original order + its seats
    x_shuf_pos = x[perm] + pe[:seq]     # shuffled order + THE SAME seats
    out_pos = self_attention(x_with_pos)
    out_shuf_pos = self_attention(x_shuf_pos)
    # THE FIX: now the two output sets are DIFFERENT — order carries meaning.
    order_now_matters = not np.allclose(sorted_rows(out_pos), sorted_rows(out_shuf_pos))
    print("outputs differ after shuffle now? ", order_now_matters, " <- the seat stamp fixed it")

    # =========================================================================
    # SELF-CHECK — three facts the lesson promised must all hold:
    #   1. plain attention is order-blind (shuffle changes nothing),
    #   2. seat 0 of the sinusoidal encoding is exactly [0, 1, 0, 1],
    #   3. after adding positions, shuffling DOES change the answer.
    # =========================================================================
    expected_seat0 = np.array([0.0, 1.0, 0.0, 1.0])  # sin(0)=0, cos(0)=1
    try:
        # fact 1: attention alone ignores order
        assert order_blind, "attention was NOT order-blind"
        # fact 2: the exact seat-0 stamp the lesson states
        assert np.allclose(pe[0], expected_seat0), \
            "seat 0 should be %s" % expected_seat0
        # fact 3: positional encoding makes order matter
        assert order_now_matters, "positions did NOT make order matter"
        print("\n✅ you got it")
    except AssertionError as err:
        print("\n❌ not yet — expected seat 0 = %s, order-blind then order-aware. (%s)"
              % (expected_seat0, err))
        raise
