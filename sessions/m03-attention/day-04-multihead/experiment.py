# day-04-multihead — experiment
#
# Today's big idea in two lines of output:
#   Ask ONE head to track two relationships and it splits the difference — both land near 0.50, muddy.
#   Give each relationship its OWN head on half the width, then join and mix with W_O — sharp, width 4 comes home.
#
# The lesson's Produce step: with d_model = 4 and h = 2 heads, predict (a) each head's answer
# width, (b) the width after joining, (c) the width after W_O. This script builds that two-head
# panel by hand and prints every width on the way.
#
# Run it:  python3 sessions/m03-attention/day-04-multihead/experiment.py

import numpy as np  # numpy gives us arrays, matrix multiply (@) and np.concatenate

# ---- Yesterday's attention, reused unchanged ------------------------------

def softmax(scores):
    # Turn each row of raw scores into shares: all positive, adding up to 1.
    # Subtract the row's biggest score first so exp() can never overflow.
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
    # One head is exactly the Day-3 formula: softmax(Q @ Kᵀ / √d_k) @ V.
    d_k = Q.shape[-1]                      # the width of one Query / Key
    scores = Q @ K.T / np.sqrt(d_k)        # one match number per pair of words
    weights = softmax(scores)              # each row of shares adds up to 1
    return weights, weights @ V            # the shares, and the blended answer

def run_head(X, W_Q, W_K, W_V):
    # A head owns three recipes. They turn every full-width word into a thin
    # d_k-wide Query, Key and Value. Then yesterday's attention runs on those.
    return attention(X @ W_Q, X @ W_K, X @ W_V)

def multi_head(X, heads, W_O):
    # heads is a list of (W_Q, W_K, W_V) — one own set of recipes per head.
    outs = [run_head(X, w_q, w_k, w_v)[1] for (w_q, w_k, w_v) in heads]
    joined = np.concatenate(outs, axis=-1)   # the staple: answers side by side
    return outs, joined, joined @ W_O        # W_O is the editor that mixes them

def close(value, expected, tol=1e-6):
    # True when a computed number lands on the number we wrote down.
    return abs(float(value) - float(expected)) < tol


if __name__ == "__main__":
    # --- Part 1: one head is yesterday's attention, sharp on ONE job ------
    # Hand-made Keys: each word owns one slot, so a Query with weight in slot 0 asks for
    # "tired" and a Query with weight in slot 2 asks for "mat". Hand-made Values: "tired"
    # carries content in slot 0 and "mat" in slot 1, so an answer reads as
    # [how much tired, how much mat].
    words = ["tired", "cat", "mat"]   # a toy sentence: "cat" and two words it cares about
    K_hand = np.eye(3)   # a Query/Key is 3 wide here, one slot per word; the lesson's d_k = 2 arrives in Part 3
    V_hand = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    # Two questions "cat" would like to ask. The 10 is match strength: a big number
    # makes the matching word win almost the whole share.
    q_describe = np.array([[10.0, 0.0, 0.0]])   # "which word DESCRIBES me?" -> tired's slot
    q_place = np.array([[0.0, 0.0, 10.0]])      # "WHERE did I happen?"      -> mat's slot

    w_describe, out_describe = attention(q_describe, K_hand, V_hand)
    w_place, out_place = attention(q_place, K_hand, V_hand)
    print("Part 1 — one head = the Day-3 formula, unchanged. words:", words)
    print("  describe-Query shares:", np.round(w_describe[0], 4), " row sum =", float(w_describe.sum()))
    print("  its blended answer   :", np.round(out_describe[0], 4), "-> nearly all 'tired': crisp")

    # --- Part 2: one head has to fold two jobs into ONE blend -------------
    # Adding the two questions asks the same head to track both relationships at
    # once. One head gives one blend, so it must average them.
    q_both = q_describe + q_place
    w_both, out_both = attention(q_both, K_hand, V_hand)
    print("\nPart 2 — the same head asked to do BOTH jobs at once")
    print("  both-Query shares    :", np.round(w_both[0], 4), "-> tired and mat get the SAME share")
    print("  its blended answer   :", np.round(out_both[0], 4), "-> half and half: muddy")
    print("  two heads instead    : head A", np.round(out_describe[0], 4), "| head B",
          np.round(out_place[0], 4), "-> each one stays sharp")

    # --- Part 3: build the two-head panel the lesson asks for -------------
    seq, d_model, h = 3, 4, 2       # 3 words in, full width 4, two heads
    d_k = d_model // h              # the lesson's rule: each head works in d_k = d_model / h
    # Predict the three widths from that rule, before running anything.
    predicted = [(seq, d_k), (seq, d_model), (seq, d_model)]

    rng = np.random.default_rng(0)             # fixed seed, so the run repeats exactly
    X = rng.standard_normal((seq, d_model))    # toy input: 3 words, width 4
    # Each head draws its OWN W_Q, W_K, W_V of shape (d_model, d_k) = (4, 2).
    heads = [(rng.standard_normal((d_model, d_k)),
              rng.standard_normal((d_model, d_k)),
              rng.standard_normal((d_model, d_k))) for _ in range(h)]
    W_O = rng.standard_normal((d_model, d_model))   # the output mixer: 4 in, 4 out

    head_outs, joined, final = multi_head(X, heads, W_O)
    print("\nPart 3 — split -> attend -> join -> mix, with d_model = 4 and h = 2")
    print("  predicted widths     :", predicted, "(one head, joined, final)")
    print("  input X", X.shape, "-> head 1", head_outs[0].shape, "+ head 2", head_outs[1].shape,
          "= half the model width each")
    print("  joined (concatenate) :", joined.shape, "= back to full width;   final (after W_O):",
          final.shape, "= one vector per word")
    print("  final values         :\n", np.round(final, 4))

    # Does W_O really MIX the heads? Replace head 2's answer with zeros and check
    # whether every slot of the final answer moves.
    joined_without_head2 = np.concatenate([head_outs[0], np.zeros_like(head_outs[1])], axis=-1)
    moved = np.abs(joined_without_head2 @ W_O - final)
    print("  smallest change in the final when head 2 is dropped:", round(float(moved.min()), 4))

    # --- Part 4: more heads, same budget ----------------------------------
    # Count the numbers stored in the panel's recipes, then in ONE full-width head,
    # whose three recipes would each be (d_model, d_model).
    params_panel = sum(w.size for head in heads for w in head)
    params_one_wide_head = 3 * d_model * d_model
    print("\nPart 4 — the panel's recipes hold", params_panel, "numbers; one full-width head would hold",
          params_one_wide_head, "-> same budget, sliced thinner")

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected number below is written down here, read off a real run of this
    # script, so a broken computation cannot quietly agree with itself.
    MUDDY_SHARE = 0.49922399    # one head doing two jobs: the share it gives each word
    SHARP_SHARE = 0.99382072    # a head with its own Query: the share it gives its own word
    TINY_SHARE = 0.00308964     # the share left over for a word that head is not asking about
    EXPECTED_FINAL = np.array([[-0.3143, -1.6484, 0.5226, 1.3313],
                               [-0.5656, 1.4477, 1.4395, -0.4749],
                               [-0.5218, -2.3046, -1.5854, 3.2423]])

    rows_sum_to_one = all(close(w.sum(), 1.0, 1e-12) for w in (w_describe, w_place, w_both))
    # The two shares come out of identical arithmetic, so ask for exact equality here.
    one_head_blurs = (float(w_both[0, 0]) == float(w_both[0, 2]) and close(w_both[0, 0], MUDDY_SHARE)
                      and close(out_both[0, 0], MUDDY_SHARE) and close(out_both[0, 1], MUDDY_SHARE))
    two_heads_stay_sharp = (close(w_describe[0, 0], SHARP_SHARE) and close(w_place[0, 2], SHARP_SHARE)
                            and close(w_describe[0, 2], TINY_SHARE))
    widths_as_predicted = ([head_outs[0].shape, joined.shape, final.shape] == predicted
                           and head_outs[1].shape == (3, 2) and joined.shape == (3, 4) and final.shape == (3, 4))
    # concatenate puts head 1 in the LEFT two columns, head 2 in the right two.
    join_layout_exact = np.array_equal(joined[:, :2], head_outs[0]) and np.array_equal(joined[:, 2:], head_outs[1])
    heads_differ = not np.allclose(head_outs[0], head_outs[1])   # own recipes -> own answers
    w_o_mixes_every_slot = bool(moved.min() > 1e-6)
    final_values_match = bool(np.array_equal(np.round(final, 4), EXPECTED_FINAL))
    budget_is_the_same = (params_panel == 48 and params_one_wide_head == 48)

    if all([rows_sum_to_one, one_head_blurs, two_heads_stay_sharp, widths_as_predicted, join_layout_exact,
            heads_differ, w_o_mixes_every_slot, final_values_match, budget_is_the_same]):
        print("\n✅ you got it — one head blurs to 0.4992 / 0.4992; two heads of width 2 stay sharp,"
              " join to width 4, and W_O hands back width 4")
    else:
        print("\n❌ not yet — expected the muddy share", MUDDY_SHARE, "the sharp share", SHARP_SHARE,
              "widths (3,2) / (3,4) / (3,4), head 1 in joined[:, :2], head 2 in joined[:, 2:], every final"
              " slot to move when a head is dropped, and 48 numbers in either layout")

    assert rows_sum_to_one, "softmax must give shares that add up to 1"
    assert one_head_blurs, "one head doing two jobs should give both words the same 0.4992 share"
    assert two_heads_stay_sharp, "a head with its own Query should give 0.9938 to its own word"
    assert widths_as_predicted, "expected (3,2) per head, (3,4) joined, (3,4) final"
    assert join_layout_exact, "joined[:, :2] must be head 1 and joined[:, 2:] must be head 2"
    assert heads_differ, "two heads with their own recipes must not return the same answer"
    assert w_o_mixes_every_slot, "dropping a head must change every slot of the final answer"
    assert final_values_match, "the final vectors, rounded to 4 places, must match the written-down ones"
    assert budget_is_the_same, "two heads of width 2 must hold as many numbers as one head of width 4"
