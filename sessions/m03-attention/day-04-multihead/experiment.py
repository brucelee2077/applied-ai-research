# day-04-multihead — experiment
#
# Today's big idea in two lines of output:
#   Ask ONE full-width head to track two relationships and it splits the difference — both land near 0.50, muddy.
#   Give each relationship its OWN head on half the width, then join and mix with W_O — sharp, and width 4 comes home.
#
# The lesson's Produce step: with d_model = 4 and h = 2 heads, predict (a) each head's answer
# width, (b) the width after joining, (c) the width after W_O. This script builds that two-head
# panel by hand and prints every width on the way.
#
# One thing to know before you read the numbers: every recipe here is HAND-TYPED, not learned.
# A real transformer learns W_Q, W_K, W_V and W_O from data. Typing them by hand is what lets
# you see WHY each head lands on the word it lands on, instead of taking it on faith.
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
    runs = [run_head(X, w_q, w_k, w_v) for (w_q, w_k, w_v) in heads]
    weights = [w for (w, _) in runs]         # each head's own shares
    outs = [o for (_, o) in runs]            # each head's own answer
    joined = np.concatenate(outs, axis=-1)   # the staple: answers side by side
    return weights, outs, joined, joined @ W_O   # W_O is the editor that mixes them

def close(value, expected, tol=1e-6):
    # True when a computed number lands on the number we wrote down.
    return abs(float(value) - float(expected)) < tol

def row(values):
    # Print a row of shares in plain decimals, so a tiny share stays readable.
    return "[" + " ".join("%8.4f" % v for v in values) + "]"


if __name__ == "__main__":
    # --- Part 1: one FULL-width head, asked ONE question ------------------
    words = ["tired", "cat", "mat"]   # a toy sentence: "cat" and two words it cares about
    seq, d_model, h = 3, 4, 2         # 3 words in, full width 4, two heads
    d_k = d_model // h                # the lesson's rule: each head works in d_k = d_model / h
    CAT = 1                           # "cat" is the word doing the asking — row 1 of X
    STRENGTH = 10.0                   # how loudly a Query asks; a big number makes its word win the share

    # The four slots of a word vector, hand-picked so we can read them:
    #   slot 0 = "I am a describing word"   slot 2 = "I want to know who describes me"
    #   slot 1 = "I am a place word"        slot 3 = "I want to know where I happened"
    X = np.array([[1.0, 0.0, 0.0, 0.0],    # tired — a describing word
                  [0.0, 0.0, 1.0, 1.0],    # cat   — asks BOTH questions
                  [0.0, 1.0, 0.0, 0.0]])   # mat   — a place word

    # One head on the FULL width, so its three recipes are (d_model, d_model) = (4, 4).
    # W_K copies slot 0 into key-slot 0 and slot 1 into key-slot 1: "tired" answers the
    # describe question, "mat" answers the place question, "cat" answers neither.
    W_K_wide = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0]])
    W_V_wide = W_K_wide.copy()   # so an answer reads [how much tired, how much mat, 0, 0]
    # Two Query recipes for that one head. Each turns "cat" into one question.
    W_Q_describe = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0],
                             [STRENGTH, 0.0, 0.0, 0.0],   # "who describes me?" -> tired's key slot
                             [0.0, 0.0, 0.0, 0.0]])
    W_Q_place = np.array([[0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0],
                          [0.0, STRENGTH, 0.0, 0.0]])      # "where did I happen?" -> mat's key slot

    w_describe, out_describe = run_head(X, W_Q_describe, W_K_wide, W_V_wide)
    print("Part 1 — one FULL-width head (d_k = 4), asked ONE question. words:", words)
    print("  X", X.shape, "-> its Q, K, V are each", (X @ W_Q_describe).shape, "— full width, nothing sliced yet")
    print("  'cat' row shares     :", row(w_describe[CAT]), " row sum =", float(w_describe[CAT].sum()))
    print("  its blended answer   :", row(out_describe[CAT]),
          "\n                         reads [how much tired, how much mat, 0, 0] -> nearly all 'tired': crisp")

    # --- Part 2: the SAME head asked both questions at once ---------------
    # Add the two Query recipes and this one head has to track both relationships.
    # One head hands back one blend, so it has to average them.
    W_Q_both = W_Q_describe + W_Q_place
    w_both, out_both = run_head(X, W_Q_both, W_K_wide, W_V_wide)
    print("\nPart 2 — the same full-width head asked to do BOTH jobs at once")
    print("  'cat' row shares     :", row(w_both[CAT]), "-> tired and mat get the SAME share")
    print("  its blended answer   :", row(out_both[CAT]), "-> half and half: muddy")

    # --- Part 3: build the two-head panel the lesson asks for -------------
    # Predict the three widths from d_k = d_model / h, before running anything.
    predicted = [(seq, d_k), (seq, d_model), (seq, d_model)]

    # Each head owns its OWN W_Q, W_K, W_V, and each is (d_model, d_k) = (4, 2) — half width.
    # Head A asks the describe question; head B asks the place question.
    W_Q_A = np.array([[0.0, 0.0], [0.0, 0.0], [STRENGTH, 0.0], [0.0, 0.0]])
    W_K_A = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])   # only a describing word answers head A
    W_Q_B = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [STRENGTH, 0.0]])
    W_K_B = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])   # only a place word answers head B
    # Both heads read Values the same way, so their two answers are easy to compare side by
    # side: each answer reads [how much tired, how much mat]. Their Query and Key recipes are
    # what differ, and that alone sends the two heads to two different words.
    W_V_slice = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    heads = [(W_Q_A, W_K_A, W_V_slice), (W_Q_B, W_K_B, W_V_slice)]
    # W_O, the mixer: every column takes something from head A's two slots AND from head B's
    # two slots, so every slot of the final answer is built out of both heads.
    W_O = np.array([[1.00, 0.50, 0.30, 0.20],
                    [0.40, 1.00, 0.60, 0.10],
                    [0.20, 0.30, 1.00, 0.70],
                    [0.60, 0.10, 0.40, 1.00]])

    head_weights, head_outs, joined, final = multi_head(X, heads, W_O)
    print("\nPart 3 — split -> attend -> join -> mix, with d_model = 4 and h = 2")
    print("  predicted widths     :", predicted, "(one head, joined, final)")
    print("  each head's recipes  : W_Q", W_Q_A.shape, "W_K", W_K_A.shape, "W_V", W_V_slice.shape,
          "-> half the model width")
    print("  head A 'cat' shares  :", row(head_weights[0][CAT]), "-> nearly all 'tired'")
    print("  head A 'cat' answer  :", row(head_outs[0][CAT]), "= [how much tired, how much mat]")
    print("  head B 'cat' shares  :", row(head_weights[1][CAT]), "-> nearly all 'mat'")
    print("  head B 'cat' answer  :", row(head_outs[1][CAT]), "= same two slots, the other word")
    print("  two heads of width", d_k, "each stay sharp where ONE head of width", d_model, "went half and half")
    print("  input X", X.shape, "-> head A", head_outs[0].shape, "+ head B", head_outs[1].shape,
          "= half the model width each")
    print("  joined (concatenate) :", joined.shape, "= back to full width;   final (after W_O):",
          final.shape, "= one vector per word")
    print("  final values         :\n", np.round(final, 4))
    print("  note: only 'cat' asks a question in this toy, so the 'tired' and 'mat' rows have no",
          "\n        Query to speak of and spread their shares evenly.")

    # --- Part 4: does W_O really let the heads mix? -----------------------
    # Replace head B's answer with zeros, then rebuild the final vector two ways. Once with
    # the mixing W_O above. Once with the SAME matrix with its two cross-head corners set to
    # zero, so each head can only write into its own two output slots. If W_O truly mixes,
    # losing head B has to move head A's slots as well.
    a_cols = head_outs[0].shape[-1]    # how many columns of `joined` head A actually filled
    W_O_no_mixing = W_O.copy()
    W_O_no_mixing[:a_cols, a_cols:] = 0.0    # head A can no longer reach head B's slots
    W_O_no_mixing[a_cols:, :a_cols] = 0.0    # and head B can no longer reach head A's
    joined_without_B = np.concatenate([head_outs[0], np.zeros_like(head_outs[1])], axis=-1)
    final_no_mixing = joined @ W_O_no_mixing
    moved_mixing = np.abs(joined_without_B @ W_O - final)
    moved_no_mixing = np.abs(joined_without_B @ W_O_no_mixing - final_no_mixing)
    print("\nPart 4 — drop head B, then compare the mixing W_O with a no-mixing one")
    print("  mixing W_O    : the first", a_cols, "final slots move by at least",
          round(float(moved_mixing[:, :a_cols].min()), 4), "-> head B had a say in them")
    print("  no-mixing W_O : the same slots move by", round(float(moved_no_mixing[:, :a_cols].max()), 4),
          "-> they never heard head B at all")
    print("                  its own last", a_cols, "slots still move by at least",
          round(float(moved_no_mixing[:, a_cols:].min()), 4))

    # --- Part 5: more heads, same budget ----------------------------------
    # Count the numbers stored in the panel's recipes, then in the ONE full-width head from
    # Part 1, whose three recipes are each (d_model, d_model).
    params_panel = sum(w.size for head in heads for w in head)
    params_one_wide_head = W_Q_both.size + W_K_wide.size + W_V_wide.size
    print("\nPart 5 — the panel's Q/K/V recipes hold", params_panel, "numbers; the one full-width head holds",
          params_one_wide_head, "-> same budget, sliced thinner")
    print("  (that counts Q/K/V only. W_O adds", W_O.size, "more numbers, and a full-width head needs",
          "\n   an output matrix of its own in a real model too.)")

    # --- Part 6: the same panel with RANDOM recipes -----------------------
    # The lesson's Produce step asks for two DIFFERENT RANDOM sets of W_Q, W_K, W_V. The width
    # story is exactly the same — that is what the three guesses were about. What random
    # recipes do NOT give you is a sharp head: nothing has tuned them to look for anything, so
    # their shares stay much closer to an even split. Sharpness is what training buys, and
    # hand-typing the recipes above is how this script stands in for training.
    rng = np.random.default_rng(0)      # fixed seed, so this run repeats exactly
    random_heads = [(rng.standard_normal((d_model, d_k)),
                     rng.standard_normal((d_model, d_k)),
                     rng.standard_normal((d_model, d_k))) for _ in range(h)]
    W_O_random = rng.standard_normal((d_model, d_model))
    rnd_weights, rnd_outs, rnd_joined, rnd_final = multi_head(X, random_heads, W_O_random)
    biggest_random_share = max(float(w[CAT].max()) for w in rnd_weights)
    print("\nPart 6 — the same panel again, now with two random sets of recipes (seed 0)")
    print("  widths               : head", rnd_outs[0].shape, "+ head", rnd_outs[1].shape, "-> joined",
          rnd_joined.shape, "-> final", rnd_final.shape, "= the same width story")
    print("  biggest 'cat' share any random head gives:", round(biggest_random_share, 4),
          "-> nowhere near the", round(float(head_weights[0][CAT].max()), 4), "above: untuned heads do not specialize")

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected number below is written down here, read off a real run of this
    # script, so a broken computation cannot quietly agree with itself.
    MUDDY_SHARE = 0.49832117      # one FULL-width head doing two jobs: the share each word gets
    ONE_JOB_SHARE = 0.98670329    # that same head asked ONE question: the share its word gets
    PANEL_SHARP = 0.99830423      # a width-2 head with its OWN Query: the share its own word gets
    PANEL_TINY = 0.00084789       # the share left for a word that head is not asking about
    MOVE_WITH_MIXING = 0.10008479 # smallest move in the first two final slots when head B is dropped
    MOVE_OF_OWN_SLOTS = 0.40016958  # smallest move in head B's own slots under the no-mixing W_O
    RANDOM_TOP_SHARE = 0.73198896  # the biggest 'cat' share either RANDOM head manages
    EXPECTED_FINAL = np.array([[0.7333, 0.6333, 0.7667, 0.6667],
                               [1.5978, 0.6001, 0.7002, 1.1986],
                               [0.7333, 0.6333, 0.7667, 0.6667]])

    rows_sum_to_one = all(close(r.sum(), 1.0, 1e-12)
                          for w in (w_describe, w_both, head_weights[0], head_weights[1]) for r in w)
    # The two shares come out of identical arithmetic, so ask for exact equality here.
    one_head_blurs = (float(w_both[CAT, 0]) == float(w_both[CAT, 2]) and close(w_both[CAT, 0], MUDDY_SHARE)
                      and close(out_both[CAT, 0], MUDDY_SHARE) and close(out_both[CAT, 1], MUDDY_SHARE))
    one_head_can_be_sharp = (close(w_describe[CAT, 0], ONE_JOB_SHARE)      # the head is not broken —
                             and close(out_describe[CAT, 0], ONE_JOB_SHARE))  # one blend is all it has
    recipes_are_half_width = all(w.shape == (d_model, d_k) for head in heads for w in head)
    panel_heads_stay_sharp = (close(head_weights[0][CAT, 0], PANEL_SHARP)   # head A -> tired
                              and close(head_weights[1][CAT, 2], PANEL_SHARP)  # head B -> mat
                              and close(head_weights[0][CAT, 2], PANEL_TINY)   # head A barely looks at mat
                              and close(head_weights[1][CAT, 0], PANEL_TINY)   # head B barely looks at tired
                              and close(head_outs[0][CAT, 0], PANEL_SHARP)
                              and close(head_outs[1][CAT, 1], PANEL_SHARP)
                              and close(head_outs[0][CAT, 1], PANEL_TINY)
                              and close(head_outs[1][CAT, 0], PANEL_TINY))
    widths_as_predicted = ([head_outs[0].shape, joined.shape, final.shape] == predicted
                           and head_outs[1].shape == (3, 2) and joined.shape == (3, 4) and final.shape == (3, 4))
    # concatenate puts head A in the LEFT two columns, head B in the right two.
    join_layout_exact = np.array_equal(joined[:, :2], head_outs[0]) and np.array_equal(joined[:, 2:], head_outs[1])
    heads_differ = not np.allclose(head_outs[0], head_outs[1])   # own recipes -> own answers
    # Mixing is what W_O adds, so measure it against a mixer that cannot mix: with the
    # cross-head corners zeroed, dropping head B must leave head A's slots bit-for-bit alone.
    w_o_reaches_across_heads = (close(moved_mixing[:, :a_cols].min(), MOVE_WITH_MIXING)
                                and np.array_equal(moved_no_mixing[:, :a_cols], np.zeros((seq, a_cols)))
                                and close(moved_no_mixing[:, a_cols:].min(), MOVE_OF_OWN_SLOTS))
    final_values_match = bool(np.array_equal(np.round(final, 4), EXPECTED_FINAL))
    budget_is_the_same = (params_panel == 48 and params_one_wide_head == 48)
    # A tolerance is only worth something if it can still tell our own numbers apart. The
    # panel's sharp share (0.9983), the full-width head's one-job share (0.9867), the muddy
    # share (0.4983) and the tiny leftover share are four different numbers — close() has to
    # say so, or every check above would pass on the wrong answer.
    tolerance_is_tight_enough = (not close(PANEL_SHARP, ONE_JOB_SHARE)
                                 and not close(ONE_JOB_SHARE, MUDDY_SHARE)
                                 and not close(PANEL_TINY, 0.0))
    # The random panel: same widths, same join layout — and no head anywhere near sharp.
    random_panel_widths_match = ([rnd_outs[0].shape, rnd_joined.shape, rnd_final.shape] == predicted
                                 and np.array_equal(rnd_joined[:, :2], rnd_outs[0])
                                 and np.array_equal(rnd_joined[:, 2:], rnd_outs[1]))
    random_heads_are_not_sharp = (close(biggest_random_share, RANDOM_TOP_SHARE)
                                  and biggest_random_share < PANEL_SHARP)

    if all([rows_sum_to_one, one_head_blurs, one_head_can_be_sharp, recipes_are_half_width,
            panel_heads_stay_sharp, widths_as_predicted, join_layout_exact, heads_differ,
            w_o_reaches_across_heads, final_values_match, budget_is_the_same,
            tolerance_is_tight_enough, random_panel_widths_match, random_heads_are_not_sharp]):
        print("\n✅ you got it — one full-width head asked both questions blurs to 0.4983 / 0.4983;"
              " two heads with their OWN width-2 recipes each hold 0.9983 on their own word,"
              " join to width 4, and W_O hands back width 4")
    else:
        print("\n❌ not yet — expected the muddy share", MUDDY_SHARE, "the panel's sharp share", PANEL_SHARP,
              "widths (3,2) / (3,4) / (3,4), head A in joined[:, :2] and head B in joined[:, 2:], the first two"
              " final slots to move by", MOVE_WITH_MIXING, "when head B is dropped under the mixing W_O but not"
              " move at all under the no-mixing one, 48 numbers in either layout, a tolerance tight enough to"
              " tell those shares apart, and the random panel to reach the same widths with a top share of only",
              RANDOM_TOP_SHARE)

    assert rows_sum_to_one, "softmax must give shares that add up to 1"
    assert one_head_blurs, "one full-width head doing two jobs should give both words the same 0.4983 share"
    assert one_head_can_be_sharp, "that same head asked ONE question should give 0.9867 to its word"
    assert recipes_are_half_width, "each head's W_Q, W_K, W_V must be (4, 2) — half the model width"
    assert panel_heads_stay_sharp, "each width-2 head with its own Query should give 0.9983 to its own word"
    assert widths_as_predicted, "expected (3,2) per head, (3,4) joined, (3,4) final"
    assert join_layout_exact, "joined[:, :2] must be head A and joined[:, 2:] must be head B"
    assert heads_differ, "two heads with their own recipes must not return the same answer"
    assert w_o_reaches_across_heads, ("dropping head B must move the first two final slots by 0.1001 under the"
                                      " mixing W_O, and not at all under the no-mixing one")
    assert final_values_match, "the final vectors, rounded to 4 places, must match the written-down ones"
    assert budget_is_the_same, "two heads of width 2 must hold as many numbers as one head of width 4"
    assert tolerance_is_tight_enough, "close() must still tell 0.9983, 0.9867, 0.4983 and 0.0008 apart"
