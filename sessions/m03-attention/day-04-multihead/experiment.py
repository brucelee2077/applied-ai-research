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

# ---- Yesterday's attention, same formula, respelled with an explicit axis ---
# Two things are worth saying out loud, because a reused helper hides them.
#  * UNMASKED. Day 3's pipeline had five beats and the fifth was the mask. This helper
#    has no mask argument: a 3-word toy sentence has nothing to hide, so every day in
#    this module runs the short form softmax(Q @ Kᵀ / √d_k) @ V. The full form is
#    softmax(Q @ Kᵀ / √d_k + mask) @ V, and the mask returns in the decoder module.
#  * QUERY-FIRST. The score grid is Q @ K.T, so ROW = the word asking and COLUMN = the
#    word offered — the same layout Days 2, 3 and 5 use. Key-first is the transpose.

def softmax(scores):
    # Turn each ROW of scores into shares: all positive, adding up to 1. The axis is the
    # last one, so every asking word keeps its own whole budget of 1.
    # Subtract the row's biggest score first so exp() can never overflow.
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
    # One head is exactly the Day-3 formula: softmax(Q @ Kᵀ / √d_k) @ V.
    d_k = Q.shape[-1]                      # the width of one Query / Key
    # scaled_scores, not "scores": the sqrt(d_k) division has already happened here, so
    # this grid is Day 3's `scaled_scores`, not its raw `raw_scores`.
    scaled_scores = Q @ K.T / np.sqrt(d_k)  # one match number per pair of words, scaled
    weights = softmax(scaled_scores)       # each row of shares adds up to 1
    return weights, weights @ V            # the shares, and the blended answer

def run_head(x, W_Q, W_K, W_V):
    # A head owns three recipes. They turn every full-width word into a thin
    # d_k-wide Query and Key, and a d_v-wide Value. Day 2 established that d_v is a
    # FREE choice, and it still is — but here we spend that freedom on purpose:
    # setting d_v = d_k is what makes h head answers join back to exactly d_model,
    # which is today's whole width story. A real transformer spends it the same way.
    return attention(x @ W_Q, x @ W_K, x @ W_V)

def split_width(d_model, h):
    # The lesson's rule, in one place so it can be run at more than one dial setting:
    # h heads share the model width, so each head works in d_k = d_model / h.
    return d_model // h

def multi_head(x, heads, W_O):
    # heads is a list of (W_Q, W_K, W_V) — one own set of recipes per head.
    runs = [run_head(x, w_q, w_k, w_v) for (w_q, w_k, w_v) in heads]
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
    d_k = split_width(d_model, h)     # the lesson's rule: each head works in d_k = d_model / h
    CAT = 1                           # "cat" is the word doing the asking — row 1 of x
    STRENGTH = 10.0                   # how loudly a Query asks; a big number makes its word win the share

    # The four slots of a word vector, hand-picked so we can read them. Slot numbering is
    # 1-BASED, the way Days 1 and 3 count: slot 1 is the FIRST column (index 0).
    #   slot 1 = "I am a describing word"   slot 3 = "I want to know who describes me"
    #   slot 2 = "I am a place word"        slot 4 = "I want to know where I happened"
    x = np.array([[1.0, 0.0, 0.0, 0.0],    # tired — a describing word
                  [0.0, 0.0, 1.0, 1.0],    # cat   — asks BOTH questions
                  [0.0, 1.0, 0.0, 0.0]])   # mat   — a place word

    # One head on the FULL width, so its three recipes are (d_model, d_model) = (4, 4).
    # W_K copies slot 1 into key-slot 1 and slot 2 into key-slot 2: "tired" answers the
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

    w_describe, out_describe = run_head(x, W_Q_describe, W_K_wide, W_V_wide)
    print("Part 1 — one FULL-width head (d_k = 4), asked ONE question. words:", words)
    print("  x", x.shape, "-> its Q, K, V are each", (x @ W_Q_describe).shape, "— full width, nothing sliced yet")
    print("  'cat' row shares     :", row(w_describe[CAT]), " row sum =", float(w_describe[CAT].sum()))
    # QUERY-FIRST, measured rather than assumed: cat is the word ASKING, so its sharp share
    # belongs to cat's ROW. Read the same grid key-first and the sharpness would have to sit
    # in cat's COLUMN, which holds nothing but the flat 1/3 of the two words that ask nothing.
    cat_row_top = float(w_describe[CAT].max())
    cat_column_top = float(w_describe[:, CAT].max())
    print("  row  vs  column      : cat's ROW peaks at", round(cat_row_top, 4),
          "but cat's COLUMN peaks at", round(cat_column_top, 4), "-> row = the asker")
    print("  its blended answer   :", row(out_describe[CAT]),
          "\n                         reads [how much tired, how much mat, 0, 0] -> nearly all 'tired': crisp")

    # --- Part 2: the SAME head asked both questions at once ---------------
    # Add the two Query recipes and this one head has to track both relationships.
    # One head hands back one blend, so it has to average them.
    W_Q_both = W_Q_describe + W_Q_place
    w_both, out_both = run_head(x, W_Q_both, W_K_wide, W_V_wide)
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
    # d_v is READ OFF W_V, not assumed: Day 2's free choice is spent here, deliberately, by
    # picking d_v = d_k. That is what makes h * d_v land back on d_model after the join.
    # Day 3 spent it the other way (d_v 2 against d_k 4) and had no join to land.
    d_v = W_V_slice.shape[-1]
    heads = [(W_Q_A, W_K_A, W_V_slice), (W_Q_B, W_K_B, W_V_slice)]
    # W_O, the mixer: every column takes something from head A's two slots AND from head B's
    # two slots, so every slot of the final answer is built out of both heads.
    W_O = np.array([[1.00, 0.50, 0.30, 0.20],
                    [0.40, 1.00, 0.60, 0.10],
                    [0.20, 0.30, 1.00, 0.70],
                    [0.60, 0.10, 0.40, 1.00]])

    head_weights, head_outs, joined, final = multi_head(x, heads, W_O)
    print("\nPart 3 — split -> attend -> join -> mix, with d_model = 4 and h = 2")
    print("  predicted widths     :", predicted, "(one head, joined, final)")
    print("  each head's recipes  : W_Q", W_Q_A.shape, "W_K", W_K_A.shape, "W_V", W_V_slice.shape,
          "-> half the model width")
    print("  d_k =", d_k, "and d_v =", d_v, "-> a Value's width is still free (Day 3 used 2 against 4);",
          "\n                         we CHOOSE d_v = d_k here so h x d_v =", h * d_v, "= d_model on the join")
    print("  head A 'cat' shares  :", row(head_weights[0][CAT]), "-> nearly all 'tired'")
    print("  head A 'cat' answer  :", row(head_outs[0][CAT]), "= [how much tired, how much mat]")
    print("  head B 'cat' shares  :", row(head_weights[1][CAT]), "-> nearly all 'mat'")
    print("  head B 'cat' answer  :", row(head_outs[1][CAT]), "= same two slots, the other word")
    print("  two heads of width", d_k, "each stay sharp where ONE head of width", d_model, "went half and half")
    print("  input x", x.shape, "-> head A", head_outs[0].shape, "+ head B", head_outs[1].shape,
          "= half the model width each")
    print("  joined (concatenate) :", joined.shape, "= back to full width;   final (after W_O):",
          final.shape, "= one vector per word")
    final_rounded = np.round(final, 4)   # ONE copy, printed below and pinned in the self-check
    final_block = str(final_rounded)     # the exact block of text the learner reads
    print("  final values         :\n", final_block)
    print("  note: only 'cat' asks a question in this toy, so the 'tired' and 'mat' rows have no",
          "\n        Query to speak of and spread their shares evenly.")

    # --- Part 3b: the same split rule at three dial settings ---------------
    # At (d_model, h) = (4, 2) the rule d_model // h cannot be told apart from d_model - h or
    # from plain h — all three give 2. So run the SAME rule at two more settings and build a
    # real panel at each, so the widths themselves have to agree, not just the arithmetic.
    dial_widths = []
    for dial_d_model, dial_h in ((d_model, h), (12, 3), (64, 8)):
        dial_d_k = split_width(dial_d_model, dial_h)
        dial_rng = np.random.default_rng(1)         # fixed seed: widths only, so any numbers do
        x_dial = dial_rng.standard_normal((seq, dial_d_model))
        dial_heads = [(dial_rng.standard_normal((dial_d_model, dial_d_k)),
                       dial_rng.standard_normal((dial_d_model, dial_d_k)),
                       dial_rng.standard_normal((dial_d_model, dial_d_k))) for _ in range(dial_h)]
        # W_O takes the JOINED width in and hands d_model back out. Shaping it that way means a
        # wrong d_k shows up as a joined width that no longer equals d_model — a number we can
        # print and pin — instead of a matmul that blows up mid-run.
        W_O_dial = dial_rng.standard_normal((dial_h * dial_d_k, dial_d_model))
        _, dial_outs, dial_joined, dial_final = multi_head(x_dial, dial_heads, W_O_dial)
        dial_widths.append((dial_d_model, dial_h, dial_d_k,
                            dial_outs[0].shape[-1], dial_joined.shape[-1], dial_final.shape[-1]))
    print("\nPart 3b — run the same rule at three dial settings, panel and all")
    for dial_row in dial_widths:
        print("  d_model %2d, h %d -> d_k %d;  one head %2d wide, joined %2d, final %2d" % dial_row)

    # --- Part 3c: a probe word that asks only ONE of the two questions -----
    # In the sentence above, "cat" is [0, 0, 1, 1] — it asks both questions equally, so W_Q_A
    # (which reads slot 3) and W_Q_B (which reads slot 4) hand both heads the SAME Query, and
    # tying the two Query recipes together would change nothing. This second, lopsided input
    # asks the PLACE question only: now head A has nothing to look for and goes flat, while
    # head B stays on "mat". That is what makes each head's OWN Query recipe observable.
    x_probe = np.array([[1.0, 0.0, 0.0, 0.0],    # tired — unchanged
                        [0.0, 0.0, 0.0, 1.0],    # asks the PLACE question only
                        [0.0, 1.0, 0.0, 0.0]])   # mat   — unchanged
    probe_weights, probe_outs, _, _ = multi_head(x_probe, heads, W_O)
    print("\nPart 3c — same two heads, a word asking ONLY the place question")
    print("  head A 'cat' shares  :", row(probe_weights[0][CAT]), "-> flat: head A's Query is silent here")
    print("  head B 'cat' shares  :", row(probe_weights[1][CAT]), "-> still nearly all 'mat'")

    # Contrast control for the KEY recipes: hand head B head A's W_K and the two heads collapse
    # onto one answer, which is the observable behind "each head looks for something different".
    _, tied_key_outs, _, _ = multi_head(
        x, [(W_Q_A, W_K_A, W_V_slice), (W_Q_B, W_K_A, W_V_slice)], W_O)
    print("  tie head B's KEY recipe to head A's and both heads answer", row(tied_key_outs[1][CAT]),
          "-> identical")

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
    # BOTH corners must be zeroed for "each head writes only its own slots" to be true, so
    # check both — zeroing only one of them used to pass unnoticed.
    corners_both_zeroed = (not W_O_no_mixing[:a_cols, a_cols:].any()
                           and not W_O_no_mixing[a_cols:, :a_cols].any()
                           and W_O[:a_cols, a_cols:].any() and W_O[a_cols:, :a_cols].any())
    # Each printed number below is computed ONCE here and then reused by the self-check. The
    # headline sentence is built as a STRING first and that string is pinned too, so a number
    # recomputed on the print line — the one way print and claim could still drift — is caught.
    smallest_mixing_move = float(moved_mixing[:, :a_cols].min())
    biggest_no_mixing_move = float(moved_no_mixing[:, :a_cols].max())
    smallest_own_slot_move = float(moved_no_mixing[:, a_cols:].min())
    mixing_line = ("  mixing W_O    : the first %d final slots move by at least %s"
                   " -> head B had a say in them" % (a_cols, round(smallest_mixing_move, 4)))
    print("\nPart 4 — drop head B, then compare the mixing W_O with a no-mixing one")
    print(mixing_line)
    print("  no-mixing W_O : the same slots move by", round(biggest_no_mixing_move, 4),
          "-> they never heard head B at all")
    print("                  its own last", a_cols, "slots still move by at least",
          round(smallest_own_slot_move, 4))
    print("  each of those first", a_cols, "slots, word by word:\n",
          np.round(moved_mixing[:, :a_cols], 5))

    # --- Part 5: more heads, same budget ----------------------------------
    # "Budget" here means STORED NUMBERS — the lesson's "same budget, sliced thinner". It is
    # NOT Day 3's budget, which was the 1.0 of attention shares a row hands out. Two words,
    # two quantities: count the numbers stored in the panel's recipes, then in the ONE
    # full-width head from Part 1, whose three recipes are each (d_model, d_model).
    params_panel = sum(w.size for head in heads for w in head)
    params_one_wide_head = W_Q_both.size + W_K_wide.size + W_V_wide.size
    print("\nPart 5 — more heads, same budget. 'Budget' = stored numbers here, not Day 3's",
          "\n         budget of attention shares. The panel's Q/K/V recipes hold", params_panel,
          "numbers; the one full-width head holds",
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
    rnd_weights, rnd_outs, rnd_joined, rnd_final = multi_head(x, random_heads, W_O_random)
    biggest_random_share = max(float(w[CAT].max()) for w in rnd_weights)
    panel_top_share = float(head_weights[0][CAT].max())   # printed below AND pinned, one copy
    print("\nPart 6 — the same panel again, now with two random sets of recipes (seed 0)")
    print("  widths               : head", rnd_outs[0].shape, "+ head", rnd_outs[1].shape, "-> joined",
          rnd_joined.shape, "-> final", rnd_final.shape, "= the same width story")
    print("  biggest 'cat' share any random head gives:", round(biggest_random_share, 4),
          "-> nowhere near the", round(panel_top_share, 4), "above: untuned heads do not specialize")

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
    TWO_BIG_SHARES = 0.99664234    # tired + mat in the muddy row — short of 1, the rest is on "cat"
    FLAT_SHARE = 1.0 / 3.0         # a head with nothing to look for spreads 3 words evenly
    SUM_TOL = 1e-12                # how close a row of shares has to sit to exactly 1
    EXPECTED_FINAL = np.array([[0.7333, 0.6333, 0.7667, 0.6667],
                               [1.5978, 0.6001, 0.7002, 1.1986],
                               [0.7333, 0.6333, 0.7667, 0.6667]])
    # The whole block of moves, not just its smallest entry: a min over a block says nothing
    # about WHICH slot moved, so transposing moved_mixing would leave the min untouched.
    EXPECTED_MOVED_MIXING_A = np.array([[0.26667, 0.13333],
                                        [0.59915, 0.10008],
                                        [0.26667, 0.13333]])
    # d_k = d_model // h, run at three settings so the two rules that also give 2 at (4, 2) —
    # d_model - h and plain h — disagree at (12, 3) and (64, 8).
    EXPECTED_DIAL_WIDTHS = [(4, 2, 2, 2, 4, 4), (12, 3, 4, 4, 12, 12), (64, 8, 8, 8, 64, 64)]

    rows_sum_to_one = (all(close(r.sum(), 1.0, SUM_TOL)
                           for w in (w_describe, w_both, head_weights[0], head_weights[1],
                                     probe_weights[0], probe_weights[1]) for r in w)
                       # A row adds up to 1 only when EVERY word is counted. Tired and mat alone
                       # come to 0.9966 — the leftover 0.0034 sits on "cat" — so SUM_TOL has to
                       # be tight enough to tell that sum apart from 1.
                       and close(w_both[CAT, 0] + w_both[CAT, 2], TWO_BIG_SHARES)
                       and not close(w_both[CAT, 0] + w_both[CAT, 2], 1.0, SUM_TOL))
    # The two shares come out of identical arithmetic, so ask for exact equality here.
    one_head_blurs = (float(w_both[CAT, 0]) == float(w_both[CAT, 2]) and close(w_both[CAT, 0], MUDDY_SHARE)
                      and close(out_both[CAT, 0], MUDDY_SHARE) and close(out_both[CAT, 1], MUDDY_SHARE))
    one_head_can_be_sharp = (close(w_describe[CAT, 0], ONE_JOB_SHARE)      # the head is not broken —
                             and close(out_describe[CAT, 0], ONE_JOB_SHARE))  # one blend is all it has
    recipes_are_half_width = all(w.shape == (d_model, d_k) for head in heads for w in head)
    # d_v is a CHOICE, so state both halves of it: it equals d_k here, and that equality is
    # what makes the join land on d_model. Day 3's d_v was 2 against a d_k of 4, so this is
    # not a rule the module discovered — it is a knob this day sets on purpose.
    value_width_is_chosen = (d_v == d_k and h * d_v == d_model
                             and W_V_slice.shape == (d_model, d_v)
                             and joined.shape[-1] == h * d_v)
    # The score grid is QUERY-first, so the asker's sharpness lives in its ROW. If it were
    # read key-first, cat's sharp share would have to appear in cat's COLUMN, which holds
    # only the flat 1/3 of the two words that ask nothing.
    grid_is_query_first = (close(cat_row_top, ONE_JOB_SHARE)
                           and close(cat_column_top, FLAT_SHARE)
                           and not close(cat_column_top, ONE_JOB_SHARE))
    split_rule_holds_at_every_dial = (dial_widths == EXPECTED_DIAL_WIDTHS
                                      and d_k == split_width(d_model, h))
    # NOTE on what actually separates these two heads. "cat" is the row [0, 0, 1, 1]: it asks
    # BOTH questions equally, so slot 3 and slot 4 hold the same 1.0, and W_Q_A (which reads
    # slot 3) and W_Q_B (which reads slot 4) therefore hand both heads the SAME Query here.
    # That is not a bug — it is the whole reason one head muddles: cat's question is genuinely
    # two questions at once. On THIS input it is the KEY recipe that separates the heads, so the
    # claim is stated as something observed, not as a comparison of two typed-in matrices: the
    # two heads read different Key tables out of x, land on different words, and collapse onto
    # one answer the moment head B is handed head A's Key recipe.
    heads_own_their_own_keys = (not np.allclose(x @ W_K_A, x @ W_K_B)
                                and int(np.argmax(head_weights[0][CAT])) == 0    # head A -> tired
                                and int(np.argmax(head_weights[1][CAT])) == 2    # head B -> mat
                                and np.allclose(tied_key_outs[0], tied_key_outs[1])
                                and not np.allclose(head_outs[0], head_outs[1]))
    # And the Query recipes are separated by the lopsided probe input, which the sentence above
    # cannot do: asked the place question only, head A has nothing to look for and goes flat at
    # 1/3, while head B still holds 0.9983 on "mat". Tying the two Query recipes flattens both.
    heads_ask_their_own_questions = (all(close(share, FLAT_SHARE) for share in probe_weights[0][CAT])
                                     and close(probe_weights[1][CAT, 2], PANEL_SHARP)
                                     and close(probe_weights[1][CAT, 0], PANEL_TINY)
                                     and close(probe_outs[1][CAT, 1], PANEL_SHARP))
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
    w_o_reaches_across_heads = (close(smallest_mixing_move, MOVE_WITH_MIXING)
                                and moved_mixing.shape == (seq, d_model)
                                and np.allclose(moved_mixing[:, :a_cols], EXPECTED_MOVED_MIXING_A, atol=1e-5)
                                and biggest_no_mixing_move == 0.0
                                and np.array_equal(moved_no_mixing[:, :a_cols], np.zeros((seq, a_cols)))
                                and close(smallest_own_slot_move, MOVE_OF_OWN_SLOTS))
    final_values_match = (bool(np.array_equal(final_rounded, EXPECTED_FINAL))
                          and final_block == str(EXPECTED_FINAL))
    # The one sentence Part 4 is about, pinned as text: whatever number reaches the page, this is
    # the number that has to be on it.
    printed_claim_matches = (mixing_line == "  mixing W_O    : the first 2 final slots move by at"
                                           " least 0.1001 -> head B had a say in them")
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
                                  and close(panel_top_share, PANEL_SHARP)
                                  and biggest_random_share < PANEL_SHARP)

    if all([rows_sum_to_one, one_head_blurs, one_head_can_be_sharp, recipes_are_half_width,
            value_width_is_chosen, grid_is_query_first,
            split_rule_holds_at_every_dial, panel_heads_stay_sharp, heads_own_their_own_keys,
            heads_ask_their_own_questions, corners_both_zeroed,
            widths_as_predicted, join_layout_exact, heads_differ,
            w_o_reaches_across_heads, final_values_match, printed_claim_matches, budget_is_the_same,
            tolerance_is_tight_enough, random_panel_widths_match, random_heads_are_not_sharp]):
        print("\n✅ you got it — one full-width head asked both questions blurs to 0.4983 / 0.4983;"
              " two heads with their OWN width-2 recipes each hold 0.9983 on their own word"
              " (here it is their KEY recipe that separates them — cat asks both questions"
              " equally, so both heads get the same Query), they join to width 4, and W_O"
              " hands back width 4")
    else:
        print("\n❌ not yet — expected the muddy share", MUDDY_SHARE, "the panel's sharp share", PANEL_SHARP,
              "widths (3,2) / (3,4) / (3,4), head A in joined[:, :2] and head B in joined[:, 2:], the first two"
              " final slots to move by", MOVE_WITH_MIXING, "when head B is dropped under the mixing W_O but not"
              " move at all under the no-mixing one, 48 numbers in either layout, a tolerance tight enough to"
              " tell those shares apart, and the random panel to reach the same widths with a top share of only",
              RANDOM_TOP_SHARE, "\n   — also d_k = d_model // h at every dial setting", EXPECTED_DIAL_WIDTHS,
              "and, on the place-only probe, head A flat at 1/3 while head B holds", PANEL_SHARP)

    assert rows_sum_to_one, ("softmax must give shares that add up to 1, with a tolerance tight enough to see"
                             " that tired + mat alone come to 0.9966 and not 1")
    assert one_head_blurs, "one full-width head doing two jobs should give both words the same 0.4983 share"
    assert one_head_can_be_sharp, "that same head asked ONE question should give 0.9867 to its word"
    assert recipes_are_half_width, ("each head's W_Q and W_K must be (4, 2) — half the model width —"
                                    " and W_V is the same width by today's deliberate choice of d_v")
    assert value_width_is_chosen, ("d_v is a free choice Day 2 established and Day 3 used (2 against 4);"
                                   " here we set d_v = d_k = 2 so h x d_v = 4 = d_model on the join")
    assert grid_is_query_first, ("the grid is QUERY-first, so cat's sharp 0.9867 must sit in cat's ROW"
                                 " while cat's COLUMN peaks at only the flat 1/3")
    assert split_rule_holds_at_every_dial, ("d_k must be d_model // h at (4,2), (12,3) and (64,8) — at (4,2)"
                                            " alone, d_model - h and h itself also give 2")
    assert panel_heads_stay_sharp, "each width-2 head should give 0.9983 to its own word"
    assert heads_own_their_own_keys, ("the two heads must look for DIFFERENT things — on this input it is"
                                      " W_K that separates them, so tying the keys must collapse them")
    assert heads_ask_their_own_questions, ("on the place-only probe head A must go flat at 1/3 while head B"
                                           " still holds 0.9983 on 'mat' — that is what separates the Queries")
    assert corners_both_zeroed, ("the no-mixing W_O must have BOTH cross-head corners zeroed, and the real"
                                 " W_O must have both non-zero")
    assert widths_as_predicted, "expected (3,2) per head, (3,4) joined, (3,4) final"
    assert join_layout_exact, "joined[:, :2] must be head A and joined[:, 2:] must be head B"
    assert heads_differ, "two heads with their own recipes must not return the same answer"
    assert w_o_reaches_across_heads, ("dropping head B must move the first two final slots by the whole"
                                      " written-down block (smallest 0.1001) under the mixing W_O, and not"
                                      " at all under the no-mixing one")
    assert final_values_match, "the final vectors, rounded to 4 places, must match the written-down ones"
    assert printed_claim_matches, ("the Part-4 sentence must reach the page reading 'move by at least 0.1001'"
                                   " — the same number the self-check pins")
    assert budget_is_the_same, "two heads of width 2 must hold as many numbers as one head of width 4"
    assert tolerance_is_tight_enough, "close() must still tell 0.9983, 0.9867, 0.4983 and 0.0008 apart"
    assert random_panel_widths_match, "the random panel must reach (3,2) per head, (3,4) joined, (3,4) final too"
    assert random_heads_are_not_sharp, "with random recipes no head should get near the hand-typed 0.9983"
