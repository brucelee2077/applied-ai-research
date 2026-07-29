# day-04-vit-classifier — experiment
#
# Today's big idea in two lines of output:
#   A Vision Transformer's whole-picture verdict is read off ONE vector — the
#   [CLS] token's final summary; every tile token is set aside for the answer.
#
# The readout path: keep [CLS] -> LayerNorm -> linear head -> logits -> softmax
# -> the predicted class, plus the cross-entropy loss that taught it, and the
# GAP alternative that averages the tile tokens instead of using [CLS].
# Run it:  python3 sessions/m05b-vision-transformer/day-04-vit-classifier/experiment.py

import numpy as np  # numpy gives us arrays and matrix multiply (@)

# The lesson writes np.random.randn(...). We use a SEEDED generator instead, so the
# numbers pinned in the self-check at the bottom are the same on every run. No trained
# ViT and no image dataset live here, so these stand-ins play the part of "what came
# out of the last block".
rng = np.random.default_rng(0)

D = 8        # length of the summary vector (the lesson's D)
C = 4        # how many classes the head can choose from (the lesson's C). Day 3 used
             # C = 3: C is a free choice, unlike D = 8, which every day must share.
N_TILES = 4  # how many tile tokens rode through the blocks next to [CLS]


def layer_norm(v):
    # Tidy a vector: subtract its own mean (re-center), divide by its own spread
    # (re-scale). Same meaning, calmer numbers — the lesson's "plating a dish". A real
    # LayerNorm adds a tiny epsilon to the bottom in case the spread is zero; these
    # vectors are far from flat, so we leave it out to keep the numbers readable.
    # axis=-1 is the SAME rule as Day 3's layer_norm: one mean and one spread per ROW.
    # For the single vectors below it means exactly v.mean() / v.std(), and it also stays
    # correct if you ever hand it the whole (N+1, D) block of tokens.
    return (v - v.mean(axis=-1, keepdims=True)) / v.std(axis=-1, keepdims=True)


def softmax(scores):
    # Exactly the lesson's formula: stretch every score with exp (so all are
    # positive), then divide by the total, so the shares add up to 1. Day 3's softmax
    # subtracted the largest score first so exp() can never overflow; this plain form is
    # safe here only because these few logits are small. Same meaning either way.
    return np.exp(scores) / np.exp(scores).sum()


if __name__ == "__main__":
    # --- Part 1: the [CLS] summary vector ---------------------------------
    # Stands in for the [CLS] token's FINAL vector, after it rode through every
    # block gathering evidence from all the tiles.
    cls = rng.standard_normal(D)
    print("cls shape:", cls.shape, " (one summary of the whole picture, length D =", D, ")")
    print("cls           :", np.round(cls, 4))

    # --- Part 2: LayerNorm — tidy the summary before the head -------------
    cls_tidied = layer_norm(cls)
    print("\nbefore LayerNorm: mean =", round(float(cls.mean()), 4),
          " spread(std) =", round(float(cls.std()), 4), "(off-center, uneven scale)")
    print("cls_tidied    :", np.round(cls_tidied, 4))
    print("after  LayerNorm: mean =", round(float(cls_tidied.mean()), 4),
          " spread(std) =", round(float(cls_tidied.std()), 4), "-> centered, scale 1")

    # --- Part 3: the linear head — one logit per class ---------------------
    W = rng.standard_normal((D, C))       # the head: D numbers in, C scores out
    # Predict the output shape BEFORE looking at it, computed from the head's own
    # shape: the head's output length IS its number of columns, i.e. C.
    predicted_logits_shape = (W.shape[1],)
    logits = cls_tidied @ W               # raw scores, one per class
    print("\nW shape:", W.shape, " -> predicted logits shape:", predicted_logits_shape,
          " actual:", logits.shape)
    print("logits        :", np.round(logits, 4), "  <- one RAW score per class")
    # Raw scores are not percentages — nothing makes them add up to 1.
    print("logits.sum()  :", round(float(logits.sum()), 4), "-> not 1.0, not readable as odds yet")

    # --- Part 4: softmax — raw scores become readable percentages ----------
    probs = softmax(logits)
    predicted_class = int(np.argmax(probs))
    ranked = np.sort(probs)
    print("\nprobs         :", np.round(probs, 4), " shape:", probs.shape,
          " sum:", round(float(probs.sum()), 12), "-> one whole pizza")
    print("argmax(probs) :", predicted_class, "-> the predicted class index (the biggest slice),",
          "winning", round(float(ranked[-1]), 4), "vs runner-up", round(float(ranked[-2]), 4))

    # --- Part 5: cross-entropy — the nudge that teaches the verdict --------
    true_class = 0                        # pretend the correct answer is class 0
    # One named rule, used for the real loss AND for the round-number table below, so the
    # table exercises this code path instead of testing numpy's log on its own.
    cross_entropy = lambda p_true: float(-np.log(p_true))
    loss = cross_entropy(probs[true_class])   # penalty for the % given to the TRUE class
    print("\ntrue_class =", true_class, " probs[true_class] =", round(float(probs[true_class]), 4),
          " loss = -log(probs[true_class]) =", round(float(loss), 4), "-> BIG: a tiny slice")
    print("had the truth been the model's pick (class", predicted_class, "), the loss would be",
          round(cross_entropy(probs[predicted_class]), 4), "-> small")
    # The same rule on round numbers, so the shape of the penalty is easy to read.
    # Built once, so the table that gets PRINTED is the table that gets CHECKED.
    loss_table = [(p_true, cross_entropy(p_true)) for p_true in [0.10, 0.50, 0.90]]
    for p_true, p_loss in loss_table:
        print("   gave the true class %.2f -> loss = %.2f" % (p_true, p_loss))

    # --- Part 6: the sequence it came from, and the GAP alternative --------
    # Rebuild the line of tokens the transformer handed back: [CLS] first, then
    # one token per tile. The readout keeps row 0 ONLY; rows 1.. are set aside.
    tiles = rng.standard_normal((N_TILES, D))
    tokens = np.vstack([cls, tiles])
    closest_rows = min(np.abs(tokens[i] - tokens[j]).max()
                       for i in range(len(tokens)) for j in range(i + 1, len(tokens)))
    print("\ntokens shape:", tokens.shape, "([CLS] in row 0, then", N_TILES, "tile tokens);",
          "closest two rows differ by", round(float(closest_rows), 4), "-> WHICH row we read matters")
    logits_row0 = layer_norm(tokens[0]) @ W        # the readout we actually used
    logits_row1 = layer_norm(tokens[1]) @ W        # a tile token instead of [CLS]
    print("row 0 ([CLS]) logits:", np.round(logits_row0, 4), "-> class", int(np.argmax(logits_row0)))
    print("row 1 (a tile) logits:", np.round(logits_row1, 4), "-> class", int(np.argmax(logits_row1)),
          "(a different verdict — one tile only saw one corner)")
    # GAP: skip [CLS] entirely and average the TILE rows into one summary.
    gap = tiles.mean(axis=0)
    print("gap shape:", gap.shape, " gap:", np.round(gap, 4), "(average DOWN the tiles)")
    print("a mean over ALL", len(tokens), "rows (wrongly including [CLS]) would sit up to",
          round(float(np.abs(gap - tokens.mean(axis=0)).max()), 4), "away")
    gap_probs = softmax(layer_norm(gap) @ W)       # same LayerNorm, same head, same softmax
    gap_class = int(np.argmax(gap_probs))
    print("gap_probs     :", np.round(gap_probs, 4), " argmax:", gap_class, " loss for true_class",
          true_class, "=", round(cross_entropy(gap_probs[true_class]), 4))
    print("-> same head, different summary:", predicted_class, "([CLS]) vs", gap_class,
          "(GAP). Which readout you pick can change the verdict.")

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected number below was READ OFF a finished run and written down here,
    # not recomputed from the code above, so a broken step cannot agree with itself.
    exp_cls = np.array([0.1257, -0.1321, 0.6404, 0.1049, -0.5357, 0.3616, 1.304, 0.9471])
    exp_tidied = np.array([-0.4065, -0.8698, 0.5182, -0.4439, -1.5948, 0.0172, 1.7104, 1.0692])
    exp_logits = np.array([-0.3368, 0.5745, 4.2194, 4.6753])
    exp_probs = np.array([0.004, 0.01, 0.3825, 0.6035])
    exp_row1 = np.array([-1.6424, 2.1446, 2.9167, 2.7125])
    exp_gap = np.array([0.3254, 0.8166, 0.5257, 0.0579, 0.1142, -0.7577, 0.4903, 0.6358])
    exp_gap_probs = np.array([0.0035, 0.0133, 0.6311, 0.3521])

    # Shapes, including the prediction that came from W's own column count.
    shapes_ok = (cls.shape == (D,) and W.shape == (D, C) and gap.shape == (D,)
                 and tokens.shape == (N_TILES + 1, D)
                 and logits.shape == predicted_logits_shape == (C,))
    cls_ok = np.array_equal(np.round(cls, 4), exp_cls)              # the seeded input
    tidied_ok = np.array_equal(np.round(cls_tidied, 4), exp_tidied)
    logits_ok = np.array_equal(np.round(logits, 4), exp_logits)
    probs_ok = np.array_equal(np.round(probs, 4), exp_probs)
    # Softmax always sums to 1 by construction, so this one is a sanity check on
    # the arithmetic, not evidence that the head or the tidy step is right.
    probs_sum_one = abs(float(probs.sum()) - 1.0) < 1e-12
    winner_ok = predicted_class == 3                                 # argmax, not argmin
    loss_ok = round(float(loss), 4) == 5.5172                        # -log(0.00401708)
    loss_table_ok = ([(round(p, 2), round(l, 2)) for p, l in loss_table]
                     == [(0.10, 2.30), (0.50, 0.69), (0.90, 0.11)])
    readout_is_row0 = np.array_equal(logits_row0, logits)            # row 0 IS the [CLS] summary
    rows_differ = round(float(closest_rows), 4) == 1.646             # so a wrong row would show
    other_row_ok = (np.array_equal(np.round(logits_row1, 4), exp_row1)
                    and int(np.argmax(logits_row1)) == 2)
    gap_ok = np.array_equal(np.round(gap, 4), exp_gap)               # mean of the TILE rows only
    gap_verdict_ok = np.array_equal(np.round(gap_probs, 4), exp_gap_probs) and gap_class == 2

    if (shapes_ok and cls_ok and tidied_ok and logits_ok and probs_ok and probs_sum_one
            and winner_ok and loss_ok and loss_table_ok and readout_is_row0
            and rows_differ and other_row_ok and gap_ok and gap_verdict_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected cls_tidied", exp_tidied, "logits (4,)", exp_logits,
              "\n   probs", exp_probs, "summing to 1.0 -> argmax 3, loss 5.5172, loss table",
              "[2.30 0.69 0.11],\n   row 0 reproducing the logits while row 1 gives", exp_row1,
              "-> class 2,\n   gap", exp_gap, "and gap_probs", exp_gap_probs, "-> class 2")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert shapes_ok, "cls (8,), W (8,4), tokens (5,8), gap (8,), and logits (4,) = one per class"
    assert cls_ok, "the seeded [CLS] summary should start [0.1257 -0.1321 0.6404 ...]"
    assert tidied_ok, "LayerNorm should give %s" % exp_tidied
    assert logits_ok, "logits should be %s" % exp_logits
    assert probs_ok, "softmax should give %s" % exp_probs
    assert probs_sum_one, "the percentages must add up to 1.0 — one whole pizza"
    assert winner_ok, "argmax(probs) should name class 3, the biggest slice"
    assert loss_ok, "-log(probs[0]) should be 5.5172 — a big penalty for a tiny slice"
    assert loss_table_ok, "true-class 0.10/0.50/0.90 should cost 2.30/0.69/0.11"
    assert readout_is_row0, "row 0 of the token sequence must be the [CLS] summary we read"
    assert rows_differ, "the closest two token rows should differ by 1.646"
    assert other_row_ok, "reading row 1 should give %s -> class 2" % exp_row1
    assert gap_ok, "GAP should average the TILE rows only: %s" % exp_gap
    assert gap_verdict_ok, "the GAP summary should give %s -> class 2" % exp_gap_probs
