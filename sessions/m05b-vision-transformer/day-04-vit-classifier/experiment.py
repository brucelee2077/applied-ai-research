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
    # safe here only because these few logits are small. Subtracting the max or not gives
    # the same shares either way — that one difference is about overflow, nothing else.
    # The AXIS is a real difference, though. ONE ROW ONLY: a bare .sum() adds up
    # EVERYTHING, which is the same as Day 3's axis=-1 rule for a single row and NOT the
    # same for a block of rows. Part 4 prints both so you can see where they part company.
    return np.exp(scores) / np.exp(scores).sum()


if __name__ == "__main__":
    # --- Part 1: the [CLS] summary vector ---------------------------------
    # Stands in for the [CLS] token's FINAL vector, after it rode through every
    # block gathering evidence from all the tiles. Day 2's `cls` and Day 3's `cls_token`
    # are the same token at the OTHER end of the pipeline — the vector that goes in.
    # POSITIONS: no position stamp is added anywhere on this day, and none is missing.
    # Day 2's stamp is ADDED once, before block 1 (Day 3 does exactly that), so by the time
    # the blocks hand these vectors back the stamp is already baked into every row. A readout
    # that stamped them a second time would be adding position twice.
    # Every printed number below is BOUND to a name first and the self-check reads that
    # same name, so a wrong printed number breaks its own claim instead of sliding past.
    cls = rng.standard_normal(D)
    shown_cls_shape = cls.shape
    printed_cls = np.round(cls, 4)
    print("cls shape:", shown_cls_shape, " (one summary of the whole picture, length D =", D, ")")
    print("cls           :", printed_cls)

    # --- Part 2: LayerNorm — tidy the summary before the head -------------
    cls_tidied = layer_norm(cls)
    mean_before, std_before = round(float(cls.mean()), 4), round(float(cls.std()), 4)
    mean_after = round(float(cls_tidied.mean()), 4)
    std_after = round(float(cls_tidied.std()), 4)
    printed_tidied = np.round(cls_tidied, 4)
    print("\nbefore LayerNorm: mean =", mean_before,
          " spread(std) =", std_before, "(off-center, uneven scale)")
    print("cls_tidied    :", printed_tidied)
    print("after  LayerNorm: mean =", mean_after,
          " spread(std) =", std_after, "-> centered, scale 1")

    # --- Part 3: the linear head — one logit per class ---------------------
    W = rng.standard_normal((D, C))       # the head: D numbers in, C scores out
    # NAMING: Day 3 called this same head `W_head` and its output "class scores". Here the
    # lesson's bare `W` and the standard word `logits` are used — one object, two names.
    # Predict the output shape BEFORE looking at it, computed from the head's own
    # shape: the head's output length IS its number of columns, i.e. C.
    predicted_logits_shape = (W.shape[1],)
    logits = cls_tidied @ W               # raw scores, one per class
    shown_W_shape, shown_logits_shape = W.shape, logits.shape
    printed_logits = np.round(logits, 4)
    logits_sum = round(float(logits.sum()), 4)
    print("\nW shape:", shown_W_shape, " -> predicted logits shape:", predicted_logits_shape,
          " actual:", shown_logits_shape)
    print("logits        :", printed_logits, "  <- one RAW score per class")
    # Raw scores are not percentages — nothing makes them add up to 1.
    print("logits.sum()  :", logits_sum, "-> not 1.0, not readable as odds yet")

    # --- Part 4: softmax — raw scores become readable percentages ----------
    probs = softmax(logits)
    predicted_class = int(np.argmax(probs))
    ranked = np.sort(probs)
    printed_probs = np.round(probs, 4)
    shown_probs_shape = probs.shape
    probs_total = round(float(probs.sum()), 12)
    winner_share, runner_up_share = round(float(ranked[-1]), 4), round(float(ranked[-2]), 4)
    print("\nprobs         :", printed_probs, " shape:", shown_probs_shape,
          " sum:", probs_total, "-> one whole pizza")
    print("argmax(probs) :", predicted_class, "-> the predicted class index (the biggest slice),",
          "winning", winner_share, "vs runner-up", runner_up_share)
    # WHICH axis? Day 3 printed the rule: the softmax runs along axis -1, one row at a
    # time. On this ONE row the bare .sum() above agrees with that rule exactly — Part 6
    # shows where the two forms part company, on a whole block of rows.
    probs_axis = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    one_row_same = bool(np.array_equal(probs, probs_axis))
    print("bare .sum() == the axis=-1 form on this one row?", one_row_same,
          "-> same rule as Day 3, for a single row")

    # --- Part 5: cross-entropy — the nudge that teaches the verdict --------
    true_class = 0                        # pretend the correct answer is class 0
    # One named rule, used for the real loss AND for the round-number table below, so the
    # table exercises this code path instead of testing numpy's log on its own.
    cross_entropy = lambda p_true: float(-np.log(p_true))
    # The LABEL read, hoisted into ONE named step that every labelled case below goes
    # through. probs[0] happens to BE the smallest slice, so the first case on its own
    # cannot tell "read the labelled class" apart from "pick the lowest percentage" — but a
    # version that looked at the lowest slice instead of the label would break the SECOND
    # case a few lines down, where the labelled class is not extremal.
    loss_for_label = lambda c: cross_entropy(probs[c])
    loss = loss_for_label(true_class)         # penalty for the % given to the TRUE class
    true_prob = round(float(probs[true_class]), 4)
    shown_loss = round(float(loss), 4)
    loss_if_right = round(loss_for_label(predicted_class), 4)
    print("\ntrue_class =", true_class, " probs[true_class] =", true_prob,
          " loss = -log(probs[true_class]) =", shown_loss, "-> BIG: a tiny slice")
    print("had the truth been the model's pick (class", predicted_class, "), the loss would be",
          loss_if_right, "-> small")
    # Class 0 happens to hold the SMALLEST slice, so on its own it cannot show that the loss
    # reads the LABEL rather than just picking the lowest percentage. A second labelled case
    # settles it: class 2 is neither the biggest slice nor the smallest, so its loss can only
    # come from the label. The gap between the two numbers is the proof.
    second_true_class = 2
    second_prob = round(float(probs[second_true_class]), 4)
    second_loss = round(loss_for_label(second_true_class), 4)
    smallest_share = round(float(probs.min()), 4)
    loss_of_smallest = round(cross_entropy(float(probs.min())), 4)
    print("label it class", second_true_class, "instead (neither biggest nor smallest):",
          "probs =", second_prob, " loss =", second_loss)
    print("  the smallest slice is", smallest_share, "-> loss", loss_of_smallest,
          "· different number, so the loss follows the LABEL, not the lowest slice")
    # The same rule on round numbers, so the shape of the penalty is easy to read.
    # Built once, so the table that gets PRINTED is the table that gets CHECKED — and the
    # printed LINES are built once too, then checked as text.
    loss_table = [(p_true, cross_entropy(p_true)) for p_true in [0.10, 0.50, 0.90]]
    loss_lines = ["   gave the true class %.2f -> loss = %.2f" % (p, l) for p, l in loss_table]
    for line in loss_lines:
        print(line)

    # --- Part 6: the sequence it came from, and the GAP alternative --------
    # Rebuild the line of tokens the transformer handed back: [CLS] first, then
    # one token per tile. The readout keeps row 0 ONLY; rows 1.. are set aside.
    # NAMING: `tiles` here holds already-EMBEDDED token vectors, width D — the same
    # meaning as Day 3's `tiles`, NOT Day 2's flattened pixel rows of width P*P.
    # No position stamp is added when we restack them: these rows come back FROM the blocks,
    # and Day 2's stamp went on once before block 1 (see Part 1).
    tiles = rng.standard_normal((N_TILES, D))
    tokens = np.vstack([cls, tiles])
    closest_rows = min(np.abs(tokens[i] - tokens[j]).max()
                       for i in range(len(tokens)) for j in range(i + 1, len(tokens)))
    shown_tokens_shape = tokens.shape
    shown_closest_rows = round(float(closest_rows), 4)
    n_rows = len(tokens)
    print("\ntokens shape:", shown_tokens_shape, "([CLS] in row 0, then", N_TILES, "tile tokens);",
          "closest two rows differ by", shown_closest_rows, "-> WHICH row we read matters")
    logits_row0 = layer_norm(tokens[0]) @ W        # the readout we actually used
    logits_row1 = layer_norm(tokens[1]) @ W        # a tile token instead of [CLS]
    printed_row0, printed_row1 = np.round(logits_row0, 4), np.round(logits_row1, 4)
    row0_class, row1_class = int(np.argmax(logits_row0)), int(np.argmax(logits_row1))
    print("row 0 ([CLS]) logits:", printed_row0, "-> class", row0_class)
    # Day 3 showed the opposite of "it only saw one corner": from block 1 every token takes
    # a share of every other, so a tile token has seen the whole picture too. What makes it
    # the wrong row to read is its JOB, not its view — it is the summary OF ITS OWN TILE,
    # while [CLS] is the only row whose whole job is to speak for the picture.
    print("row 1 (a tile) logits:", printed_row1, "-> class", row1_class,
          "(a different verdict — that row speaks for its own tile, not the picture)")
    # And here is where the bare .sum() softmax stops matching Day 3's axis=-1 rule: on the
    # whole (5, 4) block of logits the bare form spreads ONE pizza over all 20 numbers, so
    # no row adds up to 1 any more. The axis=-1 form gives every row its own whole pizza.
    all_logits = layer_norm(tokens) @ W            # every row tidied on its own (axis=-1)
    bare_rows = np.round((np.exp(all_logits) / np.exp(all_logits).sum()).sum(axis=1), 4)
    axis_rows = np.round((np.exp(all_logits)
                          / np.exp(all_logits).sum(axis=-1, keepdims=True)).sum(axis=1), 4)
    print("softmax on the whole", all_logits.shape, "block — bare .sum() row totals:", bare_rows,
          "\n  the axis=-1 form's row totals:", axis_rows, "-> axis=-1 is the rule")
    # GAP: skip [CLS] entirely and average the TILE rows into one summary.
    gap = tiles.mean(axis=0)
    shown_gap_shape, printed_gap = gap.shape, np.round(gap, 4)
    gap_vs_all_rows = round(float(np.abs(gap - tokens.mean(axis=0)).max()), 4)
    print("gap shape:", shown_gap_shape, " gap:", printed_gap, "(average DOWN the tiles)")
    print("a mean over ALL", n_rows, "rows (wrongly including [CLS]) would sit up to",
          gap_vs_all_rows, "away")
    gap_probs = softmax(layer_norm(gap) @ W)       # same LayerNorm, same head, same softmax
    gap_class = int(np.argmax(gap_probs))
    printed_gap_probs = np.round(gap_probs, 4)
    gap_loss = round(cross_entropy(gap_probs[true_class]), 4)
    print("gap_probs     :", printed_gap_probs, " argmax:", gap_class, " loss for true_class",
          true_class, "=", gap_loss)
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
    exp_loss_lines = ["   gave the true class 0.10 -> loss = 2.30",
                      "   gave the true class 0.50 -> loss = 0.69",
                      "   gave the true class 0.90 -> loss = 0.11"]

    # Shapes, including the prediction that came from W's own column count.
    shapes_ok = (shown_cls_shape == (D,) and shown_W_shape == (D, C) and shown_gap_shape == (D,)
                 and shown_tokens_shape == (N_TILES + 1, D) and n_rows == 5
                 and shown_probs_shape == (C,)
                 and shown_logits_shape == predicted_logits_shape == (C,))
    cls_ok = np.array_equal(printed_cls, exp_cls)                    # the seeded input
    tidied_ok = (np.array_equal(printed_tidied, exp_tidied)
                 # the whole point of the tidy step, in the two printed numbers
                 and (mean_before, std_before) == (0.352, 0.5566)
                 and (mean_after, std_after) == (0.0, 1.0))
    logits_ok = np.array_equal(printed_logits, exp_logits) and logits_sum == 9.1324
    probs_ok = (np.array_equal(printed_probs, exp_probs)
                and (winner_share, runner_up_share) == (0.6035, 0.3825))
    # Softmax always sums to 1 by construction, so this one is a sanity check on
    # the arithmetic, not evidence that the head or the tidy step is right.
    probs_sum_one = abs(float(probs.sum()) - 1.0) < 1e-12 and probs_total == 1.0
    # The axis rule, agreed with Day 3: same answer on one row, different on a block.
    # Row 0 of the block readout matches the single-row readout to within float noise
    # (~1e-16): a 2-D mean adds the same eight numbers in a different order, so this one
    # clause uses a tolerance while every other number on the page is pinned exactly.
    axis_rule_ok = (one_row_same and np.array_equal(axis_rows, np.array([1.0] * 5))
                    and np.array_equal(bare_rows, np.array([0.6927, 0.1648, 0.0019, 0.0077, 0.1329]))
                    and not np.allclose(bare_rows, 1.0)
                    and np.allclose(all_logits[0], logits, rtol=0, atol=1e-12))
    winner_ok = predicted_class == 3                                 # argmax, not argmin
    loss_ok = (shown_loss == 5.5172 and true_prob == 0.004           # -log(0.00401708)
               and loss_if_right == 0.5051)
    # The second labelled case: class 2 is neither the biggest nor the smallest slice, and
    # its loss differs from the loss of the smallest slice — so the loss reads the LABEL.
    labelled_not_lowest_ok = (second_prob == 0.3825 and second_loss == 0.961
                              and smallest_share == 0.004 and loss_of_smallest == 5.5172
                              and second_loss != loss_of_smallest
                              and second_true_class not in (predicted_class,
                                                            int(np.argmin(probs))))
    loss_table_ok = ([(round(p, 2), round(l, 2)) for p, l in loss_table]
                     == [(0.10, 2.30), (0.50, 0.69), (0.90, 0.11)]
                     and loss_lines == exp_loss_lines)   # the printed LINES, as text
    readout_is_row0 = (np.array_equal(logits_row0, logits)           # row 0 IS the [CLS] summary
                       and np.array_equal(printed_row0, exp_logits) and row0_class == 3)
    rows_differ = shown_closest_rows == 1.646                        # so a wrong row would show
    other_row_ok = np.array_equal(printed_row1, exp_row1) and row1_class == 2
    gap_ok = np.array_equal(printed_gap, exp_gap) and gap_vs_all_rows == 0.2239  # TILE rows only
    gap_verdict_ok = (np.array_equal(printed_gap_probs, exp_gap_probs) and gap_class == 2
                      and gap_loss == 5.6564)

    if (shapes_ok and cls_ok and tidied_ok and logits_ok and probs_ok and probs_sum_one
            and axis_rule_ok and winner_ok and loss_ok and labelled_not_lowest_ok
            and loss_table_ok and readout_is_row0
            and rows_differ and other_row_ok and gap_ok and gap_verdict_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected cls_tidied", exp_tidied, "logits (4,)", exp_logits,
              "\n   probs", exp_probs, "summing to 1.0 -> argmax 3, loss 5.5172 for class 0 and",
              "0.961 for class 2, loss table [2.30 0.69 0.11],\n   row 0 reproducing the logits",
              "while row 1 gives", exp_row1, "-> class 2,\n   gap", exp_gap, "and gap_probs",
              exp_gap_probs, "-> class 2")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert shapes_ok, "cls (8,), W (8,4), tokens (5,8), gap (8,), and logits (4,) = one per class"
    assert cls_ok, "the seeded [CLS] summary should start [0.1257 -0.1321 0.6404 ...]"
    assert tidied_ok, "LayerNorm should turn mean 0.352 / std 0.5566 into 0.0 / 1.0: %s" % exp_tidied
    assert logits_ok, "logits should be %s and sum to 9.1324, not 1.0" % exp_logits
    assert probs_ok, "softmax should give %s, winner 0.6035 over runner-up 0.3825" % exp_probs
    assert probs_sum_one, "the percentages must add up to 1.0 — one whole pizza"
    assert axis_rule_ok, "softmax runs along axis -1: same on one row, row totals 1.0 on a block"
    assert winner_ok, "argmax(probs) should name class 3, the biggest slice"
    assert loss_ok, "-log(probs[0]) should be 5.5172 — a big penalty for a tiny slice"
    assert labelled_not_lowest_ok, "labelling class 2 (neither biggest nor smallest) costs 0.961, not the smallest slice's 5.5172"
    assert loss_table_ok, "true-class 0.10/0.50/0.90 should cost 2.30/0.69/0.11"
    assert readout_is_row0, "row 0 of the token sequence must be the [CLS] summary we read"
    assert rows_differ, "the closest two token rows should differ by 1.646"
    assert other_row_ok, "reading row 1 should give %s -> class 2" % exp_row1
    assert gap_ok, "GAP should average the TILE rows only: %s" % exp_gap
    assert gap_verdict_ok, "the GAP summary should give %s -> class 2, loss 5.6564" % exp_gap_probs
