# day-08-clip-contrastive — experiment
#
# Today's big idea in two lines of output:
#   Trim every photo-arrow and caption-arrow to length 1 and one number — the cosine —
#   scores the match: in the batch grid the DIAGONAL (true pairs) wins every row.
#   Temperature stretches the cramped scores; the push is what stops collapse.
#
# Plain NumPy, no real model and nothing downloaded: the embeddings are the lesson's
# own hand-written numbers, so there is nothing random here and nothing to seed.
# Run it:  python3 sessions/m06-cnns-vision-encoders/day-08-clip-contrastive/experiment.py

import numpy as np  # numpy gives us arrays, matrix multiply (@) and vector lengths

def normalize(x):
    # L2 normalization: divide EVERY ROW by its own length, so each row becomes an
    # arrow of length 1. axis=1 means "across the numbers of one embedding".
    #
    # Reading across the module — two things this name does NOT mean here:
    #  · Days 4, 5 and 6 use the word "normalize" for a different operation: subtract a
    #    mean and divide by a spread (batch norm, the frozen train mean/std, day 6's
    #    "wrong normalization" gotcha). That one re-centres the numbers. This one only
    #    changes their LENGTH — the direction is untouched. Same word, two operations;
    #    day 6 already does THIS operation, spelled out in place as `E_unit` / `Q_unit`.
    #  · There is no eps in the divide, and day 4 taught eps AS the crumb that makes a
    #    divide-by-a-spread safe. A row of all zeros has length 0 and would come back
    #    nan here, exactly as day 4 demonstrated. It is safe in this file because no row
    #    is all zeros: the pinned row lengths ([0.9055 0.9566 0.9055] and the messy
    #    [1.6613 2.0833 1.3038]) are the standing check that the divisor is never 0.
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def softmax(z):
    # Row-wise: turn each ROW of scores into probabilities that add up to 1.
    z = z - z.max(axis=1, keepdims=True)   # subtract the row max so exp() cannot blow up
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

def contrastive_loss(logits):
    # One direction of the loss: -log(probability the row gives its OWN diagonal
    # cell), averaged over the batch. 0 means every row already picks its true pair.
    return float(-np.log(np.diag(softmax(logits))).mean())

def show_rows(labels, matrix, nd=4):
    # Argument 1 is a LIST of labels, one per row of the matrix. Day 5's printer takes a
    # single TITLE string for the whole array, so it is named `show_block(title, a)` and
    # this one `show_rows(labels, matrix)`: two contracts, two names. Sharing one name
    # would make a day-5-style call print just the first row and never complain.
    # Round ONCE, print that, and hand the rounded rows back, so the self-check can
    # verify the exact numbers that reached the screen instead of rounding a second time.
    shown = []
    for label, row in zip(labels, matrix):   # one labelled row per line, rounded
        rounded = np.round(row, nd)
        shown.append(rounded)
        print("  %-10s %s" % (label, rounded))
    return np.array(shown)

def nearer(value):
    # Is this score closer to 0 or to 1? Worked out FROM the score, never typed in,
    # so it can come out either way — and below it comes out both ways. A score sitting
    # exactly halfway is a tie, and the strict "<" hands ties to 0; Part 3 asks about
    # 0.5 on purpose so that choice is a printed, checked answer instead of an accident.
    return "1" if abs(value - 1.0) < abs(value - 0.0) else "0"

if __name__ == "__main__":
    # --- Part 1: fake embeddings for a tiny batch of 3 image-text pairs ----
    # Row i of img is photo i; row i of txt is THAT photo's own caption, so the
    # true pairs sit on the diagonal. Made up by hand — no encoder runs here.
    # Careful with the name `img`: on days 1, 2 and 5 it was a grid of PIXELS, so
    # `img.shape -> (3, 3)` read as a 3x3 picture. Here `img` is a batch of image
    # EMBEDDINGS — 3 rows, one per whole photo, 3 numbers each — which is why the shape
    # line below spells that out. Nothing in this file ever indexes a pixel.
    photos = ["dog", "beach", "car"]
    captions = ['"a photo of a dog"', '"a sunset over the ocean"', '"a red sports car"']
    img = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    txt = np.array([[0.9, 0.1, 0.], [0.05, 0.95, 0.1], [0., 0.1, 0.9]])
    shown_img_shape, shown_txt_shape = img.shape, txt.shape
    print("img shape:", shown_img_shape, " txt shape:", shown_txt_shape,
          "(3 pairs, 3 numbers each)")
    for i in range(3):
        shown_photo, shown_caption = img[i], txt[i]      # printed straight from the data
        print("  pair %d  photo %s  caption %s  %s"
              % (i, shown_photo, shown_caption, captions[i]))

    # --- Part 2: L2 normalize — same length, so only direction is left ----
    # Why this step comes FIRST, straight from day 6: day 6 ended with the rule "small L2
    # = high cosine ONLY for equal-length vectors", and showed cat vs 0.1*cat breaking it
    # (perfect cosine 1.0 at a big L2 of 0.9777). Trimming every row to length 1 is what
    # buys that condition: once all lengths are 1, the two rulers agree exactly
    # (l2^2 = 2 - 2*cosine), so one number — the cosine — can score the match.
    len_txt_before = np.linalg.norm(txt, axis=1)
    img_n, txt_n = normalize(img), normalize(txt)
    len_after = np.concatenate([np.linalg.norm(img_n, axis=1), np.linalg.norm(txt_n, axis=1)])
    shown_len_before, shown_len_after = np.round(len_txt_before, 4), np.round(len_after, 4)
    print("\ntxt row lengths BEFORE:", shown_len_before, "(not length 1)")
    print("all 6 row lengths AFTER:", shown_len_after, "(every row is on the unit sphere)")
    shown_txt_n = show_rows(["caption %d" % j for j in range(3)], txt_n)
    # Proof we divided by ROW length, not column length: a row-wise normalize leaves
    # the columns alone, so the column lengths are NOT 1.
    col_len_txt = np.linalg.norm(txt_n, axis=0)
    shown_col_len = np.round(col_len_txt, 4)
    print("column lengths of txt_n:", shown_col_len,
          "-> not 1: we normalized rows (each embedding), not columns")

    # --- Part 3: the similarity matrix — every photo x every caption ------
    # `S` here is the SIMILARITY grid (rows = photos, columns = captions). Days 1 to 3
    # drilled S as the STRIDE — the scalar in ⌊(N − K + 2P)/S⌋ + 1 saying how far the
    # stencil jumps. Both spellings are standard; they are simply different letters that
    # happen to look alike. No stride appears anywhere in this file.
    S = img_n @ txt_n.T          # rows are unit length, so this dot product IS cosine
    shown_S_shape = S.shape
    print("\nsimilarity matrix S = img_n @ txt_n.T -> shape", shown_S_shape,
          "(row = photo, column = caption)")
    shown_grid = show_rows(photos, S)
    # Each line's number and its verdict word are bound first, printed, then checked.
    cos_reads = []
    for j, note in [(0, "its OWN caption"), (2, "the car caption")]:
        shown_cos, shown_side = "%6.4f" % S[0, j], nearer(float(S[0, j]))
        cos_reads.append((shown_cos, shown_side))
        print("  cos(dog photo, %s) = %s -> closer to %s" % (note, shown_cos, shown_side))
    # Those two cells (0.9939 and 0.0000) never ask nearer() what it does with a TIE, so
    # ask it directly about a score sitting exactly halfway. The strict "<" must answer
    # "0"; loosening it to "<=" would answer "1" and this line would change.
    shown_half_side = nearer(0.5)
    print("  cos(anything, anything) = 0.5000 -> closer to %s: a tie is not a match"
          % shown_half_side)
    labels = np.arange(3)        # the true pairing: photo i belongs with caption i
    picks = S.argmax(axis=1)     # who each photo actually picks
    # Hide the diagonal (the positive pairs) and what is left in each row is that
    # photo's in-batch negatives — the free wrong answers from the same batch.
    off = np.where(np.eye(3, dtype=bool), -np.inf, S)
    margins = np.diag(S) - off.max(axis=1)   # true pair minus the best wrong one
    winner = "diagonal" if np.array_equal(picks, labels) else "off-diagonal"
    print("each photo's pick:", picks, " true pairing:", labels, "-> winner:", winner)
    shown_diag, shown_best_wrong = np.round(np.diag(S), 4), np.round(off.max(axis=1), 4)
    shown_margins = np.round(margins, 4)
    print("diagonal:", shown_diag, " best mismatch per row:", shown_best_wrong,
          " margin:", shown_margins)

    # --- Part 3b: the same pipeline, on photos that are not already tidy --
    # The three photos above are [1,0,0], [0,1,0] and [0,0,1] — what the lesson asks for,
    # but they are ALREADY length 1 and each points along its own axis, so normalizing
    # them changes nothing and S is really just txt_n turned on its side. The photo half
    # of the pipeline never shows its work. These three messy photos make it show: three
    # different lengths, no axis-aligned rows, the SAME normalize() and the same
    # @ txt_n.T — and the true pairs must still win every row.
    img_messy = np.array([[1.6, 0.4, 0.2], [0.3, 2.0, 0.5], [0.1, 0.5, 1.2]])
    img_messy_n = normalize(img_messy)
    shown_messy_before = np.round(np.linalg.norm(img_messy, axis=1), 4)
    shown_messy_after = np.round(np.linalg.norm(img_messy_n, axis=1), 4)
    S_messy = img_messy_n @ txt_n.T
    messy_picks = S_messy.argmax(axis=1)
    off_messy = np.where(np.eye(3, dtype=bool), -np.inf, S_messy)
    shown_messy_margins = np.round(np.diag(S_messy) - off_messy.max(axis=1), 4)
    shown_messy_winner = "diagonal" if np.array_equal(messy_picks, labels) else "off-diagonal"
    print("\nsame code, messy photos — photo row lengths BEFORE:", shown_messy_before,
          " AFTER:", shown_messy_after)
    shown_messy_grid = show_rows(photos, S_messy)
    print("  picks:", messy_picks, " margin:", shown_messy_margins, "-> winner:",
          shown_messy_winner, "(here normalizing the PHOTOS changes every score)")

    # --- Part 4: temperature — stretch the cramped [-1, 1] scores ---------
    temperature = 20.0                    # the lesson's value; real CLIP learns it
    probs_raw = softmax(S)                # scores used as they are: cramped
    probs_hot = softmax(S * temperature)  # stretched first, then softmax
    off_hot = np.where(np.eye(3, dtype=bool), 0.0, probs_hot)   # wrong pairs only
    shown_temp = "%g" % temperature
    shown_weak_raw = "%.4f" % np.diag(probs_raw).min()
    shown_weak_hot = "%.8f" % np.diag(probs_hot).min()
    shown_worst_wrong = "%.4e" % off_hot.max()
    print("\nrow probabilities with NO temperature:")
    shown_probs_raw = show_rows(photos, probs_raw)
    print("weakest true-pair probability: %s (no temperature) -> %s (x%s)"
          % (shown_weak_raw, shown_weak_hot, shown_temp))
    print("biggest wrong-pair probability after x%s: %s" % (shown_temp, shown_worst_wrong))
    # A row-wise softmax makes every ROW add to 1 and leaves the columns alone.
    row_sums, col_sums = probs_raw.sum(axis=1), probs_raw.sum(axis=0)
    shown_row_sums, shown_col_sums = np.round(row_sums, 6), np.round(col_sums, 6)
    print("row sums:", shown_row_sums, " column sums:", shown_col_sums,
          "-> rows add to 1, columns do not")

    # --- Part 5: the contrastive loss, and why the PUSH is needed ---------
    # Symmetric: photos picking captions (S) and captions picking photos (S.T).
    # Each direction on its own FIRST: the average of the pair is the same number if the
    # two sides get swapped, so only the average would hide a transpose inside the loss.
    loss_i2t, loss_t2i = contrastive_loss(S), contrastive_loss(S.T)
    loss_raw = (loss_i2t + loss_t2i) / 2
    loss_hot = (contrastive_loss(S * temperature) + contrastive_loss(S.T * temperature)) / 2
    shown_i2t, shown_t2i = "%.4f" % loss_i2t, "%.4f" % loss_t2i
    shown_loss_raw, shown_loss_hot = "%.4f" % loss_raw, "%.4e" % loss_hot
    print("\nphotos->captions %s and captions->photos %s — two different numbers"
          % (shown_i2t, shown_t2i))
    print("symmetric contrastive loss: %s (no temperature) -> %s (x%s)"
          % (shown_loss_raw, shown_loss_hot, shown_temp))
    # Negative control for the DIAGONAL. Shuffle the rows so no row's true pair is its
    # best cell any more. Here the loss must explode; a loss that read each row's biggest
    # score instead of its labelled cell would still print ~0 and look fine.
    S_mismatched = S[[1, 2, 0]]
    loss_mismatched = contrastive_loss(S_mismatched * temperature)
    shown_loss_mismatched = "%.4f" % loss_mismatched
    print("rows shuffled so every diagonal cell is a WRONG pair -> loss %s, not ~0"
          % shown_loss_mismatched)
    # The pull-only cheat: every photo AND every caption land on ONE single point.
    collapsed = normalize(np.ones((3, 3)))
    S_collapsed = collapsed @ collapsed.T
    loss_collapsed = (contrastive_loss(S_collapsed * temperature)
                      + contrastive_loss(S_collapsed.T * temperature)) / 2
    shown_collapsed_row = np.round(S_collapsed[0], 4)
    shown_collapsed_probs = np.round(softmax(S_collapsed * temperature)[0], 4)
    shown_loss_collapsed = "%.4f" % loss_collapsed
    print("collapsed space (every input on one point): S row 0 = %s  probabilities = %s"
          "\n-> a 1-in-3 guess, loss %s, and no temperature can rescue it"
          % (shown_collapsed_row, shown_collapsed_probs, shown_loss_collapsed))

    # --- Part 6: zero-shot — the label set is chosen at test time ---------
    scores_row0 = img_n[0] @ txt_n.T      # photo 0 against all 3 caption sentences
    pick0 = int(np.argmax(scores_row0))
    print("\nzero-shot for photo 0 (the dog photo):")
    zero_lines = []
    for j, caption in enumerate(captions):
        # The score AND the "picked" marker are built once, printed, then checked.
        shown_score = "%.4f" % scores_row0[j]
        shown_mark = "  <- picked" if j == pick0 else ""
        zero_lines.append(shown_score + shown_mark)
        print("  %-26s -> %s%s" % (caption, shown_score, shown_mark))
    shown_pred_caption = captions[pick0]
    print("argmax =", pick0, "-> predicted caption:", shown_pred_caption)
    order = [2, 0, 1]                     # same sentences, shuffled into new slots
    expected_slot = order.index(0)        # where caption 0 moved to — computed, not typed
    pick_shuffled = int(np.argmax(img_n[0] @ txt_n[order].T))
    print("captions re-ordered to", order, "-> argmax moves to", pick_shuffled,
          "= caption 0's new slot %d, so the label set is free to change at test time"
          % expected_slot)
    print("\nTakeaway: one shared space + cosine similarity scores the match; the diagonal"
          " holds the true pairs, temperature sharpens them, the push prevents collapse.")

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected number below was read off a real run and written down here, so a
    # broken change above cannot quietly agree with itself. Each claim also reads the
    # SAME name the print statement read, so a wrong printed number is a failing run
    # rather than a wrong number sitting under a green tick.
    shapes_ok = (shown_img_shape == (3, 3) and shown_txt_shape == (3, 3)
                 and shown_S_shape == (3, 3))
    # The two embedding tables are the ones the lesson asks for, exactly.
    inputs_are_spec_data = (np.array_equal(img, [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
                            and np.array_equal(txt, [[0.9, 0.1, 0.], [0.05, 0.95, 0.1],
                                                     [0., 0.1, 0.9]]))
    txt_was_not_unit = np.array_equal(shown_len_before, [0.9055, 0.9566, 0.9055])
    # round(..., 12) == 0.0 instead of "< tolerance": an exact pin cannot be loosened.
    rows_are_unit = (round(float(np.abs(len_after - 1).max()), 12) == 0.0
                     and np.array_equal(shown_len_after, [1., 1., 1., 1., 1., 1.]))
    unit_rows_printed = np.array_equal(shown_txt_n, [[0.9939, 0.1104, 0.0],
                                                     [0.0523, 0.9931, 0.1045],
                                                     [0.0, 0.1104, 0.9939]])
    normalized_by_row = np.array_equal(shown_col_len, [0.9953, 1.0054, 0.9994])
    grid_ok = np.array_equal(shown_grid, [[0.9939, 0.0523, 0.0],
                                          [0.1104, 0.9931, 0.1104],
                                          [0.0, 0.1045, 0.9939]])
    # The predicted answer to "closer to 0 or to 1?" — and the two cells must disagree,
    # which is what shows nearer() is reading the score and not returning a fixed word.
    # The halfway score pins the tie rule, which those two cells cannot reach.
    cosine_reads_direction = (cos_reads == [("0.9939", "1"), ("0.0000", "0")]
                              and shown_half_side == "0")
    diagonal_wins = (winner == "diagonal" and picks.tolist() == [0, 1, 2]
                     and labels.tolist() == [0, 1, 2]
                     and np.array_equal(shown_diag, [0.9939, 0.9931, 0.9939])
                     and np.array_equal(shown_best_wrong, [0.0523, 0.1104, 0.1045])
                     and np.array_equal(shown_margins, [0.9416, 0.8827, 0.8893])
                     and round(float(margins.min()), 4) == 0.8827)
    # The messy-photo batch: photo rows that are NOT already unit length, so skipping or
    # breaking the photo-side normalize changes these scores instead of hiding in an
    # identity matrix. The diagonal still has to win.
    messy_photos_ok = (np.array_equal(shown_messy_before, [1.6613, 2.0833, 1.3038])
                       and np.array_equal(shown_messy_after, [1., 1., 1.])
                       and np.array_equal(shown_messy_grid, [[0.9838, 0.302, 0.1462],
                                                             [0.2491, 0.9861, 0.3446],
                                                             [0.1186, 0.4811, 0.9571]])
                       and messy_picks.tolist() == [0, 1, 2]
                       and shown_messy_winner == "diagonal"
                       and np.array_equal(shown_messy_margins, [0.6817, 0.6415, 0.476]))
    softmax_by_row = (round(float(np.abs(row_sums - 1).max()), 12) == 0.0
                      and round(float(np.abs(col_sums - 1).max()), 6) == 0.002331
                      and np.array_equal(shown_row_sums, [1., 1., 1.])
                      and np.array_equal(shown_col_sums, [1.002331, 0.999543, 0.998126]))
    temp_sharpens = (np.array_equal(shown_probs_raw, [[0.5681, 0.2216, 0.2103],
                                                      [0.2264, 0.5473, 0.2264],
                                                      [0.2078, 0.2307, 0.5615]])
                     and shown_weak_raw == "0.5473" and shown_weak_hot == "0.99999996"
                     and shown_worst_wrong == "2.1520e-08" and shown_temp == "20")
    loss_drops = shown_loss_raw == "0.5819" and shown_loss_hot == "2.4349e-08"
    # The two directions are pinned apart, so swapping the sides inside the loss is caught.
    directions_differ = shown_i2t == "0.5818" and shown_t2i == "0.5819"
    # ...and with the true pairs off the diagonal the loss must be huge, which is what
    # shows it reads the labelled cell rather than each row's largest score.
    reads_the_diagonal = shown_loss_mismatched == "18.4396"
    collapse_is_chance = (shown_loss_collapsed == "1.0986"
                          and np.array_equal(shown_collapsed_row, [1., 1., 1.])
                          and np.array_equal(shown_collapsed_probs, [0.3333, 0.3333, 0.3333])
                          and np.array_equal(np.round(S_collapsed, 4), np.ones((3, 3))))
    # expected_slot != 0 keeps the re-order from being a no-op that proves nothing.
    # The scores are pinned too, not only the winner: with these three captions the
    # winner alone would still look right if the two sides were fed in swapped.
    zero_shot_ok = (pick0 == 0 and expected_slot == 1 and pick_shuffled == expected_slot
                    and shown_pred_caption == '"a photo of a dog"'
                    and zero_lines == ["0.9939  <- picked", "0.0523", "0.0000"]
                    and np.array_equal(np.round(scores_row0, 4), [0.9939, 0.0523, 0.0]))

    if all([shapes_ok, inputs_are_spec_data, txt_was_not_unit, rows_are_unit,
            unit_rows_printed, cosine_reads_direction, normalized_by_row, grid_ok,
            diagonal_wins, messy_photos_ok, softmax_by_row,
            temp_sharpens, loss_drops, directions_differ, reads_the_diagonal,
            collapse_is_chance, zero_shot_ok]):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected txt row lengths [0.9055 0.9566 0.9055] before and 1 after, "
              "txt_n column lengths [0.9953 1.0054 0.9994], S = [[0.9939 0.0523 0] [0.1104 0.9931 "
              "0.1104] [0 0.1045 0.9939]] with the diagonal winning every row by 0.8827 at the "
              "closest, the messy photos starting at lengths [1.6613 2.0833 1.3038] and still "
              "picking [0 1 2], the weakest true pair 0.5473 -> wrong pairs 2.1520e-08 with "
              "temperature, loss 0.5819 -> 2.4349e-08, a collapsed space stuck at 1.0986, and "
              "photo 0 picking caption 0 (and its caption's new slot after the re-order)")

    assert shapes_ok, "img, txt and S should all be (3, 3)"
    assert inputs_are_spec_data, "the embeddings must be the lesson's img and txt tables"
    assert txt_was_not_unit, "txt rows should start at lengths [0.9055 0.9566 0.9055]"
    assert rows_are_unit, "after normalize, all 6 row lengths must be 1"
    assert unit_rows_printed, ("txt_n rows should print as [.9939 .1104 0] / "
                               "[.0523 .9931 .1045] / [0 .1104 .9939]")
    assert normalized_by_row, "txt_n column lengths should be [0.9953 1.0054 0.9994]"
    assert grid_ok, "S should be [[.9939 .0523 0] [.1104 .9931 .1104] [0 .1045 .9939]]"
    assert cosine_reads_direction, ("the true pair should read as nearer 1, the mismatch "
                                    "as nearer 0, and a 0.5 tie as 0")
    assert diagonal_wins, "every row's biggest cell must be its diagonal, by 0.8827 at the closest"
    assert messy_photos_ok, ("the messy photos should start at lengths "
                             "[1.6613 2.0833 1.3038] and still pick [0 1 2]")
    assert softmax_by_row, "softmax must normalize rows (columns should be off by 0.002331)"
    assert temp_sharpens, "0.5473 without temperature; wrong pairs 2.1520e-08 with it"
    assert loss_drops, "symmetric loss should be 0.5819 raw and 2.4349e-08 with temperature"
    assert directions_differ, "the two directions should be 0.5818 and 0.5819, not one number"
    assert reads_the_diagonal, "with the true pairs off the diagonal the loss must be 18.4396"
    assert collapse_is_chance, "a collapsed space should sit at 1.0986 = a 1-in-3 guess"
    assert zero_shot_ok, "photo 0 should score [0.9939 0.0523 0] and follow caption 0's slot"
