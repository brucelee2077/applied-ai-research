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
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def softmax(z):
    # Row-wise: turn each ROW of scores into probabilities that add up to 1.
    z = z - z.max(axis=1, keepdims=True)   # subtract the row max so exp() cannot blow up
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

def contrastive_loss(logits):
    # One direction of the loss: -log(probability the row gives its OWN diagonal
    # cell), averaged over the batch. 0 means every row already picks its true pair.
    return float(-np.log(np.diag(softmax(logits))).mean())

def show(labels, matrix, nd=4):
    for label, row in zip(labels, matrix):   # one labelled row per line, rounded
        print("  %-10s %s" % (label, np.round(row, nd)))

def nearer(value):
    # Is this score closer to 0 or to 1? Worked out FROM the score, never typed in,
    # so it can come out either way — and below it comes out both ways.
    return "1" if abs(value - 1.0) < abs(value - 0.0) else "0"

if __name__ == "__main__":
    # --- Part 1: fake embeddings for a tiny batch of 3 image-text pairs ----
    # Row i of img is photo i; row i of txt is THAT photo's own caption, so the
    # true pairs sit on the diagonal. Made up by hand — no encoder runs here.
    photos = ["dog", "beach", "car"]
    captions = ['"a photo of a dog"', '"a sunset over the ocean"', '"a red sports car"']
    img = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    txt = np.array([[0.9, 0.1, 0.], [0.05, 0.95, 0.1], [0., 0.1, 0.9]])
    print("img shape:", img.shape, " txt shape:", txt.shape, "(3 pairs, 3 numbers each)")
    for i in range(3):
        print("  pair %d  photo %s  caption %s  %s" % (i, img[i], txt[i], captions[i]))

    # --- Part 2: L2 normalize — same length, so only direction is left ----
    len_txt_before = np.linalg.norm(txt, axis=1)
    img_n, txt_n = normalize(img), normalize(txt)
    len_after = np.concatenate([np.linalg.norm(img_n, axis=1), np.linalg.norm(txt_n, axis=1)])
    print("\ntxt row lengths BEFORE:", np.round(len_txt_before, 4), "(not length 1)")
    print("all 6 row lengths AFTER:", np.round(len_after, 4), "(every row is on the unit sphere)")
    show(["caption %d" % j for j in range(3)], txt_n)
    # Proof we divided by ROW length, not column length: a row-wise normalize leaves
    # the columns alone, so the column lengths are NOT 1.
    col_len_txt = np.linalg.norm(txt_n, axis=0)
    print("column lengths of txt_n:", np.round(col_len_txt, 4),
          "-> not 1: we normalized rows (each embedding), not columns")

    # --- Part 3: the similarity matrix — every photo x every caption ------
    S = img_n @ txt_n.T          # rows are unit length, so this dot product IS cosine
    print("\nsimilarity matrix S = img_n @ txt_n.T -> shape", S.shape,
          "(row = photo, column = caption)")
    show(photos, S)
    for j, note in [(0, "its OWN caption"), (2, "the car caption")]:
        print("  cos(dog photo, %s) = %6.4f -> closer to %s"
              % (note, S[0, j], nearer(float(S[0, j]))))
    labels = np.arange(3)        # the true pairing: photo i belongs with caption i
    picks = S.argmax(axis=1)     # who each photo actually picks
    # Hide the diagonal (the positive pairs) and what is left in each row is that
    # photo's in-batch negatives — the free wrong answers from the same batch.
    off = np.where(np.eye(3, dtype=bool), -np.inf, S)
    margins = np.diag(S) - off.max(axis=1)   # true pair minus the best wrong one
    winner = "diagonal" if np.array_equal(picks, labels) else "off-diagonal"
    print("each photo's pick:", picks, " true pairing:", labels, "-> winner:", winner)
    print("diagonal:", np.round(np.diag(S), 4), " best mismatch per row:",
          np.round(off.max(axis=1), 4), " margin:", np.round(margins, 4))

    # --- Part 4: temperature — stretch the cramped [-1, 1] scores ---------
    temperature = 20.0                    # the lesson's value; real CLIP learns it
    probs_raw = softmax(S)                # scores used as they are: cramped
    probs_hot = softmax(S * temperature)  # stretched first, then softmax
    off_hot = np.where(np.eye(3, dtype=bool), 0.0, probs_hot)   # wrong pairs only
    print("\nrow probabilities with NO temperature:")
    show(photos, probs_raw)
    print("weakest true-pair probability: %.4f (no temperature) -> %.8f (x%g)"
          % (np.diag(probs_raw).min(), np.diag(probs_hot).min(), temperature))
    print("biggest wrong-pair probability after x%g: %.4e" % (temperature, off_hot.max()))
    # A row-wise softmax makes every ROW add to 1 and leaves the columns alone.
    row_sums, col_sums = probs_raw.sum(axis=1), probs_raw.sum(axis=0)
    print("row sums:", np.round(row_sums, 6), " column sums:", np.round(col_sums, 6),
          "-> rows add to 1, columns do not")

    # --- Part 5: the contrastive loss, and why the PUSH is needed ---------
    # Symmetric: photos picking captions (S) and captions picking photos (S.T).
    # Each direction on its own FIRST: the average of the pair is the same number if the
    # two sides get swapped, so only the average would hide a transpose inside the loss.
    loss_i2t, loss_t2i = contrastive_loss(S), contrastive_loss(S.T)
    loss_raw = (loss_i2t + loss_t2i) / 2
    loss_hot = (contrastive_loss(S * temperature) + contrastive_loss(S.T * temperature)) / 2
    print("\nphotos->captions %.4f and captions->photos %.4f — two different numbers"
          % (loss_i2t, loss_t2i))
    print("symmetric contrastive loss: %.4f (no temperature) -> %.4e (x%g)"
          % (loss_raw, loss_hot, temperature))
    # Negative control for the DIAGONAL. Shuffle the rows so no row's true pair is its
    # best cell any more. Here the loss must explode; a loss that read each row's biggest
    # score instead of its labelled cell would still print ~0 and look fine.
    S_mismatched = S[[1, 2, 0]]
    loss_mismatched = contrastive_loss(S_mismatched * temperature)
    print("rows shuffled so every diagonal cell is a WRONG pair -> loss %.4f, not ~0"
          % loss_mismatched)
    # The pull-only cheat: every photo AND every caption land on ONE single point.
    collapsed = normalize(np.ones((3, 3)))
    S_collapsed = collapsed @ collapsed.T
    loss_collapsed = (contrastive_loss(S_collapsed * temperature)
                      + contrastive_loss(S_collapsed.T * temperature)) / 2
    print("collapsed space (every input on one point): S row 0 = %s  probabilities = %s"
          "\n-> a 1-in-3 guess, loss %.4f, and no temperature can rescue it"
          % (np.round(S_collapsed[0], 4),
             np.round(softmax(S_collapsed * temperature)[0], 4), loss_collapsed))

    # --- Part 6: zero-shot — the label set is chosen at test time ---------
    scores_row0 = img_n[0] @ txt_n.T      # photo 0 against all 3 caption sentences
    pick0 = int(np.argmax(scores_row0))
    print("\nzero-shot for photo 0 (the dog photo):")
    for j, caption in enumerate(captions):
        print("  %-26s -> %.4f%s" % (caption, scores_row0[j], "  <- picked" if j == pick0 else ""))
    print("argmax =", pick0, "-> predicted caption:", captions[pick0])
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
    # broken change above cannot quietly agree with itself.
    shapes_ok = img.shape == (3, 3) and txt.shape == (3, 3) and S.shape == (3, 3)
    txt_was_not_unit = np.array_equal(np.round(len_txt_before, 4), [0.9055, 0.9566, 0.9055])
    # round(..., 12) == 0.0 instead of "< tolerance": an exact pin cannot be loosened.
    rows_are_unit = round(float(np.abs(len_after - 1).max()), 12) == 0.0
    normalized_by_row = np.array_equal(np.round(col_len_txt, 4), [0.9953, 1.0054, 0.9994])
    grid_ok = np.array_equal(np.round(S, 4), [[0.9939, 0.0523, 0.0],
                                              [0.1104, 0.9931, 0.1104],
                                              [0.0, 0.1045, 0.9939]])
    # The predicted answer to "closer to 0 or to 1?" — and the two cells must disagree,
    # which is what shows nearer() is reading the score and not returning a fixed word.
    cosine_reads_direction = nearer(float(S[0, 0])) == "1" and nearer(float(S[0, 2])) == "0"
    diagonal_wins = winner == "diagonal" and round(float(margins.min()), 4) == 0.8827
    softmax_by_row = (round(float(np.abs(row_sums - 1).max()), 12) == 0.0
                      and round(float(np.abs(col_sums - 1).max()), 6) == 0.002331)
    temp_sharpens = (round(float(np.diag(probs_raw).min()), 4) == 0.5473
                     and "%.4e" % off_hot.max() == "2.1520e-08")
    loss_drops = round(loss_raw, 4) == 0.5819 and "%.4e" % loss_hot == "2.4349e-08"
    # The two directions are pinned apart, so swapping the sides inside the loss is caught.
    directions_differ = round(loss_i2t, 4) == 0.5818 and round(loss_t2i, 4) == 0.5819
    # ...and with the true pairs off the diagonal the loss must be huge, which is what
    # shows it reads the labelled cell rather than each row's largest score.
    reads_the_diagonal = round(loss_mismatched, 4) == 18.4396
    collapse_is_chance = (round(loss_collapsed, 4) == 1.0986
                          and np.array_equal(np.round(S_collapsed, 4), np.ones((3, 3))))
    # expected_slot != 0 keeps the re-order from being a no-op that proves nothing.
    # The scores are pinned too, not only the winner: with these three captions the
    # winner alone would still look right if the two sides were fed in swapped.
    zero_shot_ok = (pick0 == 0 and expected_slot != 0 and pick_shuffled == expected_slot
                    and np.array_equal(np.round(scores_row0, 4), [0.9939, 0.0523, 0.0]))

    if all([shapes_ok, txt_was_not_unit, rows_are_unit, normalized_by_row,
            cosine_reads_direction, grid_ok, diagonal_wins, softmax_by_row,
            temp_sharpens, loss_drops, directions_differ, reads_the_diagonal,
            collapse_is_chance, zero_shot_ok]):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected txt row lengths [0.9055 0.9566 0.9055] before and 1 after, "
              "txt_n column lengths [0.9953 1.0054 0.9994], S = [[0.9939 0.0523 0] [0.1104 0.9931 "
              "0.1104] [0 0.1045 0.9939]] with the diagonal winning every row by 0.8827 at the "
              "closest, the weakest true pair 0.5473 -> wrong pairs 2.1520e-08 with temperature, "
              "loss 0.5819 -> 2.4349e-08, a collapsed space stuck at 1.0986, and photo 0 picking "
              "caption 0 (and its caption's new slot after the re-order)")

    assert shapes_ok, "img, txt and S should all be (3, 3)"
    assert txt_was_not_unit, "txt rows should start at lengths [0.9055 0.9566 0.9055]"
    assert rows_are_unit, "after normalize, all 6 row lengths must be 1"
    assert normalized_by_row, "txt_n column lengths should be [0.9953 1.0054 0.9994]"
    assert grid_ok, "S should be [[.9939 .0523 0] [.1104 .9931 .1104] [0 .1045 .9939]]"
    assert cosine_reads_direction, "the true pair should read as nearer 1, the mismatch as nearer 0"
    assert diagonal_wins, "every row's biggest cell must be its diagonal, by 0.8827 at the closest"
    assert softmax_by_row, "softmax must normalize rows (columns should be off by 0.002331)"
    assert temp_sharpens, "0.5473 without temperature; wrong pairs 2.1520e-08 with it"
    assert loss_drops, "symmetric loss should be 0.5819 raw and 2.4349e-08 with temperature"
    assert directions_differ, "the two directions should be 0.5818 and 0.5819, not one number"
    assert reads_the_diagonal, "with the true pairs off the diagonal the loss must be 18.4396"
    assert collapse_is_chance, "a collapsed space should sit at 1.0986 = a 1-in-3 guess"
    assert zero_shot_ok, "photo 0 should score [0.9939 0.0523 0] and follow caption 0's slot"
