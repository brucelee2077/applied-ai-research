# day-05-data-augmentation — experiment
#
# Today's big idea in two lines of output:
#   A safe disguise changes WHERE the pixels sit and never the label — mirror the
#   photo and the same 16 values come back with the columns swapped.
#   Push one step too far (a 180 degree half-turn of a 6) and the label becomes a
#   9, so the disguise is now teaching the model a lie.
#
# The kit, in the lesson's order: a horizontal flip (c3), the always-on normalize
# step and its mismatch trap (c5), the train-only rule (c6), the lying rotation (c8).
# Run it:  python3 sessions/m06-cnns-vision-encoders/day-05-data-augmentation/experiment.py

import numpy as np  # numpy gives us arrays, slicing and np.rot90

FLIP_PROB = 0.5   # the coin: about half of the training passes get the mirror


def show(name, a):
    # One row per line plus the shape, so stdout tells the story by itself.
    print("%s -> shape %s" % (name, a.shape))
    for row in np.round(a, 4):
        print("     ", row)


def pipeline(image, rng, train, fixed_mean, fixed_std):
    """One photo through the pipeline. The random disguise runs ONLY if train."""
    # Randomness is spent only on a training pass, so a test view cannot drift.
    draw = rng.random() if train else None
    did_flip = bool(train and draw < FLIP_PROB)
    view = image[:, ::-1] if did_flip else image
    # Normalize is NOT a disguise: it runs on every photo, train and test alike,
    # always with the SAME fixed numbers measured once on the training data.
    return (view - fixed_mean) / fixed_std, draw, did_flip


if __name__ == "__main__":
    # --- Part 1: one fake photo -------------------------------------------
    # No image dataset is on this machine, so the "photo" is 16 counted-up grey
    # levels. Every column differs, which is what lets a flip show up at all.
    img = np.arange(16).reshape(4, 4).astype(float)
    show("img (fake 4x4 grayscale photo, label \"cat\")", img)

    # --- Part 2: horizontal flip — a label-preserving disguise ------------
    flipped = img[:, ::-1]              # [:, ::-1] reverses the column axis
    show("flipped = img[:, ::-1]", flipped)
    # The prediction, written down before running: column 0 <-> 3, column 1 <-> 2.
    predicted_flip = np.array([[3.0, 2.0, 1.0, 0.0], [7.0, 6.0, 5.0, 4.0],
                               [11.0, 10.0, 9.0, 8.0], [15.0, 14.0, 13.0, 12.0]])
    # The same claim measured off the data: each output column, matched back to
    # the input column it came from. Mirroring must answer 3, 2, 1, 0. The search
    # bound is read from the shape, and an unmatched column reports -1 rather than
    # raising, so a wrong flip axis reaches the self-check instead of crashing.
    col_map = [next((j for j in range(img.shape[1])
                     if np.array_equal(flipped[:, c], img[:, j])), -1)
               for c in range(flipped.shape[1])]
    print("each flipped column came from input column:", col_map, "(mirrored)")
    print("same 16 values?", np.array_equal(np.sort(flipped, axis=None),
                                            np.sort(img, axis=None)),
          " pixels in the same places?", np.array_equal(flipped, img),
          "\n-> the picture moved, the value set did not: the label stays \"cat\"")

    # --- Part 3: normalize — the always-on step, same numbers everywhere --
    # Measured ONCE on the training data, then frozen. np.std() is the population
    # spread (ddof=0); ddof=1 would give 4.761 and move every number below.
    train_mean, train_std = float(img.mean()), float(img.std())
    print("\nfrozen train numbers: mean = %.4f  spread = %.4f" % (train_mean, train_std))
    norm = (img - train_mean) / train_std
    show("norm = (img - mean) / spread", norm)
    print("norm.mean() =", round(float(norm.mean()), 6),
          " norm.std() =", round(float(norm.std()), 6), "-> centered on 0, spread 1")
    # The mismatch trap: a NEW photo of the same scene, 40 grey levels brighter.
    bright = img + 40.0
    right_way = (bright - train_mean) / train_std       # frozen numbers: brightness survives
    wrong_way = (bright - bright.mean()) / bright.std()  # its OWN numbers: brightness erased
    print("brighter photo, frozen numbers  -> mean", round(float(right_way.mean()), 4))
    print("brighter photo, its own numbers -> mean", round(float(wrong_way.mean()), 4),
          " identical to the dark photo?", np.array_equal(wrong_way, norm),
          "\n-> re-measuring at test time throws the +40 away: use the frozen numbers")

    # --- Part 4: the iron rule — disguise training photos only -----------
    # Seeded on purpose: default_rng(0) makes the coin flips repeatable, so the
    # numbers pinned in the self-check below are stable across runs.
    rng = np.random.default_rng(0)
    train_views, train_draws = [], []
    for _ in range(2):
        view, draw, did_flip = pipeline(img, rng, True, train_mean, train_std)
        train_views.append(view)
        train_draws.append(round(float(draw), 4))
        print("\ntrain pass: coin %.4f < %.2f ? %s" % (draw, FLIP_PROB, did_flip))
        show("  train view", view)
    print("two train views identical?", np.array_equal(train_views[0], train_views[1]),
          "-> a fresh disguise per pass (pass 1 drew no flip, pass 2 did)")
    # One test view per training pass, so the two columns line up side by side.
    test_views = [pipeline(img, rng, False, train_mean, train_std)[0] for _ in train_views]
    print("all test views identical?",
          all(np.array_equal(v, test_views[0]) for v in test_views),
          "-> plain and repeatable: random disguises are TRAINING ONLY")
    # Push the BRIGHTER photo through the same pipeline. Because the pipeline uses
    # the frozen numbers rather than re-measuring, the +40 is still visible here.
    # On the plain photo a re-measuring pipeline would look identical, so this is
    # the pass that actually pins which numbers the pipeline used.
    bright_view = pipeline(bright, rng, False, train_mean, train_std)[0]
    print("brighter photo through the test pipeline -> mean",
          round(float(bright_view.mean()), 4), "(frozen numbers, so the +40 survives)")
    show("  test view (no flip, normalize only)", test_views[0])

    # --- Part 5: a disguise that destroys the label ----------------------
    # A crude 5x4 "6": the closed loop sits at the BOTTOM (rows 2 to 4).
    six = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1],
                    [1, 0, 0, 1], [1, 1, 1, 1]])
    nine = np.rot90(six, 2)   # a 180 degree half-turn: both axes reversed
    show("\nsix (loop at the bottom)", six)
    show("nine = np.rot90(six, 2)", nine)
    # Where the ink sits, counted rather than eyeballed: the loop is the heavy end.
    six_top, six_bottom = int(six[:2].sum()), int(six[3:].sum())
    nine_top, nine_bottom = int(nine[:2].sum()), int(nine[3:].sum())
    print("ink — six: top rows %d, bottom rows %d (bottom-heavy, reads \"6\");  "
          "nine: top %d, bottom %d (top-heavy, reads \"9\")"
          % (six_top, six_bottom, nine_top, nine_bottom))
    print("np.array_equal(six, nine) =", np.array_equal(six, nine),
          "-> a different class now, so keeping the label \"6\" teaches a lie")
    # The lesson's footnote: a single flip is a different transform and gives no
    # clean 9 — only the half-turn (both axes at once) does.
    print("flipud(six) or fliplr(six) equal to the half-turn?",
          np.array_equal(np.flipud(six), nine) or np.array_equal(np.fliplr(six), nine))
    print("\ntakeaway: augmentation = label-preserving disguises applied to TRAINING "
          "photos only; normalize every photo with the same fixed numbers; a "
          "too-strong disguise (a 180 degree rotation turning a 6 into a 9) "
          "destroys the label.")

    # --- Self-check: one boolean per claim -------------------------------
    # Every expected value below was read off a real run and typed here, so a
    # broken change above cannot quietly agree with itself.
    exp_norm_row0 = np.array([-1.627, -1.41, -1.1931, -0.9762])
    exp_flipped_row0 = np.array([-0.9762, -1.1931, -1.41, -1.627])
    exp_nine = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1],
                         [0, 0, 0, 1], [1, 1, 1, 0]])
    shapes_ok = img.shape == (4, 4) and norm.shape == (4, 4) and six.shape == (5, 4)
    flip_ok = np.array_equal(flipped, predicted_flip)     # mirrored, not upside-down
    mirror_ok = (col_map == [3, 2, 1, 0]                  # measured permutation
                 and np.array_equal(np.sort(flipped, axis=None), np.sort(img, axis=None))
                 and not np.array_equal(flipped, img))    # same values, new places
    frozen_ok = round(train_mean, 4) == 7.5 and round(train_std, 4) == 4.6098
    norm_ok = (np.array_equal(np.round(norm[0], 4), exp_norm_row0)
               and round(float(norm.mean()), 6) == 0.0
               and round(float(norm.std()), 6) == 1.0)
    mismatch_ok = (round(float(right_way.mean()), 4) == 8.6772   # +40 survives
                   and np.array_equal(wrong_way, norm))          # +40 erased
    draws_ok = train_draws == [0.637, 0.2698]                    # the seeded coin stream
    train_ok = (np.array_equal(np.round(train_views[0][0], 4), exp_norm_row0)
                and np.array_equal(np.round(train_views[1][0], 4), exp_flipped_row0)
                and not np.array_equal(train_views[0], train_views[1]))
    test_ok = (len(test_views) > 1                            # something to compare
               and all(np.array_equal(v, norm) for v in test_views))  # plain, no disguise
    # Which numbers did the pipeline normalize with? Only the frozen ones leave the
    # +40 in place; re-measuring per photo would land the brighter photo on 0.0.
    frozen_in_pipeline = round(float(bright_view.mean()), 4) == 8.6772
    nine_ok = np.array_equal(nine, exp_nine)
    loop_moved = (six_top, six_bottom, nine_top, nine_bottom) == (4, 6, 6, 4)
    # Entailed by nine_ok, but the lesson names both explicitly, so state them.
    halfturn_lies = (not np.array_equal(six, nine)
                     and not np.array_equal(np.flipud(six), nine)
                     and not np.array_equal(np.fliplr(six), nine))

    if (shapes_ok and flip_ok and mirror_ok and frozen_ok and norm_ok and mismatch_ok
            and draws_ok and train_ok and test_ok and frozen_in_pipeline
            and nine_ok and loop_moved
            and halfturn_lies):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected columns permuted [3, 2, 1, 0], frozen mean 7.5 and "
              "spread 4.6098, norm row 0 = [-1.627 -1.41 -1.1931 -0.9762] at mean 0.0 / "
              "std 1.0, the brighter photo at mean 8.6772 on frozen numbers but identical "
              "to norm on its own, coins [0.637, 0.2698] giving a plain then a mirrored "
              "train view, one repeatable test view equal to norm, the brighter photo at 8.6772 "
              "through the pipeline, and rot90(six, 2) = "
              "[[1111],[1001],[1111],[0001],[1110]] moving the ink from 4 top / 6 bottom "
              "to 6 top / 4 bottom")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert shapes_ok, "img and norm should be (4, 4) and the digit bitmap (5, 4)"
    assert flip_ok, "img[:, ::-1] should mirror columns: row 0 becomes [3 2 1 0]"
    assert mirror_ok, "the flip should permute columns [3, 2, 1, 0] and keep all 16 values"
    assert frozen_ok, "the frozen numbers should be mean 7.5, spread 4.6098 (ddof=0)"
    assert norm_ok, "norm row 0 should be [-1.627 -1.41 -1.1931 -0.9762], mean 0, std 1"
    assert mismatch_ok, "the brighter photo should be mean 8.6772 frozen, == norm re-measured"
    assert draws_ok, "the seeded coins should be [0.637, 0.2698]"
    assert train_ok, "train pass 1 should be plain and pass 2 mirrored — a random disguise"
    assert test_ok, "both test views must equal the plain normalized image"
    assert frozen_in_pipeline, ("the pipeline must normalize with the frozen numbers, "
                               "putting the brighter photo at mean 8.6772, not 0.0")
    assert nine_ok, "rot90(six, 2) should be [[1111],[1001],[1111],[0001],[1110]]"
    assert loop_moved, "the ink should move from 4 top / 6 bottom to 6 top / 4 bottom"
    assert halfturn_lies, "only the half-turn maps the 6 to a 9 — one flip does not"
