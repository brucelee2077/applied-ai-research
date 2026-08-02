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


def show_block(name, a):
    # Argument 1 is ONE TITLE for the whole array. Day 8 has a printer whose first
    # argument is a LIST of per-row labels instead, so the two are deliberately spelled
    # differently — `show_block(title, a)` here, `show_rows(labels, matrix)` there. A
    # single name for both contracts would let a day-5-style call print only row one.
    # One row per line plus the shape, so stdout tells the story by itself.
    # Hands back exactly what it printed, so the self-check reads the SHOWN values
    # instead of computing them a second time.
    shown_shape = a.shape
    shown_rows = np.round(a, 4)
    print("%s -> shape %s" % (name, shown_shape))
    for row in shown_rows:
        print("     ", row)
    return shown_shape, shown_rows


class FixedDraw:
    """A stand-in for the rng that always hands back one chosen coin value.

    It exists so the `<` in `draw < FLIP_PROB` can be tested ON the threshold:
    the seeded stream never draws exactly 0.5, so without this a `<=` typo would
    change the rule and nothing would notice.
    """

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


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
    img_shape, shown_img = show_block("img (fake 4x4 grayscale photo, label \"cat\")", img)

    # --- Part 2: horizontal flip — a label-preserving disguise ------------
    flipped = img[:, ::-1]              # [:, ::-1] reverses the column axis
    flipped_shape, shown_flipped = show_block("flipped = img[:, ::-1]", flipped)
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
    shown_same_values = np.array_equal(np.sort(flipped, axis=None),
                                       np.sort(img, axis=None))
    shown_same_places = np.array_equal(flipped, img)
    print("same 16 values?", shown_same_values,
          " pixels in the same places?", shown_same_places,
          "\n-> the picture moved, the value set did not: the label stays \"cat\"")

    # --- Part 3: normalize — the always-on step, same numbers everywhere --
    # "Normalize" here means the STANDARDIZING sense: subtract a fixed average and
    # divide by a fixed spread — day 4's recipe. (It is not the unit-length sense of
    # dividing a vector by its own length — that is a different operation with a
    # different purpose, and it is the one day 8 gives the name `normalize` to.)
    # Note what is missing next to day 4: no eps inside the divide. Day 4 taught eps AS
    # the mechanism that keeps this divide finite and proved it returns nan without one.
    # It is safe to leave out here only because this spread is 4.6098 — printed and
    # pinned below, so the claim is checked and not just asserted in prose. A constant
    # photo would have spread 0 and would need day 4's crumb.
    # Measured ONCE on the training data, then frozen. np.std() is the population
    # spread (ddof=0); ddof=1 would give 4.761 and move every number below.
    train_mean, train_std = float(img.mean()), float(img.std())
    # The rendered line is bound first: the learner and the self-check read one string.
    shown_frozen_line = ("frozen train numbers: mean = %.4f  spread = %.4f"
                         % (train_mean, train_std))
    print("\n" + shown_frozen_line)
    norm = (img - train_mean) / train_std
    norm_shape, shown_norm = show_block("norm = (img - mean) / spread", norm)
    shown_norm_mean = round(float(norm.mean()), 6)
    shown_norm_std = round(float(norm.std()), 6)
    print("norm.mean() =", shown_norm_mean,
          " norm.std() =", shown_norm_std, "-> centered on 0, spread 1")
    # The mismatch trap: a NEW photo of the same scene, 40 grey levels brighter.
    bright = img + 40.0
    right_way = (bright - train_mean) / train_std       # frozen numbers: brightness survives
    wrong_way = (bright - bright.mean()) / bright.std()  # its OWN numbers: brightness erased
    shown_right_mean = round(float(right_way.mean()), 4)
    shown_wrong_mean = round(float(wrong_way.mean()), 4)
    shown_wrong_equals_norm = np.array_equal(wrong_way, norm)
    print("brighter photo, frozen numbers  -> mean", shown_right_mean)
    print("brighter photo, its own numbers -> mean", shown_wrong_mean,
          " identical to the dark photo?", shown_wrong_equals_norm,
          "\n-> re-measuring at test time throws the +40 away: use the frozen numbers")

    # --- Part 4: the iron rule — disguise training photos only -----------
    # Seeded on purpose: default_rng(0) makes the coin flips repeatable, so the
    # numbers pinned in the self-check below are stable across runs.
    rng = np.random.default_rng(0)
    train_views, train_draws, train_lines, shown_train_rows = [], [], [], []
    for _ in range(2):
        view, draw, did_flip = pipeline(img, rng, True, train_mean, train_std)
        train_views.append(view)
        train_draws.append(round(float(draw), 4))
        line = "train pass: coin %.4f < %.2f ? %s" % (draw, FLIP_PROB, did_flip)
        train_lines.append(line)
        print("\n" + line)
        shown_train_rows.append(show_block("  train view", view)[1])
    shown_train_views_identical = np.array_equal(train_views[0], train_views[1])
    print("two train views identical?", shown_train_views_identical,
          "-> a fresh disguise per pass (pass 1 drew no flip, pass 2 did)")
    # One test view per training pass, so the two columns line up side by side.
    test_views = [pipeline(img, rng, False, train_mean, train_std)[0] for _ in train_views]
    shown_test_views_identical = all(np.array_equal(v, test_views[0]) for v in test_views)
    print("all test views identical?", shown_test_views_identical,
          "-> plain and repeatable: random disguises are TRAINING ONLY")
    # The coin test is `draw < FLIP_PROB`, strictly less-than. The seeded stream drew
    # 0.6370 and 0.2698 — neither sits ON 0.5, so those two passes cannot tell `<`
    # from `<=`. Draw exactly 0.5 (and a hair under) to pin which one is in force.
    edge_view, edge_draw, edge_flip = pipeline(img, FixedDraw(FLIP_PROB), True,
                                              train_mean, train_std)
    under_view, under_draw, under_flip = pipeline(img, FixedDraw(FLIP_PROB - 1e-9), True,
                                                 train_mean, train_std)
    shown_edge_line = ("boundary coin exactly %.4f ? flip %s;  a hair under (%.9f) ? flip %s"
                       "  <- the test is < , not <=" % (edge_draw, edge_flip,
                                                        under_draw, under_flip))
    print(shown_edge_line)
    shown_edge_is_plain = np.array_equal(edge_view, norm)
    shown_under_is_flipped = np.array_equal(under_view, (flipped - train_mean) / train_std)
    print("  coin ON the threshold gives the PLAIN view?", shown_edge_is_plain,
          " a hair under gives the MIRRORED view?", shown_under_is_flipped)
    # Push the BRIGHTER photo through the same pipeline. Because the pipeline uses
    # the frozen numbers rather than re-measuring, the +40 is still visible here.
    # On the plain photo a re-measuring pipeline would look identical, so this is
    # the pass that actually pins which numbers the pipeline used.
    bright_view = pipeline(bright, rng, False, train_mean, train_std)[0]
    shown_bright_view_mean = round(float(bright_view.mean()), 4)
    print("brighter photo through the test pipeline -> mean",
          shown_bright_view_mean, "(frozen numbers, so the +40 survives)")
    test_shape, shown_test_view = show_block("  test view (no flip, normalize only)", test_views[0])

    # --- Part 5: a disguise that destroys the label ----------------------
    # A crude 5x4 "6": the closed loop sits at the BOTTOM (rows 2 to 4).
    six = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1],
                    [1, 0, 0, 1], [1, 1, 1, 1]])
    nine = np.rot90(six, 2)   # a 180 degree half-turn: both axes reversed
    six_shape, shown_six = show_block("\nsix (loop at the bottom)", six)
    nine_shape, shown_nine = show_block("nine = np.rot90(six, 2)", nine)
    # Where the ink sits, counted rather than eyeballed: the loop is the heavy end.
    six_top, six_bottom = int(six[:2].sum()), int(six[3:].sum())
    nine_top, nine_bottom = int(nine[:2].sum()), int(nine[3:].sum())
    shown_ink_line = ("ink — six: top rows %d, bottom rows %d (bottom-heavy, reads \"6\");  "
                      "nine: top %d, bottom %d (top-heavy, reads \"9\")"
                      % (six_top, six_bottom, nine_top, nine_bottom))
    print(shown_ink_line)
    shown_six_equals_nine = np.array_equal(six, nine)
    print("np.array_equal(six, nine) =", shown_six_equals_nine,
          "-> a different class now, so keeping the label \"6\" teaches a lie")
    # The lesson's footnote: a single flip is a different transform and gives no
    # clean 9 — only the half-turn (both axes at once) does.
    shown_single_flip_matches = (np.array_equal(np.flipud(six), nine)
                                 or np.array_equal(np.fliplr(six), nine))
    print("flipud(six) or fliplr(six) equal to the half-turn?", shown_single_flip_matches)
    print("\ntakeaway: augmentation = label-preserving disguises applied to TRAINING "
          "photos only; normalize every photo with the same fixed numbers; a "
          "too-strong disguise (a 180 degree rotation turning a 6 into a 9) "
          "destroys the label.")

    # --- Self-check: one boolean per claim -------------------------------
    # Every expected value below was read off a real run and typed here, so a
    # broken change above cannot quietly agree with itself. Each claim reads the
    # SAME `shown_*` value that was printed, so corrupting a printed number or a
    # printed sentence also breaks its claim.
    exp_norm_row0 = np.array([-1.627, -1.41, -1.1931, -0.9762])
    exp_flipped_row0 = np.array([-0.9762, -1.1931, -1.41, -1.627])
    exp_nine = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1],
                         [0, 0, 0, 1], [1, 1, 1, 0]])
    shapes_ok = (img_shape == (4, 4) and flipped_shape == (4, 4)
                 and norm_shape == (4, 4) and test_shape == (4, 4)
                 and six_shape == (5, 4) and nine_shape == (5, 4))
    flip_ok = np.array_equal(shown_flipped, predicted_flip)  # mirrored, not upside-down
    mirror_ok = (col_map == [3, 2, 1, 0]                  # measured permutation
                 and shown_same_values                    # same 16 values
                 and not shown_same_places                # in new places
                 and np.array_equal(shown_img, np.arange(16).reshape(4, 4)))
    frozen_ok = (shown_frozen_line == "frozen train numbers: mean = 7.5000  spread = 4.6098"
                 and round(train_mean, 4) == 7.5 and round(train_std, 4) == 4.6098)
    norm_ok = (np.array_equal(shown_norm[0], exp_norm_row0)
               and shown_norm_mean == 0.0
               and shown_norm_std == 1.0)
    mismatch_ok = (shown_right_mean == 8.6772       # +40 survives the frozen numbers
                   and shown_wrong_mean == 0.0      # re-measuring recentres it on 0
                   and shown_wrong_equals_norm)     # +40 erased: same as the dark photo
    draws_ok = (train_draws == [0.637, 0.2698]      # the seeded coin stream
                and train_lines == ["train pass: coin 0.6370 < 0.50 ? False",
                                    "train pass: coin 0.2698 < 0.50 ? True"])
    train_ok = (np.array_equal(shown_train_rows[0][0], exp_norm_row0)
                and np.array_equal(shown_train_rows[1][0], exp_flipped_row0)
                and not shown_train_views_identical)
    test_ok = (len(test_views) > 1                            # something to compare
               and shown_test_views_identical                 # repeatable
               and np.array_equal(shown_test_view[0], exp_norm_row0)
               and all(np.array_equal(v, norm) for v in test_views))  # plain, no disguise
    # The threshold itself: a coin exactly ON 0.5 must NOT flip (the test is `<`), and a
    # hair under must flip. Without this pair, `<` and `<=` behave the same here.
    boundary_ok = (edge_draw == 0.5 and edge_flip is False and shown_edge_is_plain
                   and under_flip is True and shown_under_is_flipped
                   and shown_edge_line == ("boundary coin exactly 0.5000 ? flip False;  "
                                           "a hair under (0.499999999) ? flip True"
                                           "  <- the test is < , not <="))
    # Which numbers did the pipeline normalize with? Only the frozen ones leave the
    # +40 in place; re-measuring per photo would land the brighter photo on 0.0.
    frozen_in_pipeline = shown_bright_view_mean == 8.6772
    nine_ok = np.array_equal(shown_nine, exp_nine) and np.array_equal(shown_six, six)
    loop_moved = ((six_top, six_bottom, nine_top, nine_bottom) == (4, 6, 6, 4)
                  and shown_ink_line == ("ink — six: top rows 4, bottom rows 6 "
                                         "(bottom-heavy, reads \"6\");  nine: top 6, "
                                         "bottom 4 (top-heavy, reads \"9\")"))
    # Entailed by nine_ok, but the lesson names both explicitly, so state them.
    halfturn_lies = not shown_six_equals_nine and not shown_single_flip_matches

    if (shapes_ok and flip_ok and mirror_ok and frozen_ok and norm_ok and mismatch_ok
            and draws_ok and train_ok and test_ok and boundary_ok and frozen_in_pipeline
            and nine_ok and loop_moved
            and halfturn_lies):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected columns permuted [3, 2, 1, 0], frozen mean 7.5 and "
              "spread 4.6098, norm row 0 = [-1.627 -1.41 -1.1931 -0.9762] at mean 0.0 / "
              "std 1.0, the brighter photo at mean 8.6772 on frozen numbers but identical "
              "to norm on its own, coins [0.637, 0.2698] giving a plain then a mirrored "
              "train view, a coin exactly on 0.5 giving NO flip and 0.5 - 1e-9 giving one, "
              "one repeatable test view equal to norm, the brighter photo at 8.6772 "
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
    assert boundary_ok, ("a coin exactly on 0.5 must NOT flip and 0.5 - 1e-9 must — "
                        "the rule is draw < 0.5, strictly less-than")
    assert frozen_in_pipeline, ("the pipeline must normalize with the frozen numbers, "
                               "putting the brighter photo at mean 8.6772, not 0.0")
    assert nine_ok, "rot90(six, 2) should be [[1111],[1001],[1111],[0001],[1110]]"
    assert loop_moved, "the ink should move from 4 top / 6 bottom to 6 top / 4 bottom"
    assert halfturn_lies, "only the half-turn maps the 6 to a 9 — one flip does not"
