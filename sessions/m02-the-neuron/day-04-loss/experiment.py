# Module 2 · Day 4 — Loss Functions
#
# Today's big idea: a loss is ONE number that says how wrong a guess is.
# Lower is better; 0 means perfect. Training is just making that number smaller.
# Keep that meaning: it needs a prediction AND a target. Days 7 and 8 practise the
# downhill walk on a toy loss SURFACE instead — a bowl in weight space with no data,
# no prediction and no target — so when those days say "the bottom is at weight = 0"
# they are describing the shape they wrote down, not a perfect prediction.
#
# We will watch two scorecards react:
#   1. MSE / MAE — the scores for guessing a NUMBER (and how ONE wild outlier
#      gets AMPLIFIED by MSE while MAE merely FEELS it).
#   2. Cross-entropy — the score for a CONFIDENT guess (and how clamping
#      keeps a "0% sure and wrong" case finite instead of blowing up to +infinity).
#
# Run me:  python3 sessions/m02-the-neuron/day-04-loss/experiment.py

import numpy as np  # numpy gives us fast array math (mean, abs, log)


# --- Score #1: for guessing a NUMBER ----------------------------------------

def mse(pred, target):
    # MSE = Mean Squared Error.
    # Step 1: pred - target is the "miss" on each example (how far off).
    # Step 2: (...)**2 squares each miss — this kills the sign (so +1 and -1
    #         are equally bad) AND makes big misses count much more.
    # Step 3: np.mean(...) averages them into one number.
    return np.mean((pred - target) ** 2)


def mae(pred, target):
    # MAE = Mean Absolute Error.
    # Same misses, but we take the PLAIN size |miss| (np.abs) instead of squaring.
    # A miss of 100 counts as 100, not 100*100 — so one outlier stays in
    # proportion instead of being blown up into a monster.
    return np.mean(np.abs(pred - target))


# --- Score #2: for a CONFIDENT (yes/no or one-of-many) guess ----------------

def cross_entropy(p):
    # Cross-entropy looks ONLY at p = the probability the model gave the
    # CORRECT answer, and scores -log(p).
    #   p near 1 (confident + right)  -> -log(p) near 0     (barely a ding)
    #   p near 0 (confident + wrong)  -> -log(p) shoots up  (a brutal sting)
    #
    # np.clip(p, 1e-7, 1-1e-7) is "epsilon clipping": it nudges p just off the
    # cliff edge so we never take log(0), which would be +infinity and later
    # poison training with a NaN.
    p_safe = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.log(p_safe)


if __name__ == "__main__":
    # ================= Part 1: one close guess, scored ======================
    # `pred` is the same object Day 3 printed as yhat (ŷ) — the model's prediction —
    # and `target` is the true answer it is graded against. Days 6 and 9 keep both names.
    pred = np.array([2.5, 0.0, 2.0])       # the network's three guesses
    target = np.array([3.0, -0.5, 2.0])    # the three true answers

    close_mse = round(float(mse(pred, target)), 3)   # a CLOSE guess
    close_mae = round(float(mae(pred, target)), 3)
    print("close guess  -> MSE", close_mse, " MAE", close_mae)

    # ================= Part 2: the outlier hijack, on a CROWD ===============
    # Twenty ordinary houses, each missed by exactly 0.5.
    clean_target = np.full(20, 3.0)
    clean_pred = np.full(20, 3.5)

    # The SAME twenty houses plus one freak mansion missed by 100.
    wild_target = np.append(clean_target, 103.0)
    wild_pred = np.append(clean_pred, 3.0)   # that last miss is -100

    clean_m = round(float(mse(clean_pred, clean_target)), 3)
    clean_a = round(float(mae(clean_pred, clean_target)), 3)
    wild_m = round(float(mse(wild_pred, wild_target)), 1)
    wild_a = round(float(mae(wild_pred, wild_target)), 1)
    print("20 ordinary misses    -> MSE", clean_m, " MAE", clean_a)
    print("same 20 + ONE mansion -> MSE", wild_m, " MAE", wild_a)

    # The honest comparison: BOTH scores grow, but by wildly different factors.
    # MSE squares the 100 (-> 10000), so the score is AMPLIFIED about 1900x.
    # MAE counts the 100 as 100, so it grows only about 10x — the outlier is
    # FELT, not amplified, and the twenty honest misses keep their say.
    grow_m = round(wild_m / clean_m)
    grow_a = round(wild_a / clean_a, 1)
    print("growth factor         -> MSE x", grow_m, " MAE x", grow_a)

    # ================= Part 2b: a LOPSIDED batch (contrast control) ==========
    # The crowd above is deliberately tidy: every miss is exactly 0.5, so its
    # mean, its middle value and its worst value are all 0.5 — the numbers alone
    # cannot show you that these scores AVERAGE every miss. So here is a batch
    # where the four misses differ, and mixed signs prove the sign is dropped.
    lop_pred = np.array([0.2, -0.4, 0.6, -3.0])
    lop_target = np.zeros(4)
    lop_m = round(float(mse(lop_pred, lop_target)), 3)
    lop_a = round(float(mae(lop_pred, lop_target)), 3)
    # The two things MAE is NOT: the middle miss, and the worst miss.
    lop_middle = round(float(np.median(np.abs(lop_pred - lop_target))), 3)
    lop_worst = round(float(np.max(np.abs(lop_pred - lop_target))), 3)
    print("lopsided misses [+0.2,-0.4,+0.6,-3.0] -> MSE", lop_m, " MAE", lop_a)
    print("   ... middle |miss|", lop_middle, " worst |miss|", lop_worst,
          "-> MAE averages ALL four, it does not pick one")

    # ================= Part 3: cross-entropy rewards confidence =============
    # As p (probability on the CORRECT answer) climbs 0.1 -> 0.99, the loss falls.
    # Score every probability ONCE, into one list. Everything below — the printed
    # table, the two drops, and the asserts at the bottom — reads THIS list. One
    # value, one expression: a wrong number cannot be printed and checked separately.
    ce_probe_ps = [0.99, 0.6, 0.1, 1e-7]
    ce_exact = [float(cross_entropy(p)) for p in ce_probe_ps]
    ce_values = [round(v, 3) for v in ce_exact]
    print("cross-entropy for p=[0.99, 0.6, 0.1, 1e-7]:", ce_values)

    # WHERE does cross-entropy change fastest? Compare two equal-looking climbs.
    # Going 0.1 -> 0.6 buys a much bigger drop than 0.6 -> 0.99, so the loss is
    # steepest down at the CONFIDENT-WRONG end (p -> 0) and flattens as p -> 1.
    # Both drops are differences of the SAME scores printed in the table above.
    drop_low = round(ce_exact[2] - ce_exact[1], 3)    # p = 0.1  ->  p = 0.6
    drop_high = round(ce_exact[1] - ce_exact[0], 3)   # p = 0.6  ->  p = 0.99
    print("drop p=0.1->0.6:", drop_low, " drop p=0.6->0.99:", drop_high,
          "-> steepest near p=0 (confident + wrong), flat near p=1")

    # The p=0 case, clamped, stays FINITE (~16.1) instead of +infinity.
    ce_zero = round(float(cross_entropy(0.0)), 1)
    print("confident+wrong p=0 (clamped) -> cross-entropy", ce_zero, "(not +inf)")

    # The clip has a SECOND guard, 1-1e-7, at the other end. It matters the
    # moment a model reports p = 1.0 exactly (the "1-p" side of binary
    # cross-entropy needs it), so here is the p=1 row that makes it visible:
    # the score is a hair ABOVE zero, not exactly zero.
    ce_one = round(float(cross_entropy(1.0)), 7)
    print("confident+right p=1 (clamped) -> cross-entropy", ce_one,
          "(a hair above 0 — the upper clamp fired)")

    # ================= The honest limit =====================================
    # A loss only SCORES a guess (and hands over a slope). It never changes a
    # single weight — that is the optimizer's job, and the optimizer gets its
    # own lesson shortly after tomorrow's gradients.
    print("reminder: a loss only SCORES; a separate optimizer step changes weights.")

    # ================= Self-check against the lesson's stated numbers =======
    # The lesson says: MSE on the close guess is ~0.167 (MAE ~0.333); the crowd
    # test goes 0.25 -> 476.4 for MSE (1906x) but only 0.5 -> 5.2 for MAE (10.4x);
    # the two cross-entropy drops are ~1.79 and ~0.50; and the clamped p=0
    # cross-entropy is ~16.1. Every one of those is pinned to a literal below.
    try:
        assert close_mse == 0.167, close_mse            # "what you should see"
        assert close_mae == 0.333, close_mae            # the plain-size twin
        assert ce_zero == 16.1, ce_zero                 # lower clip keeps it finite
        assert ce_one == 1e-07, ce_one                  # upper clip nudges p=1 off 1
        assert (clean_m, clean_a) == (0.25, 0.5), (clean_m, clean_a)
        assert wild_m == 476.4, wild_m                  # MSE amplified the outlier
        assert wild_a == 5.2, wild_a                    # MAE merely felt it
        assert grow_m == 1906, grow_m                   # ~1900x ...
        assert grow_a == 10.4, grow_a                   # ... versus only ~10x
        # The lopsided batch: MAE is the AVERAGE miss (1.05), which is neither the
        # middle miss (0.5) nor the worst (3.0), and the signs are gone.
        assert (lop_m, lop_a) == (2.39, 1.05), (lop_m, lop_a)
        assert (lop_middle, lop_worst) == (0.5, 3.0), (lop_middle, lop_worst)
        # The whole printed cross-entropy table, row by row: rewards confidence
        # (0.01 at p=0.99), stings a confident-wrong guess (2.303 at p=0.1), and
        # stays finite at the clip (16.118). The two drops below are differences of
        # these same four numbers, so the table and the drops cannot disagree.
        assert ce_values == [0.01, 0.511, 2.303, 16.118], ce_values
        assert drop_low == 1.792, drop_low              # steepest near p=0 ...
        assert drop_high == 0.501, drop_high            # ... and flat near p=1
        print("✅ you got it")
    except AssertionError as bad:
        print("❌ not yet — expected MSE 0.167, crowd MSE 476.4 vs MAE 5.2,"
              " lopsided MAE 1.05, and clamped cross-entropy 16.1, got", bad)
        raise
