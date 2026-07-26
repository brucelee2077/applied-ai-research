# Module 2 · Day 4 — Loss Functions
#
# Today's big idea: a loss is ONE number that says how wrong a guess is.
# Lower is better; 0 means perfect. Training is just making that number smaller.
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

    # ================= Part 3: cross-entropy rewards confidence =============
    # As p (probability on the CORRECT answer) climbs 0.1 -> 0.99, the loss falls.
    ce_values = [round(float(cross_entropy(p)), 3) for p in [0.99, 0.6, 0.1, 1e-7]]
    print("cross-entropy for p=[0.99, 0.6, 0.1, 1e-7]:", ce_values)

    # WHERE does cross-entropy change fastest? Compare two equal-looking climbs.
    # Going 0.1 -> 0.6 buys a much bigger drop than 0.6 -> 0.99, so the loss is
    # steepest down at the CONFIDENT-WRONG end (p -> 0) and flattens as p -> 1.
    drop_low = round(float(cross_entropy(0.1) - cross_entropy(0.6)), 3)
    drop_high = round(float(cross_entropy(0.6) - cross_entropy(0.99)), 3)
    print("drop p=0.1->0.6:", drop_low, " drop p=0.6->0.99:", drop_high,
          "-> steepest near p=0 (confident + wrong), flat near p=1")

    # The p=0 case, clamped, stays FINITE (~16.1) instead of +infinity.
    ce_zero = round(float(cross_entropy(0.0)), 1)
    print("confident+wrong p=0 (clamped) -> cross-entropy", ce_zero, "(not +inf)")

    # ================= The honest limit =====================================
    # A loss only SCORES a guess (and hands over a slope). It never changes a
    # single weight — that is the optimizer's job, and the optimizer gets its
    # own lesson shortly after tomorrow's gradients.
    print("reminder: a loss only SCORES; a separate optimizer step changes weights.")

    # ================= Self-check against the lesson's stated numbers =======
    # The lesson says: MSE on the close guess is ~0.167; the crowd test goes
    # 0.25 -> ~476 for MSE (about 1900x) but only 0.5 -> ~5.2 for MAE (~10x);
    # and the clamped p=0 cross-entropy is ~16.1.
    try:
        assert close_mse == 0.167, close_mse            # "what you should see"
        assert ce_zero == 16.1, ce_zero                 # clip keeps it finite
        assert (clean_m, clean_a) == (0.25, 0.5), (clean_m, clean_a)
        assert wild_m > 400, wild_m                     # MSE amplified the outlier
        assert wild_a < 6, wild_a                       # MAE merely felt it
        assert grow_m > 100 * grow_a, (grow_m, grow_a)  # ~1900x vs ~10x
        assert drop_low > 3 * drop_high, (drop_low, drop_high)  # steepest near p=0
        print("✅ you got it")
    except AssertionError as bad:
        print("❌ not yet — expected MSE 0.167, crowd MSE ~476 vs MAE ~5.2, and"
              " clamped cross-entropy 16.1, got", bad)
        raise
