# Module 2 · Day 4 — Loss Functions
#
# Today's big idea: a loss is ONE number that says how wrong a guess is.
# Lower is better; 0 means perfect. Training is just making that number smaller.
#
# We will watch two scorecards react:
#   1. MSE / MAE — the scores for guessing a NUMBER (and how one wild
#      outlier hijacks MSE but barely moves MAE).
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
    # A miss of 100 counts as 100, not 100*100 — so one outlier can't dominate.
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
    # ================= Part 1: MSE vs MAE, and the outlier trap =============
    pred = np.array([2.5, 0.0, 2.0])       # the network's three guesses
    target = np.array([3.0, -0.5, 2.0])    # the three true answers

    close_mse = round(float(mse(pred, target)), 3)   # a CLOSE guess
    close_mae = round(float(mae(pred, target)), 3)
    print("close guess  -> MSE", close_mse, " MAE", close_mae)

    # Now break the last guess into a wild outlier: 2.0 becomes 102.0.
    pred_wild = np.array([2.5, 0.0, 102.0])
    wild_mse = round(float(mse(pred_wild, target)), 1)   # MSE squares 100 -> huge
    wild_mae = round(float(mae(pred_wild, target)), 3)   # MAE only feels 100
    print("wild outlier -> MSE", wild_mse, " MAE", wild_mae)

    # ================= Part 2: cross-entropy rewards confidence =============
    # As p (probability on the CORRECT answer) climbs 0.1 -> 0.99, the loss falls.
    ce_values = [round(float(cross_entropy(p)), 3) for p in [0.99, 0.6, 0.1, 1e-7]]
    print("cross-entropy for p=[0.99, 0.6, 0.1, 1e-7]:", ce_values)

    # The p=0 case, clamped, stays FINITE (~16.1) instead of +infinity.
    ce_zero = round(float(cross_entropy(0.0)), 1)
    print("confident+wrong p=0 (clamped) -> cross-entropy", ce_zero, "(not +inf)")

    # ================= The honest limit =====================================
    # A loss only SCORES a guess (and hands over a slope). It never changes a
    # single weight — that is the optimizer's job, which you meet tomorrow (Day 5).
    print("reminder: a loss only SCORES; the optimizer (Day 5) changes weights.")

    # ================= Self-check against the lesson's stated numbers =======
    # The lesson says: MSE on the close guess is ~0.167, and the clamped p=0
    # cross-entropy is ~16.1. Check both, and check the outlier really hijacks MSE.
    try:
        assert close_mse == 0.167, close_mse            # "what you should see"
        assert ce_zero == 16.1, ce_zero                 # clip keeps it finite
        assert wild_mse > 1000, wild_mse                # outlier explodes MSE
        assert wild_mae < 40, wild_mae                  # but MAE barely moves
        print("✅ you got it")
    except AssertionError as bad:
        print("❌ not yet — expected MSE 0.167 and clamped cross-entropy 16.1, got", bad)
        raise
