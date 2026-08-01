# day-06-training-loop - experiment
#
# Today's artifact: run a REAL training loop for ONE neuron and watch the loss fall.
# The loop has four steps, on repeat:  forward -> loss -> backward -> update.
# We run it three times to FEEL the learning rate:
#   lr = 0.01     -> healthy: loss falls fast, then glides, then flattens near zero (converges)
#   lr = 1.5      -> too high: each step overshoots, so the PREDICTION flips sign every loop
#                              (0, 15, -195, 2745, ...) while the loss explodes (diverges)
#   lr = 0.00005  -> too low : each nudge is a grain of sand, so the loss barely moves (crawls)
#
# "healthy" / "too high" / "too low" are verdicts about THIS problem, not about the
# numbers themselves: they depend on the loss surface's curvature and on the step
# budget. The block near the bottom MEASURES that, and shows the same 0.01 earning a
# different verdict on Day 8's gentler bowl (where 0.01 is labelled a crawl).
#
# One neuron here is just: pred = w*x + b — a BEND-LESS neuron on purpose, so the four
# loop steps are the only thing on screen. Day 5's Rule 1 (the activation charges a
# toll: delta = incoming * activation_slope) is still the law; here the activation is
# the identity, whose slope is 1, so the toll is 1 and drops out of grad_w / grad_b.
# Put a real bend back and the toll returns — the probe at the bottom does exactly that.

# numpy holds the loss curve as an array so we can measure it (min, last) cleanly.
# It also lets the overflow probe at the bottom reach inf/NaN instead of crashing.
import numpy as np

# One shared setup, used by all three runs and by every check below.
X = 2.0       # the single input this neuron sees
TARGET = 1.0  # the number we want pred to become


def train_one_neuron(lr, steps=50, x=X, target=TARGET, w=0.0, b=0.0):
    """Run the four-step training loop. Returns (losses, preds, w, b).

    losses[i] / preds[i] are recorded at the START of loop i, BOTH parameters are
    returned so the checks can look at w and b separately -- the bias is half the
    model, and a curve alone cannot tell the two of them apart.

    `steps` is the same knob Day 7 spells STEPS and Day 8 spells steps: one step =
    one weight update. It is NOT an epoch (Day 9's word for one full pass over a
    training split) -- this neuron sees a single example, so the two coincide here.
    """
    # We record the loss AND the prediction at the START of each loop, so we can
    # watch the curve fall (healthy) or the guess flip sign (too high) later.
    losses = []
    preds = []
    # Repeat the loop `steps` times. Each pass is ONE iteration (one weight update).
    for _ in range(steps):
        # Step 1 - FORWARD PASS: the neuron takes its shot (a weighted sum plus bias).
        # `pred` is the same object Day 3 printed as yhat (ŷ) — one prediction.
        pred = w * x + b
        # Step 2 - LOSS: measure the miss with one number. Squared error: (guess - truth)^2.
        loss = (pred - target) ** 2
        # Remember this loop's guess and loss so we can inspect both curves.
        preds.append(pred)
        losses.append(loss)
        # Step 3 - BACKWARD PASS: which way is each weight off? These are the gradients.
        # For loss = (w*x + b - target)^2, calculus gives these two slopes. Day 5's Rule 1
        # toll is present but invisible: this neuron's activation is the identity, whose
        # slope is 1, so delta = incoming * 1 = 2*(pred - target). With a real bend the
        # toll is NOT 1 (see the ReLU-toll probe below).
        grad_w = 2 * (pred - target) * x  # how the loss changes if we nudge w
        grad_b = 2 * (pred - target)      # how the loss changes if we nudge b
        # Step 4 - UPDATE: nudge each weight a small step DOWNHILL.
        # The gradient points uphill (toward more loss), so the MINUS walks us downhill.
        w = w - lr * grad_w
        b = b - lr * grad_b
    # Return both curves as numpy arrays, plus the two parameters we ended up with.
    return np.array(losses), np.array(preds), w, b


def train_with_relu_toll(lr, steps=50, x=X, target=TARGET, w=0.0, b=0.0):
    """The SAME loop with Day 5's Rule 1 put back: a ReLU bend, so the toll can be 0.

    Identical to train_one_neuron except that pred = relu(w*x + b) and every gradient
    is multiplied by the ReLU's local slope (1 if z > 0, else 0). Start at w = b = 0
    and z = 0 is not > 0, so the toll is 0, nothing moves, and the loss never falls --
    the dead unit Day 2 warned about, inside Day 6's loop.
    """
    losses = []
    for _ in range(steps):
        z = w * x + b                       # the pre-activation Day 5 saves
        pred = max(0.0, z)                  # the bend
        losses.append((pred - target) ** 2)
        incoming = 2 * (pred - target)
        relu_slope = 1.0 if z > 0 else 0.0  # Rule 1: the activation's toll
        delta = incoming * relu_slope
        w = w - lr * delta * x
        b = b - lr * delta
    return np.array(losses), w, b


if __name__ == "__main__":
    # ---- Run 1: a sensible learning rate. The loss should slide down toward 0. ----
    # lr = 0.01 leaves a real CURVE to watch: 1.0 -> 0.12 -> 0.015 -> 0.0018 -> 0.0002.
    healthy, healthy_preds, w_healthy, b_healthy = train_one_neuron(lr=0.01)
    # Print the curve every 10 loops so we can WATCH it slide downhill.
    print("lr = 0.01  (healthy, should converge):")
    for i in range(0, len(healthy), 10):
        print(f"  loop {i:2d}: loss = {healthy[i]:.6f}")
    final_healthy = healthy[-1]
    print(f"  final loss = {final_healthy:.6g}  (started at {healthy[0]:.6g})")
    # Show WHICH parameters got there. w and b are both learned; a loss curve alone
    # cannot tell you whether the bias did any work, so print (and pin) them both.
    print(f"  learned w = {w_healthy:.6f}, b = {b_healthy:.6f}"
          f"  ->  pred = {w_healthy * X + b_healthy:.6f}  (target {TARGET:g})")
    print()

    # ---- Run 2: learning rate WAY too high. Each step overshoots -> loss explodes. ----
    too_high, high_preds, _, _ = train_one_neuron(lr=1.5)
    final_high = too_high[-1]
    # The signed miss each lap. This is the thing that flips sign; the loss (its
    # square) can only grow, so the loss alone cannot show you the overshoot.
    high_misses = high_preds - TARGET
    high_signs = np.sign(high_misses)
    # How many of the 49 lap-to-lap steps changed sign? (49 = every single one.)
    sign_flips = int(np.sum(high_signs[1:] != high_signs[:-1]))
    # By how much is the miss multiplied each lap? Negative => it jumped past the target.
    miss_ratios = high_misses[1:] / high_misses[:-1]
    print("lr = 1.5  (too high -> diverges / blows up):")
    print("  prediction, lap by lap: "
          + " -> ".join(f"{p:g}" for p in high_preds[:5])
          + " -> ...   <- it flips sign EVERY lap")
    print(f"  the miss is multiplied by {miss_ratios[0]:g} each lap"
          f"  ->  {sign_flips} of {len(high_preds) - 1} steps flip sign")
    print(f"  final loss = {final_high:.6g}   <- astronomical: the too-high failure, live")
    print("  (the error flips sign every loop and grows -> keep looping and it becomes inf / NaN)")

    # Keep looping and it becomes inf / NaN -- that is a claim, so measure it instead
    # of asserting it in prose. numpy floats saturate to inf where Python floats raise.
    with np.errstate(over="ignore", invalid="ignore"):
        long_high, _, _, _ = train_one_neuron(
            lr=1.5, steps=300, w=np.float64(0.0), b=np.float64(0.0))
    saw_inf, saw_nan = np.isinf(long_high), np.isnan(long_high)
    first_inf = int(np.argmax(saw_inf)) if saw_inf.any() else -1
    first_nan = int(np.argmax(saw_nan)) if saw_nan.any() else -1
    print(f"  measured over 300 laps: loss first becomes inf on lap {first_inf}, "
          f"then NaN on lap {first_nan}\n")

    # ---- Run 3: learning rate too low. Each nudge is tiny -> loss barely moves. ----
    too_low, _, _, _ = train_one_neuron(lr=0.00005)
    final_low = too_low[-1]
    print("lr = 0.00005  (too low -> crawls):")
    print(f"  start loss = {too_low[0]:.6f},  final loss = {final_low:.6f}")
    print("  the loss barely moved after 50 loops -> the too-low failure\n")

    # ---- Why "healthy" is RELATIVE: a rate is judged against a CURVATURE ----------
    # One lap multiplies the MISS (pred - target) by (1 - lr * C), where C is this
    # problem's curvature. Here both w and b move, so C = 2 * (x*x + 1) = 10, the run is
    # stable only while |1 - lr*C| < 1 (that is lr < 2/C = 0.2), and 0.01 lands at a
    # shrink of 0.9 per lap -> "healthy" AT THIS curvature. Day 8's bowl (loss =
    # weight**2, gradient 2*weight) has C = 2 and a limit of lr < 1, so the SAME 0.01
    # shrinks by 0.98 there and earns the label "crawl". Same number, different problem.
    curvature = 2 * (X * X + 1)
    formula_shrink = 1 - 0.01 * curvature          # what the formula predicts here
    measured_shrink = (healthy_preds[1] - TARGET) / (healthy_preds[0] - TARGET)
    day08_bowl_shrink = 1 - 0.01 * 2               # the same rate on Day 8's C = 2 bowl
    print(f"curvature here C = 2*(x*x + 1) = {curvature:g}"
          f"  ->  stable only while lr < 2/C = {2 / curvature:g}")
    print(f"  lr = 0.01: measured shrink per lap = {measured_shrink:g}"
          f" (formula 1 - lr*C = {formula_shrink:g})"
          f"  vs {day08_bowl_shrink:g} on Day 8's C = 2 bowl -> 'crawl' there")
    # Exactly ON the limit the miss flips sign at the SAME size forever: never diverges,
    # never converges. Without this row the "lr < 0.2" claim would be untested.
    on_limit, on_limit_preds, _, _ = train_one_neuron(lr=2 / curvature)
    print(f"  lr = {2 / curvature:g} (exactly 2/C): loss stays {on_limit[-1]:g} for all"
          f" {len(on_limit)} laps while the guess flips {on_limit_preds[1]:g} /"
          f" {on_limit_preds[2]:g} — on the line, neither converging nor exploding")
    # The two rates Day 8 blesses, run HERE. 0.1 makes 1 - lr*C exactly 0, so this
    # neuron lands on the target in ONE lap; 1.0 only bounces harmlessly in that bowl.
    day08_good, _, _, _ = train_one_neuron(lr=0.1)
    day08_big, _, _, _ = train_one_neuron(lr=1.0)
    print(f"  Day 8's 'just right' lr = 0.1 here: shrink {1 - 0.1 * curvature:g}"
          f" -> loss {day08_good[1]:g} after ONE lap (dead on the target)")
    print(f"  Day 8's lr = 1.0 (stuck at loss 4 in that bowl) here: final loss"
          f" {day08_big[-1]:.6g}  <- the same number, the opposite verdict\n")

    # ---- Day 5's Rule 1, put back: with a real bend the toll can be ZERO ----------
    # This day's neuron has no activation, so the toll is 1 and vanishes from the two
    # gradients above. Re-run the SAME loop with a ReLU and the toll returns: starting at
    # w = b = 0, z = 0 is not > 0, the slope is 0, and nothing ever moves.
    dead_losses, dead_w, dead_b = train_with_relu_toll(lr=0.01)
    live_losses, _, _ = train_with_relu_toll(lr=0.01, w=0.6)
    print("with Day 5's ReLU toll (same loop, same lr = 0.01):")
    print(f"  start w = 0, b = 0 -> z = 0, toll = 0: loss stays {dead_losses[-1]:g} for all"
          f" {len(dead_losses)} laps, w = {dead_w:g}, b = {dead_b:g} (a dead unit)")
    print(f"  start w = 0.6      -> z = 1.2, toll = 1: loss falls {live_losses[0]:.6g}"
          f" -> {live_losses[-1]:.6g}  <- the toll, not the loop, is the difference\n")

    # ---- Self-check: assert each run behaved exactly as the lesson says it should. ----
    # Loose bands ("final loss < 1e-3", "> 1e6") pass for almost any descent-shaped
    # update on this 1-D problem, so each printed number is PINNED to the value the
    # lesson quotes instead. x = 2 makes 2*x*x = 8 = 2*x + 4, which is exactly why
    # w and b must be pinned separately: swapping the two gradients, or dropping the
    # bias entirely, still converges and still looks fine from the loss curve alone.
    try:
        # 1) Healthy run: every printed point on the curve, pinned.
        assert healthy[0] == 1.0, f"curve must start at 1.0, got {healthy[0]}"
        assert abs(healthy[10] - 0.12157665459056932) < 1e-12, f"loop 10: {healthy[10]}"
        assert abs(healthy[20] - 0.01478088294143463) < 1e-12, f"loop 20: {healthy[20]}"
        assert abs(healthy[30] - 0.00179701029991444) < 1e-12, f"loop 30: {healthy[30]}"
        assert abs(healthy[40] - 0.00021847450052839) < 1e-12, f"loop 40: {healthy[40]}"
        assert abs(final_healthy - 3.279185047850316e-05) < 1e-15, f"final: {final_healthy}"
        # 2) Healthy run: BOTH parameters did their share of the work.
        assert abs(w_healthy - 0.397938489917072) < 1e-12, f"learned w: {w_healthy}"
        assert abs(b_healthy - 0.198969244958536) < 1e-12, f"learned b: {b_healthy}"
        # 3) Too-high run: the guess overshoots and flips sign every single lap.
        assert list(high_preds[:4]) == [0.0, 15.0, -195.0, 2745.0], \
            f"the guess should swing 0 -> 15 -> -195 -> 2745, got {list(high_preds[:4])}"
        assert sign_flips == len(high_preds) - 1, \
            f"every lap should flip sign, only {sign_flips} of {len(high_preds) - 1} did"
        assert np.max(np.abs(miss_ratios + 14.0)) < 1e-9, \
            f"the miss should be multiplied by -14 each lap, got {miss_ratios[0]}"
        # 4) Too-high run: the magnitude, pinned. 1e6 leaves 100 decades of slack, so
        #    pin the exponent: log10(2.09e112) = 112.3205. lr = 1.2 lands on 102.06.
        assert abs(np.log10(final_high) - 112.32054749646733) < 1e-6, \
            f"too-high run should reach ~1e112, got {final_high}"
        # 5) Too-high run: keep looping and it really does hit inf, then NaN.
        assert first_inf == 135, f"loss should first hit inf on lap 135, got {first_inf}"
        assert first_nan == 271, f"loss should first hit NaN on lap 271, got {first_nan}"
        # 6) Too-low run: after 50 loops it has barely left its starting loss of 1.0.
        assert too_low[0] == 1.0, f"crawl must start at 1.0, got {too_low[0]}"
        assert abs(final_low - 0.9521694616616252) < 1e-12, \
            f"too-low run should crawl to 0.952169, got {final_low}"
        # 7) The curvature block: the verdict words are relative, and the numbers say so.
        assert curvature == 10.0, f"C = 2*(x*x+1) should be 10, got {curvature}"
        assert abs(measured_shrink - 0.9) < 1e-12, \
            f"the measured shrink per lap at lr = 0.01 should be 0.9, got {measured_shrink}"
        assert abs(measured_shrink - formula_shrink) < 1e-12, \
            "the measured shrink must agree with 1 - lr*C, or C is the wrong curvature"
        assert abs(day08_bowl_shrink - 0.98) < 1e-12, \
            f"the same 0.01 on Day 8's C = 2 bowl shrinks by 0.98, got {day08_bowl_shrink}"
        # exactly on the limit: the miss keeps its size and flips sign, so loss == 1.0 flat
        assert np.all(on_limit == 1.0), \
            f"at lr = 2/C the loss should stay 1.0 on every lap, got {on_limit[:3]}"
        assert list(on_limit_preds[:3]) == [0.0, 2.0, 0.0], \
            f"at lr = 2/C the guess should flip 0 -> 2 -> 0, got {list(on_limit_preds[:3])}"
        # Day 8's two blessed rates, judged by THIS problem
        assert day08_good[1] == 0.0, \
            f"lr = 0.1 gives 1 - lr*C = 0, so lap 1's loss is exactly 0, got {day08_good[1]}"
        assert abs(np.log10(day08_big[-1]) - 93.51576592505384) < 1e-6, \
            f"lr = 1.0 should blow up to ~1e93 on this neuron, got {day08_big[-1]}"
        # 8) Day 5's toll, back in the loop: dead at z = 0, alive at z = 1.2.
        assert np.all(dead_losses == 1.0) and dead_w == 0.0 and dead_b == 0.0, \
            "with the ReLU toll and z = 0 nothing may move: loss flat at 1.0, w = b = 0"
        assert abs(live_losses[0] - 0.04) < 1e-12, \
            f"the live probe must start at loss 0.04 (z = 1.2), got {live_losses[0]}"
        # The printed end of the live probe, pinned two-sided to the value on screen.
        # "< 1e-5" left under one decade of headroom above the real 1.31167e-06, so
        # almost any wrong step size still slipped through it; this pins the number.
        assert abs(live_losses[-1] - 1.3116740191408385e-06) < 1e-15, \
            f"the live probe must end at loss 1.31167e-06, got {live_losses[-1]}"
        # and it must really have CONVERGED relative to the dead unit beside it
        assert live_losses[-1] < live_losses[0] / 1000.0 < dead_losses[-1], \
            f"with the toll open (z > 0) the same loop must converge, got {live_losses[-1]}"
        print("✅ you got it")
    except AssertionError as e:
        # If any check fails, say what we expected so it's easy to fix.
        print("❌ not yet — expected the healthy curve to read exactly "
              "1.0 / 0.121577 / 0.014781 / 0.001797 / 0.000218 / 3.27919e-05 with "
              "w = 0.397938 and b = 0.198969; the lr = 1.5 guess to swing "
              "0 -> 15 -> -195 -> 2745 (sign flipping every lap, miss x -14) to a final "
              "loss of ~1e112, hitting inf on lap 135 and NaN on lap 271; the "
              "lr = 0.00005 loss to crawl from 1.0 to 0.952169; a curvature C = 10 with "
              "a measured shrink of 0.9 at lr = 0.01 (0.98 on Day 8's C = 2 bowl), a flat "
              "1.0 exactly at lr = 0.2 and ~1e93 at Day 8's lr = 1.0; and the ReLU-toll "
              "probe to sit dead at 1.0 from w = 0 but converge from w = 0.6")
        print(f"   first failing check: {e}")
        raise
