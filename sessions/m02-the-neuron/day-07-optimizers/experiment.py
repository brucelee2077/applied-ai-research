# day-07-optimizers — experiment
#
# Today's spine: "a ball rolling downhill, three smarter ways."
# We drop THREE optimizers onto the SAME long, narrow valley and count
# how many steps each one needs to reach the bottom (loss < 0.01).
#
# The valley:  loss = 0.5 * (4*wx**2 + 0.2*wy**2)
#   - it is STEEP across the wx direction (the 4)  -> big slope
#   - it is GENTLE along the wy direction (the 0.2) -> tiny slope
# That mismatch is exactly the "narrow ravine" where plain SGD crawls.
#
# The slopes (gradients) of that valley are just the derivatives:
#   gx = 4  * wx     (slope in the wx direction)
#   gy = 0.2 * wy    (slope in the wy direction)
#
# All three racers start at the SAME spot and use the SAME learning rate
# for the same number of steps, so it is a fair race.
#
# POKE ME: the four knobs below are meant to be changed. The self-check only
# runs while they are at their defaults, so experimenting never fails.

import math  # we only need sqrt for the adaptive optimizer

# ---- the knobs (change these and re-run!) ----------------------------------
LR = 0.3       # the shared learning rate — same for all three, for a fair race
START = 1.5    # where every racer begins (wx = wy = START)
KEEP = 0.6     # Momentum's "keep this fraction of past speed"
STEPS = 30     # how many steps each optimizer gets
# ----------------------------------------------------------------------------

DEFAULTS = (0.3, 1.5, 0.6, 30)   # the settings the lesson's expected numbers assume


def loss(wx, wy):
    # "How wrong am I?" — the height of the ball on the valley.
    return 0.5 * (4 * wx ** 2 + 0.2 * wy ** 2)


def grads(wx, wy):
    # The slope under the ball in each direction (the gradient).
    gx = 4 * wx      # steep direction
    gy = 0.2 * wy    # gentle direction
    return gx, gy


def run_plain_sgd(lr, steps, start):
    """Plain SGD: read the slope, slide straight downhill. No memory."""
    wx, wy = start, start         # start at the top-corner of the valley
    first_below = None            # the first step where loss < 0.01
    for step in range(1, steps + 1):
        gx, gy = grads(wx, wy)    # look at the slope right under the ball
        wx -= lr * gx             # slide downhill in x by lr * slope
        wy -= lr * gy             # slide downhill in y by lr * slope
        # remember the FIRST step we cross below 0.01
        if loss(wx, wy) < 0.01 and first_below is None:
            first_below = step
    return first_below, loss(wx, wy)


def run_momentum(lr, steps, start, keep):
    """Momentum: keep a running velocity (memory of recent slopes), then step
    by the velocity. Same v = keep*v + slope rule as the lesson demo.

    Note on `keep`: the lesson's demo used the common 0.9, but this valley is
    very steep across wx. Try keep=0.9 and watch Momentum OVERSHOOT and finish
    slower than plain SGD — that is a real failure mode, not a bug."""
    wx, wy = start, start
    vx, vy = 0.0, 0.0             # the ball starts at rest (no speed yet)
    first_below = None
    for step in range(1, steps + 1):
        gx, gy = grads(wx, wy)
        vx = keep * vx + gx       # keep some past speed, add this slope
        vy = keep * vy + gy
        wx -= lr * vx             # step by the built-up velocity, not the raw slope
        wy -= lr * vy
        if loss(wx, wy) < 0.01 and first_below is None:
            first_below = step
    return first_below, loss(wx, wy)


def run_adaptive(lr, steps, start):
    """RMSProp-style: give EACH weight its own stride by dividing its step
    by the square root of a running average of that weight's squared slopes.

    Watch out for one coincidence: on the VERY first step the running average
    is just 0.1*g*g, so the step is lr/sqrt(0.1) ~= 0.95 in EVERY direction, no
    matter how steep the ground is. Start at 1.0 and it lands near the bottom
    in one move by luck. Change START and that luck disappears."""
    wx, wy = start, start
    sqx, sqy = 0.0, 0.0           # running average of each weight's squared slope
    first_below = None
    for step in range(1, steps + 1):
        gx, gy = grads(wx, wy)
        sqx = 0.9 * sqx + 0.1 * gx * gx   # remember how big this weight's slopes are
        sqy = 0.9 * sqy + 0.1 * gy * gy
        # divide each step by that weight's own slope size -> a well-sized stride
        wx -= lr * gx / (math.sqrt(sqx) + 1e-8)
        wy -= lr * gy / (math.sqrt(sqy) + 1e-8)
        if loss(wx, wy) < 0.01 and first_below is None:
            first_below = step
    return first_below, loss(wx, wy)


def show(step):
    # A run that never crosses 0.01 has first_below = None — say so, don't crash.
    return 'never' if step is None else 'step %2d' % step


if __name__ == "__main__":
    # Run the race: plain -> speedy -> per-weight.
    plain_step, plain_loss = run_plain_sgd(LR, STEPS, START)
    mom_step, mom_loss = run_momentum(LR, STEPS, START, KEEP)
    adap_step, adap_loss = run_adaptive(LR, STEPS, START)

    # Print the table an interviewer could read off in one glance.
    print(f"Valley 0.5*(4*wx^2 + 0.2*wy^2) | start={START} lr={LR} keep={KEEP} steps={STEPS}")
    print("First step where loss < 0.01:")
    print(f"  {'plain SGD':<18s}: {show(plain_step):>8s}   (loss@{STEPS} = {plain_loss:.2e})")
    print(f"  {'Momentum':<18s}: {show(mom_step):>8s}   (loss@{STEPS} = {mom_loss:.2e})")
    print(f"  {'Adaptive (RMSProp)':<18s}: {show(adap_step):>8s}   (loss@{STEPS} = {adap_loss:.2e})")

    got = (plain_step, mom_step, adap_step)
    if (LR, START, KEEP, STEPS) != DEFAULTS:
        # The learner is experimenting — report, never assert.
        print("🔧 knobs changed from the defaults — no self-check, just read the numbers above.")
        print("   (plain -> speedy -> per-weight is the usual order, but a bad knob can flip it,")
        print("    and that flip is exactly the thing worth understanding.)")
    else:
        # The lesson's stated "what you should see": 26 -> 13 -> 4.
        expected = (26, 13, 4)
        if got == expected:
            # The order must also go plain -> speedy -> per-weight, each sooner.
            assert plain_step > mom_step > adap_step, "order should be plain > momentum > adaptive"
            print("✅ you got it — plain 26, Momentum 13, adaptive 4: each one sooner.")
        else:
            print(f"❌ not yet — expected {expected}, got {got}")
            # A failing assert makes the self-check exit non-zero if the numbers drift.
            assert got == expected, f"expected {expected}, got {got}"
