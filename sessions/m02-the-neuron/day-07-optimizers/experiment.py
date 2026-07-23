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
# All three racers start at the SAME spot (wx = wy = 1.0) and use the
# SAME learning rate (lr = 0.3) for 30 steps, so it is a fair race.

import math  # we only need sqrt for the adaptive optimizer


def loss(wx, wy):
    # "How wrong am I?" — the height of the ball on the valley.
    return 0.5 * (4 * wx ** 2 + 0.2 * wy ** 2)


def grads(wx, wy):
    # The slope under the ball in each direction (the gradient).
    gx = 4 * wx      # steep direction
    gy = 0.2 * wy    # gentle direction
    return gx, gy


def run_plain_sgd(lr, steps):
    """Plain SGD: read the slope, slide straight downhill. No memory."""
    wx, wy = 1.0, 1.0            # start at the top-corner of the valley
    first_below = None           # the first step where loss < 0.01
    for step in range(1, steps + 1):
        gx, gy = grads(wx, wy)   # look at the slope right under the ball
        wx -= lr * gx            # slide downhill in x by lr * slope
        wy -= lr * gy            # slide downhill in y by lr * slope
        # remember the FIRST step we cross below 0.01
        if loss(wx, wy) < 0.01 and first_below is None:
            first_below = step
    return first_below, loss(wx, wy)


def run_momentum(lr, steps, keep=0.6):
    """Momentum: keep a running velocity (memory of recent slopes),
    then step by the velocity. Same v = keep*v + slope rule as the lesson demo."""
    wx, wy = 1.0, 1.0
    vx, vy = 0.0, 0.0            # the ball starts at rest (no speed yet)
    first_below = None
    for step in range(1, steps + 1):
        gx, gy = grads(wx, wy)
        vx = keep * vx + gx      # keep 60% of past speed, add this slope
        vy = keep * vy + gy
        wx -= lr * vx            # step by the built-up velocity, not the raw slope
        wy -= lr * vy
        if loss(wx, wy) < 0.01 and first_below is None:
            first_below = step
    return first_below, loss(wx, wy)


def run_adaptive(lr, steps):
    """RMSProp-style: give EACH weight its own stride by dividing its step
    by the square root of a running average of that weight's squared slopes."""
    wx, wy = 1.0, 1.0
    sqx, sqy = 0.0, 0.0          # running average of each weight's squared slope
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


if __name__ == "__main__":
    LR = 0.3      # the shared learning rate — same for all three, for a fair race
    STEPS = 30    # how many steps each optimizer gets

    # Run the race: plain -> speedy -> per-weight.
    plain_step, plain_loss = run_plain_sgd(LR, STEPS)
    mom_step, mom_loss = run_momentum(LR, STEPS)
    adap_step, adap_loss = run_adaptive(LR, STEPS)

    # Print the table an interviewer could read off in one glance.
    print("Same valley, same lr = 0.3, 30 steps. First step where loss < 0.01:")
    print(f"  {'plain SGD':<18s}: step {plain_step:>2}   (loss@30 = {plain_loss:.2e})")
    print(f"  {'Momentum':<18s}: step {mom_step:>2}   (loss@30 = {mom_loss:.2e})")
    print(f"  {'Adaptive (RMSProp)':<18s}: step {adap_step:>2}   (loss@30 = {adap_loss:.2e})")

    # The lesson's stated "what you should see": 19 -> 8 -> 1.
    expected = (19, 8, 1)
    got = (plain_step, mom_step, adap_step)
    if got == expected:
        # The order must also go plain -> speedy -> per-weight, each sooner.
        assert plain_step > mom_step > adap_step, "order should be plain > momentum > adaptive"
        print("✅ you got it")
    else:
        print(f"❌ not yet — expected {expected}, got {got}")
        # A failing assert makes the self-check exit non-zero if the numbers drift.
        assert got == expected, f"expected {expected}, got {got}"
