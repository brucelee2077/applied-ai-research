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
# Read that valley as a LOSS SURFACE in weight space — the height a loss would trace
# out as two weights move — not as a loss in Day 4's sense: there is no data, no
# prediction and no target here, so the height is not "how wrong a guess is". Its
# bottom happens to sit at wx = wy = 0 because of how this surface was written down;
# on a real model the bottom sits wherever the predictions match the answers.
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
# LR and STEPS are the same two knobs Day 6 and Day 8 pass as lr and steps; they are
# spelled in capitals here only because the lesson asks for editable module constants.
LR = 0.3       # the shared learning rate — same for all three, for a fair race
START = 1.5    # where every racer begins (wx = wy = START)
KEEP = 0.6     # Momentum's "keep this fraction of past speed"
STEPS = 30     # how many steps each optimizer gets — one step = one weight update
# ----------------------------------------------------------------------------

DEFAULTS = (0.3, 1.5, 0.6, 30)   # the settings the lesson's expected numbers assume

# The finish line for the race, in LOSS-HEIGHT units. It is NOT a target answer: Days
# 4-6 use "target" for the true value a prediction should match, and nothing here
# predicts anything, so this constant is called BAR.
BAR = 0.01   # "reached the bottom" means the loss got STRICTLY under this bar


def loss(wx, wy):
    # The height of the ball on this surface — how high above the bottom these two
    # weights sit. (Not Day 4's "how wrong is my guess": there is no guess here.)
    return 0.5 * (4 * wx ** 2 + 0.2 * wy ** 2)


def reached(height):
    # One home for the bar, so all three racers are judged by the same rule
    # and that rule can be tested on its own (see the boundary probe below).
    # Strict <: sitting exactly ON 0.01 is not yet at the bottom.
    return height < BAR


def fmt(height):
    # ONE formatter for the printed table AND the self-check, so the numbers the
    # learner reads are literally the numbers being pinned.
    return "%.2e" % height


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
        # remember the FIRST step we cross below the bar
        if reached(loss(wx, wy)) and first_below is None:
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
        if reached(loss(wx, wy)) and first_below is None:
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
        if reached(loss(wx, wy)) and first_below is None:
            first_below = step
    return first_below, loss(wx, wy)


def show(step):
    # A run that never crosses 0.01 has first_below = None — say so, don't crash.
    # This is the ONLY place a step number becomes text, and the self-check pins the
    # text this returns — not the raw number behind it. Otherwise the table could
    # print "step 27" while a check on the raw 26 still said ✅, and the three step
    # counts ARE the whole day.
    return 'never' if step is None else 'step %2d' % step


def plain_step_factors(lr, start):
    """ONE plain-SGD step, measured: what fraction of itself does each weight keep?

    Steep wx keeps 1 - lr*4, gentle wy keeps 1 - lr*0.2. This is the line that ties
    the shared learning rate to a BEHAVIOUR instead of leaving LR a number nothing
    claims anything about. At lr = 0.3 the steep factor is NEGATIVE (-0.200): plain
    SGD already hops PAST the bottom in wx and lands on the other side, while wy
    keeps 0.940 of itself — it barely budges. That gap IS the ravine."""
    wx, wy = start, start
    gx, gy = grads(wx, wy)
    return (wx - lr * gx) / start, (wy - lr * gy) / start


if __name__ == "__main__":
    # Run the race: plain -> speedy -> per-weight.
    plain_step, plain_loss = run_plain_sgd(LR, STEPS, START)
    mom_step, mom_loss = run_momentum(LR, STEPS, START, KEEP)
    adap_step, adap_loss = run_adaptive(LR, STEPS, START)

    # Print the table an interviewer could read off in one glance. The three cells are
    # rendered ONCE, here, and the self-check below pins these very strings.
    cells = (show(plain_step), show(mom_step), show(adap_step))
    print(f"Valley 0.5*(4*wx^2 + 0.2*wy^2) | start={START} lr={LR} keep={KEEP} steps={STEPS}")
    print(f"First step where loss < {BAR}:")
    print(f"  {'plain SGD':<18s}: {cells[0]:>8s}   (loss@{STEPS} = {fmt(plain_loss)})")
    print(f"  {'Momentum':<18s}: {cells[1]:>8s}   (loss@{STEPS} = {fmt(mom_loss)})")
    print(f"  {'Adaptive (RMSProp)':<18s}: {cells[2]:>8s}   (loss@{STEPS} = {fmt(adap_loss)})")

    # What does the shared learning rate DO? One plain step, measured.
    fx, fy = plain_step_factors(LR, START)
    factor_cells = ("%+.3f" % fx, "%+.3f" % fy)
    print(f"One plain step at lr={LR}: wx keeps {factor_cells[0]} of itself,"
          f" wy keeps {factor_cells[1]} — and a minus sign means that weight hopped PAST"
          f" the bottom onto the far wall. That gap between the two is the ravine.")

    # And is that rate really SHARED? Halve it and every racer must slow down. A racer
    # that quietly hardcoded 0.3 instead of reading `lr` would keep its number here.
    half_lr = LR / 2
    half_cells = (show(run_plain_sgd(half_lr, STEPS, START)[0]),
                  show(run_momentum(half_lr, STEPS, START, KEEP)[0]),
                  show(run_adaptive(half_lr, STEPS, START)[0]))
    print(f"Same race at half the rate (lr={half_lr:g}): plain {half_cells[0].strip()},"
          f" Momentum {half_cells[1].strip()}, adaptive {half_cells[2].strip()}"
          f" — smaller hops, later finishes, and plain SGD runs out of steps.")


    # No racer ever lands EXACTLY on the bar, so the bar's strictness would go
    # untested. Probe it directly, on the boundary and one hair under it.
    on_bar, under_bar = reached(BAR), reached(BAR - 1e-6)
    print(f"Bar is strict: a loss of exactly {BAR} counts as reached? {on_bar}"
          f" | a loss of {BAR - 1e-6:g}? {under_bar}")

    if (LR, START, KEEP, STEPS) != DEFAULTS:
        # The learner is experimenting — report, never assert.
        print("🔧 knobs changed from the defaults — no self-check, just read the numbers above.")
        print("   (plain -> speedy -> per-weight is the usual order, but a bad knob can flip it,")
        print("    and that flip is exactly the thing worth understanding.)")
    else:
        # The lesson's stated "what you should see": 26 -> 13 -> 4 — pinned as the
        # CELLS the table printed, so the text on screen is the text being checked.
        expected_cells = ('step 26', 'step 13', 'step  4')
        # … and the loss column that makes the race quantitative. Both are quoted
        # to the learner, so both get pinned — against strings written down HERE,
        # never re-derived from the loops above.
        expected_losses = ('5.49e-03', '3.83e-07', '7.97e-28')
        got_losses = (fmt(plain_loss), fmt(mom_loss), fmt(adap_loss))
        steps_ok = cells == expected_cells
        losses_ok = got_losses == expected_losses
        # The bar itself: on-the-line is NOT reached, a hair under it is.
        bar_ok = (on_bar is False) and (under_bar is True)
        # LR earns its value twice: the one-step factors it produces (negative across
        # the steep direction, nearly 1 along the gentle one), and the fact that
        # halving it moves ALL THREE racers.
        expected_factors = ('-0.200', '+0.940')
        expected_half = ('never', 'step 20', 'step 10')
        factors_ok = factor_cells == expected_factors
        half_ok = half_cells == expected_half
        if steps_ok and losses_ok and bar_ok and factors_ok and half_ok:
            print("✅ you got it — plain 26, Momentum 13, adaptive 4: each one sooner.")
        else:
            print(f"❌ not yet — expected steps {expected_cells} with losses {expected_losses}"
                  f" and a strict bar (False, True); got steps {cells},"
                  f" losses {got_losses}, bar {(on_bar, under_bar)};"
                  f" expected one-step factors {expected_factors} and a half-rate race"
                  f" {expected_half}, got {factor_cells} and {half_cells}")
            # A failing assert makes the self-check exit non-zero if anything drifts.
            assert steps_ok, f"expected steps {expected_cells}, got {cells}"
            assert losses_ok, f"expected losses {expected_losses}, got {got_losses}"
            assert bar_ok, f"the bar must be strict: got {(on_bar, under_bar)}, want (False, True)"
            assert factors_ok, (f"one plain step at lr={LR} should keep {expected_factors}"
                               f" of (wx, wy), got {factor_cells}")
            assert half_ok, (f"at half the rate the race should read {expected_half},"
                             f" got {half_cells} — is every racer really using lr?")

