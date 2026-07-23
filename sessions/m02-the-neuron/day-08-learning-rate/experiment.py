# day-08-learning-rate - experiment
#
# The one knob: the LEARNING RATE — how far you hop downhill each step.
# We train the same tiny model at four learning rates and watch the story change:
#   crawl (too small)  ->  just right  ->  partway crawl  ->  bounce/blow up (too big).
#
# The toy valley is the simplest one possible:
#   loss     = weight ** 2      (a U-shaped bowl; the bottom is at weight = 0)
#   gradient = 2 * weight       (the slope of that bowl: which way is uphill, how steep)
# Every run starts at weight = 2.0 and takes 40 steps. The ONLY thing we change
# between runs is the learning rate. This is the whole spine of Day 8 in one file.

import math  # only used to check for NaN when we push the rate too far in the bonus


# Run 40 downhill steps at one learning rate and return the FINAL loss.
def train(lr, steps=40, start=2.0):
    # start high up the left wall of the bowl, at weight = 2.0
    weight = start
    # take `steps` hops downhill, each hop scaled by the learning rate
    for _ in range(steps):
        # the gradient (slope) at this weight is 2 * weight
        gradient = 2 * weight
        # THE update rule: new weight = old weight - learning rate * gradient
        # the minus sign turns us to face downhill; lr sets how far we hop
        weight = weight - lr * gradient
    # loss is weight squared; the smaller, the closer we got to the bottom (0)
    return weight ** 2


# Return the loss AFTER EACH step, so we can see the shape of the loss curve.
def loss_curve(lr, steps=40, start=2.0):
    weight = start
    curve = []  # will hold the loss after every hop
    for _ in range(steps):
        weight = weight - lr * (2 * weight)  # one downhill hop
        curve.append(weight ** 2)            # record how wrong we still are
    return curve


# BONUS: a learning-rate SCHEDULE. Start the rate big (0.8) and HALVE it each
# step, but never let it drop below a 0.1 floor. Big hops early, small steady
# hops late — a gentle, precise landing instead of a bounce.
def train_with_schedule(steps=40, start=2.0, start_lr=0.8, floor=0.1):
    weight = start
    lr = start_lr
    for _ in range(steps):
        weight = weight - lr * (2 * weight)   # hop downhill at the current rate
        lr = max(floor, lr * 0.5)             # halve the rate, but never below the floor
    return weight ** 2, lr  # final loss AND the rate we settled at


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # The power-of-ten sweep: four rates, four stories.
    # -----------------------------------------------------------------
    rates = [1.0, 0.1, 0.01, 0.001]
    finals = {lr: train(lr) for lr in rates}  # final loss for each rate

    # print the sweep result, one line per rate
    print("Learning-rate sweep (loss = weight**2, start 2.0, 40 steps, bottom = 0):")
    for lr in rates:
        print("  lr = %-6s -> final loss = %.8g" % (lr, finals[lr]))

    # show the per-step loss for the two most telling rates
    print("\nlr = 1.0 loss per step (too big -> stuck bouncing at 4):",
          [round(x, 3) for x in loss_curve(1.0)[:6]], "...")
    print("lr = 0.1 loss per step (just right -> drops to ~0):",
          ["%.4g" % x for x in loss_curve(0.1)[:6]], "...")

    # bonus: the halving schedule with a 0.1 floor
    sched_loss, sched_lr = train_with_schedule()
    print("\nSchedule (start 0.8, halve, floor 0.1): settled rate = %.3g, "
          "final loss = %.8g" % (sched_lr, sched_loss))

    # -----------------------------------------------------------------
    # Self-check: the four final losses must match what the lesson says.
    # (These are the exact "what you should see" numbers from Day 8.)
    # -----------------------------------------------------------------
    # lr = 1.0 -> stuck at loss 4.0 (the weight just flips +2 <-> -2 forever)
    expected_big = 4.0
    # lr = 0.1 -> drops smoothly to almost zero (~7.07e-08)
    expected_good = 7.067388259113542e-08
    # lr = 0.01 -> crawls only partway, ~0.79
    expected_crawl = 0.7945954003281637
    # lr = 0.001 -> barely moves, ~3.41 (only a little below its start of 4.0)
    expected_frozen = 3.4080290993287403

    ok = (
        abs(finals[1.0]   - expected_big)    < 1e-9  and
        abs(finals[0.1]   - expected_good)   < 1e-12 and
        abs(finals[0.01]  - expected_crawl)  < 1e-9  and
        abs(finals[0.001] - expected_frozen) < 1e-9  and
        # the schedule must settle exactly at the 0.1 floor
        abs(sched_lr - 0.1) < 1e-9 and
        # ...and land far more gently than the too-big rate (loss well under 1)
        sched_loss < 1.0 and
        # bonus sanity: a slightly-too-big rate blows up to a huge/NaN loss
        (train(1.1, steps=60) > 1e6 or math.isnan(train(1.1, steps=60)))
    )

    if ok:
        print("\n✅ you got it — the sweep walks explode -> just-right -> crawl -> frozen,"
              " and the schedule settles at the 0.1 floor.")
    else:
        print("\n❌ not yet — expected lr=1.0->4.0, lr=0.1->~7.07e-08, "
              "lr=0.01->~0.79, lr=0.001->~3.41, and schedule rate -> 0.1")

    # a hard assert so the script FAILS loudly (non-zero exit) if the numbers drift
    assert ok, "learning-rate sweep did not match the lesson's expected values"
