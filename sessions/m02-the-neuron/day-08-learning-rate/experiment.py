# day-08-learning-rate - experiment
#
# The one knob: the LEARNING RATE — how far you hop downhill each step.
# We train the same tiny model at four learning rates and watch the story change:
#   crawl (too small)  ->  just right  ->  partway crawl  ->  bounce/blow up (too big).
#
# The toy valley is the simplest one possible. Read it as a LOSS SURFACE over one
# weight — the shape a loss traces out as a weight moves — not as a loss with data
# behind it: there is no prediction and no target anywhere in this file.
#   loss     = weight ** 2      (a U-shaped bowl whose bottom sits at weight = 0)
#   gradient = 2 * weight       (the slope of that bowl: which way is uphill, how steep)
# "The bottom is at weight = 0" is therefore a fact about THIS toy surface, not a rule
# of training: on Day 6's real neuron weight zero is the worst point, and the bottom is
# a whole line of (w, b) pairs. The block near the bottom of this file prints that.
#
# Every run starts at weight = 2.0 and takes 40 steps. The ONLY thing we change
# between runs is the learning rate. This is the whole spine of Day 8 in one file.
#
# One caution about the four verdicts below: "just right" and "crawl" are judged
# against THIS surface's curvature (C = 2) and this 40-step budget, never against the
# number alone. Day 6's neuron has C = 10, so the 0.01 that only crawls here is the
# rate that day calls healthy, and the 1.0 that merely bounces here explodes there.
# The curvature block after the sweep measures both sides of that.

import math  # only used to check for NaN when we push the rate too far in the bonus


# The ONE copy of the update rule in this file: the SIGNED weight after every hop.
# Everything below reads from here — the final loss, the loss curve, the measured
# per-step shrink — so the update cannot be right in one of them and wrong in another.
# Keeping the sign matters: loss = weight**2 throws it away, and "the weight flips
# sign" is one of the day's four headline claims.
def weight_path(lr, steps=40, start=2.0):
    # start high up the left wall of the bowl, at weight = 2.0
    weight = start
    path = []
    for _ in range(steps):
        # the gradient (slope) at this weight is 2 * weight
        gradient = 2 * weight
        # THE update rule: new weight = old weight - learning rate * gradient
        # the minus sign turns us to face downhill; lr sets how far we hop
        weight = weight - lr * gradient
        path.append(weight)   # keep the sign: at lr = 1.0 this flips +2 <-> -2
    return path


# Run 40 downhill steps at one learning rate and return the FINAL loss.
def train(lr, steps=40, start=2.0):
    # loss is weight squared; the smaller, the closer we got to this surface's bottom (0)
    return weight_path(lr, steps, start)[-1] ** 2


# Return the loss AFTER EACH step, so we can see the shape of the loss curve.
def loss_curve(lr, steps=40, start=2.0):
    # the same walk as train(), just reported at every hop instead of only the last
    return [weight ** 2 for weight in weight_path(lr, steps, start)]



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

    # Each sweep line is rendered ONCE, here, and the self-check at the bottom pins these
    # very strings. Rendering finals[lr] again at the print while the check only looked at
    # the raw float would be two expressions for one number: corrupt the printed one and
    # the learner reads a wrong final loss under a passing ✅.
    final_cells = ["lr = %-6s -> final loss = %.8g" % (lr, finals[lr]) for lr in rates]

    # print the sweep result, one line per rate
    print("Learning-rate sweep (loss = weight**2, start 2.0, 40 steps, bottom = 0):")
    for cell in final_cells:
        print("  " + cell)

    # show the per-step loss for the two most telling rates
    # (compute each path ONCE, so the numbers printed are the numbers checked below)
    path_big = weight_path(1.0)     # the bouncing rate: hops right past the bottom
    path_good = weight_path(0.1)    # the just-right rate: a smooth slide down
    curve_big = [w ** 2 for w in path_big]
    curve_good = [w ** 2 for w in path_good]
    # the six numbers each curve actually SHOWS, rounded once and pinned below as shown
    curve_big_cells = [round(x, 3) for x in curve_big[:6]]
    curve_good_cells = ["%.4g" % x for x in curve_good[:6]]
    # Each of these two lines is rendered ONCE as the text it will print, and that text is
    # pinned below. Both lists are already pinned as values, so what is left loose is which
    # list lands under which LABEL: hand the "too big -> stuck bouncing at 4" line the
    # falling curve and the day's headline evidence reads backwards with nothing to object.
    curve_big_line = ("lr = 1.0 loss per step (too big -> stuck bouncing at 4): %s ..."
                      % (curve_big_cells,))
    curve_good_line = ("lr = 0.1 loss per step (just right -> drops to ~0): %s ..."
                       % (curve_good_cells,))
    print()
    print(curve_big_line)
    print(curve_good_line)

    # Every number above is a SQUARE, and squaring hides the sign — so "lr = 1.0 makes
    # the weight flip sign" has been a sentence, not evidence. Here is the sign itself:
    # lr = 1.0 lands on the far wall each hop (-2, +2, -2, +2 — same height, other side,
    # which is exactly why its loss sits at 4 forever), while lr = 0.1 never crosses the
    # bottom and just shrinks toward it.
    signed_big = ["%+.3f" % w for w in path_big[:4]]
    signed_good = ["%+.3f" % w for w in path_good[:4]]
    # Same reasoning, and it matters more here: "flips side" vs "same side" is one of the
    # day's four headline claims, and it is carried entirely by which row of signs sits
    # under which label. So pin the rendered rows, not only the two lists.
    signed_big_line = ("lr = 1.0 SIGNED weight per step (flips side, same size): %s ..."
                       % " ".join(signed_big))
    signed_good_line = ("lr = 0.1 SIGNED weight per step (same side, shrinking):  %s ..."
                        % " ".join(signed_good))
    print(signed_big_line)
    print(signed_good_line)

    # -----------------------------------------------------------------
    # Why those four verdicts belong to THIS surface (curvature), not to the numbers.
    # One step multiplies the WEIGHT by (1 - lr*C); here C = 2 because the gradient is
    # 2*weight. So the walk is stable while |1 - lr*C| < 1, that is lr < 2/C = 1, and
    # lr = 1.0 sits exactly ON that line — which is why it bounces +2 <-> -2 forever
    # instead of exploding. Day 6's neuron has C = 10 (limit lr < 0.2): the same 0.01
    # shrinks its miss by 0.9 a lap and is called HEALTHY there, while lr = 1.0 — the
    # harmless bouncer here — blows that neuron up. Same rate, different curvature.
    # -----------------------------------------------------------------
    curvature = 2.0                                # from gradient = 2 * weight
    # the stability line lr < 2/C, computed ONCE: printed here and on the blow-up line below
    stability_limit = 2 / curvature
    # The factor 1 - lr*C, computed ONCE: these are the values printed AND checked.
    predicted_shrinks = [1 - lr * curvature for lr in rates]
    shrink_cells = ["lr %-5s -> %+.3f" % (lr, s) for lr, s in zip(rates, predicted_shrinks)]
    # measured, not assumed, and signed: the WEIGHT after one real hop / the start.
    measured_shrinks = {lr: weight_path(lr, steps=1)[0] / 2.0 for lr in rates}
    measured_cells = ["lr %-5s -> %+.3f" % (lr, measured_shrinks[lr]) for lr in rates]
    measured_good_shrink = measured_shrinks[0.1]    # 0.80 — "just right" here
    measured_crawl_shrink = measured_shrinks[0.01]  # 0.98 — "crawl" here
    # `curvature == 2.0` and `stability_limit == 1.0` are both pinned below — and that is
    # not enough, because this header reads them in an ORDER, and the order IS the lesson.
    # Swapped, the same two pinned numbers print "C = 1, stable while lr < 2", which makes
    # lr = 1.0's bounce look like a comfortably stable rate. So pin the rendered header.
    shrink_header = ("per-step weight shrink 1 - lr*C on this bowl (C = %g, stable while lr < %g):"
                     % (curvature, stability_limit))
    print()
    print(shrink_header)
    print("  formula  1 - lr*C: " + " | ".join(shrink_cells))
    print("  measured from real hops: " + " | ".join(measured_cells))
    print("  lr = 1.0 sits exactly on -1.000: the sign flips, the size does not — a bounce,"
          " not an explosion.")

    # The same rate, a different curvature. Day 6's neuron is its own loop (pred = w*2 + b,
    # target 1, and BOTH weights move), so its curvature is C = 2*(x*x + 1) = 10. Measure
    # what one lap does to its miss at three rates instead of asserting one sentence about
    # one of them — and keep the sign, because at lr = 1.0 the miss comes back NINE times
    # bigger on the other side. That is the whole reason a rate's verdict cannot travel.
    DAY06_X, DAY06_TARGET = 2.0, 1.0

    def day06_loss(w, b):
        # pred = w*x + b with x = 2 and target = 1 — Day 6's own model, no training
        return (w * DAY06_X + b - DAY06_TARGET) ** 2

    def day06_miss_shrink(lr, w=0.0, b=0.0):
        # ONE lap of Day 6's loop, with Day 6's two gradients. Returns the SIGNED factor
        # the miss got multiplied by, measured from the lap.
        miss = w * DAY06_X + b - DAY06_TARGET
        w = w - lr * 2 * miss * DAY06_X   # Day 6's grad_w = 2*(pred - target)*x
        b = b - lr * 2 * miss             # Day 6's grad_b = 2*(pred - target)
        new_miss = w * DAY06_X + b - DAY06_TARGET
        return new_miss / miss + 0.0      # + 0.0 prints a dead-on lap as +0.000, not -0.000

    day06_curvature = 2 * (DAY06_X * DAY06_X + 1)   # = 10, so stable only while lr < 0.2
    day06_limit = 2 / day06_curvature               # 0.2 — printed below, pinned below
    bridge_rates = [0.01, 0.1, 1.0]
    bridge = {lr: day06_miss_shrink(lr) for lr in bridge_rates}
    bridge_cells = ["lr %-4s -> %+.3f" % (lr, bridge[lr]) for lr in bridge_rates]
    # The four numbers the prose sentence quotes, rendered ONCE here instead of a second
    # time inside the print: the bridge is the day's cross-day claim, so the sentence the
    # learner reads and the strings the check pins must be the same strings.
    bridge_prose_cells = ("%+.3f" % measured_crawl_shrink,   # the 'crawl' rate here
                          "%+.3f" % bridge[0.01],            # …is the healthy one there
                          "%+.3f" % bridge[0.1],             # 0.1 lands dead on target
                          "%.0f" % abs(bridge[1.0]))         # 1.0 throws the miss 9x out
    # Same trap as the header above, one day further: `day06_curvature == 10.0` and
    # `day06_limit == 0.2` are both pinned, so the only thing loose is the order they are
    # read in — and swapped they print "C = 2*(x*x+1) = 0.2, stable while lr < 10", which
    # says every rate in this file is safe on a neuron that lr = 1.0 blows up.
    bridge_header = ("same rates on Day 6's neuron (C = 2*(x*x+1) = %g, stable while lr < %g):"
                     % (day06_curvature, day06_limit))
    print(bridge_header)
    print("  one lap multiplies its miss by: " + " | ".join(bridge_cells))
    print("  so 0.01 — the 'crawl' rate here (%s) — is the HEALTHY one there (%s),"
          " 0.1 lands dead on the target in ONE lap (%s), and 1.0 — the harmless"
          " bouncer here — throws the miss %sx further out, on the far side."
          % bridge_prose_cells)

    # And the bowl is a SURFACE, not a loss with data behind it, so "the bottom is at
    # weight = 0" is not a rule of training. Day 6's real neuron, evaluated at three
    # points, says the opposite: weight zero is its WORST point and the bottom is the
    # whole line 2w + b = 1. Each of the three heights is computed ONCE and pinned below,
    # so the sentence cannot quote a height the self-check never evaluated.
    zero_loss = day06_loss(0.0, 0.0)        # weight zero: Day 6's WORST point
    on_line_a = day06_loss(0.4, 0.2)        # on the line 2w + b = 1
    on_line_b = day06_loss(0.5, 0.0)        # also on that line
    bottom_cells = ("%g" % zero_loss, "%g" % on_line_a, "%g" % on_line_b)
    print("a bottom at zero is this SURFACE's property, not a rule: Day 6's neuron"
          " (pred = w*2 + b, target 1) scores %s at w = b = 0, but %s at (0.4, 0.2)"
          " and %s at (0.5, 0.0) — its bottom is the LINE 2w + b = 1"
          % bottom_cells)


    # A hair over the stability line is not a small difference. At lr = 1.1 the factor
    # is 1 - 1.1*2 = -1.2: the weight flips AND grows 20% every hop, so 60 hops leave
    # the screen. Printed, not just tested — and pinned to its exact size, because
    # "bigger than a million" is true for lr = 1.1 and for lr = 1.9 alike and would let
    # a wrong rate sit here unnoticed.
    blowup_lr = 1.1
    blowup_loss = train(blowup_lr, steps=60)
    blowup_decades = math.log10(blowup_loss)
    # rendered ONCE — the size on screen is the size pinned at the bottom
    blowup_cells = ("%g" % blowup_lr, "%.6g" % blowup_loss, "%.2f" % blowup_decades)
    # and the finished LINE rendered once as well, because read in the wrong order the same
    # pinned cells print "lr = 1.27002e+10 for 60 steps -> loss 1.1", which turns a blow-up
    # into a tiny loss. The stability line is quoted here too, so the pin ties this sentence
    # to the same `stability_limit` the shrink header printed.
    blowup_line = ("a hair over the line: lr = %s for 60 steps -> loss %s (10^%s), while"
                   " lr = 1.0 just bounces at 4. The line is at lr = %g."
                   % (blowup_cells + (stability_limit,)))
    print()
    print(blowup_line)

    # bonus: the halving schedule with a 0.1 floor
    sched_loss, sched_lr = train_with_schedule()
    # the settled rate and the final loss as PRINTED, pinned below as printed
    sched_cells = ("%.3g" % sched_lr, "%.8g" % sched_loss)
    print("\nSchedule (start 0.8, halve, floor 0.1): settled rate = %s, "
          "final loss = %s" % sched_cells)
    # the first four steps of the schedule, so the words "start 0.8, halve, floor 0.1"
    # above are numbers you can read: the rate the run settles on NEXT, and the loss
    # the hop at the CURRENT rate produced (hop first, halve after). Each row is built
    # ONCE, so the check pins the row text the learner actually sees.
    ladder = [train_with_schedule(steps=k) for k in (1, 2, 3, 4)]
    ladder_cells = ["after %d -> loss %.6g, next rate %.3g" % (k, loss, rate)
                    for k, (loss, rate) in zip((1, 2, 3, 4), ladder)]
    print("  first 4 steps: " + " | ".join(ladder_cells))

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
        # …and the four sweep lines exactly as they reached the screen, so the printed
        # column cannot drift away from the floats just checked above
        final_cells == ['lr = 1.0    -> final loss = 4',
                        'lr = 0.1    -> final loss = 7.0673883e-08',
                        'lr = 0.01   -> final loss = 0.7945954',
                        'lr = 0.001  -> final loss = 3.4080291'] and
        # the two printed curves are the day's headline evidence, so pin their SHAPE.
        # lr = 1.0 sits at exactly 4.0 on every one of the 40 steps: the weight flips
        # +2 <-> -2 and never gets closer to the bottom.
        curve_big == [4.0] * 40 and
        # lr = 0.1 starts at 2.56 (= 1.6**2, one hop down from 4.0), is at 0.2749 by
        # step 6, and falls a little on every single step in between.
        abs(curve_good[0] - 2.56) < 1e-12 and
        abs(curve_good[5] - 0.274877906944) < 1e-12 and
        all(curve_good[i + 1] < curve_good[i] for i in range(len(curve_good) - 1)) and
        # the six cells of each curve as PRINTED — the shape claims above live on the
        # full 40-step lists, and these pin the window the learner actually reads
        curve_big_cells == [4.0] * 6 and
        curve_good_cells == ['2.56', '1.638', '1.049', '0.6711', '0.4295', '0.2749'] and
        # the signed weights printed above: squares would let a sign error through, and
        # "lr = 1.0 flips the weight" is one of the day's four headline claims.
        signed_big == ['-2.000', '+2.000', '-2.000', '+2.000'] and
        signed_good == ['+1.600', '+1.280', '+1.024', '+0.819'] and
        # the curve and the sweep are two views of the same walk: after the
        # same 40 steps they must land on the same loss, or one of them has drifted.
        abs(curve_big[-1]  - finals[1.0]) < 1e-12 and
        abs(curve_good[-1] - finals[0.1]) < 1e-15 and
        # the schedule must settle exactly at the 0.1 floor
        abs(sched_lr - 0.1) < 1e-9 and
        # the settled rate and final loss exactly as the schedule line printed them
        sched_cells == ('0.1', '1.3976036e-09') and
        # ...and land far more gently than the too-big rate — pinned to the exact
        # value, not just "small", so the step count and the decay both matter
        abs(sched_loss - 1.3976036352250886e-09) < 1e-20 and
        # the halving ladder: 0.8 -> 0.4 -> 0.2 -> 0.1 (floor holds), and the loss
        # after each early step, which pins the START rate and the hop-THEN-halve order
        [round(rate, 6) for _loss, rate in ladder] == [0.4, 0.2, 0.1, 0.1] and
        abs(ladder[0][0] - 1.44)       < 1e-12 and
        abs(ladder[1][0] - 0.0576)     < 1e-12 and
        abs(ladder[2][0] - 0.020736)   < 1e-12 and
        abs(ladder[3][0] - 0.01327104) < 1e-12 and
        # …and the four ladder rows as printed, so the losses and rates on screen are
        # the losses and rates being pinned
        ladder_cells == ['after 1 -> loss 1.44, next rate 0.4',
                         'after 2 -> loss 0.0576, next rate 0.2',
                         'after 3 -> loss 0.020736, next rate 0.1',
                         'after 4 -> loss 0.013271, next rate 0.1'] and
        # the printed blow-up: pinned to its EXACT size (10^10.1), so a different
        # too-big rate cannot hide behind a "> 1e6" that almost anything passes
        blowup_lr == 1.1 and
        abs(blowup_loss / 12700169495.121557 - 1) < 1e-12 and
        abs(blowup_decades - 10.1038095) < 1e-6 and
        # the same three numbers as the line PRINTED them (rate, size, decades)
        blowup_cells == ('1.1', '1.27002e+10', '10.10') and
        # the curvature block: the measured shrink must match the printed 1 - lr*C, or C
        # is wrong — and C is the whole reason a rate's verdict cannot travel between days
        abs(measured_good_shrink - 0.8) < 1e-12 and
        abs(measured_crawl_shrink - 0.98) < 1e-12 and
        # the curvature and the stability line as printed on that header row
        curvature == 2.0 and stability_limit == 1.0 and
        abs(measured_good_shrink - predicted_shrinks[1]) < 1e-12 and
        abs(measured_crawl_shrink - predicted_shrinks[2]) < 1e-12 and
        # lr = 1.0 lands exactly ON the stability line (1 - lr*C = -1), which is why it
        # bounces at a fixed height instead of exploding: the sign flips, the size does not
        abs(predicted_shrinks[0] + 1.0) < 1e-12 and
        # the shrink row exactly as printed — sign included, since a sign flip there is
        # the difference between "settles" and "bounces"
        shrink_cells == ['lr 1.0   -> -1.000', 'lr 0.1   -> +0.800',
                         'lr 0.01  -> +0.980', 'lr 0.001 -> +0.998'] and
        measured_cells == shrink_cells and
        # the cross-day bridge, measured on Day 6's own lap at THREE rates instead of
        # asserted once in prose: 0.01 is healthy there, 0.1 lands dead on the target,
        # and 1.0 throws the miss 9x out on the far side (the sign is the point)
        day06_curvature == 10.0 and
        day06_limit == 0.2 and
        bridge_cells == ['lr 0.01 -> +0.900', 'lr 0.1  -> +0.000', 'lr 1.0  -> -9.000'] and
        # the prose sentence's own four numbers, as it printed them
        bridge_prose_cells == ('+0.980', '+0.900', '+0.000', '9') and
        all(abs(bridge[lr] - (1 - lr * day06_curvature)) < 1e-12 for lr in bridge_rates) and
        # Day 6's neuron: zero weights are its WORST point, and its bottom is a LINE
        abs(zero_loss - 1.0) < 1e-12 and
        on_line_a < 1e-28 and
        on_line_b < 1e-28 and
        # …and the three heights as the sentence printed them
        bottom_cells == ('1', '0', '0') and
        # And the seven claim-carrying LINES, pinned as the exact text that reached the
        # screen. Every number on them is pinned as a value above; what is pinned here is
        # the ORDER the line reads them in, which is where each claim lives. Without these,
        # swapping two operands at a print call prints "C = 1, stable while lr < 2" (making
        # a bouncing rate look stable), or "C = 2*(x*x+1) = 0.2, stable while lr < 10"
        # (making an exploding rate look safe), or hands the "too big" label the falling
        # curve — with every existing assert still green.
        curve_big_line == ("lr = 1.0 loss per step (too big -> stuck bouncing at 4): "
                           "[4.0, 4.0, 4.0, 4.0, 4.0, 4.0] ...") and
        curve_good_line == ("lr = 0.1 loss per step (just right -> drops to ~0): "
                            "['2.56', '1.638', '1.049', '0.6711', '0.4295', '0.2749'] ...") and
        signed_big_line == ("lr = 1.0 SIGNED weight per step (flips side, same size): "
                            "-2.000 +2.000 -2.000 +2.000 ...") and
        signed_good_line == ("lr = 0.1 SIGNED weight per step (same side, shrinking):  "
                             "+1.600 +1.280 +1.024 +0.819 ...") and
        shrink_header == ("per-step weight shrink 1 - lr*C on this bowl "
                          "(C = 2, stable while lr < 1):") and
        bridge_header == ("same rates on Day 6's neuron "
                          "(C = 2*(x*x+1) = 10, stable while lr < 0.2):") and
        blowup_line == ("a hair over the line: lr = 1.1 for 60 steps -> loss 1.27002e+10"
                        " (10^10.10), while lr = 1.0 just bounces at 4."
                        " The line is at lr = 1.")
    )

    if ok:
        print("\n✅ you got it — the sweep walks explode -> just-right -> crawl -> frozen,"
              " and the schedule settles at the 0.1 floor.")
    else:
        print("\n❌ not yet — expected lr=1.0->4.0, lr=0.1->~7.07e-08, "
              "lr=0.01->~0.79, lr=0.001->~3.41, the lr=1.0 curve flat at 4.0 with the "
              "weight flipping -2, +2, -2, +2, the "
              "lr=0.1 curve falling 2.56 -> 0.2749 over 6 steps, the schedule "
              "rate stepping 0.8 -> 0.4 -> 0.2 -> 0.1, lr=1.1 for 60 steps landing on "
              "1.27e10, per-step shrinks of -1.00/+0.80/+0.98/+1.00 matching 1 - lr*C "
              "with C = 2, and Day 6's C = 10 neuron multiplying its miss by "
              "+0.900 / +0.000 / -9.000 at lr = 0.01 / 0.1 / 1.0")


    # a hard assert so the script FAILS loudly (non-zero exit) if the numbers drift
    assert ok, "learning-rate sweep did not match the lesson's expected values"
