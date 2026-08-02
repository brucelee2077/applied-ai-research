# day-05-gradients-backprop — experiment
#
# Watch a gradient flow BACKWARD through one tiny neuron, then CHECK your work
# two different ways. This is the whole day: forward pass (save the notes),
# backprop by hand (the lesson's rules), and a gradient check (measure the
# same slope with no calculus, just two loss readings).
#
# math.isclose does the "are these two numbers the same?" comparisons at the
# bottom — exactly the judgement a gradient check asks you to make.
import math
from collections import namedtuple

# Everything one backward pass produces, kept in one bundle so we can run the
# SAME trace at several settings: a positive z, a negative z (the dead ReLU the
# lesson promises), and a z that sits exactly on 0.
# Field name note: `loss` here is the same object Day 4 and Day 6 call loss (Day 4's
# lesson writes it as L); one name for it across the module.
Trace = namedtuple("Trace", "z a loss incoming slope delta w_grad b_grad passed_back")


# --- the activation from Day 2: ReLU passes positives, zeros out negatives ---
def relu(z):
    # if z is above 0 keep it; otherwise the neuron outputs a flat 0
    return max(0.0, z)


# --- STEP 1: the FORWARD pass — make a guess, and SAVE the in-between notes ---
def forward(x, w, b, target):
    # weighted sum + bias — this in-between number is the pre-activation z
    z = w * x + b
    # bend z with the activation to get the neuron's output a (its guess)
    a = relu(z)
    # compare the guess a to the true answer with a squared-error loss
    loss = (a - target) ** 2
    # hand back everything, including the saved notes (x, z, a) for the trip back
    return z, a, loss


# --- STEP 2: BACKPROP by hand, using the rules from the lesson ---
def backward(x, w, b, target):
    # re-use the forward pass's saved notes — nothing is recomputed from scratch
    z, a, loss = forward(x, w, b, target)
    # the gradient flowing INTO the neuron from the loss side: d/da of (a-target)^2
    incoming = 2.0 * (a - target)
    # the ReLU's LOCAL slope at the saved z: a clean 1 if z>0, else 0
    relu_slope = 1.0 if z > 0 else 0.0
    # Rule 1 — the activation charges a toll: delta is what gets through to the knobs
    delta = incoming * relu_slope
    # Rule 2 — the weight's share = delta × its input value
    w_grad = delta * x
    # Rule 3 — the bias travels free (slope 1), so delta passes straight through
    b_grad = delta * 1.0
    # Rule 4 — one for the road: what gets handed one layer further back is delta × w
    passed_back = delta * w
    return Trace(z, a, loss, incoming, relu_slope, delta, w_grad, b_grad, passed_back)


# --- STEP 3: the GRADIENT CHECK — measure the SAME slopes a different way ---
def measured_grads(x, w, b, target, eps=1e-5):
    """Nudge one knob up and down by eps, see how much the loss moved, / 2eps.

    No calculus and no backprop code touches this — that is the point. It gives
    an INDEPENDENT reading of both slopes, so the day's headline (the weight's
    gradient is x times the bias's) can be checked between two MEASURED numbers
    instead of between two lines that were written from the same delta.
    """
    def loss_of_w(w_try):
        return forward(x, w_try, b, target)[2]

    def loss_of_b(b_try):
        return forward(x, w, b_try, target)[2]

    measured_w = (loss_of_w(w + eps) - loss_of_w(w - eps)) / (2 * eps)
    measured_b = (loss_of_b(b + eps) - loss_of_b(b - eps)) / (2 * eps)
    return measured_w, measured_b


def tidy(value):
    """Print helper only: turn IEEE's -0.0 into 0.0, leave every other number be.

    A dead ReLU multiplies a negative incoming gradient by a slope of 0, and
    -2.0 * 0.0 is -0.0 in floating point. That is still zero; showing it as
    "-0.0" would just look like a bug to a reader.
    """
    return value + 0.0


if __name__ == "__main__":
    # the tiny fixed setup the lesson gives us
    x = 2.0        # the neuron's input
    w = 0.5        # the weight we will trace blame to
    b = 0.1        # the bias
    target = 1.0   # the true answer we wanted

    # ---- the main trace: forward, then all four rules backward ----
    t = backward(x, w, b, target)
    # Every number below is ROUNDED FOR DISPLAY exactly once, bound to a name, and
    # then printed. The self-checks at the bottom read these same names, so the line
    # a learner reads and the line that gets checked are one thing. Rounding inside
    # the print instead would make them two independent expressions: re-round or
    # swap a cell there and the number on screen goes wrong under a passing ✅.
    shown_loss = round(t.loss, 4)
    shown_delta = round(t.delta, 6)
    shown_w_grad = round(t.w_grad, 6)
    shown_b_grad = round(t.b_grad, 6)
    shown_passed_back = round(t.passed_back, 6)
    forward_line = f"forward pass:  z = {t.z}  a = {t.a}  loss = {shown_loss}"
    backprop_line = (f"backprop:      delta = {shown_delta}"
                     f"  w_grad = {shown_w_grad}  b_grad = {shown_b_grad}")
    # the headline of the day: the weight's blame is exactly x times the bias's.
    # We print w_grad itself here, NOT the product x * b_grad — the product would
    # agree with itself no matter what w_grad came out as.
    ratio_line = f"check ratio:   w_grad = {x} × b_grad = {shown_w_grad}"
    prev_layer_line = f"to prev layer: delta × w = {shown_passed_back}"
    print(forward_line)
    print(backprop_line)
    print(ratio_line)
    print(prev_layer_line)

    # ---- gradient-check BOTH knobs, so the ratio compares two measurements ----
    measured_w, measured_b = measured_grads(x, w, b, target)
    shown_measured_w = round(measured_w, 4)
    shown_measured_b = round(measured_b, 4)
    shown_w_grad4 = round(t.w_grad, 4)
    shown_b_grad4 = round(t.b_grad, 4)
    # the ratio is its own printed claim, so it gets its own bound name — printing
    # a freshly written division here would be a number nothing else ever reads
    shown_ratio = round(measured_w / measured_b, 4)
    check_w_line = (f"grad check:    measured = {shown_measured_w}"
                    f"  backprop w_grad = {shown_w_grad4}")
    check_b_line = (f"grad check b:  measured = {shown_measured_b}"
                    f"  backprop b_grad = {shown_b_grad4}")
    measured_ratio_line = (f"measured ratio: {shown_measured_w} / {shown_measured_b}"
                           f" = {shown_ratio}  (= x, both sides measured)")
    print(check_w_line)
    print(check_b_line)
    print(measured_ratio_line)

    # ---- change x and the ratio changes WITH it — so it is a fact about x ----
    x_big = 3.0
    t_big = backward(x_big, w, b, target)
    measured_w_big, measured_b_big = measured_grads(x_big, w, b, target)
    shown_measured_w_big = round(measured_w_big, 4)
    shown_measured_b_big = round(measured_b_big, 4)
    shown_ratio_big = round(measured_w_big / measured_b_big, 4)
    # the "x = 3" in the label is READ FROM x_big, not typed again, so the label
    # cannot claim one input while the numbers came from another
    big_ratio_line = (f"same at x = {x_big:g}: {shown_measured_w_big}"
                      f" / {shown_measured_b_big} = {shown_ratio_big}"
                      "  (the ratio follows x)")
    print(big_ratio_line)

    # ---- the DEAD ReLU the lesson promises: a big negative b puts z below 0 ----
    b_dead = -3.0
    t_dead = backward(x, w, b_dead, target)
    measured_w_dead, measured_b_dead = measured_grads(x, w, b_dead, target)
    # tidy() is applied ONCE per value, here, and the checks below read these same
    # names — so the helper cannot quietly turn a non-zero gradient into a shown 0.0
    dead_w_shown = tidy(t_dead.w_grad)
    dead_b_shown = tidy(t_dead.b_grad)
    dead_prev_shown = tidy(t_dead.passed_back)
    shown_dead_incoming = round(t_dead.incoming, 4)
    shown_measured_w_dead = round(measured_w_dead, 4)
    shown_measured_b_dead = round(measured_b_dead, 4)
    dead_line = (f"dead relu:     b = {b_dead} → z = {t_dead.z}"
                 f"  slope = {t_dead.slope}  w_grad = {dead_w_shown}"
                 f"  b_grad = {dead_b_shown}  to prev = {dead_prev_shown}")
    dead_flat_line = (f"               incoming = {shown_dead_incoming}"
                      " (the loss still wants a change) but the measured slopes are"
                      f" {shown_measured_w_dead} and {shown_measured_b_dead}"
                      " — a genuinely flat spot")
    print(dead_line)
    print(dead_flat_line)

    # ---- exactly ON the boundary: z = 0.5*2.0 - 1.0 = 0.0, so > vs >= decides ----
    # (no gradient check here: z = 0 is the ReLU's kink, where a measured slope
    # reads the sharp corner rather than any single true slope.)
    b_edge = -1.0
    t_edge = backward(x, w, b_edge, target)
    edge_w_shown = tidy(t_edge.w_grad)
    edge_line = (f"z exactly 0:   b = {b_edge} → z = {t_edge.z}"
                 f"  slope = {t_edge.slope}"
                 f" (the rule is z > 0, not z >= 0)  w_grad = {edge_w_shown}")
    print(edge_line)

    # ---- self-checks: every number the lesson's "what you should see" promises ----
    # One table drives both the ❌ guidance line and the asserts, so the two can
    # never drift apart. (got, want, tol) — abs_tol, because these are small.
    expected = [
        # the forward pass's own printed numbers. The loss is what the backward pass
        # is the slope OF, so pin it: without this row a constant bolted onto the loss
        # inside forward() would print a wrong loss and every gradient below would
        # still agree (a constant shifts the loss but not its slope).
        ("the forward pass's loss should be 0.01 (a = 1.1 against target 1.0)", t.loss, 0.01, 1e-9),
        ("the forward pass's z should be 1.1", t.z, 1.1, 1e-9),
        ("the forward pass's a should be 1.1 (z > 0, so ReLU passes it through)", t.a, 1.1, 1e-9),
        # delta is printed as the toll survivor, so it is pinned like everything else
        ("delta — what survives the ReLU's toll — should be 0.2", t.delta, 0.2, 1e-9),
        ("w_grad should be 0.4", t.w_grad, 0.4, 1e-9),
        ("b_grad should be 0.2", t.b_grad, 0.2, 1e-9),
        ("Rule 4: delta × w handed one layer back should be 0.1", t.passed_back, 0.1, 1e-9),
        ("the measured w-slope should match backprop's w_grad", measured_w, t.w_grad, 1e-3),
        ("the measured b-slope should match backprop's b_grad", measured_b, t.b_grad, 1e-3),
        # the day's headline, between two INDEPENDENT measurements — not an identity
        ("the measured w-slope should be x = 2.0 times the measured b-slope",
         measured_w, x * measured_b, 1e-3),
        # and at a different x, so the claim is about x rather than about one setting
        ("at x = 3.0 the measured w-slope should be 3.0 times the measured b-slope",
         measured_w_big, x_big * measured_b_big, 1e-3),
        ("at x = 3.0 backprop's w_grad should be 3.6", t_big.w_grad, 3.6, 1e-9),
        # the dead ReLU: the toll is what zeroes these, and the probe is only
        # meaningful because a NON-zero gradient really did arrive from the loss
        ("the dead-ReLU probe must be live: incoming should be -2.0", t_dead.incoming, -2.0, 1e-9),
        # the printed z is what shows the ReLU is off; without this row it could
        # print any negative number and every zero below would still look right
        ("the dead-ReLU setting should put z at -2.0", t_dead.z, -2.0, 1e-9),
        ("with z < 0 the ReLU slope is 0", t_dead.slope, 0.0, 1e-12),
        ("with z < 0 the weight's gradient must be 0", t_dead.w_grad, 0.0, 1e-12),
        ("with z < 0 the bias's gradient must be 0", t_dead.b_grad, 0.0, 1e-12),
        ("with z < 0 nothing is handed to the previous layer", t_dead.passed_back, 0.0, 1e-12),
        ("with z < 0 the measured slope must be 0 too — a genuinely flat loss",
         measured_w_dead, 0.0, 1e-12),
        # the bias's measured slope is printed on that same line, so pin it as well
        ("with z < 0 the measured BIAS slope must be 0 too", measured_b_dead, 0.0, 1e-12),
        # the boundary: the rule is z > 0, not z >= 0, so z == 0 gets slope 0
        ("the boundary setting should put z at exactly 0", t_edge.z, 0.0, 1e-12),
        ("at exactly z = 0 the ReLU slope is 0 (the rule is z > 0)", t_edge.slope, 0.0, 1e-12),
        ("at exactly z = 0 the weight's gradient must be 0", t_edge.w_grad, 0.0, 1e-12),
        # the print helper above is allowed to erase a MINUS SIGN in front of a
        # zero and nothing else — pin it, or it could quietly show every one of
        # those zeros as 0.0 whether they really were zero or not
        ("the -0.0 print helper must leave a real number alone", tidy(-0.4), -0.4, 1e-12),
    ]
    # The other half of the self-check: the printed LINES, character for character.
    # Each line was built once and printed once, so this pins what the learner
    # actually reads — a re-rounded cell, or two correct numbers swapped into each
    # other's slots, changes the screen without changing any value above, and would
    # otherwise sail past every row in `expected`.
    printed_lines = [
        ("the forward line should read z = 1.1, a = 1.1, loss = 0.01",
         forward_line, "forward pass:  z = 1.1  a = 1.1  loss = 0.01"),
        ("the backprop line should read delta = 0.2, w_grad = 0.4, b_grad = 0.2",
         backprop_line, "backprop:      delta = 0.2  w_grad = 0.4  b_grad = 0.2"),
        ("the ratio line should show w_grad = 0.4 against x = 2.0",
         ratio_line, "check ratio:   w_grad = 2.0 × b_grad = 0.4"),
        ("the hand-back line should read 0.1",
         prev_layer_line, "to prev layer: delta × w = 0.1"),
        ("the w gradient check should show 0.4 measured against 0.4 from backprop",
         check_w_line, "grad check:    measured = 0.4  backprop w_grad = 0.4"),
        ("the b gradient check should show 0.2 measured against 0.2 from backprop",
         check_b_line, "grad check b:  measured = 0.2  backprop b_grad = 0.2"),
        ("the measured ratio should print 0.4 / 0.2 = 2.0 — the w-slope on top",
         measured_ratio_line, "measured ratio: 0.4 / 0.2 = 2.0  (= x, both sides measured)"),
        ("at x = 3 the measured ratio should print 3.6 / 1.2 = 3.0",
         big_ratio_line, "same at x = 3: 3.6 / 1.2 = 3.0  (the ratio follows x)"),
        ("the dead-ReLU line should show z = -2.0 and three zeros",
         dead_line, "dead relu:     b = -3.0 → z = -2.0  slope = 0.0"
                    "  w_grad = 0.0  b_grad = 0.0  to prev = 0.0"),
        ("the flat-spot line should show a live incoming of -2.0 against two 0.0 slopes",
         dead_flat_line, "               incoming = -2.0 (the loss still wants a change)"
                         " but the measured slopes are 0.0 and 0.0 — a genuinely flat spot"),
        ("the boundary line should show z = 0.0 with slope 0.0 and w_grad 0.0",
         edge_line, "z exactly 0:   b = -1.0 → z = 0.0  slope = 0.0"
                    " (the rule is z > 0, not z >= 0)  w_grad = 0.0"),
    ]
    failures = [(why, got, want) for why, got, want, tol in expected
                if not math.isclose(got, want, abs_tol=tol)]
    failures += [(why, got, want) for why, got, want in printed_lines if got != want]

    # verdict FIRST, so a learner always sees the guidance line — the asserts
    # below are what turn a failure into a non-zero exit.
    if not failures:
        print("✅ you got it — w_grad is exactly x times b_grad, and the gradient check matches")
    else:
        for why, got, want in failures:
            print("❌ not yet —", why, "· got", got, "· expected", want)

    for why, got, want, tol in expected:
        assert math.isclose(got, want, abs_tol=tol), why
    for why, got, want in printed_lines:
        assert got == want, why
