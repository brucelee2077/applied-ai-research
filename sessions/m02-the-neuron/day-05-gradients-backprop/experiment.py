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
    print("forward pass:  z =", t.z, " a =", t.a, " loss =", round(t.loss, 4))
    print("backprop:      delta =", round(t.delta, 6),
          " w_grad =", round(t.w_grad, 6), " b_grad =", round(t.b_grad, 6))
    # the headline of the day: the weight's blame is exactly x times the bias's.
    # We print w_grad itself here, NOT the product x * b_grad — the product would
    # agree with itself no matter what w_grad came out as.
    print("check ratio:   w_grad =", x, "× b_grad =", round(t.w_grad, 6))
    print("to prev layer: delta × w =", round(t.passed_back, 6))

    # ---- gradient-check BOTH knobs, so the ratio compares two measurements ----
    measured_w, measured_b = measured_grads(x, w, b, target)
    print("grad check:    measured =", round(measured_w, 4), " backprop w_grad =", round(t.w_grad, 4))
    print("grad check b:  measured =", round(measured_b, 4), " backprop b_grad =", round(t.b_grad, 4))
    print("measured ratio:", round(measured_w, 4), "/", round(measured_b, 4),
          "=", round(measured_w / measured_b, 4), " (= x, both sides measured)")

    # ---- change x and the ratio changes WITH it — so it is a fact about x ----
    x_big = 3.0
    t_big = backward(x_big, w, b, target)
    measured_w_big, measured_b_big = measured_grads(x_big, w, b, target)
    print("same at x = 3:", round(measured_w_big, 4), "/", round(measured_b_big, 4),
          "=", round(measured_w_big / measured_b_big, 4), " (the ratio follows x)")

    # ---- the DEAD ReLU the lesson promises: a big negative b puts z below 0 ----
    b_dead = -3.0
    t_dead = backward(x, w, b_dead, target)
    measured_w_dead, measured_b_dead = measured_grads(x, w, b_dead, target)
    print("dead relu:     b =", b_dead, "→ z =", t_dead.z, " slope =", t_dead.slope,
          " w_grad =", tidy(t_dead.w_grad), " b_grad =", tidy(t_dead.b_grad),
          " to prev =", tidy(t_dead.passed_back))
    print("               incoming =", round(t_dead.incoming, 4),
          "(the loss still wants a change) but the measured slopes are",
          round(measured_w_dead, 4), "and", round(measured_b_dead, 4),
          "— a genuinely flat spot")

    # ---- exactly ON the boundary: z = 0.5*2.0 - 1.0 = 0.0, so > vs >= decides ----
    # (no gradient check here: z = 0 is the ReLU's kink, where a measured slope
    # reads the sharp corner rather than any single true slope.)
    b_edge = -1.0
    t_edge = backward(x, w, b_edge, target)
    print("z exactly 0:   b =", b_edge, "→ z =", t_edge.z, " slope =", t_edge.slope,
          "(the rule is z > 0, not z >= 0)  w_grad =", tidy(t_edge.w_grad))

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
        ("with z < 0 the ReLU slope is 0", t_dead.slope, 0.0, 1e-12),
        ("with z < 0 the weight's gradient must be 0", t_dead.w_grad, 0.0, 1e-12),
        ("with z < 0 the bias's gradient must be 0", t_dead.b_grad, 0.0, 1e-12),
        ("with z < 0 nothing is handed to the previous layer", t_dead.passed_back, 0.0, 1e-12),
        ("with z < 0 the measured slope must be 0 too — a genuinely flat loss",
         measured_w_dead, 0.0, 1e-12),
        # the boundary: the rule is z > 0, not z >= 0, so z == 0 gets slope 0
        ("at exactly z = 0 the ReLU slope is 0 (the rule is z > 0)", t_edge.slope, 0.0, 1e-12),
        ("at exactly z = 0 the weight's gradient must be 0", t_edge.w_grad, 0.0, 1e-12),
        # the print helper above is allowed to erase a MINUS SIGN in front of a
        # zero and nothing else — pin it, or it could quietly show every one of
        # those zeros as 0.0 whether they really were zero or not
        ("the -0.0 print helper must leave a real number alone", tidy(-0.4), -0.4, 1e-12),
    ]
    failures = [(why, got, want) for why, got, want, tol in expected
                if not math.isclose(got, want, abs_tol=tol)]

    # verdict FIRST, so a learner always sees the guidance line — the asserts
    # below are what turn a failure into a non-zero exit.
    if not failures:
        print("✅ you got it — w_grad is exactly x times b_grad, and the gradient check matches")
    else:
        for why, got, want in failures:
            print("❌ not yet —", why, "· got", got, "· expected", want)

    for why, got, want, tol in expected:
        assert math.isclose(got, want, abs_tol=tol), why
