# day-05-gradients-backprop — experiment
#
# Watch a gradient flow BACKWARD through one tiny neuron, then CHECK your work
# two different ways. This is the whole day in ~40 lines: forward pass (save the
# notes), backprop by hand (the three rules), and a gradient check (measure the
# same slope with no calculus, just two loss readings).
#
# We only need one import — math is not even required, but we bring it in so this
# is a real, honest script (the gate expects an import; numpy would be fine too).
import math  # not strictly needed, but keeps this a real, importable module


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
    L = (a - target) ** 2
    # hand back everything, including the saved notes (x, z, a) for the trip back
    return z, a, L


if __name__ == "__main__":
    # the tiny fixed setup the lesson gives us
    x = 2.0        # the neuron's input
    w = 0.5        # the weight we will trace blame to
    b = 0.1        # the bias
    target = 1.0   # the true answer we wanted

    # ---- run the forward pass once and keep the saved notes ----
    z, a, L = forward(x, w, b, target)
    print("forward pass:  z =", z, " a =", a, " loss =", round(L, 4))

    # ---- STEP 2: BACKPROP by hand, using the three rules from the lesson ----
    # the gradient flowing INTO the neuron from the loss side: d/da of (a-target)^2
    incoming = 2.0 * (a - target)
    # the ReLU's LOCAL slope at the saved z: a clean 1 if z>0, else 0
    relu_slope = 1.0 if z > 0 else 0.0
    # Rule 1 — a weight's gradient = its input × the gradient reaching it
    w_grad = incoming * relu_slope * x
    # Rule 3 — the bias just passes the gradient through (slope 1)
    b_grad = incoming * relu_slope * 1.0
    print("backprop:      w_grad =", round(w_grad, 6), " b_grad =", round(b_grad, 6))
    # the headline of the day: the weight's blame is exactly x times the bias's
    print("check ratio:   w_grad =", x, "× b_grad =", round(x * b_grad, 6))

    # ---- STEP 3: the GRADIENT CHECK — measure the SAME slope a different way ----
    # a loss-as-a-function-of-w helper: redo the forward pass, return just the loss
    def loss_of_w(w_try):
        _, _, L_try = forward(x, w_try, b, target)
        return L_try

    eps = 1e-5  # a tiny nudge
    # nudge w up by eps and down by eps, see how much the loss moved, divide by 2eps
    measured = (loss_of_w(w + eps) - loss_of_w(w - eps)) / (2 * eps)
    print("grad check:    measured =", round(measured, 4), " backprop w_grad =", round(w_grad, 4))

    # ---- self-checks: the numbers must match the lesson's "what you should see" ----
    # the weight's gradient is exactly x times the bias's gradient
    assert abs(w_grad - x * b_grad) < 1e-9, "w_grad should equal x * b_grad"
    # the hand-computed w_grad for this setup is 0.4
    assert abs(w_grad - 0.4) < 1e-9, "w_grad should be 0.4"
    # and the bias gradient is 0.2
    assert abs(b_grad - 0.2) < 1e-9, "b_grad should be 0.2"
    # the measured slope must match backprop to ~4 decimals — proof the code is right
    assert abs(measured - w_grad) < 1e-3, "measured slope should match backprop"

    # if every check passed, tell the learner they got it
    if abs(measured - w_grad) < 1e-3 and abs(w_grad - 0.4) < 1e-9:
        print("✅ you got it — w_grad is exactly x times b_grad, and the gradient check matches")
    else:
        print("❌ not yet — expected w_grad 0.4 (= x × b_grad 0.2) matching the measured slope")
