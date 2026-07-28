# day-02-backward-pass — experiment
#
# Today's big idea in two lines of output:
#   One forward trip leaves breadcrumbs — each layer's input, and which ReLU units were on.
#   One backward trip multiplies slopes along that trail and gives a gradient for EVERY knob.
#
# It ends with the lesson's two cheap checks (analytic vs numerical, and shape match), then
# breaks the backward pass on purpose so you can watch the check catch the bug.
# Run it:  python3 sessions/m04-first-model-mlp/day-02-backward-pass/experiment.py

import numpy as np  # numpy gives us arrays, matrix multiply (@), and a seeded random generator

def relu(z):   # the bend: keep positive numbers, turn negative ones into 0
    return np.maximum(0.0, z)

def forward(x, target, W1, b1, W2, b2):   # one forward trip; also returns the breadcrumbs
    z1 = x @ W1 + b1                             # layer 1 raw score, before the bend
    h = relu(z1)                                 # the bend
    out = h @ W2 + b2                            # layer 2 raw score = the 10 guesses
    loss = float(((out - target) ** 2).mean())   # one honest number; lower is better
    return loss, z1, h, out

def backward(x, target, z1, h, out, W2):   # the backward pass by hand; no autograd anywhere
    delta2 = 2.0 * (out - target) / out.size   # seed delta: slope of mean((out-target)^2) at out
    dW2 = h.T @ delta2      # weight slope = delta times the layer's SAVED input
    db2 = delta2.sum(0)     # bias slope = delta itself, added up over the batch rows
    # Pass blame back a layer, then through the ReLU gate: on-units keep their slope, off-units 0.
    delta1 = (delta2 @ W2.T) * (z1 > 0)
    dW1 = x.T @ delta1
    db1 = delta1.sum(0)
    return dW1, db1, dW2, db2

def numerical_slope(knob, index, x, target, knobs, h=1e-5):
    # The slow honest second opinion: nudge ONE number by +h and -h, re-run the forward pass.
    up = {name: value.copy() for name, value in knobs.items()}
    dn = {name: value.copy() for name, value in knobs.items()}
    up[knob][index] += h
    dn[knob][index] -= h
    return (forward(x, target, **up)[0] - forward(x, target, **dn)[0]) / (2.0 * h)

def relative_gap(a, b):   # how far apart two numbers are, measured against their own size
    return abs(a - b) / max(abs(a), abs(b), 1e-8)

if __name__ == "__main__":
    # --- Part 1: the chain rule is a multiply ------------------------------
    chain_product = float(np.prod([2.0, 3.0, 3.0]))   # the lesson's three dominoes, multiplied
    upstream, local = 4.0, 0.5                # one box from the lesson's wiring picture
    downstream = upstream * local             # multiply-and-pass-back, in one line
    print("slopes 2, 3, 3 -> chain =", chain_product, "|  one box: upstream", upstream,
          "x local", local, "-> downstream", downstream)
    print("squared miss: guess 2 vs target 5 ->", (2.0 - 5.0) ** 2, "| guess 4 ->", (4.0 - 5.0) ** 2)

    # --- Part 2: push a gradient back through a ReLU gate ------------------
    z_fwd = np.array([2.0, -1.0, 3.0, -0.5])            # what the forward pass produced here
    upstream_delta = np.array([0.4, -0.7, 0.9, -0.2])   # blame arriving from the right
    gate = z_fwd > 0                                    # the saved mask: was this unit on?
    predicted_survivors = int(gate.sum())    # the prediction is read off the inputs, not guessed
    gated_delta = upstream_delta * gate + 0.0   # the "+ 0.0" only turns -0.0 into 0.0 for printing
    actual_survivors = int(np.count_nonzero(gated_delta))
    print("\nforward inputs :", z_fwd, "\nincoming delta :", upstream_delta)
    print("gate (on = 1)  :", gate.astype(int), "-> we predict", predicted_survivors, "of 4 live")
    print("outgoing delta :", gated_delta, "-> survived:", actual_survivors)

    # --- Part 3: forward pass, saving the breadcrumbs ----------------------
    rng = np.random.default_rng(0)
    n_in, n_hidden, n_out, batch = 784, 128, 10, 4
    # MNIST is not on this machine and cannot be downloaded, so these 4 "images" are a seeded
    # stand-in: pixel-like numbers in [0, 1). The shapes are the real ones, 784 -> 128 -> 10.
    x = rng.random((batch, n_in))
    target = np.eye(n_out)[[7, 2, 1, 0]]      # one row per image, 1.0 at the true digit
    # He initialization: a spread of sqrt(2 / inputs), so ReLU units are not born dead.
    W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
    W2 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(2.0 / n_hidden)
    b1, b2 = np.zeros(n_hidden), np.zeros(n_out)   # the two biases start at zero
    loss, z1, h, out = forward(x, target, W1, b1, W2, b2)
    on_count = int((z1 > 0).sum())            # a breadcrumb: how many units were on
    print("\nx", x.shape, "-> z1", z1.shape, "-> h", h.shape, "-> out", out.shape)
    print("out[0][:4] =", np.round(out[0][:4], 4), " target[0][:4] =", target[0][:4])
    print("hidden units on (z1 > 0):", on_count, "of", z1.size, " loss =", round(loss, 8))

    # --- Part 4: backward pass by hand, and the free shape check -----------
    dW1, db1, dW2, db2 = backward(x, target, z1, h, out, W2)
    for name, grad, knob in [("dW1", dW1, W1), ("db1", db1, b1), ("dW2", dW2, W2), ("db2", db2, b2)]:
        print(name, grad.shape, "vs knob", knob.shape, "-> same shape:", grad.shape == knob.shape)
    print("dW1[0][:3] =", np.round(dW1[0][:3], 8), "\n")

    # --- Part 5: gradient check — analytic vs numerical --------------------
    knobs = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    analytic = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    # np.argmax finds the W1 entry with the steepest slope — the best single entry to check.
    steepest = tuple(int(v) for v in np.unravel_index(int(np.argmax(np.abs(dW1))), dW1.shape))
    to_check = [("W1", steepest), ("W1", (0, 0)), ("W1", (17, 42)), ("b1", (5,)),
                ("W2", (20, 7)), ("W2", (100, 2)), ("b2", (4,)), ("b2", (9,))]
    worst_gap = 0.0
    for knob, index in to_check:
        hand, nudged = float(analytic[knob][index]), numerical_slope(knob, index, x, target, knobs)
        worst_gap = max(worst_gap, relative_gap(hand, nudged))
        print(f"{knob}{index}: analytic {hand: .8f}  numerical {nudged: .8f}"
              f"  gap {relative_gap(hand, nudged):.2e}")
    print("worst relative gap over", len(to_check), "entries =", f"{worst_gap:.2e}", "\n")

    # --- Part 6: break the backward pass on purpose -----------------------
    delta2 = 2.0 * (out - target) / out.size     # forget the ReLU gate on purpose:
    dW1_no_gate = x.T @ (delta2 @ W2.T)          # this line is missing the  * (z1 > 0)
    spoiled, disagreements = float((np.abs(dW1_no_gate - dW1) > 1e-9).mean()), 0
    # A unit that was on for every image hides this bug, so we look at three entries.
    for index in [steepest, (0, 0), (17, 42)]:
        gap = relative_gap(float(dW1_no_gate[index]), numerical_slope("W1", index, x, target, knobs))
        disagreements += int(gap > 1e-3)
        print(f"no-gate dW1{index}: {float(dW1_no_gate[index]): .8f}  gap {gap:.2e}"
              f"  (this unit was on for {int((z1[:, index[1]] > 0).sum())} of 4 images)")
    print("the bug spoils", spoiled, "of dW1, caught on", disagreements, "of the 3 entries")

    # --- Self-check: one boolean per claim; pinned numbers come from a real run of this file ---
    chain_ok = abs(chain_product - 18.0) < 1e-12 and abs(downstream - 2.0) < 1e-12
    golf_ok = (2.0 - 5.0) ** 2 == 9.0 and (4.0 - 5.0) ** 2 == 1.0   # lesson: L = 9, then 1
    gate_ok = (np.array_equal(z_fwd, [2.0, -1.0, 3.0, -0.5])            # the lesson's inputs
               and np.array_equal(upstream_delta, [0.4, -0.7, 0.9, -0.2])
               and np.array_equal(gated_delta, [0.4, 0.0, 0.9, 0.0]))   # and its outgoing delta
    survivors_ok = predicted_survivors == 2 and actual_survivors == 2
    shapes_ok = (dW1.shape == (784, 128) == W1.shape and db1.shape == (128,) == b1.shape
                 and dW2.shape == (128, 10) == W2.shape and db2.shape == (10,) == b2.shape)
    pinned_ok = (abs(loss - 0.92741551) < 1e-7 and on_count == 263 and steepest == (515, 82)
                 and relative_gap(float(dW1[steepest]), 0.15716058) < 1e-6
                 and relative_gap(float(dW1[0, 0]), -0.0006150285) < 1e-6)
    check_ok = worst_gap < 1e-5       # two independent code paths agree on all 8 entries
    bug_caught = disagreements == 1 and abs(spoiled - 0.6953125) < 1e-9

    if (chain_ok and golf_ok and gate_ok and survivors_ok and shapes_ok
            and pinned_ok and check_ok and bug_caught):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected chain 18.0 and box 2.0, losses 9 and 1, outgoing delta "
              "[0.4, 0, 0.9, 0] with 2 survivors, dW1 shaped (784, 128), loss 0.92741551 with 263 units "
              "on, steepest dW1 0.15716058 at (515, 82), gap < 1e-5, and the no-gate bug to spoil "
              "0.6953125 of dW1, caught on 1 of 3 entries")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert chain_ok, "slopes 2, 3, 3 must multiply to 18.0, and upstream 4 x local 0.5 to 2.0"
    assert golf_ok, "the squared miss must be 9 for guess 2 and 1 for guess 4"
    assert gate_ok, "pushing [0.4, -0.7, 0.9, -0.2] back must give exactly [0.4, 0, 0.9, 0]"
    assert survivors_ok, "2 of the 4 entries had a positive forward input, so 2 survive"
    assert shapes_ok, "every gradient must have the same shape as the knob it corrects"
    assert pinned_ok, "loss 0.92741551, 263 units on, steepest dW1 0.15716058 at (515, 82)"
    assert check_ok, "the hand-wired gradient must match the numerical one to better than 1e-5"
    assert bug_caught, "dropping the ReLU gate must spoil 0.6953125 of dW1 and be caught once"
