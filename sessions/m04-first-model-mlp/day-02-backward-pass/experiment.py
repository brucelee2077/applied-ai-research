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
    return np.maximum(0, z)

def relu_gate(z):   # the saved mask: a unit is ON only if its forward input was STRICTLY above 0
    return z > 0

def forward(x, W1, b1, W2, b2):
    """One forward trip. Returns day 1's 3-tuple UNCHANGED — (z1, hidden, out) — so the line
    `z1, h, out = forward(...)` means the same slots it meant yesterday. The loss is scored
    separately, by squared_loss below, instead of being wedged into the front of this tuple."""
    z1 = x @ W1 + b1                             # layer 1 raw score, before the bend
    h = relu(z1)                                 # the bend
    out = h @ W2 + b2                            # layer 2 raw score = the 10 guesses (day 1's logits)
    return z1, h, out

def squared_loss(out, target):
    # MSE, and its denominator, named: the mean runs over every ELEMENT of the error, so it
    # divides by out.size (batch x 10) — NOT by the row count. Day 3 keeps this chain rule but
    # switches to a different, named reduction, and prints the factor between the two.
    return float(((out - target) ** 2).mean())   # one honest number; lower is better

def backward(x, target, z1, h, out, W2):   # the backward pass by hand; no autograd anywhere
    n_elements = out.size   # the SAME denominator squared_loss used: elements, not rows
    delta2 = 2.0 * (out - target) / n_elements   # seed delta (later days call this d_out)
    dW2 = h.T @ delta2         # weight slope = delta times the layer's SAVED input
    db2 = delta2.sum(axis=0)   # bias slope = delta itself, added up over the batch rows
    # Pass blame back a layer, then through the ReLU gate: on-units keep their slope, off-units 0.
    delta1 = (delta2 @ W2.T) * relu_gate(z1)
    dW1 = x.T @ delta1
    db1 = delta1.sum(axis=0)
    return dW1, db1, dW2, db2

def numerical_slope(knob, index, x, target, knobs, eps=1e-5):
    # The slow honest second opinion: nudge ONE number by +eps and -eps, re-run the forward pass.
    # (The lesson writes this step size as h; it is eps here because h is the hidden layer.)
    up = {name: value.copy() for name, value in knobs.items()}
    dn = {name: value.copy() for name, value in knobs.items()}
    up[knob][index] += eps
    dn[knob][index] -= eps
    return (squared_loss(forward(x, **up)[2], target)
            - squared_loss(forward(x, **dn)[2], target)) / (2.0 * eps)

def relative_gap(a, b):   # how far apart two numbers are, measured against their own size
    return abs(a - b) / max(abs(a), abs(b), 1e-8)

if __name__ == "__main__":
    # --- Part 1: the chain rule is a multiply ------------------------------
    chain_product = float(np.prod([2.0, 3.0, 3.0]))   # the lesson's three dominoes, multiplied
    upstream, local = 4.0, 0.5                # one box from the lesson's wiring picture
    downstream = upstream * local             # multiply-and-pass-back, in one line
    print("slopes 2, 3, 3 -> chain =", chain_product, "|  one box: upstream", upstream,
          "x local", local, "-> downstream", downstream)
    hole_target = 5.0                                        # the lesson's target, written once
    miss_2 = (2.0 - hole_target) ** 2                        # squared miss for the far guess
    miss_4 = (4.0 - hole_target) ** 2                        # and for the closer guess
    print("squared miss: guess 2 vs target 5 ->", miss_2, "| guess 4 ->", miss_4)

    # --- Part 2: push a gradient back through a ReLU gate ------------------
    # The lesson's four entries, plus a fifth sitting exactly ON the hinge (z = 0), so that ONE
    # gate call decides both the printed exhibit and the boundary case.
    z_all = np.array([2.0, -1.0, 3.0, -0.5, 0.0])
    delta_all = np.array([0.4, -0.7, 0.9, -0.2, 0.6])
    gate_all = relu_gate(z_all)                         # the saved mask: was this unit on?
    gated_all = delta_all * gate_all + 0.0   # the "+ 0.0" only turns -0.0 into 0.0 for printing
    z_fwd, upstream_delta = z_all[:4], delta_all[:4]    # what the forward pass produced here
    gate, gated_delta = gate_all[:4], gated_all[:4]     # blame arriving from the right, then gated
    predicted_survivors = int(gate.sum())    # the prediction is read off the inputs, not guessed
    actual_survivors = int(np.count_nonzero(gated_delta))
    # Each printed number is BOUND first and the self-check reads the same name, so the line you
    # read and the line that is checked are one value, not two expressions that agree today.
    shown_gate = gate.astype(int)
    shown_hinge_z = float(z_all[4])
    shown_hinge_delta = float(delta_all[4])
    shown_hinge_out = float(gated_all[4])
    print("\nforward inputs :", z_fwd, "\nincoming delta :", upstream_delta)
    print("gate (on = 1)  :", shown_gate, "-> we predict", predicted_survivors, "of 4 live")
    print("outgoing delta :", gated_delta, "-> survived:", actual_survivors)
    # The fifth entry: z = 0 counts as OFF, so its 0.6 of blame must vanish too.
    print("on the hinge   : z =", shown_hinge_z, "with delta", shown_hinge_delta, "-> outgoing",
          shown_hinge_out, "(z = 0 is OFF, not on)")

    # --- Part 3: forward pass, saving the breadcrumbs ----------------------
    rng = np.random.default_rng(0)
    n_in, n_hidden, n_out, batch = 784, 128, 10, 4
    # MNIST is not on this machine and cannot be downloaded, so these 4 "images" are a seeded
    # stand-in: pixel-like numbers in [0, 1). The shapes are the real ones, 784 -> 128 -> 10.
    x = rng.random((batch, n_in))
    target = np.eye(n_out)[[7, 2, 1, 0]]      # one row per image, 1.0 at the true digit
    # He initialization: a spread of sqrt(2 / inputs), so ReLU units are not born dead.
    # Layout is day 1's convention, (in, out): W1 is (784, 128) and W2 is (128, 10).
    W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
    W2 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(2.0 / n_hidden)
    b1, b2 = np.zeros(n_hidden), np.zeros(n_out)   # the two biases start at zero
    z1, h, out = forward(x, W1, b1, W2, b2)   # day 1's tuple, same order
    loss = squared_loss(out, target)
    on_count = int(relu_gate(z1).sum())       # a breadcrumb: how many units were on
    x_shape, z1_shape, h_shape, out_shape = x.shape, z1.shape, h.shape, out.shape
    z1_size = z1.size
    shown_out_row0 = np.round(out[0][:4], 4)
    shown_target_row0 = target[0][:4]
    shown_loss = round(loss, 8)
    print("\nx", x_shape, "-> z1", z1_shape, "-> h", h_shape, "-> out", out_shape)
    print("out[0][:4] =", shown_out_row0, " target[0][:4] =", shown_target_row0)
    print("hidden units on (z1 > 0):", on_count, "of", z1_size, " loss =", shown_loss)
    # The denominator, made visible before it is used. This loss is a mean over ELEMENTS, so a
    # 4-row batch of 10 scores divides by 40, not by 4. Seeding the backward pass with the row
    # count instead — the mistake that reads as "average over the batch" — would multiply every
    # delta by n_out. Here that factor is 10, and it is asserted below.
    rows_seed = 2.0 * (out - target) / out.shape[0]        # the wrong denominator, written out
    element_seed = 2.0 * (out - target) / out.size         # the one backward() really uses
    seed_ratio = float(np.abs(rows_seed / element_seed).max())
    out_elements, out_rows = out.size, out.shape[0]
    shown_seed_ratio = round(seed_ratio, 1)
    print("MSE denominator :", out_elements, "elements =", out_rows, "rows x", n_out,
          "classes -> dividing by rows instead would make every delta", shown_seed_ratio,
          "x too big")

    # --- Part 4: backward pass by hand, and the free shape check -----------
    dW1, db1, dW2, db2 = backward(x, target, z1, h, out, W2)
    # Bind the shape triples ONCE, print them, and let the self-check read the same rows.
    grad_rows = [(name, grad.shape, knob.shape, grad.shape == knob.shape)
                 for name, grad, knob in [("dW1", dW1, W1), ("db1", db1, b1),
                                          ("dW2", dW2, W2), ("db2", db2, b2)]]
    for name, grad_shape, knob_shape, same_shape in grad_rows:
        print(name, grad_shape, "vs knob", knob_shape, "-> same shape:", same_shape)
    shown_dW1_row0 = np.round(dW1[0][:3], 8)
    print("dW1[0][:3] =", shown_dW1_row0, "\n")
    # The hinge case again, but through the REAL backward pass: a hand-built net whose z1 is
    # exactly [2, 0], so the second unit sits on the bend and must receive no blame at all.
    x_edge, target_edge = np.array([[1.0, 1.0]]), np.zeros((1, 1))
    W1_edge, b1_edge = np.array([[1.0, 2.0], [1.0, -2.0]]), np.zeros(2)   # column 2 cancels to 0
    W2_edge, b2_edge = np.array([[1.0], [1.0]]), np.zeros(1)
    z1_edge, h_edge, out_edge = forward(x_edge, W1_edge, b1_edge, W2_edge, b2_edge)
    loss_edge = squared_loss(out_edge, target_edge)
    dW1_edge = backward(x_edge, target_edge, z1_edge, h_edge, out_edge, W2_edge)[0]
    shown_edge_z1 = z1_edge[0]
    shown_edge_live = dW1_edge[:, 0]     # the unit whose z1 is 2.0 — it must collect blame
    shown_edge_dead = dW1_edge[:, 1]     # the unit sitting exactly on the hinge — it must not
    print("hinge net: z1 =", shown_edge_z1, "loss =", loss_edge, "-> dW1 for the live unit",
          shown_edge_live, "and for the z=0 unit", shown_edge_dead, "\n")

    # --- Part 5: gradient check — analytic vs numerical --------------------
    knobs = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    analytic = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    # np.argmax finds the W1 entry with the steepest slope — the best single entry to check.
    steepest = tuple(int(v) for v in np.unravel_index(int(np.argmax(np.abs(dW1))), dW1.shape))
    to_check = [("W1", steepest), ("W1", (0, 0)), ("W1", (17, 42)), ("b1", (5,)),
                ("W2", (20, 7)), ("W2", (100, 2)), ("b2", (4,)), ("b2", (9,))]
    worst_gap = 0.0
    checked_gaps = []
    for knob, index in to_check:
        hand, nudged = float(analytic[knob][index]), numerical_slope(knob, index, x, target, knobs)
        gap = relative_gap(hand, nudged)   # computed ONCE: the printed gap is the checked gap
        checked_gaps.append(gap)
        worst_gap = max(worst_gap, gap)
        print(f"{knob}{index}: analytic {hand: .8f}  numerical {nudged: .8f}"
              f"  gap {gap:.2e}")
    n_checked = len(to_check)
    shown_worst_gap = f"{worst_gap:.2e}"     # the rendered string is a print site too
    print("worst relative gap over", n_checked, "entries =", shown_worst_gap, "\n")

    # --- Part 6: break the backward pass on purpose -----------------------
    delta2 = element_seed                        # forget the ReLU gate on purpose:
    dW1_no_gate = x.T @ (delta2 @ W2.T)          # this line is missing the  * (z1 > 0)
    spoiled, disagreements = float((np.abs(dW1_no_gate - dW1) > 1e-9).mean()), 0
    # A unit that was on for every image hides this bug, so we look at three entries.
    no_gate_values, no_gate_on_counts = [], []
    for index in [steepest, (0, 0), (17, 42)]:
        gap = relative_gap(float(dW1_no_gate[index]), numerical_slope("W1", index, x, target, knobs))
        disagreements += int(gap > 1e-3)
        shown_no_gate = float(dW1_no_gate[index])
        shown_on_for_unit = int(relu_gate(z1[:, index[1]]).sum())
        no_gate_values.append(shown_no_gate)
        no_gate_on_counts.append(shown_on_for_unit)
        print(f"no-gate dW1{index}: {shown_no_gate: .8f}  gap {gap:.2e}"
              f"  (this unit was on for {shown_on_for_unit} of 4 images)")
    print("the bug spoils", spoiled, "of dW1, caught on", disagreements, "of the 3 entries")

    # --- Self-check: one boolean per claim; pinned numbers come from a real run of this file ---
    # Every check reads the SAME name that was printed, so a corrupted printed number is a
    # corrupted checked number.
    chain_ok = abs(chain_product - 18.0) < 1e-12 and abs(downstream - 2.0) < 1e-12
    golf_ok = miss_2 == 9.0 and miss_4 == 1.0   # the two numbers we PRINTED; lesson: L = 9, then 1
    gate_ok = (np.array_equal(z_fwd, [2.0, -1.0, 3.0, -0.5])            # the lesson's inputs
               and np.array_equal(upstream_delta, [0.4, -0.7, 0.9, -0.2])
               and np.array_equal(shown_gate, [1, 0, 1, 0])             # the printed mask
               and np.array_equal(gated_delta, [0.4, 0.0, 0.9, 0.0]))   # and its outgoing delta
    # The strictness itself: with a ">=" gate the on-the-hinge entry would keep its 0.6, and the
    # hinge net's dead column would collect blame instead of the exact zeros it must get.
    edge_ok = (bool(gate_all[4]) is False and shown_hinge_z == 0.0
               and shown_hinge_delta == 0.6 and shown_hinge_out == 0.0)
    edge_back_ok = (loss_edge == 4.0 and np.array_equal(z1_edge, [[2.0, 0.0]])
                    and np.array_equal(shown_edge_z1, [2.0, 0.0])
                    and np.array_equal(shown_edge_live, [4.0, 4.0])
                    and np.array_equal(shown_edge_dead, [0.0, 0.0])
                    and np.array_equal(dW1_edge, [[4.0, 0.0], [4.0, 0.0]]))
    survivors_ok = predicted_survivors == 2 and actual_survivors == 2
    shapes_ok = (all(same_shape for _, _, _, same_shape in grad_rows)
                 and [(name, grad_shape, knob_shape)
                      for name, grad_shape, knob_shape, _ in grad_rows]
                 == [("dW1", (784, 128), (784, 128)), ("db1", (128,), (128,)),
                     ("dW2", (128, 10), (128, 10)), ("db2", (10,), (10,))]
                 and x_shape == (4, 784) and z1_shape == (4, 128)
                 and h_shape == (4, 128) and out_shape == (4, 10))
    # The reduction, pinned: this loss divides by ELEMENTS. 40 of them, from 4 rows of 10, so
    # the row-count spelling is exactly n_out = 10 times bigger. Day 3 keeps the same chain rule
    # and deliberately switches to that other reduction — which is why the factor is named here.
    reduction_ok = (out_elements == 40 and out_rows == 4 and z1_size == 512
                    and shown_seed_ratio == 10.0
                    and np.allclose(rows_seed, n_out * element_seed, rtol=1e-12, atol=0.0))
    pinned_ok = (abs(shown_loss - 0.92741551) < 1e-7 and on_count == 263
                 and steepest == (515, 82)
                 and relative_gap(float(dW1[steepest]), 0.15716058) < 1e-6
                 and relative_gap(float(dW1[0, 0]), -0.0006150285) < 1e-6
                 # the two printed rows of numbers, pinned to what this seed really produces
                 and np.allclose(shown_out_row0, [-0.7923, 0.9492, -1.0002, 0.1708], atol=5e-5)
                 and np.array_equal(shown_target_row0, [0.0, 0.0, 0.0, 0.0])
                 and np.allclose(shown_dW1_row0, [-0.00061503, 0.01880515, 0.00692428],
                                 rtol=0.0, atol=5e-9))
    # Two independent code paths agree on all 8 entries — checked on the rendered string too,
    # so the summary line cannot say one thing while the check reads another.
    check_ok = (worst_gap < 1e-5 and n_checked == 8 and len(checked_gaps) == 8
                and max(checked_gaps) == worst_gap and float(shown_worst_gap) < 1e-5)
    # The bug only shows on a unit that was OFF for some image: entry 2 was on for 1 of 4 and is
    # the one caught, while the other two were on for all 4 and hide it.
    bug_caught = (disagreements == 1 and abs(spoiled - 0.6953125) < 1e-9
                  and no_gate_on_counts == [4, 1, 4]
                  and disagreements == sum(1 for c in no_gate_on_counts if c < batch)
                  and np.allclose(no_gate_values, [0.15716058, 0.01715807, 0.09938535],
                                  rtol=1e-6, atol=0.0))

    if (chain_ok and golf_ok and gate_ok and edge_ok and edge_back_ok and survivors_ok
            and shapes_ok and reduction_ok and pinned_ok and check_ok and bug_caught):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected chain 18.0 and box 2.0, losses 9 and 1, outgoing delta "
              "[0.4, 0, 0.9, 0] with 2 survivors, a forward input of exactly 0 to send back 0.0 "
              "(hinge net: loss 4.0, dW1 [[4, 0], [4, 0]]), dW1 shaped (784, 128), the loss to "
              "divide by 40 ELEMENTS (4 rows x 10) so the row-count seed is 10x too big, loss "
              "0.92741551 with 263 units on, steepest dW1 0.15716058 at (515, 82), gap < 1e-5, "
              "and the no-gate bug to spoil 0.6953125 of dW1, caught on 1 of 3 entries")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert chain_ok, "slopes 2, 3, 3 must multiply to 18.0, and upstream 4 x local 0.5 to 2.0"
    assert golf_ok, "the squared miss must be 9 for guess 2 and 1 for guess 4"
    assert gate_ok, "pushing [0.4, -0.7, 0.9, -0.2] back must give exactly [0.4, 0, 0.9, 0]"
    assert edge_ok, "a forward input of exactly 0 is OFF, so its outgoing delta must be 0.0"
    assert edge_back_ok, "the hinge net must give loss 4.0 and dW1 [[4, 0], [4, 0]] — no blame at z=0"
    assert survivors_ok, "2 of the 4 entries had a positive forward input, so 2 survive"
    assert shapes_ok, "every gradient must have the same shape as the knob it corrects"
    assert reduction_ok, ("this loss must divide by 40 ELEMENTS (4 rows x 10 classes), so the "
                          "row-count seed is exactly 10x too big")
    assert pinned_ok, "loss 0.92741551, 263 units on, steepest dW1 0.15716058 at (515, 82)"
    assert check_ok, "the hand-wired gradient must match the numerical one to better than 1e-5"
    assert bug_caught, "dropping the ReLU gate must spoil 0.6953125 of dW1 and be caught once"
