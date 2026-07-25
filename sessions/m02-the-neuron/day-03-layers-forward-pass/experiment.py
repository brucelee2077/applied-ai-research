# day-03-layers-forward-pass — experiment
#
# What this shows (one sentence):
#   A 2-layer forward pass reshapes a length-3 input into a length-2 prediction,
#   and TWO layers with NO bend between them collapse into ONE single layer —
#   a bend (ReLU) is what stops that collapse.
#
# Run it:  python3 sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py

import numpy as np  # numpy gives us the matrix-vector multiply "@" (the "one beep")


# --- the activation "bend" from Day 2: ReLU keeps positives, zeroes negatives ---
def relu(z):
    # np.maximum(0, z) compares each entry of z with 0 and keeps the bigger one.
    # This is "elementwise": every number in the list is bent on its own, no mixing.
    return np.maximum(0, z)


# --- identity: a "no bend" pass-through, used to demo the collapse ---
def identity(z):
    # It hands the list back unchanged — a straight line, no bend at all.
    return z


# --- one dense layer = one multiply-plus-bias, then a bend ---
def layer(x, W, b, f):
    # Step 1: W @ x mixes the input — every neuron's weighted sum at once.
    # Step 2: + b shifts each neuron's sum by its own bias.
    z = W @ x + b
    # Print the shape so we can WATCH the vector length change station to station.
    print(f"    layer: W @ x + b has shape {z.shape}  (one number per neuron)")
    # Step 3: apply the bend f to each entry on its own -> the layer's output list a.
    return f(z)


def forward_pass():
    # --- Part 1: the lesson's own 2-layer network, pushed end to end -------------
    print("Part 1 — the full forward pass (watch the length change 3 -> 4 -> 2)")

    # The input: a list of 3 numbers describing one example (same x as the lesson).
    x = np.array([1.0, 2.0, 3.0])
    print(f"  input x has length {x.shape[0]}")

    # Hidden layer: 4 neurons, each reading 3 inputs -> W1 is shape [4, 3], b1 length 4.
    W1 = np.array([
        [0.2, -0.1, 0.5],   # neuron 1's weights
        [0.0,  0.3, -0.1],  # neuron 2's weights
        [0.4,  0.1, 0.0],   # neuron 3's weights
        [-0.3, 0.2, 0.6],   # neuron 4's weights
    ])
    # One bias per neuron. Neuron 2 and neuron 4 get negative biases, which is what
    # pushes their totals below zero so you can SEE the ReLU bend actually bite.
    b1 = np.array([0.0, -0.5, 0.0, -2.0])

    # Output layer: 2 neurons, each reading the hidden layer's 4 outputs -> W2 shape [2, 4].
    # Chaining rule: the hidden layer outputs 4 numbers, so W2 MUST have 4 columns.
    W2 = np.array([
        [0.5,  0.5, 0.0, 0.5],  # output neuron 1's weights
        [-0.5, 0.0, 1.0, 0.5],  # output neuron 2's weights
    ])
    b2 = np.array([0.0, 0.0])  # one bias per output neuron

    # The raw weighted sums BEFORE the bend, so you can compare z1 with a1 by eye.
    z1 = W1 @ x + b1
    print(f"  z1 (raw, before the bend) = {np.round(z1, 2)}")

    # Run the hidden layer WITH the ReLU bend.
    a1 = layer(x, W1, b1, relu)
    print(f"  a1 (after the bend)       = {np.round(a1, 2)}   <- two entries flattened to 0")

    # Run the output layer with NO bend -> the raw prediction (as in today's lesson).
    yhat = layer(a1, W2, b2, identity)
    print(f"  final prediction yhat has length {yhat.shape[0]}  (2 output neurons)")
    print(f"  yhat = {np.round(yhat, 2)}")

    # The lesson's promise: the vector length is 3 in, 4 in the middle, 2 out,
    # and the numbers are exactly the ones printed on the page.
    assert x.shape == (3,), "input should be length 3"
    assert a1.shape == (4,), "hidden output should be length 4"
    assert yhat.shape == (2,), "final prediction should be length 2"
    assert np.allclose(a1, [1.5, 0.0, 0.6, 0.0]), "hidden output should match the lesson"
    assert np.allclose(yhat, [0.75, -0.15]), "prediction should match the lesson"

    # --- Part 2: the linear collapse — two straight layers fold into one ---------
    # We reuse the lesson's tiny by-hand pair so you can check it on paper.
    print("\nPart 2 — the collapse (no bend) vs the fix (add a bend)")
    A = np.array([[1, 2], [0, 1]])   # tiny station 1
    B = np.array([[1, 0], [3, 1]])   # tiny station 2
    u = np.array([-3, 1])            # this input makes A @ u contain a NEGATIVE entry

    # Two-step path, NO bend anywhere: run u through A then B.
    no_bend = B @ (A @ u)
    # One combined grid: multiply the two weight grids first, then apply once.
    combined_grid = B @ A
    one_step = combined_grid @ u
    print(f"  A @ u                 = {(A @ u).tolist()}   <- note the negative entry")
    print(f"  combined grid B @ A   = {combined_grid.tolist()}")
    print(f"  two-step (no bend)    = {no_bend.tolist()}")
    print(f"  one combined layer    = {one_step.tolist()}")

    # The collapse: with NO bend, both paths land on the exact same numbers.
    assert np.array_equal(no_bend, one_step), "no-bend layers should collapse into one"

    # The fix: put a ReLU bend BETWEEN the layers -> the collapse breaks.
    with_bend = B @ relu(A @ u)
    print(f"  with a ReLU bend      = {with_bend.tolist()}  (DIFFERENT — the collapse is broken)")
    assert not np.array_equal(with_bend, one_step), "the bend must change the result"

    # The trap worth knowing: if every middle number is already positive, ReLU has
    # nothing to flatten, so the bend changes NOTHING and a careless test would
    # "pass" for the wrong reason. Watch it happen with u = [1, 2].
    u_pos = np.array([1, 2])
    print(f"\n  trap check with u = [1, 2]: A @ u = {(A @ u_pos).tolist()} (both positive)")
    print(f"    no bend   = {(B @ (A @ u_pos)).tolist()}")
    print(f"    with bend = {(B @ relu(A @ u_pos)).tolist()}   <- IDENTICAL: relu had nothing to do")
    assert np.array_equal(B @ relu(A @ u_pos), B @ (A @ u_pos)), \
        "with all-positive middle numbers the bend is a no-op — that is the point"

    # Return the numbers the self-check will verify against the lesson.
    return combined_grid.tolist(), no_bend.tolist(), one_step.tolist(), with_bend.tolist()


if __name__ == "__main__":
    combined_grid, no_bend, one_step, with_bend = forward_pass()

    # The lesson's "What you should see" states the collapse demo exactly:
    #   B @ A = [[1, 2], [3, 7]]; both no-bend paths give [-1, -2]; with a bend, [0, 1].
    expected_grid = [[1, 2], [3, 7]]
    expected_collapsed = [-1, -2]
    expected_bent = [0, 1]

    if (combined_grid == expected_grid and no_bend == expected_collapsed
            and one_step == expected_collapsed and with_bend == expected_bent):
        print("\n✅ you got it")
    else:
        print(f"\n❌ not yet — expected grid {expected_grid}, collapsed {expected_collapsed}, "
              f"bent {expected_bent}; got grid {combined_grid}, no_bend {no_bend}, "
              f"one_step {one_step}, with_bend {with_bend}")
        raise SystemExit(1)
