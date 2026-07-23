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
    # --- Part 1: build a real 2-layer network and push one input all the way down ---
    print("Part 1 — the full forward pass (watch the length change 3 -> 4 -> 2)")

    # The input: a list of 3 numbers describing one example.
    x = np.array([1.0, 2.0, 3.0])
    print(f"  input x has length {x.shape[0]}")

    # Hidden layer: 4 neurons, each reading 3 inputs -> W1 is shape [4, 3], b1 length 4.
    W1 = np.array([
        [0.2, -0.1, 0.5],   # neuron 1's weights
        [0.0,  0.3, -0.2],  # neuron 2's weights
        [0.4,  0.1, 0.0],   # neuron 3's weights
        [-0.3, 0.2, 0.6],   # neuron 4's weights
    ])
    b1 = np.array([0.0, 0.0, 0.0, 0.0])  # one bias per neuron (kept 0 here for clarity)

    # Output layer: 2 neurons, each reading the hidden layer's 4 outputs -> W2 shape [2, 4].
    # Chaining rule: the hidden layer outputs 4 numbers, so W2 MUST have 4 columns.
    W2 = np.array([
        [0.5,  0.5, 0.0, 0.5],  # output neuron 1's weights
        [-0.5, 0.0, 1.0, 0.5],  # output neuron 2's weights
    ])
    b2 = np.array([0.0, 0.0])  # one bias per output neuron

    # Run the hidden layer WITH the ReLU bend.
    a1 = layer(x, W1, b1, relu)
    print(f"  after hidden layer: a1 has length {a1.shape[0]}  (4 neurons -> 4 numbers)")

    # Run the output layer with NO bend -> the raw prediction (as in today's lesson).
    yhat = layer(a1, W2, b2, identity)
    print(f"  final prediction yhat has length {yhat.shape[0]}  (2 output neurons)")
    print(f"  yhat = {yhat}")

    # The lesson's promise: the vector length is 3 in, 4 in the middle, 2 out.
    assert x.shape == (3,), "input should be length 3"
    assert a1.shape == (4,), "hidden output should be length 4"
    assert yhat.shape == (2,), "final prediction should be length 2"

    # --- Part 2: the linear collapse — two straight layers fold into one ---
    # We reuse the lesson's tiny by-hand example so you can check it on paper.
    print("\nPart 2 — the collapse (no bend) vs the fix (add a bend)")
    xc = np.array([1, 2])                    # tiny input from the lesson
    W1c = np.array([[1, 2], [0, 1]])         # first station
    W2c = np.array([[1, 0], [3, 1]])         # second station

    # Two-step path, NO bend anywhere: run x through W1 then W2.
    two_step = W2c @ (W1c @ xc)
    # One combined grid: multiply the two weight grids first, then apply once.
    combined = W2c @ W1c
    one_step = combined @ xc
    print(f"  combined grid W2 @ W1 = {combined.tolist()}")
    print(f"  two-step (no bend)    = {two_step.tolist()}")
    print(f"  one combined layer    = {one_step.tolist()}")

    # The collapse: with NO bend, both paths land on the exact same numbers.
    assert np.array_equal(two_step, one_step), "no-bend layers should collapse into one"

    # The fix: put a ReLU bend BETWEEN the layers -> the collapse breaks.
    bent = W2c @ relu(W1c @ xc)
    print(f"  with a ReLU bend      = {bent.tolist()}  (no longer matches the single grid)")
    # Both W1c@xc entries happen to be positive here, so ReLU changes nothing on THIS
    # input — so we prove the break with an input that has a negative intermediate.
    xn = np.array([-3, 1])                   # this makes W1@x have a negative entry
    two_step_n = W2c @ (W1c @ xn)            # no bend: still one linear layer
    bent_n = W2c @ relu(W1c @ xn)            # with bend: negative gets zeroed
    print(f"  x=[-3,1]: no-bend = {two_step_n.tolist()}, with-bend = {bent_n.tolist()}")
    assert not np.array_equal(two_step_n, bent_n), "the bend must change the result"

    # Return the numbers the self-check will verify against the lesson.
    return combined.tolist(), two_step.tolist(), one_step.tolist()


if __name__ == "__main__":
    combined, two_step, one_step = forward_pass()

    # The lesson's "What you should see" states the collapse demo exactly:
    #   W2 @ W1 = [[1, 2], [3, 7]], and both paths give [5, 17].
    expected_combined = [[1, 2], [3, 7]]
    expected_result = [5, 17]

    if combined == expected_combined and two_step == expected_result and one_step == expected_result:
        print("\n✅ you got it")
    else:
        print(f"\n❌ not yet — expected combined {expected_combined} and result {expected_result}, "
              f"got combined {combined}, two_step {two_step}, one_step {one_step}")
        raise SystemExit(1)
