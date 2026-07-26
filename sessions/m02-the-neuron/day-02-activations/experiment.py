# day-02-activations — experiment
#
# Today's big idea in two lines of output:
#   Two straight-line layers with NO bend between them fold into ONE layer.
#   Slip a bend (ReLU) in the middle, and the fold breaks — that is why "deep" works.
#
# This script (1) shows the five activation "bends" over a small grid, then
# (2) proves the linear collapse and (3) breaks it with a ReLU.
# Run it:  python3 sessions/m02-the-neuron/day-02-activations/experiment.py

import numpy as np  # numpy gives us arrays and matrix multiply (@)


# ---- The five bends -------------------------------------------------------
# Each one takes a number (or an array of numbers) and returns the "bent" value.

def step(z):
    # Hard on/off switch: 1 if z >= 0, else 0. Flat everywhere (slope 0).
    return (z >= 0).astype(float)

def relu(z):
    # One-way valve: pass positives, block negatives (turn them into 0).
    # np.maximum(0, z) keeps the input's dtype, so integer input prints as integers
    # (relu(np.array([-3, -1, 0, 2, 5])) -> array([0, 0, 0, 2, 5])), which is exactly
    # what the lesson's demo shows.
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    # Like ReLU, but negatives leak through as a tiny trickle (alpha * z),
    # so the neuron keeps a small slope on the negative side and can't fully die.
    return np.where(z > 0, z, alpha * z)

def sigmoid(z):
    # Soft dimmer: squashes any number smoothly into the range (0, 1).
    return 1.0 / (1.0 + np.exp(-z))

def tanh(z):
    # Balanced see-saw: squashes into (-1, 1), and tanh(0) = 0 (zero-centered).
    return np.tanh(z)


if __name__ == "__main__":
    # --- Part 1: print all five bends over a small grid -------------------
    # A row of 13 evenly spaced inputs from -6 to +6 — enough to see each shape.
    grid = np.linspace(-6, 6, 13)

    print("input z :", np.round(grid, 2))
    print("step    :", np.round(step(grid), 3))         # flat 0, then jumps to flat 1
    print("relu    :", np.round(relu(grid), 3))          # negatives -> 0, positives pass
    print("leaky   :", np.round(leaky_relu(grid), 3))    # negatives -> tiny trickle
    print("sigmoid :", np.round(sigmoid(grid), 3))       # always inside (0, 1)
    print("tanh    :", np.round(tanh(grid), 3))          # inside (-1, 1), and 0 at 0

    # --- Part 2: the slope facts that seed today's failures ---------------
    # sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)); at z=0 that is 0.5*0.5 = 0.25.
    s0 = sigmoid(0.0)
    sigmoid_slope_at_0 = s0 * (1 - s0)
    print("\nsigmoid'(0) =", round(float(sigmoid_slope_at_0), 4),
          "(peak slope ~0.25 -> shrinks the backward signal in deep sigmoid nets)")
    # ReLU's slope on the negative side is exactly 0 -> the seed of a "dead" ReLU.
    # Leaky ReLU keeps a small slope alpha there -> the seed of the cure.
    print("ReLU slope for negatives = 0   (can go 'dead')")
    print("leaky slope for negatives = 0.01 (small alpha keeps it alive)")

    # --- Part 3: watch two linear layers collapse into one ----------------
    # These fixed weights are the lesson's "anchor pair".
    W1 = np.array([[1.0, 2.0], [0.0, 1.0]])
    W2 = np.array([[1.0, 0.0], [3.0, 1.0]])

    # Two input rows, chosen so that (a) each one gives a MIXED-SIGN pre-activation
    # x@W1 — so the ReLU in Part 4 clips one component and keeps the other, leaving a
    # non-trivial output instead of all zeros — and (b) they add up to exactly zero,
    # which makes the additivity test in Part 4 easy to read.
    x_a = np.array([[1.0, -3.0]])    # x_a@W1 = [ 1, -1]  -> relu -> [1, 0]
    x_b = np.array([[-1.0, 3.0]])    # x_b@W1 = [-1,  1]  -> relu -> [0, 1]

    # Path A: push x through layer 1, THEN through layer 2 (no bend between).
    two_layers = (x_a @ W1) @ W2
    # Path B: multiply the two weight matrices FIRST, then push x through once.
    one_layer = x_a @ (W1 @ W2)

    print("\nW1 @ W2 =\n", (W1 @ W2).astype(int))     # the single combined matrix
    print("(x_a@W1)@W2 =", np.round(two_layers, 4))    # two layers...
    print("x_a@(W1@W2) =", np.round(one_layer, 4))     # ...land on the SAME numbers

    # --- Part 4: slip a ReLU in the middle and break the collapse ---------
    # How do we PROVE that no single matrix can reproduce the bent path? Not by showing
    # that one output changed — a different matrix could still match that one input. We
    # use the defining property of a linear map: it ADDS UP.
    #     linear:  g(x_a + x_b) == g(x_a) + g(x_b)   -- always, for every matrix
    # So if the bent path fails that test even once, NO single matrix can reproduce it.
    def g(x):
        return x @ W1 @ W2                 # straight path: no bend

    def f(x):
        return relu(x @ W1) @ W2           # bent path: a ReLU in the middle

    x_sum = x_a + x_b                      # = [[0., 0.]]
    lin_adds_up = np.allclose(g(x_sum), g(x_a) + g(x_b))
    bent_adds_up = np.allclose(f(x_sum), f(x_a) + f(x_b))

    print("\nstraight path  g(x_a)+g(x_b) =", np.round(g(x_a) + g(x_b), 4),
          " g(x_a+x_b) =", np.round(g(x_sum), 4), " -> adds up:", lin_adds_up)
    print("bent path      f(x_a)+f(x_b) =", np.round(f(x_a) + f(x_b), 4),
          " f(x_a+x_b) =", np.round(f(x_sum), 4), " -> adds up:", bent_adds_up)
    print("f(x_a) =", np.round(f(x_a), 4), " f(x_b) =", np.round(f(x_b), 4),
          "(each keeps one component and clips the other)")
    print("-> the bent path does NOT add up, and every single matrix does,",
          "so no single matrix can reproduce it: the bend cannot be folded flat")

    # --- Self-check: assert the lesson's stated expected values -----------
    expected_W = np.array([[7, 2], [3, 1]])              # the lesson says expect [[7,2],[3,1]]
    collapse_holds = np.allclose(two_layers, one_layer)   # two linear layers == one
    W_matches = np.array_equal((W1 @ W2).astype(int), expected_W)
    sigmoid_ok = abs(sigmoid_slope_at_0 - 0.25) < 1e-9    # sigmoid'(0) == 0.25
    bend_breaks = lin_adds_up and not bent_adds_up        # only the bent path fails additivity
    values_match = (np.allclose(f(x_a) + f(x_b), [[4.0, 1.0]])
                    and np.allclose(f(x_sum), [[0.0, 0.0]]))  # the lesson's printed numbers

    if collapse_holds and W_matches and sigmoid_ok and bend_breaks and values_match:
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected W1@W2 == [[7,2],[3,1]], "
              "(x_a@W1)@W2 == x_a@(W1@W2), sigmoid'(0) == 0.25, the straight path to "
              "add up, and the bent path to give [[4,1]] vs [[0,0]]")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert W_matches, "W1@W2 should be [[7,2],[3,1]]"
    assert collapse_holds, "(x_a@W1)@W2 must equal x_a@(W1@W2) — two linear layers are one"
    assert sigmoid_ok, "sigmoid'(0) should be 0.25"
    assert lin_adds_up, "a straight (linear) path must always add up"
    assert not bent_adds_up, "a ReLU between the layers must break additivity"
    assert values_match, "expected f(x_a)+f(x_b) == [[4,1]] and f(x_a+x_b) == [[0,0]]"
