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


# Every shape line layer() actually PRINTS gets kept here, so the self-check can pin
# the trace the learner reads instead of a second, separate copy of the same numbers.
SHAPE_TRACE_LINES = []


def printed_shape_trace():
    # Pull the shape back out of each printed line: "...has shape (4,)  (one number..."
    return [line.split("shape ")[1].split(" ")[0] for line in SHAPE_TRACE_LINES]


# --- one dense layer = one multiply-plus-bias, then a bend ---
def layer(x, W, b, f):
    # Step 1: W @ x mixes the input — every neuron's weighted sum at once.
    # Step 2: + b shifts each neuron's sum by its own bias.
    z = W @ x + b
    # Print the shape so we can WATCH the vector length change station to station.
    # We build the line first and keep it, so the assert further down checks the very
    # string that reached the screen.
    trace_line = f"    layer: W @ x + b has shape {z.shape}  (one number per neuron)"
    SHAPE_TRACE_LINES.append(trace_line)
    print(trace_line)
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
    # z1 is printed for you to compare with a1 by eye, so pin it too: ReLU clips both
    # negative entries to 0, which would otherwise hide HOW negative they were.
    assert np.allclose(z1, [1.5, -0.2, 0.6, -0.1]), "pre-activation z1 should match the lesson"
    assert np.allclose(a1, [1.5, 0.0, 0.6, 0.0]), "hidden output should match the lesson"
    assert np.allclose(yhat, [0.75, -0.15]), "prediction should match the lesson"
    # Pin the trace that was actually PRINTED, not just the shapes held in memory:
    # with the length-3 input, the screen must read 3 -> 4 -> 2.
    assert printed_shape_trace() == ["(4,)", "(2,)"], \
        "the printed shape trace should read 4 then 2 (3 -> 4 -> 2 with the input)"
    # b2 is all zeros, so its SHAPE cannot change any number above — pin it directly,
    # or "one bias per output neuron" could quietly become one bias for both.
    assert b2.shape == (W2.shape[0],), "one bias per output neuron"

    # Control probe: the same output layer, same a1, but a LOPSIDED bias. b2 stays
    # [0, 0] (the lesson's numbers are untouched) — this extra run is what makes the
    # per-neuron bias structure visible: 0.1 lifts neuron 1 only, -0.2 drops neuron 2 only.
    b2_probe = np.array([0.1, -0.2])
    yhat_probe = layer(a1, W2, b2_probe, identity)
    print(f"  control: same layer with b2 = [0.1, -0.2] -> yhat = {np.round(yhat_probe, 2)}"
          f"  (each bias shifted its OWN neuron)")
    assert np.allclose(yhat_probe, [0.85, -0.35]), \
        "each output bias should shift only its own neuron (0.75+0.1, -0.15-0.2)"

    # --- Part 2: the linear collapse — two straight layers fold into one ---------
    # We reuse the lesson's tiny by-hand pair so you can check it on paper. It is the
    # SAME numeric pair Day 2 used, under Day 2's names: A is Day 2's W1, B is Day 2's W2.
    print("\nPart 2 — the collapse (no bend) vs the fix (add a bend)")
    A = np.array([[1, 2], [0, 1]])   # tiny station 1 (Day 2 called this same grid W1)
    B = np.array([[1, 0], [3, 1]])   # tiny station 2 (Day 2 called this same grid W2)
    u = np.array([-3, 1])            # this input makes A @ u contain a NEGATIVE entry

    # Two-step path, NO bend anywhere: run u through A then B.
    no_bend = B @ (A @ u)            # Day 2 called this same two-station path two_layers
    # One combined grid: multiply the two weight grids first, then apply once.
    combined_grid = B @ A            # Day 2 called this same object combined_grid too
    one_step = combined_grid @ u     # Day 2 called this same one-station answer one_layer
    print(f"  A @ u                 = {(A @ u).tolist()}   <- note the negative entry")
    print(f"  combined grid B @ A   = {combined_grid.tolist()}")
    print(f"  two-step (no bend)    = {no_bend.tolist()}")
    print(f"  one combined layer    = {one_step.tolist()}")

    # The collapse: with NO bend, both paths land on the exact same numbers.
    assert np.array_equal(no_bend, one_step), "no-bend layers should collapse into one"

    # LAYOUT CONVENTION (today: COLUMN-on-the-right). Same two grids as Day 2, but the
    # convention flipped, so the combined grid comes out in the other multiply ORDER
    # (matrix multiply is not commutable). Day 2 put the input on the LEFT as a row
    # (x @ W1 @ W2 with W1 = A, W2 = B), so its combined grid was A @ B = [[7, 2], [3, 1]].
    # Here u is a COLUMN on the right (B @ (A @ u), the z = W @ x + b of Part 1, where
    # rows = neurons and columns = inputs), so the stations multiply right-to-left and
    # today's combined grid is B @ A = [[1, 2], [3, 7]].
    # Pin BOTH orders so the two days can never drift into a real disagreement.
    day02_row_convention_grid = A @ B
    print(f"  layout: u is a COLUMN on the right (rows = neurons, columns = inputs), so the"
          f" stations multiply right-to-left -> B @ A = {combined_grid.tolist()};")
    print(f"          Day 2's ROW layout (x on the left) multiplies left-to-right instead"
          f" -> A @ B = {day02_row_convention_grid.tolist()} (same numbers, other order)")
    assert np.array_equal(day02_row_convention_grid, [[7, 2], [3, 1]]), \
        "the other multiply order A @ B must still be Day 2's row-convention grid [[7,2],[3,1]]"
    assert not np.array_equal(combined_grid, day02_row_convention_grid), \
        "B @ A and A @ B must differ — that difference IS the convention flip, not an error"
    # The shape rule of this layout, pinned: a grid maps a length-2 column to a length-2
    # column, so the combined grid must be 2x2 and one_step must stay a length-2 column.
    assert combined_grid.shape == (2, 2) and one_step.shape == (2,), \
        "in this layout the combined grid is (neurons x inputs) and its output is a column"

    # The fix: put a ReLU bend BETWEEN the layers -> the collapse breaks.
    with_bend = B @ relu(A @ u)
    print(f"  with a ReLU bend      = {with_bend.tolist()}  (DIFFERENT — the collapse is broken)")
    assert not np.array_equal(with_bend, one_step), "the bend must change the result"

    # One differing output is NOT the proof, and Day 2 said so: some OTHER grid could still match
    # this one input, so "different from B @ A" only rules out that one grid. The proof is
    # the defining property of a linear map — it ADDS UP for every input pair:
    #     linear:  g(u + v) == g(u) + g(v)   -- always, for every grid
    # so one failure rules out every single grid at once. Same test as Day 2, run in
    # today's column layout.
    #
    # CHOOSING v — the trap that makes this whole test say nothing. If v is the exact
    # negation of u, then u + v is the all-zero column and EVERY term below collapses to
    # zeros: the straight line would print "[0, 0] vs [0, 0] -> adds up: True" for almost
    # any pair of grids, right or wrong, because zero-plus-its-negative is zero whatever
    # the grid does. So v is LOPSIDED here: u + v is a distinct, nonzero column, and
    # A @ v = [3, -2] has a negative entry of its own so the bend bites on v too.
    # (Day 2 runs this same test in its own ROW layout with its own different numbers —
    # same convention about WHY additivity is the proof, deliberately not the same data.)
    v = np.array([7, -2])              # NOT -u; A @ v = [3, -2], so relu has work to do
    u_plus_v = u + v                   # = [4, -1]; A @ [4,-1] = [2,-1], mixed-sign as well
    # Compute each quantity ONCE, print that name, then assert that same name.
    straight_sum = combined_grid @ u + combined_grid @ v      # grid applied twice, then added
    straight_of_sum = combined_grid @ u_plus_v                # grid applied to the sum
    bent_sum = B @ relu(A @ u) + B @ relu(A @ v)
    bent_of_sum = B @ relu(A @ u_plus_v)
    straight_adds_up = np.array_equal(straight_of_sum, straight_sum)
    bent_adds_up = np.array_equal(bent_of_sum, bent_sum)
    print(f"  additivity with v = {v.tolist()} (u + v = {u_plus_v.tolist()}, nonzero):"
          f" straight {straight_sum.tolist()} vs"
          f" {straight_of_sum.tolist()} -> adds up: {straight_adds_up}")
    print(f"    bent {bent_sum.tolist()} vs"
          f" {bent_of_sum.tolist()} -> adds up: {bent_adds_up}"
          f"  <- THAT is what rules out every single grid, not one differing output")
    assert straight_adds_up, "one combined grid is linear, so it must always add up"
    assert not bent_adds_up, "the bent path must fail additivity — no single grid can do it"
    # Pin the four columns those two lines printed. "adds up: True/False" on its own would
    # still read the same with the wrong grid inside, so the literals are what make the
    # lopsided pair worth using: every number here depends on A, B and v being right.
    assert straight_sum.tolist() == [2, 5] and straight_of_sum.tolist() == [2, 5], \
        "the straight path's two additivity columns should both be [2, 5]"
    assert bent_sum.tolist() == [3, 10] and bent_of_sum.tolist() == [2, 6], \
        "the bent path should print [3, 10] vs [2, 6] — a real, nonzero mismatch"

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
