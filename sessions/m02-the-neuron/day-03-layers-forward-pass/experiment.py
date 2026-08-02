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
    # Bind what we SHOW, print the bound name, then check that same name below. One
    # value read twice means a wrong number on screen can never pass its own check.
    shown_x_len = x.shape[0]
    print(f"  input x has length {shown_x_len}")

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
    shown_z1 = np.round(z1, 2)
    print(f"  z1 (raw, before the bend) = {shown_z1}")

    # Run the hidden layer WITH the ReLU bend.
    a1 = layer(x, W1, b1, relu)
    shown_a1 = np.round(a1, 2)
    print(f"  a1 (after the bend)       = {shown_a1}   <- two entries flattened to 0")

    # Run the output layer with NO bend -> the raw prediction (as in today's lesson).
    yhat = layer(a1, W2, b2, identity)
    shown_yhat_len = yhat.shape[0]
    shown_yhat = np.round(yhat, 2)
    print(f"  final prediction yhat has length {shown_yhat_len}  (2 output neurons)")
    print(f"  yhat = {shown_yhat}")

    # The lesson's promise: the vector length is 3 in, 4 in the middle, 2 out,
    # and the numbers are exactly the ones printed on the page.
    assert x.shape == (3,) and shown_x_len == 3, "input should be length 3"
    assert a1.shape == (4,), "hidden output should be length 4"
    assert yhat.shape == (2,) and shown_yhat_len == 2, "final prediction should be length 2"
    # z1 is printed for you to compare with a1 by eye, so pin it too: ReLU clips both
    # negative entries to 0, which would otherwise hide HOW negative they were.
    assert np.allclose(shown_z1, [1.5, -0.2, 0.6, -0.1]), "pre-activation z1 should match the lesson"
    assert np.allclose(shown_a1, [1.5, 0.0, 0.6, 0.0]), "hidden output should match the lesson"
    assert np.allclose(shown_yhat, [0.75, -0.15]), "prediction should match the lesson"
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
    shown_b2_probe = b2_probe.tolist()      # the bias pair we claim we used
    shown_yhat_probe = np.round(yhat_probe, 2)
    print(f"  control: same layer with b2 = {shown_b2_probe} -> yhat = {shown_yhat_probe}"
          f"  (each bias shifted its OWN neuron)")
    assert shown_b2_probe == [0.1, -0.2], "the control probe's printed bias pair is the one it ran"
    assert np.allclose(shown_yhat_probe, [0.85, -0.35]), \
        "each output bias should shift only its own neuron (0.75+0.1, -0.15-0.2)"

    # --- Part 2: the linear collapse — two straight layers fold into one ---------
    # We reuse the lesson's tiny by-hand pair so you can check it on paper. It is the
    # SAME numeric pair Day 2 used, under Day 2's names: A is Day 2's W1, B is Day 2's W2.
    print("\nPart 2 — the collapse (no bend) vs the fix (add a bend)")
    A = np.array([[1, 2], [0, 1]])   # tiny station 1 (Day 2 called this same grid W1)
    B = np.array([[1, 0], [3, 1]])   # tiny station 2 (Day 2 called this same grid W2)
    u = np.array([-3, 1])            # this input makes A @ u contain a NEGATIVE entry

    # Two-step path, NO bend anywhere: run u through A then B.
    au = A @ u                       # the middle column, with its one negative entry
    no_bend = B @ au                 # Day 2 called this same two-station path two_layers
    # One combined grid: multiply the two weight grids first, then apply once.
    combined_grid = B @ A            # Day 2 called this same object combined_grid too
    one_step = combined_grid @ u     # Day 2 called this same one-station answer one_layer
    # Bind each printed column ONCE. These same names are what the asserts read and what
    # this function returns, so the screen, the checks and the verdict share one value.
    shown_au = au.tolist()
    shown_combined = combined_grid.tolist()
    shown_no_bend = no_bend.tolist()
    shown_one_step = one_step.tolist()
    print(f"  A @ u                 = {shown_au}   <- note the negative entry")
    print(f"  combined grid B @ A   = {shown_combined}")
    print(f"  two-step (no bend)    = {shown_no_bend}")
    print(f"  one combined layer    = {shown_one_step}")

    # The collapse: with NO bend, both paths land on the exact same numbers.
    assert shown_no_bend == shown_one_step, "no-bend layers should collapse into one"
    # The middle column is printed as the reason the bend has work to do, so pin it:
    # without this, "note the negative entry" could point at a number that is not there.
    assert shown_au == [-1, 1], "A @ u should be [-1, 1] — one negative entry for ReLU to bite"

    # LAYOUT CONVENTION (today: COLUMN-on-the-right). Same two grids as Day 2, but the
    # convention flipped, so the combined grid comes out in the other multiply ORDER
    # (matrix multiply is not commutable). Day 2 put the input on the LEFT as a row
    # (x @ W1 @ W2 with W1 = A, W2 = B), so its combined grid was A @ B = [[7, 2], [3, 1]].
    # Here u is a COLUMN on the right (B @ (A @ u), the z = W @ x + b of Part 1, where
    # rows = neurons and columns = inputs), so the stations multiply right-to-left and
    # today's combined grid is B @ A = [[1, 2], [3, 7]].
    # Pin BOTH orders so the two days can never drift into a real disagreement.
    day02_row_convention_grid = A @ B
    shown_day02_grid = day02_row_convention_grid.tolist()
    print(f"  layout: u is a COLUMN on the right (rows = neurons, columns = inputs), so the"
          f" stations multiply right-to-left -> B @ A = {shown_combined};")
    print(f"          Day 2's ROW layout (x on the left) multiplies left-to-right instead"
          f" -> A @ B = {shown_day02_grid} (same numbers, other order)")
    assert shown_day02_grid == [[7, 2], [3, 1]], \
        "the other multiply order A @ B must still be Day 2's row-convention grid [[7,2],[3,1]]"
    assert shown_combined != shown_day02_grid, \
        "B @ A and A @ B must differ — that difference IS the convention flip, not an error"
    # The shape rule of this layout, pinned: a grid maps a length-2 column to a length-2
    # column, so the combined grid must be 2x2 and one_step must stay a length-2 column.
    assert combined_grid.shape == (2, 2) and one_step.shape == (2,), \
        "in this layout the combined grid is (neurons x inputs) and its output is a column"

    # The fix: put a ReLU bend BETWEEN the layers -> the collapse breaks.
    with_bend = B @ relu(au)
    shown_with_bend = with_bend.tolist()
    print(f"  with a ReLU bend      = {shown_with_bend}  (DIFFERENT — the collapse is broken)")
    assert shown_with_bend != shown_one_step, "the bend must change the result"

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
    bent_sum = B @ relu(au) + B @ relu(A @ v)
    bent_of_sum = B @ relu(A @ u_plus_v)
    straight_adds_up = np.array_equal(straight_of_sum, straight_sum)
    bent_adds_up = np.array_equal(bent_of_sum, bent_sum)
    # Render each column once, under a name, and let the asserts read those same names.
    shown_v = v.tolist()
    shown_u_plus_v = u_plus_v.tolist()
    shown_straight_sum = straight_sum.tolist()
    shown_straight_of_sum = straight_of_sum.tolist()
    shown_bent_sum = bent_sum.tolist()
    shown_bent_of_sum = bent_of_sum.tolist()
    print(f"  additivity with v = {shown_v} (u + v = {shown_u_plus_v}, nonzero):"
          f" straight {shown_straight_sum} vs"
          f" {shown_straight_of_sum} -> adds up: {straight_adds_up}")
    print(f"    bent {shown_bent_sum} vs"
          f" {shown_bent_of_sum} -> adds up: {bent_adds_up}"
          f"  <- THAT is what rules out every single grid, not one differing output")
    assert straight_adds_up, "one combined grid is linear, so it must always add up"
    assert not bent_adds_up, "the bent path must fail additivity — no single grid can do it"
    # Pin the four columns those two lines printed. "adds up: True/False" on its own would
    # still read the same with the wrong grid inside, so the literals are what make the
    # lopsided pair worth using: every number here depends on A, B and v being right.
    # The printed pair itself is pinned too — a lopsided v is the whole point, so the v on
    # screen has to be the v that ran, and u + v has to really be their sum.
    assert shown_v == [7, -2] and shown_u_plus_v == [4, -1], \
        "the additivity pair on screen is the lopsided pair that ran: v = [7,-2], u+v = [4,-1]"
    assert shown_straight_sum == [2, 5] and shown_straight_of_sum == [2, 5], \
        "the straight path's two additivity columns should both be [2, 5]"
    assert shown_bent_sum == [3, 10] and shown_bent_of_sum == [2, 6], \
        "the bent path should print [3, 10] vs [2, 6] — a real, nonzero mismatch"

    # The trap worth knowing: if every middle number is already positive, ReLU has
    # nothing to flatten, so the bend changes NOTHING and a careless test would
    # "pass" for the wrong reason. Watch it happen with u = [1, 2].
    u_pos = np.array([1, 2])
    au_pos = A @ u_pos                       # the all-positive middle column
    trap_no_bend = B @ au_pos
    trap_with_bend = B @ relu(au_pos)
    shown_au_pos = au_pos.tolist()
    shown_trap_no_bend = trap_no_bend.tolist()
    shown_trap_with_bend = trap_with_bend.tolist()
    print(f"\n  trap check with u = [1, 2]: A @ u = {shown_au_pos} (both positive)")
    print(f"    no bend   = {shown_trap_no_bend}")
    print(f"    with bend = {shown_trap_with_bend}   <- IDENTICAL: relu had nothing to do")
    assert shown_trap_with_bend == shown_trap_no_bend, \
        "with all-positive middle numbers the bend is a no-op — that is the point"
    # Equality alone would also hold if both lines printed the same WRONG column, so pin
    # what they print: an all-positive middle, and the one answer both paths reach.
    assert shown_au_pos == [5, 2] and min(shown_au_pos) > 0, \
        "the trap needs an all-positive middle column: A @ [1,2] = [5, 2]"
    assert shown_trap_no_bend == [5, 17], "both trap paths should land on [5, 17]"

    # Return the numbers the self-check will verify against the lesson — the very lists
    # the lines above printed, so the verdict judges what the learner read.
    return shown_combined, shown_no_bend, shown_one_step, shown_with_bend


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
