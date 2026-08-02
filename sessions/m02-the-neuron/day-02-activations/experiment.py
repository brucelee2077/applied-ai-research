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


# ---- How we MEASURE a slope, instead of trusting a formula ------------------
EPS = 1e-5   # a tiny step to the left and right of the point we care about

def central_slope(bend, z):
    # Rise over run, measured on the real function: (f(z+eps) - f(z-eps)) / (2*eps).
    # We use this instead of the textbook derivative formula so the slope numbers we
    # print come from the code above, not from an identity that would hold anyway.
    return float((bend(z + EPS) - bend(z - EPS)) / (2.0 * EPS))


if __name__ == "__main__":
    # --- Part 1: print all five bends over a small grid -------------------
    # A row of 13 evenly spaced inputs from -6 to +6 — enough to see each shape.
    # ("grid" here means this row of INPUT values, the lesson's own word for it. Day 3
    # uses "grid" for a weight matrix instead — two ideas, one word, so read the type.)
    grid = np.linspace(-6, 6, 13)

    # Round each row ONCE into a named value, print that name, and check that SAME
    # name in the self-check below. One expression per printed row means a corrupted
    # printed row can never pass the check that is supposed to guard it.
    shown_grid = np.round(grid, 2)
    shown_curves = {
        "step":    np.round(step(grid), 3),
        "relu":    np.round(relu(grid), 3),
        "leaky":   np.round(leaky_relu(grid), 3),
        "sigmoid": np.round(sigmoid(grid), 3),
        "tanh":    np.round(tanh(grid), 3),
    }

    print("input z :", shown_grid)
    print("step    :", shown_curves["step"])       # flat 0, then jumps to flat 1
    print("relu    :", shown_curves["relu"])       # negatives -> 0, positives pass
    print("leaky   :", shown_curves["leaky"])      # negatives -> tiny trickle
    print("sigmoid :", shown_curves["sigmoid"])    # always inside (0, 1)
    print("tanh    :", shown_curves["tanh"])       # inside (-1, 1), and 0 at 0

    # --- Part 2: the slope facts that seed today's failures ---------------
    # The textbook says sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)); at z=0 that is
    # 0.5*0.5 = 0.25. The number we headline is MEASURED on the sigmoid we wrote above,
    # not read off that formula. The formula is kept as a second opinion — and a second
    # opinion only counts as evidence if you can SEE it, so we print it too, and we
    # compare it against the measurement at every point on the grid instead of only at
    # z = 0. An unprinted, single-point "agreement" could be a hardcoded 0.25 and nobody
    # reading the output would ever know.
    sigmoid_slope_at_0 = central_slope(sigmoid, 0.0)      # measured on our sigmoid
    printed_sigmoid_slope = round(sigmoid_slope_at_0, 4)  # the number the next line prints
    formula_slopes = sigmoid(grid) * (1 - sigmoid(grid))  # the textbook formula, whole grid
    measured_slopes = np.array([central_slope(sigmoid, z) for z in grid])
    max_slope_gap = float(np.max(np.abs(measured_slopes - formula_slopes)))
    zero_idx = int(np.argmin(np.abs(grid)))               # where z == 0 sits on the grid
    # One expression, one value, two uses: the number printed below IS the number checked.
    formula_slope_at_0 = round(float(formula_slopes[zero_idx]), 4)
    # The gap is SHOWN as a formatted string, so format it once and check that string:
    # a number that only exists inside the print call is a number nothing can guard.
    shown_gap = f"{max_slope_gap:.2e}"
    print("\nsigmoid'(0) =", printed_sigmoid_slope,
          "(peak slope ~0.25 -> shrinks the backward signal in deep sigmoid nets)")
    print("  second opinion: sigmoid(0)*(1-sigmoid(0)) =", formula_slope_at_0,
          f"| worst measured-vs-formula gap across all 13 grid points = {shown_gap}")
    # ReLU's slope on the negative side is exactly 0 -> the seed of a "dead" ReLU.
    # Leaky ReLU keeps a small slope alpha there -> the seed of the cure.
    # Both numbers below are measured at z = -1, not typed in by hand.
    relu_neg_slope = central_slope(relu, -1.0)
    leaky_neg_slope = round(central_slope(leaky_relu, -1.0), 4)
    # Same rule for these two: render each once, print the rendered text, check it.
    shown_relu_slope = f"{relu_neg_slope:g}"
    shown_leaky_slope = f"{leaky_neg_slope:g}"
    print(f"ReLU slope for negatives = {shown_relu_slope}   (can go 'dead')")
    print(f"leaky slope for negatives = {shown_leaky_slope} (small alpha keeps it alive)")

    # --- Part 3: watch two linear layers collapse into one ----------------
    # These fixed weights are the lesson's "anchor pair". Day 3 re-uses this exact
    # numeric pair under the names A = W1 and B = W2, so the two days are running one
    # demo — see the layout note in Part 3b for why their printed grids differ.
    #
    # LAYOUT CONVENTION (today: ROW-on-the-left). The input rides in as a ROW and the
    # weights sit to its RIGHT: x @ W1 @ W2. In this layout a matrix's ROWS are its
    # INPUTS and its COLUMNS are its neurons, and the stations multiply LEFT-to-RIGHT,
    # so the combined grid is W1 @ W2. Day 3 uses the other layout (a COLUMN input on
    # the right: W @ x + b, "rows = neurons, columns = inputs"). Do not carry one day's
    # matrices into the other day's layout untransposed — Part 3b prints what happens.
    W1 = np.array([[1.0, 2.0], [0.0, 1.0]])
    W2 = np.array([[1.0, 0.0], [3.0, 1.0]])

    # Two input rows, chosen so that (a) each one gives a MIXED-SIGN pre-activation
    # x@W1 — so the ReLU in Part 4 clips one component and keeps the other, leaving a
    # non-trivial output instead of all zeros — and (b) they add up to exactly zero,
    # which makes the additivity test in Part 4 easy to read.
    x_a = np.array([[1.0, -3.0]])    # x_a@W1 = [ 1, -1]  -> relu -> [1, 0]
    x_b = np.array([[-1.0, 3.0]])    # x_b@W1 = [-1,  1]  -> relu -> [0, 1]

    # Path A: push x through layer 1, THEN through layer 2 (no bend between).
    two_layers = (x_a @ W1) @ W2       # Day 3 calls this same two-station path no_bend
    # Path B: multiply the two weight matrices FIRST, then push x through once.
    one_layer = x_a @ (W1 @ W2)        # Day 3 calls this same one-station answer one_step
    # ONE name for the object both paths share: the single matrix the two layers fold
    # into. Day 3 calls it combined_grid, so use its name here too.
    combined_grid = W1 @ W2

    # Bind every number these three lines SHOW, then check the same names below.
    shown_combined = combined_grid.astype(int)
    shown_two_layers = np.round(two_layers, 4)
    shown_one_layer = np.round(one_layer, 4)

    print("\nW1 @ W2 =\n", shown_combined)                # the single combined matrix
    print("(x_a@W1)@W2 =", shown_two_layers)              # two layers...
    print("x_a@(W1@W2) =", shown_one_layer)               # ...land on the SAME numbers

    # --- Part 3b: say the layout out loud, and pin BOTH orders --------------
    # Matrix multiply is not commutable, so the layout decides the ORDER the stations
    # combine in. Today's row layout gives W1 @ W2. Day 3's column layout, fed the same
    # NUMBERS, gives the other order W2 @ W1 — a different network, not a typo. And to
    # rewrite TODAY's network as a column layer you TRANSPOSE the combined grid:
    # x @ (W1@W2) == ((W1@W2).T @ x.T).T. Pinning all three keeps the two days from ever
    # drifting into a real disagreement, and shows why an untransposed carry-over breaks.
    day03_column_grid = W2 @ W1                # the order Day 3's layout multiplies in
    today_as_column = combined_grid.T          # today's OWN network, written as columns
    today_column_out = (today_as_column @ x_a.T).T
    # Three shown grids and one shown answer, each bound once before it is printed.
    shown_row_grid_list = shown_combined.tolist()
    shown_col_grid_list = day03_column_grid.astype(int).tolist()
    shown_today_col_list = today_as_column.astype(int).tolist()
    shown_today_col_out = np.round(today_column_out, 4)
    print("layout: x is a ROW on the left today, so the combined grid is W1 @ W2 =",
          shown_row_grid_list)
    print("        Day 3 puts x in a COLUMN on the right, so its order is W2 @ W1 =",
          shown_col_grid_list, "(same numbers, other order)")
    print("        today's network AS a column layer = (W1@W2).T =",
          shown_today_col_list, "-> (W1@W2).T @ x_a.T =",
          shown_today_col_out, "(same answer, transposed layout)")

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
    # Compute each side of the additivity test ONCE, under a name, then print and check
    # that name. Recomputing g(...) / f(...) inside the print calls would give the print
    # its own private copy of the answer, which no check below could ever guard.
    g_sum = g(x_a) + g(x_b)                # g of each, then added
    g_of_sum = g(x_sum)                    # g of the sum
    f_a = f(x_a)
    f_b = f(x_b)
    f_sum = f_a + f_b
    f_of_sum = f(x_sum)
    lin_adds_up = np.allclose(g_of_sum, g_sum)
    bent_adds_up = np.allclose(f_of_sum, f_sum)

    shown_g_sum = np.round(g_sum, 4)
    shown_g_of_sum = np.round(g_of_sum, 4)
    shown_f_sum = np.round(f_sum, 4)
    shown_f_of_sum = np.round(f_of_sum, 4)
    shown_f_a = np.round(f_a, 4)
    shown_f_b = np.round(f_b, 4)

    print("\nstraight path  g(x_a)+g(x_b) =", shown_g_sum,
          " g(x_a+x_b) =", shown_g_of_sum, " -> adds up:", lin_adds_up)
    print("bent path      f(x_a)+f(x_b) =", shown_f_sum,
          " f(x_a+x_b) =", shown_f_of_sum, " -> adds up:", bent_adds_up)
    print("f(x_a) =", shown_f_a, " f(x_b) =", shown_f_b,
          "(each keeps one component and clips the other)")
    print("-> the bent path does NOT add up, and every single matrix does,",
          "so no single matrix can reproduce it: the bend cannot be folded flat")

    # --- Part 4b: run the SAME test on a lopsided pair, so it can actually fail ----
    # Read the pair above again: x_b is exactly -x_a, so x_a + x_b is the all-zero row.
    # That makes the lesson's numbers easy to read, but it makes the STRAIGHT claim say
    # almost nothing: for any linear g, g(x_a) + g(-x_a) is zero and g(0) is zero, so
    # "adds up: True" and the printed [[0,0]] would show up with the WRONG matrices
    # inside g just as happily (swap the order to x@W2@W1, or use x@W1@W1 — the zeros
    # hide it). A test whose answer does not depend on the code is not a test.
    # So run it again on a lopsided pair whose sum is a real, nonzero third point, and
    # pin the numbers it lands on. Now the code has to be right for the output to match.
    # (Day 3 runs this same test in its own column layout with its own different
    # numbers — same convention, different data, deliberately not a copy of these.)
    x_c = np.array([[2.0, 1.0]])           # NOT the negation of x_a
    x_lop = x_a + x_c                      # = [[3., -2.]] — nonzero, and mixed-sign after W1
    # Compute each quantity ONCE, print that name, assert that same name.
    straight_lop_sum = g(x_a) + g(x_c)     # g of each, then added
    straight_lop_of_sum = g(x_lop)         # g of the sum
    bent_lop_sum = f(x_a) + f(x_c)
    bent_lop_of_sum = f(x_lop)
    lin_lop_adds_up = np.allclose(straight_lop_of_sum, straight_lop_sum)
    bent_lop_adds_up = np.allclose(bent_lop_of_sum, bent_lop_sum)
    # Bind the rendered form of every value these three lines show, so the numbers on
    # screen and the numbers in the pins below are literally the same objects.
    shown_x_a = x_a.tolist()
    shown_x_c = x_c.tolist()
    shown_x_lop = x_lop.tolist()
    shown_straight_lop_sum = np.round(straight_lop_sum, 4)
    shown_straight_lop_of_sum = np.round(straight_lop_of_sum, 4)
    shown_bent_lop_sum = np.round(bent_lop_sum, 4)
    shown_bent_lop_of_sum = np.round(bent_lop_of_sum, 4)
    print("\nlopsided pair x_a =", shown_x_a, "x_c =", shown_x_c,
          "-> x_a + x_c =", shown_x_lop, "(a distinct, NONZERO point)")
    print("straight path  g(x_a)+g(x_c) =", shown_straight_lop_sum,
          " g(x_a+x_c) =", shown_straight_lop_of_sum,
          " -> adds up:", lin_lop_adds_up)
    print("bent path      f(x_a)+f(x_c) =", shown_bent_lop_sum,
          " f(x_a+x_c) =", shown_bent_lop_of_sum,
          " -> adds up:", bent_lop_adds_up)

    # --- Self-check: assert the lesson's stated expected values -----------
    # (a) Every curve printed in Part 1 gets pinned to a written-down row. Without
    # these, any bend could be quietly changed (alpha 0.01 -> 0.1, a sign flip inside
    # sigmoid, tanh(z) -> tanh(z/2), step's >= -> >) and the table above would print
    # different numbers with nothing to object. z = 0 IS on the grid, so the row for
    # `step` also decides the boundary rule: >= fires at exactly 0, > does not.
    expected_curves = {
        "step":    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        "relu":    [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
        "leaky":   [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0, 1, 2, 3, 4, 5, 6],
        "sigmoid": [0.002, 0.007, 0.018, 0.047, 0.119, 0.269,
                    0.5, 0.731, 0.881, 0.953, 0.982, 0.993, 0.998],
        "tanh":    [-1, -1, -0.999, -0.995, -0.964, -0.762,
                    0, 0.762, 0.964, 0.995, 0.999, 1, 1],
    }
    printed_curves = shown_curves
    curves_match = all(np.array_equal(printed_curves[name], np.array(row, dtype=float))
                       for name, row in expected_curves.items())
    # The input row is the axis every curve above is read against, so pin it as well.
    grid_ok = shown_grid.tolist() == [float(v) for v in range(-6, 7)]

    # (b) The three slope numbers, each measured on the real bend in Part 2. These pin
    # the exact values Part 2 printed — no tolerance to loosen, so a wrong slope cannot
    # slip through by being "close enough".
    sigmoid_ok = printed_sigmoid_slope == 0.25              # measured sigmoid'(0) == 0.25
    # The second opinion is now a printed number, and it is checked at all 13 grid points
    # instead of only at z = 0 — so a hardcoded formula shows up as a widening gap. The
    # gap claim reads the STRING that was printed, not a second copy of the number.
    formula_agrees = (formula_slope_at_0 == printed_sigmoid_slope
                      and float(shown_gap) < 1e-6)
    slopes_match = (relu_neg_slope == 0.0 and leaky_neg_slope == 0.01
                    and shown_relu_slope == "0" and shown_leaky_slope == "0.01")

    # (c) The collapse. `collapse_holds` alone is matrix associativity — true for ANY
    # W1, W2, x — so it would still hold with the wrong operands. The literal pin is
    # what makes it a claim about THESE numbers.
    expected_W = np.array([[7, 2], [3, 1]])              # the lesson says expect [[7,2],[3,1]]
    expected_collapse = np.array([[-2.0, -1.0]])         # ...and both paths land here
    collapse_holds = np.allclose(two_layers, one_layer)   # two linear layers == one
    collapse_values_match = (np.allclose(shown_two_layers, expected_collapse)
                             and np.allclose(shown_one_layer, expected_collapse))
    W_matches = np.array_equal(shown_combined, expected_W)
    # (c2) The layout, pinned in both orders. Day 3 prints W2 @ W1 = [[1,2],[3,7]] for
    # the same numbers; today's own network in that column layout is the TRANSPOSE of
    # today's grid, [[7,3],[2,1]] — which is NOT W2 @ W1, and that is the whole warning.
    # Each pin reads the same list the line above printed.
    layout_ok = (shown_row_grid_list == [[7, 2], [3, 1]]
                 and shown_col_grid_list == [[1, 2], [3, 7]]
                 and shown_today_col_list == [[7, 3], [2, 1]]
                 and shown_col_grid_list != shown_today_col_list
                 and np.allclose(shown_today_col_out, expected_collapse))
    bend_breaks = lin_adds_up and not bent_adds_up        # only the bent path fails additivity
    values_match = (np.allclose(shown_f_sum, [[4.0, 1.0]])
                    and np.allclose(shown_f_of_sum, [[0.0, 0.0]])  # the lesson's printed numbers
                    and np.allclose(shown_f_a, [[1.0, 0.0]])       # ...and each half of that sum
                    and np.allclose(shown_f_b, [[3.0, 1.0]])
                    and np.allclose(shown_g_sum, [[0.0, 0.0]])
                    and np.allclose(shown_g_of_sum, [[0.0, 0.0]]))
    # (d) The lopsided pair from Part 4b. These four literals are what give the additivity
    # test teeth: with x_b = -x_a every term is zero, so the printed [[0,0]] survives a
    # wrong g; on the lopsided pair a wrong g prints different numbers and these fail.
    lop_bend_breaks = lin_lop_adds_up and not bent_lop_adds_up
    lop_values_match = (np.allclose(shown_straight_lop_sum, [[15.0, 4.0]])
                        and np.allclose(shown_straight_lop_of_sum, [[15.0, 4.0]])
                        and np.allclose(shown_bent_lop_sum, [[18.0, 5.0]])
                        and np.allclose(shown_bent_lop_of_sum, [[15.0, 4.0]])
                        # the shown inputs too: the lopsided point is the whole reason
                        # these four numbers are worth anything.
                        and shown_x_a == [[1.0, -3.0]]
                        and shown_x_c == [[2.0, 1.0]]
                        and shown_x_lop == [[3.0, -2.0]])

    if (curves_match and grid_ok and slopes_match and collapse_holds
            and collapse_values_match
            and W_matches and layout_ok and sigmoid_ok and formula_agrees and bend_breaks
            and values_match and lop_bend_breaks and lop_values_match):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected the input row to read -6..6, "
              "the five printed curves to match their pinned rows, "
              "slopes of 0 (ReLU) and 0.01 (leaky) on the negative side, "
              "W1@W2 == [[7,2],[3,1]], both collapse paths == [[-2,-1]], "
              "the column layout to read W2@W1 == [[1,2],[3,7]] and (W1@W2).T == [[7,3],[2,1]], "
              "measured sigmoid'(0) == 0.25 with the formula agreeing across the grid, "
              "the straight path to add up, the bent path to give [[4,1]] vs [[0,0]], "
              "and the lopsided pair to give straight [[15,4]] both ways with bent "
              "[[18,5]] vs [[15,4]]")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert curves_match, "each printed bend must match its pinned row over the grid"
    assert grid_ok, "the printed input row must be the 13 whole numbers -6 .. +6"
    assert slopes_match, "negative-side slopes should measure 0 (ReLU) and 0.01 (leaky)"
    assert W_matches, "W1@W2 should be [[7,2],[3,1]]"
    assert layout_ok, ("the row layout's grid is W1@W2; the column order W2@W1 is "
                       "[[1,2],[3,7]] (Day 3's number) and today's network as a column "
                       "layer is the transpose (W1@W2).T = [[7,3],[2,1]]")
    assert collapse_holds, "(x_a@W1)@W2 must equal x_a@(W1@W2) — two linear layers are one"
    assert collapse_values_match, "both collapse paths should land on [[-2,-1]]"
    assert sigmoid_ok, "measured sigmoid'(0) should be 0.25"
    assert formula_agrees, ("the printed formula sigmoid(z)*(1-sigmoid(z)) must equal the "
                           "measured slope at z=0 and match it at every grid point")
    assert lin_adds_up, "a straight (linear) path must always add up"
    assert not bent_adds_up, "a ReLU between the layers must break additivity"
    assert values_match, "expected f(x_a)+f(x_b) == [[4,1]] and f(x_a+x_b) == [[0,0]]"
    assert lop_bend_breaks, ("on the lopsided pair too, only the bent path may fail "
                             "additivity")
    assert lop_values_match, ("expected the lopsided pair to give straight [[15,4]] both "
                              "ways, and bent [[18,5]] vs [[15,4]]")

