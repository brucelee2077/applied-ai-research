# day-04-matmul-and-shapes — experiment
#
# Today's big idea in two lines of output:
#   (m, k) @ (k, n) = (m, n) — the inner numbers must match, and then they vanish.
#   A dense layer IS that one operation: y = x @ W + b.
#
# This script (1) runs the lesson's (2,3) @ (3,2), (2) catches the mismatch crash,
# (3) builds a real layer, and (4) shows @ and * are NOT the same thing.
# Run it:  python3 sessions/m01-shape-of-data/day-04-matmul-and-shapes/experiment.py

import numpy as np  # numpy gives us arrays, the @ operator, and element-wise *


# ---- Small helpers --------------------------------------------------------

def flop_cost(m, k, n):
    # FLOPs = floating-point operations, a count of the multiply/add work.
    # One (m,k)@(k,n) matmul costs about 2*m*k*n: each of the m*n outputs is a
    # k-term dot product, and every term is one multiply plus one add.
    return 2 * m * k * n


def dot_by_hand(row, col):
    # A dot product: multiply the two lists pairwise, then add the products.
    # This is the single step a matmul repeats for every output entry.
    total = 0
    for a, b in zip(row, col):
        total += a * b
    return total


if __name__ == "__main__":
    # --- Part 1: one dot product, the block everything is made of ---------
    left = [1, 2, 3]
    right = [4, 5, 6]
    print("dot product [1,2,3] · [4,5,6] = 1*4 + 2*5 + 3*6 =", dot_by_hand(left, right))
    print("-> two lists in, ONE number out. A matmul is many of these at once.")

    # --- Part 2: (2,3) @ (3,2) — the shape rule in action -----------------
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])          # shape (2, 3)
    B = np.array([[1, 0],
                  [0, 1],
                  [1, 0]])             # shape (3, 2)

    print("\nA.shape =", A.shape, " B.shape =", B.shape)
    # Say it BEFORE running it — computed from the shapes, so if either matrix
    # changes the prediction moves with it instead of quietly going stale.
    predicted_shape = (A.shape[0], B.shape[1])
    predicted_cost = flop_cost(A.shape[0], A.shape[1], B.shape[1])
    print("predicted output shape =", predicted_shape,
          "  predicted cost = 2*%d*%d*%d =" % (A.shape[0], A.shape[1], B.shape[1]),
          predicted_cost, "FLOPs")

    AB = A @ B                          # @ is matmul (np.matmul), not element-wise
    print("(A @ B).shape =", AB.shape, "  <- the inner 3's matched and vanished")
    print("A @ B =\n", AB)

    # Check one entry by hand, the way a code reviewer does:
    # result[1, 0] = row 1 of A  ·  column 0 of B.
    entry_10 = dot_by_hand(A[1, :], B[:, 0])
    print("check: row 1 of A", A[1, :], "· column 0 of B", B[:, 0], "=", entry_10,
          "== A@B[1,0] =", AB[1, 0])

    # --- Part 3: the mismatch crash, caught on purpose --------------------
    C = np.array([[1, 2, 3],
                  [4, 5, 6]])          # shape (2, 3) — same shape as A
    mismatch_message = ""
    mismatch_raised = False
    try:
        A @ C                           # (2,3) @ (2,3): inner numbers are 3 and 2
    except ValueError as err:           # numpy refuses before computing anything
        mismatch_raised = True          # the FACT is the refusal, not its wording
        mismatch_message = str(err)     # printed for the reader; not asserted on,
                                        # because numpy has reworded it across majors
    print("\ntried (2,3) @ (2,3) -> inner 3 vs 2, not equal")
    print("ValueError:", mismatch_message)

    # --- Part 4: a dense layer is a matmul plus a bias --------------------
    x = np.array([[1.0, 2.0, 3.0]])     # (1, 3) — one example with 3 features
    # The lesson uses np.random.rand for W. We seed the generator so the numbers
    # printed here are the same every run.
    rng = np.random.default_rng(0)
    W = rng.random((3, 4))              # (3, 4) — the weight table: 3 in, 4 out
    b = np.array([0.5, 0.0, -0.5, 1.0])  # (4,) — one bias per output

    print("\nx.shape =", x.shape, " W.shape =", W.shape, " b.shape =", b.shape)
    layer_cost = flop_cost(x.shape[0], W.shape[0], W.shape[1])
    print("predicted layer cost = 2*%d*%d*%d =" % (x.shape[0], W.shape[0], W.shape[1]),
          layer_cost, "FLOPs")
    print("W =\n", np.round(W, 3))

    y = x @ W + b                       # this is a neural-net layer
    print("x @ W  =", np.round(x @ W, 3), " shape", (x @ W).shape)
    print("y = x @ W + b =", np.round(y, 3), " shape", y.shape)
    print("-> (1,3) @ (3,4) = (1,4); the bias (4,) broadcasts across the row (Day 3)")

    # --- Part 5: @ and * are different operations -------------------------
    # @ needs matching INNER dims and gives (2,2). * needs matching FULL shapes
    # and gives (2,3) — same-position numbers multiplied, no summing.
    A_star_A = A * A
    print("\nA @ B (matmul)       =\n", AB, " shape", AB.shape)
    print("A * A (element-wise) =\n", A_star_A, " shape", A_star_A.shape)
    print("-> different shapes, different numbers: never swap @ and *")

    # The dangerous version: * does NOT crash here, it broadcasts.
    # y * b has the right shape (1,4) but every number is wrong.
    y_wrong = (x @ W) * b
    print("right: (x@W) + b =", np.round(y, 3))
    print("wrong: (x@W) * b =", np.round(y_wrong, 3), "  same shape", y_wrong.shape,
          "-> a silent bug: legal shape, wrong quantity")

    # --- Self-check: assert the values the lesson states ------------------
    dot_ok = dot_by_hand(left, right) == 32                      # lesson: [1,2,3]·[4,5,6] = 32
    shape_ok = AB.shape == (2, 2)                                # lesson: (2,3)@(3,2) -> (2,2)
    product_ok = np.array_equal(AB, np.array([[4, 2], [10, 5]]))  # lesson prints [[4,2],[10,5]]
    entry_ok = entry_10 == 10                                    # row·column check of A@B[1,0]
    mismatch_ok = mismatch_raised                                # a ValueError was raised
    layer_ok = y.shape == (1, 4)                                 # lesson: (x@W+b).shape = (1,4)
    # Check the cost of work this script ACTUALLY does — A@B is (2,3)@(3,2), so
    # 2*2*3*2 = 24 — plus the lesson's own worked (2,3)@(3,4) = 48 example, and
    # that the prediction printed above matched what numpy then produced.
    flops_ok = (predicted_cost == 24 and flop_cost(2, 3, 4) == 48
                and layer_cost == 24 and predicted_shape == AB.shape)
    star_ok = (A_star_A.shape == (2, 3)
               and np.array_equal(A_star_A, np.array([[1, 4, 9], [16, 25, 36]])))
    silent_ok = not np.allclose(y, y_wrong)                      # + and * give different numbers

    all_ok = (dot_ok and shape_ok and product_ok and entry_ok and mismatch_ok
              and layer_ok and flops_ok and star_ok and silent_ok)

    if all_ok:
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected [1,2,3]·[4,5,6] == 32, (A@B).shape == (2,2) with "
              "A@B == [[4,2],[10,5]], a ValueError from (2,3)@(2,3), (x@W+b).shape == (1,4), "
              "2*2*3*4 == 48 FLOPs, and A*A == [[1,4,9],[16,25,36]] with shape (2,3)")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert dot_ok, "[1,2,3] · [4,5,6] should be 32"
    assert shape_ok, "(2,3) @ (3,2) should give shape (2,2)"
    assert product_ok, "A @ B should be [[4,2],[10,5]]"
    assert entry_ok, "A@B[1,0] should equal row 1 of A dotted with column 0 of B, which is 10"
    assert mismatch_ok, "(2,3) @ (2,3) should raise a core-dimension ValueError"
    assert layer_ok, "x @ W + b should have shape (1,4)"
    assert flops_ok, "a (2,3)@(3,4) matmul should cost 2*2*3*4 = 48 FLOPs"
    assert star_ok, "A * A should be element-wise: [[1,4,9],[16,25,36]] with shape (2,3)"
    assert silent_ok, "(x@W)+b and (x@W)*b must differ — * is not a safe stand-in"
