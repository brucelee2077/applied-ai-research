# day-03-broadcasting-dtypes — experiment
#
# Today's big idea in two lines of output:
#   Line the two shapes up at their RIGHT edge: each pair fits if the sizes are
#   equal, or if one of them is 1 — and the dtype decides what each number costs.
#
# This script predicts every result shape from the rule BEFORE running the op,
# then weighs the same three numbers as float64 and as float32.
# Run it:  python3 sessions/m01-shape-of-data/day-03-broadcasting-dtypes/experiment.py

import numpy as np  # numpy gives us arrays, .shape, .dtype, .nbytes and broadcasting

# The byte table from the lesson: how many bytes ONE number costs in each dtype.
# numpy has no bfloat16, so it lives here as a plain number we can still add up.
BYTES_PER_NUMBER = {"float64": 8, "float32": 4, "bfloat16": 2}


def predict_shape(shape_a, shape_b):
    """The lesson's rule written out by hand, so we can guess before NumPy answers.

    Returns the result shape, or None when the two shapes cannot be broadcast.
    """
    out = []
    # Walk both shapes from the RIGHT edge: axis -1 first, then -2, and so on.
    for k in range(1, max(len(shape_a), len(shape_b)) + 1):
        # A missing axis (the shorter shape) is read as 1, and a 1 is a stamp.
        a = shape_a[-k] if k <= len(shape_a) else 1
        b = shape_b[-k] if k <= len(shape_b) else 1
        if a != b and a != 1 and b != 1:
            return None          # not equal, and neither side is a stamp -> error
        out.append(max(a, b))    # the stamp stretches to the bigger of the two
    out.reverse()                # we built it right-to-left, so flip it back
    return tuple(out)


def predict(label, shape_a, shape_b):
    # Print our own guess first. Running the op afterwards is the marking scheme.
    guess = predict_shape(shape_a, shape_b)
    print("  %-16s %-8s + %-8s -> predicted %s"
          % (label, shape_a, shape_b, guess if guess else "ERROR (no stamp)"))
    return guess


if __name__ == "__main__":
    # --- Part 1: predict all four shapes from the right-aligned rule ------
    print("Right-align the shapes, then check each pair (equal, or one is 1):")
    p_scalar = predict("scalar stamp", (3,), ())          # () is a plain number
    p_row = predict("row onto grid", (2, 3), (3,))
    p_bad = predict("mismatched row", (2, 3), (2,))
    p_grid = predict("two stamps", (3, 1), (1, 4))

    # --- Part 2: the simplest stamp — one number pressed across an array --
    x = np.array([1, 2, 3])
    x_plus_10 = x + 10           # the single 10 is stretched to every element
    print("\nx        =", x, " shape", x.shape)
    print("x + 10   =", x_plus_10, " shape", x_plus_10.shape,
          "  <- one number pressed across all three slots")

    # --- Part 3: a (3,) row added to a (2,3) grid -------------------------
    g = np.array([[1, 2, 3],
                  [4, 5, 6]])         # shape (2, 3)
    row = np.array([10, 20, 30])      # shape (3,)
    g_plus_row = g + row              # the row is stamped DOWN onto both rows
    print("\ng        =", g.tolist(), " shape", g.shape)
    print("row      =", row.tolist(), " shape", row.shape)
    print("g + row  =", g_plus_row.tolist(), " shape", g_plus_row.shape,
          "  <- the (3,) row reached BOTH rows, no loop")

    # A second stamp case: both sides stretch, and 12 numbers grow from 3 + 4.
    col = np.arange(1, 4).reshape(3, 1)          # shape (3, 1)
    wide = np.arange(10, 50, 10).reshape(1, 4)   # shape (1, 4)
    grid_out = col + wide
    print("(3,1) + (1,4) ->", grid_out.shape, "=", grid_out.tolist())

    # --- Part 4: when the rule says no, NumPy raises ----------------------
    # try/except means: attempt this, and if it fails, catch the error instead
    # of stopping the program.
    error_message = ""
    try:
        g + np.array([1, 2])          # (2,3) + (2,): 3 vs 2, and neither is 1
    except ValueError as err:
        error_message = str(err)
    print("\n(2,3) + (2,) raised ->", error_message)
    print("  right edge is 3 vs 2 — not equal, neither is 1, so no stamp exists")

    # --- Part 5: the silent one — a row against a column ------------------
    # This does NOT error. That is what makes it dangerous.
    n = 100
    as_row = np.arange(n)                   # shape (100,)
    as_column = np.arange(n).reshape(n, 1)  # shape (100, 1)
    accidental = as_row + as_column
    print("\n(100,) + (100,1) ->", accidental.shape,
          "with", accidental.size, "numbers instead of", n)
    print("  no error, no warning — print .shape right after the op to catch it")

    # --- Part 6: dtype — the size of the box each number sits in ----------
    a = np.array([1, 2, 3])            # whole numbers default to int64
    b = np.array([1.0, 2.0, 3.0])      # decimals default to float64
    f = b.astype(np.float32)           # the same three values, half-size box
    print("\na.dtype =", a.dtype, " b.dtype =", b.dtype, " f.dtype =", f.dtype)
    print("b (float64): nbytes =", b.nbytes, "= 3 numbers x 8 bytes")
    print("f (float32): nbytes =", f.nbytes, "= 3 numbers x 4 bytes")
    print("values still match after the cast:", np.array_equal(b, f.astype(np.float64)))

    # The same dial, at frontier scale: 7 billion weights, one dtype change.
    params = 7_000_000_000
    gb_fp32 = params * BYTES_PER_NUMBER["float32"] / 1e9
    gb_bf16 = params * BYTES_PER_NUMBER["bfloat16"] / 1e9
    print("7B weights: float32 ~%.0f GB, bfloat16 ~%.0f GB" % (gb_fp32, gb_bf16))

    # --- Self-check: the values the lesson states -------------------------
    scalar_ok = np.array_equal(x_plus_10, [11, 12, 13]) and p_scalar == (3,)
    row_ok = (np.array_equal(g_plus_row, [[11, 22, 33], [14, 25, 36]])
              and g_plus_row.shape == (2, 3) and p_row == (2, 3))
    error_ok = ("broadcast" in error_message) and p_bad is None
    predict_ok = p_grid == (3, 4) == grid_out.shape
    silent_ok = accidental.shape == (100, 100)     # the bug that never shouts
    dtype_ok = (str(b.dtype) == "float64" and str(f.dtype) == "float32"
                and b.nbytes == 24 and f.nbytes == 12 and f.nbytes * 2 == b.nbytes)
    memory_ok = round(gb_fp32) == 28 and round(gb_bf16) == 14

    if (scalar_ok and row_ok and error_ok and predict_ok
            and silent_ok and dtype_ok and memory_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected x+10 == [11,12,13], g+row == [[11,22,33],[14,25,36]], "
              "(2,3)+(2,) to raise a broadcast error, (3,1)+(1,4) -> (3,4), "
              "(100,)+(100,1) -> (100,100), b.nbytes == 24 with f.nbytes == 12, "
              "and 7B weights == 28 GB in float32 vs 14 GB in bfloat16")

    # These asserts make the check hard: a wrong fact stops the program.
    assert scalar_ok, "x + 10 should be [11,12,13] with shape (3,)"
    assert row_ok, "g + row should be [[11,22,33],[14,25,36]] with shape (2,3)"
    assert error_ok, "(2,3) + (2,) must raise a broadcasting error"
    assert predict_ok, "(3,1) + (1,4) should give (3,4) — both sides stretch"
    assert silent_ok, "(100,) + (100,1) must silently give (100,100)"
    assert dtype_ok, "float64 costs 24 bytes here and float32 exactly half, 12"
    assert memory_ok, "7B weights should be ~28 GB in float32 and ~14 GB in bfloat16"
