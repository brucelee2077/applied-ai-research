# day-01-arrays — experiment
#
# Today's big idea in two lines of output:
#   An array is one dtype + one shape over a regular grid of numbers.
#   That rigid grid is why `x + 10` touches every number at once, with no loop.
#
# This script (1) builds a 1-D and a 2-D array and prints shape/ndim/dtype,
# (2) runs a vectorized op, (3) reshapes 6 numbers into (2, 3), (4) reads shapes
# out loud, and (5) shows the silent shape bug the lesson warns about.
# Run it:  python3 sessions/m01-shape-of-data/day-01-arrays/experiment.py

import numpy as np  # numpy gives us arrays, .shape, .ndim, .dtype and reshape


# ---- Two small helpers ----------------------------------------------------

def describe(name, arr):
    # Print the three labels every array carries: shape (size along each dimension),
    # ndim (how many dimensions), dtype (the one type every element shares).
    print("%-8s value:\n%s" % (name, arr))
    print("%-8s shape: %-12s ndim: %-3d dtype: %-8s size: %d numbers"
          % (name, str(arr.shape), arr.ndim, arr.dtype, arr.size))


def read_shape(shape, names):
    # Turn a shape tuple into a sentence you can say out loud. Read left to right:
    # the first number is the biggest group, the last is the smallest slot.
    if len(shape) == 0:
        return "()          -> one single number (a scalar), 0 dimension(s)"
    parts = ["%d %s" % (size, label) for size, label in zip(shape, names)]
    return "%-11s -> %s, %d dimension(s)" % (str(shape), ", then ".join(parts), len(shape))


if __name__ == "__main__":
    # --- Part 1: a 1-D array (a row of numbers) ---------------------------
    one_d = np.array([1, 2, 3])          # a plain Python list becomes an array
    describe("one_d", one_d)
    print("-> shape (3,) means: 3 numbers along one axis. The comma says 'tuple of one'.")

    # --- Part 2: a 2-D array (rows and columns) ---------------------------
    two_d = np.array([[1, 2, 3],
                      [4, 5, 6]])        # 2 rows, each row holding 3 numbers
    print()
    describe("two_d", two_d)
    print("-> shape (2, 3) means: 2 rows, then 3 columns. Outer group first.")

    # One dtype for the whole array — no mixing. One float in a list of ints turns
    # every element into a float, so the grid stays regular.
    mixed = np.array([1, 2.5, 3])
    print("\nnp.array([1, 2.5, 3]).dtype =", mixed.dtype, "->", mixed,
          "(one type wins for the whole grid)")

    # --- Part 3: vectorization — one op, every element, no loop -----------
    # A Python list cannot do this: `[1, 2, 3] + 10` is an error, so you must loop.
    list_error = ""
    try:
        [1, 2, 3] + 10
    except TypeError as err:
        list_error = str(err)
    print("\npython list [1,2,3] + 10 -> TypeError:", list_error)

    plus_ten = one_d + 10                # no loop here: numpy adds 10 to all 3 at once
    times_two = one_d * 2                # same idea, one instruction over the grid
    print("numpy   one_d + 10 =", plus_ten, " shape", plus_ten.shape, "(no loop written)")
    print("numpy   one_d *  2 =", times_two, " shape", times_two.shape, "(no loop written)")

    # Same answer as the slow list loop: the speed costs you nothing in meaning.
    loop_answer = [n + 10 for n in [1, 2, 3]]
    print("slow list loop gives", loop_answer, "-> same numbers, written the long way")

    # --- Part 4: reshape — same numbers, new grid -------------------------
    flat = np.arange(6)                  # np.arange(6) is [0 1 2 3 4 5]
    grid = flat.reshape(2, 3)            # rearrange those 6 numbers into 2 rows x 3 cols
    print("\nflat =", flat, " shape", flat.shape, " ndim", flat.ndim)
    print("grid = flat.reshape(2, 3) ->\n", grid)
    print("grid shape", grid.shape, " ndim", grid.ndim, " size", grid.size)
    print("-> the six numbers did not change:", sorted(grid.ravel().tolist()),
          "==", sorted(flat.tolist()), "; only the grid around them changed")

    # --- Part 5: read any shape out loud, outer to inner ------------------
    scalar = np.array(21)                            # one number, 0-D
    cube = np.arange(12).reshape(2, 2, 3)            # 2 grids of 2 rows x 3 columns
    batch = np.zeros((32, 8, 4))                     # the lesson's batch of sentences
    print("\nreading shapes out loud:")
    print("  scalar", read_shape(scalar.shape, []))
    print("  one_d ", read_shape(one_d.shape, ["numbers in a row"]))
    print("  two_d ", read_shape(two_d.shape, ["rows", "columns"]))
    print("  cube  ", read_shape(cube.shape, ["grids", "rows", "columns"]))
    print("  batch ", read_shape(batch.shape, ["sentences", "words", "numbers per word"]))

    # A real frontier-lab shape. We only multiply the two sizes — the real array
    # would need about 2.4 GB of memory, so we never build it.
    gpt3_shape = (12288, 49152)
    gpt3_count = gpt3_shape[0] * gpt3_shape[1]
    print("  GPT-3 ", read_shape(gpt3_shape, ["rows", "columns"]),
          "=", format(gpt3_count, ","), "numbers in ONE matrix")

    # --- Part 6: the silent shape bug the lesson warns about --------------
    # A wrong shape usually does not crash. numpy broadcasts (32,) against
    # (32, 1) and hands back (32, 32) — 1024 numbers where you wanted 32.
    row = np.zeros(32)                   # shape (32,)
    column = np.zeros((32, 1))           # shape (32, 1) — same count, different grid
    surprise = row + column
    print("\n(32,) + (32, 1) -> shape", surprise.shape, "with", surprise.size,
          "numbers: no crash, wrong result")
    print("-> that is why a one-line `assert arr.shape == (32,)` at each boundary pays off")

    # --- Self-check: assert the values the lesson states ------------------
    one_d_ok = one_d.shape == (3,) and one_d.ndim == 1          # lesson: (3,) and 1
    two_d_ok = two_d.shape == (2, 3) and two_d.ndim == 2        # lesson: (2, 3) and 2
    dtype_ok = (np.issubdtype(one_d.dtype, np.integer)          # both arrays hold ints
                and one_d.dtype == two_d.dtype
                and np.issubdtype(mixed.dtype, np.floating))    # one float makes all floats
    vector_ok = (np.array_equal(plus_ten, [11, 12, 13])         # lesson demo: [11, 12, 13]
                 and np.array_equal(times_two, [2, 4, 6])       # lesson demo: [2, 4, 6]
                 and plus_ten.tolist() == loop_answer)          # loop agrees with vectorized
    list_ok = "can only concatenate list" in list_error         # lesson demo: TypeError
    # The layout has to be pinned, not just the contents: reshape cannot change
    # the multiset, so a `sorted(...) == sorted(...)` check passes even for
    # `flat.reshape(3, 2).T`, which prints [[0,2,4],[1,3,5]] — not the row-major
    # grid the lesson shows. So assert the exact rows.
    reshape_ok = (grid.shape == (2, 3) and grid.ndim == 2 and grid.size == 6
                  and grid.tolist() == [[0, 1, 2], [3, 4, 5]])
    ladder_ok = (scalar.shape == () and scalar.ndim == 0          # build step 1
                 and cube.shape == (2, 2, 3) and cube.ndim == 3   # build step 4
                 and batch.shape == (32, 8, 4) and batch.ndim == 3)  # quiz question 4
    gpt3_ok = gpt3_count > 600_000_000                          # lesson: over 600 million
    silent_ok = surprise.shape == (32, 32)                      # staff lens: the quiet bug

    all_ok = (one_d_ok and two_d_ok and dtype_ok and vector_ok and list_ok
              and reshape_ok and ladder_ok and gpt3_ok and silent_ok)

    if all_ok:
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected one_d.shape == (3,) with ndim 1, two_d.shape == (2,3) "
              "with ndim 2, both holding one integer dtype, one_d + 10 == [11,12,13] and "
              "one_d * 2 == [2,4,6], a TypeError from [1,2,3] + 10, arange(6).reshape(2,3) "
              "holding the same six numbers, shapes (), (2,2,3) and (32,8,4) reading as 0, 3 "
              "and 3 dimensions, over 600 million numbers in (12288, 49152), and "
              "(32,) + (32,1) broadcasting to (32, 32)")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert one_d_ok, "np.array([1,2,3]) should have shape (3,) and ndim 1"
    assert two_d_ok, "np.array([[1,2,3],[4,5,6]]) should have shape (2,3) and ndim 2"
    assert dtype_ok, "both int arrays share one integer dtype, and [1, 2.5, 3] becomes float"
    assert vector_ok, "one_d + 10 should be [11,12,13] and one_d * 2 should be [2,4,6]"
    assert list_ok, "[1,2,3] + 10 should raise a TypeError — a list cannot do math this way"
    assert reshape_ok, "arange(6).reshape(2,3) should be exactly [[0,1,2],[3,4,5]]"
    assert ladder_ok, "shapes should read as (), (3,), (2,3), (2,2,3), (32,8,4)"
    assert gpt3_ok, "a (12288, 49152) matrix should hold over 600 million numbers"
    assert silent_ok, "(32,) + (32,1) should broadcast to (32, 32) — the silent shape bug"
