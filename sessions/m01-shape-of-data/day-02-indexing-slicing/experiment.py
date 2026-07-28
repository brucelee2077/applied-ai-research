# day-02-indexing-slicing — experiment
#
# Today's big idea in two lines of output:
#   An index picks ONE element; a slice picks a RANGE, and the stop is left out.
#   So you can say the length of x[a:b] out loud before you run it: b − a.
#
# This script (1) indexes a 1-D array, (2) predicts every slice length BEFORE
# numpy runs, (3) grabs one cell, a whole row and a whole column from a 2-D
# array, and (4) shows the view-vs-copy trap the lesson warns about.
# Run it:  python3 sessions/m01-shape-of-data/day-02-indexing-slicing/experiment.py

import numpy as np  # numpy gives us arrays, comma-separated axes, and np.shares_memory

# No random numbers are used anywhere in this file, so every printed number is
# the same on every machine, every run. Nothing is downloaded and nothing is saved.


# ---- Three small helpers --------------------------------------------------

def show(expression, value, kind):
    # Print one grab: the code you wrote, what came back, its shape, and whether
    # the result is a VIEW (a window onto the same memory) or a COPY (its own memory).
    # A shape of () means one single number (a scalar), not an array.
    print("  %-10s -> %-13s shape %-7s %s" % (expression, str(value), str(np.shape(value)), kind))


def half_open_length(start, stop):
    # The half-open rule: x[start:stop] runs from start up to BUT NOT INCLUDING stop.
    # That is why you can state the length before running anything: stop − start.
    return stop - start


def view_or_copy(original, piece):
    # A view shares memory with the original array. A copy has its own memory.
    # np.shares_memory checks that for us instead of us guessing.
    return "VIEW (shares memory)" if np.shares_memory(original, piece) else "copy (own memory)"


if __name__ == "__main__":
    # --- Part 1: index a 1-D array — one position in, one number out ------
    x = np.array([10, 20, 30, 40])     # four numbers in a row, at positions 0, 1, 2, 3
    print("x =", x, " shape", x.shape, " -> positions are 0, 1, 2, 3")
    print("indexing:")
    show("x[0]", x[0], view_or_copy(x, x[0]))      # first element — counting starts at 0
    show("x[2]", x[2], view_or_copy(x, x[2]))      # third element
    show("x[-1]", x[-1], view_or_copy(x, x[-1]))   # a negative index counts from the end
    print("  -> an index returns shape () : one single number, and it is a copy of that value")

    # --- Part 2: say the length FIRST, then let numpy answer --------------
    # This is the acceptance criterion "state the length of x[a:b] as b − a".
    print("\nslicing — I predict the length, THEN numpy runs:")
    predictions_ok = True
    for start, stop in [(1, 3), (0, 2), (2, 4), (0, 4)]:
        said = half_open_length(start, stop)   # the length I can say out loud first
        piece = x[start:stop]                  # what numpy actually hands back
        matched = (said == len(piece))
        predictions_ok = predictions_ok and matched
        print("  x[%d:%d] -> I say %d − %d = %d ; numpy gives %-9s len %d  match: %s"
              % (start, stop, stop, start, said, str(piece), len(piece), matched))

    # The three slices the lesson shows in the playground.
    s_1_3 = x[1:3]      # positions 1 and 2 — position 3 is EXCLUDED, so this is 2 long
    s_head = x[:2]      # no start means "from the beginning", stop 2 is still excluded
    s_step = x[::2]     # no start, no stop, step 2 — every 2nd element
    print("the lesson's three slices:")
    show("x[1:3]", s_1_3, view_or_copy(x, s_1_3))
    show("x[:2]", s_head, view_or_copy(x, s_head))
    show("x[::2]", s_step, view_or_copy(x, s_step))
    print("  -> the stop is excluded, so x[1:3] has %d elements, not 3" % len(s_1_3))

    # --- Part 3: a 2-D array — the comma separates the axes ---------------
    g = np.array([[1, 2, 3],
                  [4, 5, 6]])          # 2 rows, 3 columns
    print("\ng =\n", g, "\n  shape", g.shape, " -> 2 rows, then 3 columns")
    cell = g[1, 2]                     # row 1, column 2 — one number
    row0 = g[0, :]                     # row fixed to 0, colon = ALL columns
    col1 = g[:, 1]                     # colon = ALL rows, column fixed to 1
    print("2-D grabs:")
    show("g[1, 2]", cell, view_or_copy(g, cell))
    show("g[0, :]", row0, view_or_copy(g, row0))
    show("g[:, 1]", col1, view_or_copy(g, col1))
    print("  -> the comma moves to the next axis; the colon means 'everything on this axis'")

    # --- Part 4: the view-vs-copy trap ------------------------------------
    # A slice is a window onto the SAME memory. Writing into the window writes
    # into the original array. This is the silent bug the lesson warns about.
    data = np.arange(20).reshape(5, 4)     # a tiny stand-in for a dataset: 5 rows, 4 columns
    batch_view = data[0:2]                 # "grab a batch" — a view, no numbers were copied
    print("\ndata (5 rows, 4 columns), first row before:", data[0])
    batch_view[0, 0] = 999                 # write into the batch...
    print("after batch_view[0,0] = 999   -> data[0] is now", data[0], "(the source changed)")
    aliasing_bit = data[0, 0]              # remember what the write did

    data[0, 0] = 0                         # put the dataset back the way it was
    batch_copy = data[0:2].copy()          # .copy() asks for separate memory
    batch_copy[0, 0] = 777                 # write into the copy...
    print("after batch_copy[0,0] = 777   -> data[0] is still", data[0], "(the source is safe)")
    print("  batch_view shares memory with data:", np.shares_memory(data, batch_view))
    print("  batch_copy shares memory with data:", np.shares_memory(data, batch_copy))
    print("  -> views are free but aliased; .copy() costs memory and buys safety")

    # --- Self-check: assert the values the lesson states ------------------
    index_ok = (x[0] == 10 and x[2] == 30 and x[-1] == 40)          # playground demo ①
    slice_values_ok = (np.array_equal(s_1_3, [20, 30])              # playground demo ②
                       and np.array_equal(s_head, [10, 20])
                       and np.array_equal(s_step, [10, 30]))
    slice_len_ok = (len(s_1_3) == 2)                                # acceptance: exactly 2
    length_rule_ok = predictions_ok                                 # b − a matched every time
    two_d_ok = (cell == 6                                           # playground demo ③
                and np.array_equal(row0, [1, 2, 3])
                and np.array_equal(col1, [2, 5]))
    view_ok = (np.shares_memory(x, s_1_3)                           # basic slices are views
               and np.shares_memory(g, row0)
               and np.shares_memory(g, col1)
               and not np.shares_memory(x, x[0])                    # an index gives a copy
               and not np.shares_memory(data, batch_copy))          # .copy() is separate memory
    aliasing_ok = (aliasing_bit == 999 and data[0, 0] == 0)         # the write went through, then was undone

    all_ok = (index_ok and slice_values_ok and slice_len_ok and length_rule_ok
              and two_d_ok and view_ok and aliasing_ok)

    if all_ok:
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected x[0] == 10, x[2] == 30, x[-1] == 40, x[1:3] == [20,30] "
              "with exactly 2 elements, x[:2] == [10,20], x[::2] == [10,30], every predicted "
              "length stop − start to match len(), g[1,2] == 6, g[0,:] == [1,2,3], "
              "g[:,1] == [2,5], slices to share memory while an index and .copy() do not, and "
              "a write into a view to change the source array")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert index_ok, "x[0] should be 10, x[2] should be 30, x[-1] should be 40"
    assert slice_values_ok, "x[1:3] == [20,30], x[:2] == [10,20], x[::2] == [10,30]"
    assert slice_len_ok, "x[1:3] must hold exactly 2 elements — the stop is excluded"
    assert length_rule_ok, "the length of x[a:b] must equal b − a for every pair tested"
    assert two_d_ok, "g[1,2] == 6, g[0,:] == [1,2,3], g[:,1] == [2,5]"
    assert view_ok, "a basic slice is a view; an index and .copy() give separate memory"
    assert aliasing_ok, "writing into a view must change the source array"
