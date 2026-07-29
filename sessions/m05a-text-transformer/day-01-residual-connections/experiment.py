# day-01-residual-connections — experiment
#
# Today's big idea in two lines of output:
#   30 PLAIN blocks shrink the signal from |x| = 2.000000e+00 to 3.365176e-06 — the fade.
#   30 SHORTCUT blocks (output = F(x) + x) leave it at 3.612264e+00 — the signal survives.
#
# Same block F, same 30 layers, same starting x: the only difference is the "+ x". Part 4
# shows why: when F(x) = 0 the shortcut hands x back, a plain stack returns all zeros.
# Run it:  python3 sessions/m05a-text-transformer/day-01-residual-connections/experiment.py

import numpy as np  # numpy gives us arrays, matrix multiply (@) and np.linalg.norm

N_BLOCKS = 30                        # how deep the stack is
CHECKPOINTS = (0, 1, 5, 10, 20, 30)  # the depths we print a signal size for

# The block's fixed recipe, written out by hand instead of drawn at random, so every
# number below is the same on every run and every machine. Its rows and its columns do NOT
# add up to the same totals, on purpose: using W the wrong way round (W.T) then changes
# the very first number this script prints.
W = np.array([[-0.5, -0.3,  0.9, -0.2],
              [ 0.3, -0.9,  0.2,  0.2],
              [-0.6,  0.4, -0.4, -0.6],
              [ 0.6, -0.2,  0.3,  0.3]])

def F(x):
    # The block's suggested change — the residual. It reshapes x and shrinks it (the 0.8).
    return 0.8 * (W @ x)             # same shape out as in, so the "+ x" can line up

def zero_block(x):
    return np.zeros_like(x)          # a block that suggests no change at all: F(x) = 0

def plain_block(x, block):
    return block(x)                  # a plain layer: keep only F(x) and drop the input x

def residual_block(x, block):
    return block(x) + x              # the shortcut: keep x and ADD the change on top

def run_stack(x_start, block, wiring):
    # Push x_start through N_BLOCKS copies of one block, writing down |x| after each.
    x = x_start.copy()               # both stacks start from the same fresh input
    norms = [np.linalg.norm(x)]      # the norm says how big the whole vector is
    for _ in range(N_BLOCKS):
        x = wiring(x, block)         # wiring is plain_block or residual_block
        norms.append(np.linalg.norm(x))
    return x, norms

def sci(value):
    # One shared number format, so the printed table and the pinned check below match.
    return "%.6e" % value

def show_norms(norms):
    # Print the signal size at a few depths, so the trend is readable.
    for k in CHECKPOINTS:
        print("   after %2d blocks   |x| = %s" % (k, sci(norms[k])))


if __name__ == "__main__":
    # --- Part 1: the block, and the shape rule that lets "+ x" work --------
    x0 = np.ones(4)                  # the starting signal: four 1.0s
    edit = F(x0)                     # the change this block suggests
    print("W shape:", W.shape, " x0 shape:", x0.shape, " x0 =", x0)
    print("F(x0) shape:", edit.shape, " F(x0) =", np.round(edit, 6), "(the change)")
    assert edit.shape == x0.shape, "F(x) must come out the same shape as x"
    print("same shape ->", edit.shape == x0.shape, " F(x0) + x0 =", np.round(edit + x0, 6))
    def narrow_block(v):             # a deliberately wrong block: 3 slots out, always
        return np.zeros(3)
    try:                             # 3 slots cannot be added onto 4 — the lesson's ✗ row
        residual_block(x0, narrow_block)
        mismatch_raised = "nothing"
    except ValueError as exc:        # catch the exception TYPE, never its message wording
        mismatch_raised = type(exc).__name__
    print("a 3-slot F(x) added onto a 4-slot x raises:", mismatch_raised)

    # --- Part 2: 30 plain blocks — watch the signal fade ------------------
    x_plain, plain_norms = run_stack(x0, F, plain_block)
    print("\nPLAIN stack, x = F(x) each time (no shortcut):")
    show_norms(plain_norms)
    fades_every_step = all(plain_norms[i + 1] < plain_norms[i] for i in range(N_BLOCKS))
    print("   final x =", np.round(x_plain, 8), " shape", x_plain.shape,
          " smaller after every single block?", fades_every_step)

    # --- Part 3: 30 shortcut blocks — watch it survive -------------------
    x_res, res_norms = run_stack(x0, F, residual_block)
    print("\nSHORTCUT stack, x = F(x) + x each time:")
    show_norms(res_norms)
    print("   final x =", np.round(x_res, 6), " shape", x_res.shape,
          " smallest |x| ever reached:", sci(min(res_norms[1:])))
    print("   it never fell back to 2.0, and it crept UP: layer norm tames that tomorrow")

    # --- Part 4: the identity trick — when F(x) = 0 ----------------------
    demo_x = np.array([2.0, -1.0, 5.0])      # the lesson's own demo numbers
    demo_out = residual_block(demo_x, zero_block)
    print("\nF(x) = 0, so output = 0 + x:")
    print("   x =", demo_x, " F(x) =", zero_block(demo_x), " output =", demo_out,
          " np.allclose(output, x) ->", np.allclose(demo_out, demo_x))
    # 30 do-nothing blocks both ways: with the shortcut the input walks out untouched.
    x_deep_res, _ = run_stack(x0, zero_block, residual_block)
    x_deep_plain, _ = run_stack(x0, zero_block, plain_block)
    print("   30 zero-blocks WITH the shortcut ->", x_deep_res, "(exactly x0);",
          "WITHOUT it ->", x_deep_plain, "(signal gone)")

    # --- Self-check: one boolean per claim -------------------------------
    # Every expected value below was copied out of a real run and written down here.
    shapes_ok = (W.shape == (4, 4) and x0.shape == (4,) and edit.shape == (4,))
    first_edit_ok = tuple(np.round(edit, 6)) == (-0.08, -0.16, -0.96, 0.8)
    depth_ok = len(plain_norms) == len(res_norms) == 31
    plain_pins_ok = tuple(sci(plain_norms[k]) for k in CHECKPOINTS) == (
        '2.000000e+00', '1.262379e+00', '3.770688e-01',
        '3.376106e-02', '3.260209e-04', '3.365176e-06')
    res_pins_ok = tuple(sci(res_norms[k]) for k in CHECKPOINTS) == (
        '2.000000e+00', '2.189429e+00', '3.695958e+00',
        '3.066812e+00', '3.391588e+00', '3.612264e+00')
    res_floor_ok = sci(min(res_norms[1:])) == '2.189429e+00'
    demo_ok = tuple(demo_out) == (2.0, -1.0, 5.0)
    deep_identity_ok = tuple(x_deep_res) == (1.0, 1.0, 1.0, 1.0)
    deep_plain_zero_ok = tuple(x_deep_plain) == (0.0, 0.0, 0.0, 0.0)
    mismatch_ok = mismatch_raised == "ValueError"

    if (shapes_ok and first_edit_ok and depth_ok and plain_pins_ok and res_pins_ok
            and fades_every_step and res_floor_ok and demo_ok and deep_identity_ok
            and deep_plain_zero_ok and mismatch_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected F(x0) == [-0.08 -0.16 -0.96 0.8]; the plain |x| to fall "
              "2.000000e+00 -> 3.365176e-06, shrinking at every one of the 30 steps; the shortcut "
              "|x| to run 2.000000e+00 -> 3.612264e+00 and never dip below 2.189429e+00; "
              "[2. -1. 5.] handed back when F(x) = 0; 30 zero-blocks to give [1 1 1 1] with the "
              "shortcut and [0 0 0 0] without it; a 3-slot F(x) + a 4-slot x to raise ValueError")

    assert shapes_ok, "W must be 4x4 and x0 and F(x0) must both be (4,)"
    assert first_edit_ok, "F(x0) should be [-0.08 -0.16 -0.96 0.8]"
    assert depth_ok, "each stack must record 31 sizes: the start plus 30 blocks"
    assert plain_pins_ok, "the plain stack should fade 2.000000e+00 -> 3.365176e-06"
    assert fades_every_step, "every plain block must leave |x| smaller than before"
    assert res_pins_ok, "the shortcut stack should run 2.000000e+00 -> 3.612264e+00"
    assert res_floor_ok, "the shortcut stack should never dip below 2.189429e+00"
    assert demo_ok, "with F(x) = 0 the shortcut must hand back exactly [2. -1. 5.]"
    assert deep_identity_ok, "30 zero-blocks with the shortcut must leave x0 untouched"
    assert deep_plain_zero_ok, "30 zero-blocks without the shortcut must give all zeros"
    assert mismatch_ok, "a 3-slot F(x) added onto a 4-slot x must raise ValueError"
