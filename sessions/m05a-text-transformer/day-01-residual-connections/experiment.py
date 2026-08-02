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
# ORIENTATION, stated once because it changes: here a signal is a COLUMN vector and the
# weight sits on its LEFT (W @ x, so W is (out_slots, in_slots)). From Day 3 onward — and
# in the prerequisite module m03 — a word is a ROW and the weight sits on the RIGHT
# (x @ W, so W is (in_slots, out_slots)). Same operation, transposed spelling.
W = np.array([[-0.5, -0.3,  0.9, -0.2],
              [ 0.3, -0.9,  0.2,  0.2],
              [-0.6,  0.4, -0.4, -0.6],
              [ 0.6, -0.2,  0.3,  0.3]])

def F(x):
    # The block's suggested change — the residual. It reshapes x and shrinks it (the 0.8).
    # WORD OF WARNING, because this word moves: TODAY "the residual" means this CHANGE,
    # F(x), the way the 2015 ResNet paper used it. Days 3, 4 and 8 say "residual stream"
    # or "residual lane" for the OTHER thing in `F(x) + x` — the pass-through x that the
    # change is added onto. Same word, two objects; today's whole lesson is the contrast
    # between them. The change itself also picks up two later names: "one block's change"
    # on Day 4 and "a correction" on Day 8. All three mean F(x).
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

def shrinks_every_step(norms):
    # STRICT: each block must leave |x| genuinely SMALLER than before.
    return all(norms[i + 1] < norms[i] for i in range(N_BLOCKS))

def never_grows(norms):
    # The weaker claim: |x| is allowed to stand still, it just may not go up.
    return all(norms[i + 1] <= norms[i] for i in range(N_BLOCKS))

def sci(value):
    # One shared number format, so the printed table and the pinned check below match.
    return "%.6e" % value

def render_norms(norms):
    # Render the checkpoint sizes ONCE. Both the printed table and the pinned check
    # read this one tuple, so a wrong printed number cannot slip past the check.
    return tuple(sci(norms[k]) for k in CHECKPOINTS)

def show_norms(shown):
    # Print the signal size at a few depths, so the trend is readable.
    for k, text in zip(CHECKPOINTS, shown):
        print("   after %2d blocks   |x| = %s" % (k, text))


if __name__ == "__main__":
    # --- Part 1: the block, and the shape rule that lets "+ x" work --------
    x0 = np.ones(4)                  # the starting signal: four 1.0s
    edit = F(x0)                     # the change this block suggests
    # Bind every number BEFORE printing it, then check the same binding below. One
    # value, read twice — never two separate expressions for one quantity.
    shown_w_shape, shown_x0_shape, shown_edit_shape = W.shape, x0.shape, edit.shape
    shown_x0 = x0
    shown_edit = np.round(edit, 6)
    shapes_line_up = shown_edit_shape == shown_x0_shape
    shown_sum = np.round(edit + x0, 6)
    print("W shape:", shown_w_shape, " x0 shape:", shown_x0_shape, " x0 =", shown_x0)
    print("F(x0) shape:", shown_edit_shape, " F(x0) =", shown_edit, "(the change)")
    assert shapes_line_up, "F(x) must come out the same shape as x"
    print("same shape ->", shapes_line_up, " F(x0) + x0 =", shown_sum)
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
    plain_shown = render_norms(plain_norms)
    print("\nPLAIN stack, x = F(x) each time (no shortcut):")
    show_norms(plain_shown)
    fades_every_step = shrinks_every_step(plain_norms)
    shown_x_plain = np.round(x_plain, 8)
    shown_plain_shape = x_plain.shape
    print("   final x =", shown_x_plain, " shape", shown_plain_shape,
          " smaller after every single block?", fades_every_step)

    # --- Part 3: 30 shortcut blocks — watch it survive -------------------
    x_res, res_norms = run_stack(x0, F, residual_block)
    res_shown = render_norms(res_norms)
    print("\nSHORTCUT stack, x = F(x) + x each time:")
    show_norms(res_shown)
    shown_x_res = np.round(x_res, 6)
    shown_res_shape = x_res.shape
    shown_res_floor = sci(min(res_norms[1:]))
    print("   final x =", shown_x_res, " shape", shown_res_shape,
          " smallest |x| ever reached:", shown_res_floor)
    # Honest about what tomorrow does and does not fix: layer norm re-levels each TOKEN's
    # own numbers, which is not the same as holding this lane down. Under the Pre-LN order
    # the module actually adopts, Day 4 measures the lane still growing, block after block.
    print("   it never fell back to 2.0, and it crept UP: tomorrow's layer norm re-levels"
          " each TOKEN going into a block, not this lane — Day 4 measures the lane still"
          " growing under Pre-LN")

    # --- Part 4: the identity trick — when F(x) = 0 ----------------------
    demo_x = np.array([2.0, -1.0, 5.0])      # the lesson's own demo numbers
    demo_out = residual_block(demo_x, zero_block)
    shown_demo_edit = zero_block(demo_x)
    demo_matches = np.allclose(demo_out, demo_x)
    print("\nF(x) = 0, so output = 0 + x:")
    print("   x =", demo_x, " F(x) =", shown_demo_edit, " output =", demo_out,
          " np.allclose(output, x) ->", demo_matches)
    # 30 do-nothing blocks both ways: with the shortcut the input walks out untouched.
    x_deep_res, _ = run_stack(x0, zero_block, residual_block)
    x_deep_plain, dead_norms = run_stack(x0, zero_block, plain_block)
    print("   30 zero-blocks WITH the shortcut ->", x_deep_res, "(exactly x0);",
          "WITHOUT it ->", x_deep_plain, "(signal gone)")
    # Why "smaller after every single block" above is a STRICT claim and not decoration:
    # this dead stack hits 0 at block 1 and stays there, so |x| never grows yet it is NOT
    # strictly shrinking. Printing both makes the difference between < and <= decidable.
    dead_monotone = never_grows(dead_norms)
    dead_strict = shrinks_every_step(dead_norms)
    print("   that dead stack: |x| never grows?", dead_monotone,
          " strictly smaller every step?", dead_strict, "(0 -> 0 is not smaller)")

    # --- Self-check: one boolean per claim -------------------------------
    # Every expected value below was copied out of a real run and written down here.
    shapes_ok = (shown_w_shape == (4, 4) and shown_x0_shape == (4,)
                 and shown_edit_shape == (4,) and shown_plain_shape == (4,)
                 and shown_res_shape == (4,))
    x0_ok = tuple(shown_x0) == (1.0, 1.0, 1.0, 1.0)
    shapes_line_up_ok = shapes_line_up is True
    first_edit_ok = tuple(shown_edit) == (-0.08, -0.16, -0.96, 0.8)
    first_sum_ok = tuple(shown_sum) == (0.92, 0.84, 0.04, 1.8)
    depth_ok = len(plain_norms) == len(res_norms) == 31
    plain_pins_ok = plain_shown == (
        '2.000000e+00', '1.262379e+00', '3.770688e-01',
        '3.376106e-02', '3.260209e-04', '3.365176e-06')
    plain_final_ok = tuple(shown_x_plain) == (2.56e-06, 1.5e-07, 1.59e-06, -1.5e-06)
    res_pins_ok = res_shown == (
        '2.000000e+00', '2.189429e+00', '3.695958e+00',
        '3.066812e+00', '3.391588e+00', '3.612264e+00')
    res_final_ok = tuple(shown_x_res) == (3.421581, 0.776002, 0.773864, -0.374418)
    res_floor_ok = shown_res_floor == '2.189429e+00'
    demo_ok = tuple(demo_out) == (2.0, -1.0, 5.0)
    demo_edit_zero_ok = tuple(shown_demo_edit) == (0.0, 0.0, 0.0)
    demo_matches_ok = demo_matches is True
    deep_identity_ok = tuple(x_deep_res) == (1.0, 1.0, 1.0, 1.0)
    deep_plain_zero_ok = tuple(x_deep_plain) == (0.0, 0.0, 0.0, 0.0)
    dead_strictness_ok = (dead_monotone is True) and (dead_strict is False)
    mismatch_ok = mismatch_raised == "ValueError"

    if (shapes_ok and x0_ok and shapes_line_up_ok and first_edit_ok and first_sum_ok
            and depth_ok and plain_pins_ok and plain_final_ok and res_pins_ok
            and res_final_ok and fades_every_step and res_floor_ok and demo_ok
            and demo_edit_zero_ok and demo_matches_ok and deep_identity_ok
            and deep_plain_zero_ok and dead_strictness_ok and mismatch_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected F(x0) == [-0.08 -0.16 -0.96 0.8]; the plain |x| to fall "
              "2.000000e+00 -> 3.365176e-06, shrinking at every one of the 30 steps; the shortcut "
              "|x| to run 2.000000e+00 -> 3.612264e+00 and never dip below 2.189429e+00; "
              "[2. -1. 5.] handed back when F(x) = 0; 30 zero-blocks to give [1 1 1 1] with the "
              "shortcut and [0 0 0 0] without it, that dead stack never growing but NOT strictly "
              "shrinking; a 3-slot F(x) + a 4-slot x to raise ValueError")

    assert shapes_ok, "W must be 4x4 and x0, F(x0) and both final x must be (4,)"
    assert x0_ok, "the starting signal must be four 1.0s"
    assert shapes_line_up_ok, "F(x0) and x0 must be reported as the same shape"
    assert first_edit_ok, "F(x0) should be [-0.08 -0.16 -0.96 0.8]"
    assert first_sum_ok, "F(x0) + x0 should be [0.92 0.84 0.04 1.8]"
    assert depth_ok, "each stack must record 31 sizes: the start plus 30 blocks"
    assert plain_pins_ok, "the plain stack should fade 2.000000e+00 -> 3.365176e-06"
    assert plain_final_ok, "the plain stack's final x should be [2.56e-06 1.5e-07 1.59e-06 -1.5e-06]"
    assert fades_every_step, "every plain block must leave |x| smaller than before"
    assert res_pins_ok, "the shortcut stack should run 2.000000e+00 -> 3.612264e+00"
    assert res_final_ok, "the shortcut stack's final x should be [3.421581 0.776002 0.773864 -0.374418]"
    assert res_floor_ok, "the shortcut stack should never dip below 2.189429e+00"
    assert demo_ok, "with F(x) = 0 the shortcut must hand back exactly [2. -1. 5.]"
    assert demo_edit_zero_ok, "the do-nothing block must suggest exactly [0. 0. 0.]"
    assert demo_matches_ok, "np.allclose(output, x) must report True for the identity case"
    assert deep_identity_ok, "30 zero-blocks with the shortcut must leave x0 untouched"
    assert deep_plain_zero_ok, "30 zero-blocks without the shortcut must give all zeros"
    assert dead_strictness_ok, ("the dead stack must never grow yet must NOT shrink strictly — "
                                "this is what makes the strict < above a testable claim")
    assert mismatch_ok, "a 3-slot F(x) added onto a 4-slot x must raise ValueError"
