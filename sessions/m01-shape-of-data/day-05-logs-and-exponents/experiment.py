# day-05-logs-and-exponents — experiment
#
# Today's big idea in two lines of output:
#   A log asks "10 to what power gives this number?", so it turns ×10 into +1.
#   That is why a power law y = a·x^b is a straight LINE on log-log axes, and
#   the slope of that line IS the exponent b.
#
# This script (1) shows log and exp undoing each other, (2) counts orders of
# magnitude, (3) straightens y = x^2 and measures slope 2, (4) reads a real
# scaling-law exponent off the line, and (5) shows the log(0) underflow trap.
# Run it:  python3 sessions/m01-shape-of-data/day-05-logs-and-exponents/experiment.py

import numpy as np  # numpy gives us arrays, np.log / np.log10, and np.polyfit

# ---- Small helpers --------------------------------------------------------

def power_law(x, a, b):
    # A power law: one quantity is a fixed power of another, y = a * x^b.
    # `a` sets the height of the curve; `b` is the exponent we want to recover.
    return a * x ** b

def slope_of_line(x_values, y_values):
    # Slope = rise / run, measured from the first point to the last one.
    return (y_values[-1] - y_values[0]) / (x_values[-1] - x_values[0])

def step_slopes(x_values, y_values):
    # The slope of every single segment. All the same number means one straight
    # line. Different numbers mean the shape still bends.
    return np.diff(y_values) / np.diff(x_values)

if __name__ == "__main__":
    # --- Part 1: an exponent multiplies, a log undoes it ------------------
    # An exponent is repeated multiplication: 2**3 means 2 * 2 * 2.
    print("2 ** 3 =", 2 ** 3, "  (multiply 2 by itself 3 times)")
    print("np.exp(1) =", np.exp(1), "  (the number e, to the power 1)")
    # A log asks the reverse question: "to what power?"
    log_of_exp_3 = np.log(np.exp(3))       # np.log is the natural log (base e)
    print("np.log(np.exp(3)) =", log_of_exp_3, " <- exp went up, log came back down")
    print("np.log2(8) =", np.log2(8), "(because 2**3 = 8)   np.log10(1000) =",
          np.log10(1000), "(because 10**3 = 1000)")
    print("-> log and exponent undo each other, the way + and - do")

    # --- Part 2: log10 counts orders of magnitude -------------------------
    # One order of magnitude = one factor of 10.
    decades = np.array([1, 10, 100, 1000])
    log_decades = np.log10(decades)
    print("\nvalues        :", decades, " shape", decades.shape)
    print("np.log10(...) :", log_decades, " shape", log_decades.shape)
    print("gap between neighbours:", np.diff(log_decades), " <- every ×10 adds 1")
    print("-> the range 1..1000 becomes the small, evenly spaced numbers 0..3")

    # --- Part 3: a power law straightens on log-log axes ------------------
    x = np.array([1, 10, 100])
    y = x ** 2                             # the power law y = x^2, so b = 2
    print("\nx =", x, " shape", x.shape)
    print("y = x ** 2 =", y, " shape", y.shape)
    log_x = np.log10(x)
    log_y = np.log10(y)
    print("np.log10(x) =", log_x, "  np.log10(y) =", log_y)

    # Every step right lifts log_y by the same 2, so the points sit on a line.
    print("slope of each segment:", step_slopes(log_x, log_y))
    measured_slope = slope_of_line(log_x, log_y)
    print("slope = rise / run = (%.1f - %.1f) / (%.1f - %.1f) = %.1f"
          % (log_y[-1], log_y[0], log_x[-1], log_x[0], measured_slope))
    print("-> the exponent 2 came back as the slope 2")

    # The common mistake: logging only ONE axis. Here x stays raw, only y is
    # logged, and the segment slopes disagree — so it is not a line.
    half_logged = step_slopes(x, log_y)
    print("if only y is logged, segment slopes =", np.round(half_logged, 4),
          "-> still bending")

    # --- Part 4: read a real scaling-law exponent -------------------------
    # Kaplan et al. (2020): loss falls with compute as a power law with a small
    # NEGATIVE exponent, about -0.05. Nothing is downloaded — we build that
    # curve ourselves, then read the exponent back off the log-log line.
    kaplan_b = -0.05
    compute = np.array([1e18, 1e19, 1e20, 1e21, 1e22, 1e23])   # FLOPs spent
    loss = power_law(compute, 2.0, kaplan_b)
    print("\ncompute :", compute, " shape", compute.shape)
    print("loss    :", np.round(loss, 4))
    recovered_b = slope_of_line(np.log10(compute), np.log10(loss))
    print("slope of the log-log line = %.4f (we put in b = %.2f) -> 10x compute "
          "multiplies loss by 10**%.2f = %.4f"
          % (recovered_b, kaplan_b, kaplan_b, 10 ** kaplan_b))

    # Real measurements scatter around the line, so we fit instead of eyeballing.
    # The seed fixes the noise, so every number printed here repeats every run.
    rng = np.random.default_rng(0)
    noisy_loss = loss * (1 + 0.02 * rng.standard_normal(loss.shape))
    fitted_b = np.polyfit(np.log10(compute), np.log10(noisy_loss), 1)[0]
    print("noisy loss :", np.round(noisy_loss, 4),
          "-> fitted slope %.4f, still close to %.2f" % (fitted_b, kaplan_b))

    # --- Part 5: the silent trap — log of an underflowed product ----------
    # Multiplying hundreds of small probabilities gives a number too small to
    # store, so it becomes exactly 0.0 with no error. Then log(0.0) is -inf.
    probs = np.full(300, 0.001)
    product = np.prod(probs)
    with np.errstate(divide="ignore"):     # we EXPECT the log(0) warning here
        log_of_product = np.log(product)
    sum_of_logs = np.sum(np.log(probs))    # the safe way: add the logs
    print("\nnp.prod(300 copies of 0.001) =", product, " <- underflowed to zero")
    print("np.log(product) =", log_of_product, " <- one -inf poisons any average")
    print("np.sum(np.log(probs)) = %.4f <- finite, and the right answer" % sum_of_logs)
    print("-> add the logs, never log a long product")

    # --- Self-check: assert the values the lesson states ------------------
    exp_log_ok = log_of_exp_3 == 3.0                             # lesson: 3.0
    decades_ok = np.array_equal(log_decades, np.array([0.0, 1.0, 2.0, 3.0]))
    step_ok = np.array_equal(np.diff(log_decades), np.ones(3))   # each ×10 adds 1
    y_ok = np.array_equal(y, np.array([1, 100, 10000]))          # lesson prints these
    log_pair_ok = (np.array_equal(log_x, np.array([0.0, 1.0, 2.0]))
                   and np.array_equal(log_y, np.array([0.0, 2.0, 4.0])))
    rise_ok = np.array_equal(step_slopes(log_x, log_y), np.array([2.0, 2.0]))
    slope_ok = measured_slope == 2.0                             # lesson: (4-0)/(2-0)
    one_axis_ok = not np.allclose(half_logged[0], half_logged[1])  # log-y-only bends
    kaplan_ok = abs(recovered_b - kaplan_b) < 1e-12              # slope == exponent
    # The line above only re-states an identity: recovered_b is the log-log slope
    # of a curve BUILT from kaplan_b, so log(a*x^b) = log a + b*log x forces them
    # to agree for ANY exponent. It would print ✅ with b = -0.7. So the real
    # claim is pinned against numbers written down here, independent of the code
    # that produced them: the six losses, and the fixed 10x step ratio.
    expected_loss = np.array([0.251785, 0.224404, 0.200000,
                              0.178250, 0.158866, 0.141589])
    loss_values_ok = np.allclose(loss, expected_loss, atol=1e-6)
    ratio_ok = np.allclose(loss[1:] / loss[:-1], 0.891251, atol=1e-6)   # 10**-0.05
    fit_ok = abs(fitted_b - (-0.050143)) < 1e-5                  # the seeded fit
    underflow_ok = (product == 0.0 and np.isneginf(log_of_product)
                    and np.isfinite(sum_of_logs))

    if (exp_log_ok and decades_ok and step_ok and y_ok and log_pair_ok and rise_ok
            and slope_ok and one_axis_ok and kaplan_ok and loss_values_ok
            and ratio_ok and fit_ok and underflow_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected np.log(np.exp(3)) == 3.0, "
              "np.log10([1,10,100,1000]) == [0,1,2,3] with gaps of exactly 1, "
              "y == [1,100,10000], np.log10(x) == [0,1,2] and np.log10(y) == [0,2,4] "
              "rising 2 per 1 for a slope of exactly 2, a b = -0.05 power law to give back "
              "slope -0.05, and 0.001**300 to underflow to 0.0 so np.log of it is -inf "
              "while the sum of the logs stays finite")
    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert exp_log_ok, "np.log(np.exp(3)) should be 3.0 — log and exp cancel"
    assert decades_ok, "np.log10([1,10,100,1000]) should be [0,1,2,3]"
    assert step_ok, "each ×10 should add exactly 1 to the log"
    assert y_ok, "y = x**2 for x = [1,10,100] should be [1,100,10000]"
    assert log_pair_ok, "np.log10(x) should be [0,1,2] and np.log10(y) should be [0,2,4]"
    assert rise_ok, "log10(y) must rise by 2 for every 1 that log10(x) rises"
    assert slope_ok, "the log-log slope should equal the exponent, 2"
    assert one_axis_ok, "logging only the y axis must NOT give a straight line"
    assert kaplan_ok, "the log-log slope of loss = 2.0 * compute**-0.05 should be -0.05"
    assert loss_values_ok, "loss = 2.0 * compute**-0.05 should start 0.251785 and end 0.141589"
    assert ratio_ok, "every 10x of compute must multiply the loss by 10**-0.05 = 0.891251"
    assert fit_ok, "the seeded noisy fit should recover -0.050143"
    assert underflow_ok, "0.001**300 underflows to 0.0, log(0.0) = -inf, sum of logs finite"
