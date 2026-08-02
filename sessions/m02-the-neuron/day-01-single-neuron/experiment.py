# day-01-single-neuron - experiment
#
# Build ONE neuron by hand and watch it do its three beats: weigh -> add -> decide.
# Then watch that same tiny neuron hit its one hard limit: it can't do XOR,
# because a single neuron only ever draws ONE straight line.
#
# Run me:  python3 sessions/m02-the-neuron/day-01-single-neuron/experiment.py

# We use numpy for tidy array math (multiply every input by its weight at once).
import numpy as np


# ---- the "decide" step, version 1: the hard on/off switch (step function) ----
def step(z):
    # If the score z reaches the bar (zero), the neuron "fires" -> 1. Otherwise 0.
    # (z >= 0) is a True/False array; .astype(float) turns it into 1.0 / 0.0.
    return (z >= 0).astype(float)


# ---- the "decide" step, version 2: the smooth dimmer (sigmoid) ----
def sigmoid(z):
    # Squash any score z smoothly into the range 0..1 (think "how confident, 0 to 1").
    # Big negative z -> near 0, big positive z -> near 1, z = 0 sits at exactly 0.5.
    return 1.0 / (1.0 + np.exp(-z))


# ---- the whole neuron: weigh -> add -> decide ----
def neuron(x, w, b):
    # 1. WEIGH: multiply each input by its own weight (how much we trust it).
    weighed = w * x
    # 2. ADD: sum all the weighed inputs, then add the one bias (the starting mood).
    z = weighed.sum() + b
    # 3. DECIDE: pass that single number z through the sigmoid to get the answer.
    out = sigmoid(z)
    # Hand back BOTH numbers so we can look at the raw score and the final answer.
    return z, out


# ---- the XOR pattern: "on when exactly ONE input is on" ----
# Each row is (input1, input2, correct_answer). Notice the answers sit on diagonals.
XOR = [
    (np.array([0, 0]), 0),  # both off  -> off (0)
    (np.array([0, 1]), 1),  # one on    -> on  (1)
    (np.array([1, 0]), 1),  # one on    -> on  (1)
    (np.array([1, 1]), 0),  # both on   -> off (0)
]


def search_single_lines():
    # Try LOTS of different straight lines (each is one choice of w1, w2, bias).
    # For each line, the step-neuron above does the "decide". Count how many of the
    # 4 XOR points it gets right; remember the best score, AND how many settings
    # reach that best score.
    best = 0.0
    ties = 0            # how many (w1, w2, b) settings score exactly 3 of 4
    searched = 0        # how many settings we actually tried (the denominator we print)
    # 17 candidate values per dial. This is a SEARCH SPACE over settings — Day 3 uses
    # the word "grid" for a weight matrix, so keep the two ideas under two names.
    dial_values = np.linspace(-4, 4, 17)
    for w1 in dial_values:
        for w2 in dial_values:
            for b in dial_values:
                searched += 1
                w = np.array([w1, w2])
                # How many of the 4 XOR points does THIS one line get right?
                correct = 0
                for x, target in XOR:
                    z = (w * x).sum() + b   # the neuron's raw score
                    guess = step(z)         # the SAME hard step "decide" as above
                    if guess == target:
                        correct += 1
                accuracy = correct / len(XOR)
                if accuracy > best:
                    best = accuracy
                if accuracy == 0.75:
                    ties += 1
    return best, ties, searched


if __name__ == "__main__":
    # ---- Part 0: pin the "decide" rule right ON the bar ----
    # step() says the neuron fires when z REACHES zero, so z = 0 must give 1, not 0.
    # A score exactly on the bar is the only place ">= 0" and "> 0" differ.
    on_the_bar = step(np.array([-0.1, 0.0, 0.1]))
    print("step(-0.1, 0.0, +0.1) =", on_the_bar, "(a score of exactly 0 fires)")
    boundary_ok = on_the_bar.tolist() == [0.0, 1.0, 1.0]

    # ---- Part 1: run one full forward pass and check the numbers ----
    x = np.array([2, 3])          # two raw inputs
    w = np.array([0.5, -1.0])     # one trust dial per input
    b = 1.0                       # one starting-mood bias for the whole neuron

    z, out = neuron(x, w, b)      # weigh -> add -> decide, all at once

    # Bind every number we are about to SHOW, then print the bound name and check
    # that same bound name. One value, read twice — so a wrong printed number can
    # never sit under a passing check.
    shown_out = round(out, 3)

    # Show the two numbers the lesson told us to watch.
    print("inputs x =", x, " weights w =", w, " bias b =", b)
    print("z (weighted sum) =", z)              # should be -1.0
    print("sigmoid(z)       =", shown_out)      # should be ~0.269 (a soft "probably no")

    # The lesson's stated expected values: z = -1.0 and sigmoid(z) rounds to 0.269.
    z_ok = (z == -1.0)
    out_ok = (shown_out == 0.269)

    # ---- Part 2: the bias is the starting lean — turn only that dial ----
    # Three settings, all measured off the SAME b, so one shared number moves the
    # score: b+4 lands above the bar, b+1 lands exactly ON it, b-4 lands below.
    z_high, out_high = neuron(x, w, b + 4.0)   # z = +3.0
    z_bar, out_bar = neuron(x, w, b + 1.0)     # z =  0.0 exactly — on the bar
    z_low, out_low = neuron(x, w, b - 4.0)     # z = -5.0

    # Bind the three shown answers (and the shown step verdict) before printing them.
    shown_high = round(out_high, 3)
    shown_bar = round(out_bar, 3)
    shown_low = round(out_low, 3)
    shown_bar_step = float(step(z_bar))

    print("bias b+4 -> z =", z_high, " sigmoid =", shown_high, "(above 0.5)")
    print("bias b+1 -> z =", z_bar, " sigmoid =", shown_bar,
          " step =", shown_bar_step, "(exactly on the bar)")
    print("bias b-4 -> z =", z_low, " sigmoid =", shown_low, "(below 0.5)")
    print("-> the answer rises as the bias rises:", shown_low, "<",
          shown_out, "<", shown_bar, "<", shown_high)

    # Pin each of the three to its own literal. That is stronger than only checking
    # the direction: it also fixes HOW FAR the bias moved the score. Every literal
    # here is compared against the SAME name the line above printed, and the three
    # raw scores are pinned too, so no shown number is left unchecked.
    lean_ok = (shown_high == 0.953
               and out_bar == 0.5 and shown_bar == 0.5
               and shown_low == 0.007
               and z_high == 3.0 and z_low == -5.0)
    bar_ok = (z_bar == 0.0 and shown_bar_step == 1.0)

    # ---- Part 3: watch the single-neuron limit on XOR ----
    best, ties, searched = search_single_lines()

    # Bind both shown forms of the ceiling, and the denominator we show as well.
    shown_best = round(best, 2)
    shown_best_of_4 = int(best * 4)

    print("best XOR accuracy any single straight line can reach =",
          shown_best, "->", shown_best_of_4, "of 4")
    print("settings on the 17-point grid that reach 3 of 4 =", ties, "of", searched)

    # The whole point: one neuron = one straight line, and XOR is NOT linearly
    # separable, so no single line ever gets all 4 right. The ceiling is 3 of 4.
    ceiling_ok = (best == 0.75 and shown_best == 0.75 and shown_best_of_4 == 3)
    # A ceiling alone is a weak claim — a search that barely ran also reports 0.75.
    # The tie count is the floor: coarsen the grid, or stop turning w1, and it moves.
    # `searched` is counted inside the loop, so the denominator we print is the
    # number of settings we really tried, not a number typed into the string.
    search_ok = (ties == 732 and searched == 4913)

    # ---- Self-check: one boolean per claim, then a verdict ----
    if boundary_ok and z_ok and out_ok and lean_ok and bar_ok and ceiling_ok and search_ok:
        print("✅ you got it")
    else:
        print("❌ not yet — expected step(0) = 1, z = -1.0, sigmoid(z) = 0.269, "
              "sigmoid at b-4/b+1/b+4 = 0.007/0.5/0.953, and best XOR accuracy "
              "0.75 (3 of 4) reached by 732 of the 4913 grid settings")

    # These asserts make the check hard (they stop the program if a fact is wrong).
    assert boundary_ok, "step() must fire at exactly z = 0: expected [0, 1, 1]"
    assert z_ok, "expected z = -1.0"
    assert out_ok, "expected sigmoid(z) = 0.269"
    assert lean_ok, "expected sigmoid = 0.007 / 0.5 / 0.953 at bias b-4 / b+1 / b+4"
    assert bar_ok, "at bias b+1 the score is exactly 0, so the step-neuron fires"
    assert ceiling_ok, "expected best single-line XOR accuracy = 0.75 (3 of 4)"
    assert search_ok, "expected 732 of the 4913 grid settings to reach 3 of 4"
