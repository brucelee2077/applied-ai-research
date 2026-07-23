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


def best_single_line_accuracy():
    # Try LOTS of different straight lines (each is one choice of w1, w2, bias).
    # For each line, a step-neuron fires 1 when z >= 0, else 0. Count how many of
    # the 4 XOR points it gets right, and remember the very best score we ever see.
    best = 0.0
    grid = np.linspace(-4, 4, 17)  # candidate values for each dial
    for w1 in grid:
        for w2 in grid:
            for b in grid:
                w = np.array([w1, w2])
                # How many of the 4 XOR points does THIS one line get right?
                correct = 0
                for x, target in XOR:
                    z = (w * x).sum() + b       # the neuron's raw score
                    guess = 1 if z >= 0 else 0  # the hard step "decide"
                    if guess == target:
                        correct += 1
                accuracy = correct / len(XOR)
                if accuracy > best:
                    best = accuracy
    return best


if __name__ == "__main__":
    # ---- Part 1: run one full forward pass and check the numbers ----
    x = np.array([2, 3])          # two raw inputs
    w = np.array([0.5, -1.0])     # one trust dial per input
    b = 1.0                       # one starting-mood bias for the whole neuron

    z, out = neuron(x, w, b)      # weigh -> add -> decide, all at once

    # Show the two numbers the lesson told us to watch.
    print("inputs x =", x, " weights w =", w, " bias b =", b)
    print("z (weighted sum) =", z)              # should be -1.0
    print("sigmoid(z)       =", round(out, 3))  # should be ~0.269 (a soft "probably no")

    # The lesson's stated expected values: z = -1.0 and sigmoid(z) rounds to 0.269.
    assert z == -1.0, "expected z = -1.0"
    assert round(out, 3) == 0.269, "expected sigmoid(z) = 0.269"

    # A quick sanity check on the bias being the "starting lean":
    # a bigger (more positive) bias should push the answer higher.
    _, out_bigger_bias = neuron(x, w, b + 4.0)
    assert out_bigger_bias > out, "a more positive bias should raise the output"

    # ---- Part 2: watch the single-neuron limit on XOR ----
    best = best_single_line_accuracy()
    print("best XOR accuracy any single straight line can reach =",
          round(best, 2), "->", int(best * 4), "of 4")

    # The whole point: one neuron = one straight line, and XOR is NOT linearly
    # separable, so no single line ever gets all 4 right. The ceiling is 3 of 4.
    assert best == 0.75, "expected best single-line XOR accuracy = 0.75 (3 of 4)"

    # If every check passed, tell the learner they got it.
    print("✅ you got it")
