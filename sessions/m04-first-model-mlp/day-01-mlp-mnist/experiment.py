# day-01-mlp-mnist — experiment
#
# Today's big idea in two lines of output:
#   A forward pass is shapes changing hands: (batch,784) -> (batch,128) -> (batch,10).
#   The wiring is already right, but every knob is random — so it guesses ~10%.
#
# This script (1) makes a stand-in digit set, (2) builds an UNTRAINED 784->128->10 MLP
# with He init, (3) runs one forward pass and reads its guess, (4) shows why layers bend,
# (5) hands the same forward pass two NON-ZERO bias vectors so you see what a bias adds.
# Run it:  python3 sessions/m04-first-model-mlp/day-01-mlp-mnist/experiment.py

import numpy as np  # numpy gives us arrays, matrix multiply (@) and argmax


def make_stand_in_digits(n_per_class, rng):
    """DEVIATION: real MNIST is not on this machine and downloading is blocked, so we draw
    our own 28x28 'handwriting'. Each class gets one ink pattern (two strokes), then every
    sample adds pen wobble. Same shapes as MNIST: 28x28 bytes 0-255, 10 classes.
    Returns the bytes, the labels, and the pre-cast floats (so the caller can check the
    clip-then-cast order, which the byte array alone can no longer show)."""
    templates = np.zeros((10, 28, 28))
    for digit in range(10):
        row, col = rng.integers(4, 22, size=2)      # where this digit's strokes sit
        templates[digit, row:row + 3, 3:25] = 1.0   # one horizontal stroke
        templates[digit, 3:25, col:col + 3] = 1.0   # one vertical stroke
    images = np.repeat(templates, n_per_class, axis=0)          # (10*n, 28, 28)
    labels = np.repeat(np.arange(10), n_per_class)              # (10*n,)
    wobble = rng.normal(0.0, 0.18, size=images.shape)           # a little noise per pixel
    pixels = np.clip(images + wobble, 0.0, 1.0) * 255.0         # clip first, then to bytes
    return pixels.round().astype(np.uint8), labels, pixels


def he_init(n_in, n_out, rng):
    """He initialization: normal random numbers scaled by sqrt(2 / n_in). Built for
    ReLU, so fewer hidden units are born stuck at zero (the 'dead ReLU' failure mode)."""
    return rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)


def relu(z):
    # The bend: keep positives, turn negatives into 0.
    return np.maximum(0, z)


def forward(x, W1, b1, W2, b2):
    """One forward pass, returning the module's shared 3-tuple (z1, hidden, logits): the hidden
    layer's raw score BEFORE the bend, the same values after it, and the 10 output scores.
    Days 2 and 3 return that same triple in that same order, so the unpacking line carries
    over; day 3 only differs in passing the four weights as one `params` list."""
    z1 = x @ W1 + b1                # station 1: (batch,784) @ (784,128) -> (batch,128)
    hidden = relu(z1)               # the bend zeros the negative notes
    logits = hidden @ W2 + b2       # station 2: (batch,128) @ (128,10) -> (batch,10)
    return z1, hidden, logits       # z1 is the "pre-activation"; every later day calls it z1


def nearest_template_accuracy(x, labels):
    """A no-learning check on the DATA: label each image by the class whose average image
    it sits closest to. A high score means the data is readable, so a ~10% score later
    belongs to the untrained model and not to a broken dataset."""
    centroids = np.stack([x[labels == d].mean(axis=0) for d in range(10)])  # (10,784)
    distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)  # (N,10)
    return float((distances.argmin(axis=1) == labels).mean())


if __name__ == "__main__":
    rng = np.random.default_rng(0)  # one seed for data and weights -> same output each run

    # --- Part 1: the images, flattened and scaled to [0,1] ----------------
    raw_images, labels, pre_cast = make_stand_in_digits(n_per_class=50, rng=rng)
    n_images = raw_images.shape[0]
    # Every number below is BOUND before it is printed, and the self-check at the bottom
    # reads these same names. So the line you read and the line that is checked are one
    # value, not two expressions that happen to agree today.
    shown_raw_shape = raw_images.shape
    shown_raw_dtype = raw_images.dtype
    shown_raw_min = int(raw_images.min())
    shown_raw_max = int(raw_images.max())
    print("raw images       :", shown_raw_shape, shown_raw_dtype,
          "min", shown_raw_min, "max", shown_raw_max)
    x = raw_images.reshape(n_images, 28 * 28) / 255.0   # flatten the grid, then scale
    shown_x_shape = x.shape
    shown_x_min = round(float(x.min()), 3)
    shown_x_max = round(float(x.max()), 3)
    print("flattened+scaled :", shown_x_shape, "min", shown_x_min,
          "max", shown_x_max, "(one image is now a row of 784 numbers)")
    # min/max of x cannot see a broken pipeline: a negative float cast to uint8 WRAPS to a
    # bright pixel, and 0.0/1.0 stay 0.0/1.0 either way. So measure the pre-cast floats, how
    # far the cast moved each pixel, how much of the picture is still exactly black, and
    # whether two samples of one class actually differ.
    cast_gap = float(np.abs(raw_images.astype(np.float64) - pre_cast).max())
    black_share = float((raw_images == 0).mean())
    pair_gap = float(np.abs(x[0] - x[1]).max())
    shown_pre_min = round(float(pre_cast.min()), 1)
    shown_pre_max = round(float(pre_cast.max()), 1)
    shown_cast_gap = round(cast_gap, 3)
    shown_black_pct = round(black_share * 100, 1)
    shown_pair_gap = round(pair_gap, 3)
    print("pre-cast floats  : min", shown_pre_min,
          "max", shown_pre_max,
          "-> rounding moved a pixel by at most", shown_cast_gap,
          "| still exactly black:", shown_black_pct, "%")
    print("two samples of class 0 differ by up to", shown_pair_gap,
          "-> the pen wobble is alive; 50 samples of a class are not one repeated image")
    # Flattening lays the ROWS end to end, so slot k holds pixel (k // 28, k % 28). We
    # rebuild that map by hand and compare, which catches a column-first flatten.
    slot_row, slot_col = np.divmod(np.arange(784), 28)
    shown_slots = [(int(slot_row[k]), int(slot_col[k])) for k in (0, 1, 28)]
    layout_ok = (np.array_equal(x[7], (raw_images[7] / 255.0)[slot_row, slot_col])
                 and shown_slots == [(0, 0), (0, 1), (1, 0)])
    print("flat slots 0,1,28 of image 7 hold grid cells", shown_slots,
          "-> row-major:", layout_ok)

    # --- Part 2: predict first, from numbers we already have ---------------
    n_classes = len(np.unique(labels))
    predicted_logit_shape = (n_images, n_classes)      # one row of 10 scores per image
    predicted_accuracy = 1.0 / n_classes               # blind guessing over 10 digits
    shown_predicted_accuracy = round(predicted_accuracy, 3)
    print("\nprediction: logits", predicted_logit_shape, "and accuracy about",
          shown_predicted_accuracy)

    # --- Part 3: the knobs (He initialization) ----------------------------
    # LAYOUT, the convention days 1-3 all use: a weight matrix is (in, out). So W1 is
    # (784, 128) and W2 is (128, 10), and the forward pass reads x @ W1. PyTorch's nn.Linear
    # stores the TRANSPOSE, (out, in) — days 5 and 6 print that and multiply by W.T instead.
    W1 = he_init(28 * 28, 128, rng)
    b1 = np.zeros(128)          # biases start at zero
    W2 = he_init(128, 10, rng)
    b2 = np.zeros(10)
    w1_shape, b1_shape, w2_shape, b2_shape = W1.shape, b1.shape, W2.shape, b2.shape
    w1_count, b1_count, w2_count, b2_count = W1.size, b1.size, W2.size, b2.size
    n_params = w1_count + b1_count + w2_count + b2_count
    print("W1", w1_shape, "b1", b1_shape, "W2", w2_shape, "b2", b2_shape)
    # "knobs" here means weights PLUS biases. Day 4 prints a count of the WEIGHTS only, which
    # is a different quantity under a similar word; day 5 prints this same 101770 from torch.
    print("parameters       :", w1_count, "+", b1_count, "+", w2_count, "+", b2_count,
          "=", n_params, "knobs")
    shown_w1_std = round(float(W1.std()), 4)
    shown_w2_std = round(float(W2.std()), 4)
    print("He spread        : W1 std", shown_w1_std,
          " W2 std", shown_w2_std)

    # --- Part 4: one forward pass, printing the shape at every step -------
    z1, hidden, logits = forward(x, W1, b1, W2, b2)
    z1_shape, hidden_shape, logits_shape = z1.shape, hidden.shape, logits.shape
    shown_z1_min = round(float(z1.min()), 3)
    shown_hidden_min = round(float(hidden.min()), 3)
    shown_zero_pct = round(float((hidden == 0).mean()) * 100, 1)
    shown_logits_row0 = np.round(logits[0], 3)
    print("\nstep 1 input     :", shown_x_shape)
    print("step 2 x@W1+b1   :", z1_shape, "min", shown_z1_min)
    print("step 3 relu      :", hidden_shape, "min", shown_hidden_min,
          "-> zeroed", shown_zero_pct, "% of the notes")
    print("step 4 logits    :", logits_shape, "= one row of 10 scores per image")
    print("logits for image 0:", shown_logits_row0)

    # --- Part 5: read the guess with argmax, then score it -----------------
    guesses = logits.argmax(axis=1)          # position of the loudest of 10 voices
    accuracy = float((guesses == labels).mean())
    data_accuracy = nearest_template_accuracy(x, labels)
    # The guess must point at the LOUDEST score, so the logit at the picked slot has to be
    # that row's largest. Accuracy cannot check this: the quietest voice also scores ~10%.
    picked_score = logits[np.arange(n_images), guesses]
    argmax_ok = np.array_equal(picked_score, logits.max(axis=1))
    shown_guess0 = int(guesses[0])
    shown_label0 = int(labels[0])
    shown_guess_counts = np.bincount(guesses, minlength=10)
    shown_accuracy = round(accuracy, 4)
    shown_data_accuracy = round(data_accuracy, 4)
    print("\nimage 0: guess", shown_guess0, " true label", shown_label0,
          " guess counts per digit:", shown_guess_counts)
    print("every picked slot holds its row's biggest logit:", argmax_ok)
    print("untrained accuracy:", shown_accuracy, " chance level:", predicted_accuracy)
    print("same data, nearest-average-image labeller:", shown_data_accuracy,
          "-> the data is readable; the random knobs are what score ~10%")

    # --- Part 6: why the two layers need a bend between them --------------
    # Small hand-checkable stand-ins for W1 and W2 (the lesson's 3-input, 2-judge demo).
    Wa = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    Wb = np.array([[1.0, -1.0], [2.0, 1.0]])
    x_demo = np.array([[1.0, 2.0, 3.0]])
    # The lesson's z = x@W + b, run through the SAME forward() the big model uses, so its
    # printed [[7.5, 5.5]] depends on the "+ b1" inside that function.
    lesson_pre, _, _ = forward(x_demo, Wa, np.array([0.5, 0.5]), Wb, np.zeros(2))
    two_layer = (x_demo @ Wa) @ Wb           # two matmuls, no bend between them
    folded_pair = x_demo @ (Wa @ Wb)         # the same two matrices multiplied first
    print("\nlesson demo z = x@Wa + b :", lesson_pre)
    print("two layers (x@Wa)@Wb =", two_layer,
          " one layer x@(Wa@Wb) =", folded_pair, "-> the same numbers")
    x_mix = np.array([[1.0, -3.0, 0.5]])     # this input makes x@Wa mixed-sign
    mix_pre = x_mix @ Wa                     # the raw notes, before the bend
    mix_bent = relu(mix_pre)                 # the bend zeros the negative one
    bent = mix_bent @ Wb                     # bend first, then the second matmul
    folded = x_mix @ (Wa @ Wb)               # the two layers folded into one matrix
    print("x_mix@Wa =", mix_pre, "-> relu ->", mix_bent)
    print("with the bend:", bent, " folded flat:", folded,
          "-> NOT the same, so with a bend the stack is no longer one linear layer")

    # --- Part 7: what the two bias vectors do, once they are not zero ------
    # Above, b1 and b2 are all zeros, so "+ b1" and "+ b2" add nothing yet. Give the SAME
    # forward() two non-zero biases on the small demo weights, and the numbers stay small
    # enough to check by hand. Judge 2's raw note is -2.5, so the bend would delete it; a
    # +3.0 bias lifts it to +0.5 and it survives.
    b1_demo = np.array([0.5, 3.0])       # a head start for each of the 2 hidden judges
    b2_demo = np.array([-1.0, 0.5])      # a head start for each of the 2 output scores
    zero_pre, zero_hidden, zero_logits = forward(x_mix, Wa, np.zeros(2), Wb, np.zeros(2))
    biased_pre, biased_hidden, biased_logits = forward(x_mix, Wa, b1_demo, Wb, b2_demo)
    print("\nzero biases : pre", zero_pre, "-> relu", zero_hidden,
          "-> logits", zero_logits)
    print("with biases : pre", biased_pre, "-> relu", biased_hidden,
          "-> logits", biased_logits)
    print("adding b1 =", b1_demo, "and b2 =", b2_demo,
          "moved every number -> the two '+ b' terms are doing work")

    # --- Self-check: one boolean per claim, expected values written down here
    # Every check below reads the SAME name that was printed, so corrupting a printed
    # number is now corrupting the checked number too.
    shapes_ok = (shown_x_shape == (500, 784) and z1_shape == (500, 128)
                 and hidden_shape == (500, 128)
                 and logits_shape == (500, 10) and logits_shape == predicted_logit_shape)
    scaled_ok = (shown_x_min == 0.0 and shown_x_max == 1.0
                 and shown_raw_min == 0 and shown_raw_max == 255
                 and shown_raw_shape == (500, 28, 28) and shown_raw_dtype == np.uint8)
    # Clip BEFORE the cast, and round rather than truncate. Drop the clip and the pre-cast
    # floats run to [-210.9, 448.2] while wraparound moves a byte by 256.5; drop the .round()
    # and truncation moves a byte by a whole 1.0. Neither shows up in scaled_ok.
    cast_ok = (shown_pre_min == 0.0 and shown_pre_max == 255.0
               and shown_cast_gap <= 0.5)
    # A clipped stand-in keeps a big exactly-black background — but not an all-black one:
    # wraparound leaves 0.9% black, a dead wobble (std 0.0) leaves 84.3% and identical twins.
    wobble_ok = 30.0 < shown_black_pct < 60.0 and shown_pair_gap > 0.05
    params_ok = (n_params == 101770                      # the lesson's ~100k knobs
                 and (w1_count, b1_count, w2_count, b2_count) == (100352, 128, 1280, 10)
                 and w1_shape == (784, 128) and b1_shape == (128,)
                 and w2_shape == (128, 10) and b2_shape == (10,))
    he_ok = (abs(shown_w1_std - 0.0505) < 0.002          # sqrt(2/784) = 0.0505...
             and abs(shown_w2_std - 0.1250) < 0.008      # sqrt(2/128) = 0.125
             and not b1.any() and not b2.any())
    # He init is zero-mean, so roughly half the hidden notes start negative and the bend
    # deletes them. An all-positive or all-negative layer would be a broken bend.
    bend_ok = (shown_z1_min < 0.0 and shown_hidden_min == 0.0
               and 30.0 < shown_zero_pct < 70.0)
    # The printed row of 10 scores IS the evidence for the argmax below, so pin it to the
    # numbers this seed really produced (checked by running it, not by expecting it).
    row0_ok = np.allclose(shown_logits_row0,
                          [0.084, 1.124, -1.071, 0.695, -0.244,
                           0.543, 0.178, 0.502, -0.312, -0.269])
    # Image 0 is a class-0 sample the random model calls a 1, and the 500 guesses pile up on
    # a handful of digits instead of spreading evenly — both printed, so both pinned.
    guesses_ok = (shown_guess0 == 1 and shown_label0 == 0
                  and shown_guess_counts.tolist() == [2, 192, 0, 97, 0, 125, 60, 20, 0, 4]
                  and int(shown_guess_counts.sum()) == n_images)
    # A tight band around chance: 0.05 rules out the lesson's other offers (50%, 95%).
    chance_ok = (abs(shown_accuracy - predicted_accuracy) < 0.05
                 and shown_predicted_accuracy == 0.1)
    data_ok = shown_data_accuracy > 0.95                 # so ~10% is not the data's fault
    # Both paths must land on the SAME written-down numbers — not merely on each other.
    numbers_ok = (np.allclose(lesson_pre, [[7.5, 5.5]])
                  and np.allclose(two_layer, [[17.0, -2.0]])
                  and np.allclose(folded_pair, [[17.0, -2.0]])
                  and np.allclose(mix_pre, [[2.0, -2.5]])
                  and np.allclose(mix_bent, [[2.0, 0.0]])
                  and np.allclose(bent, [[2.0, -2.0]]) and np.allclose(folded, [[-3.0, -4.5]]))
    # The biases must change the answer. x_mix@Wa = [2.0,-2.5], so with b1 = [0.5,3.0] the
    # pre-bend row is [2.5,0.5] (both survive relu), and [2.5,0.5]@Wb = [3.5,-2.0], which
    # b2 = [-1.0,0.5] moves to [2.5,-1.5]. Drop either "+ b" and these numbers change.
    bias_ok = (np.allclose(zero_pre, [[2.0, -2.5]]) and np.allclose(zero_hidden, [[2.0, 0.0]])
               and np.allclose(zero_logits, [[2.0, -2.0]]) and np.allclose(zero_logits, bent)
               and np.allclose(biased_pre, [[2.5, 0.5]])
               and np.allclose(biased_hidden, [[2.5, 0.5]])
               and np.allclose(biased_logits, [[2.5, -1.5]]))

    claims = [
        (shapes_ok, "shapes flowing (500,784) -> (500,128) -> (500,10)"),
        (scaled_ok, "byte pixels spanning 0-255 that span exactly 0.0 to 1.0 after /255"),
        (cast_ok, "clip-then-cast: pre-cast floats inside [0.0, 255.0] and the round moving "
                  "a pixel by at most 0.5 (no uint8 wraparound, no truncation)"),
        (wobble_ok, "live pen wobble: 30-60% of pixels exactly black and two samples of one "
                    "class differing by more than 0.05"),
        (layout_ok, "slot k of the flat row holding grid pixel (k // 28, k % 28), so slots "
                    "0,1,28 are cells (0,0), (0,1), (1,0)"),
        (params_ok, "784*128 + 128 + 128*10 + 10 = 101770 parameters, from W1 (784,128), "
                    "b1 (128,), W2 (128,10), b2 (10,)"),
        (he_ok, "He init: W1 std ~0.0505, W2 std ~0.125, biases exactly zero"),
        (bend_ok, "relu clipping real negatives to exactly 0, zeroing 30-70% of the notes"),
        (row0_ok, "image 0's printed logit row being the row this seed produces, "
                  "starting [0.084, 1.124, -1.071, ...]"),
        (argmax_ok, "the guessed slot being the position of the row's largest logit"),
        (guesses_ok, "image 0 (a true 0) being guessed as a 1, and the 500 guesses landing "
                     "as [2,192,0,97,0,125,60,20,0,4]"),
        (chance_ok, "untrained accuracy near chance (1/10), nowhere near 50% or 95%"),
        (data_ok, "readable stand-in digits, else ~10% would prove nothing"),
        (numbers_ok, "z=[[7.5,5.5]], both two-layer paths [[17,-2]], x_mix@Wa=[[2,-2.5]] "
                     "bending to [[2,0]], and the bend giving [[2,-2]] where the folded "
                     "single matrix gives [[-3,-4.5]]"),
        (bias_ok, "the '+ b' terms carrying their weight: b1=[0.5,3.0] turning pre-bend "
                  "[[2,-2.5]] into [[2.5,0.5]], then b2=[-1,0.5] turning logits [[3.5,-2]] "
                  "into [[2.5,-1.5]] (and zero biases reproducing [[2,-2]])"),
    ]
    if all(ok for ok, _ in claims):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected " + "; ".join(why for ok, why in claims if not ok))

    for ok, why in claims:      # each claim stops the program on its own, with its message
        assert ok, why
