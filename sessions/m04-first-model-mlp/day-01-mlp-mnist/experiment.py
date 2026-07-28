# day-01-mlp-mnist — experiment
#
# Today's big idea in two lines of output:
#   A forward pass is shapes changing hands: (batch,784) -> (batch,128) -> (batch,10).
#   The wiring is already right, but every knob is random — so it guesses ~10%.
#
# This script (1) makes a stand-in digit set, (2) builds an UNTRAINED 784->128->10 MLP
# with He init, (3) runs one forward pass and reads its guess, (4) shows why layers bend.
# Run it:  python3 sessions/m04-first-model-mlp/day-01-mlp-mnist/experiment.py

import numpy as np  # numpy gives us arrays, matrix multiply (@) and argmax


def make_stand_in_digits(n_per_class, rng):
    """DEVIATION: real MNIST is not on this machine and downloading is blocked, so we draw
    our own 28x28 'handwriting'. Each class gets one ink pattern (two strokes), then every
    sample adds pen wobble. Same shapes as MNIST: 28x28 bytes 0-255, 10 classes."""
    templates = np.zeros((10, 28, 28))
    for digit in range(10):
        row, col = rng.integers(4, 22, size=2)      # where this digit's strokes sit
        templates[digit, row:row + 3, 3:25] = 1.0   # one horizontal stroke
        templates[digit, 3:25, col:col + 3] = 1.0   # one vertical stroke
    images = np.repeat(templates, n_per_class, axis=0)          # (10*n, 28, 28)
    labels = np.repeat(np.arange(10), n_per_class)              # (10*n,)
    wobble = rng.normal(0.0, 0.18, size=images.shape)           # a little noise per pixel
    pixels = np.clip(images + wobble, 0.0, 1.0) * 255.0         # clip first, then to bytes
    return pixels.round().astype(np.uint8), labels


def he_init(n_in, n_out, rng):
    """He initialization: normal random numbers scaled by sqrt(2 / n_in). Built for
    ReLU, so fewer hidden units are born stuck at zero (the 'dead ReLU' failure mode)."""
    return rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)


def relu(z):
    # The bend: keep positives, turn negatives into 0.
    return np.maximum(0, z)


def forward(x, W1, b1, W2, b2):
    """One forward pass: hidden values before the bend, after the bend, and the logits."""
    hidden_pre = x @ W1 + b1        # station 1: (batch,784) @ (784,128) -> (batch,128)
    hidden = relu(hidden_pre)       # the bend zeros the negative notes
    logits = hidden @ W2 + b2       # station 2: (batch,128) @ (128,10) -> (batch,10)
    return hidden_pre, hidden, logits


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
    raw_images, labels = make_stand_in_digits(n_per_class=50, rng=rng)
    n_images = raw_images.shape[0]
    print("raw images       :", raw_images.shape, raw_images.dtype,
          "min", int(raw_images.min()), "max", int(raw_images.max()))
    x = raw_images.reshape(n_images, 28 * 28) / 255.0   # flatten the grid, then scale
    print("flattened+scaled :", x.shape, "min", round(float(x.min()), 3),
          "max", round(float(x.max()), 3), "(one image is now a row of 784 numbers)")
    # Flattening lays the ROWS end to end, so slot k holds pixel (k // 28, k % 28). We
    # rebuild that map by hand and compare, which catches a column-first flatten.
    slot_row, slot_col = np.divmod(np.arange(784), 28)
    layout_ok = np.array_equal(x[7], (raw_images[7] / 255.0)[slot_row, slot_col])
    print("flat slots 0,1,28 of image 7 hold grid cells",
          [(int(slot_row[k]), int(slot_col[k])) for k in (0, 1, 28)],
          "-> row-major:", layout_ok)

    # --- Part 2: predict first, from numbers we already have ---------------
    n_classes = len(np.unique(labels))
    predicted_logit_shape = (n_images, n_classes)      # one row of 10 scores per image
    predicted_accuracy = 1.0 / n_classes               # blind guessing over 10 digits
    print("\nprediction: logits", predicted_logit_shape, "and accuracy about",
          round(predicted_accuracy, 3))

    # --- Part 3: the knobs (He initialization) ----------------------------
    W1 = he_init(28 * 28, 128, rng)
    b1 = np.zeros(128)          # biases start at zero
    W2 = he_init(128, 10, rng)
    b2 = np.zeros(10)
    n_params = W1.size + b1.size + W2.size + b2.size
    print("W1", W1.shape, "b1", b1.shape, "W2", W2.shape, "b2", b2.shape)
    print("parameters       :", W1.size, "+", b1.size, "+", W2.size, "+", b2.size,
          "=", n_params, "knobs")
    print("He spread        : W1 std", round(float(W1.std()), 4),
          " W2 std", round(float(W2.std()), 4))

    # --- Part 4: one forward pass, printing the shape at every step -------
    hidden_pre, hidden, logits = forward(x, W1, b1, W2, b2)
    print("\nstep 1 input     :", x.shape)
    print("step 2 x@W1+b1   :", hidden_pre.shape, "min", round(float(hidden_pre.min()), 3))
    print("step 3 relu      :", hidden.shape, "min", round(float(hidden.min()), 3),
          "-> zeroed", round(float((hidden == 0).mean()) * 100, 1), "% of the notes")
    print("step 4 logits    :", logits.shape, "= one row of 10 scores per image")
    print("logits for image 0:", np.round(logits[0], 3))

    # --- Part 5: read the guess with argmax, then score it -----------------
    guesses = logits.argmax(axis=1)          # position of the loudest of 10 voices
    accuracy = float((guesses == labels).mean())
    data_accuracy = nearest_template_accuracy(x, labels)
    # The guess must point at the LOUDEST score, so the logit at the picked slot has to be
    # that row's largest. Accuracy cannot check this: the quietest voice also scores ~10%.
    picked_score = logits[np.arange(n_images), guesses]
    argmax_ok = np.array_equal(picked_score, logits.max(axis=1))
    print("\nimage 0: guess", int(guesses[0]), " true label", int(labels[0]),
          " guess counts per digit:", np.bincount(guesses, minlength=10))
    print("every picked slot holds its row's biggest logit:", argmax_ok)
    print("untrained accuracy:", round(accuracy, 4), " chance level:", predicted_accuracy)
    print("same data, nearest-average-image labeller:", round(data_accuracy, 4),
          "-> the data is readable; the random knobs are what score ~10%")

    # --- Part 6: why the two layers need a bend between them --------------
    # Small hand-checkable stand-ins for W1 and W2 (the lesson's 3-input, 2-judge demo).
    Wa = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    Wb = np.array([[1.0, -1.0], [2.0, 1.0]])
    x_demo = np.array([[1.0, 2.0, 3.0]])
    print("\nlesson demo z = x@Wa + b :", x_demo @ Wa + np.array([0.5, 0.5]))
    print("two layers (x@Wa)@Wb =", (x_demo @ Wa) @ Wb,
          " one layer x@(Wa@Wb) =", x_demo @ (Wa @ Wb), "-> the same numbers")
    x_mix = np.array([[1.0, -3.0, 0.5]])     # this input makes x@Wa mixed-sign
    bent = relu(x_mix @ Wa) @ Wb             # bend first, then the second matmul
    folded = x_mix @ (Wa @ Wb)               # the two layers folded into one matrix
    print("x_mix@Wa =", x_mix @ Wa, "-> relu ->", relu(x_mix @ Wa))
    print("with the bend:", bent, " folded flat:", folded,
          "-> NOT the same, so with a bend the stack is no longer one linear layer")

    # --- Self-check: one boolean per claim, expected values written down here
    shapes_ok = (x.shape == (500, 784) and hidden.shape == (500, 128)
                 and logits.shape == (500, 10) and logits.shape == predicted_logit_shape)
    scaled_ok = x.min() == 0.0 and x.max() == 1.0 and raw_images.dtype == np.uint8
    params_ok = n_params == 101770                       # the lesson's ~100k knobs
    he_ok = (abs(float(W1.std()) - 0.0505) < 0.002       # sqrt(2/784) = 0.0505...
             and abs(float(W2.std()) - 0.1250) < 0.008   # sqrt(2/128) = 0.125
             and not b1.any() and not b2.any())
    bend_ok = float(hidden_pre.min()) < 0.0 and float(hidden.min()) == 0.0
    # A tight band around chance: 0.05 rules out the lesson's other offers (50%, 95%).
    chance_ok = abs(accuracy - predicted_accuracy) < 0.05
    data_ok = data_accuracy > 0.95                       # so ~10% is not the data's fault
    # Both paths must land on the SAME written-down numbers — not merely on each other.
    numbers_ok = (np.allclose(x_demo @ Wa + np.array([0.5, 0.5]), [[7.5, 5.5]])
                  and np.allclose((x_demo @ Wa) @ Wb, [[17.0, -2.0]])
                  and np.allclose(x_demo @ (Wa @ Wb), [[17.0, -2.0]])
                  and np.allclose(bent, [[2.0, -2.0]]) and np.allclose(folded, [[-3.0, -4.5]]))

    claims = [
        (shapes_ok, "shapes flowing (500,784) -> (500,128) -> (500,10)"),
        (scaled_ok, "byte pixels that span exactly 0.0 to 1.0 after dividing by 255"),
        (layout_ok, "slot k of the flat row holding grid pixel (k // 28, k % 28)"),
        (params_ok, "784*128 + 128 + 128*10 + 10 = 101770 parameters"),
        (he_ok, "He init: W1 std ~0.0505, W2 std ~0.125, biases exactly zero"),
        (bend_ok, "relu clipping real negatives to exactly 0"),
        (argmax_ok, "the guessed slot being the position of the row's largest logit"),
        (chance_ok, "untrained accuracy near chance (1/10), nowhere near 50% or 95%"),
        (data_ok, "readable stand-in digits, else ~10% would prove nothing"),
        (numbers_ok, "z=[[7.5,5.5]], both two-layer paths [[17,-2]], and the bend giving "
                     "[[2,-2]] where the folded single matrix gives [[-3,-4.5]]"),
    ]
    if all(ok for ok, _ in claims):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected " + "; ".join(why for ok, why in claims if not ok))

    for ok, why in claims:      # each claim stops the program on its own, with its message
        assert ok, why
