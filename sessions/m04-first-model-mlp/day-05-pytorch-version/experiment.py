# day-05-pytorch-version — experiment
#
# Today's big idea in two lines of output:
#   The same 784→128→10 MLP you hand-wrote in numpy fits in a handful of PyTorch lines.
#   Its loss falls the same way, because autograd redoes your hand-derived backward pass.
#
# Parts: (1) stand-in data, (2) the model and how few lines it takes, (3) the five-line loop,
# (4) a taste test against a hand-written numpy twin, (5) the same loop with zero_grad() deleted,
# (6) the hand-derived ReLU gate at exactly z = 0, where (z > 0) and (z >= 0) part ways.
# Run it:  python3 sessions/m04-first-model-mlp/day-05-pytorch-version/experiment.py

import ast                   # ast reads this file's own code, to count how short it is
import math                  # math.log(10) is the loss we expect BEFORE any learning
import numpy as np           # numpy makes the stand-in data and the hand-written twin
import torch                 # torch gives us tensors: grids of numbers that can record their history
import torch.nn as nn        # nn holds the ready-made pieces: Linear, ReLU, CrossEntropyLoss
import torch.optim as optim  # optim holds SGD, the helper that applies each nudge

torch.set_num_threads(1)                 # one thread keeps every run byte-identical
PIXELS, HIDDEN, CLASSES = 784, 128, 10   # the three sizes the lesson uses
EPOCHS, LR = 25, 0.3                     # rounds of practice, and how big each nudge is
HAND_WRITTEN_LINES = 500                 # the lesson's count for the numpy build of days 1-4


def make_digit_data(distinct=128, copies=8, agree=5):
    """A seeded stand-in for MNIST — this machine has no MNIST on disk and no network. One
    random 784-pixel prototype per class, plus noise, gives `distinct` fake images; each is
    repeated `copies` times and only `agree` copies keep the true label, like real people
    disagreeing about smudged handwriting. That disagreement is deliberate: it stops the loss
    reaching 0, so gradients stay alive to the last epoch and the Part 5 bug stays visible."""
    rng = np.random.default_rng(0)
    prototypes = rng.normal(0, 1, size=(CLASSES, PIXELS)).astype(np.float32)
    true_class = np.tile(np.arange(CLASSES), distinct // CLASSES + 1)[:distinct]
    images = (prototypes[true_class] + rng.normal(0, 1, size=(distinct, PIXELS))).astype(np.float32)
    images = images / images.std()        # keep pixel values near 1, like scaled pixels
    X, y = np.repeat(images, copies, axis=0), np.repeat(true_class, copies).copy()
    for i in range(distinct):
        for j in range(agree, copies):    # the disagreeing copies get a random label
            y[i * copies + j] = rng.integers(0, CLASSES)
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)

class MLP(nn.Module):
    """The lesson's model: 784 → Linear → ReLU → Linear → 10."""
    def __init__(self):
        super().__init__()
        # Days 1-3 built these by hand as (in, out): W1 was (784, 128) and W2 was (128, 10).
        # nn.Linear keeps the TRANSPOSE of that — the printout below says (128, 784) and
        # (10, 128) — which is exactly why the Part 4 twin has to write X @ W1.T.
        self.fc1 = nn.Linear(PIXELS, HIDDEN)    # replaces the hand-made W1 (784, 128) and b1
        self.relu = nn.ReLU()                   # the bend: keeps positives, zeroes negatives
        self.fc2 = nn.Linear(HIDDEN, CLASSES)   # replaces the hand-made W2 (128, 10) and b2

    def forward(self, x):                       # this IS the hand-written forward pass
        return self.fc2(self.relu(self.fc1(x)))

def fresh_model():
    torch.manual_seed(0)   # same seed each time, so every run below starts from equal weights
    return MLP()

def code_lines(*names):
    """How many real lines of code do these definitions take? Read out of this file itself, so
    the number cannot drift from the code it describes. Blank lines, comment-only lines and
    docstrings do not count — they are not work the reader has to think through."""
    lines = open(__file__, encoding="utf-8").read().splitlines()
    counted = 0
    for node in ast.parse("\n".join(lines)).body:
        if getattr(node, "name", None) not in names:
            continue
        # a bare string on its own line is a docstring, not code, so skip those line numbers
        prose = {i for inner in ast.walk(node)
                 if isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Constant)
                 and isinstance(inner.value.value, str)
                 for i in range(inner.lineno, inner.end_lineno + 1)}
        counted += sum(1 for i in range(node.lineno, node.end_lineno + 1)
                       if lines[i - 1].strip() and not lines[i - 1].strip().startswith("#")
                       and i not in prose)
    return counted

def gradient_size(model):
    # One number for the whole model: how big is the gradient the optimizer will apply?
    # A weight whose .grad slot is still empty counts as 0 — that is a cleared slot, not a crash.
    return math.sqrt(sum(float(p.grad.norm()) ** 2 if p.grad is not None else 0.0
                         for p in model.parameters()))

def epoch_row(epoch, losses, grad_sizes):
    """Everything one row of the training table shows: the epoch, its loss and gradient size
    rounded EXACTLY as printed, and the "<- went UP" marker. The loop below prints what this
    returns and the self-check reads the same rows back, so the table on the page and the numbers
    the claims quote are one thing, computed once. Note it is deliberately NOT counted as part of
    the loop's length below — it is this artifact's bookkeeping, not the PyTorch loop."""
    return (epoch, round(losses[epoch], 4), round(grad_sizes[epoch], 3),
            "   <- went UP" if epoch and losses[epoch] > losses[epoch - 1] else "")

def epoch_line(row):
    """The exact TEXT one table row puts on the screen, built from that row and nothing else. The
    loop below prints what this returns, and the self-check pins these strings against the text
    written down from a real run — so swapping the loss and gradient columns, wrapping either one
    in arithmetic, or flipping the marker all change this string and are caught. Like epoch_row,
    it is deliberately NOT counted as part of the loop's length."""
    return f"  epoch {row[0]:2d}  loss {row[1]:.4f}  gradient size {row[2]:.3f}{row[3]}"

def run_loop(model, X, y, clear_gradients):
    """The whole training loop. Line 1 is the one everybody forgets, so Part 5 can switch
    it off — that single line is the ONLY difference between the two runs.
    Note what `criterion` eats: the model's RAW SCORES. nn.CrossEntropyLoss runs the softmax
    itself, so it must be fed logits — not the probabilities day 4's cross_entropy(probs, ...)
    wanted. Part 3 measures what a leftover softmax does to this number."""
    criterion = nn.CrossEntropyLoss()                 # the built-in loss module: logits in
    optimizer = optim.SGD(model.parameters(), lr=LR)  # .parameters() hands over every weight
    losses, grad_sizes = [], []
    for epoch in range(EPOCHS):
        if clear_gradients:
            optimizer.zero_grad()             # 1 clear the piled-up gradients
        out = model(X)                        # 2 forward -> raw scores, no softmax applied
        loss = criterion(out, y)              # 3 score
        loss.backward()                       # 4 retrace the trail -> fills every .grad
        grad_sizes.append(gradient_size(model))
        optimizer.step()                      # 5 nudge every weight
        losses.append(loss.item())            # .item() pulls a plain number out of the tensor
        row = epoch_row(epoch, losses, grad_sizes)     # the one rule for what this row shows
        print(epoch_line(row))                         # the one rule for how that row is WRITTEN
    return losses, grad_sizes

def relu_gate(z):
    # The gate the hand-written backward pass multiplies by: STRICTLY greater than zero. A unit
    # sitting exactly ON the bend (z = 0) is OFF and must receive no blame. No z1 in the twin below
    # is ever exactly 0.0, so writing (z >= 0) here would change nothing you could see — which is
    # why Part 6 builds a net whose z1 IS 0 and runs this gate both ways.
    return z > 0

def twin_step(W1, b1, W2, b2, X, y, onehot, gate=relu_gate):
    """One forward + backward of the hand-written twin: the chain rule days 1-4 derived, with no
    autograd anywhere. Pulled out of the epoch loop so Part 6 can run the SAME lines on a small
    hand-built net whose z1 lands exactly on the ReLU bend."""
    z1 = X @ W1.T + b1                       # nn.Linear, by hand
    hidden = np.maximum(0, z1)               # nn.ReLU, by hand
    out = hidden @ W2.T + b2                 # the second nn.Linear -> day 1's logits
    P = np.exp(out - out.max(axis=1, keepdims=True)); P /= P.sum(axis=1, keepdims=True)
    rows = len(y)
    loss = float(-np.log(P[np.arange(rows), y]).mean())    # cross-entropy, by hand
    d_out = (P - onehot) / rows              # the chain rule, derived by hand
    dW2, db2, d_hidden = d_out.T @ hidden, d_out.sum(axis=0), d_out @ W2
    d_z1 = d_hidden * gate(z1)               # the ReLU gate blocks the non-positive positions
    dW1, db1 = d_z1.T @ X, d_z1.sum(axis=0)
    return loss, z1, (dW1, db1, dW2, db2)

def numpy_twin_losses(model, X, y, epochs):
    """The day-2 / day-4 version: same starting weights, but forward, backward and update
    written out by hand in numpy. Nothing here uses autograd, so agreement means something.
    The names inside twin_step are days 1-4's: z1 for the pre-activation, hidden for the bent
    values, out for the raw scores, d_out for day 2's seed delta. W1/W2 come out of nn.Linear, so
    they are stored (out, in) — the transpose of days 1-3 — which is why every matmul takes a .T."""
    W1 = model.fc1.weight.detach().numpy().copy(); b1 = model.fc1.bias.detach().numpy().copy()
    W2 = model.fc2.weight.detach().numpy().copy(); b2 = model.fc2.bias.detach().numpy().copy()
    X, y, losses = X.numpy(), y.numpy(), []
    # onehot[i] is a row of zeros with a single 1 in the true class — the target we compare to
    onehot = np.eye(CLASSES, dtype=np.float32)[y]
    for _ in range(epochs):
        loss, _, (dW1, db1, dW2, db2) = twin_step(W1, b1, W2, b2, X, y, onehot)
        losses.append(loss)
        W1 -= LR * dW1; b1 -= LR * db1; W2 -= LR * dW2; b2 -= LR * db2   # optimizer.step(), by hand
    return losses


if __name__ == "__main__":
    # --- Part 1: the data (a seeded stand-in, not real MNIST) --------------
    X, y = make_digit_data()
    # Bound once, printed, and read back by the self-check: the two shapes, the first labels, the
    # class name, and the requires_grad flag are all evidence the day rests on.
    shown_X_shape, shown_y_shape = tuple(X.shape), tuple(y.shape)
    shown_first_labels = y[:8].tolist()
    shown_X_type = type(X).__name__
    shown_X_requires_grad = X.requires_grad
    print("X shape:", shown_X_shape, " y shape:", shown_y_shape,
          " first labels:", shown_first_labels)
    print("X type:", shown_X_type, "- the same grid of numbers a numpy array holds, plus the"
          " power to record how it was computed, so gradients can flow back")
    print("this X is plain data, so there is nothing to record:",
          f"X.requires_grad = {shown_X_requires_grad}",
          "- the weights below say True, which is why gradients reach them")

    # --- Part 2: the model, how short it is, and the handle that finds every weight ---
    model = fresh_model()
    layout = [(name, tuple(p.shape)) for name, p in model.named_parameters()]
    total_numbers = sum(p.numel() for p in model.parameters())
    weights_track_history = all(p.requires_grad for p in model.parameters())
    # torch reports fc1.weight as (128, 784): the TRANSPOSE of the (784, 128) days 1-3 built by
    # hand. Same 101770 numbers as day 1's "knobs" — biases included, same formula.
    print("\nparameters:", layout, "\nnumbers to learn:", total_numbers,
          "= 784*128 + 128 + 128*10 + 10", "\nevery weight records its history:", weights_track_history)
    # Predict before you look: how many lines is this PyTorch version, next to the ~500 you
    # hand-wrote across days 1-4? The count comes from this file, not from a number typed in.
    # (The Part 4 twin is a squeezed-down copy of that build, so it is not the ~500-line rival.)
    print(f"\npredict: fewer lines than the ~{HAND_WRITTEN_LINES} hand-written ones, or more?")
    torch_lines = code_lines("MLP", "run_loop")
    shown_shrink = HAND_WRITTEN_LINES // torch_lines    # the "about 20x" the next line prints
    # Built as ONE string, pinned below, then printed: the line's whole point is which number is
    # the line count and which is the shrink, so swapping them would sell the opposite story.
    shrink_line = (f"counted from this file: the model + its whole training loop = {torch_lines}"
                   f" lines of code -> about {shown_shrink}x less typing for the same network")
    print(shrink_line)

    # --- Part 3: the five-line loop, with zero_grad in place ---------------
    # Predict before running: an untrained model has learned nothing, so it should be no better
    # than guessing 1 class out of CLASSES. Guesses that flat put the first loss near ln(CLASSES).
    # The guessing is measured here; ln(CLASSES) is computed here. Neither is typed in.
    predicted_start = math.log(CLASSES)
    with torch.no_grad():                      # a look at the untrained model, recording nothing
        start_probs = torch.softmax(model(X), dim=1)
    untrained_accuracy = float((start_probs.argmax(dim=1) == y).float().mean())
    top_probability = float(start_probs.max())
    guarded = start_probs.requires_grad is False and start_probs.grad_fn is None
    watched = torch.softmax(model(X), dim=1)   # the SAME forward, outside the no_grad block
    # The two grad_fn readings and the chance line, bound before they are printed. `guarded` above
    # asks the tensor itself; these are what the page shows, and both are checked below.
    shown_quiet_grad_fn = start_probs.grad_fn                     # None, inside no_grad
    shown_watched_grad_fn = type(watched.grad_fn).__name__        # a real recorded step, outside it
    shown_chance = 1 / CLASSES
    print(f"\ninside torch.no_grad() that forward recorded nothing: grad_fn {shown_quiet_grad_fn};"
          f" the same line outside records one: grad_fn {shown_watched_grad_fn}"
          " - recording is a switch you control, not something a tensor always does")
    print(f"\nuntrained model: {untrained_accuracy:.1%} of labels right, and its most confident"
          f" single guess is only {top_probability:.3f} - flat guessing, chance is {shown_chance:.1%}"
          f"\npredict: so epoch 0 loss should sit near ln({CLASSES}) = {predicted_start:.4f},"
          " then fall")
    # What nn.CrossEntropyLoss eats — the one day-4 habit that must NOT carry over. Day 4's
    # cross_entropy() took PROBABILITIES, so it was always handed softmax output. This loss runs
    # log-softmax itself, so it takes the RAW scores. Both halves are measured on the untrained
    # model below: day 4's formula on softmax(logits) is the SAME number as this loss on logits,
    # and handing it probabilities instead raises no error at all — it just softmaxes twice,
    # flattening the model's confidence and quietly scoring a different objective.
    eats = nn.CrossEntropyLoss()
    with torch.no_grad():
        loss_on_logits = float(eats(model(X), y))
        loss_on_probs = float(eats(start_probs, y))     # the double-softmax bug, run on purpose
        day4_on_probs = float(-torch.log(start_probs[torch.arange(len(y)), y]).mean())
        twice_top = float(torch.softmax(start_probs, dim=1).max())
    print(f"nn.CrossEntropyLoss(raw scores) {loss_on_logits:.5f} == day 4's -log(p) on softmax"
          f" output {day4_on_probs:.5f} -> the same objective, entered from the other side")
    print(f"the same loss handed PROBABILITIES: {loss_on_probs:.5f} — no error, but softmax ran"
          f" twice and the top probability collapsed {top_probability:.3f} -> {twice_top:.3f},"
          " so the number looks better while the model it scores has been flattened")
    print("with zero_grad():")
    good_losses, good_grads = run_loop(model, X, y, clear_gradients=True)
    # The table that was just printed, read back through the SAME row rule the prints used, plus
    # the gap the next line shows. Nothing here re-derives a printed number a second way.
    good_table = [epoch_row(e, good_losses, good_grads) for e in range(EPOCHS)]
    good_lines = [epoch_line(row) for row in good_table]     # the 25 lines the table just printed
    shown_start_gap = round(abs(good_losses[0] - predicted_start), 4)
    # One string, pinned below, then printed: the measured loss and ln(10) sit side by side here,
    # so swapping them (or wrapping either) would read as a prediction that was never made.
    check_line = (f"prediction check: epoch 0 loss was {good_losses[0]:.4f}, ln({CLASSES}) ="
                  f" {predicted_start:.4f}, gap {shown_start_gap:.4f}")
    print(check_line)

    # --- Part 4: the taste test against the hand-written twin --------------
    twin_epochs = 8
    twin_losses = numpy_twin_losses(fresh_model(), X, y, twin_epochs)
    biggest_gap = max(abs(a - b) for a, b in zip(good_losses[:twin_epochs], twin_losses))
    # The two rows of numbers, bound before printing so the claims quote the printed rows.
    shown_torch_row = [round(v, 5) for v in good_losses[:twin_epochs]]
    shown_twin_row = [round(v, 5) for v in twin_losses]
    print("\nPyTorch:", shown_torch_row,
          "\nby hand:", shown_twin_row,
          f"\nbiggest gap: {biggest_gap:.2e}  <- same weights, same math, same learning")

    # --- Part 5: the same loop with zero_grad() deleted --------------------
    # First the two small facts behind the bug, using the lesson's own worked numbers.
    w = torch.tensor([2.0], requires_grad=True)
    empty_slot = w.grad                                  # None — nothing recorded yet
    ((3 * w + 1) ** 2).backward()                         # the slope at w=2 is 2*(3*2+1)*3 = 42
    filled_slot = w.grad.item()                           # autograd put that 42 in the slot
    w = torch.tensor([2.0], requires_grad=True)           # a fresh weight, so the slot is empty again
    (3 * w).backward(); one_backward = w.grad.item()      # the true slope of 3w is 3
    (3 * w).backward(); two_backwards = w.grad.item()     # ...and now the slot holds 6
    print(f"\n.grad before backward: {empty_slot}   after: {filled_slot}   <- autograd filled it"
          f"\nafter 1 backward: {one_backward}   after 2 with no clearing: {two_backwards}"
          "  <- it added, it did not replace\nno zero_grad():")
    bug_losses, bug_grads = run_loop(fresh_model(), X, y, clear_gradients=False)
    # Both tables, read back through the row rule that printed them. `rises` now COUNTS THE MARKERS
    # the table showed instead of re-deciding which epochs rose, and the good run's rise count is
    # counted the same way rather than typed as a 0 into the sentence below.
    bug_table = [epoch_row(e, bug_losses, bug_grads) for e in range(EPOCHS)]
    bug_lines = [epoch_line(row) for row in bug_table]       # the 25 lines the bug table printed
    rises = sum(1 for row in bug_table if row[3])
    good_rises = sum(1 for row in good_table if row[3])
    shown_good_first, shown_good_last = round(good_grads[0], 2), round(good_grads[-1], 2)
    shown_bug_first, shown_bug_last = round(bug_grads[0], 2), round(bug_grads[-1], 2)
    shown_steps = EPOCHS - 1        # 25 epochs, so 24 steps BETWEEN them — bound, not left at the
                                    # print, where "of 25 steps" would slip past every check
    # The day's headline contrast, built as ONE string and pinned below before it is printed. Both
    # halves quote the same two columns, so a swap here would say the buggy run is the tidy one.
    contrast_lines = (
        f"\nwith zero_grad: loss rose in {good_rises} of {shown_steps} steps, gradient size"
        f" {shown_good_first:.2f} -> {shown_good_last:.2f} (shrinks, as it should)\n"
        f"without it:     loss rose in {rises} of {shown_steps} steps, gradient size"
        f" {shown_bug_first:.2f} -> {shown_bug_last:.2f} (a growing pile-up)")
    print(contrast_lines)

    # --- Part 6: the ReLU gate is strict — a unit exactly on the bend -------
    # Every z1 in the twin above came from random weights, so not one of them was ever exactly 0.0
    # and the difference between (z > 0) and (z >= 0) could not show. This hand-built net puts the
    # second hidden unit exactly ON the bend — 2*2 + 4*(-1) = 0 — and runs the twin's own step
    # twice, once with each spelling. The numbers are chosen to differ from the earlier days'
    # hinge nets, and the layout is PyTorch's (out, in), the same one the twin uses.
    hinge_X = np.array([[2.0, 4.0]], dtype=np.float32)
    hinge_W1 = np.array([[1.0, 0.5], [2.0, -1.0]], dtype=np.float32)   # row 2 cancels to exactly 0
    hinge_W2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)    # one hidden unit per class
    hinge_b1, hinge_b2 = np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
    hinge_y = np.array([1])                          # the true class is the dead unit's own class
    hinge_onehot = np.eye(2, dtype=np.float32)[hinge_y]
    hinge_loss, hinge_z1, hinge_grads = twin_step(
        hinge_W1, hinge_b1, hinge_W2, hinge_b2, hinge_X, hinge_y, hinge_onehot)
    loose_loss, loose_z1, loose_grads = twin_step(
        hinge_W1, hinge_b1, hinge_W2, hinge_b2, hinge_X, hinge_y, hinge_onehot,
        gate=lambda z: z >= 0)
    shown_hinge_z1 = [round(float(v), 4) for v in hinge_z1[0]]
    shown_hinge_loss, shown_loose_loss = round(hinge_loss, 4), round(loose_loss, 4)
    # dW1 rows: one per hidden unit, rounded to what the two lines below print
    shown_live_row = [round(float(v), 4) for v in hinge_grads[0][0]]
    shown_dead_row = [round(float(v), 4) for v in hinge_grads[0][1]]
    shown_loose_dead_row = [round(float(v), 4) for v in loose_grads[0][1]]
    # Two strings, pinned below, then printed. Which row belongs to the LIVE unit and which to the
    # z = 0 unit is the entire claim, so the labels and the rows are frozen together as text.
    hinge_lines = [
        f"\nhinge net: z1 = {shown_hinge_z1} (loss {shown_hinge_loss}) -> dW1 for the live unit"
        f" {shown_live_row}, for the z = 0 unit {shown_dead_row}",
        f"the same step with a (z >= 0) gate (loss {shown_loose_loss}, unchanged) hands that dead"
        f" unit {shown_loose_dead_row} instead — same forward, different weights learned: the"
        " one-character change no random run can show you",
    ]
    for hinge_line in hinge_lines:
        print(hinge_line)

    # --- Self-check: one boolean per claim ---------------------------------
    # The pinned numbers below were WRITTEN DOWN after running this file, so they do not come
    # from the code they test. Break the model or the loop and they stop matching.
    want_layout = [("fc1.weight", (128, 784)), ("fc1.bias", (128,)),
                   ("fc2.weight", (10, 128)), ("fc2.bias", (10,))]
    claims = {
        "the 784->128->10 layout and 101770 numbers in total":
            layout == want_layout and total_numbers == 101770,
        "1024 stand-in images of 784 pixels, 1024 labels, and [0, 0, 0, 0, 0, 1, 3, 7] first up":
            shown_X_shape == (1024, 784) and shown_y_shape == (1024,)
            and shown_first_labels == [0, 0, 0, 0, 0, 1, 3, 7] and shown_X_type == "Tensor",
        "plain data with requires_grad False, every weight with requires_grad True, and the"
        " no_grad block recording nothing while the same forward outside it records the"
        " SoftmaxBackward0 step by name":
            shown_X_requires_grad is False and X.requires_grad is False and weights_track_history
            and guarded and shown_quiet_grad_fn is None
            and watched.requires_grad and watched.grad_fn is not None
            and shown_watched_grad_fn == "SoftmaxBackward0",
        "the model plus its whole loop in 24 lines of code, 20x shorter than the ~500 by hand,"
        " printed in that order":
            torch_lines == 24 and shown_shrink == 20 and HAND_WRITTEN_LINES == 500
            and shrink_line == ("counted from this file: the model + its whole training loop = 24"
                                " lines of code -> about 20x less typing for the same network"),
        "an untrained model no better than guessing: 7.5% right, no single guess above 0.25,"
        " against a 1-in-10 chance":
            abs(untrained_accuracy - 0.07520) < 0.002 and top_probability < 0.25
            and shown_chance == 0.1,
        "nn.CrossEntropyLoss to eat RAW SCORES — landing on the same 2.32391 as day 4's -log(p)"
        " applied to softmax output, and as epoch 0 of the real loop — while handing it"
        " probabilities instead softmaxes twice: 2.30287, with the top probability squashed from"
        " 0.20852 to 0.11135":
            abs(loss_on_logits - day4_on_probs) < 1e-5
            and abs(loss_on_logits - 2.32391) < 0.002
            and abs(loss_on_logits - good_losses[0]) < 1e-6
            and loss_on_probs < loss_on_logits
            and abs(loss_on_probs - 2.30287) < 0.002
            and abs(top_probability - 0.20852) < 0.002
            and abs(twice_top - 0.11135) < 0.002,
        "loss 2.32391 at epoch 0 — that is within 0.1 of the untrained ln(10) = 2.30259, a printed"
        " gap of 0.0213 — 1.15495 at epoch 24, gradient size 1.4597 -> 0.14857":
            abs(good_losses[0] - 2.32391) < 0.002
            and abs(predicted_start - 2.30259) < 1e-5
            and shown_start_gap == 0.0213 and shown_start_gap < 0.1
            and abs(good_losses[-1] - 1.15495) < 0.002
            and abs(good_grads[0] - 1.4597) < 0.005
            and abs(good_grads[-1] - 0.14857) < 0.002
            # and the prediction-check line as TEXT, so the measured loss and ln(10) cannot trade
            # places and turn the check into a claim about numbers nobody computed
            and check_line == ("prediction check: epoch 0 loss was 2.3239, ln(10) = 2.3026,"
                               " gap 0.0213"),
        "the printed table to show exactly those numbers: epoch 0 at loss 2.3239 / gradient 1.46"
        " and epoch 24 at 1.155 / 0.149, and the summary line to quote 1.46 -> 0.15":
            good_table[0] == (0, 2.3239, 1.46, "") and good_table[-1] == (24, 1.155, 0.149, "")
            and shown_good_first == 1.46 and shown_good_last == 0.15,
        "the loss to fall in every one of the 24 steps, with no '<- went UP' marker printed":
            all(good_losses[i] < good_losses[i - 1] for i in range(1, EPOCHS))
            and good_rises == 0 and all(row[3] == "" for row in good_table),
        # The good table's claim-carrying rows, pinned as the exact TEXT that reached the screen:
        # the row it OPENS on and the row it ENDS on. Those two are what keep the loss and gradient
        # COLUMNS honest — swap them or wrap one in arithmetic and these strings stop matching.
        # What the 23 rows BETWEEN them were there to show is that not one of them wore the
        # "<- went UP" marker, and that is said once, structurally, in the last clause: one
        # sentence about the whole table that a change of field width cannot defeat, where 25
        # verbatim rows said it 25 times and broke on one. The fall itself is asserted, step by
        # step, by the monotonicity clause in the claim above.
        "the with-zero_grad table to open on epoch 0's row, end on epoch 24's, and print no"
        " '<- went UP' marker on any of its 25 rows":
            len(good_lines) == 25
            and good_lines[0] == '  epoch  0  loss 2.3239  gradient size 1.460'
            and good_lines[-1] == '  epoch 24  loss 1.1550  gradient size 0.149'
            and not any('went UP' in line for line in good_lines),
        "autograd to match the hand-derived backward pass within 1e-5": biggest_gap < 1e-5,
        "the hand-written twin to reach 1.28603 after its 8 epochs, printed beside PyTorch's"
        " 2.32391 -> 1.28603 row":
            len(twin_losses) == 8 and abs(twin_losses[-1] - 1.28603) < 0.002
            and shown_twin_row[-1] == 1.28603 and shown_torch_row[0] == 2.32391
            and len(shown_torch_row) == 8 == len(shown_twin_row),
        "an empty .grad slot before backward and 42.0 in it after, at w=2":
            empty_slot is None and filled_slot == 42.0,
        "two backwards with no clearing to give 3.0 then 6.0 (it adds, it does not replace)":
            one_backward == 3.0 and two_backwards == 6.0,
        "without zero_grad: the same epoch 0 as the good run (loss and gradient identical, so"
        " zero_grad really is the only difference), then the loss rising in 10 of 24 steps — one"
        " printed '<- went UP' marker each — loss 1.27627 at epoch 24, and the gradient piling up"
        " 1.4597 -> 3.10903, quoted as 1.46 -> 3.11":
            bug_losses[0] == good_losses[0] and abs(bug_grads[0] - good_grads[0]) < 1e-9
            and rises == 10 and abs(bug_losses[-1] - 1.27627) < 0.002
            and abs(bug_grads[-1] - 3.10903) < 0.002
            and shown_bug_first == 1.46 and shown_bug_last == 3.11,
        # The buggy table's claim-carrying rows as TEXT. Here the MARKER column is the story, so
        # every marked row is pinned — read out of the table by filtering on the marker itself, so
        # the list also says that these are the ONLY marked rows. An inverted marker (UP on the
        # epochs that FELL) turns that list into the 14 other rows, and a deleted one empties it.
        # The row the table opens on and the row it ends on are pinned beside them. The unmarked
        # rows in between carry no claim: "10 of 24 steps rose" is already counted above.
        "the no-zero_grad table to open on epoch 0's row, end on epoch 24's, and wear the"
        " '<- went UP' marker on exactly these 10 rows":
            len(bug_lines) == 25
            and bug_lines[0] == '  epoch  0  loss 2.3239  gradient size 1.460'
            and bug_lines[-1] == '  epoch 24  loss 1.2763  gradient size 3.109'
            and [line for line in bug_lines if 'went UP' in line] == [
                '  epoch  3  loss 1.6196  gradient size 2.068   <- went UP',
                '  epoch  4  loss 1.7832  gradient size 2.254   <- went UP',
                '  epoch  6  loss 1.5579  gradient size 2.292   <- went UP',
                '  epoch  7  loss 1.7247  gradient size 2.428   <- went UP',
                '  epoch 10  loss 1.5858  gradient size 2.538   <- went UP',
                '  epoch 13  loss 1.4461  gradient size 2.755   <- went UP',
                '  epoch 16  loss 1.2513  gradient size 2.867   <- went UP',
                '  epoch 17  loss 1.2770  gradient size 2.922   <- went UP',
                '  epoch 20  loss 1.3143  gradient size 3.118   <- went UP',
                '  epoch 23  loss 1.2946  gradient size 3.104   <- went UP',
            ],
        # The summary contrast, pinned as the two lines the reader compares. 24 is the number of
        # STEPS between 25 epochs, printed from a bound name, and the two runs' columns are frozen
        # in place — so neither the denominator nor the direction of the story can drift.
        "the summary to read '0 of 24' with the gradient shrinking 1.46 -> 0.15 against '10 of 24'"
        " with it piling up 1.46 -> 3.11":
            shown_steps == 24 and contrast_lines == (
                "\nwith zero_grad: loss rose in 0 of 24 steps, gradient size 1.46 -> 0.15"
                " (shrinks, as it should)\n"
                "without it:     loss rose in 10 of 24 steps, gradient size 1.46 -> 3.11"
                " (a growing pile-up)"),
        "the hand-written gate to be STRICT: a hidden unit whose z1 is exactly 0.0 must receive a"
        " dW1 row of exactly [0.0, 0.0], where the (z >= 0) spelling hands it [-1.964, -3.9281]"
        " from the very same forward pass and loss 4.0181 — a difference no random run can show":
            shown_hinge_z1 == [4.0, 0.0] and loose_z1.tolist() == hinge_z1.tolist()
            and shown_hinge_loss == 4.0181 and shown_loose_loss == shown_hinge_loss
            and shown_dead_row == [0.0, 0.0] and shown_live_row == [1.964, 3.9281]
            and shown_loose_dead_row == [-1.964, -3.9281]
            and bool(relu_gate(0.0)) is False and bool(relu_gate(1e-12)) is True
            # and the two lines as TEXT: which row is labelled "live" and which "z = 0" IS the
            # claim, so the labels and the rows are pinned together, not one value at a time
            and hinge_lines == [
                "\nhinge net: z1 = [4.0, 0.0] (loss 4.0181) -> dW1 for the live unit"
                " [1.964, 3.9281], for the z = 0 unit [0.0, 0.0]",
                "the same step with a (z >= 0) gate (loss 4.0181, unchanged) hands that dead unit"
                " [-1.964, -3.9281] instead — same forward, different weights learned: the"
                " one-character change no random run can show you",
            ],
    }
    if all(claims.values()):
        print("\n✅ you got it")
    else:
        for claim, held in claims.items():
            if not held:
                print("\n❌ not yet — expected", claim)
    for claim, held in claims.items():   # these stop the program if a claim is wrong
        assert held, "expected " + claim
