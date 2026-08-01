# day-05-pytorch-version — experiment
#
# Today's big idea in two lines of output:
#   The same 784→128→10 MLP you hand-wrote in numpy fits in a handful of PyTorch lines.
#   Its loss falls the same way, because autograd redoes your hand-derived backward pass.
#
# Parts: (1) stand-in data, (2) the model and how few lines it takes, (3) the five-line loop,
# (4) a taste test against a hand-written numpy twin, (5) the same loop with zero_grad() deleted.
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
        rose = "   <- went UP" if epoch and losses[-1] > losses[-2] else ""
        print(f"  epoch {epoch:2d}  loss {losses[-1]:.4f}  gradient size {grad_sizes[-1]:.3f}{rose}")
    return losses, grad_sizes

def numpy_twin_losses(model, X, y, epochs):
    """The day-2 / day-4 version: same starting weights, but forward, backward and update
    written out by hand in numpy. Nothing here uses autograd, so agreement means something.
    The names are days 1-4's: z1 for the pre-activation, hidden for the bent values, out for
    the raw scores, d_out for day 2's seed delta. W1/W2 come out of nn.Linear, so they are
    stored (out, in) — the transpose of days 1-3 — which is why every matmul takes a .T."""
    W1 = model.fc1.weight.detach().numpy().copy(); b1 = model.fc1.bias.detach().numpy().copy()
    W2 = model.fc2.weight.detach().numpy().copy(); b2 = model.fc2.bias.detach().numpy().copy()
    X, y, losses = X.numpy(), y.numpy(), []
    # onehot[i] is a row of zeros with a single 1 in the true class — the target we compare to
    rows, onehot = len(y), np.eye(CLASSES, dtype=np.float32)[y]
    for _ in range(epochs):
        z1 = X @ W1.T + b1                       # nn.Linear, by hand
        hidden = np.maximum(0, z1)               # nn.ReLU, by hand
        out = hidden @ W2.T + b2                 # the second nn.Linear -> day 1's logits
        P = np.exp(out - out.max(axis=1, keepdims=True)); P /= P.sum(axis=1, keepdims=True)
        losses.append(float(-np.log(P[np.arange(rows), y]).mean()))   # cross-entropy, by hand
        d_out = (P - onehot) / rows              # the chain rule, derived by hand
        dW2, db2, d_hidden = d_out.T @ hidden, d_out.sum(axis=0), d_out @ W2
        d_z1 = d_hidden * (z1 > 0)               # the ReLU gate blocks the negative positions
        dW1, db1 = d_z1.T @ X, d_z1.sum(axis=0)
        W1 -= LR * dW1; b1 -= LR * db1; W2 -= LR * dW2; b2 -= LR * db2   # optimizer.step(), by hand
    return losses


if __name__ == "__main__":
    # --- Part 1: the data (a seeded stand-in, not real MNIST) --------------
    X, y = make_digit_data()
    print("X shape:", tuple(X.shape), " y shape:", tuple(y.shape), " first labels:", y[:8].tolist())
    print("X type:", type(X).__name__, "- the same grid of numbers a numpy array holds, plus the"
          " power to record how it was computed, so gradients can flow back")
    print("this X is plain data, so there is nothing to record:", f"X.requires_grad = {X.requires_grad}",
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
    print("counted from this file: the model + its whole training loop =", torch_lines,
          "lines of code", f"-> about {HAND_WRITTEN_LINES // torch_lines}x less typing"
          " for the same network")

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
    print(f"\ninside torch.no_grad() that forward recorded nothing: grad_fn {start_probs.grad_fn};"
          f" the same line outside records one: grad_fn {type(watched.grad_fn).__name__}"
          " - recording is a switch you control, not something a tensor always does")
    print(f"\nuntrained model: {untrained_accuracy:.1%} of labels right, and its most confident"
          f" single guess is only {top_probability:.3f} - flat guessing, chance is {1 / CLASSES:.1%}"
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
    print(f"prediction check: epoch 0 loss was {good_losses[0]:.4f}, ln({CLASSES}) ="
          f" {predicted_start:.4f}, gap {abs(good_losses[0] - predicted_start):.4f}")

    # --- Part 4: the taste test against the hand-written twin --------------
    twin_epochs = 8
    twin_losses = numpy_twin_losses(fresh_model(), X, y, twin_epochs)
    biggest_gap = max(abs(a - b) for a, b in zip(good_losses[:twin_epochs], twin_losses))
    print("\nPyTorch:", [round(v, 5) for v in good_losses[:twin_epochs]],
          "\nby hand:", [round(v, 5) for v in twin_losses],
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
    rises = sum(1 for i in range(1, EPOCHS) if bug_losses[i] > bug_losses[i - 1])
    print(f"\nwith zero_grad: loss rose in 0 of {EPOCHS - 1} steps, gradient size"
          f" {good_grads[0]:.2f} -> {good_grads[-1]:.2f} (shrinks, as it should)\n"
          f"without it:     loss rose in {rises} of {EPOCHS - 1} steps, gradient size"
          f" {bug_grads[0]:.2f} -> {bug_grads[-1]:.2f} (a growing pile-up)")

    # --- Self-check: one boolean per claim ---------------------------------
    # The pinned numbers below were WRITTEN DOWN after running this file, so they do not come
    # from the code they test. Break the model or the loop and they stop matching.
    want_layout = [("fc1.weight", (128, 784)), ("fc1.bias", (128,)),
                   ("fc2.weight", (10, 128)), ("fc2.bias", (10,))]
    claims = {
        "the 784->128->10 layout and 101770 numbers in total":
            layout == want_layout and total_numbers == 101770,
        "plain data with requires_grad False, every weight with requires_grad True, and the"
        " no_grad block recording nothing while the same forward outside it records a grad_fn":
            X.requires_grad is False and weights_track_history
            and guarded and watched.requires_grad and watched.grad_fn is not None,
        "the model plus its whole loop in 24 lines of code, 20x shorter than the ~500 by hand":
            torch_lines == 24 and HAND_WRITTEN_LINES // torch_lines == 20,
        "an untrained model no better than guessing: 7.5% right, no single guess above 0.25":
            abs(untrained_accuracy - 0.07520) < 0.002 and top_probability < 0.25,
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
        "loss 2.32391 at epoch 0 — that is within 0.1 of the untrained ln(10) = 2.30259 —"
        " 1.15495 at epoch 24, gradient size 1.4597 -> 0.14857":
            abs(good_losses[0] - 2.32391) < 0.002
            and abs(predicted_start - 2.30259) < 1e-5
            and abs(good_losses[-1] - 1.15495) < 0.002
            and abs(good_grads[0] - 1.4597) < 0.005
            and abs(good_grads[-1] - 0.14857) < 0.002,
        "the loss to fall in every one of the 24 steps":
            all(good_losses[i] < good_losses[i - 1] for i in range(1, EPOCHS)),
        "autograd to match the hand-derived backward pass within 1e-5": biggest_gap < 1e-5,
        "the hand-written twin to reach 1.28603 after its 8 epochs":
            len(twin_losses) == 8 and abs(twin_losses[-1] - 1.28603) < 0.002,
        "an empty .grad slot before backward and 42.0 in it after, at w=2":
            empty_slot is None and filled_slot == 42.0,
        "two backwards with no clearing to give 3.0 then 6.0 (it adds, it does not replace)":
            one_backward == 3.0 and two_backwards == 6.0,
        "without zero_grad: the same epoch 0 as the good run (loss and gradient identical, so"
        " zero_grad really is the only difference), then the loss rising in 10 of 24 steps, loss"
        " 1.27627 at epoch 24, and the gradient piling up 1.4597 -> 3.10903":
            bug_losses[0] == good_losses[0] and abs(bug_grads[0] - good_grads[0]) < 1e-9
            and rises == 10 and abs(bug_losses[-1] - 1.27627) < 0.002
            and abs(bug_grads[-1] - 3.10903) < 0.002,
    }
    if all(claims.values()):
        print("\n✅ you got it")
    else:
        for claim, held in claims.items():
            if not held:
                print("\n❌ not yet — expected", claim)
    for claim, held in claims.items():   # these stop the program if a claim is wrong
        assert held, "expected " + claim
