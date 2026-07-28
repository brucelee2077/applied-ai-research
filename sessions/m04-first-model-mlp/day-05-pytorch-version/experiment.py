# day-05-pytorch-version — experiment
#
# Today's big idea in two lines of output:
#   The same 784→128→10 MLP you hand-wrote in numpy fits in a handful of PyTorch lines.
#   Its loss falls the same way, because autograd redoes your hand-derived backward pass.
#
# Parts: (1) stand-in data, (2) the model, (3) the five-line loop, (4) a taste test against a
# hand-written numpy twin, (5) the same loop with zero_grad() deleted.
# Run it:  python3 sessions/m04-first-model-mlp/day-05-pytorch-version/experiment.py

import math                  # math.log(10) is the loss we expect BEFORE any learning
import numpy as np           # numpy makes the stand-in data and the hand-written twin
import torch                 # torch gives us tensors: arrays that remember how they were made
import torch.nn as nn        # nn holds the ready-made pieces: Linear, ReLU, CrossEntropyLoss
import torch.optim as optim  # optim holds SGD, the helper that applies each nudge

torch.set_num_threads(1)                 # one thread keeps every run byte-identical
PIXELS, HIDDEN, CLASSES = 784, 128, 10   # the three sizes the lesson uses
EPOCHS, LR = 25, 0.3                     # rounds of practice, and how big each nudge is


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
        self.fc1 = nn.Linear(PIXELS, HIDDEN)    # replaces hand-made W1 (128, 784) and b1
        self.relu = nn.ReLU()                   # the bend: keeps positives, zeroes negatives
        self.fc2 = nn.Linear(HIDDEN, CLASSES)   # replaces hand-made W2 (10, 128) and b2

    def forward(self, x):                       # this IS the hand-written forward pass
        return self.fc2(self.relu(self.fc1(x)))

def fresh_model():
    torch.manual_seed(0)   # same seed each time, so every run below starts from equal weights
    return MLP()

def gradient_size(model):
    # One number for the whole model: how big is the gradient the optimizer will apply?
    # A weight whose .grad slot is still empty counts as 0 — that is a cleared slot, not a crash.
    return math.sqrt(sum(float(p.grad.norm()) ** 2 if p.grad is not None else 0.0
                         for p in model.parameters()))

def run_loop(model, X, y, clear_gradients):
    """The whole training loop. Line 1 is the one everybody forgets, so Part 5 can switch
    it off — that single line is the ONLY difference between the two runs."""
    criterion = nn.CrossEntropyLoss()                 # the built-in loss module
    optimizer = optim.SGD(model.parameters(), lr=LR)  # .parameters() hands over every weight
    losses, grad_sizes = [], []
    for epoch in range(EPOCHS):
        if clear_gradients:
            optimizer.zero_grad()             # 1 clear the piled-up gradients
        out = model(X)                        # 2 forward
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
    written out by hand in numpy. Nothing here uses autograd, so agreement means something."""
    W1 = model.fc1.weight.detach().numpy().copy(); b1 = model.fc1.bias.detach().numpy().copy()
    W2 = model.fc2.weight.detach().numpy().copy(); b2 = model.fc2.bias.detach().numpy().copy()
    X, y, losses = X.numpy(), y.numpy(), []
    # onehot[i] is a row of zeros with a single 1 in the true class — the target we compare to
    rows, onehot = len(y), np.eye(CLASSES, dtype=np.float32)[y]
    for _ in range(epochs):
        Z1 = X @ W1.T + b1                       # nn.Linear, by hand
        A1 = np.maximum(0, Z1)                   # nn.ReLU, by hand
        Z2 = A1 @ W2.T + b2                      # the second nn.Linear
        P = np.exp(Z2 - Z2.max(axis=1, keepdims=True)); P /= P.sum(axis=1, keepdims=True)
        losses.append(float(-np.log(P[np.arange(rows), y]).mean()))   # cross-entropy, by hand
        dZ2 = (P - onehot) / rows                # the chain rule, derived by hand
        dW2, db2, dA1 = dZ2.T @ A1, dZ2.sum(axis=0), dZ2 @ W2
        dZ1 = dA1 * (Z1 > 0)                     # the ReLU gate blocks the negative positions
        dW1, db1 = dZ1.T @ X, dZ1.sum(axis=0)
        W1 -= LR * dW1; b1 -= LR * db1; W2 -= LR * dW2; b2 -= LR * db2   # optimizer.step(), by hand
    return losses


if __name__ == "__main__":
    # --- Part 1: the data (a seeded stand-in, not real MNIST) --------------
    X, y = make_digit_data()
    print("X shape:", tuple(X.shape), " y shape:", tuple(y.shape), " first labels:", y[:8].tolist(),
          "\nX type:", type(X).__name__, "- a numpy array that also remembers how it was made")

    # --- Part 2: the model, and the handle that finds every weight ---------
    model = fresh_model()
    layout = [(name, tuple(p.shape)) for name, p in model.named_parameters()]
    total_numbers = sum(p.numel() for p in model.parameters())
    print("parameters:", layout, "\nnumbers to learn:", total_numbers,
          "= 784*128 + 128 + 128*10 + 10")

    # --- Part 3: the five-line loop, with zero_grad in place ---------------
    # Predict before running: an untrained model spreads its guess evenly over the classes,
    # so the first loss should sit near ln(CLASSES). Computed here, not typed in.
    predicted_start = math.log(CLASSES)
    print(f"\npredict: epoch 0 loss near ln({CLASSES}) = {predicted_start:.4f}, then falling"
          "\nwith zero_grad():")
    good_losses, good_grads = run_loop(model, X, y, clear_gradients=True)

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
        "a first loss within 0.1 of ln(10), the untrained prediction":
            abs(good_losses[0] - predicted_start) < 0.1,
        "loss 2.32391 at epoch 0, 1.15495 at epoch 24, first gradient size 1.4597":
            abs(good_losses[0] - 2.32391) < 0.002 and abs(good_losses[-1] - 1.15495) < 0.002
            and abs(good_grads[0] - 1.4597) < 0.005,
        "the loss to fall in every one of the 24 steps":
            all(good_losses[i] < good_losses[i - 1] for i in range(1, EPOCHS)),
        "autograd to match the hand-derived backward pass within 1e-5": biggest_gap < 1e-5,
        "the hand-written twin to reach 1.28603 after its 8 epochs":
            len(twin_losses) == 8 and abs(twin_losses[-1] - 1.28603) < 0.002,
        "an empty .grad slot before backward and 42.0 in it after, at w=2":
            empty_slot is None and filled_slot == 42.0,
        "two backwards with no clearing to give 3.0 then 6.0 (it adds, it does not replace)":
            one_backward == 3.0 and two_backwards == 6.0,
        "without zero_grad: the loss rising in 5+ steps, the gradient more than doubling":
            rises >= 5 and bug_grads[-1] > 2 * bug_grads[0] and good_grads[-1] < good_grads[0],
    }
    if all(claims.values()):
        print("\n✅ you got it")
    else:
        for claim, held in claims.items():
            if not held:
                print("\n❌ not yet — expected", claim)
    for claim, held in claims.items():   # these stop the program if a claim is wrong
        assert held, "expected " + claim
