# day-06-why-pytorch — experiment
#
# Today's big idea in two lines of output:
#   loss.backward() lands the SAME numbers as the backward pass you derived by hand.
#   A framework is not magic — it is your own numpy gradient, computed for you in one line.
#
# It (1) watches an empty .grad slot fill, (2) checks autograd against a gradient
# hand-derived in numpy, (3) runs the loop zero_grad → forward → loss → backward → step,
# (4) removes zero_grad() on purpose, and (5) switches the graph off with torch.no_grad().
# Run it:  python3 sessions/m04-first-model-mlp/day-06-why-pytorch/experiment.py

import numpy as np           # numpy writes the by-hand backward pass we compare against
import torch                 # a torch tensor is a numpy array that remembers its own steps
import torch.nn as nn        # nn holds the ready-made pieces: nn.Linear, nn.MSELoss
import torch.optim as optim  # optim holds SGD, the update rule w := w - lr * grad

torch.set_num_threads(1)     # one thread keeps every run byte-identical
ROWS, INPUTS = 8, 4          # the shape the lesson's demo uses: nn.Linear(4, 1)
EPOCHS, LR = 12, 0.1         # rounds of practice, and how big each nudge is


def make_data():
    """A seeded stand-in: no dataset is on this machine and there is no network. The
    target follows a hidden straight-line rule plus noise, so there is something to learn."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(ROWS, INPUTS)).astype(np.float32)
    hidden_rule = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float32)
    Y = X @ hidden_rule + 0.5 + rng.normal(0, 0.1, size=ROWS).astype(np.float32)
    return torch.tensor(X), torch.tensor(Y.reshape(ROWS, 1))

def hand_backward(W, b, X, Y):
    """The day-2 way: forward, mean-squared error, then the chain rule BY HAND.
    W has shape (1, 4) and b has shape (1,) — exactly what nn.Linear(4, 1) keeps."""
    out = X @ W.T + b                  # what nn.Linear does for you
    err = out - Y
    loss = float(np.mean(err ** 2))    # what nn.MSELoss does for you
    d_out = 2.0 * err / err.size       # first chain-rule step: d loss / d out
    dW = d_out.T @ X                   # (1, 4) — the number W.grad should hold
    db = d_out.sum(axis=0)             # (1,)  — the number b.grad should hold
    return loss, dW, db

def numpy_twin(W, b, X, Y, epochs):
    """The same loop with no PyTorch at all: hand gradient, hand update, every epoch."""
    W, b = W.copy(), b.copy()
    losses = []
    for _ in range(epochs):
        loss, dW, db = hand_backward(W, b, X, Y)
        losses.append(loss)
        W -= LR * dW                   # what optimizer.step() does for you...
        b -= LR * db                   # ...for every parameter, in a single call
    return losses

def fresh_model():
    torch.manual_seed(0)               # same seed, so every run starts from one weight set
    return nn.Linear(INPUTS, 1)

def grad_size(model):
    # One number for the whole model: how big is the nudge waiting in the .grad slots?
    return float(sum((p.grad ** 2).sum() for p in model.parameters()) ** 0.5)


if __name__ == "__main__":
    # --- Part 1: an empty .grad slot, then autograd fills it ----------------
    w = torch.tensor([2.0], requires_grad=True)     # flag ON: keep the margin notes
    grad_before = w.grad                            # nothing has been computed yet
    print("w:", w.tolist(), " shape:", tuple(w.shape), " dtype:", w.dtype,
          " device:", w.device, " grad slot before backward:", grad_before)
    # By hand, the slope of (3w+1)^2 is 2*(3w+1)*3, worked out from w itself — a real
    # prediction, not a typed-in answer: change w and the prediction changes with it.
    hand_slope = 2.0 * (3.0 * w.item() + 1.0) * 3.0
    demo_loss = (3 * w + 1) ** 2                    # forward: autograd records each step
    demo_recorded = demo_loss.grad_fn is not None   # the tensor carries a graph
    demo_loss.backward()                            # walk that graph back -> fill w.grad
    print("the loss tensor carries a graph:", demo_recorded,
          f"| hand-derived slope at w={w.item()}: {hand_slope}  autograd w.grad: {w.grad.item()}")

    # --- Part 2: one gradient, two ways (the whole point of today) ----------
    X, Y = make_data()
    model = fresh_model()
    layout = [(name, tuple(p.shape)) for name, p in model.named_parameters()]
    print("\nX shape:", tuple(X.shape), " Y shape:", tuple(Y.shape), " parameters:", layout)
    start_W = model.weight.detach().numpy().copy()  # one starting point for both paths
    start_b = model.bias.detach().numpy().copy()
    hand_loss, hand_dW, hand_db = hand_backward(start_W, start_b, X.numpy(), Y.numpy())
    criterion = nn.MSELoss()
    torch_loss = criterion(model(X), Y)
    torch_loss.backward()
    grad_gap = max(float(np.abs(model.weight.grad.numpy() - hand_dW).max()),
                   float(np.abs(model.bias.grad.numpy() - hand_db).max()))
    print(f"loss  by hand {hand_loss:.6f}  PyTorch {torch_loss.item():.6f}")
    print("dW    by hand", np.round(hand_dW[0], 4), " db by hand", np.round(hand_db, 4))
    print("dW    W.grad ", np.round(model.weight.grad.numpy()[0], 4),
          " db b.grad  ", np.round(model.bias.grad.numpy(), 4),
          f"\n-> biggest gap between hand-derived and autograd: {grad_gap:.2e} (the same numbers)")

    # --- Part 3: the five-line loop, with the drop predicted first ----------
    # One SGD step of size LR lowers a smooth loss by at most LR * (gradient size)^2.
    # We compute that bound from the HAND gradient above, then watch what PyTorch does.
    grad_sq = float((hand_dW ** 2).sum() + (hand_db ** 2).sum())
    predicted_drop = LR * grad_sq
    direction = "down" if predicted_drop > 0 else "up"
    print(f"\npredict: step 1 moves the loss {direction}, by at most {predicted_drop:.4f}")
    model = fresh_model()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    losses, good_grads = [], []
    for epoch in range(EPOCHS):
        optimizer.zero_grad()          # 1 clear the piled-up gradients
        out = model(X)                 # 2 forward
        loss = criterion(out, Y)       # 3 score
        loss.backward()                # 4 retrace the trail -> fills every .grad
        good_grads.append(grad_size(model))
        optimizer.step()               # 5 nudge every parameter at once
        losses.append(loss.item())     # .item() pulls a plain number out of the tensor
        print(f"epoch {epoch:2d}  loss {losses[-1]:8.5f}  gradient size {good_grads[-1]:6.3f}")
    actual_drop = losses[0] - losses[1]
    twin_losses = numpy_twin(start_W, start_b, X.numpy(), Y.numpy(), EPOCHS)
    twin_gap = max(abs(a - b) for a, b in zip(losses, twin_losses))
    print(f"step 1 really moved it down by {actual_drop:.4f}"
          f"  ({actual_drop / predicted_drop:.3f} of the predicted bound)")
    print("PyTorch:", [round(v, 5) for v in losses[:4]], "... | all numpy:",
          [round(v, 5) for v in twin_losses[:4]], f"... | biggest gap {twin_gap:.2e}")

    # --- Part 4: remove zero_grad() on purpose ------------------------------
    # First the plain fact: two backwards with nothing cleared ADD, they do not replace.
    piled = fresh_model()
    criterion(piled(X), Y).backward()
    first_grad = piled.weight.grad.clone()          # the gradient after one backward
    criterion(piled(X), Y).backward()               # a second backward, nothing cleared
    ratios = (piled.weight.grad / first_grad).numpy()[0]
    pile_gap = float(np.abs(ratios - 2.0).max())
    print("\ngrad after 1 backward:", np.round(first_grad.numpy()[0], 3),
          "| after a 2nd backward, divided by the 1st:", np.round(ratios, 4))
    bug_model = fresh_model()
    bug_optimizer = optim.SGD(bug_model.parameters(), lr=LR)
    bug_losses, bug_grads = [], []
    for epoch in range(EPOCHS):
        # optimizer.zero_grad() is MISSING here on purpose — that is the experiment.
        bug_loss = criterion(bug_model(X), Y)
        bug_loss.backward()
        bug_grads.append(grad_size(bug_model))
        bug_optimizer.step()
        bug_losses.append(bug_loss.item())
        rose = epoch > 0 and bug_losses[-1] > bug_losses[-2]
        print(f"no zero_grad  epoch {epoch:2d}  loss {bug_losses[-1]:8.4f}"
              f"  gradient size {bug_grads[-1]:7.3f}{'   <- went UP' if rose else ''}")
    rises = sum(1 for i in range(1, EPOCHS) if bug_losses[i] > bug_losses[i - 1])
    good_rises = sum(1 for i in range(1, EPOCHS) if losses[i] > losses[i - 1])
    print(f"with zero_grad: {good_rises} rises, ends {losses[-1]:.4f}, gradient"
          f" {good_grads[0]:.2f} -> {good_grads[-1]:.2f} (shrinks, as it should)")
    print(f"without it:     {rises} rises, ends {bug_losses[-1]:.4f}, gradient"
          f" {bug_grads[0]:.2f} -> {bug_grads[-1]:.2f} (a pile-up that never settles)")

    # --- Part 5: victory lap — switch the graph off -------------------------
    tracked = model(X)                              # graph on: this output is recorded
    with torch.no_grad():                           # graph off: plain arithmetic only
        untracked = model(X)
    print("\ngraph on: requires_grad", tracked.requires_grad, " grad_fn is None:",
          tracked.grad_fn is None, "| in no_grad: requires_grad", untracked.requires_grad,
          " grad_fn is None:", untracked.grad_fn is None)
    same_numbers = torch.equal(tracked.detach(), untracked)
    print("same numbers either way:", same_numbers, "- no_grad drops the record, not the answer")
    # The graph is freed after the first backward, so a second one must fail. We check the
    # exception TYPE, not its wording, because wording changes between torch versions.
    once = criterion(model(X), Y)
    once.backward()
    try:
        once.backward()
        second_backward = None
    except RuntimeError:
        second_backward = RuntimeError
    print("a 2nd backward() on one loss raised:", getattr(second_backward, "__name__", "nothing"))

    # --- Self-check: one boolean per claim ---------------------------------
    # The expected numbers below were WRITTEN DOWN after running this file, so they are
    # not re-derived from the code they test. Break a computation and they fail. They are
    # quoted to five decimals, so PIN is the tolerance for "matches what we wrote down".
    PIN = 2e-4
    autograd_ok = (grad_before is None and demo_recorded
                   and w.grad.item() == 42.0 and hand_slope == 42.0)
    start_pinned = (layout == [("weight", (1, 4)), ("bias", (1,))]
                    and tuple(X.shape) == (8, 4) and tuple(Y.shape) == (8, 1)
                    and np.allclose(hand_dW[0], [-3.1470, 0.1902, -4.9423, -3.5771], atol=PIN)
                    and abs(float(hand_db[0]) - 0.3394) < PIN
                    and abs(hand_loss - 11.34160) < PIN)
    # Two independent code paths must agree: autograd, and the numpy chain rule by hand.
    mirror_ok = (grad_gap < 1e-6 and abs(hand_loss - torch_loss.item()) < 1e-6
                 and twin_gap < 1e-5)
    # A real prediction that can disagree: too big an LR would overshoot and push loss UP.
    prediction_ok = (direction == ("down" if actual_drop > 0 else "up")
                     and abs(predicted_drop - 4.72771) < PIN
                     and 0.5 * predicted_drop < actual_drop <= predicted_drop * 1.000001)
    loop_ok = (good_rises == 0 and all(losses[i] < losses[i - 1] for i in range(1, EPOCHS))
               and abs(losses[0] - 11.34160) < PIN and abs(losses[-1] - 0.68164) < PIN
               and abs(good_grads[0] - 6.87584) < PIN      # the gradient shrinks as it learns
               and abs(good_grads[-1] - 0.85665) < PIN)
    bug_is_visible = (pile_gap < 1e-6 and rises == 5 and bug_losses[-1] > losses[-1]
                      and abs(bug_losses[-1] - 5.21676) < PIN     # stuck, not learning
                      and abs(bug_grads[-1] - 11.60718) < PIN)    # and piled up, not shrinking
    no_grad_ok = (tracked.grad_fn is not None and tracked.requires_grad and same_numbers
                  and untracked.grad_fn is None and not untracked.requires_grad)
    freed_graph_ok = second_backward is RuntimeError

    if (autograd_ok and start_pinned and mirror_ok and prediction_ok and loop_ok
            and bug_is_visible and no_grad_ok and freed_graph_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected .grad None before backward then 42.0, parameters "
              "[('weight',(1,4)),('bias',(1,))], hand-derived dW == [-3.1470, 0.1902, -4.9423, "
              "-3.5771] and db == 0.3394 matched by autograd within 1e-6, step 1's drop inside "
              "the predicted bound 4.72771, loss 11.34160 -> 0.68164 falling every epoch with "
              "the gradient 6.87584 -> 0.85665, a 2nd backward doubling .grad, the no-zero_grad "
              "run rising 5 times and ending at 5.21676, no grad_fn in no_grad, and a "
              "RuntimeError on backward 2")

    # These asserts stop the program if a claim is wrong.
    assert autograd_ok, ".grad must start as None, then hold 42.0 for (3w+1)^2 at w=2"
    assert start_pinned, "the model shape or the pinned starting gradient is wrong: %r" % hand_dW
    assert mirror_ok, ("autograd must mirror the hand-derived numpy gradient: one-step gap "
                       "%.2e, 12-epoch gap %.2e" % (grad_gap, twin_gap))
    assert prediction_ok, "step 1 must drop the loss by at most LR*|grad|^2 = 4.72771, at least half"
    assert loop_ok, "expected the loss to fall every epoch from 11.34160 to 0.68164"
    assert bug_is_visible, "without zero_grad .grad must double and the loss must rise 5 times"
    assert no_grad_ok, "inside torch.no_grad() the output must carry no grad_fn"
    assert freed_graph_ok, "a second backward on a freed graph must raise RuntimeError"
