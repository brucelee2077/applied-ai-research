# day-06-why-pytorch — experiment
#
# Today's big idea in two lines of output:
#   loss.backward() lands the SAME numbers as the backward pass you derived by hand.
#   A framework is not magic — it is your own numpy gradient, computed for you in one line.
#
# It (1) watches an empty .grad slot fill, (2) checks autograd against a gradient
# hand-derived in numpy — twice, the second time on a 2-column target so the MSE
# denominator is visible, (3) predicts the exact effect of one step and runs the loop
# zero_grad → forward → loss → backward → step at two step sizes, one of them too big,
# (4) removes zero_grad() on purpose, and (5) switches the graph off with torch.no_grad().
# Run it:  python3 sessions/m04-first-model-mlp/day-06-why-pytorch/experiment.py

import numpy as np           # numpy writes the by-hand backward pass we compare against
import torch                 # a torch tensor holds the same grid of numbers a numpy array does,
                             # and also records how it was computed so gradients can flow back
import torch.nn as nn        # nn holds the ready-made pieces: nn.Linear, nn.MSELoss
import torch.optim as optim  # optim holds SGD, the update rule w := w - lr * grad

torch.set_num_threads(1)     # one thread keeps every run byte-identical
ROWS, INPUTS = 8, 4          # the shape the lesson's demo uses: nn.Linear(4, 1)
EPOCHS, LR = 12, 0.1         # rounds of practice, and how big each nudge is
BIG_LR = 1.0                 # ten times LR: too big on purpose, for the prediction in Part 3


def make_data():
    """A seeded stand-in: no dataset is on this machine and there is no network. Each target
    follows a hidden straight-line rule plus noise, so there is something to learn. We build
    two: the 1-column target the lesson's demo uses, and a 2-column target (two things to
    predict per row) that Part 2 needs to make the MSE denominator visible."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(ROWS, INPUTS)).astype(np.float32)
    hidden_rule = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float32)
    Y = X @ hidden_rule + 0.5 + rng.normal(0, 0.1, size=ROWS).astype(np.float32)
    second_rule = np.array([-1.0, 0.5, 2.0, 1.0], dtype=np.float32)   # the second column's rule
    Y2 = X @ second_rule - 0.25 + rng.normal(0, 0.1, size=ROWS).astype(np.float32)
    wide_Y = np.stack([Y, Y2], axis=1).astype(np.float32)
    return torch.tensor(X), torch.tensor(Y.reshape(ROWS, 1)), torch.tensor(wide_Y)

def hand_backward(W, b, X, Y):
    """Day 2's chain rule, by hand: forward, mean-squared error, then the slopes. The layout is
    PyTorch's, not day 2's: nn.Linear(4, OUT) stores W as (OUT, 4) — the transpose of the
    (in, out) matrices days 1-3 built — so this reads X @ W.T where day 2 read x @ W1.
    MSE averages over every ELEMENT of the error, so the denominator is err.size."""
    out = X @ W.T + b                  # what nn.Linear does for you
    err = out - Y
    loss = float(np.mean(err ** 2))    # what nn.MSELoss does for you
    d_out = 2.0 * err / err.size       # first chain-rule step: d loss / d out
    dW = d_out.T @ X                   # (OUT, 4) — the number W.grad should hold
    db = d_out.sum(axis=0)             # (OUT,)  — the number b.grad should hold
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

def fresh_model(outputs=1):
    torch.manual_seed(0)               # same seed, so every run starts from one weight set
    return nn.Linear(INPUTS, outputs)

def chain_slope(value):
    """By hand: the slope of (3w+1)^2 is 2*(3w+1)*3. Written ONCE here and used at every w
    below, so a number typed in from a previous run cannot pass itself off as a derivation."""
    return 2.0 * (3.0 * value + 1.0) * 3.0

def one_step_terms(X_with_bias, dW, db, n_elements):
    """The two pieces of the exact one-step change (Part 3): the downhill size |g|^2, and the
    overshoot |X_with_bias g|^2 / N. N is the number of ELEMENTS the loss averaged over — the
    same denominator Part 2 makes visible — so pass the wrong count and the prediction misses."""
    stacked = np.concatenate([dW.T, db.reshape(1, -1)], axis=0)   # (INPUTS+1, OUT)
    return (float((stacked ** 2).sum()),
            float(((X_with_bias @ stacked) ** 2).sum()) / n_elements)

def predicted_drop_of(lr, grad_sq, overshoot):
    """How much step 1 lowers the loss at this step size. Negative means it RAISES it."""
    return lr * grad_sq - lr * lr * overshoot

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
    hand_slope = chain_slope(w.item())
    demo_loss = (3 * w + 1) ** 2                    # forward: autograd records each step
    demo_recorded = demo_loss.grad_fn is not None   # the tensor carries a graph
    demo_loss.backward()                            # walk that graph back -> fill w.grad
    print("the loss tensor carries a graph:", demo_recorded,
          f"| hand-derived slope at w={w.item()}: {hand_slope}  autograd w.grad: {w.grad.item()}")
    # The same formula at a second w, to show it really is a formula. At w=-1 the slope is
    # NEGATIVE, so the single number 42 cannot stand in for the derivation.
    w_neg = torch.tensor([-1.0], requires_grad=True)
    ((3 * w_neg + 1) ** 2).backward()
    neg_slope_matches = w_neg.grad.item() == chain_slope(w_neg.item()) == -12.0
    print(f"same formula at w={w_neg.item()}: hand-derived {chain_slope(w_neg.item())}"
          f"  autograd w.grad: {w_neg.grad.item()}  (the slope flipped sign — 42 was no constant)")

    # --- Part 2: one gradient, two ways (the whole point of today) ----------
    X, Y, wide_Y = make_data()
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

    # The check above has one blind spot. With 8 rows and 1 column, "divide by the number of
    # ELEMENTS" and "divide by the number of ROWS" are the same division, so a mismatched
    # denominator would change nothing here and the check could not see it. nn.MSELoss divides
    # by elements, so for THIS loss the row count is simply the wrong number. (Day 3 divided by
    # rows on purpose, because its loss was a different, named one: the squared error summed
    # over the 10 classes first, then averaged over rows. Two conventions, each with its own
    # matching denominator — the bug is mixing them, not either one on its own.) So we run the
    # mirror once more on the 2-column target, where 16 elements and 8 rows differ.
    wide_model = fresh_model(outputs=2)
    wide_W = wide_model.weight.detach().numpy().copy()
    wide_b = wide_model.bias.detach().numpy().copy()
    wide_hand_loss, wide_hand_dW, wide_hand_db = hand_backward(
        wide_W, wide_b, X.numpy(), wide_Y.numpy())
    wide_torch_loss = criterion(wide_model(X), wide_Y)
    wide_torch_loss.backward()
    wide_gap = max(float(np.abs(wide_model.weight.grad.numpy() - wide_hand_dW).max()),
                   float(np.abs(wide_model.bias.grad.numpy() - wide_hand_db).max()))
    # The same chain rule with a denominator that does not match THIS loss (rows, where
    # nn.MSELoss used elements), written out on purpose to show that this target really can tell
    # the two spellings apart.
    wide_err = (X.numpy() @ wide_W.T + wide_b) - wide_Y.numpy()
    rows_dW = (2.0 * wide_err / wide_err.shape[0]).T @ X.numpy()   # rows, not elements
    rows_ratio_off = float(np.abs(rows_dW / wide_hand_dW - 2.0).max())
    rows_gap = float(np.abs(rows_dW - wide_model.weight.grad.numpy()).max())
    print(f"\nwide target shape: {tuple(wide_Y.shape)}  -> {wide_err.size} elements over"
          f" {wide_err.shape[0]} rows, so the denominator now shows")
    print(f"loss  by hand {wide_hand_loss:.6f}  PyTorch {wide_torch_loss.item():.6f}")
    print("dW row 0  by hand", np.round(wide_hand_dW[0], 4), " W.grad",
          np.round(wide_model.weight.grad.numpy()[0], 4), f"| gap {wide_gap:.2e}")
    print("dW row 0  dividing by rows instead", np.round(rows_dW[0], 4),
          f"| {wide_err.size}/{wide_err.shape[0]} = {wide_err.size // wide_err.shape[0]}x too"
          f" big, {rows_gap:.3f} away from autograd")

    # --- Part 3: the five-line loop, with the drop predicted first ----------
    # For a linear model scored by MSE, the effect of ONE step is exactly predictable:
    #   loss(after) - loss(before) = -lr * |g|^2 + (lr^2 / N) * |X_with_bias g|^2
    # g is the hand gradient from Part 2, X_with_bias is X with a column of 1s glued on for
    # the bias, and N is the number of elements MSE averaged over. The first term is what you
    # gain by stepping downhill; the second is the overshoot, and it grows with lr SQUARED. So
    # the sign is a real prediction, not a slogan: a small lr predicts DOWN, a big one predicts
    # UP (day 3's "learning rate too large" bug). We run both and check both.
    X_with_bias = np.concatenate([X.numpy(), np.ones((ROWS, 1), dtype=np.float32)], axis=1)
    grad_sq, overshoot = one_step_terms(X_with_bias, hand_dW, hand_db, Y.numel())
    flip_lr = grad_sq / overshoot        # the step size where downhill and overshoot cancel
    predicted_drop = predicted_drop_of(LR, grad_sq, overshoot)
    big_predicted_drop = predicted_drop_of(BIG_LR, grad_sq, overshoot)
    direction = "down" if predicted_drop > 0 else "up"
    big_direction = "down" if big_predicted_drop > 0 else "up"
    print(f"\npredict: step 1 at lr={LR} moves the loss {direction} by {abs(predicted_drop):.4f};"
          f" at lr={BIG_LR} it moves {big_direction} by {abs(big_predicted_drop):.4f}"
          f"  (the sign flips at lr={flip_lr:.4f})")
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
    # Now the same five lines once, with the step size the prediction says will fail.
    big_model = fresh_model()
    big_optimizer = optim.SGD(big_model.parameters(), lr=BIG_LR)
    big_optimizer.zero_grad()
    big_before = criterion(big_model(X), Y)
    big_before.backward()
    big_optimizer.step()
    big_after = criterion(big_model(X), Y).item()
    big_actual_drop = big_before.item() - big_after
    print(f"step 1 at lr={LR} really moved it {'down' if actual_drop > 0 else 'up'} by"
          f" {abs(actual_drop):.4f} (predicted {predicted_drop:+.4f})")
    print(f"step 1 at lr={BIG_LR} really moved it {'down' if big_actual_drop > 0 else 'up'}:"
          f" {big_before.item():.4f} -> {big_after:.4f} (predicted {big_predicted_drop:+.4f})")
    # That prediction has Part 2's blind spot inside it: N above was 8 elements over 8 rows, so
    # the wrong denominator would have predicted the same number. So run the SAME formula once
    # on the 2-column model, where N is 16 and rows are 8, and let the real step judge both.
    wide_grad_sq, wide_overshoot = one_step_terms(
        X_with_bias, wide_hand_dW, wide_hand_db, wide_err.size)
    _, rows_overshoot = one_step_terms(
        X_with_bias, wide_hand_dW, wide_hand_db, wide_err.shape[0])
    wide_predicted = predicted_drop_of(LR, wide_grad_sq, wide_overshoot)
    rows_predicted = predicted_drop_of(LR, wide_grad_sq, rows_overshoot)
    step_model = fresh_model(outputs=2)               # same seed -> the wide start of Part 2
    step_optimizer = optim.SGD(step_model.parameters(), lr=LR)
    step_optimizer.zero_grad()
    wide_before = criterion(step_model(X), wide_Y)
    wide_before.backward()
    step_optimizer.step()
    wide_actual_drop = wide_before.item() - criterion(step_model(X), wide_Y).item()
    wide_pred_gap = abs(wide_predicted - wide_actual_drop)
    rows_pred_gap = abs(rows_predicted - wide_actual_drop)
    print(f"same formula, 2-column model (N = {wide_err.size} elements, not {wide_err.shape[0]}"
          f" rows): predicted {wide_predicted:+.5f}, step 1 really moved {wide_actual_drop:+.5f}"
          f" | with rows as N it would predict {rows_predicted:+.5f}, {rows_pred_gap:.5f} off")
    twin_losses = numpy_twin(start_W, start_b, X.numpy(), Y.numpy(), EPOCHS)
    twin_gap = max(abs(a - b) for a, b in zip(losses, twin_losses))
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
    bug_losses, bug_grads, marked_up = [], [], 0
    for epoch in range(EPOCHS):
        # optimizer.zero_grad() is MISSING here on purpose — that is the experiment.
        bug_loss = criterion(bug_model(X), Y)
        bug_loss.backward()
        bug_grads.append(grad_size(bug_model))
        bug_optimizer.step()
        bug_losses.append(bug_loss.item())
        rose = epoch > 0 and bug_losses[-1] > bug_losses[-2]
        marked_up += int(rose)         # count the markers we print, so the count is checkable
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
    grad_mode_outside = torch.is_grad_enabled()
    with torch.no_grad():                           # graph off: plain arithmetic only
        untracked = model(X)
        grad_mode_inside = torch.is_grad_enabled()
        # The block switches recording off for EVERYTHING inside it, weights included — which
        # is why a fresh combination of a weight built here carries no history either.
        derived_tracks = (untracked * 2 + model.bias).requires_grad
    print("\ngraph on: requires_grad", tracked.requires_grad, " grad_fn is None:",
          tracked.grad_fn is None, "| in no_grad: requires_grad", untracked.requires_grad,
          " grad_fn is None:", untracked.grad_fn is None)
    print("recording on outside the block:", grad_mode_outside, " inside:", grad_mode_inside,
          "| a weight combined inside still tracks history:", derived_tracks,
          "- detaching one tensor would not have done that")
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
                   and w.grad.item() == 42.0 and hand_slope == 42.0
                   and neg_slope_matches)     # the same formula, checked at a second w
    start_pinned = (layout == [("weight", (1, 4)), ("bias", (1,))]
                    and tuple(X.shape) == (8, 4) and tuple(Y.shape) == (8, 1)
                    and np.allclose(hand_dW[0], [-3.1470, 0.1902, -4.9423, -3.5771], atol=PIN)
                    and abs(float(hand_db[0]) - 0.3394) < PIN
                    and abs(hand_loss - 11.34160) < PIN
                    and abs(wide_hand_loss - 7.72439) < PIN)
    # Two independent code paths must agree: autograd, and the numpy chain rule by hand — on
    # the 1-column target AND on the 2-column one, where a wrong denominator cannot hide.
    mirror_ok = (grad_gap < 1e-6 and abs(hand_loss - torch_loss.item()) < 1e-6
                 and wide_gap < 1e-6 and abs(wide_hand_loss - wide_torch_loss.item()) < 1e-6
                 and twin_gap < 1e-5)
    # And this is what gives that second mirror its teeth: 16 elements over 8 rows, so
    # dividing by rows is exactly 2x wrong and lands a measured distance from autograd.
    reduction_visible = (tuple(wide_Y.shape) == (ROWS, 2) and wide_err.size == 16
                         and wide_err.shape[0] == 8 and rows_ratio_off < 1e-6
                         and abs(rows_gap - 2.50695) < PIN)
    # A real prediction that can disagree: one formula, two step sizes, opposite signs —
    # and each sign is checked against the step SGD actually took. The same formula is then
    # run at a second dial setting (2 output columns) where its N is 16, not 8 = the row
    # count, so the denominator inside the prediction is measured, not assumed.
    prediction_ok = (direction == "down" and big_direction == "up"
                     and abs(predicted_drop - 4.14032) < PIN
                     and abs(big_predicted_drop + 11.46238) < PIN
                     and abs(flip_lr - 0.80486) < PIN
                     and abs(actual_drop - predicted_drop) < PIN
                     and abs(big_actual_drop - big_predicted_drop) < PIN
                     and actual_drop > 0 > big_actual_drop
                     and abs(big_after - 22.80399) < PIN
                     and abs(wide_predicted - 1.52394) < PIN and wide_pred_gap < 1e-5
                     and rows_pred_gap > 0.05)  # rows as N misses the real step by 0.10036
    loop_ok = (good_rises == 0 and all(losses[i] < losses[i - 1] for i in range(1, EPOCHS))
               and abs(losses[0] - 11.34160) < PIN and abs(losses[-1] - 0.68164) < PIN
               and abs(good_grads[0] - 6.87584) < PIN      # the gradient shrinks as it learns
               and abs(good_grads[-1] - 0.85665) < PIN)
    bug_is_visible = (pile_gap < 1e-6 and rises == 5 and marked_up == rises
                      and bug_losses[-1] > losses[-1]
                      and abs(bug_losses[-1] - 5.21676) < PIN     # stuck, not learning
                      and abs(bug_grads[-1] - 11.60718) < PIN)    # and piled up, not shrinking
    no_grad_ok = (tracked.grad_fn is not None and tracked.requires_grad and same_numbers
                  and untracked.grad_fn is None and not untracked.requires_grad
                  and grad_mode_outside and not grad_mode_inside and not derived_tracks)
    freed_graph_ok = second_backward is RuntimeError

    if (autograd_ok and start_pinned and mirror_ok and reduction_visible and prediction_ok
            and loop_ok and bug_is_visible and no_grad_ok and freed_graph_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected .grad None before backward then 42.0 (and -12.0 from the "
              "same formula at w=-1), parameters "
              "[('weight',(1,4)),('bias',(1,))], hand-derived dW == [-3.1470, 0.1902, -4.9423, "
              "-3.5771] and db == 0.3394 matched by autograd within 1e-6, the 2-column mirror "
              "(loss 7.72439, 16 elements over 8 rows) matching too while the same chain rule "
              "divided by rows lands 2.50695 away, step 1 moving DOWN 4.14032 at lr=0.1 and UP "
              "11.46238 at lr=1.0 (the sign flipping at lr=0.80486), the same formula predicting "
              "+1.52394 on the 2-column model where rows-as-N misses by 0.10036, loss 11.34160 "
              "-> 0.68164 falling every epoch with the gradient 6.87584 -> 0.85665, a 2nd "
              "backward doubling .grad, the no-zero_grad run rising 5 times (5 printed markers) "
              "and ending at 5.21676, no grad_fn and no recording inside no_grad, and a "
              "RuntimeError on backward 2")

    # These asserts stop the program if a claim is wrong.
    assert autograd_ok, (".grad must start as None, then hold 42.0 for (3w+1)^2 at w=2, and the "
                         "same formula must give -12.0 at w=-1")
    assert start_pinned, "the model shape or the pinned starting gradient is wrong: %r" % hand_dW
    assert mirror_ok, ("autograd must mirror the hand-derived numpy gradient: 1-column gap "
                       "%.2e, 2-column gap %.2e, 12-epoch gap %.2e"
                       % (grad_gap, wide_gap, twin_gap))
    assert reduction_visible, ("the 2-column target must expose the MSE denominator: dividing "
                               "by rows must be 2x wrong and land 2.50695 from autograd, got "
                               "%.5f" % rows_gap)
    assert prediction_ok, ("the one-step formula must predict DOWN 4.14032 at lr=0.1 and UP "
                           "11.46238 at lr=1.0, match the step SGD really took, and land within "
                           "1e-5 on the 2-column model (%.5f off) where rows-as-N misses by %.5f"
                           % (wide_pred_gap, rows_pred_gap))
    assert loop_ok, "expected the loss to fall every epoch from 11.34160 to 0.68164"
    assert bug_is_visible, ("without zero_grad .grad must double and the loss must rise 5 times, "
                            "each rise printed: %d rises, %d markers" % (rises, marked_up))
    assert no_grad_ok, ("inside torch.no_grad() the output must carry no grad_fn and recording "
                        "must be off for the whole block")
    assert freed_graph_ok, "a second backward on a freed graph must raise RuntimeError"
