# day-06-training-loop - experiment
#
# Today's artifact: run a REAL training loop for ONE neuron and watch the loss fall.
# The loop has four steps, on repeat:  forward -> loss -> backward -> update.
# We run it three times to FEEL the learning rate:
#   lr = 0.1     -> healthy: loss falls fast, then flattens near zero (converges)
#   lr = 1.5     -> too high: each step overshoots, so the loss blows up (diverges)
#   lr = 0.0005  -> too low : each nudge is a grain of sand, so the loss barely moves (crawls)
#
# One neuron here is just: pred = w*x + b.  We want pred to match a target number.

# numpy holds the loss curve as an array so we can measure it (min, last) cleanly.
import numpy as np


def train_one_neuron(lr, num_iters=50, x=2.0, target=1.0, w=0.0, b=0.0):
    """Run the four-step training loop and return the loss curve, one loss per loop."""
    # We record the loss at the START of each loop, so we can watch the curve later.
    losses = []
    # Repeat the loop num_iters times. Each pass is ONE iteration (one weight update).
    for _ in range(num_iters):
        # Step 1 - FORWARD PASS: the neuron takes its shot (a weighted sum plus bias).
        pred = w * x + b
        # Step 2 - LOSS: measure the miss with one number. Squared error: (guess - truth)^2.
        loss = (pred - target) ** 2
        # Remember this loop's loss so we can plot / inspect the loss curve.
        losses.append(loss)
        # Step 3 - BACKWARD PASS: which way is each weight off? These are the gradients.
        # For loss = (w*x + b - target)^2, calculus gives these two slopes:
        grad_w = 2 * (pred - target) * x  # how the loss changes if we nudge w
        grad_b = 2 * (pred - target)      # how the loss changes if we nudge b
        # Step 4 - UPDATE: nudge each weight a small step DOWNHILL.
        # The gradient points uphill (toward more loss), so the MINUS walks us downhill.
        w = w - lr * grad_w
        b = b - lr * grad_b
    # Return the whole loss curve as a numpy array. curve[0] is the very first loss.
    return np.array(losses)


if __name__ == "__main__":
    # ---- Run 1: a sensible learning rate. The loss should fall to ~0 (converge). ----
    healthy = train_one_neuron(lr=0.1)
    # Print the curve every 10 loops so we can WATCH it slide downhill.
    print("lr = 0.1  (healthy, should converge):")
    for i in range(0, len(healthy), 10):
        print(f"  loop {i:2d}: loss = {healthy[i]:.6f}")
    final_healthy = healthy[-1]
    print(f"  final loss = {final_healthy:.6g}  (lowest along the curve = {healthy.min():.6g})\n")

    # ---- Run 2: learning rate WAY too high. Each step overshoots -> loss blows up. ----
    too_high = train_one_neuron(lr=1.5)
    final_high = too_high[-1]
    print("lr = 1.5  (too high -> diverges / blows up):")
    print(f"  final loss = {final_high:.6g}   <- huge: the too-high failure, live\n")

    # ---- Run 3: learning rate too low. Each nudge is tiny -> loss barely moves. ----
    too_low = train_one_neuron(lr=0.0005)
    final_low = too_low[-1]
    print("lr = 0.0005  (too low -> crawls):")
    print(f"  start loss = {too_low[0]:.6f},  final loss = {final_low:.6f}")
    print(f"  the loss barely moved after 50 loops -> the too-low failure\n")

    # ---- Self-check: assert each run behaved exactly as the lesson says it should. ----
    # 1) Healthy run converges: the final loss should be essentially zero.
    #    (w -> 0.4, b -> 0.2, so pred -> 0.4*2 + 0.2 = 1.0 = target, an exact hit.)
    expected_healthy = 0.0
    # 2) Too-high run diverges: the final loss should be astronomically large.
    # 3) Too-low run crawls: it drops only a little from its starting loss of 1.0.
    try:
        assert abs(final_healthy - expected_healthy) < 1e-6, \
            f"healthy run did not converge: final loss {final_healthy}"
        assert final_high > 1e6, \
            f"too-high run should blow up but final loss was {final_high}"
        assert 0.5 < final_low < 0.9, \
            f"too-low run should barely move but final loss was {final_low}"
        print("✅ you got it")
    except AssertionError as e:
        # If any check fails, say what we expected so it's easy to fix.
        print(f"❌ not yet — expected healthy loss {expected_healthy}, "
              f"too-high loss > 1e6, too-low loss in (0.5, 0.9)")
        raise
