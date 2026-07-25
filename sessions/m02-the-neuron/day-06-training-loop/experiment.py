# day-06-training-loop - experiment
#
# Today's artifact: run a REAL training loop for ONE neuron and watch the loss fall.
# The loop has four steps, on repeat:  forward -> loss -> backward -> update.
# We run it three times to FEEL the learning rate:
#   lr = 0.01     -> healthy: loss falls fast, then glides, then flattens near zero (converges)
#   lr = 1.5      -> too high: each step overshoots, so the PREDICTION flips sign every loop
#                              (0, 15, -195, 2745, ...) while the loss explodes (diverges)
#   lr = 0.00005  -> too low : each nudge is a grain of sand, so the loss barely moves (crawls)
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
    # ---- Run 1: a sensible learning rate. The loss should slide down toward 0. ----
    # lr = 0.01 leaves a real CURVE to watch: 1.0 -> 0.12 -> 0.015 -> 0.0018 -> 0.0002.
    healthy = train_one_neuron(lr=0.01)
    # Print the curve every 10 loops so we can WATCH it slide downhill.
    print("lr = 0.01  (healthy, should converge):")
    for i in range(0, len(healthy), 10):
        print(f"  loop {i:2d}: loss = {healthy[i]:.6f}")
    final_healthy = healthy[-1]
    print(f"  final loss = {final_healthy:.6g}  (started at {healthy[0]:.6g})\n")

    # ---- Run 2: learning rate WAY too high. Each step overshoots -> loss explodes. ----
    too_high = train_one_neuron(lr=1.5)
    final_high = too_high[-1]
    print("lr = 1.5  (too high -> diverges / blows up):")
    print(f"  final loss = {final_high:.6g}   <- astronomical: the too-high failure, live")
    print("  (the error flips sign every loop and grows -> keep looping and it becomes inf / NaN)\n")

    # ---- Run 3: learning rate too low. Each nudge is tiny -> loss barely moves. ----
    too_low = train_one_neuron(lr=0.00005)
    final_low = too_low[-1]
    print("lr = 0.00005  (too low -> crawls):")
    print(f"  start loss = {too_low[0]:.6f},  final loss = {final_low:.6f}")
    print("  the loss barely moved after 50 loops -> the too-low failure\n")

    # ---- Self-check: assert each run behaved exactly as the lesson says it should. ----
    # 1) Healthy run converges: the loss should fall by a factor of ~1000+ but NOT be an
    #    instant one-step hit -- we want a real curve with a steep drop then a flat tail.
    # 2) Too-high run diverges: the final loss should be astronomically large.
    # 3) Too-low run crawls: after 50 loops it has barely left its starting loss of 1.0.
    try:
        assert final_healthy < 1e-3, \
            f"healthy run did not converge: final loss {final_healthy}"
        assert healthy[10] > 1e-3 and healthy[10] > healthy[20] > healthy[30] > 0, \
            "healthy run should show a real falling curve, not a one-step jump to zero"
        assert final_high > 1e6, \
            f"too-high run should blow up but final loss was {final_high}"
        assert 0.9 < final_low < 0.99, \
            f"too-low run should barely move but final loss was {final_low}"
        print("✅ you got it")
    except AssertionError as e:
        # If any check fails, say what we expected so it's easy to fix.
        print("❌ not yet — expected healthy final loss < 1e-3 along a real falling curve, "
              "too-high loss > 1e6, too-low loss in (0.9, 0.99)")
        raise
