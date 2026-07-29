# day-02-layer-norm — experiment
#
# Today's big idea in two lines of output:
#   A token whose numbers drifted to [10, 20, 30] is re-leveled to average 0 and
#   spread 1 — and the two learned knobs γ, β can put the old scale right back.
#
# It also shows why the tiny ε lives inside the square root and what RMSNorm drops.
# Nothing is random (every number is a lesson literal), so two runs match exactly.
# Run it:  python3 sessions/m05a-text-transformer/day-02-layer-norm/experiment.py

import numpy as np  # numpy gives us arrays plus per-axis mean and sqrt


# ---- Part 1: the re-level itself ------------------------------------------
def layer_norm(x, gamma, beta, eps=1e-5):
    """Re-level the LAST axis of x: average -> 0, spread -> 1, then γ and β."""
    # axis=-1 is the whole point: one token's own numbers, never the batch.
    mu = x.mean(axis=-1, keepdims=True)                   # μ, the average
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)     # σ², the spread squared
    norm = (x - mu) / np.sqrt(var + eps)                   # eps sits INSIDE the root
    return gamma * norm + beta                             # γ stretches, β slides


def rms_norm(x, gamma, eps=1e-5):
    """RMSNorm: rescale by the typical size only — no centring, and no β."""
    # Root mean square: square, average, square-root. It never subtracts μ.
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / rms * gamma


if __name__ == "__main__":
    # --- Part 2: a drifted token, re-leveled ------------------------------
    x = np.array([10.0, 20.0, 30.0])      # one token, three numbers, drifted high
    mu, sigma = float(x.mean()), float(x.std())      # μ and σ, measured on x alone
    print("x            :", x, " shape", x.shape, " mu %.4f  sigma %.4f" % (mu, sigma))
    ln_out = layer_norm(x, 1.0, 0.0)      # neutral knobs -> the bare re-level
    print("re-leveled   :", np.round(ln_out, 4), " shape", ln_out.shape,
          " average %.4f  spread %.4f (really %.9f — the hair below 1 is eps)"
          % (ln_out.mean(), ln_out.std(), ln_out.std()))

    # --- Part 3: eps barely moves a token with a normal spread ------------
    ln_no_eps = layer_norm(x, 1.0, 0.0, eps=0.0)     # the same re-level, no safety crumb
    eps_shift = float(np.abs(ln_out - ln_no_eps).max())
    print("\nwithout eps  :", np.round(ln_no_eps, 4), " with eps :", np.round(ln_out, 4))
    print("biggest change eps caused :", eps_shift, "-> too small to see")

    # --- Part 4: a degenerate token, where eps earns its keep -------------
    # Every number equal, so the numbers do not spread at all: sigma is exactly 0.
    degenerate = np.full(4, 7.0)
    print("\ndegenerate token :", degenerate, " shape", degenerate.shape,
          " mu %.4f  sigma %.4f" % (degenerate.mean(), degenerate.std()))
    # errstate silences the warning text only — the nan it produces is real.
    with np.errstate(invalid="ignore", divide="ignore"):
        deg_no_eps = layer_norm(degenerate, 1.0, 0.0, eps=0.0)    # 0 / 0
    deg_with_eps = layer_norm(degenerate, 1.0, 0.0)               # 0 / sqrt(0 + eps)
    print("  without eps :", deg_no_eps, "-> all finite?", bool(np.isfinite(deg_no_eps).all()))
    print("  with eps    :", deg_with_eps, "-> all finite?", bool(np.isfinite(deg_with_eps).all()))

    # --- Part 5: the two knobs, and the undo trick ------------------------
    knobbed = layer_norm(x, 2.0, 0.5)     # gain 2 stretches the spread, bias 0.5 slides it
    print("\ngamma=2, beta=0.5    :", np.round(knobbed, 4), "-> spread doubled, centre 0.5")
    # Now set the gain to the ORIGINAL spread and the bias to the ORIGINAL average.
    recovered = layer_norm(x, sigma, mu)
    undo_gap = float(np.abs(recovered - x).max())
    print("gamma=sigma, beta=mu :", np.round(recovered, 4), " vs x :", x, " gap :", undo_gap)
    # Control: with the neutral knobs the original numbers do NOT come back.
    neutral_gap = float(np.abs(ln_out - x).max())
    print("  same gap with gamma=1, beta=0 :", round(neutral_gap, 4),
          "-> the knobs are what did the undoing")

    # --- Part 6: RMSNorm, the cousin that skips the centring --------------
    rms_out = rms_norm(x, 1.0)
    ln_rms_gap = float(np.abs(ln_out - rms_out).max())
    print("\nlayer_norm :", np.round(ln_out, 4), " average %.4f" % ln_out.mean())
    print("rms_norm   :", np.round(rms_out, 4), " average %.4f" % rms_out.mean(),
          "-> never centred: it only rescales, so each number keeps its own sign")
    print("the two disagree by", round(ln_rms_gap, 4), "on this token · rms_norm, gamma=2 :",
          np.round(rms_norm(x, 2.0), 4))
    # A token whose average is already 0 makes centring a no-op, so the two agree.
    centered = np.array([-3.0, 1.0, 2.0])            # these three add up to 0
    print("already-centred token", centered, " average %.4f" % centered.mean())
    print("  layer_norm :", np.round(layer_norm(centered, 1.0, 0.0), 4),
          " rms_norm :", np.round(rms_norm(centered, 1.0), 4), "-> centring was the ONLY gap")

    # --- Part 7: per token, never across the batch ------------------------
    # Two tokens stacked. Row 1 is lopsided on purpose, so a wrong axis would show.
    batch = np.array([[10.0, 20.0, 30.0],
                      [1.0, 2.0, 100.0]])
    batch_out = layer_norm(batch, 1.0, 0.0)
    row_mean_worst = float(np.abs(batch_out.mean(axis=-1)).max())
    same_alone = bool(np.array_equal(batch_out[0], ln_out))
    print("\nbatch shape", batch.shape, "-> out shape", batch_out.shape)
    print("token 0 in :", batch[0], "  out :", np.round(batch_out[0], 4))
    print("token 1 in :", batch[1], " out :", np.round(batch_out[1], 4),
          " (its own mu %.4f, its own sigma %.4f)" % (batch[1].mean(), batch[1].std()))
    print("worst row average :", row_mean_worst, "-> each row was levelled on its own")
    print("token 0 in the batch == token 0 alone :", same_alone, "-> batch-independent")

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected number below was read off a real run and written down, not recomputed.
    level_ok = (round(mu, 4) == 20.0 and round(sigma, 4) == 8.165
                and np.array_equal(np.round(ln_out, 4), np.array([-1.2247, 0.0, 1.2247])))
    # snap_ok: average lands on 0 (allclose beats the 4-digit pin above), spread on 1 minus eps.
    snap_ok = np.allclose(ln_out.mean(), 0.0) and round(float(ln_out.std()), 10) == 0.999999925
    eps_shift_ok = round(eps_shift * 1e8, 4) == 9.1856   # scaled up so it is easy to pin
    eps_rescues_ok = (degenerate.shape == (4,) and round(float(degenerate.mean()), 4) == 7.0
                      and bool(np.isnan(deg_no_eps).all())             # 0/0 without eps
                      and np.array_equal(deg_with_eps, np.zeros(4)))   # finite with eps
    knobs_ok = np.array_equal(np.round(knobbed, 4), np.array([-1.9495, 0.5, 2.9495]))
    undo_ok = (np.allclose(recovered, x) and round(undo_gap * 1e7, 6) == 7.499999
               and round(neutral_gap, 4) == 28.7753)   # neutral knobs do NOT undo
    rms_ok = (np.array_equal(np.round(rms_out, 4), np.array([0.4629, 0.9258, 1.3887]))
              and round(float(rms_out.mean()), 4) == 0.9258       # NOT 0: no centring
              and np.array_equal(np.round(rms_norm(x, 2.0), 4), [0.9258, 1.8516, 2.7775]))
    rms_vs_ln_ok = (round(ln_rms_gap, 4) == 1.6877   # differ on x, agree on a centred token
                    and np.array_equal(layer_norm(centered, 1.0, 0.0), rms_norm(centered, 1.0)))
    per_token_ok = (batch_out.shape == (2, 3) and same_alone
                    and np.array_equal(np.round(batch_out[1], 4), [-0.7178, -0.6963, 1.4142])
                    and round(row_mean_worst, 12) == 0.0)   # each row average is 0, not near 0

    if (level_ok and snap_ok and eps_shift_ok and eps_rescues_ok
            and knobs_ok and undo_ok and rms_ok and rms_vs_ln_ok and per_token_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected mu 20.0, sigma 8.165, re-leveled [-1.2247 0. 1.2247] "
              "(average 0, spread 0.999999925), eps to shift it 9.1856e-08, the all-7 token nan "
              "without eps and [0 0 0 0] with it, gamma=2/beta=0.5 -> [-1.9495 0.5 2.9495], "
              "gamma=sigma/beta=mu back on x within 7.499999e-07 (neutral knobs 28.7753 off), "
              "rms_norm [0.4629 0.9258 1.3887] average 0.9258 not 0, norms 1.6877 apart on x but "
              "equal on [-3 1 2], token 1 -> [-0.7178 -0.6963 1.4142] on its own row")

    assert level_ok, "x = [10, 20, 30]: mu 20.0, sigma 8.165, re-leveled [-1.2247, 0, 1.2247]"
    assert snap_ok, "the re-leveled average sits on 0 and the spread on 0.999999925"
    assert eps_shift_ok, "eps should move a normal-spread token by 9.1856e-08, no more"
    assert eps_rescues_ok, "all-equal numbers give nan without eps and [0 0 0 0] with it"
    assert knobs_ok, "gamma=2, beta=0.5 should give [-1.9495, 0.5, 2.9495]"
    assert undo_ok, "gamma=sigma/beta=mu returns x (off 7.499999e-07); neutral stays 28.7753 off"
    assert rms_ok, "rms_norm(x) should be [0.4629, 0.9258, 1.3887], average 0.9258 not 0"
    assert rms_vs_ln_ok, "the norms differ by 1.6877 on x and agree exactly on [-3, 1, 2]"
    assert per_token_ok, "token 1 -> [-0.7178, -0.6963, 1.4142], and each token levels alone"
