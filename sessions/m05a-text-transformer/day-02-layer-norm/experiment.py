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
# SCOPE, so yesterday's hand-off stays honest: everything below re-levels ONE token's own
# numbers. That is not the same job as holding down the residual lane whose size crept up
# on Day 1. Day 4 wires this function into a block in the Pre-LN order (re-level, run the
# part, add x back) and measures the lane growing anyway; only the Post-LN order, which
# Day 4 presents as the rejected 2017 original, re-levels the lane itself.
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
    # Every number that carries a claim is BOUND to a name, printed from that name, and
    # checked through that same name — one value read twice, not two expressions for one
    # quantity. Corrupt the printed number and the self-check below fails with it.

    # --- Part 2: a drifted token, re-leveled ------------------------------
    x = np.array([10.0, 20.0, 30.0])      # one token, three numbers, drifted high
    mu, sigma = float(x.mean()), float(x.std())      # μ and σ, measured on x alone
    shown_x, shown_x_shape = x, x.shape
    shown_mu, shown_sigma = "%.4f" % mu, "%.4f" % sigma
    print("x            :", shown_x, " shape", shown_x_shape,
          " mu %s  sigma %s" % (shown_mu, shown_sigma))
    ln_out = layer_norm(x, 1.0, 0.0)      # neutral knobs -> the bare re-level
    shown_ln = np.round(ln_out, 4)
    shown_ln_shape = ln_out.shape
    shown_ln_mean = "%.4f" % ln_out.mean()
    shown_ln_spread = "%.4f" % ln_out.std()
    shown_ln_spread_exact = "%.9f" % ln_out.std()
    print("re-leveled   :", shown_ln, " shape", shown_ln_shape,
          " average %s  spread %s (really %s — the hair below 1 is eps)"
          % (shown_ln_mean, shown_ln_spread, shown_ln_spread_exact))

    # --- Part 3: eps barely moves a token with a normal spread ------------
    ln_no_eps = layer_norm(x, 1.0, 0.0, eps=0.0)     # the same re-level, no safety crumb
    eps_shift = float(np.abs(ln_out - ln_no_eps).max())
    shown_no_eps = np.round(ln_no_eps, 4)
    print("\nwithout eps  :", shown_no_eps, " with eps :", shown_ln)
    print("biggest change eps caused :", eps_shift, "-> too small to see")

    # --- Part 4: a degenerate token, where eps earns its keep -------------
    # Every number equal, so the numbers do not spread at all: sigma is exactly 0.
    degenerate = np.full(4, 7.0)
    shown_deg, shown_deg_shape = degenerate, degenerate.shape
    shown_deg_mu = "%.4f" % degenerate.mean()
    shown_deg_sigma = "%.4f" % degenerate.std()
    print("\ndegenerate token :", shown_deg, " shape", shown_deg_shape,
          " mu %s  sigma %s" % (shown_deg_mu, shown_deg_sigma))
    # errstate silences the warning text only — the nan it produces is real.
    with np.errstate(invalid="ignore", divide="ignore"):
        deg_no_eps = layer_norm(degenerate, 1.0, 0.0, eps=0.0)    # 0 / 0
    deg_with_eps = layer_norm(degenerate, 1.0, 0.0)               # 0 / sqrt(0 + eps)
    deg_no_eps_finite = bool(np.isfinite(deg_no_eps).all())
    deg_with_eps_finite = bool(np.isfinite(deg_with_eps).all())
    print("  without eps :", deg_no_eps, "-> all finite?", deg_no_eps_finite)
    print("  with eps    :", deg_with_eps, "-> all finite?", deg_with_eps_finite)
    # That token shows eps stops the 0/0 — but it says nothing about eps's VALUE, because
    # (x - mu) is exactly 0 there, so ANY eps gives the same [0 0 0 0]. A NEARLY flat token
    # is where the value becomes visible: its spread (variance 1.87e-09) is far smaller
    # than eps, so eps — not the data — decides how big the answer comes out.
    nearly_flat = np.array([7.0, 7.0, 7.0, 7.0001])
    flat_no_eps = layer_norm(nearly_flat, 1.0, 0.0, eps=0.0)      # the true spread alone
    flat_default = layer_norm(nearly_flat, 1.0, 0.0)              # eps = 1e-5, the default
    flat_big_eps = layer_norm(nearly_flat, 1.0, 0.0, eps=1e-3)    # 100x more eps
    shown_flat_no_eps = np.round(flat_no_eps, 4)
    shown_flat_default = np.round(flat_default, 4)
    shown_flat_big_eps = np.round(flat_big_eps, 4)
    shown_flat_ratio = round(float(np.abs(flat_default).max() / np.abs(flat_big_eps).max()), 4)
    print("  nearly-flat token", nearly_flat, " with eps=0 :", shown_flat_no_eps)
    print("    eps=1e-5 (the default) :", shown_flat_default,
          "  eps=1e-3 :", shown_flat_big_eps)
    print("    100x the eps shrinks the answer", shown_flat_ratio,
          "x -> on a nearly flat token the eps VALUE sets the size")

    # --- Part 5: the two knobs, and the undo trick ------------------------
    knobbed = layer_norm(x, 2.0, 0.5)     # gain 2 stretches the spread, bias 0.5 slides it
    shown_knobbed = np.round(knobbed, 4)
    print("\ngamma=2, beta=0.5    :", shown_knobbed, "-> spread doubled, centre 0.5")
    # Now set the gain to the ORIGINAL spread and the bias to the ORIGINAL average.
    recovered = layer_norm(x, sigma, mu)
    undo_gap = float(np.abs(recovered - x).max())
    shown_recovered = np.round(recovered, 4)
    print("gamma=sigma, beta=mu :", shown_recovered, " vs x :", shown_x, " gap :", undo_gap)
    # Control: with the neutral knobs the original numbers do NOT come back.
    neutral_gap = float(np.abs(ln_out - x).max())
    shown_neutral_gap = round(neutral_gap, 4)
    print("  same gap with gamma=1, beta=0 :", shown_neutral_gap,
          "-> the knobs are what did the undoing")

    # --- Part 6: RMSNorm, the cousin that skips the centring --------------
    rms_out = rms_norm(x, 1.0)
    ln_rms_gap = float(np.abs(ln_out - rms_out).max())
    shown_rms = np.round(rms_out, 4)
    shown_rms_mean = "%.4f" % rms_out.mean()
    shown_gap = round(ln_rms_gap, 4)
    shown_rms_gain2 = np.round(rms_norm(x, 2.0), 4)
    print("\nlayer_norm :", shown_ln, " average %s" % shown_ln_mean)
    print("rms_norm   :", shown_rms, " average %s" % shown_rms_mean,
          "-> never centred: it only rescales, so each number keeps its own sign")
    print("the two disagree by", shown_gap, "on this token · rms_norm, gamma=2 :",
          shown_rms_gain2)
    # A token whose average is already 0 makes centring a no-op, so the two agree.
    centered = np.array([-3.0, 1.0, 2.0])            # these three add up to 0
    shown_centered = centered
    shown_centered_mean = "%.4f" % centered.mean()
    shown_ln_centered = np.round(layer_norm(centered, 1.0, 0.0), 4)
    shown_rms_centered = np.round(rms_norm(centered, 1.0), 4)
    agree_on_centred = bool(np.array_equal(shown_ln_centered, shown_rms_centered))
    agree_on_drifted = bool(np.array_equal(shown_ln, shown_rms))
    print("already-centred token", shown_centered, " average %s" % shown_centered_mean)
    print("  layer_norm :", shown_ln_centered,
          " rms_norm :", shown_rms_centered, "-> centring was the ONLY gap")
    # "They agree here" holds for EVERY mean-zero token, so on its own it tests nothing —
    # a layer_norm that forgot to subtract mu would pass it too. The drifted token is the
    # half that CAN fail: there the two must come out different.
    print("  equal on this centred token?", agree_on_centred,
          " equal on the drifted [10 20 30]?", agree_on_drifted)

    # --- Part 7: per token, never across the batch ------------------------
    # Two tokens stacked. Row 1 is lopsided on purpose, so a wrong axis would show.
    batch = np.array([[10.0, 20.0, 30.0],
                      [1.0, 2.0, 100.0]])
    batch_out = layer_norm(batch, 1.0, 0.0)
    row_mean_worst = float(np.abs(batch_out.mean(axis=-1)).max())
    same_alone = bool(np.array_equal(batch_out[0], ln_out))
    shown_batch_shape, shown_batch_out_shape = batch.shape, batch_out.shape
    shown_batch0, shown_batch1 = batch[0], batch[1]
    shown_out0 = np.round(batch_out[0], 4)
    shown_out1 = np.round(batch_out[1], 4)
    shown_b1_mu = "%.4f" % batch[1].mean()
    shown_b1_sigma = "%.4f" % batch[1].std()
    print("\nbatch shape", shown_batch_shape, "-> out shape", shown_batch_out_shape)
    print("token 0 in :", shown_batch0, "  out :", shown_out0)
    print("token 1 in :", shown_batch1, " out :", shown_out1,
          " (its own mu %s, its own sigma %s)" % (shown_b1_mu, shown_b1_sigma))
    print("worst row average :", row_mean_worst, "-> each row was levelled on its own")
    print("token 0 in the batch == token 0 alone :", same_alone, "-> batch-independent")

    # --- Self-check: one boolean per claim --------------------------------
    # Every expected number below was read off a real run and written down, not recomputed.
    level_ok = (shown_mu == '20.0000' and shown_sigma == '8.1650'
                and tuple(shown_x) == (10.0, 20.0, 30.0) and shown_x_shape == (3,)
                and np.array_equal(shown_ln, np.array([-1.2247, 0.0, 1.2247]))
                and shown_ln_shape == (3,))
    # snap_ok: average lands on 0 (allclose beats the 4-digit pin above), spread on 1 minus eps.
    snap_ok = (np.allclose(ln_out.mean(), 0.0) and round(float(ln_out.std()), 10) == 0.999999925
               and shown_ln_mean == '0.0000' and shown_ln_spread == '1.0000'
               and shown_ln_spread_exact == '0.999999925')
    eps_shift_ok = (round(eps_shift * 1e8, 4) == 9.1856   # scaled up so it is easy to pin
                    and np.array_equal(shown_no_eps, np.array([-1.2247, 0.0, 1.2247])))
    eps_rescues_ok = (shown_deg_shape == (4,) and tuple(shown_deg) == (7.0, 7.0, 7.0, 7.0)
                      and shown_deg_mu == '7.0000' and shown_deg_sigma == '0.0000'
                      and bool(np.isnan(deg_no_eps).all())             # 0/0 without eps
                      and deg_no_eps_finite is False
                      and np.array_equal(deg_with_eps, np.zeros(4))    # finite with eps
                      and deg_with_eps_finite is True)
    # eps's VALUE, pinned at the one place it is observable: raise eps 100x on a nearly
    # flat token and the answer comes out ~10x smaller. Change the default and this fails.
    eps_value_ok = (tuple(nearly_flat) == (7.0, 7.0, 7.0, 7.0001)
                    and np.array_equal(shown_flat_no_eps,
                                       np.array([-0.5774, -0.5774, -0.5774, 1.7321]))
                    and np.array_equal(shown_flat_default,
                                       np.array([-0.0079, -0.0079, -0.0079, 0.0237]))
                    and np.array_equal(shown_flat_big_eps,
                                       np.array([-0.0008, -0.0008, -0.0008, 0.0024]))
                    and shown_flat_ratio == 9.9991)
    knobs_ok = np.array_equal(shown_knobbed, np.array([-1.9495, 0.5, 2.9495]))
    undo_ok = (np.allclose(recovered, x) and round(undo_gap * 1e7, 6) == 7.499999
               and np.array_equal(shown_recovered, np.array([10.0, 20.0, 30.0]))
               and shown_neutral_gap == 28.7753)   # neutral knobs do NOT undo
    rms_ok = (np.array_equal(shown_rms, np.array([0.4629, 0.9258, 1.3887]))
              and shown_rms_mean == '0.9258'                     # NOT 0.0000: no centring
              and np.array_equal(shown_rms_gain2, np.array([0.9258, 1.8516, 2.7775])))
    rms_vs_ln_ok = (shown_gap == 1.6877          # differ on x, agree on a centred token
                    and tuple(shown_centered) == (-3.0, 1.0, 2.0)
                    and shown_centered_mean == '0.0000'
                    and np.array_equal(shown_ln_centered, np.array([-1.3887, 0.4629, 0.9258]))
                    and np.array_equal(shown_rms_centered, np.array([-1.3887, 0.4629, 0.9258]))
                    and agree_on_centred is True       # an identity for any mean-zero token
                    and agree_on_drifted is False)     # the half that can actually fail
    per_token_ok = (shown_batch_shape == (2, 3) and shown_batch_out_shape == (2, 3)
                    and same_alone
                    and tuple(shown_batch0) == (10.0, 20.0, 30.0)
                    and tuple(shown_batch1) == (1.0, 2.0, 100.0)
                    and np.array_equal(shown_out0, np.array([-1.2247, 0.0, 1.2247]))
                    and np.array_equal(shown_out1, np.array([-0.7178, -0.6963, 1.4142]))
                    and shown_b1_mu == '34.3333' and shown_b1_sigma == '46.4351'
                    and round(row_mean_worst, 12) == 0.0)   # each row average is 0, not near 0

    if (level_ok and snap_ok and eps_shift_ok and eps_rescues_ok and eps_value_ok
            and knobs_ok and undo_ok and rms_ok and rms_vs_ln_ok and per_token_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected mu 20.0, sigma 8.165, re-leveled [-1.2247 0. 1.2247] "
              "(average 0, spread 0.999999925), eps to shift it 9.1856e-08, the all-7 token nan "
              "without eps and [0 0 0 0] with it, the nearly-flat token [-0.0079 -0.0079 "
              "-0.0079 0.0237] at eps=1e-5 and 9.9991x smaller at eps=1e-3, "
              "gamma=2/beta=0.5 -> [-1.9495 0.5 2.9495], "
              "gamma=sigma/beta=mu back on x within 7.499999e-07 (neutral knobs 28.7753 off), "
              "rms_norm [0.4629 0.9258 1.3887] average 0.9258 not 0, norms 1.6877 apart on x but "
              "equal on [-3 1 2], token 1 -> [-0.7178 -0.6963 1.4142] on its own row")

    assert level_ok, "x = [10, 20, 30]: mu 20.0, sigma 8.165, re-leveled [-1.2247, 0, 1.2247]"
    assert snap_ok, "the re-leveled average sits on 0 and the spread on 0.999999925"
    assert eps_shift_ok, "eps should move a normal-spread token by 9.1856e-08, no more"
    assert eps_rescues_ok, "all-equal numbers give nan without eps and [0 0 0 0] with it"
    assert eps_value_ok, ("on a nearly flat token eps's VALUE decides the answer: eps=1e-5 gives "
                          "[-0.0079, -0.0079, -0.0079, 0.0237], and 100x eps shrinks it 9.9991x")
    assert knobs_ok, "gamma=2, beta=0.5 should give [-1.9495, 0.5, 2.9495]"
    assert undo_ok, "gamma=sigma/beta=mu returns x (off 7.499999e-07); neutral stays 28.7753 off"
    assert rms_ok, "rms_norm(x) should be [0.4629, 0.9258, 1.3887], average 0.9258 not 0"
    assert rms_vs_ln_ok, ("the norms agree exactly on the centred [-3, 1, 2] — an identity for any "
                          "mean-zero token — and MUST differ, by 1.6877, on the drifted x")
    assert per_token_ok, "token 1 -> [-0.7178, -0.6963, 1.4142], and each token levels alone"
