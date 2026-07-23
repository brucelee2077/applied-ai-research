# day-04-multihead — experiment
#
# GOAL (from the lesson's Produce step):
#   Build a tiny TWO-HEAD attention panel by hand and WATCH the widths split
#   and rejoin. With input width d_model = 4 and h = 2 heads:
#     (a) each head answers in width d_k = d_model / h = 2
#     (b) joining the two head answers gives width 2 + 2 = 4
#     (c) after the output matrix W_O the width is back to 4 — one vector per word.
#
# This is the exact move inside every real transformer: split into a panel,
# let each head work on its own thin slice, join the answers, mix with W_O.
#
# Run it:  python3 sessions/m03-attention/day-04-multihead/experiment.py

import numpy as np  # numpy does the matrix math for us


# ---------------------------------------------------------------------------
# Step 0 — yesterday's attention, unchanged. A head is just this on a slice.
# ---------------------------------------------------------------------------
def softmax(scores):
    # Turn a row of raw scores into shares that add up to 1.
    # Subtract the row max first so exp() never blows up (a safety trick).
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def attention(Q, K, V):
    # One attention head = score every Query against every Key,
    # softmax the scores into blend weights, then blend the Values.
    d_k = Q.shape[-1]                       # the per-head width
    scores = Q @ K.T / np.sqrt(d_k)         # (seq, seq): how much each word listens to each
    weights = softmax(scores)               # (seq, seq): rows sum to 1
    return weights @ V                      # (seq, d_k): the blended answer per word


# ---------------------------------------------------------------------------
# Step 1 — one head: project the input into Q, K, V of width d_k, then attend.
# ---------------------------------------------------------------------------
def run_head(X, W_Q, W_K, W_V):
    # X is (seq, d_model). Each projection matrix is (d_model, d_k),
    # so it turns each full-width word into a thin d_k-wide Query/Key/Value.
    Q = X @ W_Q                             # (seq, d_k)
    K = X @ W_K                             # (seq, d_k)
    V = X @ W_V                             # (seq, d_k)
    return attention(Q, K, V)               # (seq, d_k): this head's answer


# ---------------------------------------------------------------------------
# Step 2 — the full two-head panel: split -> attend -> join -> mix with W_O.
# ---------------------------------------------------------------------------
def multi_head(X, heads, W_O):
    # heads is a list of (W_Q, W_K, W_V) tuples — one per head, each OWNING
    # its own recipes so each head can chase a different relationship.
    head_outputs = [run_head(X, wq, wk, wv) for (wq, wk, wv) in heads]
    # Join the head answers side by side into one long vector per word.
    joined = np.concatenate(head_outputs, axis=-1)   # (seq, h * d_k) = (seq, d_model)
    # Mix the joined answers with the output matrix W_O so heads can share findings.
    final = joined @ W_O                             # (seq, d_model)
    return head_outputs, joined, final


if __name__ == "__main__":
    # -- The tiny setup the lesson asks for --------------------------------
    seq = 3          # three words in our toy sentence
    d_model = 4      # the full width of each word's vector
    h = 2            # two heads in the panel
    d_k = d_model // h   # each head's slice: 4 / 2 = 2

    # Fixed random numbers so the run is repeatable.
    rng = np.random.default_rng(0)

    # Our toy input: 3 words, each a width-4 vector.
    X = rng.standard_normal((seq, d_model))

    # Two heads, each with its OWN Q/K/V recipes (shape d_model x d_k = 4 x 2).
    heads = [
        (rng.standard_normal((d_model, d_k)),   # head 1's W_Q
         rng.standard_normal((d_model, d_k)),   # head 1's W_K
         rng.standard_normal((d_model, d_k))),  # head 1's W_V
        (rng.standard_normal((d_model, d_k)),   # head 2's W_Q
         rng.standard_normal((d_model, d_k)),   # head 2's W_K
         rng.standard_normal((d_model, d_k))),  # head 2's W_V
    ]

    # The output matrix W_O mixes the joined heads: width d_model in, d_model out.
    W_O = rng.standard_normal((d_model, d_model))

    # -- Run the panel -----------------------------------------------------
    head_outputs, joined, final = multi_head(X, heads, W_O)

    # -- Print the key numbers: the widths splitting and rejoining ---------
    print("input X shape       :", X.shape, "  (seq, d_model)")
    print("head 1 answer shape :", head_outputs[0].shape, "  (seq, d_k) — half the model width")
    print("head 2 answer shape :", head_outputs[1].shape, "  (seq, d_k) — half the model width")
    print("joined shape        :", joined.shape, "  (seq, d_model) — back to full width")
    print("final (after W_O)   :", final.shape, "  (seq, d_model) — one vector per word")

    # -- Self-check: the shapes must match the lesson's stated answers -----
    # (a) each head answers in d_k = 2
    expected_head = (seq, d_k)          # (3, 2)
    # (b) joining gives width d_model = 4
    expected_joined = (seq, d_model)    # (3, 4)
    # (c) after W_O the width is still d_model = 4
    expected_final = (seq, d_model)     # (3, 4)

    ok = (head_outputs[0].shape == expected_head
          and head_outputs[1].shape == expected_head
          and joined.shape == expected_joined
          and final.shape == expected_final)

    if ok:
        assert head_outputs[0].shape == expected_head
        assert head_outputs[1].shape == expected_head
        assert joined.shape == expected_joined
        assert final.shape == expected_final
        print("✅ you got it — 2 heads of width 2 → join to 4 → W_O keeps it 4")
    else:
        print("❌ not yet — expected head", expected_head,
              "joined", expected_joined, "final", expected_final)
        raise SystemExit(1)
