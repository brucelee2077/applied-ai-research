"""
Module 4 · Day 5 artifact — Positional Encoding & the Shuffle Problem.

Shows three things you can prove to yourself:
  1. Plain self-attention is permutation-invariant: shuffle the input rows and
     each word gets the SAME output vector (just reordered). The model cannot
     read word order on its own.
  2. Sinusoidal positional encoding gives every position a unique fingerprint
     built from sine and cosine waves at different speeds.
  3. Once you ADD the position signal to the embeddings, two different word
     orders produce different attention outputs — order is recovered.

Run:  python3 m04_positional.py
"""

import numpy as np


# ----------------------------------------------------------------------
# A tiny scaled dot-product self-attention (the M4 Day 3 formula).
# Fixed random projections so results are reproducible across calls.
# ----------------------------------------------------------------------
_RNG = np.random.default_rng(0)
_D_MODEL = 4
_D_K = 4
# One shared set of Q/K/V projection matrices, made once and reused.
_WQ = _RNG.standard_normal((_D_MODEL, _D_K))
_WK = _RNG.standard_normal((_D_MODEL, _D_K))
_WV = _RNG.standard_normal((_D_MODEL, _D_K))


def _softmax(scores):
    # Subtract the row max for numerical stability, then normalize per row.
    shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def self_attention(x):
    """
    x: array of shape (seq_len, d_model) — one row per word.
    Returns: array of shape (seq_len, d_k) — one output row per word.

    This is softmax(Q @ K.T / sqrt(d_k)) @ V with fixed projections.
    Note: no position information enters anywhere.
    """
    q = x @ _WQ
    k = x @ _WK
    v = x @ _WV
    scores = q @ k.T / np.sqrt(_D_K)
    weights = _softmax(scores)
    return weights @ v


def _same_output_set(out_a, out_b):
    # Compare the two outputs as an unordered SET of word rows.
    # We sort the rows so a pure reordering counts as "the same set".
    a = np.array(sorted(out_a.round(6).tolist()))
    b = np.array(sorted(out_b.round(6).tolist()))
    return a.shape == b.shape and np.allclose(a, b)


# ----------------------------------------------------------------------
# [1] The shuffle problem: attention ignores order.
# ----------------------------------------------------------------------
def permutation_invariance_demo():
    # Three word embeddings (seq=3, d_model=4). Pretend these are dog/bites/man.
    x = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # word at position 0
            [0.0, 1.0, 0.0, 0.0],  # word at position 1
            [0.0, 0.0, 1.0, 0.0],  # word at position 2
        ]
    )
    # A shuffle (permutation) of the SAME three word rows — a different order.
    perm = [2, 1, 0]
    x_shuffled = x[perm]

    out_a = self_attention(x)
    out_b = self_attention(x_shuffled)

    print("[1] shuffle problem — plain self-attention (no position signal)")
    print("    output for original order:\n", out_a.round(3))
    print("    output for shuffled order:\n", out_b.round(3))
    same = _same_output_set(out_a, out_b)
    print("    same SET of word-output rows under shuffle:", same)
    print("    -> permutation-invariant: order is invisible to attention.")
    print()


# ----------------------------------------------------------------------
# [2] Sinusoidal positional encoding: a unique fingerprint per position.
# ----------------------------------------------------------------------
def sinusoidal_encoding(max_len, d_model):
    """
    Build position encodings using sine and cosine waves.

    max_len: number of positions to encode.
    d_model: size of each position vector (same as the embedding size).
    Returns: array of shape (max_len, d_model).

    Formula, per position `pos` and pair index `i`:
        PE(pos, 2i)   = sin( pos / 10000 ** (2i / d_model) )
        PE(pos, 2i+1) = cos( pos / 10000 ** (2i / d_model) )
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len).reshape(-1, 1)  # column: [0, 1, 2, ...]
    # div_term sets the wave speed: slow waves for low dims, fast for high dims.
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # even columns: sine
    pe[:, 1::2] = np.cos(position * div_term)  # odd columns: cosine
    return pe


def pe_table_demo():
    pe = sinusoidal_encoding(max_len=6, d_model=8)
    print("[2] sinusoidal PE table — max_len=6, d_model=8")
    for pos, row in enumerate(pe):
        print("    pos", pos, "->", row.round(3))
    # Confirm every row is unique (no two positions share a fingerprint).
    rows = {tuple(r.round(6)) for r in pe}
    print("    all", len(pe), "rows unique:", len(rows) == len(pe))
    print()


# ----------------------------------------------------------------------
# [3] Add the position signal -> order is recovered.
# ----------------------------------------------------------------------
def order_recovered_demo():
    # Same three words, two different orders.
    order_a = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    order_b = order_a[[2, 1, 0]]  # a different word order, same words

    pe = sinusoidal_encoding(max_len=3, d_model=_D_MODEL)

    # Add the position row to each word BEFORE attention.
    out_a = self_attention(order_a + pe)
    out_b = self_attention(order_b + pe)

    print("[3] add position -> attention can read order")
    print("    output for order A (with PE):\n", out_a.round(3))
    print("    output for order B (with PE):\n", out_b.round(3))
    same = _same_output_set(out_a, out_b)
    print("    same SET of word-output rows under shuffle:", same)
    print("    -> different orders now give different outputs. Order recovered.")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Module 4 Day 5 — Positional Encoding & the Shuffle Problem")
    print("=" * 60)
    print()
    permutation_invariance_demo()
    pe_table_demo()
    order_recovered_demo()
    print("Takeaway: plain attention is permutation-invariant; adding a")
    print("per-position sinusoidal signal to the embeddings restores word order.")
