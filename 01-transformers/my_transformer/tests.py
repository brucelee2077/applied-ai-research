"""
my_transformer/tests.py — Test each component you build.

Usage:
    python -m my_transformer.tests embedding
    python -m my_transformer.tests qkv
    python -m my_transformer.tests attention
    python -m my_transformer.tests all
"""

import sys
import numpy as np

# ── Test helpers ──────────────────────────────────────────────────────────────

def _pass(name: str, detail: str = ""):
    msg = f"  PASS: {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return True


def _fail(name: str, detail: str = ""):
    msg = f"  FAIL: {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return False


# ── Day 1: Embedding ─────────────────────────────────────────────────────────

def test_embedding():
    """Test that embed() returns correct shape."""
    try:
        from my_transformer.embedding import embed
    except ImportError:
        return _fail("embedding", "Cannot import embed from my_transformer.embedding")

    vocab_size, d_model, seq_len = 10, 4, 3
    # Create a simple embedding table
    table = np.random.randn(vocab_size, d_model)
    token_ids = [1, 5, 3]

    result = embed(token_ids, table)
    result = np.array(result)

    if result.shape != (seq_len, d_model):
        return _fail("embedding", f"Expected shape ({seq_len}, {d_model}), got {result.shape}")

    # Check that the right rows were selected
    expected = table[token_ids]
    if not np.allclose(result, expected):
        return _fail("embedding", "Output values don't match expected rows from the table")

    return _pass("embedding", f"Shape {result.shape} correct, values match")


# ── Day 2: Q/K/V Projection ──────────────────────────────────────────────────

def test_qkv():
    """Test that make_qkv() produces Q, K, V of correct shapes."""
    try:
        from my_transformer.attention import make_qkv
    except ImportError:
        return _fail("qkv", "Cannot import make_qkv from my_transformer.attention")

    seq_len, d_model = 3, 4
    x = np.random.randn(seq_len, d_model)
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)

    Q, K, V = make_qkv(x, W_q, W_k, W_v)
    Q, K, V = np.array(Q), np.array(K), np.array(V)

    for name, mat in [("Q", Q), ("K", K), ("V", V)]:
        if mat.shape != (seq_len, d_model):
            return _fail("qkv", f"{name} shape is {mat.shape}, expected ({seq_len}, {d_model})")

    return _pass("qkv", f"Q, K, V all have shape ({seq_len}, {d_model})")


# ── Day 3: Dot Product ───────────────────────────────────────────────────────

def test_dot_product():
    """Test attention_scores() = Q @ K^T."""
    try:
        from my_transformer.attention import attention_scores
    except ImportError:
        return _fail("dot_product", "Cannot import attention_scores from my_transformer.attention")

    seq_len, d_model = 3, 4
    Q = np.random.randn(seq_len, d_model)
    K = np.random.randn(seq_len, d_model)

    scores = np.array(attention_scores(Q, K))
    expected = Q @ K.T

    if scores.shape != (seq_len, seq_len):
        return _fail("dot_product", f"Shape is {scores.shape}, expected ({seq_len}, {seq_len})")

    if not np.allclose(scores, expected):
        return _fail("dot_product", "Values don't match Q @ K^T")

    return _pass("dot_product", f"Shape ({seq_len}, {seq_len}), values match Q @ K^T")


# ── Day 4: Scaling ────────────────────────────────────────────────────────────

def test_scaling():
    """Test scale_scores() divides by sqrt(d_k)."""
    try:
        from my_transformer.attention import scale_scores
    except ImportError:
        return _fail("scaling", "Cannot import scale_scores from my_transformer.attention")

    d_k = 64
    scores = np.ones((3, 3)) * 8.0
    scaled = np.array(scale_scores(scores, d_k))
    expected = 8.0 / np.sqrt(d_k)

    if not np.allclose(scaled, expected):
        return _fail("scaling", f"Expected {expected:.4f}, got {scaled[0,0]:.4f}")

    return _pass("scaling", f"8.0 / sqrt({d_k}) = {expected:.4f}")


# ── Day 5: Softmax ────────────────────────────────────────────────────────────

def test_softmax():
    """Test attention_weights() applies softmax row-wise."""
    try:
        from my_transformer.attention import attention_weights
    except ImportError:
        return _fail("softmax", "Cannot import attention_weights from my_transformer.attention")

    scores = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    weights = np.array(attention_weights(scores))

    # Each row should sum to 1
    row_sums = weights.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        return _fail("softmax", f"Rows don't sum to 1. Got {row_sums}")

    # All values should be positive
    if np.any(weights < 0):
        return _fail("softmax", "Negative weights found")

    return _pass("softmax", f"All rows sum to 1.0, all values positive")


# ── Day 6: Attention Output ──────────────────────────────────────────────────

def test_attention_output():
    """Test attention_output() = weights @ V."""
    try:
        from my_transformer.attention import attention_output
    except ImportError:
        return _fail("attention_output", "Cannot import attention_output from my_transformer.attention")

    seq_len, d_model = 3, 4
    # Uniform weights (each row sums to 1)
    weights = np.ones((seq_len, seq_len)) / seq_len
    V = np.random.randn(seq_len, d_model)

    out = np.array(attention_output(weights, V))
    expected = weights @ V

    if out.shape != (seq_len, d_model):
        return _fail("attention_output", f"Shape is {out.shape}, expected ({seq_len}, {d_model})")

    if not np.allclose(out, expected):
        return _fail("attention_output", "Values don't match weights @ V")

    return _pass("attention_output", f"Shape ({seq_len}, {d_model}), values match weights @ V")


# ── Day 7: Self Attention (Milestone) ─────────────────────────────────────────

def test_self_attention():
    """Test self_attention() end-to-end."""
    try:
        from my_transformer.attention import self_attention
    except ImportError:
        return _fail("self_attention", "Cannot import self_attention from my_transformer.attention")

    seq_len, d_model = 3, 4
    x = np.random.randn(seq_len, d_model)
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)

    out = np.array(self_attention(x, W_q, W_k, W_v))

    if out.shape != (seq_len, d_model):
        return _fail("self_attention", f"Shape is {out.shape}, expected ({seq_len}, {d_model})")

    return _pass("self_attention", f"Full attention pipeline works! Shape ({seq_len}, {d_model})")


# ── Day 9: Split Heads ────────────────────────────────────────────────────────

def test_split_heads():
    """Test split_heads() reshapes correctly."""
    try:
        from my_transformer.multi_head import split_heads
    except ImportError:
        return _fail("split_heads", "Cannot import split_heads from my_transformer.multi_head")

    seq_len, d_model, n_heads = 3, 8, 2
    x = np.random.randn(seq_len, d_model)
    result = np.array(split_heads(x, n_heads))

    expected_shape = (n_heads, seq_len, d_model // n_heads)
    if result.shape != expected_shape:
        return _fail("split_heads", f"Shape is {result.shape}, expected {expected_shape}")

    return _pass("split_heads", f"Shape {expected_shape} correct")


# ── Day 11: Merge Heads ───────────────────────────────────────────────────────

def test_merge_heads():
    """Test merge_heads() reshapes back."""
    try:
        from my_transformer.multi_head import merge_heads
    except ImportError:
        return _fail("merge_heads", "Cannot import merge_heads from my_transformer.multi_head")

    n_heads, seq_len, d_head = 2, 3, 4
    heads = np.random.randn(n_heads, seq_len, d_head)
    result = np.array(merge_heads(heads))

    expected_shape = (seq_len, n_heads * d_head)
    if result.shape != expected_shape:
        return _fail("merge_heads", f"Shape is {result.shape}, expected {expected_shape}")

    return _pass("merge_heads", f"Shape {expected_shape} correct")


# ── Day 14: Positional Encoding ───────────────────────────────────────────────

def test_positional_encoding():
    """Test sinusoidal_encoding() shape and properties."""
    try:
        from my_transformer.positional import sinusoidal_encoding
    except ImportError:
        return _fail("positional_encoding", "Cannot import sinusoidal_encoding from my_transformer.positional")

    max_len, d_model = 10, 8
    pe = np.array(sinusoidal_encoding(max_len, d_model))

    if pe.shape != (max_len, d_model):
        return _fail("positional_encoding", f"Shape is {pe.shape}, expected ({max_len}, {d_model})")

    # Each position should be different
    if np.allclose(pe[0], pe[1]):
        return _fail("positional_encoding", "Position 0 and 1 are identical — they should differ")

    return _pass("positional_encoding", f"Shape ({max_len}, {d_model}), positions are unique")


# ── Day 18: Residual Connection ───────────────────────────────────────────────

def test_residual():
    """Test residual() = x + sublayer(x)."""
    try:
        from my_transformer.transformer import residual
    except ImportError:
        return _fail("residual", "Cannot import residual from my_transformer.transformer")

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    sublayer_out = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = np.array(residual(x, sublayer_out))
    expected = x + sublayer_out

    if not np.allclose(result, expected):
        return _fail("residual", "Result doesn't match x + sublayer_out")

    return _pass("residual", "x + sublayer(x) works")


# ── Day 19: Layer Norm ────────────────────────────────────────────────────────

def test_layer_norm():
    """Test layer_norm() normalizes each row."""
    try:
        from my_transformer.transformer import layer_norm
    except ImportError:
        return _fail("layer_norm", "Cannot import layer_norm from my_transformer.transformer")

    x = np.array([[10.0, 20.0, 30.0, 40.0]])
    result = np.array(layer_norm(x))

    # After normalization, mean should be ~0 and std ~1
    mean = result.mean()
    std = result.std()

    if abs(mean) > 0.1:
        return _fail("layer_norm", f"Mean is {mean:.4f}, expected ~0")
    if abs(std - 1.0) > 0.2:
        return _fail("layer_norm", f"Std is {std:.4f}, expected ~1")

    return _pass("layer_norm", f"Mean={mean:.4f}, Std={std:.4f}")


# ── Day 20: Feed-Forward ──────────────────────────────────────────────────────

def test_feed_forward():
    """Test feed_forward() shape."""
    try:
        from my_transformer.transformer import feed_forward
    except ImportError:
        return _fail("feed_forward", "Cannot import feed_forward from my_transformer.transformer")

    seq_len, d_model, d_ff = 3, 4, 16
    x = np.random.randn(seq_len, d_model)
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.zeros(d_model)

    result = np.array(feed_forward(x, W1, b1, W2, b2))

    if result.shape != (seq_len, d_model):
        return _fail("feed_forward", f"Shape is {result.shape}, expected ({seq_len}, {d_model})")

    return _pass("feed_forward", f"Shape ({seq_len}, {d_model}) correct")


# ── Day 22: Causal Mask ──────────────────────────────────────────────────────

def test_causal_mask():
    """Test causal_mask() is lower-triangular."""
    try:
        from my_transformer.transformer import causal_mask
    except ImportError:
        return _fail("causal_mask", "Cannot import causal_mask from my_transformer.transformer")

    n = 4
    mask = np.array(causal_mask(n))

    if mask.shape != (n, n):
        return _fail("causal_mask", f"Shape is {mask.shape}, expected ({n}, {n})")

    # Upper triangle should be -inf or very negative
    for i in range(n):
        for j in range(i + 1, n):
            if mask[i, j] > -1e6:
                return _fail("causal_mask", f"mask[{i}][{j}] = {mask[i,j]}, expected -inf")

    # Diagonal and below should be 0
    for i in range(n):
        for j in range(i + 1):
            if abs(mask[i, j]) > 1e-6:
                return _fail("causal_mask", f"mask[{i}][{j}] = {mask[i,j]}, expected 0")

    return _pass("causal_mask", f"Lower-triangular mask, shape ({n}, {n})")


# ── Test runner ───────────────────────────────────────────────────────────────

TESTS = {
    "embedding": test_embedding,
    "qkv": test_qkv,
    "dot_product": test_dot_product,
    "scaling": test_scaling,
    "softmax": test_softmax,
    "attention_output": test_attention_output,
    "self_attention": test_self_attention,
    "attention": test_self_attention,  # alias
    "split_heads": test_split_heads,
    "merge_heads": test_merge_heads,
    "positional_encoding": test_positional_encoding,
    "positional": test_positional_encoding,  # alias
    "residual": test_residual,
    "layer_norm": test_layer_norm,
    "feed_forward": test_feed_forward,
    "causal_mask": test_causal_mask,
}


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python -m my_transformer.tests <component>")
        print(f"\nAvailable: {', '.join(sorted(set(TESTS.keys())))}")
        print("       or: python -m my_transformer.tests all\n")
        return

    target = sys.argv[1].lower()
    print()

    if target == "all":
        passed, failed = 0, 0
        for name, fn in TESTS.items():
            if name in ("attention", "positional"):
                continue  # skip aliases
            if fn():
                passed += 1
            else:
                failed += 1
        print(f"\n  Results: {passed} passed, {failed} failed\n")
    elif target in TESTS:
        TESTS[target]()
        print()
    else:
        print(f"  Unknown test: '{target}'")
        print(f"  Available: {', '.join(sorted(set(TESTS.keys())))}\n")


if __name__ == "__main__":
    main()
