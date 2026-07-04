"""
Week 1 Day 2 — JAX PRNG keys: build a tiny MLP's weights from SPLIT keys.

JAX has no hidden global random-number generator. Randomness is an explicit
`key` you hold and pass around by hand. Same key in -> same numbers out. To get
fresh, INDEPENDENT randomness you must SPLIT the key. This script:

  1. starts from one key = jax.random.PRNGKey(0),
  2. splits it into one independent subkey per layer (plus one for the input),
  3. draws W1 (4x8) and W2 (8x2) from those separate keys (biases = zeros),
  4. runs forward(x) = tanh(x @ W1 + b1) @ W2 + b2 in raw jax.numpy, and
  5. runs the whole pipeline twice from the same seed to prove the output is
     bit-for-bit identical -> reproducible.

It also demonstrates the #1 beginner bug: reusing one key gives correlated
(identical) draws, while splitting gives independent ones.

Run:   python3 experiments/week01_jax/mlp_prng.py
Exit:  0 = layers used independent keys AND the forward pass is reproducible.
"""
import sys

import jax
import jax.numpy as jnp
from jax import random

# --- network shape (tiny on purpose) --------------------------------------
IN_DIM = 4     # size of the input vector
HIDDEN = 8     # size of the hidden layer
OUT_DIM = 2    # size of the output vector


def init_params(key):
    """Split ONE key into one independent subkey per weight matrix, then draw."""
    # split(key, 2) deterministically makes two brand-new, independent keys.
    # One key per layer means the two layers never share random values.
    k_w1, k_w2 = random.split(key, 2)

    # each weight matrix is drawn from ITS OWN key
    W1 = random.normal(k_w1, (IN_DIM, HIDDEN))   # shape (4, 8)
    W2 = random.normal(k_w2, (HIDDEN, OUT_DIM))  # shape (8, 2)

    # biases start at zero: no randomness -> they need no key
    b1 = jnp.zeros((HIDDEN,))
    b2 = jnp.zeros((OUT_DIM,))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(params, x):
    """One hidden-layer MLP in raw jax.numpy: tanh(x @ W1 + b1) @ W2 + b2."""
    # (4,) @ (4,8) -> (8,) ; + b1 (8,) ; tanh bends it (the nonlinearity)
    hidden = jnp.tanh(x @ params["W1"] + params["b1"])
    # (8,) @ (8,2) -> (2,) ; + b2 (2,) -> final output
    return hidden @ params["W2"] + params["b2"]


def run_once(seed):
    """Full reproducible pipeline: seed -> key -> split -> init -> input -> forward."""
    # the key IS the whole random state; a seed pins it down completely
    key = random.PRNGKey(seed)
    # split once to get independent keys for (a) the input and (b) the weights
    key_input, key_params = random.split(key, 2)
    params = init_params(key_params)          # weights, each from its own subkey
    x = random.normal(key_input, (IN_DIM,))   # the "fixed random input"
    return x, params, forward(params, x)


def demo_reuse_vs_split():
    """Answer the log question: what breaks if two layers share one key?"""
    key = random.PRNGKey(0)
    # BUG: draw two same-shaped matrices from the SAME key -> bit-for-bit identical
    a = random.normal(key, (IN_DIM, HIDDEN))
    b = random.normal(key, (IN_DIM, HIDDEN))
    same = bool(jnp.array_equal(a, b))
    # FIX: split first -> two independent keys -> different matrices
    k1, k2 = random.split(key, 2)
    c = random.normal(k1, (IN_DIM, HIDDEN))
    d = random.normal(k2, (IN_DIM, HIDDEN))
    diff = not bool(jnp.array_equal(c, d))
    return same, diff


def main():
    print(f"jax {jax.__version__} | devices: {jax.devices()}")
    print(f"MLP shape: in={IN_DIM} -> hidden={HIDDEN} -> out={OUT_DIM}\n")

    # -- run the whole pipeline TWICE from the same seed -------------------
    x1, p1, out1 = run_once(0)
    x2, p2, out2 = run_once(0)

    print("Run #1 (seed=0)")
    print(f"  input x   shape={x1.shape}  values={x1}")
    print(f"  W1 shape={p1['W1'].shape}   W2 shape={p1['W2'].shape}")
    print(f"  output    shape={out1.shape}  values={out1}\n")

    print("Run #2 (seed=0)")
    print(f"  output    shape={out2.shape}  values={out2}\n")

    # reproducibility: same seed -> bit-for-bit identical output
    assert bool(jnp.array_equal(out1, out2)), "expected identical output across runs"
    print("REPRODUCIBLE: run #1 and run #2 gave bit-for-bit identical output.\n")

    # why split? show the reuse bug (correlated) vs the split fix (independent)
    same_key_bug, split_key_fix = demo_reuse_vs_split()
    print("Why split? (the #1 beginner bug)")
    print(f"  same key twice -> identical draws?  {same_key_bug}  (bug: correlated randomness)")
    print(f"  split first    -> different draws?  {split_key_fix}  (fix: independent randomness)\n")
    assert same_key_bug, "expected reusing one key to give identical draws"
    assert split_key_fix, "expected split keys to give different draws"

    print("ALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
