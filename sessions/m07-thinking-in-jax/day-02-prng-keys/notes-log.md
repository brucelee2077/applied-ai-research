# Week 1 · Day 2 — JAX PRNG keys (log)

**Artifact:** [`experiments/week01_jax/mlp_prng.py`](../../experiments/week01_jax/mlp_prng.py) — a tiny `4 → 8 → 2` MLP seeded from split keys. Ran clean, exit 0, output `[-1.8783567 1.8897852]` reproduced bit-for-bit across two separate processes.

**Reflection — what would break if I used one key for both layers instead of splitting?**

The two layers would draw from the *same* random state, so their weights would be correlated (identical wherever the shapes line up) instead of independent. That kills the point of random init — the network starts with duplicated, symmetric structure, which is exactly the symmetry random initialization is supposed to break. Splitting the key once per layer is what guarantees each weight matrix gets its own fresh, independent randomness while keeping the whole thing reproducible from one seed.
