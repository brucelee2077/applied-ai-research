# Ring Attention — from-scratch JAX experiment

Verifies two claims from **Liu, Zaharia, Abbeel — "Ring Attention with Blockwise
Transformers for Near-Infinite Context"** ([arXiv:2310.01889](https://arxiv.org/abs/2310.01889)):
(1) splitting a sequence into blocks and rotating key/value blocks around a
ring of devices with an online-softmax merge reproduces full attention
**exactly** (no approximation), and (2) Ring Attention's per-device memory
footprint (`6*b*c*h`) stays flat as sequence length grows, unlike vanilla
attention (`2*b*h*s^2`, quadratic) or the Blockwise Parallel Transformer
(`2*b*s*h`, linear).

## Files

| File | What it does |
|---|---|
| `ring_attention.py` | From-scratch JAX implementation: blockwise online-softmax attention (`_blockwise_update`) plus the ring rotation (`_ring_attention_single_device`, built with `jax.pmap` + `jax.lax.ppermute`, matching the paper's Algorithm 1). |
| `test_ring_attention.py` | Correctness tests: ring-computed attention output vs. plain single-device full-softmax attention, `jnp.allclose(..., atol=1e-5)`, across three shape/seed configurations. Primary case: `seq_len=512`, `4` blocks of `128`, `head_dim=64`. |
| `memory_formula_benchmark.py` | Numerically reproduces the paper's Table 1 memory formulas for a sweep of sequence lengths, printing a table and asserting Ring Attention's per-device byte count is exactly flat while vanilla/BPT grow. |

## Running it

Ring Attention needs at least 2 devices to be a "ring" at all. On a machine
without multiple physical GPUs, simulate multiple CPU devices with an
`XLA_FLAGS` environment variable before starting Python:

```bash
# from this directory
XLA_FLAGS="--xla_force_host_platform_device_count=4" python3 ring_attention.py
XLA_FLAGS="--xla_force_host_platform_device_count=4" python3 test_ring_attention.py
XLA_FLAGS="--xla_force_host_platform_device_count=4" python3 -m pytest test_ring_attention.py -v

# the memory-formula benchmark is pure arithmetic — no JAX/devices needed
python3 memory_formula_benchmark.py
```

## What was verified

Running `ring_attention.py` and `test_ring_attention.py` on 4 simulated CPU
devices confirms that Ring Attention's blockwise online-softmax + ring-rotation
output matches plain single-device full-softmax attention within `atol=1e-5`
(observed max absolute difference ≈ `3.6e-07`) across three different shapes and
random seeds — i.e. the ring never changes the answer, only how the computation
is distributed. Running `memory_formula_benchmark.py` confirms the paper's Table 1
memory formulas numerically: as sequence length sweeps from 1,000 to 1,000,000
tokens, vanilla attention memory grows ~1,000,000x (quadratic), BPT grows ~1,000x
(linear), and Ring Attention's per-device memory grows exactly 1.0x — completely
flat, because it depends only on the fixed block size `c`, never on total
sequence length `s`.

See [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) for the original paper
and [haoliuhl/ringattention](https://github.com/haoliuhl/ringattention) for the
authors' reference implementation.
