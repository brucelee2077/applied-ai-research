"""Ring Attention — from-scratch JAX implementation.

Reference: Liu, Zaharia, Abbeel — "Ring Attention with Blockwise Transformers
for Near-Infinite Context" (arXiv:2310.01889).

This file implements the two pieces that make Ring Attention work:

1. Blockwise online-softmax attention: a way to compute attention scores
   block by block, maintaining a running max and a running (rescaled) sum,
   so the exact softmax result comes out at the end without ever holding
   the full row of scores in memory at once. This is the same trick
   FlashAttention (Week 13) uses inside one chip's SRAM — here it runs once
   per ring step instead of once per SRAM tile.

2. The ring rotation: N devices, each permanently holding one query block,
   pass their key/value blocks around a ring using `jax.lax.ppermute` (a
   collective communication primitive — the same one the paper's Algorithm 1
   specifies). After N steps, every query block has attended to every
   key/value block exactly once.

Two entry points are provided:
  - `full_attention(q, k, v)`      : plain single-device reference attention.
  - `ring_attention_pmap(q, k, v, num_devices)` : the ring-simulated version,
    built with `jax.pmap` + `jax.lax.ppermute` across `num_devices` simulated
    CPU devices (set XLA_FLAGS=--xla_force_host_platform_device_count=N
    before importing jax to get more than 1 simulated device).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# 1. Plain, single-device reference attention (the "ground truth" to check
#    Ring Attention against). No blocking, no online softmax — just the
#    textbook formula, softmax(Q K^T / sqrt(d)) V.
# ---------------------------------------------------------------------------
def full_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Standard scaled dot-product attention, computed all at once.

    q, k, v: shape (seq_len, head_dim). No batch/head dims for clarity —
    the block/ring logic below generalizes the same way regardless.
    """
    head_dim = q.shape[-1]
    scores = (q @ k.T) / jnp.sqrt(jnp.array(head_dim, dtype=q.dtype))  # (s, s)
    weights = jax.nn.softmax(scores, axis=-1)  # normalize each query's row
    return weights @ v  # (s, head_dim)


# ---------------------------------------------------------------------------
# 2. Blockwise online-softmax update — the core numerical trick.
#    Given a device's current running statistics (m, l, o) for its fixed
#    query block, and a freshly-arrived key/value block, fold that block's
#    contribution in WITHOUT ever seeing the full row of scores.
#
#    m = running max score seen so far (for numerical stability)
#    l = running softmax denominator (sum of exp(score - m))
#    o = running numerator (sum of exp(score - m) * value), same shape as q
#
#    This is exactly the FlashAttention merge rule, verified algebraically
#    exact (not an approximation) in the "PLAYGROUND" demo of the lesson.
# ---------------------------------------------------------------------------
def _blockwise_update(q_block, k_block, v_block, m, l, o):
    """One ring step's local computation for one device.

    q_block: (c, d) — this device's fixed query block, never rotates.
    k_block, v_block: (c, d) — whichever key/value block just arrived.
    m: (c,)      running max score per query row.
    l: (c,)      running softmax denominator per query row.
    o: (c, d)    running weighted-value sum per query row.
    """
    head_dim = q_block.shape[-1]
    # scores between the fixed query block and the block that just arrived
    scores = (q_block @ k_block.T) / jnp.sqrt(jnp.array(head_dim, dtype=q_block.dtype))  # (c, c)

    block_max = jnp.max(scores, axis=-1)          # (c,) max score in this block
    m_new = jnp.maximum(m, block_max)              # updated running max

    # rescale the OLD running sum/output to the new max before adding the
    # new block's contribution — this rescale step is what keeps the
    # running total exact instead of approximate.
    scale_old = jnp.exp(m - m_new)                 # (c,)
    p = jnp.exp(scores - m_new[:, None])           # (c, c) unnormalized weights for this block
    l_new = l * scale_old + jnp.sum(p, axis=-1)    # (c,)
    o_new = o * scale_old[:, None] + p @ v_block   # (c, d)

    return m_new, l_new, o_new


# ---------------------------------------------------------------------------
# 3. The ring step, run identically and concurrently on every device.
#    `axis_name` is the pmap axis representing "which device am I."
#    `jax.lax.ppermute` sends this device's current (k, v) block to its
#    ring-neighbor and receives the next block from the other neighbor —
#    this is the exact collective op the Ring Attention paper specifies.
# ---------------------------------------------------------------------------
def _ring_attention_single_device(q_block, k_block, v_block, *, num_devices, axis_name):
    """Runs on one (simulated) device inside pmap.

    q_block never changes — it belongs to this device for the whole loop.
    k_block/v_block start as this device's own key/value block and get
    replaced by the ring rotation on every step.
    """
    c, head_dim = q_block.shape
    m0 = jnp.full((c,), -jnp.inf, dtype=q_block.dtype)
    l0 = jnp.zeros((c,), dtype=q_block.dtype)
    o0 = jnp.zeros((c, head_dim), dtype=q_block.dtype)

    # ppermute needs an explicit list of (source_index, destination_index)
    # pairs describing the ring: device i sends to device (i+1) % N and
    # therefore receives from device (i-1) % N.
    perm = [(i, (i + 1) % num_devices) for i in range(num_devices)]

    def ring_step(carry, _):
        m, l, o, k_cur, v_cur = carry
        # 1. local compute: fold the currently-held (k_cur, v_cur) block
        #    into this device's running statistics for its fixed q_block.
        m, l, o = _blockwise_update(q_block, k_cur, v_cur, m, l, o)
        # 2. ring rotation: hand off the block we just used, receive the
        #    next block from our other neighbor. In real deployments this
        #    communication overlaps with step 1's compute on the NEXT
        #    iteration; lax.scan here models the data dependency exactly,
        #    which is what correctness requires (the overlap is a wall-clock
        #    scheduling optimization, not a change to the result).
        k_next = lax.ppermute(k_cur, axis_name, perm)
        v_next = lax.ppermute(v_cur, axis_name, perm)
        return (m, l, o, k_next, v_next), None

    init_carry = (m0, l0, o0, k_block, v_block)
    (m_final, l_final, o_final, _, _), _ = lax.scan(
        ring_step, init_carry, xs=None, length=num_devices
    )
    return o_final / l_final[:, None]  # final exact softmax normalization


def ring_attention_pmap(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, num_devices: int) -> jnp.ndarray:
    """Splits (q, k, v) into `num_devices` blocks and runs the ring across
    `num_devices` simulated devices, returning the concatenated output.

    q, k, v: shape (seq_len, head_dim); seq_len must be divisible by num_devices.
    """
    seq_len, head_dim = q.shape
    assert seq_len % num_devices == 0, "seq_len must be divisible by num_devices"
    block_size = seq_len // num_devices

    q_blocks = q.reshape(num_devices, block_size, head_dim)
    k_blocks = k.reshape(num_devices, block_size, head_dim)
    v_blocks = v.reshape(num_devices, block_size, head_dim)

    pmapped = jax.pmap(
        partial(_ring_attention_single_device, num_devices=num_devices, axis_name="ring"),
        axis_name="ring",
    )
    out_blocks = pmapped(q_blocks, k_blocks, v_blocks)  # (num_devices, block_size, head_dim)
    return out_blocks.reshape(seq_len, head_dim)


if __name__ == "__main__":
    import numpy as np

    print(f"jax.devices(): {jax.devices()}  (count={jax.device_count()})")
    if jax.device_count() < 2:
        raise SystemExit(
            "Need >=2 simulated devices. Re-run with, e.g.:\n"
            "  XLA_FLAGS='--xla_force_host_platform_device_count=4' python3 ring_attention.py"
        )

    seq_len, head_dim, num_devices = 512, 64, min(4, jax.device_count())
    rng = np.random.default_rng(0)
    q = jnp.array(rng.standard_normal((seq_len, head_dim)), dtype=jnp.float32)
    k = jnp.array(rng.standard_normal((seq_len, head_dim)), dtype=jnp.float32)
    v = jnp.array(rng.standard_normal((seq_len, head_dim)), dtype=jnp.float32)

    ref = full_attention(q, k, v)
    ring = ring_attention_pmap(q, k, v, num_devices=num_devices)

    max_abs_diff = float(jnp.max(jnp.abs(ref - ring)))
    print(f"seq_len={seq_len} head_dim={head_dim} num_devices={num_devices}")
    print(f"max |full_attention - ring_attention| = {max_abs_diff:.3e}")
    print("MATCH (atol=1e-5)" if max_abs_diff < 1e-5 else "MISMATCH")
