"""Correctness test: Ring Attention must reproduce full attention exactly.

Run with:
  XLA_FLAGS='--xla_force_host_platform_device_count=4' python3 -m pytest test_ring_attention.py -v
or
  XLA_FLAGS='--xla_force_host_platform_device_count=4' python3 test_ring_attention.py
"""

import numpy as np
import jax
import jax.numpy as jnp

from ring_attention import full_attention, ring_attention_pmap


def _random_qkv(seq_len, head_dim, seed):
    rng = np.random.default_rng(seed)
    q = jnp.array(rng.standard_normal((seq_len, head_dim)), dtype=jnp.float32)
    k = jnp.array(rng.standard_normal((seq_len, head_dim)), dtype=jnp.float32)
    v = jnp.array(rng.standard_normal((seq_len, head_dim)), dtype=jnp.float32)
    return q, k, v


def test_ring_matches_full_attention_primary_case():
    """The brief's primary case: seq_len=512, split into 4 blocks of 128, head_dim=64."""
    assert jax.device_count() >= 2, (
        "Need >=2 simulated devices. Re-run with: "
        "XLA_FLAGS='--xla_force_host_platform_device_count=4' python3 test_ring_attention.py"
    )
    seq_len, head_dim, num_devices = 512, 64, 4
    q, k, v = _random_qkv(seq_len, head_dim, seed=0)

    ref = full_attention(q, k, v)
    ring = ring_attention_pmap(q, k, v, num_devices=num_devices)

    assert ref.shape == ring.shape == (seq_len, head_dim)
    assert jnp.allclose(ref, ring, atol=1e-5), (
        f"max abs diff = {float(jnp.max(jnp.abs(ref - ring))):.3e}"
    )


def test_ring_matches_full_attention_different_seed():
    """Same shapes, different random data — the merge rule should not depend on
    the particular values, only on the algebra being exact."""
    seq_len, head_dim, num_devices = 512, 64, 4
    q, k, v = _random_qkv(seq_len, head_dim, seed=42)

    ref = full_attention(q, k, v)
    ring = ring_attention_pmap(q, k, v, num_devices=num_devices)

    assert jnp.allclose(ref, ring, atol=1e-5), (
        f"max abs diff = {float(jnp.max(jnp.abs(ref - ring))):.3e}"
    )


def test_ring_matches_full_attention_two_devices():
    """Fewer, larger blocks (2 devices instead of 4) should still be exact —
    the number of ring steps changes, the math does not."""
    if jax.device_count() < 2:
        return
    num_devices = min(2, jax.device_count())
    seq_len, head_dim = 256, 32
    q, k, v = _random_qkv(seq_len, head_dim, seed=7)

    ref = full_attention(q, k, v)
    ring = ring_attention_pmap(q, k, v, num_devices=num_devices)

    assert jnp.allclose(ref, ring, atol=1e-5), (
        f"max abs diff = {float(jnp.max(jnp.abs(ref - ring))):.3e}"
    )


if __name__ == "__main__":
    print(f"jax.devices(): {jax.devices()}  (count={jax.device_count()})")
    if jax.device_count() < 2:
        raise SystemExit(
            "Need >=2 simulated devices. Re-run with:\n"
            "  XLA_FLAGS='--xla_force_host_platform_device_count=4' python3 test_ring_attention.py"
        )

    tests = [
        test_ring_matches_full_attention_primary_case,
        test_ring_matches_full_attention_different_seed,
        test_ring_matches_full_attention_two_devices,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
