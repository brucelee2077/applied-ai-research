"""Unit tests for the SnapKV eviction simulator.

Run directly (also works under pytest):

    python experiments/week16_jax/day2_snapkv_eviction/test_snap_kv.py

Exercises the three acceptance criteria from the Week 16 Day 2 lesson:
  (a) output cache size == max_capacity_prompt whenever q_len >= max_capacity_prompt
  (b) the observation window's own K/V survive unmodified in the output
  (c) kernel_size=1 (no pooling) vs kernel_size=5 select measurably different
      index sets on a synthetic vote signal with "spiky" vs "clustered"
      importance regions — confirming the clustering effect from the lesson's
      pooling demo.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from snap_kv import snap_kv, votes_to_indices


def _make_synthetic_qkv(key, batch, num_heads, q_len, head_dim):
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (batch, num_heads, q_len, head_dim))
    k = jax.random.normal(k2, (batch, num_heads, q_len, head_dim))
    v = jax.random.normal(k3, (batch, num_heads, q_len, head_dim))
    return q, k, v


def test_output_cache_size_equals_max_capacity_prompt():
    """(a) Whenever q_len >= max_capacity_prompt, output length == max_capacity_prompt exactly."""
    key = jax.random.PRNGKey(0)
    window_size, max_capacity_prompt, kernel_size = 16, 128, 5
    head_dim = 32

    for q_len in [128, 256, 1024, 4096]:
        assert q_len >= max_capacity_prompt, "test setup invariant"
        q, k, v = _make_synthetic_qkv(key, batch=1, num_heads=4, q_len=q_len, head_dim=head_dim)
        new_k, new_v = snap_kv(q, k, v, window_size, max_capacity_prompt, kernel_size)
        assert new_k.shape[2] == max_capacity_prompt, (
            f"q_len={q_len}: expected cache size {max_capacity_prompt}, got {new_k.shape[2]}"
        )
        assert new_v.shape[2] == max_capacity_prompt, (
            f"q_len={q_len}: expected cache size {max_capacity_prompt}, got {new_v.shape[2]}"
        )
    print("[PASS] test_output_cache_size_equals_max_capacity_prompt")


def test_guard_clause_skips_eviction_when_prompt_fits():
    """Sanity check on the guard clause: q_len < max_capacity_prompt returns the input unchanged."""
    key = jax.random.PRNGKey(1)
    window_size, max_capacity_prompt, kernel_size = 16, 128, 5
    q_len = 64  # < max_capacity_prompt
    q, k, v = _make_synthetic_qkv(key, batch=1, num_heads=2, q_len=q_len, head_dim=16)
    new_k, new_v = snap_kv(q, k, v, window_size, max_capacity_prompt, kernel_size)
    assert new_k.shape[2] == q_len
    assert jnp.array_equal(new_k, k)
    assert jnp.array_equal(new_v, v)
    print("[PASS] test_guard_clause_skips_eviction_when_prompt_fits")


def test_observation_window_kv_survives_unmodified():
    """(b) The observation window's own K/V are always present unmodified in the output."""
    key = jax.random.PRNGKey(2)
    window_size, max_capacity_prompt, kernel_size = 16, 128, 5
    q_len = 2048
    head_dim = 32
    q, k, v = _make_synthetic_qkv(key, batch=2, num_heads=4, q_len=q_len, head_dim=head_dim)

    new_k, new_v = snap_kv(q, k, v, window_size, max_capacity_prompt, kernel_size)

    # snap_kv concatenates [k_past, k_obs] along the sequence axis, so the
    # observation window occupies the LAST window_size positions of the output.
    out_k_obs = new_k[..., -window_size:, :]
    out_v_obs = new_v[..., -window_size:, :]
    expected_k_obs = k[..., -window_size:, :]
    expected_v_obs = v[..., -window_size:, :]

    assert jnp.array_equal(out_k_obs, expected_k_obs), (
        "observation window keys were modified or misplaced by eviction"
    )
    assert jnp.array_equal(out_v_obs, expected_v_obs), (
        "observation window values were modified or misplaced by eviction"
    )
    print("[PASS] test_observation_window_kv_survives_unmodified")


def test_pooling_changes_selection_on_spiky_vs_clustered_signal():
    """(c) kernel_size=1 vs kernel_size=5 select measurably different index sets
    on a synthetic vote signal with implanted "spiky" and "clustered" regions,
    confirming the clustering effect described in the pooling demo.
    """
    L_prefix = 512
    k_keep = 13

    # Strictly-decreasing, all-negative baseline: guarantees no ties, so
    # top_k's selection among the "filler" region is fully deterministic
    # and never competes with the real signal below.
    baseline = jnp.linspace(-10.0, -10.001, L_prefix)
    vote = baseline
    vote = vote.at[100].set(50.0)  # isolated spike #1
    vote = vote.at[250].set(50.0)  # isolated spike #2
    vote = vote.at[400].set(50.0)  # isolated spike #3
    # A "clustered" region: a broad, moderately-high plateau at 300..309.
    vote = vote.at[300:310].set(20.0)
    vote = vote[None, None, :]  # [batch=1, heads=1, L_prefix]

    idx_no_pool = votes_to_indices(vote, k=k_keep, kernel_size=1)
    idx_pooled = votes_to_indices(vote, k=k_keep, kernel_size=5)

    set_no_pool = set(idx_no_pool[0, 0].tolist())
    set_pooled = set(idx_pooled[0, 0].tolist())

    assert set_no_pool != set_pooled, (
        "expected kernel_size=1 and kernel_size=5 to select different index sets"
    )

    # Without pooling, each isolated spike is selected alone — neither of
    # its immediate neighbors makes the cut, because pooling never spread
    # the spike's score outward.
    for spike in (100, 250, 400):
        assert spike in set_no_pool
        assert spike - 1 not in set_no_pool and spike + 1 not in set_no_pool, (
            f"expected spike at {spike} to stay isolated without pooling"
        )

    # With pooling, the max-pool spreads each spike's vote to its
    # neighbors before top-k ranks them, so at least one neighbor of every
    # spike now survives too — confirming contiguous clusters form.
    for spike in (100, 250, 400):
        assert spike in set_pooled
        assert (spike - 1 in set_pooled) or (spike + 1 in set_pooled), (
            f"expected pooling to pull at least one neighbor of spike {spike} "
            "into the surviving set, forming a cluster"
        )
    print("[PASS] test_pooling_changes_selection_on_spiky_vs_clustered_signal")


def test_pool1d_kernel_size_one_is_identity():
    """kernel_size=1 must be a true no-op, matching the paper's ablation baseline."""
    from snap_kv import pool1d

    key = jax.random.PRNGKey(3)
    vote = jax.random.normal(key, (2, 4, 100))
    pooled = pool1d(vote, kernel_size=1)
    assert jnp.array_equal(pooled, vote)
    print("[PASS] test_pool1d_kernel_size_one_is_identity")


ALL_TESTS = [
    test_output_cache_size_equals_max_capacity_prompt,
    test_guard_clause_skips_eviction_when_prompt_fits,
    test_observation_window_kv_survives_unmodified,
    test_pooling_changes_selection_on_spiky_vs_clustered_signal,
    test_pool1d_kernel_size_one_is_identity,
]


def main():
    failures = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {test_fn.__name__}: {exc}")
    if failures:
        print(f"\n{failures}/{len(ALL_TESTS)} tests FAILED")
        sys.exit(1)
    print(f"\nAll {len(ALL_TESTS)} tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
