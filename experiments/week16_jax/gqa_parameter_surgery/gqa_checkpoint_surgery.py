"""
GQA checkpoint surgery — Week 16, Day 3 artifact.

Curriculum line being made checkable (not just asserted):
  "Analyze how GQA physically changes the parameter structure of the model
   to reduce the KV footprint during the initial training phase."

What this script does, concretely:
  1. Defines a toy Multi-Head Attention (MHA) block's parameters as an explicit
     weight dict (W_Q, W_K, W_V, W_O), with configurable d_model / num_heads / head_dim.
  2. Implements mha_to_gqa(params, num_groups): a pure function that mean-pools
     the K and V projection weights from [d_model, num_heads, head_dim] down to
     [d_model, num_groups, head_dim], leaving W_Q (and W_O) untouched. This is the
     literal "checkpoint surgery" the GQA paper (arXiv:2305.13245) describes for
     converting an MHA checkpoint into a GQA/MQA one via mean-pooling.
  3. Prints exact parameter-count reductions for num_groups in {1 (MQA), 8 (GQA-8),
     num_heads (identity/MHA)}.
  4. Computes projected KV-cache bytes for a stated (num_layers, seq_len, dtype)
     config, for each num_groups setting, to make the "reduces the KV footprint"
     claim checkable rather than asserted.
  5. Includes unit tests (run via `python3 gqa_checkpoint_surgery.py --test`, or
     automatically at the end of a normal run) asserting mha_to_gqa's shapes and
     numerics are correct.

Uses jax.numpy when JAX is available (this is a JAX-curriculum artifact), and
falls back transparently to plain NumPy so the script is runnable and testable
in any environment with only NumPy installed. The math is identical either way
because this experiment is about parameter *structure*, not about compiling or
running on an accelerator — no GPU/TPU is required.
"""

from __future__ import annotations

import sys

try:
    import jax.numpy as jnp
    _BACKEND = "jax.numpy"
except ImportError:  # pragma: no cover - exercised when jax isn't installed
    import numpy as jnp  # type: ignore[no-redef]
    _BACKEND = "numpy (jax not installed — falling back; math is identical)"

import numpy as np


# ---------------------------------------------------------------------------
# 1. Toy MHA block — explicit weight dict, no framework dependency required.
# ---------------------------------------------------------------------------

def init_mha_params(d_model: int, num_heads: int, head_dim: int, seed: int = 0) -> dict:
    """Create a toy MHA parameter dict with the shapes real transformers use.

    Shapes (all projections are per-head, stacked on axis 1):
      W_Q: [d_model, num_heads, head_dim]
      W_K: [d_model, num_heads, head_dim]
      W_V: [d_model, num_heads, head_dim]
      W_O: [num_heads, head_dim, d_model]   (output projection, unaffected by GQA)
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(d_model)
    params = {
        "W_Q": jnp.asarray(rng.normal(size=(d_model, num_heads, head_dim)) * scale),
        "W_K": jnp.asarray(rng.normal(size=(d_model, num_heads, head_dim)) * scale),
        "W_V": jnp.asarray(rng.normal(size=(d_model, num_heads, head_dim)) * scale),
        "W_O": jnp.asarray(rng.normal(size=(num_heads, head_dim, d_model)) * scale),
    }
    return params


# ---------------------------------------------------------------------------
# 2. The actual "parameter surgery": mean-pool K/V heads into groups.
# ---------------------------------------------------------------------------

def mha_to_gqa(params: dict, num_groups: int) -> dict:
    """Pure function: convert MHA weights into GQA/MQA weights by mean-pooling
    K and V projection heads into `num_groups` groups. W_Q and W_O pass through
    unchanged — GQA only shrinks the K/V projections, never the query side.

    Args:
      params: dict with W_Q, W_K, W_V of shape [d_model, num_heads, head_dim]
              and W_O of shape [num_heads, head_dim, d_model].
      num_groups: target number of K/V head-groups.
                  num_groups == num_heads  -> identity (plain MHA).
                  1 < num_groups < num_heads -> GQA.
                  num_groups == 1           -> MQA (single shared K/V head).

    Returns:
      New params dict with W_K, W_V of shape [d_model, num_groups, head_dim].
      W_Q and W_O are copied through unmodified (same object values, new dict).

    Mechanism: heads are partitioned into `num_groups` contiguous, equal-sized
    chunks along the head axis, and each chunk is averaged (mean-pooled) into
    one representative head. This mirrors the GQA paper's checkpoint-conversion
    recipe (arXiv:2305.13245, Section 3.1): initialize each group's K/V head as
    the mean of the original heads that fall into that group, so the converted
    checkpoint starts close to the original MHA model's behavior before any
    further "uptraining".
    """
    w_k, w_v = params["W_K"], params["W_V"]
    d_model, num_heads, head_dim = w_k.shape

    if num_heads % num_groups != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_groups ({num_groups}) "
            "for even head-group partitioning."
        )

    heads_per_group = num_heads // num_groups

    def pool_heads(w):
        # [d_model, num_heads, head_dim] -> [d_model, num_groups, heads_per_group, head_dim]
        reshaped = w.reshape(d_model, num_groups, heads_per_group, head_dim)
        # Mean over the heads-within-a-group axis -> [d_model, num_groups, head_dim]
        return reshaped.mean(axis=2)

    new_params = dict(params)  # shallow copy; W_Q, W_O deliberately untouched
    new_params["W_K"] = pool_heads(w_k)
    new_params["W_V"] = pool_heads(w_v)
    return new_params


# ---------------------------------------------------------------------------
# 3. Parameter-count accounting.
# ---------------------------------------------------------------------------

def count_params(shape) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def report_parameter_reduction(d_model: int, num_heads: int, head_dim: int, num_groups: int) -> dict:
    """Print and return the before/after K+V parameter counts for one num_groups setting."""
    original_shape = (d_model, num_heads, head_dim)
    new_shape = (d_model, num_groups, head_dim)

    original_count_per_matrix = count_params(original_shape)
    new_count_per_matrix = count_params(new_shape)

    # K and V are both affected identically -> factor of 2.
    original_kv_total = 2 * original_count_per_matrix
    new_kv_total = 2 * new_count_per_matrix
    reduction_pct = 100.0 * (1.0 - new_kv_total / original_kv_total)

    label = {1: "MQA (num_groups=1)", num_heads: "MHA/identity (num_groups=num_heads)"}.get(
        num_groups, f"GQA-{num_groups} (num_groups={num_groups})"
    )

    print(f"\n--- {label} ---")
    print(f"  W_K/W_V shape : {original_shape} -> {new_shape}")
    print(f"  params/matrix : {original_count_per_matrix:,} -> {new_count_per_matrix:,}")
    print(f"  total K+V     : {original_kv_total:,} -> {new_kv_total:,}")
    print(f"  reduction     : {reduction_pct:.1f}%")

    return {
        "num_groups": num_groups,
        "original_shape": original_shape,
        "new_shape": new_shape,
        "original_kv_total": original_kv_total,
        "new_kv_total": new_kv_total,
        "reduction_pct": reduction_pct,
    }


# ---------------------------------------------------------------------------
# 4. KV cache memory projection (makes the "reduces KV footprint" claim checkable).
# ---------------------------------------------------------------------------

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}


def projected_kv_cache_bytes(
    num_layers: int, seq_len: int, num_kv_heads: int, head_dim: int, dtype: str = "fp16"
) -> int:
    """cache_bytes = 2 (K and V) x num_layers x num_kv_heads x head_dim x seq_len x bytes_per_element

    This is the standard KV-cache memory formula (see e.g. Pope et al., arXiv:2211.05102):
    at every layer, for every KV head, you store one head_dim-sized vector per cached
    token, for both K and V.
    """
    bytes_per_element = DTYPE_BYTES[dtype]
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_element


def report_kv_cache_projection(
    num_layers: int, seq_len: int, head_dim: int, num_groups_settings, dtype: str = "fp16"
) -> list:
    print(f"\n=== Projected KV cache size: num_layers={num_layers}, seq_len={seq_len}, dtype={dtype} ===")
    results = []
    for num_kv_heads in num_groups_settings:
        n_bytes = projected_kv_cache_bytes(num_layers, seq_len, num_kv_heads, head_dim, dtype)
        gb = n_bytes / 1e9
        print(f"  num_kv_heads={num_kv_heads:>3}  ->  {n_bytes:,} bytes  ({gb:.4f} GB)")
        results.append({"num_kv_heads": num_kv_heads, "bytes": n_bytes, "gb": gb})
    return results


# ---------------------------------------------------------------------------
# 5. Unit tests.
# ---------------------------------------------------------------------------

def _to_numpy(x):
    return np.asarray(x)


def run_tests():
    print(f"\n=== Running unit tests (backend: {_BACKEND}) ===")
    d_model, num_heads, head_dim = 32, 8, 4
    params = init_mha_params(d_model, num_heads, head_dim, seed=42)

    # --- Test 1: shape correctness for a mid-range num_groups ---
    num_groups = 2
    gqa_params = mha_to_gqa(params, num_groups)
    expected_shape = (d_model, num_groups, head_dim)
    assert _to_numpy(gqa_params["W_K"]).shape == expected_shape, (
        f"W_K shape mismatch: got {_to_numpy(gqa_params['W_K']).shape}, expected {expected_shape}"
    )
    assert _to_numpy(gqa_params["W_V"]).shape == expected_shape, (
        f"W_V shape mismatch: got {_to_numpy(gqa_params['W_V']).shape}, expected {expected_shape}"
    )
    print(f"  [PASS] mha_to_gqa(num_groups={num_groups}) produces W_K/W_V shape {expected_shape}")

    # --- Test 2: W_Q and W_O must be untouched (this catches "wrong axis pooled" bugs) ---
    assert np.array_equal(_to_numpy(gqa_params["W_Q"]), _to_numpy(params["W_Q"])), (
        "W_Q was modified by mha_to_gqa — query weights must never change!"
    )
    assert np.array_equal(_to_numpy(gqa_params["W_O"]), _to_numpy(params["W_O"])), (
        "W_O was modified by mha_to_gqa — output projection must never change!"
    )
    print("  [PASS] W_Q and W_O are byte-for-byte identical to the original MHA params")

    # --- Test 3: num_groups == num_heads is a no-op (mean-pool of a group of 1) ---
    identity_params = mha_to_gqa(params, num_heads)
    assert np.allclose(_to_numpy(identity_params["W_K"]), _to_numpy(params["W_K"])), (
        "num_groups=num_heads should leave W_K numerically unchanged (mean of 1 element = itself)"
    )
    assert np.allclose(_to_numpy(identity_params["W_V"]), _to_numpy(params["W_V"])), (
        "num_groups=num_heads should leave W_V numerically unchanged (mean of 1 element = itself)"
    )
    print(f"  [PASS] mha_to_gqa(num_groups=num_heads={num_heads}) is numerically a no-op (MHA identity)")

    # --- Test 4: num_groups == 1 produces a single head equal to the mean of ALL original heads ---
    mqa_params = mha_to_gqa(params, 1)
    expected_k_mean = _to_numpy(params["W_K"]).mean(axis=1, keepdims=True)  # [d_model, 1, head_dim]
    expected_v_mean = _to_numpy(params["W_V"]).mean(axis=1, keepdims=True)
    assert np.allclose(_to_numpy(mqa_params["W_K"]), expected_k_mean), (
        "num_groups=1 (MQA) should produce a single K head equal to the mean of all original K heads"
    )
    assert np.allclose(_to_numpy(mqa_params["W_V"]), expected_v_mean), (
        "num_groups=1 (MQA) should produce a single V head equal to the mean of all original V heads"
    )
    print("  [PASS] mha_to_gqa(num_groups=1) (MQA) equals the mean of all original heads")

    # --- Test 5: invalid num_groups (non-divisor of num_heads) raises loudly ---
    raised = False
    try:
        mha_to_gqa(params, 3)  # 8 heads is not divisible by 3
    except ValueError:
        raised = True
    assert raised, "mha_to_gqa must raise ValueError when num_heads is not divisible by num_groups"
    print("  [PASS] mha_to_gqa raises ValueError on a non-divisor num_groups (8 heads, num_groups=3)")

    # --- Test 6: parameter count reduction arithmetic matches direct computation ---
    result = report_parameter_reduction(d_model, num_heads, head_dim, num_groups=2)
    expected_original = 2 * d_model * num_heads * head_dim
    expected_new = 2 * d_model * 2 * head_dim
    assert result["original_kv_total"] == expected_original
    assert result["new_kv_total"] == expected_new
    print("  [PASS] parameter-count report matches direct hand computation")

    print("\nAll tests passed.")


# ---------------------------------------------------------------------------
# Main: demonstrate everything end-to-end.
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("GQA checkpoint surgery — parameter structure + KV cache footprint")
    print(f"Backend: {_BACKEND}")
    print("=" * 72)

    # A realistic-ish toy config: small enough to run instantly on CPU,
    # but with head counts that mirror a real model (Llama-2-7B style: 32 heads).
    d_model, num_heads, head_dim = 4096, 32, 128
    print(f"\nToy MHA config: d_model={d_model}, num_heads={num_heads}, head_dim={head_dim}")

    params = init_mha_params(d_model, num_heads, head_dim, seed=0)
    print(f"  W_Q shape: {tuple(_to_numpy(params['W_Q']).shape)}  (never touched by GQA surgery)")
    print(f"  W_K shape: {tuple(_to_numpy(params['W_K']).shape)}  (before surgery)")
    print(f"  W_V shape: {tuple(_to_numpy(params['W_V']).shape)}  (before surgery)")

    # --- Parameter-count reduction for 3 required settings ---
    group_settings = [1, 8, num_heads]  # MQA, GQA-8, identity/MHA
    param_reports = [
        report_parameter_reduction(d_model, num_heads, head_dim, num_groups=g) for g in group_settings
    ]

    # --- Also actually run mha_to_gqa for each setting, to prove the function works
    #     on real arrays, not just on the arithmetic. ---
    print("\n=== Verifying mha_to_gqa() produces exactly these shapes on real arrays ===")
    for g in group_settings:
        converted = mha_to_gqa(params, g)
        actual_shape = tuple(_to_numpy(converted["W_K"]).shape)
        expected_shape = (d_model, g, head_dim)
        status = "OK" if actual_shape == expected_shape else "MISMATCH"
        print(f"  num_groups={g:>3}: mha_to_gqa(...)['W_K'].shape = {actual_shape}  [{status}]")
        assert actual_shape == expected_shape

    # --- Projected KV cache bytes, reproducing the "reduce the KV footprint" claim ---
    num_layers, seq_len, dtype = 32, 128_000, "fp16"
    kv_reports = report_kv_cache_projection(
        num_layers=num_layers, seq_len=seq_len, head_dim=head_dim,
        num_groups_settings=group_settings, dtype=dtype,
    )

    mha_gb = next(r["gb"] for r in kv_reports if r["num_kv_heads"] == num_heads)
    mqa_gb = next(r["gb"] for r in kv_reports if r["num_kv_heads"] == 1)
    gqa8_gb = next(r["gb"] for r in kv_reports if r["num_kv_heads"] == 8)
    print(f"\nSummary at seq_len={seq_len:,}, num_layers={num_layers}, dtype={dtype}:")
    print(f"  MHA  (num_kv_heads=32): {mha_gb:.2f} GB")
    print(f"  GQA-8(num_kv_heads=8) : {gqa8_gb:.2f} GB  ({mha_gb / gqa8_gb:.1f}x smaller than MHA)")
    print(f"  MQA  (num_kv_heads=1) : {mqa_gb:.2f} GB  ({mha_gb / mqa_gb:.1f}x smaller than MHA)")

    # --- Unit tests ---
    run_tests()

    print("\nDone. Exit code 0.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        main()
