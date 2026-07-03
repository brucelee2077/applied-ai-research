"""Benchmark: baseline (no eviction) vs. SnapKV decode-step read cost.

Reproduces the *shape* of the SnapKV paper's Figure 7: as input length
grows, the baseline's decode-step KV-read cost grows linearly (it must
scan the whole cache every step), while SnapKV's stays flat once the
prompt exceeds max_capacity_prompt (its cache is capped).

The "cost" here is a simulated proxy — total K/V positions read per
decode step, per (layer, head) — not a wall-clock GPU benchmark. This
matches the lesson's PRODUCE brief: "simulate on synthetic or
capstone-model attention matrices, no live HF model needed."

Run:
    python experiments/week16_jax/day2_snapkv_eviction/benchmark.py
"""

from __future__ import annotations

import os

import jax

from snap_kv import snap_kv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# A 7B-class model shape, matching the lesson's Demo (1) cache-size arithmetic.
NUM_LAYERS = 32
NUM_KV_HEADS = 32
HEAD_DIM = 128

WINDOW_SIZE = 16
MAX_CAPACITY_PROMPT = 1024
KERNEL_SIZE = 5

INPUT_LENGTHS = [1024, 4096, 16384, 65536]


def simulate_snapkv_cache_len(q_len: int) -> int:
    """Run snap_kv on a tiny synthetic Q/K/V (batch=1, 1 head) just to read
    off the resulting cache length — this is the real eviction pipeline,
    not a hand-computed shortcut. Using 1 head/1 batch is enough because
    every head's cache length after SnapKV is exactly max_capacity_prompt
    (or q_len, if the guard clause fires) — the *number of kept
    positions* per head does not depend on num_heads or batch size.
    """
    key = jax.random.PRNGKey(q_len)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (1, 1, q_len, HEAD_DIM))
    k = jax.random.normal(k2, (1, 1, q_len, HEAD_DIM))
    v = jax.random.normal(k3, (1, 1, q_len, HEAD_DIM))
    new_k, _ = snap_kv(q, k, v, WINDOW_SIZE, MAX_CAPACITY_PROMPT, KERNEL_SIZE)
    return int(new_k.shape[2])


def baseline_decode_step_reads(input_len: int) -> int:
    """Baseline (no eviction): every decode step reads the ENTIRE KV cache
    — every layer, every KV head, every position seen so far.
    Proxy cost = num_layers * num_kv_heads * cache_len (per decode step).
    """
    return NUM_LAYERS * NUM_KV_HEADS * input_len


def snapkv_decode_step_reads(input_len: int) -> int:
    """SnapKV: after the one-time eviction at end-of-prefill, the cache is
    bounded at max_capacity_prompt per head regardless of how long the
    original prompt was (as long as input_len >= max_capacity_prompt).
    """
    cache_len = simulate_snapkv_cache_len(input_len)
    return NUM_LAYERS * NUM_KV_HEADS * cache_len


def main():
    print(f"Model shape: {NUM_LAYERS} layers x {NUM_KV_HEADS} KV heads x head_dim {HEAD_DIM}")
    print(f"SnapKV config: window_size={WINDOW_SIZE}, max_capacity_prompt={MAX_CAPACITY_PROMPT}, kernel_size={KERNEL_SIZE}")
    print()
    header = f"{'input_len':>10} | {'baseline reads':>16} | {'snapkv reads':>14} | {'snapkv cache_len':>16} | {'compression':>11}"
    print(header)
    print("-" * len(header))

    rows = []
    for input_len in INPUT_LENGTHS:
        baseline_reads = baseline_decode_step_reads(input_len)
        snapkv_reads = snapkv_decode_step_reads(input_len)
        cache_len = snapkv_reads // (NUM_LAYERS * NUM_KV_HEADS)
        compression = input_len / cache_len
        rows.append((input_len, baseline_reads, snapkv_reads, cache_len, compression))
        print(
            f"{input_len:>10} | {baseline_reads:>16,} | {snapkv_reads:>14,} | "
            f"{cache_len:>16} | {compression:>10.1f}x"
        )

    print()
    span = INPUT_LENGTHS[-1] / INPUT_LENGTHS[0]
    print(f"Input length span tested: {span:.0f}x ({INPUT_LENGTHS[0]} -> {INPUT_LENGTHS[-1]})")

    baseline_growth = rows[-1][1] / rows[0][1]
    snapkv_growth = rows[-1][2] / rows[0][2]
    print(f"Baseline read cost growth over that span: {baseline_growth:.1f}x (grows ~linearly with input length)")
    print(f"SnapKV read cost growth over that span:   {snapkv_growth:.1f}x (flat — bounded by max_capacity_prompt)")

    # Try to save a plot; degrade gracefully if matplotlib is unavailable.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lens = [r[0] for r in rows]
        baseline_vals = [r[1] for r in rows]
        snapkv_vals = [r[2] for r in rows]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(lens, baseline_vals, marker="o", label="Baseline (no eviction)", color="#C93B3B")
        ax.plot(lens, snapkv_vals, marker="o", label="SnapKV (bounded cache)", color="#2D8B55")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Input length (tokens)")
        ax.set_ylabel("Simulated decode-step KV reads (proxy cost)")
        ax.set_title("Decode-step KV-read cost vs. input length\n(reproduces the shape of SnapKV paper Fig. 7)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        out_path = os.path.join(SCRIPT_DIR, "benchmark_figure7_shape.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved plot to {out_path}")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"\n(Plot skipped: {exc})")


if __name__ == "__main__":
    main()
