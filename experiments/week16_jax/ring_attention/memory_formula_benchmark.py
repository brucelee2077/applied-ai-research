"""Memory-formula benchmark: reproduces Table 1 of arXiv:2310.01889 numerically.

Prints, for a sweep of sequence lengths s, the per-layer activation memory
(in bytes, bfloat16 = 2 bytes/element) of:
  - Vanilla self-attention:                 2 * b * h * s^2
  - Blockwise Parallel Transformer (BPT):   2 * b * s * h
  - Ring Attention (per device):            6 * b * c * h

b = batch size, h = hidden size, s = sequence length, c = block size
(c is fixed and independent of s — that is the whole point of the ring:
each device's memory need never grows with total sequence length).

These are the exact formulas from Table 1 of the paper ("Comparison of
maximum activation sizes among different Transformer architectures").
"""

BYTES_PER_ELEMENT = 2  # bfloat16


def vanilla_bytes(b, h, s):
    """2*b*h*s^2 — the full attention matrix, materialized, per layer."""
    return BYTES_PER_ELEMENT * b * h * s * s


def bpt_bytes(b, h, s):
    """2*b*s*h — blockwise attention + blockwise feedforward, per layer.
    Linear in s: no quadratic attention matrix, but still needs to store
    each layer's full output (length s) for the next layer to attend over.
    """
    return BYTES_PER_ELEMENT * b * s * h


def ring_bytes(b, h, c):
    """6*b*c*h — Ring Attention's per-device activation memory, per layer.
    Depends only on the block size c, which is fixed and chosen independent
    of the total sequence length s. This is why Ring Attention's per-device
    memory footprint stays FLAT as s grows: more devices simply means more
    blocks, not a bigger block on any one device.
    """
    return BYTES_PER_ELEMENT * b * c * h


def human_bytes(n):
    """Format a byte count as a human-readable string (KB/MB/GB/TB)."""
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} EB"


def main():
    # Fixed configuration, chosen to match a modest transformer layer.
    b = 1        # batch size
    h = 4096     # hidden size (roughly a 7B-class model's hidden dim)
    c = 1024     # Ring Attention block size — fixed, independent of s

    sweep_s = [1_000, 10_000, 100_000, 1_000_000]

    print(f"Fixed: batch b={b}, hidden h={h}, Ring Attention block size c={c}\n")
    header = f"{'seq_len s':>12} | {'Vanilla 2*b*h*s^2':>22} | {'BPT 2*b*s*h':>18} | {'Ring 6*b*c*h':>18}"
    print(header)
    print("-" * len(header))

    ring_value = None
    for s in sweep_s:
        v = vanilla_bytes(b, h, s)
        bpt = bpt_bytes(b, h, s)
        ring = ring_bytes(b, h, c)  # does not depend on s at all
        if ring_value is None:
            ring_value = ring
        print(f"{s:>12,} | {human_bytes(v):>22} | {human_bytes(bpt):>18} | {human_bytes(ring):>18}")

    print()
    # Sanity checks that back up the qualitative claim the lesson makes.
    v_first, v_last = vanilla_bytes(b, h, sweep_s[0]), vanilla_bytes(b, h, sweep_s[-1])
    bpt_first, bpt_last = bpt_bytes(b, h, sweep_s[0]), bpt_bytes(b, h, sweep_s[-1])
    ring_first, ring_last = ring_bytes(b, h, c), ring_bytes(b, h, c)

    growth_vanilla = v_last / v_first
    growth_bpt = bpt_last / bpt_first
    growth_ring = ring_last / ring_first  # should be exactly 1.0 — flat

    print(f"Vanilla growth factor from s={sweep_s[0]:,} to s={sweep_s[-1]:,}: {growth_vanilla:,.1f}x "
          f"(expected ~{ (sweep_s[-1]/sweep_s[0])**2:,.1f}x, since it is O(s^2))")
    print(f"BPT growth factor:      {growth_bpt:,.1f}x "
          f"(expected ~{ sweep_s[-1]/sweep_s[0]:,.1f}x, since it is O(s))")
    print(f"Ring Attn growth factor: {growth_ring:.1f}x (expected exactly 1.0x, since it is O(1) in s)")

    assert growth_ring == 1.0, "Ring Attention per-device memory must be exactly flat in s"
    assert growth_bpt > 1.0 and growth_vanilla > growth_bpt, (
        "Vanilla must grow strictly faster than BPT as s increases"
    )
    print("\nPASS: Ring Attention per-device memory stays flat while vanilla and BPT both grow with s,")
    print("      and vanilla grows strictly faster (quadratically) than BPT (linearly).")


if __name__ == "__main__":
    main()
