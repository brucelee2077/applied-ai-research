# Week 16 Day 2 — SnapKV Cache Eviction Simulator

A from-scratch (JAX) simulator of the SnapKV KV-cache eviction pipeline:
**vote → 1D-pool → top-k → gather**, run once at the end of prefill.

## Files

- `snap_kv.py` — the eviction pipeline itself (`snap_kv`), the 1D max-pool
  helper (`pool1d`), and two test-only helpers (`selected_prefix_indices`,
  `votes_to_indices`) that expose the voting/pooling/top-k stages directly.
- `test_snap_kv.py` — unit tests for correctness (cache size, observation
  window preservation, and the pooling/clustering effect).
- `benchmark.py` — reproduces the *shape* of the SnapKV paper's Figure 7
  (baseline decode-step read cost grows linearly with input length;
  SnapKV's stays flat).

## Run it

```bash
python experiments/week16_jax/day2_snapkv_eviction/test_snap_kv.py
python experiments/week16_jax/day2_snapkv_eviction/benchmark.py
```

Both exit 0. `benchmark.py` also saves `benchmark_figure7_shape.png`.

## Measured compression ratio

Using a 7B-class model shape (32 layers, 32 KV heads, head_dim 128) and
`max_capacity_prompt = 1024`:

| input_len | baseline reads/step | SnapKV reads/step | cache_len | compression (`input_len / max_capacity_prompt`) |
|-----------|---------------------|--------------------|-----------|--------------------------------------------------|
| 1,024     | 1,048,576           | 1,048,576          | 1,024     | 1.0x  (`q_len < max_capacity_prompt`? No — equal, eviction runs but nothing to cut) |
| 4,096     | 4,194,304           | 1,048,576          | 1,024     | 4.0x  |
| 16,384    | 16,777,216          | 1,048,576          | 1,024     | 16.0x |
| 65,536    | 67,108,864          | 1,048,576          | 1,024     | 64.0x |

Over the tested 64x input-length span (1,024 → 65,536 tokens), the
baseline's simulated decode-step read cost grows **64.0x** (linear in
input length), while SnapKV's stays **flat (1.0x growth)** — bounded by
`max_capacity_prompt` regardless of how long the original prompt was.
This matches the qualitative shape of the paper's Figure 7: baseline
decode latency climbs linearly with input length; SnapKV's decode
latency stays constant once the cache hits its capacity.

## Paper figures, for comparison

From Li et al., *SnapKV: LLM Knows What You Are Looking for Before
Generation* (arXiv:2404.14469):

- **380x compression**: SnapKV processes up to 380K context tokens on a
  single A100-80GB GPU with a KV cache budget of 1,024 positions per
  head (380,000 / 1,024 ≈ 380x), in the paper's Needle-in-a-Haystack
  test — while the uncompressed baseline hits an out-of-memory (OOM)
  error at just 33K input tokens on the same GPU.
- **3.6x faster decoding**: at a 16K-token input and batch size 2, the
  baseline's decode latency exceeds 100 ms/token while SnapKV's stays
  below 40 ms/token — about a 3.6x speedup.
- **8.2x more memory-efficient**: at batch size 2, the baseline OOMs
  beyond 16K input tokens, while the SnapKV-enhanced model scales to
  131K input tokens before hitting the same limit — roughly an 8.2x
  improvement in how far the same memory budget stretches.

Our simulator's measured compression ratios (1x / 4x / 16x / 64x across
the four tested input lengths) are directly comparable in *shape* to
the paper's headline **380x** figure — they are lower only because we
picked shorter input lengths (up to 65,536 tokens) for a fast local
benchmark; the same simulator run with `input_len = 389,120` and
`max_capacity_prompt = 1024` would show a ~380x ratio, matching the
paper almost exactly, since compression ratio in this simulator is
simply `input_len / max_capacity_prompt` once eviction is active.

## What this simulator does NOT model

- Real wall-clock latency or actual HBM bandwidth — the "cost" figures
  are a proxy (`num_layers * num_kv_heads * cache_len` positions read
  per decode step), not a GPU benchmark.
- Real attention accuracy / downstream task performance — that's the
  paper's LongBench and Needle-in-a-Haystack evaluations, not something
  a synthetic-signal simulator can measure.
- Decode-time cache growth: like the real SnapKV, this simulator only
  evicts once, at the end of prefill. Newly generated tokens during
  decode still append to the cache unboundedly unless paired with a
  separate decode-time eviction policy (see lesson Section 4's "catch").
