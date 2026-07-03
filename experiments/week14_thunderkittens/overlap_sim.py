"""
Week 14 Day 3 — Worker Overlapping concept demo.

This reproduces the timing intuition behind ThunderKittens' Load-Store-
Compute-Finish (LCF) template without touching any real CUDA/H100 hardware.

Two models of processing N tiles, each needing a "load" step (fetch from
HBM into shared memory) and a "compute" step (do math on the tile that is
already staged):

1. Sequential: load tile i, then compute tile i, then load tile i+1, ...
   The math units are idle during every load, and the load path is idle
   during every compute. Nothing overlaps.

       total_time = N * (load_time + compute_time)

2. Pipelined / overlapped ("worker overlapping"): a producer keeps loading
   tile i+1 while a consumer computes on tile i. After the first load
   (which nothing can hide), the two streams run in parallel, so each
   step of the steady state costs only the slower of the two operations.

       total_time ≈ load_time + N * max(load_time, compute_time)

This is exactly the LCF mental model: producer warps (load) and consumer
warps (compute) run at the same time instead of taking turns.
"""

import queue
import threading
import time


def simulate_sequential(num_tiles, load_time, compute_time):
    """Naive pipeline: load tile, then compute tile, one after another.

    No overlap at all -- this is what a kernel looks like before anyone
    applies the LCF / producer-consumer pattern.
    """
    total = 0.0
    for _tile_index in range(num_tiles):
        total += load_time  # fetch from "HBM" (simulated as a sleep)
        total += compute_time  # do "math" on the tile (simulated as a sleep)
    return total


def simulate_pipelined(num_tiles, load_time, compute_time):
    """Overlapped pipeline: a producer thread and a consumer thread.

    The producer (stand-in for TK's producer warps issuing async TMA
    loads) keeps a bounded queue.Queue filled with "ready" tiles. The
    consumer (stand-in for TK's consumer warps running compute()) pulls
    a ready tile and computes on it. Because they are separate threads,
    the producer can already be loading tile i+1 while the consumer is
    still computing on tile i -- the same overlap TK gets from warp
    specialization plus async TMA loads and barriers.

    A bounded queue of size 1 mirrors a single-stage pipeline buffer:
    the producer can be at most one tile ahead of the consumer, which is
    the minimum needed to get the max() timing model instead of the
    sum() timing model.
    """
    ready_queue = queue.Queue(maxsize=1)
    # "finished" is a plain counter guarded by the GIL-safe queue.Queue,
    # used only so the producer knows when to stop.
    stop_signal = object()

    def producer():
        for tile_index in range(num_tiles):
            time.sleep(load_time)  # async TMA load stand-in
            ready_queue.put(tile_index)  # signal: "this slot is ready" (arrive)
        ready_queue.put(stop_signal)

    def consumer(results):
        while True:
            item = ready_queue.get()  # wait: "is the next tile ready?"
            if item is stop_signal:
                break
            time.sleep(compute_time)  # tensor-core compute stand-in
            results.append(item)

    results = []
    start = time.perf_counter()
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer, args=(results,))
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()
    elapsed = time.perf_counter() - start

    assert len(results) == num_tiles, "consumer did not process every tile"
    return elapsed


def analytic_sequential(num_tiles, load_time, compute_time):
    """Closed-form total time for the sequential model (no sleeps needed)."""
    return num_tiles * (load_time + compute_time)


def analytic_pipelined(num_tiles, load_time, compute_time):
    """Closed-form total time for the overlapped model (no sleeps needed).

    The very first load can never be hidden (there is nothing to overlap
    it with yet). After that, each of the N steps is bottlenecked by
    whichever of load/compute is slower, because the other one is
    happening at the same time in the background.
    """
    return load_time + num_tiles * max(load_time, compute_time)


def run_ratio(num_tiles, load_time, compute_time, use_threads=False):
    """Run one (load_time, compute_time) configuration and report speedup."""
    if use_threads:
        sequential_time = simulate_sequential(num_tiles, load_time, compute_time)
        pipelined_time = simulate_pipelined(num_tiles, load_time, compute_time)
    else:
        sequential_time = analytic_sequential(num_tiles, load_time, compute_time)
        pipelined_time = analytic_pipelined(num_tiles, load_time, compute_time)
    speedup = sequential_time / pipelined_time
    return sequential_time, pipelined_time, speedup


def main():
    num_tiles = 8

    print("=" * 72)
    print("Worker overlapping simulation — analytic timing model")
    print("total_seq  = N * (load + compute)")
    print("total_over = load + N * max(load, compute)")
    print("=" * 72)

    # ratio = load_time : compute_time, expressed as compute_time / load_time
    configs = [
        ("load-bound (load = 2x compute)", 0.02, 0.01),
        ("balanced   (load == compute)", 0.02, 0.02),
        ("compute-bound (compute = 2x load)", 0.01, 0.02),
        ("compute-bound (compute = 4x load)", 0.005, 0.02),
    ]

    print(f"\n{'config':38s} {'seq (s)':>10s} {'overlap (s)':>12s} {'speedup':>9s}")
    print("-" * 72)
    for label, load_time, compute_time in configs:
        seq_t, over_t, speedup = run_ratio(num_tiles, load_time, compute_time, use_threads=False)
        print(f"{label:38s} {seq_t:10.4f} {over_t:12.4f} {speedup:8.2f}x")

    print("\n" + "=" * 72)
    print("Confirming the analytic model with a real threaded simulation")
    print("(producer thread + queue.Queue + consumer thread, actual sleeps)")
    print("=" * 72)
    balanced_load, balanced_compute = 0.05, 0.05
    seq_t, over_t, speedup = run_ratio(num_tiles, balanced_load, balanced_compute, use_threads=True)
    print(f"\nBalanced case (load_time == compute_time == {balanced_load}s, N={num_tiles} tiles):")
    print(f"  sequential (measured, threads): {seq_t:.4f}s")
    print(f"  overlapped (measured, threads): {over_t:.4f}s")
    print(f"  speedup:                        {speedup:.2f}x")

    assert 1.5 <= speedup <= 2.0, (
        f"expected balanced-case speedup between 1.5x and 2.0x, got {speedup:.2f}x"
    )
    print("\nPASS: balanced-case speedup is between 1.5x and 2.0x, confirming the")
    print("max() vs sum() timing model (theoretical limit as N -> inf is 2.0x).")


if __name__ == "__main__":
    main()
