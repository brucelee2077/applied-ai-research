# Results: worker overlapping simulation

Simulated with `overlap_sim.py`, `num_tiles = 8`, `load_time = 0.02s` held
fixed while `compute_time` varies so the ratio `load_time : compute_time`
sweeps across 0.5x, 1x, 2x, and 4x. Timings below are from the analytic
model (`analytic_sequential` / `analytic_pipelined`), which is exact —
no measurement noise:

```
total_sequential  = N * (load_time + compute_time)
total_overlapped  = load_time + N * max(load_time, compute_time)
speedup           = total_sequential / total_overlapped
```

## Speedup vs. load:compute ratio

| load : compute ratio | load_time (s) | compute_time (s) | sequential total (s) | overlapped total (s) | speedup |
|---|---|---|---|---|---|
| 0.5x (compute is 2x load — compute-bound) | 0.02 | 0.04 | 0.4800 | 0.3400 | 1.41x |
| 1x (balanced — load == compute) | 0.02 | 0.02 | 0.3200 | 0.1800 | **1.78x** |
| 2x (load is 2x compute — load-bound) | 0.02 | 0.01 | 0.2400 | 0.1800 | 1.33x |
| 4x (load is 4x compute — heavily load-bound) | 0.02 | 0.005 | 0.2000 | 0.1800 | 1.11x |

## ASCII plot: speedup vs. ratio

```
speedup
1.8 |                *
1.7 |
1.6 |
1.5 |
1.4 |     *
1.3 |                       *
1.2 |
1.1 |                                *
    +----+----+----+----+----+----+----+---> load:compute ratio
        0.5x       1x        2x        4x
```

The curve peaks at the balanced point (ratio = 1x, load_time == compute_time)
and falls off as either the load or the compute step comes to dominate. This
matches the `max()` vs. `sum()` timing model directly:

- When load and compute are equal, overlap hides the largest possible
  fraction of the sequential time — the steady-state cost per tile drops
  from `load + compute` to `max(load, compute)`, which is exactly half as
  much, giving a speedup that approaches 2x as `N -> infinity` (measured
  here as 1.78x at N=8, limited by the one unhideable first load).
- When one step is much larger than the other (ratio 0.5x or 4x), that step
  already dominates the sequential total almost by itself, so there is very
  little slack left for the other step's time to hide inside — overlap
  gains shrink toward 1x (no speedup) as the ratio moves away from 1.

## Threaded (measured, not just analytic) confirmation

Running the actual `threading` + `queue.Queue` producer/consumer simulation
in `overlap_sim.py` for the balanced case (`load_time = compute_time =
0.05s`, `N = 8`) gives a **measured** speedup of approximately **1.6x–1.7x**
across repeated runs (thread-scheduling jitter accounts for the small
variance), comfortably inside the required 1.5x–2.0x band and consistent
with the analytic model above.

## Takeaway for the interview

Worker overlapping does not remove work — it only hides one of the two
costs (load or compute) behind the other. The maximum possible speedup is
bounded by `2x` as `N -> infinity`, and only when load and compute are
balanced. If a kernel is heavily load-bound (HBM-bandwidth-bound), no amount
of overlap fixes that — the steady-state throughput is bounded below by
`max(load_time, compute_time)`, so you have to shrink `load_time` itself
(better tiling, less HBM traffic, higher-bandwidth memory) to go faster.
