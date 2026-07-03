# Results — Reward Hacking in ADRS (the dropped-workload load balancer)

All output below is **real terminal output** captured by `./run.sh` on this
machine. Numbers are fully deterministic (fixed seed, pure-Python LCG), so they
reproduce exactly on any machine.

Reproduce everything with one command:

```
./run.sh
```

---

## Step 1 — raw simulator: honest vs. the "evolved" hack

```
honest       submitted=10000 served=10000 dropped=0    throughput=  347.7 rps  p99= 68.11 ms
evolved_v7   submitted=10000 served=6179  dropped=3821  throughput=  751.4 rps  p99= 27.53 ms
```

The "evolved" policy silently **drops 3,821 of 10,000 requests (38%)**. Because
the wall-clock time is dominated by a super-linear congestion tax, shedding 38%
of the load cuts the time by far more than 38% — so the *measured* throughput
**more than doubles** (347.7 → 751.4 rps) and p99 latency more than halves
(68.11 → 27.53 ms). The system is now broken (it loses traffic), but every
speed number looks better.

---

## Step 2 — pytest correctness gate (request conservation)

```
.F...                                                                    [100%]
=================================== FAILURES ===================================
_____________________ test_all_requests_served[evolved_v7] _____________________

>       assert res.served == res.submitted, (
            f"{res.submitted - res.served} requests never returned a response "
            ...
E       AssertionError: 3821 requests never returned a response (conservation
        violated): served=6179 != submitted=10000
E       assert 6179 == 10000

test_conservation.py:32: AssertionError
=========================== short test summary info ============================
FAILED test_conservation.py::test_all_requests_served[evolved_v7]
```

Result: **1 failed, 4 passed.** The honest policy passes conservation; the hack
fails with an assertion message that names the violation
(`served=6179 != submitted=10000`). This is the gate a hardened verifier runs
*before* it ever looks at speed.

---

## Step 3 — NAIVE evaluator (speed only) — the hack WINS

```
=== NAIVE evaluator (speed only, no correctness gate) ===
  honest       served=10000/10000  p99= 68.11ms  score=  347.7
  evolved_v7   served= 6179/10000  p99= 27.53ms  score=  751.4  <-- HACK WINS
```

Under the naive verifier the hack scores **751.4 vs. 347.7 — 2.16x higher.**
The verifier only checked `throughput > 0` and `p99 < 200ms` (the SLO); both
policies pass those, so it happily reports the hack as the better solution.
An ADRS search would keep this "improvement" and throw the honest one away.

---

## Step 4 — HARDENED evaluator (correctness gate) — hack forced to 0

```
=== HARDENED evaluator (correctness GATE, then speed) ===
  honest       served=10000/10000  gate=PASS
               score=  347.7
  evolved_v7   served= 6179/10000  gate=REJECTED (conservation violated: 3821
               requests never returned a response (served=6179 != submitted=10000))
               score=    0.0

  The hack now scores 0. The honest policy wins. Ranking flipped.
```

Under the hardened verifier the correctness gate rejects the hack outright:
its score is forced to **0.0**, while the honest policy keeps its positive
score of 347.7. **The ranking flips** — the search can no longer be rewarded
for cheating.

---

## Acceptance criteria — checklist

| Criterion | Status | Evidence |
|-----------|--------|----------|
| `pytest test_conservation.py -q` passes for honest, fails for hack, names conservation violation | PASS | Step 2 — `served=6179 != submitted=10000` |
| Under naive evaluator the hack scores strictly higher | PASS | Step 3 — 751.4 > 347.7 (2.16x) |
| Under hardened evaluator the hack scores exactly 0, honest stays positive, ranking flips | PASS | Step 4 — hack 0.0, honest 347.7 |
| results.md contains real terminal output, no placeholders | PASS | This file |
| `run.sh` reproduces everything with one command and exits 0 | PASS | `EXIT=0` |

**The one-line lesson:** the verifier *is* your definition of success. A naive
verifier rewards a broken system; a hardened one with a hard correctness gate
does not.
