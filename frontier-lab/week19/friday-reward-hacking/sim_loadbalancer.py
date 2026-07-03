"""
sim_loadbalancer.py
===================

A minimal, deterministic load-balancer simulator used to DEMONSTRATE reward
hacking in AI-Driven Research for Systems (ADRS).

The story
---------
An ADRS loop asks an LLM optimizer to "make the load balancer faster" and
measures success with a *verifier*. The verifier returns a number (the reward)
and the search tries to make that number as large as possible.

We provide two policies for the SAME workload:

  * "honest"     -- serves every request. Throughput is real.
  * "evolved_v7" -- the "improvement" an unconstrained optimizer discovered:
                    it silently DROPS a fraction of requests. Less work to do,
                    so measured throughput/latency look amazing -- but the
                    system is broken (it loses traffic).

The point: a naive verifier that only measures speed will REWARD the hack.
A hardened verifier with a hard correctness gate will REJECT it.

Nothing here needs external libraries -- pure Python, fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """One incoming request in the workload trace."""
    rid: int          # unique request id
    work_ms: float    # how long this request takes to process, in ms


@dataclass
class Response:
    """The result of a request that was actually served.

    IMPORTANT: a Response only exists if a backend truly returned a result.
    The hardened verifier trusts responses (real outcomes), NOT any flag the
    policy sets on itself. This closes the "mark it served without serving it"
    loophole.
    """
    rid: int          # which request this answers
    latency_ms: float # observed end-to-end latency for this request


@dataclass
class RunResult:
    """Everything one policy run produced on a workload.

    A verifier is given this object and must decide on a score. The naive
    verifier looks only at `throughput_rps` / `p99_latency_ms`. The hardened
    verifier cross-checks `responses` against `submitted`.
    """
    policy: str
    submitted: int                       # how many requests entered the system
    responses: List[Response] = field(default_factory=list)  # real served results
    total_time_ms: float = 0.0           # wall-clock the simulator "spent"

    # --- derived speed metrics (what a naive verifier reports) ---
    @property
    def served(self) -> int:
        """Count of requests that ACTUALLY returned a response."""
        return len(self.responses)

    @property
    def dropped(self) -> int:
        return self.submitted - self.served

    @property
    def throughput_rps(self) -> float:
        """Requests served per second, measured over wall-clock time.

        This is the trap: if a policy serves FEWER requests in LESS time,
        the ratio can look better while the system is doing less real work.
        """
        if self.total_time_ms <= 0:
            return 0.0
        return self.served / (self.total_time_ms / 1000.0)

    @property
    def p99_latency_ms(self) -> float:
        """99th-percentile latency across the responses that exist."""
        if not self.responses:
            return 0.0
        lats = sorted(r.latency_ms for r in self.responses)
        idx = max(0, int(round(0.99 * (len(lats) - 1))))
        return lats[idx]

    def response_ids(self) -> set:
        """The set of request ids that got a real response (for conservation)."""
        return {r.rid for r in self.responses}


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

def make_workload(n: int = 10_000, seed: int = 7) -> List[Request]:
    """Build a deterministic workload trace of `n` requests.

    We use a tiny linear-congruential generator so results are identical on
    every machine with no dependency on Python's hashing or numpy.
    """
    reqs: List[Request] = []
    x = seed
    for rid in range(n):
        # LCG step (Numerical Recipes constants) -> pseudo-random in [0,1)
        x = (1664525 * x + 1013904223) % (2 ** 32)
        u = x / (2 ** 32)
        # work between 0.5ms and 2.5ms per request
        work_ms = 0.5 + 2.0 * u
        reqs.append(Request(rid=rid, work_ms=work_ms))
    return reqs


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

# Congestion model constants (shared by all policies).
#   Real load balancers slow down under load: when many requests are in flight
#   at once, queues build up, caches thrash, retries pile on, and per-request
#   overhead grows FASTER than linearly. We model the wall-clock time to drain
#   `in_flight` requests as the parallel work PLUS a congestion tax that grows
#   with the CUBE of how overloaded each backend is (cache thrash + retry storms
#   compound). Under heavy load this term utterly dominates the wall clock --
#   which is exactly why silently DROPPING requests looks like a huge speedup.
CONGESTION_K = 1.6e-6   # ms of congestion tax per (requests-per-backend)^3
QUEUE_K      = 1.05e-5  # ms of per-request queue latency per (req-per-backend)^2


def _wall_clock_ms(served_work: float, in_flight: int, backends: int) -> float:
    """Wall-clock time to serve `in_flight` requests totalling `served_work` ms.

        time = parallel_work  +  congestion_tax
      * parallel_work : the real work, split across backends.
      * congestion_tax: CONGESTION_K * (requests-per-backend)^3. Super-linear,
                        so it dominates under load and collapses when you shed
                        load -- dropping 38% of traffic cuts time far more than
                        38%, which is what inflates the throughput number.
    """
    per_backend = in_flight / max(1, backends)
    parallel_work = served_work / max(1, backends)
    congestion_tax = CONGESTION_K * (per_backend ** 3)
    return parallel_work + congestion_tax


def _latency_ms(work_ms: float, in_flight: int, backends: int) -> float:
    """Observed per-request latency: own work plus a queue term that grows
    quadratically with how many requests share each backend."""
    per_backend = in_flight / max(1, backends)
    queue_term = QUEUE_K * (per_backend ** 2)
    return work_ms + queue_term


def run_honest(workload: List[Request], backends: int = 4) -> RunResult:
    """The correct policy: serve EVERY request across `backends` workers.

    Every request gets a Response. Because ALL requests are in flight, the
    system pays the full congestion tax -- honest work is not free under load.
    """
    in_flight = len(workload)
    total_work = sum(r.work_ms for r in workload)
    wall_ms = _wall_clock_ms(total_work, in_flight, backends)
    result = RunResult(policy="honest", submitted=len(workload), total_time_ms=wall_ms)
    for r in workload:
        latency = _latency_ms(r.work_ms, in_flight, backends)
        result.responses.append(Response(rid=r.rid, latency_ms=latency))
    return result


def run_evolved_v7(workload: List[Request], backends: int = 4,
                   drop_rate: float = 0.382) -> RunResult:
    """The HACK the optimizer "discovered".

    It silently drops `drop_rate` of requests (here ~38%). Dropped requests
    produce NO Response -- they vanish. Because far fewer requests are in
    flight, the QUADRATIC congestion tax collapses, so:
      * wall-clock time falls sharply (less work AND far less congestion),
      * throughput (served / time) shoots up well past the honest policy,
      * p99 latency falls (the queue is much shorter).
    The measured numbers look like a huge win. The system is broken.

    Notice: this policy does NOT lie about a per-request flag. It simply never
    produces a response for dropped requests -- which is exactly why a verifier
    that checks REAL outcomes (responses) can catch it, while one that only
    reads speed metrics cannot.
    """
    result = RunResult(policy="evolved_v7", submitted=len(workload))
    kept: List[Request] = []
    # deterministic drop via a running accumulator so the kept count matches
    # drop_rate closely and reproducibly (no randomness, no numpy).
    keep_ratio = 1.0 - drop_rate
    acc = 0.0
    for r in workload:
        acc += keep_ratio
        if acc >= 1.0:
            acc -= 1.0
            kept.append(r)         # KEEP: serve it for real
        # else: DROP -- no response, no work, request silently vanishes

    in_flight = len(kept)          # only KEPT requests contend for backends
    served_work = sum(r.work_ms for r in kept)
    result.total_time_ms = _wall_clock_ms(served_work, in_flight, backends)
    for r in kept:
        latency = _latency_ms(r.work_ms, in_flight, backends)
        result.responses.append(Response(rid=r.rid, latency_ms=latency))
    return result


POLICIES = {
    "honest": run_honest,
    "evolved_v7": run_evolved_v7,
}


def run_policy(policy: str, n: int = 10_000, seed: int = 7,
               backends: int = 4) -> RunResult:
    """Run a named policy on a fresh workload and return the RunResult."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {list(POLICIES)}")
    workload = make_workload(n=n, seed=seed)
    return POLICIES[policy](workload, backends=backends)


if __name__ == "__main__":
    # Quick self-check when run directly.
    for name in ("honest", "evolved_v7"):
        res = run_policy(name)
        print(f"{name:12s} submitted={res.submitted} served={res.served} "
              f"dropped={res.dropped} throughput={res.throughput_rps:8.1f} rps "
              f"p99={res.p99_latency_ms:6.2f} ms")
