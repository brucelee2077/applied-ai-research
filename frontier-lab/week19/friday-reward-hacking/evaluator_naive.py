"""
evaluator_naive.py
==================

The NAIVE verifier -- the kind that gets an ADRS loop reward-hacked.

It measures ONLY speed and latency. It never checks whether every request was
actually served. This is the "ruler with a hole in it" the optimizer exploits.

    score = throughput_rps        (subject to a soft latency SLO)

Run this and you will see the hack policy score MUCH higher than the honest
policy -- because it threw away 38% of the traffic.
"""

from __future__ import annotations

from sim_loadbalancer import RunResult, run_policy


# A Service Level Objective (SLO): a promised maximum p99 latency, in ms.
P99_SLO_MS = 200.0


def score(result: RunResult) -> float:
    """Return the reward the NAIVE verifier assigns to a run.

    It checks two things -- both about SPEED:
        1. p99 latency under the SLO (else score 0)
        2. throughput is positive
    It reports throughput as the reward. It NEVER checks served == submitted.
    """
    # checks speed
    assert result.throughput_rps > 0, "throughput must be positive"
    # checks latency SLO
    assert result.p99_latency_ms < P99_SLO_MS, "p99 latency over SLO"
    # NOTE: there is NO check that served == submitted.  <-- the loophole.
    return result.throughput_rps


def evaluate(policy: str) -> float:
    """Run a policy and score it with the naive verifier."""
    return score(run_policy(policy))


if __name__ == "__main__":
    print("=== NAIVE evaluator (speed only, no correctness gate) ===")
    for name in ("honest", "evolved_v7"):
        res = run_policy(name)
        s = score(res)
        tag = "  <-- HACK WINS" if name == "evolved_v7" else ""
        print(f"  {name:12s} served={res.served:5d}/{res.submitted} "
              f"p99={res.p99_latency_ms:6.2f}ms  score={s:8.1f}{tag}")
    print("\n  The 'evolved' policy scores higher -- by silently dropping traffic.")
    print("  The naive verifier is happy. The system is broken.")
