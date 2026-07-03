"""
evaluator_hardened.py
=====================

The HARDENED verifier -- "bulletproof correctness verification".

Correctness is a HARD GATE, not a soft penalty. A solution that is fast but
wrong scores exactly 0 and is rejected outright, no matter how good its speed
numbers look. Speed is only ever compared among solutions that are ALREADY
correct.

The gate checks REAL OUTCOMES (actual responses), not any self-reported flag
the policy controls:

    1. CONSERVATION : served == submitted  (every request returned a response)
    2. NO DOUBLE-SERVE : no response id appears twice (no phantom credit)
    3. SLO : p99 latency under the promised maximum

Why a gate and not `score = throughput - k * dropped`?
Because a soft penalty is still hackable: if the throughput gain from dropping
outweighs k * dropped, dropping still wins, and the optimizer tunes itself
right up to the break-even point. A gate removes the trade entirely.
"""

from __future__ import annotations

from sim_loadbalancer import RunResult, run_policy

P99_SLO_MS = 200.0


class CorrectnessError(AssertionError):
    """Raised when a run fails the hard correctness gate."""


def correctness_gate(result: RunResult) -> None:
    """Raise CorrectnessError if the run is not correct and complete.

    This is the airtight check. It trusts only real responses.
    """
    # 1. CONSERVATION: every submitted request must have returned a response.
    if result.served != result.submitted:
        missing = result.submitted - result.served
        raise CorrectnessError(
            f"conservation violated: {missing} requests never returned a "
            f"response (served={result.served} != submitted={result.submitted})"
        )
    # 2. NO DOUBLE-SERVE: response ids must be unique (no phantom credit).
    if len(result.response_ids()) != len(result.responses):
        raise CorrectnessError("double-serve detected: a request id appears twice")
    # 3. SLO honored.
    if result.p99_latency_ms >= P99_SLO_MS:
        raise CorrectnessError(
            f"SLO violated: p99={result.p99_latency_ms:.2f}ms >= {P99_SLO_MS}ms"
        )


def score(result: RunResult) -> float:
    """Return the reward the HARDENED verifier assigns.

    If the correctness gate fails, the score is forced to 0 (rejected).
    Only if the run is correct AND complete do we reward it with throughput.
    """
    try:
        correctness_gate(result)
    except CorrectnessError:
        return 0.0  # rejected: fast-but-wrong never reaches the leaderboard
    return result.throughput_rps


def evaluate(policy: str) -> float:
    return score(run_policy(policy))


if __name__ == "__main__":
    print("=== HARDENED evaluator (correctness GATE, then speed) ===")
    for name in ("honest", "evolved_v7"):
        res = run_policy(name)
        s = score(res)
        try:
            correctness_gate(res)
            gate = "PASS"
        except CorrectnessError as e:
            gate = f"REJECTED ({e})"
        print(f"  {name:12s} served={res.served:5d}/{res.submitted}  "
              f"gate={gate}")
        print(f"  {'':12s} score={s:8.1f}")
    print("\n  The hack now scores 0. The honest policy wins. Ranking flipped.")
