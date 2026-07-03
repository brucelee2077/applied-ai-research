# Experiment Log: Capstone environment check

## What was tried
Built `experiments/week08_jax/env_check/src/smoke_test.py`, a script that:
1. Imports `jax`, `flax`, `optax`.
2. Prints `jax.devices()`.
3. Prints `flax.__version__` and `optax.__version__`.
4. Builds a 2-layer `flax.linen.Dense` MLP (32 units -> ReLU -> 8 units).
5. Runs one forward pass on a dummy `(1, 14)` array and prints the output shape.
6. Exits 0 on success, 1 on any import failure or missing device.

Ran it with `python3 experiments/week08_jax/env_check/src/smoke_test.py` in this
sandbox, and wrote the exact captured stdout + exit code to `results/run_output.txt`.

## What happened
`jax`, `flax`, and `optax` are not installed in this sandbox, and `pip install`
is proxy-blocked (no route to PyPI, and the local package mirror on
`localhost:11211` does not serve Python packages — that endpoint is an LLM
gateway, not a PyPI mirror). The script correctly detected this and exited 1
with a clear `ImportError` message rather than crashing unhelpfully or
silently succeeding.

## Interpretation
The falsifiable exit criterion from the brief — exit code 0, a printed device,
non-empty flax/optax version strings, and a printed forward-pass shape — is
**not met in this sandbox**, and cannot be met here because the required
packages cannot be installed in this environment. This is not a defect in the
smoke test: the test is doing exactly its job, which is to fail loudly and
specifically when the stack is not ready, rather than let a real problem slip
through unnoticed to Week 9 Monday.

The actionable conclusion for the learner: **do not treat this sandbox run as
proof the capstone environment is ready.** Re-run
`python3 experiments/week08_jax/env_check/src/smoke_test.py` on the actual
machine that will be used for the Week 9 capstone (after
`pip install jax flax optax` there), and only proceed to Week 9 Monday once
that real run exits 0.

## Limitations
- This sandbox has no working `pip`/PyPI access, so the "green path" of this
  experiment (stack installed, forward pass runs) was not directly observed
  here — only the "stack missing" failure path was observed and verified correct.
- The script does not test training (`optax` optimizer step), multi-device
  sharding, or actual TPU/GPU behavior — only import + one CPU-or-whatever-is-
  available forward pass.

## Next experiment
Once run successfully on a real dev machine with the packages installed,
extend this into a one-optimizer-step smoke test (loss goes down after one
`optax.adam` step) as the Week 9 Monday morning gate.
