# Experiment: Capstone environment check

## Question
Is the JAX/Flax/Optax stack ready for the Week 9 Addition Transformer capstone?

## Hypothesis
If `jax`, `flax`, and `optax` are installed correctly and a device is visible, a
minimal 2-layer Flax MLP should build and run a forward pass on a dummy input
with no exceptions. If any of the three libraries is missing, mismatched, or
misconfigured, this smoke test should fail loudly and immediately — cheaply,
today, instead of on Week 9 Monday when it would cost a day of the capstone.

## Minimal setup
The smallest version that tests this is not the Addition Transformer itself —
it's a 2-layer `flax.linen.Dense` MLP run on one dummy input. That is enough to
exercise every layer of the stack that the capstone depends on:

- `jax` — device discovery, PRNG keys, array ops
- `flax.linen` — module definition, `init`, `apply`
- `optax` — imported and version-checked (not yet exercised with a real optimizer step,
  since this check only needs to prove the stack imports and a forward pass runs)

## Implementation
- `src/smoke_test.py` — imports jax, flax, optax; prints `jax.devices()`; prints
  `flax.__version__` and `optax.__version__`; builds a 2-layer `flax.linen.Dense`
  MLP; runs one forward pass on a dummy `(1, 14)` one-hot-ish input array; prints
  the output shape. Exits with code 0 on success, non-zero on any exception.
- Run with: `python3 experiments/week08_jax/env_check/src/smoke_test.py`

## Results
See `results/run_output.txt` for the captured stdout of the actual run in this
environment, and the note at the bottom of that file if the run could not
complete (e.g. missing packages, blocked package installs).

## Interpretation
See the bottom of `results/run_output.txt` and `EXPERIMENT_LOG.md` for what the
captured run tells us about readiness for Week 9 Monday.

## Limitations
- This checks import + one forward pass, not training. It does not verify GPU/TPU
  performance, multi-device sharding, or the full Addition Transformer training loop —
  those are Week 9's job, not this weekend's.
- If a device other than CPU (GPU/TPU) is expected for the capstone, this script only
  confirms `jax.devices()` returns *something* — it does not fail if that something
  is a CPU-only fallback. Read the printed device list, don't just check the exit code.

## Next experiment
Once Week 9 Monday starts, extend this into a training-loop smoke test: one
`optax` optimizer step on the MLP, confirming gradients flow and a loss value
changes after one step.
