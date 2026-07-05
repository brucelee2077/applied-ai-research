# Experiment Log: JAX immutability proof

## What was tried
Built `experiments/week01_jax/immutability_proof.py`, which proves NumPy
mutability vs JAX immutability in three self-checking steps:
1. NumPy: build `[1,2,3,4]`, edit element 0 in place, print values and `id()`
   before/after, and assert the `id()` is unchanged (same object, mutated).
2. JAX: build the same array, attempt `x[0] = 99`, catch the `TypeError`, and
   print the message.
3. JAX: `y = x.at[0].set(99)`, assert `x` is unchanged and `x is not y`, and
   print `x`, `y`, and both `id()`s.

Ran it with `python3 experiments/week01_jax/immutability_proof.py` and saved the
captured stdout + exit code to `results/run_output.txt`.

## What happened
Exit code **0**. All three demonstrations passed.
- Step 1: `id` was identical before and after (`4465932016`), values changed
  `[1 2 3 4]` -> `[99 2 3 4]`.
- Step 2: `x[0] = 99` raised `TypeError` with the message
  "JAX arrays are immutable and do not support in-place item assignment.
  Instead of x[idx] = y, use x = x.at[idx].set(y) ...".
- Step 3: `x` stayed `[1 2 3 4]`, `y` became `[99 2 3 4]`, the two `id()`s
  differed, and `x is not y` held.

Environment: system Python 3.11.8, numpy 2.4.6, jax 0.10.2.

## Interpretation
JAX enforces immutability at the type level, so the in-place idiom that works in
NumPy fails fast with a helpful message, and the functional `.at[].set()` idiom
returns a fresh array without touching the original. This is the reason JAX
transforms (`jit`, `grad`, `vmap`) can treat array inputs as side-effect-free.

## Limitations
- Only 1-D scalar item assignment is exercised (no slices, masks, `.add()`,
  or donated buffers under `jit`).
- Ran on the default CPU backend; GPU/TPU not exercised (behavior is identical).

## Next experiment
Inspect the jaxpr / lowered HLO of a `jit`-ed function doing several
`.at[].set()` calls to confirm the "copies" fuse away and no real copy remains
in the compiled program.

---

# Experiment Log: MLP weights from split PRNG keys (Day 2)

## What was tried
Built `experiments/week01_jax/mlp_prng.py`, a tiny `4 -> 8 -> 2` MLP in raw
`jax.numpy` whose weights are seeded from explicit, split PRNG keys:
1. `key = random.PRNGKey(0)`, then `split` into an input key + a params key.
2. params key `split` into one subkey per weight matrix; `W1` (4x8) and `W2`
   (8x2) drawn with `random.normal`; biases `b1`, `b2` initialized to zeros.
3. `forward(x) = tanh(x @ W1 + b1) @ W2 + b2`.
4. `run_once(0)` called twice; asserted the two outputs are bit-for-bit equal.
5. `demo_reuse_vs_split()` — drew two matrices from the same key (identical) and
   from split keys (different).

Ran with `python3 experiments/week01_jax/mlp_prng.py`, once inline and once more
in a separate process; saved stdout to `results/mlp_prng_output.txt`.

## What happened
Exit code **0**. All assertions passed.
- Input `x` (4,) = `[ 1.0040143 -0.9063372 -0.7481722 -1.1713669]`.
- Output (2,) = `[-1.8783567  1.8897852]` on both in-process runs from seed 0.
- A second, fresh process produced the identical output — reproducible across
  processes, not just within one.
- Same key twice -> identical draws (`True`); split first -> different draws
  (`True`).

Environment: system Python 3.11, jax 0.10.2, CPU backend.

## Interpretation
The key holds the entire random state, so `split` (not a hidden counter) is how
you move forward. One split per layer gives each weight matrix independent
randomness, and the fully deterministic seed -> split -> draw chain is what makes
the network's initialization reproducible on any machine — the property JAX
transforms rely on.

## Limitations
- One hidden layer, single input vector (batching via `vmap` is Day 3).
- Uses legacy `PRNGKey`; newer JAX prefers `jax.random.key`.
- Initialization only; no training step.

## Next experiment
Run this MLP over a batch with `jax.vmap` and check the batched outputs match a
plain Python loop over single examples (Day 3).
