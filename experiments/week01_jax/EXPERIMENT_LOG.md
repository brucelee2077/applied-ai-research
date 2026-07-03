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
