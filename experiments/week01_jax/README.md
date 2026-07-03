# Experiment: JAX immutability proof (Week 1, Day 1)

## Question
Do NumPy and JAX arrays behave the same way when you edit an element in place?

## Hypothesis
- NumPy edits in place: the values change, but it is the *same* object.
- JAX arrays are immutable: `x[0] = 99` fails, and the only way to "change" an
  element is `x.at[0].set(99)`, which returns a *new* array and leaves the
  original untouched.

## Minimal setup
One script, no external data, three self-checking steps:
1. NumPy array — edit element 0 in place, compare `id()` before/after (must match).
2. JAX array — try `x[0] = 99`, catch the `TypeError`, print its message.
3. JAX array — `y = x.at[0].set(99)`, assert `x` unchanged and `x is not y`.

## Implementation
- `immutability_proof.py` — the whole experiment.

Run it:

```bash
python3 experiments/week01_jax/immutability_proof.py
```

Exit code `0` means all three demonstrations behaved as expected. Any failed
assertion (or an uncaught error) makes it exit non-zero.

## Results
See [`results/run_output.txt`](./results/run_output.txt). Highlights from the run
(numpy 2.4.6, jax 0.10.2):
- NumPy: `id` identical before and after (`4465932016` -> `4465932016`) while
  values went `[1 2 3 4]` -> `[99 2 3 4]`. Same object, mutated.
- JAX in-place edit raised:
  `JAX arrays are immutable and do not support in-place item assignment.
  Instead of x[idx] = y, use x = x.at[idx].set(y) ...`
- JAX `.at[0].set(99)`: `x` stayed `[1 2 3 4]`, `y` became `[99 2 3 4]`, and the
  two `id()`s differed — a genuinely new array.

## Interpretation
Immutability is not a style preference in JAX — it is enforced by the type
itself. NumPy's `arr[0] = 99` and JAX's `x.at[0].set(99)` look like they do the
same thing, but one *mutates* and the other *returns a copy*. This is the
foundation for why JAX code is safe to `jit`, `vmap`, and `grad`: functions have
no hidden side effects on their array inputs.

## Limitations
- Only tests scalar item assignment on a 1-D array. Does not cover slices,
  `.at[].add()`/`.mul()`, boolean masks, or donated buffers inside `jit`.
- Uses whatever JAX backend is present (CPU here); behavior is identical on
  GPU/TPU, but that was not exercised.

## Next experiment
Show that `.at[]` updates fuse away under `jit` (no real copy in the compiled
program) by inspecting the jaxpr / lowered HLO for a function that does several
`.at[].set()` calls in a row.
