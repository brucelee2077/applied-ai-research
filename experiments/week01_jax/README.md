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

---

# Experiment: MLP weights from split PRNG keys (Week 1, Day 2)

## Question
Can I seed a tiny neural network's weights from explicit JAX keys so that (a) each
layer gets *independent* randomness and (b) the whole forward pass is exactly
reproducible from one seed?

## Hypothesis
- JAX randomness is an explicit `key`, not hidden global state. The same key
  always yields the same numbers.
- `jax.random.split(key, n)` deterministically makes `n` independent keys. Give
  one key per layer and the layers never share random values.
- Because the whole chain (seed -> split -> draw) is deterministic, re-running
  from `PRNGKey(0)` reproduces the output bit-for-bit — even in a fresh process.

## Minimal setup
One script, no frameworks (raw `jax.numpy`), a tiny MLP `4 -> 8 -> 2`:
1. `key = random.PRNGKey(0)` -> `split` into an input key + a params key.
2. params key -> `split` into one subkey per weight matrix; draw `W1` (4x8) and
   `W2` (8x2) with `random.normal`; biases start at zero (no key needed).
3. `forward(x) = tanh(x @ W1 + b1) @ W2 + b2`.
4. Run the full pipeline twice from seed 0 and assert identical output.
5. Bonus check: two draws from the *same* key are identical (the bug); two draws
   from split keys differ (the fix).

## Implementation
- `mlp_prng.py` — the whole experiment.

Run it:

```bash
python3 experiments/week01_jax/mlp_prng.py
```

Exit code `0` means the layers used independent keys **and** the forward pass was
reproducible. Any failed assertion exits non-zero.

## Results
See [`results/mlp_prng_output.txt`](./results/mlp_prng_output.txt). From the run
(jax 0.10.2, CPU):
- Input `x` (4,) = `[ 1.0040143 -0.9063372 -0.7481722 -1.1713669]`.
- Output (2,) = `[-1.8783567  1.8897852]` on **both** runs from seed 0 — identical.
- Running the script again in a **separate process** gave the exact same output.
- `same key twice -> identical draws? True` (the correlated-randomness bug).
- `split first -> different draws? True` (independent randomness, fixed).

## Interpretation
The key *is* the random state — there is no hidden counter. Splitting is the only
correct way to advance it, and one split per layer guarantees independent weights.
Determinism is not a lucky side effect: it is the whole design, which is exactly
what lets JAX `jit`/`vmap`/`pmap` random code and still reproduce results on any
machine.

## Limitations
- Single hidden layer, single input vector (no batch — that is Day 3 with `vmap`).
- `PRNGKey(0)` is the legacy key API; newer JAX prefers `jax.random.key(0)`.
- No training — weights are only initialized, never updated.

## Next experiment
Take this same MLP and run it on a whole batch of inputs at once with `jax.vmap`,
confirming the per-example outputs match a manual Python loop (Week 1, Day 3).
