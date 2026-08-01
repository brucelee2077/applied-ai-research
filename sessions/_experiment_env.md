# What a lesson `experiment.py` can rely on

Written for whoever (human or agent) fills in a day's artifact. Everything below
was checked on this machine on 2026-07-27 — re-check before trusting it.

The acceptance gate is `python3 sessions/_experiment_check.py <path>`. It runs
the structural contract (`gates/experiment_contract.py`) and then **executes the
file**, requiring exit 0, a `✅` in stdout with no `❌`, no network, and no
timeout. Design to that, not to the contract alone.

## Hard rules

- **No network.** The gate injects a shim that makes socket connections raise, so
  a download does not fail gracefully — it fails loudly. This is deliberate: an
  artifact that downloads something is not reproducible for the reader.
- **No file writes, no `savefig`, no plot window.** `MPLBACKEND=Agg` is forced.
  ⚠️ Unlike the rules above and below, this one is **convention only — no gate checks it.**
  Neither `experiment_contract.py` nor `_experiment_check.py` inspects for writes or `savefig`,
  so nothing will catch you breaking it. (The portfolio/evidence artifact is deliberately
  exempt: it MAY `savefig` into its own `assets/`.)
- **Deterministic.** Seed everything (`np.random.default_rng(0)`,
  `torch.manual_seed(0)`). Running twice must print byte-identical output — the
  reviewer diffs two runs.
- **Fast.** Default timeout 180s; aim for under 10s. The reader will rerun this
  many times.
- **Every asserted number must be one you actually ran**, never one you expect.

## Libraries installed

`numpy 2.4.6` · `torch 2.8.0` · `jax 0.10.2` · `scipy 1.15.3` · `pandas 2.2.3`
`scikit-learn 1.6.1` · `matplotlib 3.10.1` · `transformers 4.52.4` · `tqdm 4.67.1`

Prefer **numpy** unless the day is specifically about another one.

**NOT installed, despite a day teaching them:** `flax` and `optax` both raise
`ModuleNotFoundError` (verified 2026-08-01) while `jax` itself is present — and
`m07-thinking-in-jax/day-05-flax-optax` specifies `nn.Dense` / `optax.sgd` /
`optax.apply_updates` throughout. **Check that a day's imports actually import before you write
its artifact.** Where a library is absent, implement the same mechanism in numpy/jax and declare
the substitution in a comment at the point of use; do not stub a fake API that appears to be the
real one.

## Hardware

- **No CUDA.** `torch.cuda.is_available()` is `False`. Never require a GPU, and
  never print a number that implies one was used.
- **MPS is available** (Apple Silicon). Fine to *mention*, but do not branch on
  it — the artifact must give the same answer on any machine.
- Days about GPU performance (rooflines, arithmetic intensity, memory bandwidth,
  FLOP counts, KV-cache size) are **arithmetic**, not benchmarks. Compute them
  from published spec numbers stated in the lesson and check the arithmetic. Do
  not time a kernel and pretend it is an H100.

## Models and data

- **No dataset is cached.** MNIST, CIFAR and friends are *not* on disk and cannot
  be fetched. If a day's produce step names one, **synthesize a small stand-in
  with a fixed seed** and say so in a comment — e.g. 8×8 digit-like blobs instead
  of MNIST. Keep the shapes the lesson talks about (`784 → 128 → 10`) so the
  reader still sees the numbers they were taught.
- **Some Hugging Face models ARE cached** and load with `HF_HUB_OFFLINE=1`:
  `gpt2`, `distilbert-base-uncased`, `sentence-transformers/all-MiniLM-L6-v2`,
  `sentence-transformers/all-mpnet-base-v2`, `BAAI/bge-{small,base,large}-en-v1.5`,
  `BAAI/bge-m3`, `Qwen/Qwen1.5-1.8B-Chat`, `Qwen/Qwen2.5-7B-Instruct`,
  `Qwen/Qwen3-Embedding-8B`, `colbert-ir/colbertv2.0`.
  `bert-base-uncased` is **not** cached — use `distilbert-base-uncased` instead.
  Set `os.environ["HF_HUB_OFFLINE"] = "1"` before importing, and wrap the load in
  a `try` that falls back to a tiny hand-built stand-in, so the artifact still
  runs for a reader without the cache.
- A tokenizer-only load (`AutoTokenizer.from_pretrained("gpt2")`) is cheap and is
  usually all a tokenization day needs.

## When the produce step asks for something impossible here

Do not silently drop it and do not fake it. Build the closest honest thing, and
add a short comment saying what changed and why — the reader is allowed to know
they are running a scaled-down version. Then say so in your report.
