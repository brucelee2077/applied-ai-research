export const meta = {
  name: 'phase1-lessons',
  description: 'Generate the 17 remaining Phase 1 daily lesson pages (Weeks 1-4, Days 4-20) by cp-ing the Day-3 template and editing only content, then adversarially verify+fix each',
  phases: [{ title: 'Generate' }, { title: 'Verify' }],
}

const ROOT = (args && args.root) || '.'  // pass {root:"<abs repo path>"} if agents need absolute paths
const TEMPLATE = ROOT + '/sessions/week-01/day-03-vmap.html'

// Shared build instructions — the cp+edit method keeps CSS/JS byte-identical.
const INSTRUCTIONS = `You are generating ONE self-contained interactive HTML lesson for a JAX/ML-systems curriculum.

METHOD (follow exactly — do NOT regenerate the file from scratch):
1. Read the template: ${TEMPLATE}. Study its exact structure, CSS classes, and JS.
2. Copy it to the target path with Bash: cp "${TEMPLATE}" "<TARGET>".
3. Then use Edit calls to swap ONLY these regions (leave ALL CSS and ALL JS engine code — state/markDone/refresh/reset/stepper/quiz-render/copy — byte-identical):
   a. <title> ... </title>
   b. the top nav-title span text ("Frontier Lab · Week 1 Day 3" -> correct week/day)
   c. <body data-quest-id="w01-d03-vmap"> -> the new quest id
   d. the <header class="hero"> block (eyebrow, h1, lead, goal) for this topic
   e. all 7 <section> bodies: keep the SAME section ids (s1..s7) and data-sec names
      (what/intuition/play/why/code/quiz/produce) and the SAME badge classes — only change
      the visible text, the sec-badge label, the sec-h title, and sec-body content.
   f. the DEMOS object in the <script> — 3 entries keyed to the 3 playground buttons; keep the
      same HTML markup style (spans with classes prompt/kw/str/num/fn/ok/bad/dim/hl) and set
      each entry's {html, take}. Also update the 3 <button class="play-btn" data-demo="..."> labels
      and their data-demo keys to match your 3 DEMOS keys.
   g. the QS array in the <script> — exactly 4 questions, each {q, opts:[4], ans, fb}. ans is the
      0-based index of the correct option.
   h. the step-screen <span class="step-line" data-ln="N"> lines (the walkthrough code) and the
      initial "0 / NN" text in #stepprog to match the new line count.
   i. the <nav class="lesson-nav"> block: set prev href+title and next href+title as given below.
   j. the <div class="fin"> celebration text.
   k. the produce <section> including the <pre class="prompt-t" id="pp"> copy-prompt.

HARD RULES:
- English only. ZERO Chinese characters anywhere.
- Ground Zero: Section 1 ("What is it") must explain what the topic IS, why it exists, and how
  it relates to something the reader already knows, BEFORE any comparison/mechanism/code.
  Assume a capable programmer (Python/NumPy) with ZERO knowledge of this specific topic.
  Define every topic-specific term/tool/syntax on first use.
- Keep it self-contained: no new external CSS/JS (Google Fonts links are fine).
- The playground gate, stepper gate, and quiz gate rely on the untouched JS — keep the
  play-btn/data-demo, step-line, and QS structures intact so gating still works.
- Put the official doc link(s) in a small callout in Section 1 (labeled "Official docs (optional)").
- Keep the tone warm, concrete, bite-sized. Short paragraphs.

Write the final file to <TARGET>. Return {file, done, notes}.`

const GEN_SCHEMA = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    done: { type: 'boolean' },
    notes: { type: 'string' },
  },
  required: ['file', 'done'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    passed: { type: 'boolean' },
    issuesFound: { type: 'array', items: { type: 'string' } },
    issuesFixed: { type: 'array', items: { type: 'string' } },
    issuesRemaining: { type: 'array', items: { type: 'string' } },
  },
  required: ['file', 'passed', 'issuesFound', 'issuesFixed', 'issuesRemaining'],
}

// Each day: full spec. brief carries the technically-correct content core.
const DAYS = [
  {
    id: 'w01-d04', file: 'sessions/week-01/day-04-jit.html', qid: 'w01-d04-jit',
    navTitle: 'Frontier Lab · Week 1 Day 4', eyebrow: 'Phase 1 · Week 1 · Day 4 (Thu)',
    h1: 'jit<br>Compilation', prev: ['day-03-vmap.html', 'vmap: Vectorization'],
    next: ['day-05-flax-optax.html', 'Flax & Optax'],
    refs: 'https://docs.jax.dev/en/latest/jit-compilation.html',
    brief: `TOPIC: jax.jit — compiling a Python function to fast fused code via XLA.
WHAT-IS-IT: jax.jit takes a function and compiles it with XLA (JAX's compiler). The FIRST call "traces" the function (records the operations on abstract shapes), XLA compiles that into one fused, optimized program, and the result is cached keyed by input shape+dtype. Later calls with the same shapes reuse the compiled code -> big speedup. Ground zero: define "compiler", "trace", "fused". Relate to Days 1-3: jit only works safely because JAX functions are pure (immutable arrays, explicit keys).
ANALOGY: A chef the first time works out the optimal order of steps (trace+compile); after that they run the memorized routine at full speed. The prep is a one-time cost.
DEMOS (3): (1) uncompiled f(x): runs op-by-op, slow, dispatched from Python each op. (2) f_jit = jax.jit(f): FIRST call is slow (compiling), subsequent calls are much faster (show timing). (3) The retrace trap: calling with a NEW input shape triggers a recompile; and a print() inside a jitted fn only fires at TRACE time, not every call.
WALKTHROUGH code: import jax, jax.numpy as jnp, time; def f(x): return jnp.tanh(x @ x.T).sum(); x=jnp.ones((512,512)); f_jit=jax.jit(f); f_jit(x).block_until_ready() (compiles); then time many calls of f vs f_jit -> f_jit is far faster.
QUIZ (4, ans index): 1) What does jax.jit do? -> compiles the function with XLA and caches by input shape (ans: that option). 2) Why is the first jitted call slow? -> it traces + compiles once. 3) What triggers a recompile? -> calling with a new input shape/dtype. 4) Why must the function be pure for jit to be safe? -> no hidden side effects means XLA can trace/reorder/fuse safely (Days 1-2).
PRODUCE: jit the Day 2/3 MLP forward and benchmark jit vs no-jit (use block_until_ready for honest timing). Path: experiments/week01_jax/mlp_jit.py. Skill: frontier-experiment-lab.
GOTCHA to mention in the why-section: no Python side effects inside jit; Python control flow on traced values needs jax.lax.cond/scan.`,
  },
  {
    id: 'w01-d05', file: 'sessions/week-01/day-05-flax-optax.html', qid: 'w01-d05-flax',
    navTitle: 'Frontier Lab · Week 1 Day 5', eyebrow: 'Phase 1 · Week 1 · Day 5 (Fri)',
    h1: 'Flax & Optax<br>Neural Nets, Functionally', prev: ['day-04-jit.html', 'jit: Compilation'],
    next: ['../week-02/day-01-rooflines.html', 'Rooflines & Arithmetic Intensity'],
    refs: 'https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html · https://optax.readthedocs.io/en/latest/getting_started.html',
    brief: `TOPIC: Flax (neural-net library) and Optax (optimizer library) for JAX.
WHAT-IS-IT: Raw JAX makes you carry parameter dicts and optimizer state by hand. Flax gives you nn.Module: you define layers, call model.init(key, x) to get a params pytree, and model.apply(params, x) to run the forward pass. Optax gives you composable optimizers: optax.adam(lr) returns an object with init/update; you hold opt_state, compute grads with jax.grad, then optax.apply_updates. Ground zero: what Flax and Optax ARE and why they exist (manage state functionally, no hidden mutation). Relate: params/state are explicit because JAX is pure (Days 1-2).
ANALOGY: Raw JAX = cooking with loose ingredients you track by hand. Flax/Optax = a labeled kitchen where each module knows its own parts and the optimizer tracks its own state for you.
DEMOS (3): (1) raw JAX: a params dict you build and thread manually. (2) Flax: model = nn.Dense(8); params = model.init(key, x); y = model.apply(params, x) -> params is a pytree. (3) Optax: tx = optax.adam(1e-3); opt_state = tx.init(params); updates, opt_state = tx.update(grads, opt_state); params = optax.apply_updates(params, updates).
WALKTHROUGH code: define a tiny Flax MLP (nn.Dense(8) -> tanh -> nn.Dense(2)) via nn.Module or nn.Sequential; params = model.init(key, x); set tx = optax.adam(1e-3), opt_state = tx.init(params); one step: loss_fn -> grads = jax.grad(loss_fn)(params); updates, opt_state = tx.update(grads, opt_state); params = optax.apply_updates(params, updates).
QUIZ (4): 1) What does model.init(key, x) return? -> a params pytree. 2) What does Optax provide? -> composable gradient transformations / optimizers with init+update. 3) How are params updated? -> optax.apply_updates(params, updates). 4) Why keep params/state explicit rather than hidden? -> purity, so it composes with jit/grad.
PRODUCE: refactor the raw MLP into a Flax module + one Optax update step. Path: experiments/week01_jax/mlp_flax_optax.py. Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w02-d01', file: 'sessions/week-02/day-01-rooflines.html', qid: 'w02-d01-roofline',
    navTitle: 'Frontier Lab · Week 2 Day 1', eyebrow: 'Phase 1 · Week 2 · Day 1 (Mon)',
    h1: 'Rooflines<br>Arithmetic Intensity', prev: ['../week-01/day-05-flax-optax.html', 'Flax & Optax'],
    next: ['day-02-tpu-architecture.html', 'TPU Architecture'],
    refs: 'https://jax-ml.github.io/scaling-book/roofline/',
    brief: `TOPIC: The roofline model and arithmetic intensity — is an operation compute-bound or memory-bound?
WHAT-IS-IT: Hardware has two ceilings: peak compute (FLOP/s) and peak memory bandwidth (bytes/s). Arithmetic Intensity (AI) = FLOPs performed / bytes moved to/from memory. Achievable performance = min(peak_FLOPs, AI x bandwidth). Low AI -> memory-bound (starved for data); high AI -> compute-bound. The crossover is the "ridge point". Ground zero: define FLOP, memory bandwidth, AI.
ANALOGY: A kitchen. Chefs = compute; the delivery door = memory bandwidth. If you cook a lot per delivery (high AI), the chefs are the limit. If you barely cook per delivery (low AI), the door is the bottleneck no matter how many chefs you add.
DEMOS (3): (1) Matmul (M=N=K=1024): ~2*N^3 FLOPs vs a few N^2 bytes -> high AI -> compute-bound. (2) Elementwise add x+y: 1 FLOP per element but ~12 bytes moved -> AI ~ 0.08 -> memory-bound. (3) Show the roofline: as AI rises, achievable perf climbs along AI x BW until it hits the flat compute ceiling at the ridge point.
WALKTHROUGH code (numbers, no GPU needed): compute AI of a 1024^3 matmul (FLOPs=2*1024^3 approx 2.1e9, bytes approx 3*1024^2*4 approx 12.6e6, AI approx 170 -> compute-bound); compute AI of x+y for 1e6 floats (FLOPs=1e6, bytes=3e6*4=12e6, AI approx 0.08 -> memory-bound).
QUIZ (4): 1) Arithmetic intensity = ? -> FLOPs / bytes moved. 2) Memory-bound means? -> limited by bandwidth, not compute. 3) Roofline achievable perf = ? -> min(peak compute, AI x bandwidth). 4) Why is LLM decode memory-bound? -> low AI (little compute per weight byte loaded).
PRODUCE: a roofline calculator: given FLOPs, bytes, peak FLOP/s, bandwidth -> print AI, the bound, and achievable throughput. Path: experiments/week02/roofline_calc.py. Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w02-d02', file: 'sessions/week-02/day-02-tpu-architecture.html', qid: 'w02-d02-tpu',
    navTitle: 'Frontier Lab · Week 2 Day 2', eyebrow: 'Phase 1 · Week 2 · Day 2 (Tue)',
    h1: 'TPU Architecture<br>The Matrix Multiply Unit', prev: ['day-01-rooflines.html', 'Rooflines & Arithmetic Intensity'],
    next: ['day-03-sharding.html', 'Sharding Fundamentals'],
    refs: 'https://jax-ml.github.io/scaling-book/tpus/',
    brief: `TOPIC: How a TPU is built and why it differs from a GPU.
WHAT-IS-IT: A TPU (Tensor Processing Unit, Google's accelerator) is built around a systolic-array Matrix Multiply Unit (MXU): a grid of multiply-accumulate cells that data flows through, doing one huge matmul per pass extremely efficiently. A GPU instead has many general-purpose Streaming Multiprocessors (SMs) — flexible, but not purpose-built for one op. Both pair the compute with fast on-chip memory + HBM (high-bandwidth memory), and favor bf16 (a 16-bit float) for throughput. Ground zero: define MXU, systolic array, SM, HBM, bf16.
ANALOGY: The MXU is a purpose-built assembly line for multiplying matrices — inputs stream through a grid, partial sums accumulate as they flow. A GPU is a big room of flexible general workers. For the one job of matmul, the assembly line wins.
DEMOS (3): conceptual click-reveals (not a REPL): (1) MXU: show a 3x3 systolic grid; click to step data flowing through, partial sums accumulating. (2) GPU SM contrast: many independent flexible cores. (3) bf16 vs fp32: half the bytes -> more throughput + fits more in memory, at some precision cost.
WALKTHROUGH: step-by-step of how a small (2x2) matmul flows through a systolic array — inputs enter staggered, each cell multiplies+adds, results emerge. Keep it visual/conceptual in the step lines.
QUIZ (4): 1) What is the MXU? -> a systolic array that does matmuls. 2) MXU vs GPU SM? -> MXU is purpose-built for matmul; SMs are general. 3) Why bf16? -> fewer bytes, more throughput, fits more in memory. 4) Why do TPUs suit transformer training? -> the workload is dominated by big matmuls.
PRODUCE: write a short comparison note (MXU vs GPU SM, when each wins) AND a tiny matmul timing in JAX. Path: experiments/week02/tpu_vs_gpu.py (+ a short markdown note). Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w02-d03', file: 'sessions/week-02/day-03-sharding.html', qid: 'w02-d03-sharding',
    navTitle: 'Frontier Lab · Week 2 Day 3', eyebrow: 'Phase 1 · Week 2 · Day 3 (Wed)',
    h1: 'Sharding<br>Splitting Arrays Across Devices', prev: ['day-02-tpu-architecture.html', 'TPU Architecture'],
    next: ['day-04-multi-device.html', 'Multi-Device Execution'],
    refs: 'https://jax-ml.github.io/scaling-book/sharding/',
    brief: `TOPIC: Sharding arrays across devices, and the three collective operations.
WHAT-IS-IT: When an array or model is too big for one device, you SHARD it — split it into pieces, one per device. To compute across shards you need COLLECTIVES: AllGather (every device ends up with everyone's shards = a full copy), ReduceScatter (sum contributions across devices, each device keeps one slice of the sum), AllReduce (everyone ends up with the full summed result; = ReduceScatter then AllGather). Ground zero: define device, mesh, shard, collective.
ANALOGY: A group project. Split a big document among people (shard). AllGather = everyone photocopies everyone else's section so all hold the whole doc. ReduceScatter = everyone adds up tallies but each keeps only their portion of the total. AllReduce = everyone adds up and all end with the same grand total.
DEMOS (3): 4 devices each holding numbers. (1) AllGather: [a][b][c][d] -> every device now has [a,b,c,d]. (2) ReduceScatter: each device gets the SUM of one position. (3) AllReduce: every device gets the full elementwise sum (same on all).
WALKTHROUGH: 4 devices hold partial vectors; show the array on each device before and after AllReduce (they all converge to the summed vector). Note AllReduce = ReduceScatter + AllGather.
QUIZ (4): 1) What is sharding? -> splitting an array across devices. 2) AllReduce produces? -> every device holds the full summed result. 3) When do you need AllGather? -> to reassemble a full copy from shards. 4) Why is communication the scaling bottleneck? -> collectives move lots of data over the interconnect.
PRODUCE: a tiny numpy simulation of AllGather / ReduceScatter / AllReduce across 4 simulated devices, printing arrays before/after. Path: experiments/week02/collectives_sim.py. Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w02-d04', file: 'sessions/week-02/day-04-multi-device.html', qid: 'w02-d04-multidevice',
    navTitle: 'Frontier Lab · Week 2 Day 4', eyebrow: 'Phase 1 · Week 2 · Day 4 (Thu)',
    h1: 'Multi-Device Execution<br>pmap & SPMD', prev: ['day-03-sharding.html', 'Sharding Fundamentals'],
    next: ['day-05-memory-footprint.html', 'Memory Footprint'],
    refs: 'https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html',
    brief: `TOPIC: Running the same program across many devices with jax.pmap and jax.lax.pmean.
WHAT-IS-IT: SPMD = Single Program, Multiple Data: run the SAME function on every device, each over its own shard. jax.pmap maps a function across physical devices (it is vmap's cousin — vmap maps over a batch axis, pmap maps over devices). jax.lax.pmean averages a value across devices (used to average gradients in data-parallel training so all replicas agree). Ground zero: define SPMD; relate pmap to vmap (Day 3).
ANALOGY: pmap is vmap that maps across real machines instead of a batch dimension. Each machine runs the same recipe on its own ingredients; pmean is them agreeing on the average at the end.
DEMOS (3): (1) pmap(lambda x: x*2) over an array shaped (n_devices, ...) -> each device doubles its slice. (2) pmean: each device has a number; pmean gives all of them the mean. (3) data-parallel gradient sync: each device computes a grad on its shard, pmean averages them so every replica updates identically.
WALKTHROUGH: jax.pmap(lambda x: x**2)(shard_array); then a pmean example inside a pmapped fn (axis_name). Mention that on a single CPU you simulate N devices via XLA_FLAGS xla_force_host_platform_device_count.
QUIZ (4): 1) What does pmap do? -> runs a function across devices (SPMD). 2) How does pmap relate to vmap? -> same idea, mapped over devices not a batch axis. 3) What is pmean for? -> averaging (e.g. gradients) across devices. 4) What does SPMD mean? -> single program, multiple data.
PRODUCE: pmap the MLP forward across simulated devices and pmean a dummy gradient. Path: experiments/week02/pmap_mlp.py. Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w02-d05', file: 'sessions/week-02/day-05-memory-footprint.html', qid: 'w02-d05-memory',
    navTitle: 'Frontier Lab · Week 2 Day 5', eyebrow: 'Phase 1 · Week 2 · Day 5 (Fri)',
    h1: 'Memory Footprint<br>Where Your GPU RAM Goes', prev: ['day-04-multi-device.html', 'Multi-Device Execution'],
    next: ['../week-03/day-01-transformer-arithmetic.html', 'Transformer Arithmetic'],
    refs: 'https://huggingface.co/spaces/nanotron/ultrascale-playbook',
    brief: `TOPIC: The four things that consume memory during training, and the Adam 2x multiplier.
WHAT-IS-IT: Training memory = weights + gradients + optimizer state + activations. In fp32 (4 bytes/param): weights = 4N, gradients = 4N, and Adam's optimizer state = 2x params (first moment m + second moment v) = 8N. So the "static" cost is ~16N bytes for N params, BEFORE activations. Activations (saved for the backward pass) scale with batch x sequence x layers and are often the largest term. Ground zero: define weights/gradients/optimizer-state/activations and the Adam 2x.
ANALOGY: Packing for a trip. Clothes = weights. A mirror-image dirty-laundry bag = gradients. Two toiletry kits = Adam's two moments. Everything you unpack along the way = activations (often the bulkiest).
DEMOS (3): (1) weights only (4N). (2) + gradients + Adam state -> 16N. (3) + activations (batch-dependent), showing it can dwarf the rest.
WALKTHROUGH: a 1B-parameter model in fp32 -> 4GB weights + 4GB grads + 8GB Adam = 16GB before activations; note gradient checkpointing trades compute to shrink activation memory.
QUIZ (4): 1) What 4 things live in training memory? -> weights, grads, optimizer state, activations. 2) Adam's multiplier? -> 2x params (m and v). 3) Which term scales with batch/sequence? -> activations. 4) What does gradient checkpointing trade? -> recompute (more compute) for less activation memory.
PRODUCE: a memory calculator: input params, dtype bytes, batch/seq/layers -> print the full breakdown (weights/grads/optstate/activations). Path: experiments/week02/memory_calc.py. Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w03-d01', file: 'sessions/week-03/day-01-transformer-arithmetic.html', qid: 'w03-d01-arith',
    navTitle: 'Frontier Lab · Week 3 Day 1', eyebrow: 'Phase 1 · Week 3 · Day 1 (Mon)',
    h1: 'Transformer Arithmetic<br>C &asymp; 6ND', prev: ['../week-02/day-05-memory-footprint.html', 'Memory Footprint'],
    next: ['day-02-qkv-matrices.html', 'Q, K, V Matrix Anatomy'],
    refs: 'https://jax-ml.github.io/scaling-book/transformers/',
    brief: `TOPIC: The training-compute rule of thumb C approx 6 N D.
WHAT-IS-IT: To estimate the FLOPs to train a model: C approx 6 * N * D, where N = number of parameters and D = number of training tokens. The 6 breaks down as 2 FLOPs per parameter per token for the forward pass (a multiply + an add per weight), plus about 4 for the backward pass (backprop is roughly 2x the forward work). Ground zero: define FLOP; explain "per parameter per token".
ANALOGY: Every parameter "touches" every token with a fixed amount of arithmetic. Forward is one pass (2 ops), backward is roughly double (4 ops), so 6 total — like each worker (parameter) shaking hands with each guest (token) a fixed number of times.
DEMOS (3): (1) forward only = 2ND. (2) + backward (2x) -> 6ND. (3) plug real numbers: GPT-3 (N=175e9, D=300e9) -> C approx 6*175e9*300e9 = 3.15e23 FLOPs.
WALKTHROUGH: compute C for a small run (N=10e6, D=1e9): 6*10e6*1e9 = 6e16 FLOPs; then divide by an H100's ~1e15 FLOP/s (bf16, with utilization) to estimate wall-clock.
QUIZ (4): 1) The compute rule is? -> C approx 6ND. 2) Why 6? -> 2 forward + 4 backward FLOPs per param per token. 3) What are N and D? -> parameters and training tokens. 4) Double the tokens D -> compute? -> doubles.
PRODUCE: a training-FLOP + cost calculator (params, tokens -> FLOPs, and approx H100-hours). Path: experiments/week03/flops_calc.py. Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w03-d02', file: 'sessions/week-03/day-02-qkv-matrices.html', qid: 'w03-d02-qkv',
    navTitle: 'Frontier Lab · Week 3 Day 2', eyebrow: 'Phase 1 · Week 3 · Day 2 (Tue)',
    h1: 'Q, K, V<br>Matrix Anatomy', prev: ['day-01-transformer-arithmetic.html', 'Transformer Arithmetic'],
    next: ['day-03-kv-cache.html', 'The KV Cache'],
    refs: 'https://jax-ml.github.io/scaling-book/transformers/',
    brief: `TOPIC: The weight matrices inside a transformer block and how to count their parameters.
WHAT-IS-IT: A transformer block has two sub-parts. Attention has 4 weight matrices — Wq, Wk, Wv, Wo — each roughly d_model x d_model. The feed-forward network (FFN/MLP) has 2 matrices, typically d_model x (4*d_model) then (4*d_model) x d_model. So per layer, params approx 4*d^2 (attention) + 8*d^2 (FFN) = 12*d^2 (ignoring biases). Multi-head attention splits d_model into n_heads x head_dim (so n_heads * head_dim = d_model). Ground zero: what Q/K/V projections and the FFN do at the shape level (no need to re-derive attention math).
ANALOGY: Each token's vector gets reshaped by these matrices — Wq/Wk/Wv ask three different "questions" of it, Wo recombines the answer; the FFN is a wide-then-narrow funnel (expand 4x, then back down).
DEMOS (3): (1) attention param count: 4 matrices of d^2 -> 4*d^2. (2) FFN: d*(4d) + (4d)*d = 8*d^2. (3) per-layer total 12*d^2, then multiply by n_layers for the model.
WALKTHROUGH: for d_model=768, n_heads=12 (head_dim=64): Wq,Wk,Wv,Wo each 768x768 (=589,824 each); FFN 768x3072 and 3072x768; sum per layer approx 12*768^2 approx 7.08M params.
QUIZ (4): 1) How many big matrices in attention? -> 4 (Wq,Wk,Wv,Wo). 2) FFN hidden multiplier? -> about 4x d_model. 3) Per-layer param count approx? -> 12*d_model^2. 4) n_heads * head_dim = ? -> d_model.
PRODUCE: a parameter counter: given d_model, n_heads, n_layers -> per-block and full-model param counts. Path: experiments/week03/param_counter.py. Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w03-d03', file: 'sessions/week-03/day-03-kv-cache.html', qid: 'w03-d03-kv',
    navTitle: 'Frontier Lab · Week 3 Day 3', eyebrow: 'Phase 1 · Week 3 · Day 3 (Wed)',
    h1: 'The KV Cache<br>Why Decoding Gets Cheaper', prev: ['day-02-qkv-matrices.html', 'Q, K, V Matrix Anatomy'],
    next: ['day-04-production-code.html', 'Production Code Review'],
    refs: 'https://jax-ml.github.io/scaling-book/inference/ · https://arxiv.org/abs/2211.05102',
    brief: `TOPIC: The KV cache — storing past Keys and Values so autoregressive decoding avoids recompute.
WHAT-IS-IT: When a model generates text one token at a time (autoregressive decoding), each new token attends to the Keys (K) and Values (V) of ALL previous tokens. Without a cache you'd recompute every past token's K and V at every step — O(n^2) wasted work. The KV cache stores past K and V so each step only computes the new token's K,V and appends them. Cost: memory = 2 (K and V) x n_layers x n_heads x head_dim x seq_len x batch x bytes_per_value. This memory grows with sequence length and becomes the bottleneck for long context. Ground zero: define autoregressive decoding; why recompute is wasteful.
ANALOGY: Writing a story where each new word must reconsider every previous word. Without a cache you re-read the whole story for each word (O(n^2)); with a cache you keep notes (K,V) for what you've written and only add the new word's note.
DEMOS (3): (1) recompute-every-step: work grows with the whole prefix each step (wasteful). (2) with cache: only the NEW token's K,V is computed; past ones are read from the cache. (3) cache memory growth: it rises linearly with each decoded token / show the 7B number.
WALKTHROUGH: compute KV cache size for a 7B-style model: n_layers=32, n_heads=32, head_dim=128, seq=8192, batch=1, fp16=2 bytes -> 2*32*32*128 = 262,144 values/token; *8192 tokens = 2.15e9; *2 bytes = approx 4.3 GB.
QUIZ (4): 1) What does the KV cache store? -> past tokens' Keys and Values. 2) Why does it save compute? -> avoids recomputing K,V every step (O(n^2) -> O(n)). 3) What does it cost? -> memory that grows with seq_len x layers x heads. 4) Consequence for decode? -> it becomes memory-bandwidth-bound (ties to the roofline, Week 2).
PRODUCE: a KV-cache memory model: given layers/heads/head_dim/seq/batch/dtype -> cache size, and print growth over decode steps. Path: experiments/week03/kv_cache_model.py. Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w03-d04', file: 'sessions/week-03/day-04-production-code.html', qid: 'w03-d04-prodcode',
    navTitle: 'Frontier Lab · Week 3 Day 4', eyebrow: 'Phase 1 · Week 3 · Day 4 (Thu)',
    h1: 'Production Code Review<br>Anatomy of a Training Loop', prev: ['day-03-kv-cache.html', 'The KV Cache'],
    next: ['day-05-pretraining-logic.html', 'Advanced Pretraining Logic'],
    refs: 'https://github.com/huggingface/nanotron · https://github.com/huggingface/picotron',
    brief: `TOPIC: Tracing a real distributed training loop (nanotron / picotron) with the abstractions stripped away.
WHAT-IS-IT: Production training frameworks like nanotron/picotron deliberately expose every step of a training iteration instead of hiding it behind a .fit() call. One training STEP is: (1) load + shard a batch, (2) forward pass, (3) compute loss, (4) backward pass (gradients), (5) synchronize gradients across devices (all-reduce, Week 2), (6) optimizer step (update weights), and periodically (7) checkpoint. Ground zero: what a training loop is; what "without abstractions" means and why it's worth reading.
ANALOGY: Reading the factory blueprint instead of pressing the big "make product" button — you trace each pipe and gear: data in -> forward -> loss -> backward -> sync -> update -> (save).
DEMOS (3, click-reveal of a pseudo training step): (1) the 6 stages in order. (2) WHERE the cross-device gradient all-reduce happens (between backward and optimizer step). (3) where checkpointing fits (periodically, to resume after crashes).
WALKTHROUGH: annotated pseudo-code of ONE distributed training step, revealing each stage line by line (batch -> logits -> loss -> grads -> all_reduce(grads) -> optimizer.step -> maybe save).
QUIZ (4): 1) The stages of one training step? -> forward, loss, backward, sync, update (in order). 2) Where does gradient sync happen? -> after backward, before the optimizer step. 3) Why read code without abstractions? -> to see and control every cost. 4) What does checkpointing protect against? -> crashes / enables resume.
PRODUCE: build a minimal single-file training loop (no framework) and annotate each stage, OR trace nanotron's step and write the mapping. Path: experiments/week03/training_loop.py (+ notes). Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w03-d05', file: 'sessions/week-03/day-05-pretraining-logic.html', qid: 'w03-d05-pretrain',
    navTitle: 'Frontier Lab · Week 3 Day 5', eyebrow: 'Phase 1 · Week 3 · Day 5 (Fri)',
    h1: 'Advanced Pretraining Logic<br>What Actually Shapes a Model', prev: ['day-04-production-code.html', 'Production Code Review'],
    next: ['../week-04/day-01-data-parallel-fsdp.html', 'Data Parallelism & FSDP'],
    refs: 'https://vladfeinberg.com/2025/04/24/gemini-flash-pretraining.html',
    brief: `TOPIC: How real pretraining decisions are made (from Vlad Feinberg's Gemini Flash pretraining notes / Princeton COS 568).
WHAT-IS-IT: "Pretraining" is the initial large-scale training of a base model on huge text corpora. Which model you actually train is decided less by elegance and more by three intersecting constraints: scaling laws (compute-optimal size vs data), data availability/quality, and systems limits (memory, bandwidth, interconnect). This is a synthesis day tying together Weeks 1-3. Ground zero: define pretraining; explain the three constraint families before the insights.
ANALOGY: Pretraining is planning a massive expedition — the route (architecture, size, token budget) is chosen not for beauty but for what the terrain (hardware, data, budget) allows.
DEMOS (3, click-reveal): (1) compute-optimal vs practical: the Chinchilla-optimal point vs what you actually train given serving cost. (2) the data constraint: high-quality tokens are finite. (3) the systems constraint: memory + bandwidth shape feasible model shapes (ties to Weeks 1-2).
WALKTHROUGH: reveal 4-5 key insights from the Feinberg talk one at a time (e.g., inference cost pushes you to train smaller-but-longer than compute-optimal; data quality caps returns; systems constraints pick the architecture).
QUIZ (4): 1) What shapes pretraining choices? -> scaling laws + data + systems, not elegance. 2) Why train smaller than compute-optimal sometimes? -> to cut inference/serving cost. 3) What caps data returns? -> finite high-quality tokens. 4) How do systems constrain choices? -> memory/bandwidth limit feasible shapes.
PRODUCE: read the Feinberg blog/talk and write a notes summary + one insight to remember. Path: notes/papers/feinberg-gemini-flash-pretraining.md. Skill: frontier-paper-course.`,
  },
  {
    id: 'w04-d01', file: 'sessions/week-04/day-01-data-parallel-fsdp.html', qid: 'w04-d01-fsdp',
    navTitle: 'Frontier Lab · Week 4 Day 1', eyebrow: 'Phase 1 · Week 4 · Day 1 (Mon)',
    h1: 'Data Parallelism & FSDP<br>Replicate vs Shard', prev: ['../week-03/day-05-pretraining-logic.html', 'Advanced Pretraining Logic'],
    next: ['day-02-tensor-parallel.html', 'Tensor Parallelism'],
    refs: 'https://arxiv.org/abs/1910.02054 · https://jax-ml.github.io/scaling-book/training/',
    brief: `TOPIC: Data Parallelism (DP) vs Fully Sharded Data Parallel (FSDP / ZeRO-3).
WHAT-IS-IT: Data Parallel: put a FULL COPY of the model on every device, split the batch across devices, and all-reduce the gradients so all copies stay in sync. Simple, but every device stores the full 16N (weights+grads+Adam, Week 2) -> memory-heavy. FSDP (a.k.a. ZeRO stage 3): SHARD the params, gradients, and optimizer state across devices so no device holds a full copy; each layer's params are all-gathered just-in-time for its forward/backward, then discarded. Ground zero: what DP is; what "sharded" means here (ties to Day w02-d03 sharding + w02-d05 memory).
ANALOGY: DP = every chef owns the entire cookbook (wasteful copies). FSDP = the cookbook is split among chefs, and they pass around just the one page needed for the dish being cooked right now.
DEMOS (3): (1) DP: full replica per device, all-reduce gradients each step. (2) FSDP: params sharded; gather one layer's params, use, discard. (3) memory comparison: DP holds 16N per device; FSDP holds about 16N / num_devices per device.
WALKTHROUGH: for a model needing 16N bytes, 8 devices: DP = 16N on each; FSDP = 2N on each (plus transient gathered layer). Cost of FSDP: more communication (gathers every layer).
QUIZ (4): 1) DP replicates what, splits what? -> replicates the model, splits the batch. 2) FSDP/ZeRO-3 shards what? -> params + gradients + optimizer state. 3) The memory win of FSDP? -> ~divide the per-device static memory by num_devices. 4) The cost? -> extra communication (param all-gathers).
PRODUCE: compute + diagram DP vs FSDP per-device memory for a given model size and device count. Path: experiments/week04/dp_vs_fsdp.py. Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w04-d02', file: 'sessions/week-04/day-02-tensor-parallel.html', qid: 'w04-d02-tp',
    navTitle: 'Frontier Lab · Week 4 Day 2', eyebrow: 'Phase 1 · Week 4 · Day 2 (Tue)',
    h1: 'Tensor Parallelism<br>Splitting a Single Matmul', prev: ['day-01-data-parallel-fsdp.html', 'Data Parallelism & FSDP'],
    next: ['day-03-pipeline-parallel.html', 'Pipeline Parallelism'],
    refs: 'https://jax-ml.github.io/scaling-book/training/',
    brief: `TOPIC: Tensor Parallelism (TP) — splitting one big matmul across devices.
WHAT-IS-IT: When a single weight matrix is too big or a matmul too slow for one device, split the matmul itself. Column-parallel: split the weight's COLUMNS across devices; each device computes part of the output; concatenate the pieces. Row-parallel: split the weight's ROWS; each device computes a partial sum; all-reduce to combine. Transformers pair a column-parallel layer followed by a row-parallel layer (in both attention and the MLP) so only ONE all-reduce is needed per block. Ground zero: what it means to split a matmul; contrast with DP (which replicates the whole model).
ANALOGY: A giant multiplication done by a team. Column-parallel: each person computes some output columns, you lay them side by side. Row-parallel: each computes partial sums that must be added up (the all-reduce) to get the true total.
DEMOS (3): (1) column-parallel: split W by columns -> concat outputs. (2) row-parallel: split W by rows -> all-reduce partial sums. (3) the transformer block's column-then-row pattern needing one all-reduce.
WALKTHROUGH: Y = X @ W with W split by columns across 2 devices: device0 computes X@W[:,:half], device1 computes X@W[:,half:]; concatenate -> equals single-device Y.
QUIZ (4): 1) What does TP split? -> a single matmul/weight matrix across devices. 2) Column vs row parallel? -> column concatenates outputs; row all-reduces partial sums. 3) Where is the all-reduce in a block? -> after the row-parallel layer. 4) Why does TP need a fast interconnect? -> it communicates within every layer.
PRODUCE: simulate column- and row-parallel matmul across 2 "devices" and assert it equals the single-device result. Path: experiments/week04/tensor_parallel_sim.py. Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w04-d03', file: 'sessions/week-04/day-03-pipeline-parallel.html', qid: 'w04-d03-pp',
    navTitle: 'Frontier Lab · Week 4 Day 3', eyebrow: 'Phase 1 · Week 4 · Day 3 (Wed)',
    h1: 'Pipeline Parallelism<br>Layers Across Devices & the Bubble', prev: ['day-02-tensor-parallel.html', 'Tensor Parallelism'],
    next: ['day-04-llama3-tpu.html', 'Training LLaMA 3 on TPUs'],
    refs: 'https://arxiv.org/abs/1811.06965',
    brief: `TOPIC: Pipeline Parallelism (PP) — splitting the model's layers across devices, and the "bubble".
WHAT-IS-IT: Put different LAYERS on different devices: device 1 holds layers 1..k, device 2 holds k+1.., etc. A batch flows through the pipeline stage by stage. Problem: the BUBBLE — while device 1 works on the first layers, later devices sit idle waiting; and at the end the early devices idle. Micro-batching (splitting the batch into many small micro-batches fed back-to-back) keeps the pipeline full and shrinks the bubble. Bubble fraction approx (p-1)/(m+p-1) for p stages and m micro-batches. Ground zero: what a pipeline stage is; what idle "bubble" time is.
ANALOGY: An assembly line. Until the first product reaches the last station, the later workers stand idle (the bubble). Feed many small products (micro-batches) back-to-back and everyone stays busy.
DEMOS (3): (1) naive pipeline (m=1): huge bubble, most devices idle. (2) micro-batching (m large): bubble shrinks, high utilization. (3) the bubble-fraction formula: change p and m, watch the idle fraction drop.
WALKTHROUGH: p=4 stages: m=1 -> bubble = 3/4 = 75% idle; m=8 -> bubble = 3/11 approx 27%; m=32 -> approx 9%.
QUIZ (4): 1) What does PP split? -> the model's layers across devices. 2) What is the bubble? -> idle time while stages wait for the pipeline to fill/drain. 3) How does micro-batching help? -> keeps stages busy, shrinking the bubble. 4) Bubble fraction with more micro-batches? -> gets smaller.
PRODUCE: a bubble-fraction calculator/plot over varying micro-batch counts (and stages). Path: experiments/week04/pipeline_bubble.py. Skill: frontier-experiment-lab.`,
  },
  {
    id: 'w04-d04', file: 'sessions/week-04/day-04-llama3-tpu.html', qid: 'w04-d04-llama3',
    navTitle: 'Frontier Lab · Week 4 Day 4', eyebrow: 'Phase 1 · Week 4 · Day 4 (Thu)',
    h1: 'Training LLaMA 3 on TPUs<br>Combining the Parallelisms', prev: ['day-03-pipeline-parallel.html', 'Pipeline Parallelism'],
    next: ['day-05-distillation.html', 'Distillation'],
    refs: 'https://jax-ml.github.io/scaling-book/llama3/',
    brief: `TOPIC: A real case study — interleaving DP/FSDP, TP, and PP to train 8B and 70B models on TPU pods.
WHAT-IS-IT: No single parallelism is enough at scale; real training COMBINES them. This day recaps the four (data parallel/FSDP for splitting the batch + sharding memory, tensor parallel for splitting big matmuls, pipeline parallel for splitting layers) and shows how a real config assigns each to the bottleneck it solves. Ground zero: brief recap of the 4 parallelisms (Days w04-d01..d03 + w02-d04) before analyzing how they compose.
ANALOGY: An orchestra — each parallelism is a section; the conductor (the training config) blends them so no single resource (compute, memory, or interconnect bandwidth) is the sole bottleneck.
DEMOS (3): (1) an 8B config: which parallelisms and why. (2) a 70B config: more sharding/TP/PP. (3) which bottleneck each parallelism solves (FSDP->memory, TP->big-matmul latency, PP->fit huge models, DP->throughput).
WALKTHROUGH: reveal a realistic plan: FSDP to shard optimizer/params for memory, TP within a fast-interconnect node, PP across nodes, DP on top for throughput — one line at a time.
QUIZ (4): 1) Which parallelism mainly saves memory? -> FSDP/ZeRO. 2) Which needs the fastest interconnect (per-layer comms)? -> tensor parallel. 3) Which introduces a bubble? -> pipeline parallel. 4) Why interleave them? -> each handles a different bottleneck; combined they scale.
PRODUCE: read Scaling Book ch.6 and write an analysis of the parallelization plan for an 8B and 70B model on a given topology. Path: notes/papers/llama3-on-tpus.md. Skill: frontier-paper-course.`,
  },
  {
    id: 'w04-d05', file: 'sessions/week-04/day-05-distillation.html', qid: 'w04-d05-distill',
    navTitle: 'Frontier Lab · Week 4 Day 5', eyebrow: 'Phase 1 · Week 4 · Day 5 (Fri)',
    h1: 'Distillation<br>A Small Model That Learned From a Big One', prev: ['day-04-llama3-tpu.html', 'Training LLaMA 3 on TPUs'],
    next: ['../index.html', 'Phase 1 complete → map'],
    refs: 'https://arxiv.org/abs/1503.02531',
    brief: `TOPIC: Knowledge distillation — training a small "student" to mimic a large "teacher".
WHAT-IS-IT: Distillation trains a small student model to copy a large teacher model's OUTPUT DISTRIBUTION, not just the hard labels. The teacher's softmax is softened with a TEMPERATURE T (higher T -> softer, more spread-out probabilities), exposing "dark knowledge" — the relative probabilities among wrong answers (e.g. a "3" looks a bit like an "8"). The student trains to match these soft targets (plus optionally the true labels). Result: a much smaller model that keeps a lot of the teacher's quality, so inference stays cheap/fast. Ground zero: define teacher, student, soft targets, temperature.
ANALOGY: An expert tutor doesn't just give the answer — they show how confident they are across options ("mostly A, but B is plausible, definitely not C"). The student absorbs that nuance, learning more than a bare answer key would teach.
DEMOS (3): (1) hard labels: a one-hot target (all info except the right class thrown away). (2) soft targets: the teacher's temperature-softened probabilities (rich relative info). (3) the distillation loss = KL(student_soft || teacher_soft) [+ optional cross-entropy on true labels].
WALKTHROUGH: softmax with temperature T on teacher logits -> soft targets; student produces its own softened probs; loss pushes them together. Show that higher T spreads the probabilities.
QUIZ (4): 1) What does distillation transfer? -> the teacher's soft output distribution (dark knowledge), not just hard labels. 2) Role of temperature T? -> softens the probabilities to expose relative info. 3) Teacher vs student? -> large accurate teacher; small cheap student that mimics it. 4) Why does it keep inference cheap? -> the deployed student is small.
PRODUCE: a toy distillation — soften teacher logits with temperature T and compute the KL distillation loss against a student's outputs. Path: experiments/week04/distillation_toy.py. Skill: frontier-experiment-lab.`,
  },
]

function genPrompt(d) {
  var prevLink = d.prev[0] === null ? 'DISABLED (this is the first lesson)' : (d.prev[0] + '  title: ' + d.prev[1])
  return INSTRUCTIONS +
`

======= THIS LESSON =======
TARGET FILE (write here): ${ROOT}/${d.file}
data-quest-id: ${d.qid}
nav-title (top bar): ${d.navTitle}
hero eyebrow: ${d.eyebrow}
hero <h1> (may contain <br>): ${d.h1}
lesson-nav PREV: href="${d.prev[0]}"  title="${d.prev[1]}"
lesson-nav NEXT: href="${d.next[0]}"  title="${d.next[1]}"
official docs link(s) for Section 1: ${d.refs}

CONTENT BRIEF (author the lesson from this; it is technically correct — do not contradict it):
${d.brief}

Remember: cp the template first, then Edit only the content regions. Keep all CSS + JS engine byte-identical. English only. Section 1 = "What is it" (ground zero). 7 sections. Self-contained.`
}

function verifyPrompt(d) {
  return `Adversarially verify (and FIX in place) the generated lesson at ${ROOT}/${d.file}.

Use Bash/Read to check each item; use Edit to fix any failure directly in the file:
1. SELF-CONTAINED: no external <link rel=stylesheet> or <script src> except the Google Fonts links. (inline <style> and <script> only)
2. ENGLISH ONLY: zero CJK characters. (grep for them)
3. STRUCTURE: exactly 7 <section class="sec"> with ids s1..s7 and data-sec = what,intuition,play,why,code,quiz,produce (order matters).
4. ENGINE HOOKS present and intact: id="bar", id="count", id="reset", id="console", id="take", id="screen", id="quiz", id="fin", id="pp"; <body data-quest-id="${d.qid}">.
5. PLAYGROUND: exactly 3 <button class="play-btn" data-demo="..."> and a DEMOS object whose keys match those 3 data-demo values (3 entries).
6. QUIZ: the QS array has exactly 4 questions, each with 4 opts and a valid ans index (0-3) that matches the intended correct answer in the brief.
7. STEPPER: the #stepprog initial text "0 / N" equals the number of <span class="step-line"> elements.
8. NAV: <nav class="lesson-nav"> present with prev href="${d.prev[0]}" and next href="${d.next[0]}".
9. NO TEMPLATE LEFTOVERS: the file must NOT still contain vmap-specific content from the template (search for "vmap", "vectoriz", "arange(12)", "sum(x ** 2)") unless this lesson is genuinely about that. If found, they are leftovers -> fix them to match this lesson's topic.
10. TAG BALANCE: <section>, <script>, <style> tags balanced.
11. TECHNICAL ACCURACY: the concept matches the brief (spot-check the key formula/fact). Fix errors.

The lesson's topic and correct content are defined by this brief:
${d.brief}

Fix everything you can with Edit. Return {file, passed, issuesFound, issuesFixed, issuesRemaining}.`
}

phase('Generate')

const results = await pipeline(
  DAYS,
  (d) => agent(genPrompt(d), { label: 'gen:' + d.id, phase: 'Generate', schema: GEN_SCHEMA })
           .then((r) => ({ day: d, gen: r })),
  (prev) => {
    if (!prev) return null
    const d = prev.day
    return agent(verifyPrompt(d), { label: 'verify:' + d.id, phase: 'Verify', schema: VERIFY_SCHEMA })
             .then((v) => ({ id: d.id, file: d.file, gen: prev.gen, verify: v }))
  }
)

const clean = results.filter(Boolean)
const passed = clean.filter((r) => r.verify && r.verify.passed)
const withRemaining = clean.filter((r) => r.verify && r.verify.issuesRemaining && r.verify.issuesRemaining.length)

log(`Generated ${clean.length}/${DAYS.length}; verified-clean ${passed.length}; with remaining issues ${withRemaining.length}`)

return {
  total: DAYS.length,
  generated: clean.map((r) => r.id),
  passed: passed.map((r) => r.id),
  remaining: withRemaining.map((r) => ({ id: r.id, issues: r.verify.issuesRemaining })),
}
