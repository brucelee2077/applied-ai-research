# Roadmap — The Capability Spiral

> **Goal:** become a **staff researcher at a frontier AI research lab.**
> **Shape:** one integrated plan that takes a total beginner from "what is an array?" to the artifacts a frontier lab hires on — a from-scratch JAX transformer, your own scaling law, custom GPU kernels, an autonomous research loop, and a portfolio of **20 system-design case studies (plus 2 design frameworks)**.
> **This is a single arc, not layers.** Foundations, the frontier curriculum, and the design case studies are woven together. Each module builds only on earlier ones.

> **This file is the single source of truth.** It merges the former narrative roadmap, the former `sessions/MANIFEST.md` build tracker, and the build-execution queue into one document: the *why* (rationale, dependencies, risks), the *what/where/status* (source, action, files, build status), **and** the *how* (the [Build queue](#build-queue--execution--divide-and-conquer): protocol, divide-and-conquer rules, work packages). `sessions/MANIFEST.md` is now a stub that points here.

---

## How this plan works

The plan threads everything through **8 capabilities you unlock in order**:

**REPRESENT → TRAIN → SCALE → SERVE → OPTIMIZE → GENERATE → AUTOMATE → COMMUNICATE.**

- Each capability is **introduced** with a foundations concept, **deepened** with its matching frontier topic, and **cemented** with a design case study you could not build without the skill you just learned. Design breadth is the *applied checkpoint* of a capability — never a parked add-on.
- **Topics spiral.** You meet the same idea several times, each time deeper. Attention: concept (M4) → exact FLOP/dimension math (M9) → KV-cache serving (M12) → FlashAttention tiling (M18b) → long-context decoding (M20c). Hardware: simulated (M10a) → parallel (M10b) → real (M10c) → kernel roofline (M18b).
- **No forward dependencies.** Every module lists what earlier module(s) it stands on. If a module ever feels impossible, the fix is to go back one module — not to push through.

### 23 capability anchors, ~30 build-modules

There are **23 capability anchors** (labelled M1–M24) — the milestones of the spiral. Two kinds of adjustment keep each anchor to "one clear job": large anchors **split into lettered sub-modules** (M6a/M6b, M10a/b/c, M16a/b, M18a/b, M20a/b/c, M24a/b), and thin adjacent anchors **merge** (M2 + M3 → the single **M2–M3** module). This keeps the 8-capability spiral intact while being honest about the true build load: **~30 build-modules, ~200 short lessons.**

> **Why the split matters:** the first draft hid, for example, "close the multi-week capstone" *and* "learn GPU kernels from zero" inside one module (old M18). A beginner should never be dropped into two hard things at once. Splitting makes each unit one clear job.

### Read this before you start: this is self-paced

These are **self-paced modules, not calendar weeks.** A module is a themed milestone that takes the calendar time *you* need. At 20+ hours/week, a true beginner should honestly expect **~10–14 months** end to end. Foundations (M1–M7) are the slowest per concept — budget 2–3 weeks each. That is normal and expected, not falling behind.

**The hard rule:** if **M2–M3** (the neuron & how it learns) through **M6** (attention, transformers) feel shaky, do **not** advance to M8+. The entire frontier spine assumes that intuition is solid.

---

## Global rules (apply to every module)

**Pacing**
- 5 study days per week + 1–2 rest/catch-up days. Never put new heavy content on a catch-up day.
- If a day's step isn't finished, it rolls to the catch-up day — not into the next new topic.
- One atomic build per day: one function, one figure, one small experiment. **Lesson counts per module vary from 2 to 12** — that is by design. A module is exactly as many short lessons as the topic needs.
- Expect a full catch-up week after the capstone (M14), after generative modeling (M19), and after native kernels (M20a).

**Cost (set caps BEFORE any job runs)**
- Modules M1–M9, M11, M15–M17, M19, M23–M24 target **$0** (simulated / toy scale).
- Real-hardware caps: **M10c** first real TPU ~**$50**; **M13–M14** capstone + IsoFlops sweep ~**$150–250** total; **M18/M20** kernels/quant ~**$40**; **M22** ADRS loop ~**$40** with a per-run token ceiling.
- Every paid job gets a toy-scale dry-run first. Set billing alerts at 50% and 90% of each cap.
- Request GPU/TPU quota one full module *before* you need it (approval can take days). Provision the persistent capstone TPU during M12.
- Checkpoint to durable storage — a run that isn't checkpointed doesn't count as done.

**Kernel bar:** "numerically correct first." Matching a reference within tolerance is a pass. Beating the vendor primitive is always an *optional* stretch.

**Live-interactive bar:** every foundation lesson whose topic *rewards manipulation* opens its "Build it up" section with a **live interactive** (`sessions/viz/*.html`, embedded via the `.build-embed` iframe), then keeps the static scroll-reveal diagram underneath as the piece-by-piece explanation. Pure "anatomy" frames stay static. See the [visualization plan](#visualization-plan-live-vs-static).

**Legend**

| Marker | Meaning |
|---|---|
| ⭐ | Module added to close a prerequisite gap |
| 🎨 | Design-case-study checkpoint |
| 🔴 | Paid / gated milestone |
| 🆕 | Content authored from scratch (no reusable source in the repo) |
| **Action** | `author` (new HTML from notebooks/MD) · `reuse` (edit existing `sessions/week-*` in place) · `wrap` (short HTML linking to a full case study) · `convert` (notebook module → HTML) |
| **Viz state** | ∘ reuse existing verbatim · ● built · 🆕 still to build · — static SVG only |
| **Status** | ⬜ not started · 🔄 in progress · ✅ done |

**Tracking:** HTML lessons → `sessions/progress.json` (`w-d` quest ids). Experiment notebooks → COACH (`coach/core.py` ids). Build-system pipeline: [`docs/superpowers/plans/2026-07-03-capability-spiral-build.md`](docs/superpowers/plans/2026-07-03-capability-spiral-build.md). **The dividable remaining-work queue is the [Build queue](#build-queue--execution--divide-and-conquer) section below.**

---

## Master build table

The authoritative build target. `~L` = expected short-lesson count. `Src` = primary source (frontier `week-*` folder, foundation dir `00–10`, or design study). Every count is `reuse`+`author` where relevant.

| M | Build-module | Cap. | Source | Action | ~L | Viz | Cost | Status |
|---|---|---|---|---|----|-----|------|--------|
| 1 | Numbers, Arrays & Shapes | Represent | Python/NumPy + light math | author | 6 | 🆕 matmul, broadcasting | $0 | 🔄 5/6 + gate |
| 2–3 | The Neuron & How It Learns *(M2+M3 merged)* | Represent→Train | `00-neural-networks/fundamentals` 01–08 + optimizers/LR/split | author | 9 | 🆕 gradient-descent | $0 | 🔄 6/9 + gates |
| 4 | Attention as a Pure Idea | Represent | `01-transformers/days` 01–17 | author | 6 | ● attention-heatmap/-pipeline/softmax-scaling/-multihead | $0 | 🔄 4/6 + gate |
| 5 | MLP → PyTorch | Train | `…fundamentals` 09–10 | author | 6 | ∘ gradient-descent | $0 | 🔄 1/6 (stub) |
| 6a | Full Text Transformer | Represent | `01-transformers/days` 18–28 | author | 8 | ∘ attention-* | $0 | ⬜ |
| 6b | Vision Transformer (ViT) | Represent | frontier Phase-1 ViT | author | 4 | ∘ attention-* | $0 | ⬜ |
| 7 | CNNs & Vision Encoders ⭐ | Represent | `00-neural-networks/cnn` + `05-multimodal/vision-language` | author | 8 | — | $0 | ⬜ |
| 8 | Thinking in JAX | Train | `sessions/week-01` (+ author MLP capstone 🆕) | reuse+author | 6 | — | $0 | ⬜ |
| 9 | Transformer Math + Search 🎨 | Represent | `week-03` + design: ml-design-prep *(fw)*, visual-search, youtube-video-search, street-view | reuse+wrap | 8 | ∘ transformer-flops | $0 | ⬜ |
| 10a | Hardware Physics: One Device | Scale | `week-02` | reuse | 5 | ∘ roofline | $0 | ⬜ |
| 10b | Sharding & Parallelism | Scale | `week-04` | reuse | 6 | ∘ parallelism | $0 | ⬜ |
| 10c | First Real Multi-Device Run 🔴 | Scale | `week-04` bridge | reuse | 2 | — | ~$50 | ⬜ |
| 11 | Scaling Laws + Simulator | Scale | `week-05` | reuse | 6 | ∘ scaling-laws | $0 | ⬜ |
| 12 | Inference + Language-Assist 🎨 | Serve | `week-06` + design: smart-compose, translate + authored BPE | reuse+wrap+author | 8 | ∘ kv-cache, flash-attention | $0 | ⬜ |
| 13 | Addition Transformer — Build 🔴 | Train | `week-09` + `week-10`(d01–03) + `week-08` profiling | reuse | 10 | ∘ transformer-flops | paid TPU | ⬜ |
| 14 | Your Own IsoFlops Law 🔴 | Scale | `week-10`(d04–06) + `week-11`(d01–03) | reuse | 6 | ∘ scaling-laws | ~$150–250 | ⬜ |
| 15 | Ranking & Recommendation 🎨 | Serve | design: video-rec, ad-click, event-rec, similar-listings, news-feed (+ authored foundation) | author+wrap | 8 | — | $0 | ⬜ |
| 16a | Mixture of Experts | Scale | `week-07` | reuse | 7 | ∘ moe-routing | $0 | ⬜ |
| 16b | Graph Systems / GNN 🎨 | Scale | design: people-you-may-know | wrap | 4 | — | $0 | ⬜ |
| 17 | Post-Training: RLHF/DPO/GRPO ⭐ | Train | `10-reinforcement-learning/rlhf` **+ `02-fine-tuning`** | convert | 6 | — | $0 | ⬜ |
| 18a | Capstone Close-Out: MoE + Report | Optimize | `week-11`(d04–06) + `week-12` (+ `week-17` optional polish) | reuse | 3 | — | small TPU | ⬜ |
| 18b | Custom Kernels I: Pallas | Optimize | `week-13` + what-is-a-kernel on-ramp 🆕 | reuse+author | 6 | ∘ flash-attention | small TPU | ⬜ |
| 19 | Generative Modeling ⭐ 🆕 | Generate | 🆕 authored (source notebooks intuition-only; VAE reparam absent) | author | 12 | 🆕 diffusion-noising | $0 | ⬜ |
| 20a | Native Kernels (CUDA/TK) 🔴 | Optimize | `week-14` | reuse | 5 | — | ~$40 | ⬜ |
| 20b | Quantization | Optimize | `week-15` | reuse | 5 | ∘ quantization | small | ⬜ |
| 20c | Long-Context Decoding | Optimize | `week-16` | reuse | 5 | ∘ speculative-decoding | small | ⬜ |
| 21 | Image & Text Generation 🎨 | Generate | genAI design: intro *(fw)*, captioning, RAG, face, high-res, text-to-image, headshots, text-to-video | wrap | 9 | — | small | ⬜ |
| 22 | AI-Driven Research (ADRS) | Automate | `week-19` + `week-20` **+ `06-evaluation`** + `frontier-lab/week19` reward-hacking | reuse | 8 | — | ~$40 | ⬜ |
| 23 | Assistant & Safety Systems 🎨 | Automate | genAI chatbot; ML Design harmful-content-detection | wrap | 6 | — | $0 | ⬜ |
| 24a | Formal Methods (TLA+ / Hypothesis) | Communicate | `week-21` (1 reuse + 5 authored) | reuse+author | 6 | — | $0 | ⬜ |
| 24b | Ship & Outreach | Communicate | `week-23` / `week-24` | reuse | 7 | — | $0 | ⬜ |

**Totals:** 23 anchors (M2 & M3 merged) · ~30 build-modules · **~200 short lessons.**

> **Lesson-count honesty (thin-week notes):** M13's 10 lessons = `week-09` (6) + `week-10` d01–03 (3) + `week-08` profiling curated to ~1. M18a's 3 lessons *curate* 5 source days (`week-11` d04–06 + `week-12`) down to 3. M24a's 6 lessons = 1 reused (`week-21` TLA+ on-ramp) **+ 5 authored**. Where a module authors more than it reuses, that is called out in its detail block.

---

## The modules

> **Foundations are grouped into three Parts (A/B/C).** Each Part is a coherent unit; the modules inside are its chapters — so daily lessons stay bite-sized while the roadmap reads as evenly-weighted phases, not a string of small modules.

### Foundations · Part A — The Neural Net From Scratch

*Capability: REPRESENT → TRAIN. Build a working neural network from nothing but NumPy — the data and math it runs on (M1), then a single neuron and exactly how it learns (M2–M3).*

#### M1 — Numbers, Arrays, and the Shape of Data · 6 lessons
- **Goal:** get fluent enough in Python/NumPy and light math that every later shape-and-matmul statement reads as English.
- **You build:** a NumPy notebook that loads a small dataset, reshapes and broadcasts arrays, does matrix multiplies by hand and with NumPy (printing every shape), plus a one-page logs/exponents worksheet.
- **Topics:** arrays, indexing, slicing · broadcasting & dtypes · matrix multiply & shape bookkeeping · **vectors, dot products, cosine similarity** (folded into matmul) · **random seeds / PRNG** (needed before M4 embeddings and M8 JAX keys) · logs/exponents (for power laws later).
- **Source:** Python + NumPy warm-up; light-math primer · **Action:** author · **Builds on:** nothing — start here · **Cost:** $0.
- *(Fix: a random-seeds lesson so PRNG isn't first met cold in JAX; teach dot-product/cosine explicitly, not just implicitly inside matmul.)*

| Day | Lesson | File | quest id | Status |
|-----|--------|------|----------|--------|
| 1 | Arrays & the shape of data | `week-f1/day-01-arrays.html` | `wf1-d01-arrays` | ✅ done |
| 2 | Indexing & slicing | `week-f1/day-02-indexing-slicing.html` | `wf1-d02-indexing` | ✅ done |
| 3 | Broadcasting & dtypes | `week-f1/day-03-broadcasting-dtypes.html` | `wf1-d03-broadcasting` | ✅ done (⬜ embed `broadcasting.html`) |
| 4 | Matmul & shapes | `week-f1/day-04-matmul-and-shapes.html` | `wf1-d04-matmul` | ✅ done (⬜ embed `matmul.html`) |
| 5 | Logs & exponents | `week-f1/day-05-logs-and-exponents.html` | `wf1-d05-logs` | ✅ done |
| 6 | **Random seeds / PRNG** | `week-f1/day-06-random-seeds.html` | `wf1-d06-seeds` | ⬜ **finish-first** |
| — | Review gate | `week-f1/review.html` | `wf1-review` | ✅ done |

#### M2–M3 — The Neuron and How It Learns · 9 lessons *(M2 + M3 merged)*
- **Goal:** build a single neuron, stack neurons into a layer and run a forward pass — then learn exactly how that network measures error and updates its weights. The complete from-scratch story of one learning unit, in one module.
- **You build:** a from-scratch neuron (weighted sum + bias + activation) → a one-layer forward pass in NumPy (printing every shape) → MSE + cross-entropy loss → backprop through a 2-layer net → SGD → momentum → Adam on the same net (watching the loss curve) → a train/val/test split with an overfitting demo.
- **Topics:** what a neural network is · neuron = weighted sum + bias · activations (ReLU, sigmoid) · a layer as a matrix multiply · forward propagation · loss (MSE, cross-entropy, and **softmax as a named function**) · gradients as slopes · backprop as the chain rule · **optimizers: SGD → momentum → Adam** · **learning-rate intuition** · **train/val/test split & overfitting** · the training-loop skeleton.
- **Source:** `00-neural-networks/fundamentals` 01–08 (spans `week-f2` + `week-f3`) · **Action:** author · **Builds on:** M1 · **Cost:** $0.
- *(Merge rationale: the thin 4-lesson forward-pass module (old M2) and the 6-lesson backward-pass module (old M3) are one continuous story — a neuron and how it learns — so they are a single ~9-lesson module. Softmax lands here; optimizers (esp. Adam) are taught here, before Optax in M8 and before Adam's 2× optimizer-state memory in M10a.)*

| Day | Lesson | File | quest id | Status |
|-----|--------|------|----------|--------|
| 1 | Single neuron | `week-f2/day-01-single-neuron.html` | `wf2-d01-neuron` | ✅ done |
| 2 | Activations | `week-f2/day-02-activations.html` | `wf2-d02-activations` | ✅ done |
| 3 | Layers & forward pass | `week-f2/day-03-layers-forward-pass.html` | `wf2-d03-layers` | ✅ done |
| 4 | Loss (MSE, cross-entropy, softmax) | `week-f3/day-01-loss.html` | `wf3-d01-loss` | ✅ done |
| 5 | Gradients & backprop | `week-f3/day-02-gradients-backprop.html` | `wf3-d02-backprop` | ✅ done |
| 6 | The training loop | `week-f3/day-03-training-loop.html` | `wf3-d03-loop` | ✅ done |
| 7 | **Optimizers: SGD → momentum → Adam** | `week-f3/day-04-optimizers.html` | `wf3-d04-optimizers` | ⬜ **finish-first** (⬜ embed `gradient-descent.html`) |
| 8 | **Learning-rate intuition** | `week-f3/day-05-learning-rate.html` | `wf3-d05-lr` | ⬜ **finish-first** |
| 9 | **Train/val/test split & overfitting** | `week-f3/day-06-train-val-test.html` | `wf3-d06-split` | ⬜ **finish-first** |
| — | Review gates | `week-f2/review.html` · `week-f3/review.html` | `wf2-review` · `wf3-review` | ✅ done |

### Foundations · Part B — Attention & First Models

*Capability: REPRESENT → TRAIN. Meet attention as a pure idea (M4), then train your first real model and see what a framework gives you (M5).*

#### M4 — Attention as a Pure Idea · 6 lessons
- **Goal:** learn embeddings, self-attention, **and position** as concepts so no later transformer build depends on an unseen idea.
- **You build:** from-scratch scaled-dot-product attention on a toy sequence — embeddings, Q/K/V, scores, softmax, weighted sum, multi-head — printing every matrix shape; then a positional-encoding demo showing the "shuffle problem."
- **Topics:** embeddings · Query/Key/Value intuition · attention scores & √d_k scaling & softmax · multi-head attention · **the shuffle problem / permutation invariance** · **sinusoidal positional encoding** (RoPE/ALiBi as awareness) · a short **"tokens are the bridge between text and embeddings"** note (so BPE in M12 isn't a cold surprise).
- **Source:** `01-transformers/days` 01–17 · **Action:** author · **Builds on:** M1, M2–M3 · **Cost:** $0.
- **Live viz:** `attention-heatmap` (day-01), `attention-pipeline` + `softmax-scaling` (day-03), `attention-multihead` (day-04) — reuse the Attention Lab's verified D3 logic.
- *(Fix: positional encoding was orphaned — assumed known by M6, built nowhere. It lives here now.)*

| Day | Lesson | File | quest id | Live viz | Status |
|-----|--------|------|----------|----------|--------|
| 1 | Embeddings (words → numbers) | `week-f4/day-01-embeddings.html` | `wf4-d01-embeddings` | `attention-heatmap.html` | ✅ built · viz verified |
| 2 | Query / Key / Value | `week-f4/day-02-qkv.html` | `wf4-d02-qkv` | — (static cards) | ✅ built |
| 3 | Attention scores, √d_k & softmax | `week-f4/day-03-attention-scores.html` | `wf4-d03-scores` | `attention-pipeline` + `softmax-scaling` | ✅ built · viz verified |
| 4 | Multi-head attention | `week-f4/day-04-multihead.html` | `wf4-d04-multihead` | `attention-multihead.html` | ✅ built · viz verified |
| 5 | **Positional encoding (+ shuffle problem)** | `week-f4/day-05-positional.html` | `wf4-d05-position` | (candidate: new) | ⬜ **finish-first** |
| — | Review gate | `week-f4/review.html` | `wf4-review` | — | ✅ done |

#### M5 — Your First From-Scratch Model: an MLP, Then the PyTorch Version · 6 lessons
- **Goal:** train a plain MLP on MNIST from scratch, then rebuild it in PyTorch to see what a framework gives you for free.
- **You build:** forward pass → backward + wiring → mini-batch loop with an overfit-one-batch sanity check → full MNIST train with a **loss-curve read and dropout** → the same model re-implemented in PyTorch → a review gate.
- **Topics:** from-scratch MLP forward+backward · mini-batch loop · overfit-one-batch sanity check · **reading a loss curve** · **dropout / basic regularization** · the same model in PyTorch (autograd / nn.Module / optimizer) · why PyTorch tensors mirror NumPy.
- **Source:** `00-neural-networks/fundamentals` 09–10 · **Action:** author · **Builds on:** M2–M3 (M4 seeds intuition) · **Cost:** $0.
- **⚠️ Status: 1/6 — stub.** Only `week-f5/day-01-mlp-mnist.html` exists. Missing: backward+wiring, mini-batch loop, loss-curve+dropout, PyTorch re-implementation, review gate. **This is the most stalled foundation module — finish before advancing to M6.**

### Foundations · Part C — Full Architectures

*Capability: REPRESENT. Assemble complete architectures — the full text transformer and a vision transformer (M6a/M6b), then CNNs and vision encoders (M7).*

#### M6a — The Full Text Transformer · 8 lessons *(split from M6)*
- **Goal:** assemble a complete decoder-only transformer that generates text.
- **You build:** a full decoder-only block (skip connections, layer norm, feed-forward, causal masking) that generates text, one sublayer at a time.
- **Topics:** skip connections & layer norm · feed-forward sublayer · the full block · causal masking · encoder vs decoder · text generation / sampling.
- **Source:** `01-transformers/days` 18–28 · **Action:** author · **Builds on:** M4–M5 · **Cost:** $0.

#### M6b — A Vision Transformer (ViT) · 4 lessons *(split from M6)*
- **Goal:** build the Vision Transformer the frontier Week-1 exercise assumed — correctly deferred until *after* attention and the text transformer.
- **You build:** a small ViT on CIFAR-10 image patches (patch embeddings → the same transformer block from M6a → a classifier head).
- **Topics:** images as patches · patch embeddings · reusing the transformer block for vision · classification head.
- **Source:** the frontier Phase-1 ViT exercise, re-ordered here · **Action:** author · **Builds on:** M6a · **Cost:** $0.
- *(Optional consolidation, deferred: M6b could fold into M6a as one "Transformers: Text & Vision" module if the shared attention block is genuine reuse. Kept split for now.)*

#### M7 — Seeing With Convolutions: CNNs and Vision Encoders ⭐ · 8 lessons
- **Goal:** learn how images become embeddings so the visual design studies are real hands-on builds, not hand-waving.
- **You build:** a from-scratch convolution + pooling op → **output-dimension arithmetic** → a small CNN classifier → **batch normalization** → **data augmentation** → transfer learning → fixed-length image embeddings → a toy CLIP-style contrastive demo.
- **Topics:** convolution & pooling · **output-dim arithmetic (stride/padding)** · a complete CNN classifier · **batch normalization** · **data augmentation** · transfer learning → image embeddings · object-detection intuition (awareness) · contrastive image-text (CLIP) as its own lesson.
- **Source:** `00-neural-networks/cnn` + `05-multimodal/vision-language` · **Action:** author · **Builds on:** M5–M6 · **Cost:** $0.
- *(⭐ closes the vision-encoder gap. Fix: batchnorm, output-dim arithmetic, and augmentation were missing across M1–M7.)*

### Capability: TRAIN → SCALE → SERVE — Frontier spine + first checkpoints

#### M8 — Thinking in JAX · 6 lessons
- **Goal:** rebuild the MLP in pure JAX/Flax to internalize immutability, PRNG keys, vmap, and jit — the substrate for the whole spine.
- **You build:** a simple **MLP** trained in Flax/Optax on CIFAR-10, with small experiments showing vmap and jit speedups. 🆕 The MLP capstone lesson is authored (the only week-01 source capstone is a ViT, which belongs in M6b).
- **Topics:** array immutability vs NumPy · PRNG keys & explicit splitting · vmap vectorization · jit / XLA compilation · Flax nn.Module + Optax.
- **Source:** `sessions/week-01` · **Action:** reuse + author (MLP capstone) · **Builds on:** M2–M3, M5, M6a (and M1 for the array + PRNG contrast) · **Cost:** $0.
- *(Fix: first JAX build is an MLP, not a ViT — matches the "first model is an MLP" principle from M5. Confirm the ViT build lives in M6b so nothing is dropped in the swap.)*

#### M9 — Transformer Math and the Embedding Search Engine 🎨 · 8 lessons
- **Goal:** derive exact FLOP and dimension formulas, then build real embedding-based search systems as the applied checkpoint.
- **You build:** a pen-and-code derivation of C≈6ND, exact Q/K/V/FFN dims, and a KV-cache size for a 7B model; then, via ~2 orientation lessons routing into the deeper design studies — a visual-search pipeline (image embeddings + ANN index + re-ranking), a YouTube-style video search, and a Street-View blurring pipeline — all through the 7-step design framework.
- **Topics:** C≈6ND FLOP accounting · exact Q/K/V & FFN dims · KV-cache sizing (7B @ 8k) · the 7-step ML design framework · embedding + ANN retrieval · detection-based privacy pipeline.
- **Source:** `sessions/week-03` (days 1–3 strict reuse; days 4–6 optional deeper material) + design: **ml-design-prep** *(framework)*, visual-search, youtube-video-search, google-street-view-blurring · **Action:** reuse + wrap · **Builds on:** M6–M7, M8, M1 · **Cost:** $0.

#### M10a — Hardware Physics: One Device · 5 lessons *(split from M10)*
- **Goal:** reason about rooflines, arithmetic intensity, and memory on a single simulated device.
- **You build:** roofline & arithmetic-intensity calculations for your ViT; a memory calculator including **Adam's 2× optimizer state** and activation checkpointing.
- **Topics:** roofline & arithmetic intensity · TPU/GPU memory hierarchy · memory footprint (weights + Adam 2× + activations) · activation checkpointing · a short **numerical-precision / mixed-precision** note.
- **Source:** `sessions/week-02` · **Action:** reuse · **Builds on:** M8–M9, M2–M3 (Adam) · **Cost:** $0.

#### M10b — Sharding and Parallelism · 6 lessons *(split from M10)*
- **Goal:** reason about sharding math and the parallelism strategies on *simulated* multi-device setups.
- **You build:** activation checkpointing across 4 simulated devices via pmean; a memory/parallelism calculator; DP / FSDP / tensor-parallel / pipeline-bubble simulations.
- **Topics:** sharding math (AllGather / ReduceScatter / AllReduce) · DP, FSDP/ZeRO-3, tensor parallel, pipeline bubbles · distillation · a memory/parallelism calculator.
- **Source:** `sessions/week-04` · **Action:** reuse · **Builds on:** M10a · **Cost:** $0 (simulated).

#### M10c — Your First Real Multi-Device Run 🔴 · 2 lessons *(split from M10)*
- **Goal:** move exactly one sharded run to real hardware with strict discipline and compare against the calculator.
- **You build:** one real multi-device sharded run vs the M10b calculator's predictions, with a dry-run and billing alerts.
- **Topics:** quota lead-time · dry-run discipline · real vs predicted memory/throughput.
- **Source:** `sessions/week-04` (bridge) · **Action:** reuse · **Builds on:** M10b · **Cost:** ~$50 cap, quota lead-time, dry-run first.
- *(Deliberately small: this is a cost/funding gate, not a content unit. Do not merge.)*

#### M11 — Scaling Laws and the Budget Simulator · 6 lessons
- **Goal:** derive scaling laws from Kaplan through the Chinchilla correction and build a compute-budget → optimal-model simulator.
- **You build:** a budget → optimal-(N,D) simulator: implement the IsoFlops method, fit power laws to toy loss curves (using the logs/exponents from M1), reproduce the Chinchilla-style parabola plots.
- **Topics:** Kaplan vs Chinchilla · the IsoFlops method · power-law derivation & fitting · the data wall · a budget → optimal (N, D) simulator.
- **Source:** `sessions/week-05` · **Action:** reuse · **Builds on:** M9 (FLOP accounting), M1 (logs/exponents) · **Cost:** $0 (toy curves).
- *(Cleanest 1:1 reuse in the plan; the power-law-derivation lesson is the payoff — do not split.)*

#### M12 — Inference Economics and Language-Assist Features 🎨 · 8 lessons
- **Goal:** understand prefill vs decode and the memory-bandwidth wall, then serve low-latency generative-assist features.
- **You build:** an inference/roofline calculator (prefill vs decode, batching economics, bandwidth wall); a dedicated **BPE/subword tokenization** lesson; then two decode-bound serving builds — Gmail Smart Compose autocomplete and a Google-Translate-style seq2seq with BLEU evaluation. Optionally add a **½-day profiling orientation** (TensorBoard basics) here so M13 isn't debugged cold.
- **Topics:** prefill vs decode · the bandwidth wall · batching economics & throughput · serving LLaMA3 + a calculator · **BPE/subword tokenization (authored lesson)** · latency/BLEU trade-offs.
- **Source:** `sessions/week-06` + design: gmail-smart-compose, google-translate · **Action:** reuse + wrap + author (BPE) · **Builds on:** M9 (KV-cache math), M10 (serving on real hardware) · **Cost:** $0.
- *(Fix: BPE was hand-waved in M4 and buried inside case studies — it gets one authored lesson here. Provision the persistent capstone TPU now for M13 lead-time.)*

### Capability: TRAIN & SCALE — The Capstone

#### M13 — The Addition Transformer: Build (Capstone Part 1) 🔴 · 10 lessons *(soft split: build ~6 / make-efficient+eval ~4)*
- **Goal:** build a ~10M-param dense transformer in JAX/Flax on synthetic digit-addition, with a config-driven runner.
- **You build:** a custom digit tokenizer, a synthetic addition-data generator, a static-shape ~10M-param Flax dense transformer, a config-driven training loop (overfit one batch, then train to convergence), exact-match evaluation, and a profiling trace to catch inefficient TPU compiles.
- **Topics:** custom tokenizer & synthetic data · static-shape engineering (avoid JIT recompiles) · ~10M-param Flax transformer + config runner · overfit-then-train + exact-match eval · reading a JAX/TensorBoard profiling trace.
- **Source:** `sessions/week-09` + `week-10` (d01–03) + `week-08` profiling · **Action:** reuse · **Builds on:** M6–M8, M10, M11, M12 · **Cost:** persistent paid TPU, capped, static shapes.

#### M14 — Your Own IsoFlops Scaling Law (Capstone Part 2) 🔴 · 6 lessons
- **Goal:** run *your own* shrunk IsoFlops sweep on the persistent paid TPU, aggregate losses, and fit a power law.
- **You build:** an automated launch/monitor/checkpoint harness over a shrunk matrix (N ~1M–15M across a few token budgets), aggregation and cleaning of the loss data, per-budget parabola fits, and your own derived data-hunger exponent — a Chinchilla-method reproduction on your own model. Includes a **billing-alert / dry-run gate lesson**.
- **Topics:** a shrunk run matrix · automated launch/monitor/checkpoint scripts · loss aggregation & cleaning · parabola fits per FLOP budget · deriving your own exponent · billing discipline.
- **Source:** `sessions/week-10` (d04–06) + `week-11` (d01–03) · **Action:** reuse · **Builds on:** M11 (method + simulator), M13 (the model + runner) · **Cost:** ~$150–250 total, billing alerts, durable checkpoints, dry-run first.

### Capability: SCALE deepened + checkpoints

#### M15 — Ranking and Recommendation at Serving Scale 🎨 · 8 lessons
- **Goal:** build two-stage retrieve-then-rank systems, applying your inference/serving skills to large-scale recommendation.
- **You build:** first **one from-scratch two-tower retrieval + ranking foundation lesson** (with negative sampling), then a **calibration/metrics** lesson, then five hands-on studies — video-recommendation, ad-click CTR, event-recommendation, similar-listings (session embeddings), personalized-news-feed (multi-task) — each with offline vs online metrics and drift monitoring.
- **Topics:** two-tower retrieval + ranking · negative sampling · CTR prediction & calibration · session/co-occurrence embeddings · multi-task ranking, feature freshness · offline vs online metrics & drift.
- **Source:** design: video-recommendation, ad-click-prediction, event-recommendation, similar-listings, personalized-news-feed · **Action:** author (foundation) + wrap · **Builds on:** M9 (embeddings), M12 (serving economics) · **Cost:** $0 (small-scale).
- *(Fix: the plan promised "you build a two-tower system" but the source is 5 case-study READMEs — a shared from-scratch foundation lesson is authored once, before the wraps.)*

#### M16a — Mixture of Experts · 7 lessons *(split from M16)*
- **Goal:** master MoE routing and build a toy router that becomes your reusable artifact.
- **You build:** a toy MoE router (top-k gating + auxiliary load-balancing loss + distributed AllToAll intuition) — *your reusable artifact* for the capstone MoE upgrade (M18a), later kernels (M18b, M20a), and the ADRS loop (M22).
- **Topics:** expert gating & top-k routing · load balancing & the aux loss · distributed AllToAll · MoE vs dense · a toy router (reused downstream).
- **Source:** `sessions/week-07` · **Action:** reuse · **Builds on:** M10b (parallelism), M11 (scaling) · **Cost:** $0.

#### M16b — Graph Systems and Link Prediction 🎨 · 4 lessons *(split from M16)*
- **Goal:** apply graph learning to a link-prediction design study.
- **You build:** a People-You-May-Know build teaching graph neural networks for edge/link prediction.
- **Topics:** graphs as data · message passing · GNNs · link/edge prediction · negative sampling for graphs.
- **Source:** design: people-you-may-know · **Action:** wrap · **Builds on:** M15 (ranking contrast) · **Cost:** $0.
- *(Optional consolidation, deferred: M16b could fold into M15 as "Retrieval & Ranking" if GNN is framed as an alternative retrieval method. Kept split for now.)*

#### M17 — Post-Training: How a Base Model Becomes an Assistant · 6 lessons ⭐
- **Goal:** understand the pretrain → SFT → RLHF/DPO/GRPO pipeline that a frontier staff hire is expected to know and the chatbot study assumes.
- **You build:** an orientation of the stages; **fine-tuning mechanics (full FT, LoRA, QLoRA, instruction tuning)** drawn from `02-fine-tuning`; a small reward model on toy preference pairs (Bradley-Terry + margin-ranking loss, last-token head); a from-scratch DPO objective; a demonstration of the KL penalty preventing reward hacking; and **GRPO** awareness.
- **Topics:** the stages (pretrain, SFT, RLHF) · **SFT & parameter-efficient fine-tuning (LoRA/QLoRA)** · reward models & margin-ranking loss · **PPO-for-LM** (KL penalty + value head) · DPO as a simpler alternative · **GRPO** (critic-free, DeepSeek-R1) · instruction-following vs base behavior.
- **Source:** `10-reinforcement-learning/rlhf` (Core-depth) **+ `02-fine-tuning`** (LoRA/QLoRA/full-FT/instruction-tuning notebooks + interview MD) · **Action:** convert · **Builds on:** M2–M3 (loss/gradients), M6a (decoder), M13 (a trained transformer) · **Cost:** $0 (toy).
- *(⭐ closes the RLHF gap; the reward-hacking intuition is reused by the ADRS sandbox in M22. **`02-fine-tuning` is woven in here** — it was previously an orphaned module.)*

### Capability: OPTIMIZE & GENERATE

#### M18a — Capstone Close-Out: MoE Upgrade + Report · 3 lessons *(split from M18; curates 5 source days → 3)*
- **Goal:** finish the capstone before starting anything new.
- **You build:** upgrade the dense model to MoE using your M16a router, run a shrunk MoE IsoFlops matrix, derive its scaling law vs dense, and write a scientific report + single-command reproducible repo.
- **Topics:** MoE capstone upgrade · MoE IsoFlops + scaling-law-vs-dense derivation · scientific report + reproducible repo.
- **Source:** `sessions/week-11` (d04–06) + `week-12` (+ `week-17` `day-01-consolidation-publishing` as optional polish) · **Action:** reuse · **Builds on:** M13–M14, M16a · **Cost:** small paid TPU for the MoE sweep.
- *(Fix: `week-18` never existed — the old "Weeks 17–18 consolidation" reference is corrected here. `week-17` (previously orphaned) is now folded in as optional polish.)*

#### M18b — Custom Kernels I: Pallas and FlashAttention · 6 lessons *(split from M18)*
- **Goal:** learn what a kernel is, then write numerically-correct Pallas kernels.
- **You build:** a "what is a GPU kernel" on-ramp → a numpy online-softmax warm-up → a Pallas matmul kernel → a FlashAttention-tiled kernel (its own day) → a fused MoE up/down-projection kernel over *your own* router.
- **Topics:** what a kernel is (threads/blocks/SRAM) · the Pallas abstraction · matmul & SRAM bank conflicts · online softmax · FlashAttention tiling · a fused MoE kernel (correct first, beat-vendor optional). A first **einsum / tensor-contraction** primer fits naturally here.
- **Source:** `sessions/week-13` + what-is-a-kernel on-ramp 🆕 · **Action:** reuse + author · **Builds on:** M18a, M16a, M10, M8 · **Cost:** small paid TPU · **Kernel bar:** correctness first.
- *(Fix: old M18 fused "close the capstone" with "learn Pallas from zero" — two hard things at once. Split; FlashAttention gets its own day; a what-is-a-kernel on-ramp is added.)*

#### M19 — Generative Modeling From Scratch: VAE, VQ-VAE, GAN, Diffusion ⭐ 🆕 · 12 lessons *(author from scratch)*
- **Goal:** learn the generative paradigms every image/video design study assumes, before building any of them.
- **You build:** a progression of tiny from-scratch, **trainable** builds — a VAE (encoder/latent/decoder, the **reparameterization trick**, the ELBO), a VQ-VAE codebook (the **straight-through estimator**), a small GAN (generator vs discriminator, **mode collapse**), a DDPM (forward noising, reverse denoising, noise-prediction loss, schedule), and a latent-diffusion sketch (CLIP/T5 conditioning + classifier-free guidance) — generating simple images to prove each works.
- **Suggested day breakdown:** VAE d1–2 · VQ-VAE d3–4 · GAN d5–6 · DDPM d7–10 · latent diffusion d11–12. Consider splitting into **M19a Autoencoders & GANs** / **M19b Diffusion** across two calendar blocks.
- **Topics:** VAE + reparameterization + ELBO · VQ-VAE + straight-through estimator · GAN + mode collapse · DDPM forward/reverse + noise-prediction loss + schedule · latent diffusion + text conditioning + CFG · a one-page ConvTranspose/upsampling reminder (M7 dependency).
- **Source:** 🆕 **must be authored** — genAI notebooks 07/08/09 are design-study *intuition* notebooks with no training loops, and the **VAE reparameterization trick appears nowhere in the repo.** · **Action:** author · **Builds on:** M6–M7, M5 · **Cost:** $0 (toy images).
- *(⭐ closes the single biggest content gap and is a scope *addition* beyond the frontier companion. Densest non-capstone module — pace one paradigm per few days; expect a catch-up week after. Build `diffusion-noising.html` viz.)*

#### M20a — Native Kernels (C++/CUDA + ThunderKittens) 🔴 · 5 lessons *(split ×3 from M20)*
- **Goal:** get a C++/CUDA on-ramp and write a native ThunderKittens kernel verified against PyTorch.
- **You build:** a C++/CUDA on-ramp (toolchain, threads/blocks/warps, vector-add, naive matmul) → a native ThunderKittens kernel (16×16 tiles, TMA, worker overlap, CUDA 12.8/g++ build) verified against a PyTorch baseline.
- **Topics:** C++/CUDA on-ramp · threads/blocks/warps · ThunderKittens tiles/TMA/overlap · native build · correctness vs PyTorch.
- **Source:** `sessions/week-14` · **Action:** reuse · **Builds on:** M18b (Pallas/tiling), M5 (PyTorch) · **Cost:** ~$40 cap; expect a catch-up week after.

#### M20b — Quantization · 5 lessons *(split ×3 from M20)*
- **Goal:** quantize to int8 and reason about the memory-vs-error trade.
- **You build:** int8 theory + LLM.int8 outliers → PTQ + a perplexity benchmark → QuIP incoherence processing → QTIP/AQLM awareness → a memory-economics synthesis.
- **Topics:** int8 quantization · LLM.int8 outliers · PTQ + perplexity · QuIP# / QTIP / AQLM (awareness) · memory economics.
- **Source:** `sessions/week-15` · **Action:** reuse · **Builds on:** M12 (inference/KV), M9 (KV math) · **Cost:** small.

#### M20c — Long-Context Decoding · 5 lessons *(split ×3 from M20)*
- **Goal:** implement the tricks that make long-context serving cheap.
- **You build:** a KV-cache problem recap → SnapKV eviction → GQA/MQA → speculative decoding → ring attention; capped by the **"hardware lottery" essay**.
- **Topics:** KV-cache eviction (SnapKV) · GQA/MQA · speculative decoding · ring attention · the hardware-lottery essay.
- **Source:** `sessions/week-16` · **Action:** reuse · **Builds on:** M12, M20b · **Cost:** small.
- *(Fix: old M20 fused native CUDA + quantization + long-context — three staff specialties, ~18 source-days. Split into three; a catch-up week is budgeted after M20a.)*

#### M21 — Image and Text Generation Systems 🎨 · 9 lessons
- **Goal:** build the generative design studies now that you can train diffusion models and reason about retrieval and serving.
- **You build:** hands-on wraps via the genAI framework, ordered — image captioning (multimodal encoder-decoder), RAG (embeddings + ANN + generator), then the diffusion-applied trio (realistic-face, high-res synthesis, text-to-image), then personalized headshots (LoRA/DreamBooth), then **text-to-video split into two** (image-to-video foundations, then temporal modeling) — with serving and safety trade-offs for each.
- **Topics:** image captioning · RAG end-to-end · diffusion applied (face, high-res, text-to-image) · LoRA/DreamBooth · text-to-video (temporal consistency, video latent diffusion) · generation serving, safety, evaluation.
- **Source:** genAI design: intro-and-framework *(framework)*, image-captioning, RAG, realistic-face-generation, high-res-image-synthesis, text-to-image, personalized-headshots, text-to-video · **Action:** wrap · **Builds on:** M19 (the unblocking prerequisite), M9 (retrieval → RAG), M12 (serving), M6–M7, M20 · **Cost:** small paid cap for any diffusion fine-tuning.
- *(These are wraps over strong source READMEs — the risk is breadth/fatigue, not missing content. Pace flexibly; text-to-video is the heaviest.)*

### Capability: AUTOMATE & COMMUNICATE

#### M22 — AI-Driven Research for Systems (ADRS) · 8 lessons
- **Goal:** build a fortified eval sandbox and an autonomous optimization loop that improves *your own* MoE router / Pallas / TK kernel over an LLM API.
- **You build:** (1) why ADRS needs a reliable verifier → (2) the 5-component loop → (3) **evolutionary-search basics** → (4) **agentic scaffolding over an LLM API** → (5) reward hacking as a *gate* → (6) evaluation-engineering / hidden holdout → (7) a case-study tour (FunSearch/AlphaGeometry/EvoPrompt/RuleFlow) → (8) run the loop on your own M16a router or M18/M20 kernel with cost caps.
- **Topics:** the ADRS paradigm & its 5 components · reliable verifier · evolutionary search · agentic scaffolding over an LLM API · reward hacking & a fortified sandbox · evaluation engineering · a loop optimizing your own kernel/router.
- **Source:** `sessions/week-19` + `week-20` **+ `06-evaluation`** (metrics + benchmarks for the verifier/holdout lessons) + `frontier-lab/week19/friday-reward-hacking/` (naive vs hardened evaluator, load-balancer hack demo, conservation test) · **Action:** reuse · **Builds on:** M16a (your router), M18/M20 (your kernels), M17 (reward-hacking intuition) · **Cost:** ~$40 cap, per-run token ceiling, dry-run.
- *(Fix: 12 rich source lessons were compressed into one module; genetic-algorithm and agentic-scaffold on-ramps are the true gap — added. **`06-evaluation` is woven in here** — it was previously orphaned. Consider a 1-day formal-methods mini-intro (from M24a) *before* running the loop, so the loop is designed against a safety invariant rather than verified after the fact.)*

#### M23 — Assistant and Safety Systems 🎨 · 6 lessons
- **Goal:** build the studies that need a full transformer + RLHF + a robust eval sandbox — a chat assistant and a harmful-content detector.
- **You build:** a **"two kinds of safety" framing lesson** (behavioral alignment vs content classification); a ChatGPT-style chatbot build (decoder-only + SFT + RLHF/DPO from M17 + serving); a multi-modal harmful-content detector (fused text + image embeddings, precision/recall, late-vs-early fusion); and a **drift-monitoring lesson**.
- **Topics:** two kinds of safety · chatbot: decoder-only + SFT/RLHF pipeline in production · serving a chat assistant (latency, safety filters, red-teaming) · harmful-content detection as multi-modal fusion · precision vs recall for safety; late vs early fusion · monitoring & drift.
- **Source:** genAI design: chatgpt-chatbot; ML Design: harmful-content-detection · **Action:** wrap · **Builds on:** M17 (RLHF), M22 (eval/robustness), M12 (serving), M7 (multimodal embeddings) · **Cost:** $0.

#### M24a — Formal Methods: TLA+ and Property-Based Testing · 6 lessons *(split from M24; 1 reuse + 5 authored)*
- **Goal:** formally verify one property of your automated loop.
- **You build:** a TLA+ on-ramp (install → states/Init/Next → invariants → TLC model-check + state-explosion/bounding) → one safety-invariant spec for the M22 ADRS loop/eval sandbox → a **refinement** map from the abstract spec to your Python sorter → **Hypothesis** property-based testing as a complement.
- **Topics:** TLA+ states/invariants · TLC model-checking & bounding · refinement (spec ↔ concrete Python) · Hypothesis property-based testing (probabilistic vs exhaustive).
- **Source:** `sessions/week-21` (`day-01-formal-verification` on-ramp) **+ 5 authored TLA+ lessons** · **Action:** reuse + author · **Builds on:** M22 (the loop to verify) · **Cost:** $0.
- *(Fix: `week-21` has only 1 day on disk — the other 5 lessons must be authored. TLA+ sits adjacent to M22, which it verifies; refinement gets its own worked lesson. `week-22` never existed.)*

#### M24b — Ship and Outreach: Portfolio, Report, Interview · 7 lessons *(split from M24)*
- **Goal:** package everything and run ruthlessly technical outreach and interview prep.
- **You build:** a code-walkthrough screencast → a math-walkthrough screencast → a kernel/systems technical report → a clean GitHub portfolio → a ruthlessly technical cold email → targeted outreach to the Phase-4/5 paper authors → staff-researcher interview prep tying each artifact to a hiring signal (a capstone review of your M9/M12/M15/M16/M21/M23 breadth).
- **Topics:** code + math screencasts · kernel/systems report · clean GitHub portfolio · a ruthlessly technical cold email + targeted outreach · interview prep using the 20 design studies as system-design ammunition.
- **Source:** `sessions/week-23` / `week-24` · **Action:** reuse · **Builds on:** all prior — especially M13–M14 (capstone), M18/M20 (kernels), M22 (ADRS loop), M21/M23 (design breadth) · **Cost:** $0.

---

## Source inventory & coverage map

Every source directory on disk, and the module that consumes it. This section exists so no built material is ever silently orphaned again.

### Frontier session weeks (`sessions/week-*`)

| Week folder | Files | Consumed by |
|---|---|---|
| `week-01` | 6 | M8 |
| `week-02` | 6 | M10a |
| `week-03` | 6 | M9 (d1–3 reuse; d4–6 optional) |
| `week-04` | 6 | M10b + M10c (bridge) |
| `week-05` | 6 | M11 |
| `week-06` | 6 | M12 |
| `week-07` | 6 | M16a |
| `week-08` | 2 | M13 (profiling + systems review) |
| `week-09` | 6 | M13 |
| `week-10` | 6 | M13 (d1–3) + M14 (d4–6) |
| `week-11` | 6 | M14 (d1–3) + M18a (d4–6) |
| `week-12` | 2 | M18a (consolidation) |
| `week-13` | 6 | M18b |
| `week-14` | 6 | M20a |
| `week-15` | 6 | M20b |
| `week-16` | 6 | M20c |
| `week-17` | 1 | M18a (**optional polish** — previously orphaned) |
| `week-18` | — | **never existed** (not in the 24-week plan) |
| `week-19` | 6 | M22 |
| `week-20` | 6 | M22 |
| `week-21` | 1 | M24a (on-ramp; +5 authored) |
| `week-22` | — | **never existed** (not in the 24-week plan) |
| `week-23` | 4 | M24b |
| `week-24` | 4 | M24b |
| `week-f1`…`week-f5` | 6/4/4/5/1 | M1 (f1), M2–M3 (f2+f3), M4 (f4), M5 (f5) |

### Foundation course modules (`00-` … `10-`)

| Module | Disposition |
|---|---|
| `00-neural-networks` | ✅ **In spiral** → M2–M3 (fundamentals 01–08), M5 (09–10), M7 (cnn) |
| `01-transformers` | ✅ **In spiral** → M4 (days 01–17), M6a (days 18–28) |
| `02-fine-tuning` | ✅ **In spiral** → **M17** (LoRA/QLoRA/full-FT/instruction-tuning) |
| `03-rag` | 🚫 **Out-of-spiral / not covered** *(the RAG **design case study** in M21 is separate and stays)* |
| `04-prompt-engineering` | 🚫 **Out-of-spiral / not covered** |
| `05-multimodal` | ✅ **Partial** → M7 (vision-language); `audio-language` out-of-spiral |
| `06-evaluation` | ✅ **In spiral** → **M22** (metrics + benchmarks for the verifier/holdout) |
| `07-deployment` | 🚫 **Out-of-spiral / not covered** *(M12 covers serving from frontier `week-06`)* |
| `08-ai-agents` | 🚫 **Out-of-spiral / not covered** |
| `09-engineering-productivity` | 🚫 **Out-of-spiral / not covered** |
| `10-reinforcement-learning` | ✅ **Partial** → M17 (`rlhf` subfolder); other RL subfolders out-of-spiral |

> **Out-of-spiral** means "intentionally not part of the frontier-staff journey" — kept in the repo as a standalone reference, not a prerequisite. This disposition was chosen deliberately (only fine-tuning and evaluation were needed).

### Design studies

**20 case studies + 2 frameworks.** Frameworks: `ML Design/01-ml-design-prep` and `genAI design/01-intro-and-framework` (they teach the 7-step framework, not a specific system).

| Study | Module |
|---|---|
| visual-search, youtube-video-search, google-street-view | M9 |
| gmail-smart-compose, google-translate | M12 |
| video-recommendation, ad-click-prediction, event-recommendation, similar-listing, personalized-news-feed | M15 |
| people-you-may-know | M16b |
| image-captioning, RAG, realistic-face-generation, high-res-image-synthesis, text-to-image, personalized-headshots, text-to-video | M21 |
| chatgpt-chatbot, harmful-content-detection | M23 |

---

## Build queue — execution & divide-and-conquer

> The actionable queue for finishing the curriculum. The **master build table** above is the module-level status; this section adds the build protocol, the rules for splitting work across sessions, and the finer-grained foundation work packages (individual lessons + viz). **To hand this off to another session, point it at this section.**

### How a picking-up session works
1. **Read first:** `CLAUDE.md` (writing style, Layer rules, COACH), the roadmap block for your target module, and this section.
2. **Claim** a work package (WP): set its status to 🔄 with your session date. Prefer an unclaimed WP whose dependencies are ✅.
3. **Build** the lesson(s) — see the pipeline table for your action type.
4. **Verify (headless — the browser bridge is blocked here):** render the page with **jsdom** (install from the internal npm registry) to confirm no runtime throw and that `d3` degrades gracefully (see the `headless-html-d3-verification` note); run `python3 sessions/lesson_audit.py` and `python3 sessions/coverage_audit.py` if present.
5. **Wire tracking:** add the `quest-id` to `sessions/progress.json`, then `python3 sessions/wire_index.py` so `sessions/index.html` shows it.
6. **Tick off:** flip the WP to ✅ here **and** update the module's Status in the master build table.

### Divide-and-conquer rules
- Each WP touches different files → **safe to run in parallel across sessions.** Never let two sessions take the same WP.
- **One session owns a module's lessons end-to-end** — do not split one module's lessons across sessions (tone/analogy drift, `CLAUDE.md §4`). Different modules in parallel is fine.
- Within a WP, build lessons in the listed order (intra-module dependencies).
- **Priority 0 (foundations) blocks the spine** — finish it before starting M6a+.

### Build pipeline per action type
| Action | How to produce a lesson |
|---|---|
| **author** (foundations, M1–M7) | `cp` the closest existing sibling day in the same `week-f*` folder (inherits the format, scroll-reveal §5, quiz, viz wiring), then replace **only the content** per the ROADMAP topic list. The `frontier-session-coach` skill can generate a fresh day if no sibling fits. Keep Layer-1 beginner voice (`CLAUDE.md §5`); §5 must be a **scroll-reveal** build-up (`lesson-walkthrough-scroll-reveal`), not a click-stepper. |
| **reuse** (most M8–M24) | The source `sessions/week-NN/day-*.html` already exists. Audit/edit it in place to match the module's framing and re-wire its quest-id. Usually light. |
| **wrap** (design studies) | Author a short HTML lesson that frames the concept and links into the full case study under `ML Design/` or `genAI design/`. Don't duplicate the case study. |
| **convert** (M17) | Turn notebook/MD content (`10-reinforcement-learning/rlhf`, `02-fine-tuning`) into HTML lessons in the coach format. |

### Priority 0 — finish the foundations *(do first; ~62% built)*
| WP | Module | Deliverable(s) | Source | Depends on | Status |
|----|--------|----------------|--------|-----------|--------|
| **P0-1** | M1 | `week-f1/day-06-random-seeds.html` (`wf1-d06-seeds`) | Python/NumPy PRNG; bridge to M8 JAX keys | — | ⬜ |
| **P0-2** | M2–M3 | `week-f3/day-04-optimizers.html`, `day-05-learning-rate.html`, `day-06-train-val-test.html` | `00-nn/fundamentals` 06–08 + optimizer content | P0-V3 (day-04 embed) | ⬜ |
| **P0-3** | M4 | `week-f4/day-05-positional.html` (`wf4-d05-position`) | `01-transformers/days` (positional day) | — | ⬜ |
| **P0-4** ⚠️ | M5 | `week-f5/day-02`…`day-06` + `review.html` (5 + gate) | `00-nn/fundamentals` 09–10 | P0-V3 (loss-curve viz) | ⬜ |
| **P0-V1** | viz | `sessions/viz/matmul.html` (M1 d04 embed) | `visual-tutorial/d3-viz.js` `drawMatrix` + contrib highlight | — | ⬜ |
| **P0-V2** | viz | `sessions/viz/broadcasting.html` (M1 d03 embed) | new (resize operands → broadcast rule fires/fails) | — | ⬜ |
| **P0-V3** | viz | `sessions/viz/gradient-descent.html` (M2–M3 d07, M5) | new (loss surface + LR slider + step button) | — | ⬜ |

- **P0-1** (M1 PRNG): why reproducibility matters, seeding NumPy, the *idea* of a PRNG key so JAX's explicit `key`/`split` in M8 isn't cold. `cp week-f1/day-05-logs-and-exponents.html`. No viz needed.
- **P0-2** (M2–M3 learning half — completes the merged module, now 6/9): `day-04` SGD→momentum→Adam on the 2-layer net (embeds `gradient-descent.html`); `day-05` LR intuition (too-big diverges / too-small crawls); `day-06` train/val/test + overfitting demo. `cp week-f3/day-03-training-loop.html`.
- **P0-4** (M5 stub — biggest gap): in order — `day-02` from-scratch backward + wiring; `day-03` mini-batch loop + overfit-one-batch; `day-04` full MNIST train, loss curve + dropout; `day-05` same model in PyTorch (autograd / `nn.Module`); `day-06` why PyTorch mirrors NumPy; then `review.html`. `cp week-f5/day-01-mlp-mnist.html`.
- **P0-V\*:** each viz is self-contained, vendors `./d3.v7.min.js`, degrades gracefully, and is `.build-embed`-compatible (see the [visualization plan](#visualization-plan-live-vs-static)). Reuse the Attention-Lab / `d3-viz.js` primitives; the `d3-viz` / `frontier-d3-visual-lab` skills help.

**P0 exit check:** M1, M2–M3, M4, M5 all ✅ in the master table; every foundation `quest-id` in `progress.json`; `index.html` shows Parts A/B/C complete; all pages pass jsdom verification.

### Priority 1 & 2 — the rest of the spiral
Build in **master-build-table order** (per-module source / action / ~L / status live there; each module's block above carries its "builds-on" and framing). **Claim one module per session.**
- **Priority 1 — spine (mostly `reuse`/`wrap`):** M6a → M16b. Light edits + re-wiring of existing `week-*` lessons.
- **Priority 2 — specialties + ship:** M17 → M24b.

**Heavy / net-new — assign a focused session (not simple reuse):**
- **M19 Generative Modeling** (12L, author from scratch; VAE reparameterization trick is absent from the repo; also build `sessions/viz/diffusion-noising.html`). Suggested split M19a (autoencoders + GAN) / M19b (diffusion).
- **M12** BPE lesson · **M15** two-tower foundation lesson · **M17** RLHF + fine-tuning conversion · **M24a** 5 authored TLA+ lessons.

### Status glance
- **Priority 0 (foundations): 0 / 7 WPs — start here.**
- Priority 1 (spine): 0 / 15 modules · Priority 2 (specialties + ship): 0 / 12 modules.
- **Legend:** ⬜ unclaimed · 🔄 in progress (add session date) · ✅ done & verified · ⛔ blocked (note blocker).

---

## Visualization plan (live vs static)

The learner's benchmark is the **Attention Lab** (`visual-tutorial/d3-viz.js`, `attention-lab-course/`, `sessions/viz/*.html`): manipulate a control → watch the math re-render → reach a felt takeaway.

- **Live interactive** where *manipulation is the lesson.* Embed via the `.build-embed` iframe as the first "Build it up" step, then the static scroll-reveal SVG underneath as the explanation.
- **Static SVG** where a fixed anatomy picture is enough: the scalar→tensor ladder, the single-neuron diagram, "here is the structure" frames.

**The `.build-embed` iframe contract** (from e.g. `sessions/week-03/day-01`):
```html
<div class="build-embed"><iframe src="../viz/<name>.html" title="…" loading="lazy"></iframe>
<div class="cap">↑ interactive — drag the controls. <a href="../viz/<name>.html" target="_blank" rel="noopener">open full screen ↗</a></div></div>
```
`../viz/` resolves from any `sessions/week-*/` folder (including `week-f1..f5`). Each page is self-contained, vendors `./d3.v7.min.js`, and degrades gracefully if d3 is missing.

### Live-viz asset registry

| Viz page | State | Embedded by |
|----------|-------|-------------|
| `transformer-flops.html` | ∘ exists | M9, M13 |
| `kv-cache.html` | ∘ exists | M12 |
| `flash-attention.html` | ∘ exists | M12, M18b |
| `roofline.html` | ∘ exists | M10a |
| `parallelism.html` | ∘ exists | M10b |
| `scaling-laws.html` | ∘ exists | M11, M14 |
| `moe-routing.html` | ∘ exists | M16a |
| `quantization.html` | ∘ exists | M20b |
| `speculative-decoding.html` | ∘ exists | M20c |
| `attention-heatmap.html` | ● built | M4 day-01 |
| `attention-pipeline.html` | ● built | M4 day-03 |
| `softmax-scaling.html` | ● built | M4 day-03 |
| `attention-multihead.html` | ● built | M4 day-04 |
| `gradient-descent.html` | 🆕 to build | M2–M3, M5 |
| `matmul.html` | 🆕 to build | M1 day-04 |
| `broadcasting.html` | 🆕 to build | M1 day-03 |
| `diffusion-noising.html` | 🆕 to build | M19 |

---

## Cross-cutting dependency fixes

These concepts were once used downstream before being taught. Each now has a home:

| Concept | Was assumed by | Now taught in |
|---|---|---|
| Random seeds / PRNG | M8 (JAX keys) | **M1** |
| Dot product / cosine similarity | M4, M9 | **M1** (explicit) |
| Softmax as a named function | M4, cross-entropy | **M2–M3** |
| Optimizers (SGD/momentum/Adam) | M8 (Optax), M10a (Adam 2×) | **M2–M3** |
| Learning-rate intuition, train/val/test | M5+ | **M2–M3** |
| Dropout / regularization | M5+ | **M5** |
| Loss-curve reading | M5+ | **M5** |
| Positional encoding (+shuffle problem) | M6 | **M4** |
| Batch normalization | M7 (ResNets) | **M7** |
| Output-dim arithmetic, data augmentation | M7 | **M7** |
| BPE / subword tokenization | M12, chat | **M12** (authored lesson; awareness note in M4) |
| Fine-tuning (LoRA/QLoRA/SFT) | M17, M21 | **M17** (from `02-fine-tuning`) |
| Evaluation metrics/benchmarks | M22 verifier | **M22** (from `06-evaluation`) |

### Deferred depth (optional — not currently scheduled)

Flagged as genuine frontier-staff foundations but **not** added as net-new modules (kept out of scope for now to prioritize finishing the foundations). If revisited, prefer folding *light-touch* versions into existing lessons rather than new modules:

- **Probability & statistics basics** (likelihood, cross-entropy as divergence, entropy, sampling) → fold a short primer into M1's logs lesson or early M2–M3.
- **Autodiff internals** (tape-based reverse-mode, Jacobian-vector products, `requires_grad`/`grad_fn`) → fold into M2–M3's backprop lesson; M8's `jit`/`grad` assumes it.
- **Einsum / tensor-contraction notation** → a first primer fits in M18b; one page before M8 would help.
- **Numerical stability / precision** (overflow/underflow, mixed precision) → a short note in M7 or M10a.

---

## Known risks / honest caveats

- **Scope stretch (deliberate):** four modules pull in content beyond the original frontier plan — CNN/vision encoders (M7), GNNs (M16b), RLHF/post-training + fine-tuning (M17), and generative modeling (M19). All are sourced from existing repo modules (except M19) and are *required* to make the visual, graph, chatbot, and image builds real rather than hand-waved.
- **M19 must be authored from scratch** (VAE → VQ-VAE → GAN → DDPM → latent diffusion). The source notebooks are intuition-only with no training loops, and the reparameterization trick is nowhere in the repo. This is the densest non-capstone module — split it across two calendar blocks if it overruns.
- **Foundations are only ~62% built.** M5 is a 1/6 stub. Finish the foundation queue above before advancing — the whole spine assumes M2–M6 intuition is solid.
- **TLA+ (M24a)** is the least-familiar toolchain and `week-21` has only 1 day on disk — 5 lessons must be authored. Keep it small (one spec + one refinement), with Hypothesis as a *complement*.
- **Real-hardware cost is the biggest financial risk.** The caps assume a shrunk matrix and disciplined dry-runs; a careless recompile-heavy sweep on a paid TPU can multiply cost. Enforce static shapes (M13) and billing alerts strictly.
- **Calendar honesty:** a true beginner may need ~10–14 months. If M2–M6 are shaky, do not advance.

---

## Start here

1. ✅ Plan approved (this document — the single source of truth for scope **and** execution).
2. 🔄 **Finish the foundations first** — see the [Build queue](#build-queue--execution--divide-and-conquer) → **Priority 0** (M1 PRNG → M2–M3 optimizers/LR/split → M4 positional → M5 stub → the 3 M1/M3 viz).
3. ⬜ Then continue module by module in master-build-table order (Priority 1 → 2); gate on the "can I explain it?" check before advancing.
4. ⬜ After M7, you have full foundations; M8 begins the frontier spine.

> **Index map:** `sessions/index.html` is the Capability Spiral map. It links built lessons + reused frontier lessons and shows planned modules with their sources, reading progress from `localStorage` per quest-id.
