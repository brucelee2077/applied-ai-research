# Roadmap — The Capability Spiral

> **Goal:** become a **staff researcher at a frontier AI research lab.**
> **Shape:** one integrated plan that takes a total beginner from "what is an array?" to the artifacts a frontier lab hires on — a from-scratch JAX transformer, your own scaling law, custom GPU kernels, an autonomous research loop, and a portfolio of **20 system-design case studies (plus 2 design frameworks)**.
> **This is a single arc, not layers.** Foundations, the frontier curriculum, and the design case studies are woven together. Each module builds only on earlier ones.

> **This file is the single source of truth.** It merges the former narrative roadmap, the former `sessions/MANIFEST.md` build tracker (now removed), and the build-execution queue into one document: the *why* (rationale, dependencies, risks), the *what/where/status* (source, action, files, build status), **and** the *how* (the [Build queue](#build-queue--execution--divide-and-conquer): protocol, divide-and-conquer rules, work packages). There is no other planning doc to keep in sync — edit this file.

---

## How this plan works

The plan threads everything through **8 capabilities you unlock in order**:

**REPRESENT → TRAIN → SCALE → SERVE → OPTIMIZE → GENERATE → AUTOMATE → COMMUNICATE.**

- Each capability is **introduced** with a foundations concept, **deepened** with its matching frontier topic, and **cemented** with a design case study you could not build without the skill you just learned. Design breadth is the *applied checkpoint* of a capability — never a parked add-on.
- **Topics spiral.** You meet the same idea several times, each time deeper. Attention: concept (M3) → exact FLOP/dimension math (M8) → KV-cache serving (M16a) → FlashAttention tiling (M15a) → long-context decoding (M17b). Hardware: single-device (M9a) → parallel (M9c) → real (M9d) → kernel roofline (M15a).
- **No forward dependencies.** Every module lists what earlier module(s) it stands on. If a module ever feels impossible, the fix is to go back one module — not to push through.

### 29 anchors, ~44 build-modules, 10 phases

There are **29 capability anchors** (labelled M1–M29), grouped into **10 phases** — the milestones of the spiral. Each anchor is "one clear job": large anchors **split into lettered sub-modules** (M5a/b, M9a–e, M14a/b, M15a/b, M17a–c, M21a/b, M22a–c, M27a/b), and the thin one-neuron and loss/gradient modules were **merged** into the single **M2**. The v2 → v3 pass added **12 ⭐ modules** to hit the frontier-lab staff-researcher bar (training stability, fp8, orchestration, research method, RL foundations, reasoning-RL, eval science, interpretability, red-teaming, serving-in-practice, state-space models, flow matching) — roughly 40% of that new content is wired in from already-built out-of-spiral folders. **~44 build-modules, ~250 short lessons.**

> **Why the split matters:** an early draft hid, for example, "close the multi-week capstone" *and* "learn GPU kernels from zero" inside one overloaded module. A beginner should never be dropped into two hard things at once. Splitting makes each unit one clear job.

### Read this before you start: this is self-paced

These are **self-paced modules, not calendar weeks.** A module is a themed milestone that takes the calendar time *you* need. At 20+ hours/week, a true beginner should honestly expect **~12–16 months** end to end (the v3 expansion adds depth). Foundations (M1–M7) are the slowest per concept — budget 2–3 weeks each. That is normal and expected, not falling behind.

**The hard rule:** if **M2** (the neuron & how it learns) through **M5b** (attention, transformers) feel shaky, do **not** advance to M8+. The entire frontier spine assumes that intuition is solid.

---

## Global rules (apply to every module)

**Pacing**
- 5 study days per week + 1–2 rest/catch-up days. Never put new heavy content on a catch-up day.
- If a day's step isn't finished, it rolls to the catch-up day — not into the next new topic.
- One atomic build per day: one function, one figure, one small experiment. **Lesson counts per module vary from 2 to 10** — that is by design. A module is exactly as many short lessons as the topic needs.
- Expect a full catch-up week after the capstone (M13), after generative modeling (M22a–c), and after native kernels (M15b).

**Cost (set caps BEFORE any job runs)**
- Most modules target **$0** (simulated / toy scale).
- Real-hardware caps: **M9d** first real TPU ~**$50**; **M12–M13** capstone + IsoFlops sweep ~**$150–250** total; **M14b/M15a/M15b** MoE-sweep + kernels ~**$40**; **M17a** quant small; **M26** ADRS loop ~**$40** with a per-run token ceiling.
- Every paid job gets a toy-scale dry-run first. Set billing alerts at 50% and 90% of each cap.
- Request GPU/TPU quota one full module *before* you need it (approval can take days). Provision the persistent capstone TPU during Phase 1, before the M12 capstone.
- Checkpoint to durable storage — a run that isn't checkpointed doesn't count as done.

**Kernel bar:** "numerically correct first." Matching a reference within tolerance is a pass. Beating the vendor primitive is always an *optional* stretch.

**Live-interactive bar:** every foundation lesson whose topic *rewards manipulation* opens its "Build it up" section with a **live interactive** (`sessions/viz/*.html`, embedded via the `.build-embed` iframe), then keeps the static scroll-reveal diagram underneath as the piece-by-piece explanation. Pure "anatomy" frames stay static. See the [visualization plan](#visualization-plan-live-vs-static).

**Staff-lens bar** *(added 2026-07-04)*: any lesson with a Mechanism/math section needs a named **failure-mode** callout and a named **trade-off** callout, plus at least one diagnostic-style quiz question — see the Staff Lens rule in `.claude/skills/frontier-session-coach/TEACHING_PRINCIPLES.md`. Model: `attention-lab-course/modules/04-the-math.html`.

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

**Tracking:** HTML lessons → `sessions/progress.json` (`w-d` quest ids). Experiment notebooks → COACH (`coach/core.py` ids). This roadmap is self-contained — the [Build queue](#build-queue--execution--divide-and-conquer) below is the dividable remaining-work queue.

---

## Master build table

The authoritative build target. `~L` = expected short-lesson count. `Src` = primary source (frontier `week-*` folder, foundation dir `00–10`, or design study). Every count is `reuse`+`author` where relevant.

| M | Build-module | Cap. | Source | Action | ~L | Cost | Status |
|---|---|---|---|---|----|------|--------|
| **Phase 0 · Foundations — The Neural Net From Scratch** | | | | | | | |
| 1 | Numbers, Arrays & Shapes | Represent | Python/NumPy + light math | author | 6 | $0 | ✅ 6/6 + gate |
| 2 | The Neuron & How It Learns *(merged)* | Represent→Train | `00-nn/fundamentals` 01–08 | author | 9 | $0 | ✅ 9/9 + gates |
| 3 | Attention as a Pure Idea *(+ RoPE build)* | Represent | `01-transformers/days` 01–17 | author | 6 | $0 | ✅ 5/5 + gate |
| 4 | MLP → PyTorch *(+ einsum primer)* | Train | `…fundamentals` 09–10 | author | 6 | $0 | ✅ 6/6 + gate |
| 5a | Full Text Transformer *(+ modern components)* | Represent | `01-transformers/days` 18–28 | author | 8 | $0 | ✅ 8/8 + gate |
| 5b | Vision Transformer (ViT) | Represent | frontier ViT | author | 4 | $0 | ✅ 4/4 + gate |
| 6 | CNNs & Vision Encoders ⭐ | Represent | `00-nn/cnn` + `05-multimodal` | author | 8 | $0 | ✅ 8/8 + gate |
| 7 | Thinking in JAX *(+ autodiff internals)* | Train | `week-01` (+ MLP capstone 🆕) | reuse+author | 6 | $0 | ⬜ |
| **Phase 1 · Hardware, Distribution & Scaling** | | | | | | | |
| 8 | Transformer Math + Search 🎨 *(+ FlashAttention concept)* | Represent | `week-03` + 4 design studies | reuse+wrap | 8 | $0 | ⬜ |
| 9a | Hardware: One Device *(+ activation recomputation)* | Scale | `week-02` | reuse | 5 | $0 | ⬜ |
| 9b | Mixed-Precision & fp8 Training ⭐ | Scale | 🆕 authored (bf16/fp8 numerics) | author | 4 | $0 | ✅ 4/4 + gate |
| 9c | Sharding & Parallelism *(+ modern parallelism depth)* | Scale | `week-04` | reuse+author | 8 | $0 | ⬜ |
| 9d | First Real Multi-Device Run 🔴 | Scale | `week-04` bridge | reuse | 2 | ~$50 | ⬜ |
| 9e | Distributed Orchestration & Fault Tolerance ⭐ | Scale | 🆕 authored (bit-exact resume, preemption) | author | 5 | $0 | ✅ 5/5 + gate |
| 10a | Scaling Laws + Simulator *(+ data curation & mixtures)* | Scale | `week-05` + 🆕 curation | reuse+author | 8 | $0 | ⬜ |
| 10b | Training Stability & Optimization Dynamics ⭐ | Scale | 🆕 authored (spikes, z-loss, muP, LR schedules) | author | 5 | $0 | ✅ 5/5 + gate |
| 11 | Experimental Method & Statistics ⭐ | Scale | 🆕 authored (absorbs prob/stats) | author | 5 | $0 | ✅ 5/5 + gate |
| **Phase 2 · The Capstone (paid)** | | | | | | | |
| 12 | Addition Transformer — Build 🔴 | Train | `week-09` + `week-10`(d1–3) + `week-08` | reuse | 10 | paid TPU | ⬜ |
| 13 | Your Own IsoFlops Law 🔴 | Scale | `week-10`(d4–6) + `week-11`(d1–3) | reuse | 6 | ~$150–250 | ⬜ |
| **Phase 3 · Mixture-of-Experts & Kernels** | | | | | | | |
| 14a | Mixture of Experts *(+ router refinements)* | Scale | `week-07` | reuse | 7 | $0 | ⬜ |
| 14b | Capstone Close-Out: MoE Upgrade + Report | Optimize | `week-11`(d4–6) + `week-12` (+ `week-17`) | reuse | 3 | small TPU | ⬜ |
| 15a | Custom Kernels I: Pallas & FlashAttention *(+ Triton)* | Optimize | `week-13` + on-ramp 🆕 | reuse+author | 6 | small TPU | ⬜ |
| 15b | Native Kernels: C++/CUDA + ThunderKittens 🔴 | Optimize | `week-14` | reuse | 5 | ~$40 | ⬜ |
| **Phase 4 · Serving & Efficiency** | | | | | | | |
| 16a | Inference Economics & Language-Assist 🎨 | Serve | `week-06` + smart-compose, translate + BPE 🆕 | reuse+wrap+author | 8 | $0 | ⬜ |
| 16b | Serving Systems in Practice ⭐ | Serve | 🆕 authored (vLLM/SGLang, prefix cache, routing) | author | 4 | small | ⬜ |
| 17a | Quantization *(+ FP8/AWQ/GPTQ, KV-quant)* | Optimize | `week-15` | reuse+author | 5 | small | ⬜ |
| 17b | Long-Context Decoding | Optimize | `week-16` | reuse | 5 | small | ⬜ |
| 17c | State-Space Models & Attention Alternatives ⭐ | Optimize | 🆕 authored (Mamba/S6, selective scan) | author | 4 | small | ⬜ |
| **Phase 5 · Applied Ranking & Graphs (design)** | | | | | | | |
| 18 | Ranking & Recommendation 🎨 *(+ retrieval depth)* | Serve | 5 design studies (+ authored foundation) | author+wrap | 8 | $0 | ⬜ |
| 19 | Graph Systems & Link Prediction 🎨 | Scale | design: people-you-may-know | wrap | 4 | $0 | ⬜ |
| **Phase 6 · Post-Training, RL & Alignment** | | | | | | | |
| 20 | RL Foundations ⭐ | Train | `10-reinforcement-learning/{fundamentals,policy-gradient}` | convert | 6 | $0 | ⬜ |
| 21a | Core Post-Training: SFT/PEFT/RM/PPO/DPO | Train | `10-rl/rlhf` + `02-fine-tuning` | convert | 8 | $0 | ⬜ |
| 21b | Reasoning-RL & Verifiable Rewards (GRPO/RLVR) ⭐ | Train | `10-rl/rlhf` (grpo) + 🆕 authored | convert+author | 5 | $0 | ⬜ |
| **Phase 7 · Generative Modeling** | | | | | | | |
| 22a | Generative Modeling I: Autoencoders & GANs 🆕 | Generate | 🆕 authored | author | 5 | $0 | ⬜ |
| 22b | Generative Modeling II: Diffusion 🆕 | Generate | 🆕 authored | author | 5 | $0 | ⬜ |
| 22c | Generative Modeling III: Flow Matching ⭐🆕 | Generate | 🆕 authored (rectified/consistency) | author | 3 | $0 | ⬜ |
| 23 | Image & Text Generation Systems 🎨 | Generate | 8 genAI design studies | wrap | 9 | small | ⬜ |
| **Phase 8 · Evaluation, Interpretability & Safety** | | | | | | | |
| 24 | Eval Science & LLM-as-a-Judge ⭐ | Automate | `06-evaluation` (promoted) | convert | 4 | $0 | ⬜ |
| 25 | Interpretability Foundations ⭐ | Automate | 🆕 authored (probes, logit lens, patching) | author | 6 | $0 | ⬜ |
| 26 | AI-Driven Research (ADRS) | Automate | `week-19` + `week-20` + `frontier-lab/week19` | reuse | 8 | ~$40 | ⬜ |
| 27a | Assistant & Safety Systems 🎨 | Automate | genAI chatbot; harmful-content-detection | wrap | 6 | $0 | ⬜ |
| 27b | Red-Teaming & Adversarial Robustness ⭐ | Automate | 🆕 authored (reuses M26 sandbox) | author | 4 | $0 | ⬜ |
| **Phase 9 · Ship & Communicate** | | | | | | | |
| 28 | Formal Methods: TLA+ & Property Testing | Communicate | `week-21` (1 reuse + 5 authored) | reuse+author | 6 | $0 | ⬜ |
| 29 | Ship & Outreach *(+ paper reproduction)* | Communicate | `week-23` / `week-24` | reuse | 7 | $0 | ⬜ |

**Totals:** 29 anchors · ~44 build-modules · **~250 short lessons** across 10 phases. (⭐ = 12 modules added/promoted in the frontier-readiness pass; ~40% of new content is wired in from already-built out-of-spiral folders — RL stack, `06-evaluation`, `02-fine-tuning`.)

> **Viz column moved out of this table for width** — see the [Live-viz asset registry](#live-viz-asset-registry). Ordering note: some phases are grouped thematically but sequenced for prerequisites — e.g. MoE (14a) precedes the MoE capstone close-out (14b); the capstone (Phase 2) comes before serving/efficiency (Phase 4) because it doesn't depend on them and is the motivating culmination of the training arc.

> **Lesson-count honesty (thin-week notes):** M12's 10 lessons = `week-09` (6) + `week-10` d01–03 (3) + `week-08` profiling curated to ~1. M14b's 3 lessons *curate* 5 source days (`week-11` d04–06 + `week-12`) down to 3. M28's 6 lessons = 1 reused (`week-21` TLA+ on-ramp) **+ 5 authored**. Where a module authors more than it reuses, that is called out in its detail block.

---

## The modules

> **Foundations are grouped into three Parts (A/B/C).** Each Part is a coherent unit; the modules inside are its chapters — so daily lessons stay bite-sized while the roadmap reads as evenly-weighted phases, not a string of small modules.

### Foundations · Part A — The Neural Net From Scratch

*Capability: REPRESENT → TRAIN. Build a working neural network from nothing but NumPy — the data and math it runs on (M1), then a single neuron and exactly how it learns (M2).*

#### M1 — Numbers, Arrays, and the Shape of Data · 6 lessons
- **Goal:** get fluent enough in Python/NumPy and light math that every later shape-and-matmul statement reads as English.
- **You build:** a NumPy notebook that loads a small dataset, reshapes and broadcasts arrays, does matrix multiplies by hand and with NumPy (printing every shape), plus a one-page logs/exponents worksheet.
- **Topics:** arrays, indexing, slicing · broadcasting & dtypes · matrix multiply & shape bookkeeping · **vectors, dot products, cosine similarity** (folded into matmul) · **random seeds / PRNG** (needed before M3 embeddings and M7 JAX keys) · logs/exponents (for power laws later).
- **Source:** Python + NumPy warm-up; light-math primer · **Action:** author · **Builds on:** nothing — start here · **Cost:** $0.
- *(Fix: a random-seeds lesson so PRNG isn't first met cold in JAX; teach dot-product/cosine explicitly, not just implicitly inside matmul.)*

| Day | Lesson | File | quest id | Status |
|-----|--------|------|----------|--------|
| 1 | Arrays & the shape of data | `week-f1/day-01-arrays.html` | `wf1-d01-arrays` | ✅ done |
| 2 | Indexing & slicing | `week-f1/day-02-indexing-slicing.html` | `wf1-d02-indexing` | ✅ done |
| 3 | Broadcasting & dtypes | `week-f1/day-03-broadcasting-dtypes.html` | `wf1-d03-broadcasting` | ✅ done (✅ embedded `broadcasting.html`) |
| 4 | Matmul & shapes | `week-f1/day-04-matmul-and-shapes.html` | `wf1-d04-matmul` | ✅ done (✅ embedded `matmul.html`) |
| 5 | Logs & exponents | `week-f1/day-05-logs-and-exponents.html` | `wf1-d05-logs` | ✅ done |
| 6 | **Random seeds / PRNG** | `week-f1/day-06-random-seeds.html` | `wf1-d06-seeds` | ✅ done |
| — | Review gate | `week-f1/review.html` | `wf1-review` | ✅ done |

#### M2 — The Neuron and How It Learns · 9 lessons *(merged: forward pass + how it learns)*
- **Goal:** build a single neuron, stack neurons into a layer and run a forward pass — then learn exactly how that network measures error and updates its weights. The complete from-scratch story of one learning unit, in one module.
- **You build:** a from-scratch neuron (weighted sum + bias + activation) → a one-layer forward pass in NumPy (printing every shape) → MSE + cross-entropy loss → backprop through a 2-layer net → SGD → momentum → Adam on the same net (watching the loss curve) → a train/val/test split with an overfitting demo.
- **Topics:** what a neural network is · neuron = weighted sum + bias · activations (ReLU, sigmoid) · a layer as a matrix multiply · forward propagation · loss (MSE, cross-entropy, and **softmax as a named function**) · gradients as slopes · backprop as the chain rule · **optimizers: SGD → momentum → Adam** · **learning-rate intuition** · **train/val/test split & overfitting** · the training-loop skeleton.
- **Source:** `00-neural-networks/fundamentals` 01–08 (spans `week-f2` + `week-f3`) · **Action:** author · **Builds on:** M1 · **Cost:** $0.
- *(Merge rationale: a thin 4-lesson forward-pass module and a 6-lesson backward-pass module are one continuous story — a neuron and how it learns — so they are merged into this single ~9-lesson module. Softmax lands here; optimizers (esp. Adam) are taught here, before Optax in M7 and before Adam's 2× optimizer-state memory in M9a.)*

| Day | Lesson | File | quest id | Status |
|-----|--------|------|----------|--------|
| 1 | Single neuron | `week-f2/day-01-single-neuron.html` | `wf2-d01-neuron` | ✅ done |
| 2 | Activations | `week-f2/day-02-activations.html` | `wf2-d02-activations` | ✅ done |
| 3 | Layers & forward pass | `week-f2/day-03-layers-forward-pass.html` | `wf2-d03-layers` | ✅ done |
| 4 | Loss (MSE, cross-entropy, softmax) | `week-f3/day-01-loss.html` | `wf3-d01-loss` | ✅ done |
| 5 | Gradients & backprop | `week-f3/day-02-gradients-backprop.html` | `wf3-d02-backprop` | ✅ done |
| 6 | The training loop | `week-f3/day-03-training-loop.html` | `wf3-d03-loop` | ✅ done |
| 7 | **Optimizers: SGD → momentum → Adam** | `week-f3/day-04-optimizers.html` | `wf3-d04-optimizers` | ✅ done (✅ embedded `gradient-descent.html`) |
| 8 | **Learning-rate intuition** | `week-f3/day-05-learning-rate.html` | `wf3-d05-lr` | ✅ done |
| 9 | **Train/val/test split & overfitting** | `week-f3/day-06-train-val-test.html` | `wf3-d06-split` | ✅ done |
| — | Review gates | `week-f2/review.html` · `week-f3/review.html` | `wf2-review` · `wf3-review` | ✅ done |

### Foundations · Part B — Attention & First Models

*Capability: REPRESENT → TRAIN. Meet attention as a pure idea (M3), then train your first real model and see what a framework gives you (M4).*

#### M3 — Attention as a Pure Idea · 6 lessons
- **Goal:** learn embeddings, self-attention, **and position** as concepts so no later transformer build depends on an unseen idea.
- **You build:** from-scratch scaled-dot-product attention on a toy sequence — embeddings, Q/K/V, scores, softmax, weighted sum, multi-head — printing every matrix shape; then a positional-encoding demo showing the "shuffle problem."
- **Topics:** embeddings · Query/Key/Value intuition · attention scores & √d_k scaling & softmax · multi-head attention · **the shuffle problem / permutation invariance** · **positional encoding — implement sinusoidal + a real RoPE build** (ALiBi/NTK/YaRN as awareness) · a short **"tokens are the bridge between text and embeddings"** note (so BPE in M16a isn't a cold surprise).
- **Source:** `01-transformers/days` 01–17 · **Action:** author · **Builds on:** M1, M2 · **Cost:** $0.
- **Live viz:** `attention-heatmap` (day-01), `attention-pipeline` + `softmax-scaling` (day-03), `attention-multihead` (day-04) — reuse the Attention Lab's verified D3 logic.
- *(Fix: positional encoding was orphaned — assumed known by M5, built nowhere. It lives here now.)*

| Day | Lesson | File | quest id | Live viz | Status |
|-----|--------|------|----------|----------|--------|
| 1 | Embeddings (words → numbers) | `week-f4/day-01-embeddings.html` | `wf4-d01-embeddings` | `attention-heatmap.html` | ✅ built · viz verified |
| 2 | Query / Key / Value | `week-f4/day-02-qkv.html` | `wf4-d02-qkv` | — (static cards) | ✅ built |
| 3 | Attention scores, √d_k & softmax | `week-f4/day-03-attention-scores.html` | `wf4-d03-scores` | `attention-pipeline` + `softmax-scaling` | ✅ built · viz verified |
| 4 | Multi-head attention | `week-f4/day-04-multihead.html` | `wf4-d04-multihead` | `attention-multihead.html` | ✅ built · viz verified |
| 5 | **Positional encoding (+ shuffle problem, RoPE build)** | `week-f4/day-05-positional.html` | `wf4-d05-position` | (static SVG) | ✅ done (RoPE = awareness) |
| — | Review gate | `week-f4/review.html` | `wf4-review` | — | ✅ done |

#### M4 — Your First From-Scratch Model: an MLP, Then the PyTorch Version · 6 lessons
- **Goal:** train a plain MLP on MNIST from scratch, then rebuild it in PyTorch to see what a framework gives you for free.
- **You build:** forward pass → backward + wiring → mini-batch loop with an overfit-one-batch sanity check → full MNIST train with a **loss-curve read and dropout** → the same model re-implemented in PyTorch → a review gate.
- **Topics:** from-scratch MLP forward+backward · mini-batch loop · overfit-one-batch sanity check · **reading a loss curve** · **dropout / basic regularization** · the same model in PyTorch (autograd / nn.Module / optimizer) · why PyTorch tensors mirror NumPy.
- **Source:** `00-neural-networks/fundamentals` 09–10 · **Action:** author · **Builds on:** M2 (M3 seeds intuition) · **Cost:** $0.
- **✅ Status: 6/6 + gate — complete.** All six days built (`day-01-mlp-mnist` … `day-06-why-pytorch`) + `review.html`: from-scratch backward pass, mini-batch loop + overfit-one-batch, full training with loss-curve reading + dropout, PyTorch re-implementation, and why PyTorch mirrors NumPy. `day-04` embeds `gradient-descent.html`. *(Note: the revamp's "+einsum primer" is not yet in these lessons — flag for follow-up.)*

### Foundations · Part C — Full Architectures

*Capability: REPRESENT. Assemble complete architectures — the full text transformer and a vision transformer (M5a/M5b), then CNNs and vision encoders (M6).*

#### M5a — The Full Text Transformer · 8 lessons *(split from M5)*
- **Goal:** assemble a complete decoder-only transformer that generates text.
- **You build:** a full decoder-only block (skip connections, layer norm, feed-forward, causal masking) that generates text, one sublayer at a time.
- **Topics:** skip connections & layer norm · feed-forward sublayer · the full block · causal masking · encoder vs decoder · text generation / sampling.
- **Source:** `01-transformers/days` 18–28 · **Action:** author · **Builds on:** M3–M4 · **Cost:** $0.

#### M5b — A Vision Transformer (ViT) · 4 lessons *(split from M5)*
- **Goal:** build the Vision Transformer the frontier Week-1 exercise assumed — correctly deferred until *after* attention and the text transformer.
- **You build:** a small ViT on CIFAR-10 image patches (patch embeddings → the same transformer block from M5a → a classifier head).
- **Topics:** images as patches · patch embeddings · reusing the transformer block for vision · classification head.
- **Source:** the frontier Phase-1 ViT exercise, re-ordered here · **Action:** author · **Builds on:** M5a · **Cost:** $0.
- *(Optional consolidation, deferred: M5b could fold into M5a as one "Transformers: Text & Vision" module if the shared attention block is genuine reuse. Kept split for now.)*

#### M6 — Seeing With Convolutions: CNNs and Vision Encoders ⭐ · 8 lessons
- **Goal:** learn how images become embeddings so the visual design studies are real hands-on builds, not hand-waving.
- **You build:** a from-scratch convolution + pooling op → **output-dimension arithmetic** → a small CNN classifier → **batch normalization** → **data augmentation** → transfer learning → fixed-length image embeddings → a toy CLIP-style contrastive demo.
- **Topics:** convolution & pooling · **output-dim arithmetic (stride/padding)** · a complete CNN classifier · **batch normalization** · **data augmentation** · transfer learning → image embeddings · object-detection intuition (awareness) · contrastive image-text (CLIP) as its own lesson.
- **Source:** `00-neural-networks/cnn` + `05-multimodal/vision-language` · **Action:** author · **Builds on:** M4–M5 · **Cost:** $0.
- *(⭐ closes the vision-encoder gap. Fix: batchnorm, output-dim arithmetic, and augmentation were missing across M1–M6.)*

### Foundations · Part D — Into JAX

*Capability: TRAIN. Re-express your MLP in the JAX/Flax substrate the whole frontier spine runs on.*

#### M7 — Thinking in JAX · 6 lessons
- **Goal:** rebuild the MLP in pure JAX/Flax to internalize immutability, PRNG keys, vmap, and jit — the substrate for the whole spine.
- **You build:** a simple **MLP** trained in Flax/Optax on CIFAR-10, with vmap/jit speedup experiments; 🆕 the MLP capstone lesson is authored; **+ an autodiff-internals lesson** (tape-based reverse-mode, JVP/VJP, why reverse-mode gives O(1) gradients — `requires_grad`/`grad_fn`).
- **Topics:** array immutability vs NumPy · PRNG keys & explicit splitting · vmap vectorization · jit / XLA compilation · Flax nn.Module + Optax · **autodiff internals**.
- **Source:** `sessions/week-01` (+ authored MLP capstone 🆕 + autodiff lesson) · **Action:** reuse+author · **Builds on:** M2, M4, M5a (and M1 for the array + PRNG contrast) · **Cost:** $0.
- *(First JAX build is an MLP, not a ViT — the ViT lives in M5b. Absorbs the deferred autodiff-internals gap.)*

### Phase 1 · Hardware, Distribution & Scaling  *(SCALE)*

#### M8 — Transformer Math and the Embedding Search Engine 🎨 · 8 lessons
- **Goal:** derive exact FLOP/dimension formulas, then build embedding-search systems as the applied checkpoint.
- **You build:** pen-and-code C≈6ND, exact Q/K/V/FFN dims, KV-cache size for a 7B model; a **FlashAttention-as-concept** lesson (memory wall, exact-not-approximate, why tiling); then visual-search, YouTube-search, and Street-View-blur pipelines via the 7-step design framework.
- **Source:** `week-03` + design: ml-design-prep *(fw)*, visual-search, youtube-video-search, street-view · **Action:** reuse+wrap · **Builds on:** M5–M7 · **Cost:** $0.

#### M9a — Hardware Physics: One Device · 5 lessons
- **You build:** roofline & arithmetic-intensity for your ViT; a memory calculator (weights + Adam 2× + activations); **activation recomputation (remat)** as the memory/compute trade.
- **Source:** `week-02` (rename its "distributed-checkpointing" day → **Activation Recomputation** — it teaches `jax.checkpoint`, not run recovery) · **Action:** reuse · **Builds on:** M8, M2 (Adam). · **Cost:** $0.

#### M9b — Mixed-Precision & fp8 Training ⭐ · 4 lessons *(NEW)*
- **Goal:** understand the numerics that let frontier runs use half/eighth precision without diverging.
- **You build:** bf16 vs fp16 (dynamic range), fp32 master weights + loss scaling, accumulation dtype, then fp8 (E4M3/E5M2, per-tensor/block/delayed scaling) with a failure-mode demo.
- **Source:** 🆕 authored · **Action:** author · **Builds on:** M9a (memory/precision), M2 (optimizers). · **Cost:** $0. *(Absorbs the deferred numerical-stability gap.)*

#### M9c — Sharding and Parallelism · 8 lessons
- **You build:** sharding math (AllGather/ReduceScatter/AllReduce); DP, FSDP/ZeRO-1/2/3, tensor / **sequence** / pipeline (**1F1B + interleaved**) parallel; **selective recompute, gradient accumulation, composed DP×TP×PP×SP mesh**; a memory/parallelism calculator; distillation as a serving-side lever.
- **Source:** `week-04` (+ authored parallelism-depth lessons) · **Action:** reuse+author · **Builds on:** M9a–M9b. · **Cost:** $0 (simulated).

#### M9d — Your First Real Multi-Device Run 🔴 · 2 lessons
- **You build:** one real sharded run vs the M9c calculator, with dry-run and billing alerts. *(Deliberately small — a cost/quota gate, not a content unit.)*
- **Source:** `week-04` bridge · **Action:** reuse · **Builds on:** M9c · **Cost:** ~$50 cap, quota lead-time, dry-run first.

#### M9e — Distributed Orchestration & Fault Tolerance ⭐ · 5 lessons *(NEW)*
- **Goal:** keep a 10k-chip, month-long run alive — the defining staff systems skill.
- **You build:** multi-host bring-up, mesh→ICI/DCN topology mapping, **async/sharded checkpointing of weights+optimizer+RNG+dataloader position**, bit-exact resume, preemption/elastic recovery, straggler mitigation, goodput measurement.
- **Source:** 🆕 authored · **Action:** author · **Builds on:** M9c–M9d · **Cost:** $0 (design) — **de-risks the paid M12/M13 capstone.**

#### M10a — Scaling Laws + Budget Simulator · 8 lessons
- **You build:** Kaplan→Chinchilla→IsoFlops, power-law fitting, the data wall, a budget→optimal-(N,D) simulator; **plus a data-curation & mixture sub-cluster** (quality filtering, near-dedup MinHash/suffix-array, domain-mixture weighting/DoReMi, multi-epoch trade-off, contamination/decontamination).
- **Source:** `week-05` + 🆕 data-curation lessons · **Action:** reuse+author · **Builds on:** M8 (FLOPs), M1 (logs). · **Cost:** $0 (toy curves).

#### M10b — Training Stability & Optimization Dynamics ⭐ · 5 lessons *(NEW)*
- **Goal:** the single highest-value gap — "your run spiked at 40k steps" must have an answer.
- **You build:** spike diagnosis + recovery (rollback/skip-batch), **z-loss / QK-norm / logit soft-capping / global-norm clipping**, LR schedules (**warmup rationale, WSD vs cosine, LR↔batch coupling**), and **muP + depth-scaled init** for zero-shot HP transfer.
- **Source:** 🆕 authored · **Action:** author · **Builds on:** M2 (optimizers), M9a (precision). · **Cost:** $0.

#### M11 — Experimental Method & Statistics for ML ⭐ · 5 lessons *(NEW)*
- **Goal:** make everything downstream trustworthy — a research discipline, not a byproduct.
- **You build:** one-variable ablations, honest baselines, seed variance, error bars / confidence intervals, "is this delta inside the noise band?"; a compact probability/statistics primer (likelihood, cross-entropy as divergence, sampling).
- **Source:** 🆕 authored · **Action:** author · **Builds on:** M1, M2. · **Cost:** $0. *(Absorbs the deferred prob/stats gap; gates the capstone.)*

### Phase 2 · The Capstone  *(TRAIN & SCALE — paid)*

#### M12 — The Addition Transformer: Build (Part 1) 🔴 · 10 lessons
- **You build:** a custom digit tokenizer, synthetic addition data, a static-shape ~10M-param Flax dense transformer, a config-driven training loop (overfit-one-batch → converge), exact-match eval, and a profiling trace to catch bad TPU compiles.
- **Source:** `week-09` + `week-10`(d1–3) + `week-08` profiling · **Action:** reuse · **Builds on:** M5–M11 · **Cost:** persistent paid TPU, static shapes.

#### M13 — Your Own IsoFlops Scaling Law (Part 2) 🔴 · 6 lessons
- **You build:** an automated launch/monitor/checkpoint harness over a shrunk (N,D) matrix, loss aggregation/cleaning, per-budget parabola fits, and your own derived data-hunger exponent + a billing-alert/dry-run gate.
- **Source:** `week-10`(d4–6) + `week-11`(d1–3) · **Action:** reuse · **Builds on:** M10a (method), M11 (rigor), M12 (model). · **Cost:** ~$150–250 total, durable checkpoints.

### Phase 3 · Mixture-of-Experts & Kernels  *(OPTIMIZE)*

#### M14a — Mixture of Experts · 7 lessons
- **You build:** a toy MoE router (top-k gating + aux load-balancing loss + AllToAll intuition) + refinements (**router-z loss, capacity/token-drop, fine-grained/shared experts, upcycling**) — your reusable artifact for M14b, M15, M26.
- **Source:** `week-07` · **Action:** reuse · **Builds on:** M9c (parallelism), M10a (scaling). · **Cost:** $0.

#### M14b — Capstone Close-Out: MoE Upgrade + Report · 3 lessons
- **You build:** upgrade the dense M12 model to MoE using your M14a router, run a shrunk MoE IsoFlops matrix, derive scaling-law-vs-dense, and ship a scientific report + one-command reproducible repo.
- **Source:** `week-11`(d4–6) + `week-12` (+ `week-17` optional polish) · **Action:** reuse · **Builds on:** M12–M13, M14a · **Cost:** small paid TPU.

#### M15a — Custom Kernels I: Pallas & FlashAttention · 6 lessons
- **You build:** what-is-a-kernel on-ramp → numpy online-softmax → Pallas matmul → FlashAttention-tiled kernel → fused MoE kernel over your own router; **+ a Triton lesson**. Kernel bar: correctness first.
- **Source:** `week-13` + on-ramp 🆕 · **Action:** reuse+author · **Builds on:** M14b, M14a, M9. · **Cost:** small paid TPU.

#### M15b — Native Kernels: C++/CUDA + ThunderKittens 🔴 · 5 lessons
- **You build:** a C++/CUDA on-ramp (threads/blocks/warps, vector-add, naive matmul) → a native ThunderKittens kernel (16×16 tiles, TMA, worker overlap) verified against PyTorch.
- **Source:** `week-14` · **Action:** reuse · **Builds on:** M15a, M4 (PyTorch). · **Cost:** ~$40 cap; catch-up week after.

### Phase 4 · Serving & Efficiency  *(SERVE → OPTIMIZE)*

#### M16a — Inference Economics & Language-Assist 🎨 · 8 lessons
- **You build:** an inference/roofline calculator (prefill vs decode, bandwidth wall, batching economics, **SLO/goodput/Little's-law**); an authored **BPE/subword tokenization** lesson; then two decode-bound serving wraps — Gmail Smart Compose and Google-Translate seq2seq (BLEU).
- **Source:** `week-06` + design: smart-compose, translate + 🆕 BPE · **Action:** reuse+wrap+author · **Builds on:** M8 (KV math), M9. · **Cost:** $0.

#### M16b — Serving Systems in Practice ⭐ · 4 lessons *(NEW)*
- **Goal:** the serving lens has no hands-on artifact today — fix that.
- **You build:** deploy + instrument a real engine (vLLM/SGLang), enable PagedAttention / continuous batching / **prefix (radix) caching**, profile TTFT/ITL/goodput, and reason about fleet routing.
- **Source:** 🆕 authored · **Action:** author · **Builds on:** M16a · **Cost:** small.

#### M17a — Quantization · 5 lessons
- **You build:** int8 theory + LLM.int8 outliers → PTQ + perplexity → **production quant (FP8 / AWQ / GPTQ / SmoothQuant) + a KV-cache-quant lesson** → QuIP/QTIP/AQLM (awareness) → memory economics.
- **Source:** `week-15` (rebalanced toward what labs actually serve) · **Action:** reuse+author · **Builds on:** M16a, M8. · **Cost:** small.

#### M17b — Long-Context Decoding · 5 lessons
- **You build:** KV-cache recap → SnapKV eviction → GQA/MQA → speculative decoding → ring attention (cross-referenced to M9c as a *training* primitive too); capped by the hardware-lottery essay.
- **Source:** `week-16` · **Action:** reuse · **Builds on:** M16a, M17a. · **Cost:** small.

#### M17c — State-Space Models & Attention Alternatives ⭐ · 4 lessons *(NEW)*
- **Goal:** the clearest architecture gap — Mamba/SSMs are live at frontier labs and absent today.
- **You build:** a tiny selective-scan (S6) build, linear attention / RWKV / RetNet awareness, hybrids (Jamba/Griffin), the recurrence↔attention duality, and a "when SSMs lose on recall/copying" table.
- **Source:** 🆕 authored · **Action:** author · **Builds on:** M3 (attention), M17b. · **Cost:** small.

### Phase 5 · Applied Ranking & Graphs  *(SERVE — design checkpoints)*

#### M18 — Ranking and Recommendation at Serving Scale 🎨 · 8 lessons
- **You build:** one authored **two-tower retrieval + ranking foundation** lesson (negative sampling) + a calibration/metrics lesson + **retrieval-as-architecture** (InfoNCE, cross-encoder rerank, ColBERT, Matryoshka); then five wraps — video-rec, ad-click CTR, event-rec, similar-listings, news-feed — each with offline/online metrics + drift.
- **Source:** design: video-rec, ad-click, event-rec, similar-listings, news-feed (+ authored foundation) · **Action:** author+wrap · **Builds on:** M8 (embeddings), M16a (serving). · **Cost:** $0.

#### M19 — Graph Systems and Link Prediction 🎨 · 4 lessons
- **You build:** a People-You-May-Know build — message passing, GNNs, edge/link prediction, negative sampling for graphs.
- **Source:** design: people-you-may-know · **Action:** wrap · **Builds on:** M18 (ranking contrast). · **Cost:** $0.

### Phase 6 · Post-Training, RL & Alignment  *(TRAIN)*

#### M20 — RL Foundations ⭐ · 6 lessons *(NEW — wired in from existing repo)*
- **Goal:** fix the broken prerequisite — M21 assumes PPO/advantage the spiral never taught.
- **You build:** MDP framing, policy gradient / REINFORCE, baselines & variance reduction, advantage/GAE, actor-critic, on- vs off-policy — on toy environments.
- **Source:** `10-reinforcement-learning/{fundamentals,policy-gradient}` (already authored to interview depth) · **Action:** convert · **Builds on:** M2 (loss/gradients). · **Cost:** $0.

#### M21a — Core Post-Training: SFT / PEFT / RM / PPO / DPO · 8 lessons
- **You build:** the pretrain→SFT→RLHF pipeline; **fine-tuning (full, LoRA, QLoRA, instruction tuning)**; a reward model (Bradley-Terry + margin-ranking) split into **RM basics + RM failure modes** (Goodhart/overoptimization, length/sycophancy bias, ensembles); PPO-for-LM (KL penalty/value head); DPO; **a preference & instruction-data-engineering lesson** and an alignment-threat-model map.
- **Source:** `10-rl/rlhf` (Core-depth) + `02-fine-tuning` · **Action:** convert · **Builds on:** M20, M12 (a trained transformer), M2. · **Cost:** $0 (toy).

#### M21b — Reasoning-RL & Verifiable Rewards (GRPO / RLVR) ⭐ · 5 lessons *(NEW)*
- **Goal:** the 2026 frontier core (R1 / o-series) — was mis-scoped as "awareness."
- **You build:** GRPO (critic-free) **as core**, RLVR, PRM vs ORM, long-CoT credit assignment, rejection-sampling/STaR, best-of-N, the RL→distill flywheel; a toy verifiable-reward loop on your M12 model.
- **Source:** `10-rl/rlhf` (grpo) + 🆕 authored · **Action:** convert+author · **Builds on:** M21a. · **Cost:** $0 (toy).

### Phase 7 · Generative Modeling  *(GENERATE)*

#### M22a — Generative Modeling I: Autoencoders & GANs 🆕 · 5 lessons *(author from scratch)*
- **You build:** a trainable VAE (encoder/latent/decoder, **reparameterization trick**, ELBO), a VQ-VAE codebook (**straight-through estimator**), a small GAN (generator vs discriminator, **mode collapse**) — generating simple images.
- **Source:** 🆕 authored (reparam trick is nowhere in the repo) · **Action:** author · **Builds on:** M5–M6, M4. · **Cost:** $0.

#### M22b — Generative Modeling II: Diffusion 🆕 · 5 lessons *(author from scratch)*
- **You build:** a DDPM (forward noising, reverse denoising, noise-prediction loss, schedule) + a latent-diffusion sketch (CLIP/T5 conditioning + classifier-free guidance); build `sessions/viz/diffusion-noising.html`.
- **Source:** 🆕 authored · **Action:** author · **Builds on:** M22a. · **Cost:** $0. *(Catch-up week after Phase 7.)*

#### M22c — Generative Modeling III: Flow Matching ⭐ 🆕 · 3 lessons *(NEW)*
- **You build:** normalizing-flow intuition → flow matching / rectified flow → consistency models — the family powering frontier SD3/Flux-class image & video.
- **Source:** 🆕 authored · **Action:** author · **Builds on:** M22b. · **Cost:** $0.

#### M23 — Image and Text Generation Systems 🎨 · 9 lessons
- **You build:** genAI wraps — image captioning, RAG end-to-end, the diffusion-applied trio (face, high-res, text-to-image), LoRA/DreamBooth headshots, text-to-video (image-to-video → temporal) — with serving/safety trade-offs.
- **Source:** genAI design: intro *(fw)*, captioning, RAG, face, high-res, text-to-image, headshots, text-to-video · **Action:** wrap · **Builds on:** M22 (unblocking prereq), M8 (retrieval→RAG), M16a. · **Cost:** small.

### Phase 8 · Evaluation, Interpretability & Safety  *(AUTOMATE)*

#### M24 — Eval Science & LLM-as-a-Judge ⭐ · 4 lessons *(promoted from existing)*
- **Goal:** the highest-frequency real task — was buried inside the ADRS module.
- **You build:** contamination detection, elicitation/prompt-sensitivity, saturation; **model-graded eval design + judge biases + human calibration**; held-out-by-construction.
- **Source:** `06-evaluation` (promoted; feeds M26) · **Action:** convert · **Builds on:** M21a (where judges/labels originate). · **Cost:** $0.

#### M25 — Interpretability Foundations ⭐ · 6 lessons *(NEW)*
- **Goal:** a flagship Anthropic/GDM direction with **zero** coverage today.
- **You build:** linear probes, the logit lens, activation patching / causal interventions on your own M12 model; awareness of superposition / SAEs / circuits. Working literacy, not production SAEs.
- **Source:** 🆕 authored · **Action:** author · **Builds on:** M12 (a trained model), M5a. · **Cost:** $0.

#### M26 — AI-Driven Research for Systems (ADRS) · 8 lessons
- **You build:** why ADRS needs a reliable verifier → the 5-component loop → evolutionary-search basics → agentic scaffolding over an LLM API → reward hacking as a *gate* → evaluation engineering → a case-study tour (FunSearch/AlphaGeometry/EvoPrompt) → run the loop on your own M14a router or M15 kernel.
- **Source:** `week-19` + `week-20` + `frontier-lab/week19/friday-reward-hacking` · **Action:** reuse · **Builds on:** M24 (eval), M14a/M15 (your artifacts), M21 (reward-hacking intuition). · **Cost:** ~$40 cap.

#### M27a — Assistant and Safety Systems 🎨 · 6 lessons
- **You build:** a "two kinds of safety" framing; a ChatGPT-style chatbot (decoder-only + SFT/RLHF from M21 + serving); a multimodal harmful-content detector (fused text+image, precision/recall, late-vs-early fusion); a drift-monitoring lesson.
- **Source:** genAI chatbot; ML Design harmful-content-detection · **Action:** wrap · **Builds on:** M21 (RLHF), M26 (eval/robustness), M16a, M6. · **Cost:** $0.

#### M27b — Red-Teaming & Adversarial Robustness ⭐ · 4 lessons *(NEW)*
- **Goal:** a named launch-gating job function reduced to one word today.
- **You build:** jailbreak families, prompt injection, automated attack generation, robustness eval; RLAIF / Constitutional AI and scalable-oversight awareness.
- **Source:** 🆕 authored (reuses the M26 eval sandbox) · **Action:** author · **Builds on:** M26, M27a. · **Cost:** $0.

### Phase 9 · Ship & Communicate  *(COMMUNICATE)*

#### M28 — Formal Methods: TLA+ and Property-Based Testing · 6 lessons
- **You build:** a TLA+ on-ramp (states/Init/Next → invariants → TLC model-check + bounding) → one safety-invariant spec for the M26 loop → a refinement map to your Python → Hypothesis property-based testing.
- **Source:** `week-21` (1 reuse + 5 authored) · **Action:** reuse+author · **Builds on:** M26 (the loop to verify). · **Cost:** $0.

#### M29 — Ship and Outreach: Portfolio, Report, Interview · 7 lessons
- **You build:** code + math screencasts → a kernel/systems report → a clean GitHub portfolio → a ruthlessly technical cold email + targeted outreach → staff-researcher interview prep tying each artifact to a hiring signal; **+ a reading/reproducing-papers-as-craft lesson**.
- **Source:** `week-23` / `week-24` · **Action:** reuse · **Builds on:** all prior — especially M12–M13, M15, M26, M23/M27 breadth. · **Cost:** $0.



## Source inventory & coverage map

Every source directory on disk, and the module that consumes it. This section exists so no built material is ever silently orphaned again.

### Frontier session weeks (`sessions/week-*`)

| Week folder | Files | Consumed by |
|---|---|---|
| `week-01` | 6 | M7 |
| `week-02` | 6 | M9a |
| `week-03` | 6 | M8 (d1–3 reuse; d4–6 optional) |
| `week-04` | 6 | M9c + M9d (bridge) |
| `week-05` | 6 | M10a |
| `week-06` | 6 | M16a |
| `week-07` | 6 | M14a |
| `week-08` | 2 | M12 (profiling + systems review) |
| `week-09` | 6 | M12 |
| `week-10` | 6 | M12 (d1–3) + M13 (d4–6) |
| `week-11` | 6 | M13 (d1–3) + M14b (d4–6) |
| `week-12` | 2 | M14b (consolidation) |
| `week-13` | 6 | M15a |
| `week-14` | 6 | M15b |
| `week-15` | 6 | M17a |
| `week-16` | 6 | M17b |
| `week-17` | 1 | M14b (**optional polish** — previously orphaned) |
| `week-18` | — | **never existed** (not in the 24-week plan) |
| `week-19` | 6 | M26 |
| `week-20` | 6 | M26 |
| `week-21` | 1 | M28 (on-ramp; +5 authored) |
| `week-22` | — | **never existed** (not in the 24-week plan) |
| `week-23` | 4 | M29 |
| `week-24` | 4 | M29 |
| `week-f1`…`week-f5` | 6/4/4/5/1 | M1 (f1), M2 (f2+f3), M3 (f4), M4 (f5) |

### Foundation course modules (`00-` … `10-`)

| Module | Disposition |
|---|---|
| `00-neural-networks` | ✅ **In spiral** → M2 (fundamentals 01–08), M4 (09–10), M6 (cnn) |
| `01-transformers` | ✅ **In spiral** → M3 (days 01–17), M5a (days 18–28) |
| `02-fine-tuning` | ✅ **In spiral** → **M21a** (LoRA/QLoRA/full-FT/instruction-tuning) |
| `03-rag` | 🚫 **Out-of-spiral / not covered** *(the RAG **design case study** in M23 is separate and stays)* |
| `04-prompt-engineering` | 🚫 **Out-of-spiral / not covered** |
| `05-multimodal` | ✅ **Partial** → M6 (vision-language); `audio-language` out-of-spiral |
| `06-evaluation` | ✅ **In spiral** → **M24** (promoted to Eval Science; feeds M26) |
| `07-deployment` | 🚫 **Out-of-spiral / not covered** *(M16a/M16b cover serving)* |
| `08-ai-agents` | 🚫 **Out-of-spiral / not covered** |
| `09-engineering-productivity` | 🚫 **Out-of-spiral / not covered** |
| `10-reinforcement-learning` | ✅ **In spiral** → **M20** (fundamentals + policy-gradient), **M21a/M21b** (rlhf/grpo); remaining subfolders out-of-spiral |

> **Out-of-spiral** means "intentionally not part of the frontier-staff journey" — kept in the repo as a standalone reference, not a prerequisite. This disposition was chosen deliberately (only fine-tuning and evaluation were needed).

### Design studies

**20 case studies + 2 frameworks.** Frameworks: `ML Design/01-ml-design-prep` and `genAI design/01-intro-and-framework` (they teach the 7-step framework, not a specific system).

| Study | Module |
|---|---|
| visual-search, youtube-video-search, google-street-view | M8 |
| gmail-smart-compose, google-translate | M16a |
| video-recommendation, ad-click-prediction, event-recommendation, similar-listing, personalized-news-feed | M18 |
| people-you-may-know | M19 |
| image-captioning, RAG, realistic-face-generation, high-res-image-synthesis, text-to-image, personalized-headshots, text-to-video | M23 |
| chatgpt-chatbot, harmful-content-detection | M27a |

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
- **Priority 0 (foundations) blocks the spine** — finish it before starting M5a+.

### Build pipeline per action type
| Action | How to produce a lesson |
|---|---|
| **author** (foundations, M1–M6) | `cp` the closest existing sibling day in the same `week-f*` folder (inherits the format, scroll-reveal §5, quiz, viz wiring), then replace **only the content** per the ROADMAP topic list. The `frontier-session-coach` skill can generate a fresh day if no sibling fits. Keep Layer-1 beginner voice (`CLAUDE.md §5`); §5 must be a **scroll-reveal** build-up (`lesson-walkthrough-scroll-reveal`), not a click-stepper. |
| **reuse** (most M7–M23) | The source `sessions/week-NN/day-*.html` already exists. Audit/edit it in place to match the module's framing and re-wire its quest-id. Usually light. |
| **wrap** (design studies) | Author a short HTML lesson that frames the concept and links into the full case study under `ML Design/` or `genAI design/`. Don't duplicate the case study. |
| **convert** (M20/M21) | Turn notebook/MD content (`10-reinforcement-learning`, `02-fine-tuning`, `06-evaluation`) into HTML lessons in the coach format. |

### Priority 0 — finish the foundations *(do first; ~62% built)*
| WP | Module | Deliverable(s) | Source | Depends on | Status |
|----|--------|----------------|--------|-----------|--------|
| **P0-1** | M1 | `week-f1/day-06-random-seeds.html` (`wf1-d06-seeds`) | Python/NumPy PRNG; bridge to M7 JAX keys | — | ✅ |
| **P0-2** | M2 | `week-f3/day-04-optimizers.html`, `day-05-learning-rate.html`, `day-06-train-val-test.html` | `00-nn/fundamentals` 06–08 + optimizer content | P0-V3 (day-04 embed) | ✅ |
| **P0-3** | M3 | `week-f4/day-05-positional.html` (`wf4-d05-position`) | `01-transformers/days` (positional day) | — | ✅ |
| **P0-4** ⚠️ | M4 | `week-f5/day-02`…`day-06` + `review.html` (5 + gate) | `00-nn/fundamentals` 09–10 | P0-V3 (loss-curve viz) | ✅ |
| **P0-V1** | viz | `sessions/viz/matmul.html` (M1 d04 embed) | `visual-tutorial/d3-viz.js` `drawMatrix` + contrib highlight | — | ✅ |
| **P0-V2** | viz | `sessions/viz/broadcasting.html` (M1 d03 embed) | new (resize operands → broadcast rule fires/fails) | — | ✅ |
| **P0-V3** | viz | `sessions/viz/gradient-descent.html` (M2 d07, M4) | new (loss surface + LR slider + step button) | — | ✅ |

- **P0-1** (M1 PRNG): why reproducibility matters, seeding NumPy, the *idea* of a PRNG key so JAX's explicit `key`/`split` in M7 isn't cold. `cp week-f1/day-05-logs-and-exponents.html`. No viz needed.
- **P0-2** (M2 learning half — completes the merged module, now 6/9): `day-04` SGD→momentum→Adam on the 2-layer net (embeds `gradient-descent.html`); `day-05` LR intuition (too-big diverges / too-small crawls); `day-06` train/val/test + overfitting demo. `cp week-f3/day-03-training-loop.html`.
- **P0-4** (M4 stub — biggest gap): in order — `day-02` from-scratch backward + wiring; `day-03` mini-batch loop + overfit-one-batch; `day-04` full MNIST train, loss curve + dropout; `day-05` same model in PyTorch (autograd / `nn.Module`); `day-06` why PyTorch mirrors NumPy; then `review.html`. `cp week-f5/day-01-mlp-mnist.html`.
- **P0-V\*:** each viz is self-contained, vendors `./d3.v7.min.js`, degrades gracefully, and is `.build-embed`-compatible (see the [visualization plan](#visualization-plan-live-vs-static)). Reuse the Attention-Lab / `d3-viz.js` primitives; the `d3-viz` / `frontier-d3-visual-lab` skills help.

**P0 exit check:** M1, M2, M3, M4 all ✅ in the master table; every foundation `quest-id` in `progress.json`; `index.html` shows Parts A/B/C complete; all pages pass jsdom verification.

### Priority 1 & 2 — the rest of the spiral
Build in **master-build-table order** (per-module source / action / ~L / status live there; each module's block above carries its "builds-on" and framing). **Claim one module per session.**
- **Priority 1 — build the spine (Phases 0–5):** finish foundations, then M5a → M19 in table order — mostly `reuse`/`wrap` of existing `week-*` lessons, plus the ⭐ training-science modules (M9b, M9e, M10b, M11) which gate the paid capstone.
- **Priority 2 — specialties + ship (Phases 6–9):** M20 → M29.

**Heavy / net-new — assign a focused session (not simple reuse):**
- The **12 ⭐ modules** are net-new authoring or promotion: M9b (fp8), M9e (orchestration), M10b (training stability), M11 (research method), M16b (serving-in-practice), M17c (state-space), M20 (RL foundations — wired from `10-rl`), M21b (reasoning-RL), M22c (flow matching), M24 (eval science — from `06-evaluation`), M25 (interpretability), M27b (red-teaming).
- **Generative M22a–c** (author from scratch; VAE reparameterization trick absent; also build `sessions/viz/diffusion-noising.html`).
- Authored lessons inside otherwise-reuse modules: **M16a** BPE · **M18** two-tower foundation · **M21a** RLHF + fine-tuning conversion · **M28** 5 TLA+ lessons.

### Status glance
- **Priority 0 (foundations): 7 / 7 WPs — ✅ complete (M1, M2, M3, M4 + all 3 viz built, verified, wired).**
- Priority 1 (Phases 0–5): 7 / ~30 build-modules (M5a, M5b, M6, M9b, M9e, M10b, M11 ✅) · Priority 2 (Phases 6–9): 0 / ~14 build-modules · **~44 total.**
- **Legend:** ⬜ unclaimed · 🔄 in progress (add session date) · ✅ done & verified · ⛔ blocked (note blocker).

### Staff-lens retrofit watchlist *(added 2026-07-04)*
Not a work package — a deferred check. The Staff Lens bar (Global rules, above) was added
after comparing built lessons against `attention-lab-course`, which already names failure
modes and trade-offs explicitly and asks diagnostic-style quiz questions. Existing lessons
were **not** retrofitted in that pass. When a future session picks up one of the still-⬜
math-heavy modules below, check the topic's already-built source lesson (if `reuse`-sourced)
against the Staff Lens bar and upgrade it opportunistically rather than treating this as
separate mandatory work:

- **M8** — transformer math (`week-03`): transformer arithmetic, QKV, KV-cache
- **M10a** — scaling laws (`week-05`): Kaplan, Chinchilla, IsoFlops methodology
- **M14a** — Mixture of Experts (`week-07`)
- **M17a** — quantization (`week-15`)
- **M17b** — long-context decoding (`week-16`): speculative decoding, KV eviction

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
| `transformer-flops.html` | ∘ exists | M8, M12 |
| `kv-cache.html` | ∘ exists | M16a |
| `flash-attention.html` | ∘ exists | M15a, M16a |
| `roofline.html` | ∘ exists | M9a |
| `parallelism.html` | ∘ exists | M9c |
| `scaling-laws.html` | ∘ exists | M10a, M13 |
| `moe-routing.html` | ∘ exists | M14a |
| `quantization.html` | ∘ exists | M17a |
| `speculative-decoding.html` | ∘ exists | M17b |
| `attention-heatmap.html` | ● built | M3 day-01 |
| `attention-pipeline.html` | ● built | M3 day-03 |
| `softmax-scaling.html` | ● built | M3 day-03 |
| `attention-multihead.html` | ● built | M3 day-04 |
| `gradient-descent.html` | ● built | M2, M4 |
| `matmul.html` | ● built | M1 day-04 |
| `broadcasting.html` | ● built | M1 day-03 |
| `diffusion-noising.html` | 🆕 to build | M22b |

---

## Cross-cutting dependency fixes

These concepts were once used downstream before being taught. Each now has a home:

| Concept | Was assumed by | Now taught in |
|---|---|---|
| Random seeds / PRNG | M7 (JAX keys) | **M1** |
| Dot product / cosine similarity | M3, M8 | **M1** (explicit) |
| Softmax as a named function | M3, cross-entropy | **M2** |
| Optimizers (SGD/momentum/Adam) | M7 (Optax), M9a (Adam 2×) | **M2** |
| Learning-rate intuition, train/val/test | M4+ | **M2** |
| Dropout / regularization | M4+ | **M4** |
| Loss-curve reading | M4+ | **M4** |
| Positional encoding (+shuffle problem) | M5 | **M3** |
| Batch normalization | M6 (ResNets) | **M6** |
| Output-dim arithmetic, data augmentation | M6 | **M6** |
| BPE / subword tokenization | M16a, chat | **M16a** (authored lesson; awareness note in M3) |
| Fine-tuning (LoRA/QLoRA/SFT) | M21a, M23 | **M21a** (from `02-fine-tuning`) |
| Evaluation metrics/benchmarks | M26 verifier | **M24** (from `06-evaluation`) |

### Formerly-deferred gaps — now incorporated (v3)

The four gaps previously parked as "optional" now have real homes in the expanded plan:

- **Probability & statistics basics** (likelihood, cross-entropy as divergence, entropy, sampling) → **M11** (Experimental Method & Statistics).
- **Autodiff internals** (tape-based reverse-mode, JVP/VJP, `requires_grad`/`grad_fn`) → folded into **M7** (Thinking in JAX).
- **Einsum / tensor-contraction notation** → einsum primer folded into **M4** (MLP → PyTorch).
- **Numerical stability / precision** (overflow/underflow, mixed precision, fp8) → **M9b** (Mixed-Precision & fp8 Training).

---

## Known risks / honest caveats

- **Scope stretch (deliberate):** several anchors pull in content beyond the original frontier plan — CNN/vision encoders (M6), GNNs (M19), RL foundations + post-training + fine-tuning (M20/M21), generative modeling (M22), interpretability (M25), red-teaming (M27b). All are sourced from existing repo modules (except the generative and interpretability builds) and are *required* to hit the staff-researcher bar.
- **M22 (generative) must be authored from scratch** (VAE → VQ-VAE → GAN → DDPM → latent diffusion → flow). The source notebooks are intuition-only with no training loops, and the reparameterization trick is nowhere in the repo. Split across two calendar blocks (M22a/b/c) with a catch-up week after.
- **Foundations are only ~62% built.** M4 is a 1/6 stub. Finish the Priority-0 queue before advancing — the whole spine assumes M2–M5b intuition is solid.
- **TLA+ (M28)** is the least-familiar toolchain and `week-21` has only 1 day on disk — 5 lessons must be authored. Keep it small (one spec + one refinement), with Hypothesis as a *complement*.
- **Real-hardware cost is the biggest financial risk.** The caps assume a shrunk matrix and disciplined dry-runs; a careless recompile-heavy sweep on a paid TPU can multiply cost. Enforce static shapes (M12), M9e orchestration discipline, and billing alerts strictly.
- **Calendar honesty:** a true beginner may need ~12–16 months with the v3 depth. If M2–M5b are shaky, do not advance.

---

## Start here

1. ✅ Plan approved (this document — the single source of truth for scope **and** execution).
2. 🔄 **Finish the foundations first** — see the [Build queue](#build-queue--execution--divide-and-conquer) → **Priority 0** (M1 PRNG → M2 optimizers/LR/split → M3 positional → M4 stub → the 3 M1/M2 viz).
3. ⬜ Then continue module by module in master-build-table order (Priority 1 → 2); gate on the "can I explain it?" check before advancing.
4. ⬜ After M7, you have full foundations; M8 begins the frontier spine (Phase 1).

> **Index map:** `sessions/index.html` is the Capability Spiral map. It links built lessons + reused frontier lessons and shows planned modules with their sources, reading progress from `localStorage` per quest-id.
