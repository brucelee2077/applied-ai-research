# 24-Week Frontier AI Lab Transition Curriculum — Link Companion
Source: `Strategic Blueprint for Transitioning to Frontier AI Research Laboratories` uploaded by the user.
Legend: **primary/paper/official/video/repo** = recommended source to study; **support** = supporting implementation or reference; **candidate** = useful but less direct.

## Phase 1 — Functional Compilers and Hardware Mathematics

### Week 1

#### Monday — The JAX Functional Paradigm
**Blueprint reading/viewing:** JAX Official Documentation: Thinking in JAX.

**Study links**
- [Thinking in JAX — JAX docs](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html) `[official]`
- [JAX Quickstart](https://docs.jax.dev/en/latest/quickstart.html) `[official]`

**Practical execution:** Implement foundational array manipulations. Prove the immutability of JAX arrays compared to NumPy.

#### Tuesday — State Management & PRNG
**Blueprint reading/viewing:** JAX PRNG design documentation.

**Study links**
- [JAX PRNG Design Note](https://docs.jax.dev/en/latest/jep/263-prng.html) `[official]`
- [jax.random API](https://docs.jax.dev/en/latest/jax.random.html) `[official]`

**Practical execution:** Build a multi-layer perceptron (MLP) forward pass using raw jax.numpy and explicit key splitting.

#### Wednesday — Vectorization (vmap)
**Blueprint reading/viewing:** JAX documentation on vmap.

**Study links**
- [Automatic Vectorization in JAX](https://docs.jax.dev/en/latest/automatic-vectorization.html) `[official]`
- [jax.vmap API](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) `[official]`

**Practical execution:** Rewrite the MLP to process batched data without utilizing explicit for-loops.

#### Thursday — Compilation (jit)
**Blueprint reading/viewing:** JAX documentation on XLA compilation.

**Study links**
- [JIT Compilation in JAX](https://docs.jax.dev/en/latest/jit-compilation.html) `[official]`
- [jax.jit API](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) `[official]`
- [OpenXLA / XLA documentation](https://openxla.org/xla) `[official]`

**Practical execution:** Benchmark the raw execution time of the batched MLP against the JIT-compiled version.

#### Friday — State Abstraction (Flax & Optax)
**Blueprint reading/viewing:** Flax and Optax introductory guides.

**Study links**
- [Flax Linen basics](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html) `[official]`
- [flax.linen.Module API](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) `[official]`
- [Optax getting started](https://optax.readthedocs.io/en/latest/getting_started.html) `[official]`
- [Optax API reference](https://optax.readthedocs.io/en/latest/api/api.html) `[official]`

**Practical execution:** Refactor the raw JAX MLP using Flax nn.Module and initialize Optax gradient transformations.

#### Saturday — Capstone Architecture
**Blueprint reading/viewing:** The Peterman Pod: Google DeepMind Pre-Training Lead (YouTube: cDyi91onoJ8).

**Study links**
- [Flax Linen basics](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html) `[official]`
- [flax.linen.Module API](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) `[official]`
- [Optax getting started](https://optax.readthedocs.io/en/latest/getting_started.html) `[official]`
- [Optax API reference](https://optax.readthedocs.io/en/latest/api/api.html) `[official]`
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) `[paper]`
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) `[support]`
- [The Peterman Pod — Google DeepMind Pre-Training Lead Vlad Feinberg](https://www.youtube.com/watch?v=cDyi91onoJ8) `[video]`

**Practical execution:** Code a Vision Transformer (ViT) in Flax from scratch. Overfit a single batch of CIFAR-10 data.

#### Sunday — Debugging & Synthesis
**Blueprint reading/viewing:** Review podcast transcript detailing the dichotomy between software engineering and ML research.

**Study links**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) `[paper]`
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) `[support]`
- [Flax Linen basics](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html) `[official]`
- [The Peterman Pod Vlad Feinberg transcript/article](https://www.developing.dev/p/google-deepmind-pre-training-lead) `[support]`
- [The Peterman Pod — Google DeepMind Pre-Training Lead Vlad Feinberg](https://www.youtube.com/watch?v=cDyi91onoJ8) `[video]`

**Practical execution:** Train the ViT to convergence on the full dataset. Document the functional state-passing architecture.

### Week 2

#### Monday — Hardware Physics & Rooflines
**Blueprint reading/viewing:** How to Scale Your Model (The Scaling Book), Chapter 1: A Brief Intro to Roofline Analysis.

**Study links**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) `[paper]`
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) `[support]`
- [Flax Linen basics](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html) `[official]`
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`
- [How to Scale Your Model — The Scaling Book](https://jax-ml.github.io/scaling-book/) `[primary]`

**Practical execution:** Calculate the arithmetic intensity of the ViT. Determine if it is compute-bound or memory-bound.

#### Tuesday — TPU Architecture
**Blueprint reading/viewing:** The Scaling Book, Chapter 2: How to Think About TPUs.

**Study links**
- [Scaling Book: How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) `[primary]`
- [Run a calculation on Cloud TPU VM using JAX](https://docs.cloud.google.com/tpu/docs/run-calculation-jax) `[official]`

**Practical execution:** Diagram the physical difference between TPU matrix multiply units and GPU streaming multiprocessors.

#### Wednesday — Sharding Fundamentals
**Blueprint reading/viewing:** The Scaling Book, Chapter 3: Sharded Matrices and How to Multiply Them.

**Study links**
- [Scaling Book: Sharding](https://jax-ml.github.io/scaling-book/sharding/) `[primary]`
- [Distributed arrays and automatic parallelization in JAX](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) `[official]`

**Practical execution:** Write out the mathematical mechanics of AllGather, ReduceScatter, and AllReduce operations.

#### Thursday — Multi-Device Execution
**Blueprint reading/viewing:** JAX documentation on jax.Array and distributed sharding.

**Study links**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) `[paper]`
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) `[support]`
- [Flax Linen basics](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html) `[official]`
- [Distributed arrays and automatic parallelization in JAX](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) `[official]`
- [jax.pmap API](https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html) `[official]`
- [jax.lax.pmean API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.pmean.html) `[official]`

**Practical execution:** Modify the ViT code to execute across 4 simulated local devices using jax.lax.pmean.

#### Friday — Memory Footprint Theory
**Blueprint reading/viewing:** Hugging Face Ultrascale Playbook: Step 1: Fitting a training step in memory.

**Study links**
- [Hugging Face Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) `[primary]`
- [Scaling Book: Transformer math](https://jax-ml.github.io/scaling-book/transformers/) `[primary]`
- [Gradient checkpointing with jax.checkpoint / remat](https://docs.jax.dev/en/latest/gradient-checkpointing.html) `[official]`
- [Profiling computation in JAX](https://docs.jax.dev/en/latest/profiling.html) `[official]`
- [Scaling Book: Profiling and debugging](https://jax-ml.github.io/scaling-book/profiling/) `[primary]`

**Practical execution:** Map the memory footprint: weights, gradients, optimizer states (Adam's 2x multiplier), and activations.

#### Saturday — Distributed Implementation
**Blueprint reading/viewing:** Ultrascale Playbook: Typical Scales in LLM Training.

**Study links**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) `[paper]`
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) `[support]`
- [Flax Linen basics](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html) `[official]`
- [Hugging Face Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) `[primary]`
- [Scaling Book: Transformer math](https://jax-ml.github.io/scaling-book/transformers/) `[primary]`
- [Gradient checkpointing with jax.checkpoint / remat](https://docs.jax.dev/en/latest/gradient-checkpointing.html) `[official]`
- [Hugging Face nanotron repository](https://github.com/huggingface/nanotron) `[repo]`

**Practical execution:** Provision a multi-GPU cloud instance. Deploy the sharded ViT model across physical hardware.

#### Sunday — Activation Checkpointing
**Blueprint reading/viewing:** Research papers on gradient checkpointing techniques.

**Study links**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) `[paper]`
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) `[support]`
- [Flax Linen basics](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html) `[official]`
- [Gradient checkpointing with jax.checkpoint / remat](https://docs.jax.dev/en/latest/gradient-checkpointing.html) `[official]`
- [jax.checkpoint API](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html) `[official]`

**Practical execution:** Implement activation recomputation in the JAX ViT to manually trade compute cycles for memory savings.

### Week 3

#### Monday — Transformer Arithmetic
**Blueprint reading/viewing:** The Scaling Book, Chapter 4: All the Transformer Math You Need to Know.

**Study links**
- [Scaling Book: Transformer math](https://jax-ml.github.io/scaling-book/transformers/) `[primary]`
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) `[paper]`

**Practical execution:** Derive the base FLOP equation: C ≈ 6ND, where N is parameters and D is tokens.

#### Tuesday — Matrix Anatomies
**Blueprint reading/viewing:** The Scaling Book, Chapter 4 continued.

**Study links**
- [Scaling Book: Transformer math](https://jax-ml.github.io/scaling-book/transformers/) `[primary]`
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) `[paper]`

**Practical execution:** Calculate the exact matrix dimensions for the Q, K, V projections and Feed-Forward networks.

#### Wednesday — The KV Cache
**Blueprint reading/viewing:** Research literature on auto-regressive generation bottlenecks.

**Study links**
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`
- [Efficiently Scaling Transformer Inference — Pope et al.](https://arxiv.org/abs/2211.05102) `[paper]`
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) `[paper]`
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) `[paper]`
- [bitsandbytes GitHub repository](https://github.com/bitsandbytes-foundation/bitsandbytes) `[repo]`
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) `[paper]`

**Practical execution:** Mathematically model the memory size of the KV cache for a 7B parameter model at a sequence length of 8,192.

#### Thursday — Production Code Review
**Blueprint reading/viewing:** The nanotron and picotron repositories.

**Study links**
- [Hugging Face nanotron repository](https://github.com/huggingface/nanotron) `[repo]`
- [Hugging Face Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) `[primary]`
- [Hugging Face picotron repository](https://github.com/huggingface/picotron) `[repo]`
- [Scaling Book: Parallelize a Transformer for training](https://jax-ml.github.io/scaling-book/training/) `[primary]`
- [Optax getting started](https://optax.readthedocs.io/en/latest/getting_started.html) `[official]`
- [Flax Linen basics](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html) `[official]`
- [JIT Compilation in JAX](https://docs.jax.dev/en/latest/jit-compilation.html) `[official]`

**Practical execution:** Trace the execution flow of a production-grade distributed training loop without high-level abstractions.

#### Friday — Advanced Pretraining Logic
**Blueprint reading/viewing:** Vlad Feinberg Blog: Gemini Flash Pretraining.

**Study links**
- [Vlad Feinberg — Gemini Flash Pretraining](https://vladfeinberg.com/2025/04/24/gemini-flash-pretraining.html) `[primary]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`

**Practical execution:** Review the Princeton COS 568 lecture slides on ML systems and scaling constraints.

#### Saturday — Mathematical Verification
**Blueprint reading/viewing:** The Scaling Book exercises.

**Study links**
- [How to Scale Your Model — The Scaling Book](https://jax-ml.github.io/scaling-book/) `[primary]`
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`
- [Scaling Book: Transformer math](https://jax-ml.github.io/scaling-book/transformers/) `[primary]`

**Practical execution:** Complete every paper-and-pencil exercise from Chapters 1-4 of the Scaling Book manually.

#### Sunday — Screencast Rehearsal
**Blueprint reading/viewing:** Dwarkesh Podcast featuring Reiner Pope.

**Study links**
- [Dwarkesh Podcast — Reiner Pope](https://www.dwarkesh.com/p/reiner-pope) `[primary]`
- [Dwarkesh Podcast Reiner Pope — YouTube](https://www.youtube.com/watch?v=xmkSf5IS-zw) `[video]`
- [Efficiently Scaling Transformer Inference — Pope et al.](https://arxiv.org/abs/2211.05102) `[paper]`

**Practical execution:** Record a private presentation explaining the manual Transformer math derivations out loud.

### Week 4

#### Monday — Parallelization Strategies
**Blueprint reading/viewing:** The Scaling Book, Chapter 5: How to Parallelize a Transformer for Training.

**Study links**
- [Scaling Book: Parallelize a Transformer for training](https://jax-ml.github.io/scaling-book/training/) `[primary]`
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) `[paper]`
- [PyTorch Fully Sharded Data Parallel documentation](https://pytorch.org/docs/stable/fsdp.html) `[official]`

**Practical execution:** Diagram Data Parallelism (DP) versus Fully Sharded Data Parallel (FSDP / ZeRO-3).

#### Tuesday — Tensor Parallelism (TP)
**Blueprint reading/viewing:** The Scaling Book, Chapter 5 continued.

**Study links**
- [Scaling Book: Sharding](https://jax-ml.github.io/scaling-book/sharding/) `[primary]`
- [Distributed arrays and automatic parallelization in JAX](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) `[official]`
- [Scaling Book: Parallelize a Transformer for training](https://jax-ml.github.io/scaling-book/training/) `[primary]`
- [NVIDIA NCCL collectives documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) `[official]`

**Practical execution:** Map the internal communication overhead (AllReduce) required inside the attention and MLP layers for TP.

#### Wednesday — Pipeline Parallelism (PP)
**Blueprint reading/viewing:** Research literature on micro-batching and pipeline bubbles.

**Study links**
- [Scaling Book: Parallelize a Transformer for training](https://jax-ml.github.io/scaling-book/training/) `[primary]`
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) `[paper]`

**Practical execution:** Analyze the "bubble" problem in PP and mathematically calculate idle hardware time across pipeline stages.

#### Thursday — Real-World Scaling
**Blueprint reading/viewing:** The Scaling Book, Chapter 6: Training LLaMA 3 on TPUs.

**Study links**
- [Scaling Book: Training LLaMA 3 on TPUs](https://jax-ml.github.io/scaling-book/llama3/) `[primary]`
- [Scaling Book: Parallelize a Transformer for training](https://jax-ml.github.io/scaling-book/training/) `[primary]`

**Practical execution:** Analyze the specific parallelization interleaving utilized to train an 8B and 70B parameter architecture.

#### Friday — Distillation Techniques
**Blueprint reading/viewing:** Vlad Feinberg Blog: Distillation Walkthrough.

**Study links**
- [Vlad Feinberg — Distillation Walkthrough](https://vladfeinberg.com/2024/02/04/distillation-walkthrough.html) `[primary]`
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) `[paper]`

**Practical execution:** Study how model distillation improves output quality while keeping inference serving latency mathematically constant.

#### Saturday — Parallel PyTorch Sync
**Blueprint reading/viewing:** The picotron repository.

**Study links**
- [Hugging Face picotron repository](https://github.com/huggingface/picotron) `[repo]`
- [Scaling Book: Parallelize a Transformer for training](https://jax-ml.github.io/scaling-book/training/) `[primary]`
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) `[paper]`
- [PyTorch Fully Sharded Data Parallel documentation](https://pytorch.org/docs/stable/fsdp.html) `[official]`

**Practical execution:** Clone the repository and map JAX sharding concepts back to PyTorch DP, FSDP, TP, and PP primitives.

#### Sunday — Consolidation
**Blueprint reading/viewing:** Review notes from the first 28 days.

**Study links**
- [Scaling Book: Transformer math](https://jax-ml.github.io/scaling-book/transformers/) `[primary]`
- [Scaling Book: Parallelize a Transformer for training](https://jax-ml.github.io/scaling-book/training/) `[primary]`
- [SciPy curve_fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) `[official]`

**Practical execution:** Build an interactive Python calculator that accepts model parameters and sequence lengths, outputting memory requirements and parallelization limits.

## Phase 2 — Empirical Scaling Laws and Inference Architecture

### Week 5

#### Monday — The Kaplan Paradigm
**Blueprint reading/viewing:** Kaplan et al. (2020) Scaling Laws paper.

**Study links**
- [Scaling Laws for Neural Language Models — Kaplan et al.](https://arxiv.org/abs/2001.08361) `[paper]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`

**Practical execution:** Understand the initial conclusion that data requirements grow slowly (D ∼ C^{0.27}) and focus should heavily favor parameter count.

#### Tuesday — The Chinchilla Correction
**Blueprint reading/viewing:** Hoffmann et al. (2022) Chinchilla paper.

**Study links**
- [Scaling Laws for Neural Language Models — Kaplan et al.](https://arxiv.org/abs/2001.08361) `[paper]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`

**Practical execution:** Analyze the mathematical evidence proving the Kaplan paradigm resulted in severely undertrained models.

#### Wednesday — The IsoFlops Methodology
**Blueprint reading/viewing:** Princeton COS 568 lecture slides (Feinberg).

**Study links**
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`
- [SciPy curve_fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) `[official]`
- [Matplotlib documentation](https://matplotlib.org/stable/) `[official]`

**Practical execution:** Document the IsoFlops loop: fixing a FLOP budget, training models of varying sizes, and fitting a parabola to find the minimal loss.

#### Thursday — Power Law Derivation
**Blueprint reading/viewing:** Princeton COS 568 lecture slides (Feinberg).

**Study links**
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`
- [SciPy curve_fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) `[official]`
- [Scaling Laws for Neural Language Models — Kaplan et al.](https://arxiv.org/abs/2001.08361) `[paper]`
- [Matplotlib documentation](https://matplotlib.org/stable/) `[official]`

**Practical execution:** Understand how multiple IsoFlop parabolas are chained together to fit a global power law dictating the exact relationship between N and D.

#### Friday — The Data Wall
**Blueprint reading/viewing:** Literature on The End of Scaling and synthetic data loops.

**Study links**
- [Will we run out of data? Limits of LLM scaling based on human-generated data](https://arxiv.org/abs/2211.04325) `[paper]`
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) `[paper]`
- [Scaling Laws of Synthetic Data for Language Models](https://arxiv.org/html/2503.19551v2) `[paper]`

**Practical execution:** Analyze the implications of exhausting high-quality human text and the pivot toward multimodal or synthetic data streams.

#### Saturday — Scaling Simulator
**Blueprint reading/viewing:** Chinchilla methodology formulas.

**Study links**
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`
- [Scaling Laws for Neural Language Models — Kaplan et al.](https://arxiv.org/abs/2001.08361) `[paper]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`

**Practical execution:** Write a Python simulation that takes a monetary budget, converts it to H100 FLOPs, and outputs the mathematically optimal model size and dataset size.

#### Sunday — Visualization
**Blueprint reading/viewing:** Matplotlib/Seaborn documentation.

**Study links**
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`
- [Scaling Laws for Neural Language Models — Kaplan et al.](https://arxiv.org/abs/2001.08361) `[paper]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [SciPy curve_fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) `[official]`
- [Matplotlib documentation](https://matplotlib.org/stable/) `[official]`
- [Seaborn documentation](https://seaborn.pydata.org/) `[official]`

**Practical execution:** Plot the simulated IsoFlop parabolas and the resulting power-law frontier, matching the visualizations found in the Chinchilla paper.

### Week 6

#### Monday — Inference Mechanics
**Blueprint reading/viewing:** The Scaling Book, Chapter 7: All About Transformer Inference.

**Study links**
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`
- [Efficiently Scaling Transformer Inference — Pope et al.](https://arxiv.org/abs/2211.05102) `[paper]`

**Practical execution:** Contrast the linear algebra of the prompt prefill phase against the auto-regressive decoding phase.

#### Tuesday — The Memory Wall
**Blueprint reading/viewing:** Pope et al. (2022) Efficiently Scaling Transformer Inference.

**Study links**
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`
- [Efficiently Scaling Transformer Inference — Pope et al.](https://arxiv.org/abs/2211.05102) `[paper]`
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`

**Practical execution:** Prove mathematically why generating one token is entirely bound by the time it takes to load the weights from HBM to SRAM.

#### Wednesday — Batching Economics
**Blueprint reading/viewing:** Pope et al. (2022) continued.

**Study links**
- [Hugging Face Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) `[primary]`
- [Scaling Book: Transformer math](https://jax-ml.github.io/scaling-book/transformers/) `[primary]`
- [Gradient checkpointing with jax.checkpoint / remat](https://docs.jax.dev/en/latest/gradient-checkpointing.html) `[official]`
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`
- [Efficiently Scaling Transformer Inference — Pope et al.](https://arxiv.org/abs/2211.05102) `[paper]`
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) `[paper]`
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) `[paper]`

**Practical execution:** Understand how increasing the batch size during inference amortizes the weight-loading penalty but exacerbates the KV cache memory footprint.

#### Thursday — Real-World Serving
**Blueprint reading/viewing:** The Scaling Book, Chapter 8: Serving LLaMA 3 on TPUs.

**Study links**
- [Scaling Book: Serving LLaMA 3 on TPUs](https://jax-ml.github.io/scaling-book/serving/) `[primary]`
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`

**Practical execution:** Analyze the deployment architecture necessary to serve concurrent users without catastrophic memory fragmentation.

#### Friday — Advanced Quantization Intro
**Blueprint reading/viewing:** Dettmers et al. LLM.int8().

**Study links**
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) `[paper]`
- [bitsandbytes GitHub repository](https://github.com/bitsandbytes-foundation/bitsandbytes) `[repo]`

**Practical execution:** Understand the basic premise of post-training quantization as a mechanism to artificially bypass the memory bandwidth wall.

#### Saturday — Inference Calculator
**Blueprint reading/viewing:** The inference arithmetic logic.

**Study links**
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`
- [Efficiently Scaling Transformer Inference — Pope et al.](https://arxiv.org/abs/2211.05102) `[paper]`

**Practical execution:** Build a programmatic calculator that predicts tokens-per-second based on specific GPU memory bandwidths and FLOP capacities.

#### Sunday — Roofline Graphing
**Blueprint reading/viewing:** Roofline model methodology.

**Study links**
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`
- [How to Scale Your Model — The Scaling Book](https://jax-ml.github.io/scaling-book/) `[primary]`
- [Matplotlib documentation](https://matplotlib.org/stable/) `[official]`
- [Seaborn documentation](https://seaborn.pydata.org/) `[official]`
- [SciPy curve_fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) `[official]`
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`
- [Efficiently Scaling Transformer Inference — Pope et al.](https://arxiv.org/abs/2211.05102) `[paper]`

**Practical execution:** Plot the operational space of an H100 GPU and map where the prefill phase and decode phase sit relative to the memory and compute ceilings.

### Week 7

#### Monday — MoE Fundamentals
**Blueprint reading/viewing:** Foundational Sparse Mixture of Experts literature.

**Study links**
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`
- [Outrageously Large Neural Networks: Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) `[paper]`
- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668) `[paper]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [Efficiently Scaling Transformer Inference — Pope et al.](https://arxiv.org/abs/2211.05102) `[paper]`

**Practical execution:** Understand the gating network mechanics and the token routing paradigm that defines MoE architectures.

#### Tuesday — MoE Scaling Deviations
**Blueprint reading/viewing:** Princeton COS 568 lecture slides on MoE Scaling.

**Study links**
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`
- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668) `[paper]`

**Practical execution:** Analyze why MoE scaling laws possess a different data-dependent exponent (\beta), making them significantly more data-hungry than dense models.

#### Wednesday — The Load Balancing Problem
**Blueprint reading/viewing:** Literature on MoE expert capacity and token dropping.

**Study links**
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`
- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668) `[paper]`

**Practical execution:** Study the algorithmic penalties associated with overloaded experts and the necessity of auxiliary load-balancing loss functions.

#### Thursday — Distributed MoE routing
**Blueprint reading/viewing:** Literature on AllToAll communication primitives.

**Study links**
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`
- [NVIDIA NCCL collectives documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) `[official]`
- [jax.lax.ragged_dot API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html) `[official]`

**Practical execution:** Map the intense network communication overhead generated when tokens must cross physical device boundaries to reach specific experts.

#### Friday — MoE vs. Dense Benchmarks
**Blueprint reading/viewing:** Princeton COS 568 lecture slides.

**Study links**
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`

**Practical execution:** Compare the theoretical active parameter count of a 64-expert MoE model against a dense equivalent trained on 100 billion tokens.

#### Saturday — Routing Algorithm Design
**Blueprint reading/viewing:** JAX jax.lax.ragged_dot documentation.

**Study links**
- [jax.lax.ragged_dot API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html) `[official]`
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`
- [jax.lax.ragged_dot_general API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot_general.html) `[official]`

**Practical execution:** Implement a toy routing algorithm in Python that accepts a batch of tokens and distributes them cleanly across a simulated set of 8 experts.

#### Sunday — Deep Synthesis
**Blueprint reading/viewing:** Review all notes on scaling and MoE architectures.

**Study links**
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`
- [NVIDIA NCCL collectives documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) `[official]`
- [jax.lax.ragged_dot API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html) `[official]`
- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668) `[paper]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [Efficiently Scaling Transformer Inference — Pope et al.](https://arxiv.org/abs/2211.05102) `[paper]`

**Practical execution:** Draft a 1,500-word theoretical essay detailing why frontier labs transitioned from massive dense models to highly distributed MoE architectures for inference economics.

### Week 8

#### Monday - Friday — Comprehensive Profiling
**Blueprint reading/viewing:** The Scaling Book, Sections 9 & 10 (Profiling and Debugging).

**Study links**
- [Scaling Book: Profiling and debugging](https://jax-ml.github.io/scaling-book/profiling/) `[primary]`
- [Profiling computation in JAX](https://docs.jax.dev/en/latest/profiling.html) `[official]`

**Practical execution:** Master the JAX TensorBoard profiling plugin. Learn to read trace maps to identify host-to-device transfer bottlenecks and compiler inefficiencies.

#### Saturday - Sunday — Systems Review
**Blueprint reading/viewing:** The Peterman Pod: Dropbox's Former Most Senior Eng.

**Study links**
- [The Peterman Pod — James Cowling: Dropbox’s Former Most Senior Eng](https://www.youtube.com/watch?v=3XkmNSuHFmY) `[video]`
- [The Peterman Pod — James Cowling episode page](https://creators.spotify.com/pod/profile/peterman-pod/episodes/Dropboxs-Former-Most-Senior-Eng-Building-Great-Systems-and-Advice-for-the-AI-Era--James-Cowling-e3jp0us) `[support]`

**Practical execution:** Synthesize architectural lessons on simplicity versus complexity. Prepare the local environment for the intensive 4-week capstone project commencing in Week 9.

## Phase 3 — The Addition Transformer Crucible

### Week 9

#### Monday — Infrastructure Setup
**Blueprint reading/viewing:** Google Colab TPU documentation.

**Study links**
- [Flax Linen basics](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html) `[official]`
- [flax.linen.Module API](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) `[official]`
- [Optax getting started](https://optax.readthedocs.io/en/latest/getting_started.html) `[official]`
- [Optax API reference](https://optax.readthedocs.io/en/latest/api/api.html) `[official]`
- [Google Colab TPU example / documentation](https://colab.research.google.com/) `[official]` — Use a TPU runtime; pair with Cloud TPU docs for production details.
- [Run a calculation on Cloud TPU VM using JAX](https://docs.cloud.google.com/tpu/docs/run-calculation-jax) `[official]`
- [JAX Quickstart](https://docs.jax.dev/en/latest/quickstart.html) `[official]`

**Practical execution:** Provision a Google Colab TPU instance. Establish the pure JAX, Flax, and Optax environment exactly as requested by the prompt.

#### Tuesday — Custom Tokenizer Design
**Blueprint reading/viewing:** Literature on tokenizer compression limits.

**Study links**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) `[paper]`
- [Scaling Book: Transformer math](https://jax-ml.github.io/scaling-book/transformers/) `[primary]`

**Practical execution:** Build a strict, custom tokenizer where the vocabulary is limited exclusively to digits 0-9, space, +, and =.

#### Wednesday — Synthetic Data Generation
**Blueprint reading/viewing:** Python random generation libraries.

**Study links**
- [Will we run out of data? Limits of LLM scaling based on human-generated data](https://arxiv.org/abs/2211.04325) `[paper]`
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) `[paper]`
- [Scaling Laws of Synthetic Data for Language Models](https://arxiv.org/html/2503.19551v2) `[paper]`
- [JAX Quickstart](https://docs.jax.dev/en/latest/quickstart.html) `[official]`
- [pandas documentation](https://pandas.pydata.org/docs/) `[official]`

**Practical execution:** Write an optimized data generator capable of producing millions of exact addition sequences (e.g., 1 2 3 + 4 5 = 1 6 8).

#### Thursday — Static Shape Engineering
**Blueprint reading/viewing:** JAX documentation on recompilation triggers.

**Study links**
- [JIT Compilation in JAX](https://docs.jax.dev/en/latest/jit-compilation.html) `[official]`
- [jax.jit API](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) `[official]`
- [OpenXLA / XLA documentation](https://openxla.org/xla) `[official]`
- [Thinking in JAX — JAX docs](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html) `[official]`

**Practical execution:** Implement deterministic padding logic to ensure all sequences are a fixed length, preventing disastrous JIT recompilations during training.

#### Friday — Architecture Sizing
**Blueprint reading/viewing:** The Scaling Book, Chapter 4.

**Study links**
- [Flax Linen basics](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html) `[official]`
- [flax.linen.Module API](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) `[official]`
- [Optax getting started](https://optax.readthedocs.io/en/latest/getting_started.html) `[official]`
- [Optax API reference](https://optax.readthedocs.io/en/latest/api/api.html) `[official]`
- [Scaling Book: Transformer math](https://jax-ml.github.io/scaling-book/transformers/) `[primary]`
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) `[paper]`

**Practical execution:** Construct the Dense Transformer in Flax. Size the layers, attention heads, and embedding dimensions to total exactly ~10 million parameters.

#### Saturday — The Training Loop
**Blueprint reading/viewing:** Optax documentation on gradient clipping and learning rates.

**Study links**
- [Flax Linen basics](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html) `[official]`
- [flax.linen.Module API](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) `[official]`
- [Optax getting started](https://optax.readthedocs.io/en/latest/getting_started.html) `[official]`
- [Optax API reference](https://optax.readthedocs.io/en/latest/api/api.html) `[official]`
- [JIT Compilation in JAX](https://docs.jax.dev/en/latest/jit-compilation.html) `[official]`

**Practical execution:** Write the core training loop. Define the cross-entropy loss function. Overfit a small batch of 10 examples to absolute zero loss to prove gradient flow.

#### Sunday — Baseline Training
**Blueprint reading/viewing:** Review execution traces.

**Study links**
- [Profiling computation in JAX](https://docs.jax.dev/en/latest/profiling.html) `[official]`
- [pytest documentation](https://docs.pytest.org/) `[official]`

**Practical execution:** Scale the training to the full dataset. Establish a baseline exact-match accuracy metric on unseen hold-out addition problems.

### Week 10

#### Monday — Evaluation Infrastructure
**Blueprint reading/viewing:** Literature on LLM arithmetic evaluation.

**Study links**
- [pytest documentation](https://docs.pytest.org/) `[official]`
- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/) `[official]`

**Practical execution:** Finalize the evaluation scripts. Ensure the model is graded on strict digit-by-digit autoregressive generation accuracy, not just teacher-forced loss.

#### Tuesday — Profiling the Step Time
**Blueprint reading/viewing:** JAX profiling tools documentation.

**Study links**
- [Profiling computation in JAX](https://docs.jax.dev/en/latest/profiling.html) `[official]`
- [Scaling Book: Profiling and debugging](https://jax-ml.github.io/scaling-book/profiling/) `[primary]`

**Practical execution:** Execute a profiling trace on a training step. Identify any operations that are failing to compile efficiently to the TPU hardware.

#### Wednesday — Hardware Optimization
**Blueprint reading/viewing:** TPU memory layout guides.

**Study links**
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`
- [Profiling computation in JAX](https://docs.jax.dev/en/latest/profiling.html) `[official]`
- [Run a calculation on Cloud TPU VM using JAX](https://docs.cloud.google.com/tpu/docs/run-calculation-jax) `[official]`

**Practical execution:** Refine the batch dimensions and model configurations to ensure peak utilization of the hardware, eliminating host-side Python bottlenecks.

#### Thursday — IsoFlops Experiment Design
**Blueprint reading/viewing:** Chinchilla methodology.

**Study links**
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`
- [Scaling Laws for Neural Language Models — Kaplan et al.](https://arxiv.org/abs/2001.08361) `[paper]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [SciPy curve_fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) `[official]`

**Practical execution:** Design a strict matrix of training runs, varying the parameter count (N) from 1M to 15M, and varying the allocated token budgets (D).

#### Friday — Automation Scripting
**Blueprint reading/viewing:** Python argparse and subprocess management.

**Study links**
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [SciPy curve_fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) `[official]`
- [Python argparse documentation](https://docs.python.org/3/library/argparse.html) `[official]`
- [Python subprocess documentation](https://docs.python.org/3/library/subprocess.html) `[official]`

**Practical execution:** Write the automation scripts required to launch, monitor, and save checkpoints for the dozens of parallel training runs required for the IsoFlops matrix.

#### Saturday — Execution
**Blueprint reading/viewing:** None.

**Study links**
- [How to Scale Your Model — The Scaling Book](https://jax-ml.github.io/scaling-book/) `[primary]`

**Practical execution:** Execute the massive matrix of training runs. Monitor memory usage and ensure cloud instances do not crash during the intensive workload.

#### Sunday — Data Aggregation
**Blueprint reading/viewing:** Pandas and JSON processing.

**Study links**
- [pandas documentation](https://pandas.pydata.org/docs/) `[official]`

**Practical execution:** Aggregate the final loss values from every training run. Clean the data and prepare the output arrays for power-law fitting.

### Week 11

#### Monday — Plotting the Parabolas
**Blueprint reading/viewing:** Scipy optimization libraries (curve_fit).

**Study links**
- [Matplotlib documentation](https://matplotlib.org/stable/) `[official]`
- [Seaborn documentation](https://seaborn.pydata.org/) `[official]`
- [SciPy curve_fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) `[official]`
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`

**Practical execution:** Fit parabolas to the aggregated data points to find the minimum loss for each specific FLOP budget computationally.

#### Tuesday — Deriving the Scaling Law
**Blueprint reading/viewing:** Chinchilla methodology.

**Study links**
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`
- [Scaling Laws for Neural Language Models — Kaplan et al.](https://arxiv.org/abs/2001.08361) `[paper]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`

**Practical execution:** Fit the power law curve across the minima to determine the exact relationship and data-hunger exponents for the synthetic addition task.

#### Wednesday — Scientific Documentation
**Blueprint reading/viewing:** Academic formatting guidelines (LaTeX).

**Study links**
- [Overleaf LaTeX learning guide](https://www.overleaf.com/learn) `[support]`

**Practical execution:** Draft the methodology section of a scientific report detailing the exact procedure used to derive the scaling laws.

#### Thursday — MoE Architecture Upgrade
**Blueprint reading/viewing:** Literature on Switch Transformers.

**Study links**
- [Flax Linen basics](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html) `[official]`
- [flax.linen.Module API](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) `[official]`
- [Optax getting started](https://optax.readthedocs.io/en/latest/getting_started.html) `[official]`
- [Optax API reference](https://optax.readthedocs.io/en/latest/api/api.html) `[official]`
- [Scaling Book: Transformer math](https://jax-ml.github.io/scaling-book/transformers/) `[primary]`
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) `[paper]`
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`

**Practical execution:** Modify the Flax model to replace the standard feed-forward block with a Mixture of Experts layer. Implement the gating logic.

#### Friday — MoE Baseline Training
**Blueprint reading/viewing:** JAX jax.lax.ragged_dot documentation.

**Study links**
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`
- [jax.lax.ragged_dot API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html) `[official]`
- [jax.lax.ragged_dot_general API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot_general.html) `[official]`

**Practical execution:** Execute a baseline training run of the 10M parameter MoE model to ensure the routers are distributing tokens evenly.

#### Saturday — MoE IsoFlops Matrix
**Blueprint reading/viewing:** Previous automation scripts.

**Study links**
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [SciPy curve_fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) `[official]`
- [Python argparse documentation](https://docs.python.org/3/library/argparse.html) `[official]`
- [Python subprocess documentation](https://docs.python.org/3/library/subprocess.html) `[official]`

**Practical execution:** Repeat the entire IsoFlops experimental matrix using the newly constructed MoE architecture.

#### Sunday — MoE Scaling Law Derivation
**Blueprint reading/viewing:** Scipy optimization libraries.

**Study links**
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`
- [Scaling Laws for Neural Language Models — Kaplan et al.](https://arxiv.org/abs/2001.08361) `[paper]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`
- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668) `[paper]`

**Practical execution:** Derive the Chinchilla scaling laws for the MoE model. Compare the data hunger (\beta) against the Dense model findings.

### Week 12

#### Monday - Friday — Report Finalization
**Blueprint reading/viewing:** LaTeX formatting tools.

**Study links**
- [Matplotlib documentation](https://matplotlib.org/stable/) `[official]`
- [Seaborn documentation](https://seaborn.pydata.org/) `[official]`
- [SciPy curve_fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) `[official]`
- [Overleaf LaTeX learning guide](https://www.overleaf.com/learn) `[support]`
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`

**Practical execution:** Finalize the rigorous scientific report. Generate high-fidelity Matplotlib graphs comparing the Dense vs. MoE performance frontiers.

#### Saturday - Sunday — Repository Cleansing
**Blueprint reading/viewing:** GitHub Actions and Markdown documentation.

**Study links**
- [GitHub Actions documentation](https://docs.github.com/en/actions) `[official]`
- [GitHub README documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) `[official]`

**Practical execution:** Clean the codebase. Modularize all JAX scripts. Write a comprehensive README. Ensure the entire experiment can be reproduced with a single command.

## Phase 4 — Extreme Optimization, Custom Kernels, and ThunderKittens

### Week 13

#### Monday — The Pallas Abstraction
**Blueprint reading/viewing:** JAX Pallas API documentation.

**Study links**
- [JAX Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html) `[official]`
- [Pallas Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html) `[official]`
- [Pallas: extending JAX for kernels — video](https://www.youtube.com/watch?v=jyaxuWae2QU) `[video]`

**Practical execution:** Understand how Pallas bypasses the XLA compiler to allow direct manual control over memory grids and block execution.

#### Tuesday — Basic Kernel Construction
**Blueprint reading/viewing:** JAX Pallas examples.

**Study links**
- [JAX Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html) `[official]`
- [Pallas Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html) `[official]`
- [Pallas: extending JAX for kernels — video](https://www.youtube.com/watch?v=jyaxuWae2QU) `[video]`
- [NVIDIA CUTLASS CuTe DSL documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/00_quickstart.html) `[official]`
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`

**Practical execution:** Write a simple matrix multiplication kernel in Pallas. Manually configure the block sizes to fit within the hardware's shared memory limits.

#### Wednesday — Hardware Conflicts
**Blueprint reading/viewing:** NVIDIA documentation on SRAM bank conflicts.

**Study links**
- [JAX Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html) `[official]`
- [Pallas Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html) `[official]`
- [Pallas: extending JAX for kernels — video](https://www.youtube.com/watch?v=jyaxuWae2QU) `[video]`
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) `[official]`
- [NVIDIA CUTLASS CuTe DSL documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/00_quickstart.html) `[official]`
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`

**Practical execution:** Analyze the compiled execution of the Pallas kernel. Modify the memory striding to avoid simultaneous thread access to the same memory banks.

#### Thursday — The FlashAttention Paradigm
**Blueprint reading/viewing:** Dao et al. FlashAttention.

**Study links**
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) `[paper]`
- [FlashAttention GitHub repository](https://github.com/dao-ailab/flash-attention) `[repo]`
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) `[paper]`
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) `[paper]`
- [FlashAttention-3 blog post](https://tridao.me/blog/2024/flash3/) `[support]`

**Practical execution:** Understand the physics of FlashAttention: tiling the QK^V computation so that intermediate state matrices are never written back to slow HBM.

#### Friday — Hardware Evolution
**Blueprint reading/viewing:** Literature on FlashAttention-2 and FlashAttention-3.

**Study links**
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) `[paper]`
- [FlashAttention GitHub repository](https://github.com/dao-ailab/flash-attention) `[repo]`
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) `[paper]`
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) `[paper]`
- [FlashAttention-3 blog post](https://tridao.me/blog/2024/flash3/) `[support]`

**Practical execution:** Map the evolution of kernel optimizations tailored to the specific asynchronous capabilities of Hopper (H100) and Blackwell (B200) architectures.

#### Saturday — The MoE Kernel Challenge
**Blueprint reading/viewing:** Vlad Feinberg's MoE Kernel Exercise specification.

**Study links**
- [JAX Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html) `[official]`
- [Pallas Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html) `[official]`
- [Pallas: extending JAX for kernels — video](https://www.youtube.com/watch?v=jyaxuWae2QU) `[video]`
- [Vlad Feinberg — How to land a job at a frontier lab](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html) `[primary]`
- [jax.lax.ragged_dot API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html) `[official]`
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`

**Practical execution:** Revisit the MoE transformer from Week 12. Begin drafting a custom Pallas kernel intended to fuse the up and down projections of the MoE layer.

#### Sunday — Performance Proof
**Blueprint reading/viewing:** JAX benchmarking tools.

**Study links**
- [JAX Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html) `[official]`
- [Pallas Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html) `[official]`
- [Pallas: extending JAX for kernels — video](https://www.youtube.com/watch?v=jyaxuWae2QU) `[video]`
- [jax.lax.ragged_dot API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html) `[official]`
- [jax.lax.ragged_dot_general API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot_general.html) `[official]`

**Practical execution:** Complete the Pallas kernel. Execute a benchmark proving that when feature dimension F > D, the custom kernel physically outpaces jax.lax.ragged_dot.

### Week 14

#### Monday — Modern Kernel DSLs
**Blueprint reading/viewing:** The ThunderKittens GitHub Repository.

**Study links**
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) `[paper]`
- [ThunderKittens GitHub repository](https://github.com/HazyResearch/ThunderKittens) `[repo]`
- [ThunderKittens launch blog](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk) `[support]`

**Practical execution:** Internalize the ThunderKittens philosophy: GPUs are not massive matrix multipliers, but collections of cores highly optimized for 16 × 16 tile operations.

#### Tuesday — Tensor Memory Acceleration
**Blueprint reading/viewing:** ThunderKittens documentation on TMA.

**Study links**
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) `[paper]`
- [ThunderKittens GitHub repository](https://github.com/HazyResearch/ThunderKittens) `[repo]`
- [ThunderKittens launch blog](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk) `[support]`
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) `[official]`

**Practical execution:** Study how ThunderKittens uses the hardware Tensor Memory Accelerator (TMA) to automate address generation and load data asynchronously.

#### Wednesday — Worker Overlapping
**Blueprint reading/viewing:** ThunderKittens templates.

**Study links**
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) `[paper]`
- [ThunderKittens GitHub repository](https://github.com/HazyResearch/ThunderKittens) `[repo]`
- [ThunderKittens launch blog](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk) `[support]`

**Practical execution:** Analyze the Load-Store-Compute-Finish template, which allows compute cores to calculate mathematics while memory cores simultaneously fetch the next data tile.

#### Thursday — ThunderKittens Architecture
**Blueprint reading/viewing:** Spector et al. ThunderKittens: Simple, Fast, and Adorable AI Kernels.

**Study links**
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) `[paper]`
- [ThunderKittens GitHub repository](https://github.com/HazyResearch/ThunderKittens) `[repo]`
- [ThunderKittens launch blog](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk) `[support]`

**Practical execution:** Understand the abstraction mapping across warp-level tiles, thread-block asynchronous overlap, and grid-level scheduling.

#### Friday — Comparative Benchmarking
**Blueprint reading/viewing:** GitHub issue #7276 discussion on llama.cpp integration.

**Study links**
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) `[paper]`
- [FlashAttention GitHub repository](https://github.com/dao-ailab/flash-attention) `[repo]`
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) `[paper]`
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) `[paper]`
- [FlashAttention-3 blog post](https://tridao.me/blog/2024/flash3/) `[support]`
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) `[paper]`
- [ThunderKittens GitHub repository](https://github.com/HazyResearch/ThunderKittens) `[repo]`

**Practical execution:** Analyze how ThunderKittens matches or outperforms CuBLAS and hand-written FlashAttention code by strictly adhering to silicon constraints.

#### Saturday — CUDA / TK Setup
**Blueprint reading/viewing:** CUDA 12.8 and C++20 setup guides.

**Study links**
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) `[paper]`
- [ThunderKittens GitHub repository](https://github.com/HazyResearch/ThunderKittens) `[repo]`
- [ThunderKittens launch blog](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk) `[support]`
- [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) `[official]`
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) `[official]`

**Practical execution:** Configure a local or cloud Linux environment with the strict CUDA 12.8 and g++-11 requirements to compile ThunderKittens natively.

#### Sunday — TK Implementation
**Blueprint reading/viewing:** ThunderKittens simple-gemm examples.

**Study links**
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) `[paper]`
- [ThunderKittens GitHub repository](https://github.com/HazyResearch/ThunderKittens) `[repo]`
- [ThunderKittens launch blog](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk) `[support]`

**Practical execution:** Write a custom linear attention kernel using the ThunderKittens DSL. Compile and verify correctness against a standard PyTorch baseline.

### Week 15

#### Monday — Quantization Theory
**Blueprint reading/viewing:** Dettmers et al. LLM.int8().

**Study links**
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) `[paper]`
- [bitsandbytes GitHub repository](https://github.com/bitsandbytes-foundation/bitsandbytes) `[repo]`

**Practical execution:** Understand the statistical presence of outlier features in LLMs and how isolating them prevents catastrophic accuracy degradation during extreme quantization.

#### Tuesday — Incoherent Processing
**Blueprint reading/viewing:** Chris De Sa’s Group: QuiP.

**Study links**
- [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304) `[paper]`
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396) `[paper]`
- [QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235) `[paper]`

**Practical execution:** Study how multiplying weight matrices by orthogonal matrices (like the Hadamard transform) smooths outliers, enabling coherent low-bit processing.

#### Wednesday — Extreme Compression
**Blueprint reading/viewing:** Chris De Sa’s Group: QuiP#.

**Study links**
- [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304) `[paper]`
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396) `[paper]`
- [QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235) `[paper]`
- [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118) `[paper]`

**Practical execution:** Analyze the mathematical mechanisms behind pushing models to 2-bit quantization using advanced lattice codebooks without catastrophic failure.

#### Thursday — Additive Quantization
**Blueprint reading/viewing:** QTIP and AQLM papers.

**Study links**
- [QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235) `[paper]`
- [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118) `[paper]`
- [AQLM GitHub repository](https://github.com/Vahe1994/AQLM) `[repo]`

**Practical execution:** Understand how additive quantization improves upon standard post-training techniques by jointly optimizing codebooks across layers.

#### Friday — Memory Economics Synthesis
**Blueprint reading/viewing:** Review Roofline derivations from Phase 2.

**Study links**
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`
- [How to Scale Your Model — The Scaling Book](https://jax-ml.github.io/scaling-book/) `[primary]`
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) `[paper]`
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396) `[paper]`
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`

**Practical execution:** Formulate a proof detailing why spending intensive compute cycles on de-quantizing weights during inference is vastly superior to loading uncompressed FP16 weights from HBM.

#### Saturday — PTQ Implementation
**Blueprint reading/viewing:** bitsandbytes or custom quantization scripts.

**Study links**
- [bitsandbytes GitHub repository](https://github.com/bitsandbytes-foundation/bitsandbytes) `[repo]`
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) `[paper]`

**Practical execution:** Implement a script that loads a 7B parameter Hugging Face model, quantizes it to INT8, and measures the exact memory reduction footprint.

#### Sunday — Perplexity Benchmarking
**Blueprint reading/viewing:** EleutherAI LM Evaluation Harness.

**Study links**
- [bitsandbytes GitHub repository](https://github.com/bitsandbytes-foundation/bitsandbytes) `[repo]`
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) `[paper]`
- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) `[repo]`

**Practical execution:** Execute a perplexity benchmark comparing the FP16 base model against the INT8 quantized model. Document the mathematical degradation curve.

### Week 16

#### Monday — Decoding Innovations
**Blueprint reading/viewing:** Review Phase 1 KV cache math.

**Study links**
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`
- [Efficiently Scaling Transformer Inference — Pope et al.](https://arxiv.org/abs/2211.05102) `[paper]`
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) `[paper]`
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) `[paper]`
- [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469) `[paper]`

**Practical execution:** Transition from weight memory bottlenecks to KV cache memory bottlenecks. Re-calculate the catastrophic growth of the KV cache over 100k tokens.

#### Tuesday — Cache Eviction Algorithms
**Blueprint reading/viewing:** SnapKV Paper.

**Study links**
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`
- [Efficiently Scaling Transformer Inference — Pope et al.](https://arxiv.org/abs/2211.05102) `[paper]`
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) `[paper]`
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) `[paper]`
- [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469) `[paper]`
- [SnapKV GitHub repository](https://github.com/FasterDecoding/SnapKV) `[repo]`
- [How to Scale Your Model — The Scaling Book](https://jax-ml.github.io/scaling-book/) `[primary]`

**Practical execution:** Understand how SnapKV algorithms monitor attention maps to intelligently evict "useless" tokens from the KV cache, maintaining context without overflowing memory.

#### Wednesday — Architectural Alterations
**Blueprint reading/viewing:** Literature on Multi-Query Attention (MQA) and Grouped-Query Attention (GQA).

**Study links**
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) `[paper]`
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) `[paper]`

**Practical execution:** Analyze how GQA physically changes the parameter structure of the model to reduce the KV footprint during the initial training phase.

#### Thursday — Speculative Decoding
**Blueprint reading/viewing:** Papers on parallel token verification.

**Study links**
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) `[paper]`
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) `[paper]`
- [Google Research blog — Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/) `[support]`

**Practical execution:** Study how an auxiliary, highly compressed "draft" model can rapidly generate speculative tokens for a larger model to verify in parallel, breaking the autoregressive sequence lock.

#### Friday — Ring Attention
**Blueprint reading/viewing:** Literature on long-context window scaling.

**Study links**
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) `[paper]`
- [Ring Attention GitHub repository](https://github.com/haoliuhl/ringattention) `[repo]`

**Practical execution:** Understand how sequence lengths exceeding hardware limits are accommodated by passing attention states in a ring architecture across multiple GPUs.

#### Saturday - Sunday — Systems Review Essay
**Blueprint reading/viewing:** Synthesize notes from Weeks 13-16.

**Study links**
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) `[paper]`
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) `[paper]`
- [Efficiently Scaling Transformer Inference — Pope et al.](https://arxiv.org/abs/2211.05102) `[paper]`

**Practical execution:** Draft a comprehensive essay explaining the "hardware lottery"—how the physical realities of HBM interconnects dictate which neural network architectures survive and which fail.

### Week 17 - 18

#### Two Weeks — Consolidation & Publishing
**Blueprint reading/viewing:** All code repositories and reports.

**Study links**
- [JAX Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html) `[official]`
- [Pallas Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html) `[official]`
- [Pallas: extending JAX for kernels — video](https://www.youtube.com/watch?v=jyaxuWae2QU) `[video]`
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) `[paper]`
- [ThunderKittens GitHub repository](https://github.com/HazyResearch/ThunderKittens) `[repo]`
- [ThunderKittens launch blog](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk) `[support]`

**Practical execution:** Thoroughly refine the Pallas MoE kernel code and the ThunderKittens implementation. Ensure all mathematical proofs and C++/CUDA code are immaculately documented and pushed to GitHub.

## Phase 5 — AI-Driven Research for Systems

### Week 19

#### Monday — The ADRS Paradigm
**Blueprint reading/viewing:** Cheng et al. Barbarians at the Gate.

**Study links**
- [Barbarians at the Gate: AI-Driven System Design](https://arxiv.org/abs/2510.06189) `[paper]`
- [SIGOPS post: AI-Driven System Design](https://www.sigops.org/2026/barbarians-at-the-gate-ai-driven-system-design/) `[support]`

**Practical execution:** Understand the thesis: systems research is highly vulnerable to AI automation because its performance metrics provide a perfect deterministic verifier.

#### Tuesday — ADRS Architecture
**Blueprint reading/viewing:** Barbarians at the Gate methodology.

**Study links**
- [Barbarians at the Gate: AI-Driven System Design](https://arxiv.org/abs/2510.06189) `[paper]`
- [SIGOPS post: AI-Driven System Design](https://www.sigops.org/2026/barbarians-at-the-gate-ai-driven-system-design/) `[support]`

**Practical execution:** Map the five core components of ADRS: Prompt Generator, Solution Generator, Evaluator, Storage, and Solution Selector.

#### Wednesday — Case Studies in Automation
**Blueprint reading/viewing:** Barbarians at the Gate evaluation.

**Study links**
- [Barbarians at the Gate: AI-Driven System Design](https://arxiv.org/abs/2510.06189) `[paper]`
- [SIGOPS post: AI-Driven System Design](https://www.sigops.org/2026/barbarians-at-the-gate-ai-driven-system-design/) `[support]`
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `[paper]`

**Practical execution:** Analyze how AI frameworks successfully optimized multi-region cloud scheduling, MoE load balancing, and complex transaction scheduling.

#### Thursday — The "Less is More" Principle
**Blueprint reading/viewing:** Barbarians at the Gate "Lessons Learned".

**Study links**
- [Barbarians at the Gate: AI-Driven System Design](https://arxiv.org/abs/2510.06189) `[paper]`
- [SIGOPS post: AI-Driven System Design](https://www.sigops.org/2026/barbarians-at-the-gate-ai-driven-system-design/) `[support]`

**Practical execution:** Understand why providing an LLM with weaker baselines and restricting its access to high-level APIs forces deeper, more creative algorithm discovery.

#### Friday — The Reward Hacking Threat
**Blueprint reading/viewing:** ADRS literature critiques.

**Study links**
- [Barbarians at the Gate: AI-Driven System Design](https://arxiv.org/abs/2510.06189) `[paper]`
- [pytest documentation](https://docs.pytest.org/) `[official]`

**Practical execution:** Analyze case studies where LLMs "optimized" load balancers by secretly dropping workloads to inflate speed metrics. Understand the necessity of bulletproof correctness verification.

#### Saturday — Evaluation Engineering
**Blueprint reading/viewing:** Python pytest and benchmarking libraries.

**Study links**
- [pytest documentation](https://docs.pytest.org/) `[official]`
- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/) `[official]`

**Practical execution:** Build a heavily fortified Python evaluation sandbox designed to test the execution speed and absolute correctness of a basic sorting algorithm.

#### Sunday — Hold-out Workloads
**Blueprint reading/viewing:** ADRS best practices.

**Study links**
- [Profiling computation in JAX](https://docs.jax.dev/en/latest/profiling.html) `[official]`
- [pytest documentation](https://docs.pytest.org/) `[official]`
- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/) `[official]`

**Practical execution:** Expand the sandbox to include hidden, edge-case test distributions to prevent the LLM from overfitting to the primary training workload.

### Week 20

#### Monday — Mathematical Discovery
**Blueprint reading/viewing:** DeepMind FunSearch literature.

**Study links**
- [Barbarians at the Gate: AI-Driven System Design](https://arxiv.org/abs/2510.06189) `[paper]`
- [FunSearch: discovering new mathematics and algorithms using LLMs](https://www.nature.com/articles/s41586-023-06924-6) `[paper]`
- [DeepMind blog: FunSearch](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/) `[support]`
- [How to Scale Your Model — The Scaling Book](https://jax-ml.github.io/scaling-book/) `[primary]`
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`
- [NeetCode / algorithms practice](https://neetcode.io/roadmap) `[support]`
- [SWE-bench benchmark](https://www.swebench.com/) `[support]`

**Practical execution:** Analyze how Google DeepMind utilized an LLM paired with an automated evaluator to discover entirely novel mathematical routing algorithms.

#### Tuesday — Formal Theorem Proving
**Blueprint reading/viewing:** DeepMind AlphaGeometry literature.

**Study links**
- [Solving olympiad geometry without human demonstrations / AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5) `[paper]`
- [DeepMind blog: AlphaGeometry](https://deepmind.google/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/) `[support]`

**Practical execution:** Study how AI frameworks synthesize complex geometric theorems and proofs without relying on previous human demonstrations.

#### Wednesday — Evolutionary Prompts
**Blueprint reading/viewing:** EvoPrompt and AdaEvolve papers.

**Study links**
- [EvoPrompt: Connecting LLMs with Evolutionary Algorithms](https://arxiv.org/abs/2309.08532) `[paper]`
- [EvoPrompt GitHub repository](https://github.com/beeevita/EvoPrompt) `[repo]`
- [AdaEvolve: Adaptive Evolutionary Algorithms via LLMs](https://arxiv.org/abs/2602.20133) `[paper]`
- [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) `[paper]`
- [How to Scale Your Model — The Scaling Book](https://jax-ml.github.io/scaling-book/) `[primary]`
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`
- [NeetCode / algorithms practice](https://neetcode.io/roadmap) `[support]`

**Practical execution:** Understand the mechanics of connecting LLMs with genetic algorithms to iteratively mutate and cross-breed code generation outputs.

#### Thursday — Data Engineering Automation
**Blueprint reading/viewing:** RuleFlow framework literature.

**Study links**
- [pandas documentation](https://pandas.pydata.org/docs/) `[official]`
- [RuleFlow: Pandas Optimization via LLM Agents](https://arxiv.org/abs/2602.09051) `[paper]`
- [RuleFlow GitHub repository](https://github.com/ADAPT-uiuc/RuleFlow) `[repo]`

**Practical execution:** Study how a hybrid 3-stage LLM approach decoupled discovery from deployment to become the state-of-the-art Pandas optimization framework.

#### Friday — Agentic Scaffolding
**Blueprint reading/viewing:** Andrej Karpathy autoresearch repository.

**Study links**
- [Andrej Karpathy autoresearch repository](https://github.com/karpathy/autoresearch) `[repo]`
- [OpenAI API documentation](https://platform.openai.com/docs) `[official]`
- [Anthropic API documentation](https://docs.anthropic.com/) `[official]`

**Practical execution:** Review the system-level python code required to orchestrate an LLM agent, manage its context window, and parse its code outputs reliably.

#### Saturday — ADRS Implementation
**Blueprint reading/viewing:** OpenAI/Anthropic API documentation.

**Study links**
- [OpenAI API documentation](https://platform.openai.com/docs) `[official]`
- [OpenAI API reference](https://platform.openai.com/docs/api-reference) `[official]`
- [OpenAI Cookbook](https://cookbook.openai.com/) `[support]`
- [Anthropic API documentation](https://docs.anthropic.com/) `[official]`
- [Anthropic Get Started guide](https://docs.anthropic.com/en/docs/get-started) `[official]`
- [Anthropic client SDKs](https://docs.anthropic.com/en/api/client-sdks) `[official]`

**Practical execution:** Write an autonomous Python script that connects to an LLM API, inputs the sorting sandbox from Week 19, and enters an evolutionary loop to optimize the algorithm's speed.

#### Sunday — Simulation Execution
**Blueprint reading/viewing:** Local system monitoring tools.

**Study links**
- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/) `[official]`
- [pandas documentation](https://pandas.pydata.org/docs/) `[official]`

**Practical execution:** Run the ADRS script for 12 hours. Track the iterative mutations, log the evaluation scores, and extract the final, highly optimized algorithm generated by the AI.

### Week 21 - 22

#### Two Weeks — Formal Verification & Refinement
**Blueprint reading/viewing:** Literature on TLA+ specifications and formal software design.

**Study links**
- [TLA+ homepage by Leslie Lamport](https://lamport.azurewebsites.net/tla/tla.html) `[primary]`
- [Specifying Systems — Leslie Lamport PDF](https://lamport.azurewebsites.net/tla/book-02-08-08.pdf) `[primary]`
- [Learn TLA+](https://learntla.com/) `[support]`

**Practical execution:** Refine the ADRS pipeline. Implement formal mathematical verification techniques within the sandbox to absolutely guarantee that the AI's hyper-optimized output is safe for production deployment.

## Phase 6 — Artifact Synthesis and Targeted Outreach

### Week 23

#### Monday - Tuesday — Screencast Production (Code)
**Blueprint reading/viewing:** Vlad Feinberg's hiring requirements.

**Study links**
- [Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla](https://arxiv.org/abs/2203.15556) `[paper]`
- [Scaling Laws for Neural Language Models — Kaplan et al.](https://arxiv.org/abs/2001.08361) `[paper]`
- [Vlad Feinberg Princeton COS 568 scaling slides](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf) `[primary]`
- [Vlad Feinberg — How to land a job at a frontier lab](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html) `[primary]`
- [The Peterman Pod — Google DeepMind Pre-Training Lead Vlad Feinberg](https://www.youtube.com/watch?v=cDyi91onoJ8) `[video]`
- [The Peterman Pod Vlad Feinberg transcript/article](https://www.developing.dev/p/google-deepmind-pre-training-lead) `[support]`

**Practical execution:** Record a high-fidelity screencast explicitly walking through the JAX Addition Transformer code from scratch. Defend the architectural choices and demonstrate the Chinchilla scaling law derivation on screen.

#### Wednesday - Thursday — Screencast Production (Math)
**Blueprint reading/viewing:** Vlad Feinberg's hiring requirements.

**Study links**
- [Vlad Feinberg — How to land a job at a frontier lab](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html) `[primary]`
- [The Peterman Pod — Google DeepMind Pre-Training Lead Vlad Feinberg](https://www.youtube.com/watch?v=cDyi91onoJ8) `[video]`
- [The Peterman Pod Vlad Feinberg transcript/article](https://www.developing.dev/p/google-deepmind-pre-training-lead) `[support]`
- [How to Scale Your Model — The Scaling Book](https://jax-ml.github.io/scaling-book/) `[primary]`
- [Scaling Book: Transformer math](https://jax-ml.github.io/scaling-book/transformers/) `[primary]`
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`

**Practical execution:** Record the second phase of the screencast: walk through the paper-and-pencil exercises from the Scaling Book. Explain the FLOP and memory math audibly, mirroring the depth of a senior systems review.

#### Friday — Kernel Systems Report
**Blueprint reading/viewing:** Hardware architecture documentation.

**Study links**
- [JAX Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html) `[official]`
- [Pallas Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html) `[official]`
- [Pallas: extending JAX for kernels — video](https://www.youtube.com/watch?v=jyaxuWae2QU) `[video]`
- [Scaling Book: Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/) `[primary]`
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) `[paper]`
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) `[paper]`

**Practical execution:** Finalize the written report detailing the MoE Pallas kernel optimization. The report must exhaustively emphasize systems reasoning—proving why the kernel outpaced standard implementations via SRAM management and FLOP utilization.

#### Saturday - Sunday — Portfolio Assembly
**Blueprint reading/viewing:** GitHub portfolio best practices.

**Study links**
- [GitHub Actions documentation](https://docs.github.com/en/actions) `[official]`
- [GitHub README documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) `[official]`
- [YouTube: Change video privacy settings](https://support.google.com/youtube/answer/157177) `[official]`
- [GitHub profile README documentation](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/customizing-your-profile/about-your-profile) `[official]`

**Practical execution:** Upload all videos as unlisted YouTube links. Ensure all code across the 24 weeks is impeccably formatted in a master GitHub repository with a flawless technical README.

### Week 24

#### Monday — Cold Email Engineering
**Blueprint reading/viewing:** Principles of highly technical communication.

**Study links**
- [JAX Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html) `[official]`
- [Pallas Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html) `[official]`
- [Pallas: extending JAX for kernels — video](https://www.youtube.com/watch?v=jyaxuWae2QU) `[video]`
- [Vlad Feinberg — How to land a job at a frontier lab](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html) `[primary]`
- [GitHub README documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) `[official]`

**Practical execution:** Draft a concise, ruthlessly technical outreach email. Exclude all mentions of generalized passion. Immediately present the Transformer implementation, Scaling Law proofs, and Pallas kernel optimizations.

#### Tuesday — Targeted Leadership Outreach
**Blueprint reading/viewing:** Vlad Feinberg's blog and contact details.

**Study links**
- [Vlad Feinberg — How to land a job at a frontier lab](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html) `[primary]`
- [Vlad Feinberg — Gemini Flash Pretraining](https://vladfeinberg.com/2025/04/24/gemini-flash-pretraining.html) `[primary]`

**Practical execution:** Send the portfolio directly to targeted hiring managers. For example, Vlad Feinberg explicitly authorized candidates who complete these specific exercises to email their artifacts directly to him (noting his operations in NYC).

#### Wednesday - Friday — Researcher Network Outreach
**Blueprint reading/viewing:** Bibliographies from ThunderKittens and ADRS papers.

**Study links**
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) `[paper]`
- [ThunderKittens GitHub repository](https://github.com/HazyResearch/ThunderKittens) `[repo]`
- [ThunderKittens launch blog](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk) `[support]`
- [Barbarians at the Gate: AI-Driven System Design](https://arxiv.org/abs/2510.06189) `[paper]`
- [FunSearch: discovering new mathematics and algorithms using LLMs](https://www.nature.com/articles/s41586-023-06924-6) `[paper]`
- [RuleFlow: Pandas Optimization via LLM Agents](https://arxiv.org/abs/2602.09051) `[paper]`
- [EvoPrompt: Connecting LLMs with Evolutionary Algorithms](https://arxiv.org/abs/2309.08532) `[paper]`

**Practical execution:** Expand the outreach to the principal investigators and research scientists who authored the specialized papers studied in Phase 4 and 5. Provide the custom TK kernels and ADRS implementations as proof of immediate infrastructural utility.

#### Saturday - Sunday — Interview Preparation
**Blueprint reading/viewing:** Algorithms, systems design, and mathematical proofs.

**Study links**
- [How to Scale Your Model — The Scaling Book](https://jax-ml.github.io/scaling-book/) `[primary]`
- [Scaling Book: Transformer inference](https://jax-ml.github.io/scaling-book/inference/) `[primary]`
- [NeetCode / algorithms practice](https://neetcode.io/roadmap) `[support]`
- [SWE-bench benchmark](https://www.swebench.com/) `[support]`

**Practical execution:** Rest, review the foundational Transformer and memory bandwidth mathematics, and prepare for the grueling technical screening interviews that follow successful artifact submission.

## Deduplicated Master Resource Index
- **Anthropic API documentation** (official): https://docs.anthropic.com/
- **Anthropic client SDKs** (official): https://docs.anthropic.com/en/api/client-sdks
- **Anthropic Get Started guide** (official): https://docs.anthropic.com/en/docs/get-started
- **Automatic Vectorization in JAX** (official): https://docs.jax.dev/en/latest/automatic-vectorization.html
- **CUDA Installation Guide for Linux** (official): https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- **Distributed arrays and automatic parallelization in JAX** (official): https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- **Flax Linen basics** (official): https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html
- **flax.linen.Module API** (official): https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html
- **GitHub Actions documentation** (official): https://docs.github.com/en/actions
- **GitHub profile README documentation** (official): https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/customizing-your-profile/about-your-profile
- **GitHub README documentation** (official): https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes
- **Google Colab TPU example / documentation** (official): https://colab.research.google.com/ — Use a TPU runtime; pair with Cloud TPU docs for production details.
- **Gradient checkpointing with jax.checkpoint / remat** (official): https://docs.jax.dev/en/latest/gradient-checkpointing.html
- **JAX Pallas documentation** (official): https://docs.jax.dev/en/latest/pallas/index.html
- **JAX PRNG Design Note** (official): https://docs.jax.dev/en/latest/jep/263-prng.html
- **JAX Quickstart** (official): https://docs.jax.dev/en/latest/quickstart.html
- **jax.checkpoint API** (official): https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html
- **jax.jit API** (official): https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
- **jax.lax.pmean API** (official): https://docs.jax.dev/en/latest/_autosummary/jax.lax.pmean.html
- **jax.lax.ragged_dot API** (official): https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html
- **jax.lax.ragged_dot_general API** (official): https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot_general.html
- **jax.pmap API** (official): https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html
- **jax.random API** (official): https://docs.jax.dev/en/latest/jax.random.html
- **jax.vmap API** (official): https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html
- **JIT Compilation in JAX** (official): https://docs.jax.dev/en/latest/jit-compilation.html
- **Matplotlib documentation** (official): https://matplotlib.org/stable/
- **NVIDIA CUDA Toolkit Documentation** (official): https://docs.nvidia.com/cuda/
- **NVIDIA CUTLASS CuTe DSL documentation** (official): https://docs.nvidia.com/cutlass/media/docs/cpp/cute/00_quickstart.html
- **NVIDIA NCCL collectives documentation** (official): https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
- **OpenAI API documentation** (official): https://platform.openai.com/docs
- **OpenAI API reference** (official): https://platform.openai.com/docs/api-reference
- **OpenXLA / XLA documentation** (official): https://openxla.org/xla
- **Optax API reference** (official): https://optax.readthedocs.io/en/latest/api/api.html
- **Optax getting started** (official): https://optax.readthedocs.io/en/latest/getting_started.html
- **Pallas Quickstart** (official): https://docs.jax.dev/en/latest/pallas/quickstart.html
- **pandas documentation** (official): https://pandas.pydata.org/docs/
- **Profiling computation in JAX** (official): https://docs.jax.dev/en/latest/profiling.html
- **pytest documentation** (official): https://docs.pytest.org/
- **pytest-benchmark documentation** (official): https://pytest-benchmark.readthedocs.io/
- **Python argparse documentation** (official): https://docs.python.org/3/library/argparse.html
- **Python subprocess documentation** (official): https://docs.python.org/3/library/subprocess.html
- **PyTorch Fully Sharded Data Parallel documentation** (official): https://pytorch.org/docs/stable/fsdp.html
- **Run a calculation on Cloud TPU VM using JAX** (official): https://docs.cloud.google.com/tpu/docs/run-calculation-jax
- **SciPy curve_fit documentation** (official): https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
- **Seaborn documentation** (official): https://seaborn.pydata.org/
- **Thinking in JAX — JAX docs** (official): https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html
- **YouTube: Change video privacy settings** (official): https://support.google.com/youtube/answer/157177
- **Accelerating Large Language Model Decoding with Speculative Sampling** (paper): https://arxiv.org/abs/2302.01318
- **AdaEvolve: Adaptive Evolutionary Algorithms via LLMs** (paper): https://arxiv.org/abs/2602.20133
- **AlphaEvolve: A coding agent for scientific and algorithmic discovery** (paper): https://arxiv.org/abs/2506.13131
- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (paper): https://arxiv.org/abs/2010.11929
- **Attention Is All You Need** (paper): https://arxiv.org/abs/1706.03762
- **Barbarians at the Gate: AI-Driven System Design** (paper): https://arxiv.org/abs/2510.06189
- **Distilling the Knowledge in a Neural Network** (paper): https://arxiv.org/abs/1503.02531
- **Efficiently Scaling Transformer Inference — Pope et al.** (paper): https://arxiv.org/abs/2211.05102
- **EvoPrompt: Connecting LLMs with Evolutionary Algorithms** (paper): https://arxiv.org/abs/2309.08532
- **Extreme Compression of Large Language Models via Additive Quantization** (paper): https://arxiv.org/abs/2401.06118
- **Fast Inference from Transformers via Speculative Decoding** (paper): https://arxiv.org/abs/2211.17192
- **Fast Transformer Decoding: One Write-Head is All You Need** (paper): https://arxiv.org/abs/1911.02150
- **FlashAttention-2** (paper): https://arxiv.org/abs/2307.08691
- **FlashAttention-3** (paper): https://arxiv.org/abs/2407.08608
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (paper): https://arxiv.org/abs/2205.14135
- **FunSearch: discovering new mathematics and algorithms using LLMs** (paper): https://www.nature.com/articles/s41586-023-06924-6
- **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism** (paper): https://arxiv.org/abs/1811.06965
- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (paper): https://arxiv.org/abs/2305.13245
- **GShard: Scaling Giant Models with Conditional Computation** (paper): https://arxiv.org/abs/2006.16668
- **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** (paper): https://arxiv.org/abs/2208.07339
- **Outrageously Large Neural Networks: Sparsely-Gated Mixture-of-Experts Layer** (paper): https://arxiv.org/abs/1701.06538
- **QTIP: Quantization with Trellises and Incoherence Processing** (paper): https://arxiv.org/abs/2406.11235
- **QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks** (paper): https://arxiv.org/abs/2402.04396
- **QuIP: 2-Bit Quantization of Large Language Models With Guarantees** (paper): https://arxiv.org/abs/2307.13304
- **Ring Attention with Blockwise Transformers for Near-Infinite Context** (paper): https://arxiv.org/abs/2310.01889
- **RuleFlow: Pandas Optimization via LLM Agents** (paper): https://arxiv.org/abs/2602.09051
- **Scaling Data-Constrained Language Models** (paper): https://arxiv.org/abs/2305.16264
- **Scaling Laws for Neural Language Models — Kaplan et al.** (paper): https://arxiv.org/abs/2001.08361
- **Scaling Laws of Synthetic Data for Language Models** (paper): https://arxiv.org/html/2503.19551v2
- **SnapKV: LLM Knows What You are Looking for Before Generation** (paper): https://arxiv.org/abs/2404.14469
- **Solving olympiad geometry without human demonstrations / AlphaGeometry** (paper): https://www.nature.com/articles/s41586-023-06747-5
- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** (paper): https://arxiv.org/abs/2101.03961
- **ThunderKittens: Simple, Fast, and Adorable AI Kernels** (paper): https://arxiv.org/abs/2410.20399
- **Training Compute-Optimal Large Language Models — Hoffmann et al. / Chinchilla** (paper): https://arxiv.org/abs/2203.15556
- **Will we run out of data? Limits of LLM scaling based on human-generated data** (paper): https://arxiv.org/abs/2211.04325
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (paper): https://arxiv.org/abs/1910.02054
- **Dwarkesh Podcast — Reiner Pope** (primary): https://www.dwarkesh.com/p/reiner-pope
- **How to Scale Your Model — The Scaling Book** (primary): https://jax-ml.github.io/scaling-book/
- **Hugging Face Ultra-Scale Playbook** (primary): https://huggingface.co/spaces/nanotron/ultrascale-playbook
- **Scaling Book: How to Think About TPUs** (primary): https://jax-ml.github.io/scaling-book/tpus/
- **Scaling Book: Parallelize a Transformer for training** (primary): https://jax-ml.github.io/scaling-book/training/
- **Scaling Book: Profiling and debugging** (primary): https://jax-ml.github.io/scaling-book/profiling/
- **Scaling Book: Roofline Analysis** (primary): https://jax-ml.github.io/scaling-book/roofline/
- **Scaling Book: Serving LLaMA 3 on TPUs** (primary): https://jax-ml.github.io/scaling-book/serving/
- **Scaling Book: Sharding** (primary): https://jax-ml.github.io/scaling-book/sharding/
- **Scaling Book: Training LLaMA 3 on TPUs** (primary): https://jax-ml.github.io/scaling-book/llama3/
- **Scaling Book: Transformer inference** (primary): https://jax-ml.github.io/scaling-book/inference/
- **Scaling Book: Transformer math** (primary): https://jax-ml.github.io/scaling-book/transformers/
- **Specifying Systems — Leslie Lamport PDF** (primary): https://lamport.azurewebsites.net/tla/book-02-08-08.pdf
- **TLA+ homepage by Leslie Lamport** (primary): https://lamport.azurewebsites.net/tla/tla.html
- **Vlad Feinberg Princeton COS 568 scaling slides** (primary): https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf
- **Vlad Feinberg — Distillation Walkthrough** (primary): https://vladfeinberg.com/2024/02/04/distillation-walkthrough.html
- **Vlad Feinberg — Gemini Flash Pretraining** (primary): https://vladfeinberg.com/2025/04/24/gemini-flash-pretraining.html
- **Vlad Feinberg — How to land a job at a frontier lab** (primary): https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html
- **Andrej Karpathy autoresearch repository** (repo): https://github.com/karpathy/autoresearch
- **AQLM GitHub repository** (repo): https://github.com/Vahe1994/AQLM
- **bitsandbytes GitHub repository** (repo): https://github.com/bitsandbytes-foundation/bitsandbytes
- **EleutherAI LM Evaluation Harness** (repo): https://github.com/EleutherAI/lm-evaluation-harness
- **EvoPrompt GitHub repository** (repo): https://github.com/beeevita/EvoPrompt
- **FlashAttention GitHub repository** (repo): https://github.com/dao-ailab/flash-attention
- **Hugging Face nanotron repository** (repo): https://github.com/huggingface/nanotron
- **Hugging Face picotron repository** (repo): https://github.com/huggingface/picotron
- **Ring Attention GitHub repository** (repo): https://github.com/haoliuhl/ringattention
- **RuleFlow GitHub repository** (repo): https://github.com/ADAPT-uiuc/RuleFlow
- **SnapKV GitHub repository** (repo): https://github.com/FasterDecoding/SnapKV
- **ThunderKittens GitHub repository** (repo): https://github.com/HazyResearch/ThunderKittens
- **CIFAR-10 dataset** (support): https://www.cs.toronto.edu/~kriz/cifar.html
- **DeepMind blog: AlphaGeometry** (support): https://deepmind.google/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/
- **DeepMind blog: FunSearch** (support): https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/
- **FlashAttention-3 blog post** (support): https://tridao.me/blog/2024/flash3/
- **Google Research blog — Looking back at speculative decoding** (support): https://research.google/blog/looking-back-at-speculative-decoding/
- **Learn TLA+** (support): https://learntla.com/
- **NeetCode / algorithms practice** (support): https://neetcode.io/roadmap
- **OpenAI Cookbook** (support): https://cookbook.openai.com/
- **Overleaf LaTeX learning guide** (support): https://www.overleaf.com/learn
- **SIGOPS post: AI-Driven System Design** (support): https://www.sigops.org/2026/barbarians-at-the-gate-ai-driven-system-design/
- **SWE-bench benchmark** (support): https://www.swebench.com/
- **The Peterman Pod Vlad Feinberg transcript/article** (support): https://www.developing.dev/p/google-deepmind-pre-training-lead
- **The Peterman Pod — James Cowling episode page** (support): https://creators.spotify.com/pod/profile/peterman-pod/episodes/Dropboxs-Former-Most-Senior-Eng-Building-Great-Systems-and-Advice-for-the-AI-Era--James-Cowling-e3jp0us
- **ThunderKittens launch blog** (support): https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk
- **Dwarkesh Podcast Reiner Pope — YouTube** (video): https://www.youtube.com/watch?v=xmkSf5IS-zw
- **Pallas: extending JAX for kernels — video** (video): https://www.youtube.com/watch?v=jyaxuWae2QU
- **The Peterman Pod — Google DeepMind Pre-Training Lead Vlad Feinberg** (video): https://www.youtube.com/watch?v=cDyi91onoJ8
- **The Peterman Pod — James Cowling: Dropbox’s Former Most Senior Eng** (video): https://www.youtube.com/watch?v=3XkmNSuHFmY

## Manual Confirmation Queue
These items in the original blueprint are intentionally broad rather than single named sources. I attached the most useful primary/supporting links above, but you may want to replace or expand them as your project narrows:
- Hardware architecture documentation for the final kernel systems report.
- GitHub portfolio best practices beyond README and Actions docs.
- Local system monitoring tools for the 12-hour ADRS run; choose based on your actual environment.
- Principles of highly technical cold outreach; use Vlad Feinberg’s article as the anchor and keep the email artifact-focused.
- Interview preparation corpus; tune to target lab role: ML systems, inference, kernels, agentic systems, or research engineer.
