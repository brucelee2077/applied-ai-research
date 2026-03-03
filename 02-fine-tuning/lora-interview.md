> **What this file covers**
> - 🎯 The low-rank hypothesis: why fine-tuning weight deltas are low-rank
> - 🧮 Full LoRA math: h = Wx → h = Wx + BAx, scaling factor α/r, parameter count formula
> - ⚠️ 4 failure modes: rank too low, rank too high, alpha miscalibration, wrong target modules
> - 📊 Parameter count and memory comparison: LoRA vs full FT with exact formulas
> - 💡 LoRA vs adapter layers vs prefix tuning: comparison table with trade-offs
> - 🏭 Production: adapter merging, multi-adapter serving, adapter composition
> - Staff/Principal Q&A with all four hiring levels shown

---

# LoRA — Interview Deep-Dive

This file assumes you have read [lora.md](./lora.md) and understand the basic idea: freeze the pre-trained model and add small trainable adapters. Everything here is for Staff/Principal depth.

---

## 🗺️ Concept Flow

```
Pre-trained weight matrix W (d_out × d_in)  ← FROZEN
                    │
                    ▼
Original output: h = Wx
                    │
                    ▼
Add LoRA:       h = Wx + (α/r) · BAx
                         │
                    ┌────┴─────┐
                    │          │
              A (r × d_in)   B (d_out × r)
              init: random    init: zeros
              TRAINABLE       TRAINABLE
                    │          │
                    └────┬─────┘
                         │
              ΔW = BA (d_out × d_in)
              rank = r (typically 4-64)
              params = r × (d_in + d_out) << d_in × d_out
```

🎯 The key insight: ΔW = W_fine_tuned − W_pretrained has low intrinsic rank. The LoRA paper showed that for GPT-3 175B fine-tuned on downstream tasks, 90%+ of the weight delta's energy concentrates in the top few singular values. This means r = 4 to 16 captures most of the task-specific adaptation.

---

## 🧮 The Full Formula

### Step 1: Standard linear layer

In a pre-trained model, each linear layer computes:

```
🧮 Standard forward pass:

    h = Wx

    Where:
      W  = weight matrix  (d_out × d_in) — frozen pre-trained weights
      x  = input vector   (d_in × 1)
      h  = output vector  (d_out × 1)
```

### Step 2: Add the low-rank update

LoRA replaces the weight matrix with W + ΔW, where ΔW is factored into two small matrices:

```
🧮 LoRA forward pass:

    h = Wx + (α/r) · BAx

    Where:
      B  = down-projection  (d_out × r) — initialized to zeros
      A  = up-projection    (r × d_in)  — initialized from N(0, σ²)
      r  = rank (hyperparameter, typically 4-64)
      α  = scaling factor (hyperparameter, typically equal to r)
```

The factor (α/r) controls the magnitude of the LoRA update relative to the original output. When α = r, the scaling factor is 1 and the LoRA output is simply h = Wx + BAx.

### Step 3: Why this factorization works

The full weight change would be ΔW with shape (d_out × d_in). For a typical transformer layer with d_in = d_out = 4096, that is 4096 × 4096 = 16.7M parameters per layer.

By factoring ΔW = BA with rank r:
- B has shape (d_out × r) = 4096 × 16 = 65,536 parameters
- A has shape (r × d_in) = 16 × 4096 = 65,536 parameters
- Total: 131,072 parameters — **127× fewer** than the full update

```
🧮 Parameter count per LoRA layer:

    params_lora = r × (d_in + d_out)

    Example: d_in = d_out = 4096, r = 16
    params_lora = 16 × (4096 + 4096) = 131,072

    Compare to full update:
    params_full = d_in × d_out = 16,777,216

    Ratio: 131,072 / 16,777,216 = 0.78%
```

### Initialization matters

- **B is initialized to zeros.** This means the LoRA update starts as zero: BAx = 0·Ax = 0. At the beginning of training, the model behaves identically to the pre-trained model.
- **A is initialized from a random distribution.** Typically Kaiming or Gaussian.

💡 Zero-initializing B is a deliberate design choice. It guarantees that training starts from the pre-trained model's behavior, not from some random perturbation. This is why LoRA preserves pre-trained knowledge better than full fine-tuning — the optimization starts at the pre-trained point and moves only as far as the data requires.

---

## 🔬 The Low-Rank Hypothesis

### Why are weight deltas low-rank?

The LoRA paper (Hu et al., 2021) investigated this empirically. They fine-tuned GPT-3 175B on multiple tasks and computed the SVD of ΔW = W_fine_tuned − W_pretrained.

Key findings:
- The top singular value captures 60-80% of the total energy (Frobenius norm)
- The top 4 singular values capture 90%+ for most layers
- This holds across different tasks and different layers

**Why does this happen?** Pre-training explores a vast parameter space and finds a good general solution. Fine-tuning for a specific task requires moving in a small number of directions in parameter space — the task-specific directions. These directions form a low-dimensional subspace, which is captured by the top singular vectors of ΔW.

```
🔬 SVD perspective:

    ΔW = UΣVᵀ

    Where:
      U  = left singular vectors  (d_out × d_out)
      Σ  = diagonal of singular values (d_out × d_in)
      Vᵀ = right singular vectors (d_in × d_in)

    If rank of ΔW is effectively r << min(d_in, d_out):
      ΔW ≈ U_r Σ_r V_rᵀ

    This is exactly what LoRA learns:
      B ≈ U_r Σ_r^{1/2}  and  A ≈ Σ_r^{1/2} V_rᵀ
```

---

## 📊 Memory and Parameter Analysis

### Total trainable parameters

For a transformer with L layers, each containing attention (Q, K, V, O projections) and FFN (up and down projections):

```
📊 LoRA parameter count:

    If applying LoRA to Q and V projections only (the default):
    params_total = L × 2 × r × (d_in + d_out)

    Example: LLaMA-7B (32 layers, d_model=4096)
    params_total = 32 × 2 × 16 × (4096 + 4096)
                 = 32 × 2 × 16 × 8192
                 = 8,388,608 (8.4M)

    vs. full model: 7,000,000,000 (7B)
    Ratio: 8.4M / 7B = 0.12%
```

### Memory comparison

| Component | Full Fine-Tuning | LoRA |
|-----------|-----------------|------|
| Model weights (FP16) | 14 GB | 14 GB (frozen, no grad) |
| Gradients | 14 GB (all params) | ~16 MB (adapters only) |
| Adam m + v (FP32) | 56 GB | ~64 MB |
| Master weights (FP32) | 28 GB | ~32 MB |
| **Total (ex. activations)** | **~112 GB** | **~14.1 GB** |

⚠️ LoRA still needs to store the frozen model in GPU memory for the forward pass. The savings come from not storing gradients or optimizer states for the frozen parameters.

---

## ⚠️ Failure Modes

### 1. Rank too low

If r is too small, the adapter cannot capture the complexity of the task-specific change. Symptoms: training loss plateaus above an acceptable threshold; validation performance significantly below full fine-tuning.

**Detection:** Compare to full FT on a small dataset. If the gap is >3%, increase rank.

### 2. Rank too high

If r is too large, the adapter overfits, especially with small datasets. Symptoms: large train-val gap; performance on held-out data degrades as training continues.

**Rule of thumb:** Start with r = 16. Double if quality is insufficient; halve if overfitting. The LoRA paper found r = 4 sufficient for many tasks.

### 3. Alpha miscalibration

The scaling factor α/r controls the magnitude of the LoRA update. If α is too large, the LoRA output dominates the pre-trained output, effectively breaking the pre-trained features. If α is too small, the adapter barely contributes.

**Common practice:** Set α = r (so the scaling factor is 1) or α = 2r. The scaling factor should not exceed 2 for most tasks.

### 4. Wrong target modules

LoRA can be applied to any linear layer. Applying it to the wrong layers wastes capacity and can hurt performance.

**The default (Q, V projections)** works well for most NLP tasks. The LoRA paper showed this outperforms applying to K only or to FFN only. However, for generation-heavy tasks, applying to all attention projections (Q, K, V, O) and the FFN gives better results at the cost of more parameters.

| Target modules | Params (7B model, r=16) | Best for |
|---------------|------------------------|---------|
| Q, V only | 8.4M (0.12%) | Classification, most NLP |
| Q, K, V, O | 16.8M (0.24%) | Generation tasks |
| All linear | ~40M (0.57%) | Large domain shifts |

---

## 💡 LoRA vs Other PEFT Methods

| Method | How it works | Trainable params | Inference overhead | Merge into base? |
|--------|-------------|-----------------|-------------------|-------------------|
| **LoRA** | Low-rank adapters on weight matrices | r × (d_in + d_out) per layer | Zero (after merge) | Yes |
| **Adapter layers** | Insert small bottleneck layers | 2 × d_model × r per layer | Adds latency (extra layers) | No |
| **Prefix tuning** | Prepend trainable tokens to input | n_prefix × d_model | Reduces max sequence length | No |
| **Prompt tuning** | Prepend trainable soft prompts | n_tokens × d_model | Reduces max sequence length | No |
| **BitFit** | Train bias terms only | ~0.1% of model | Zero | Yes (trivially) |

💡 LoRA's unique advantage is **zero inference overhead after merging**. You add the LoRA weights into the base model (W_new = W + BA) and serve a single model with no extra computation. No other adapter method has this property.

---

## 🏭 Production and Scaling

### Adapter merging

After training, merge the LoRA weights into the base model:

```
🏭 Merge formula:

    W_merged = W + (α/r) · BA

    The merged model has exactly the same architecture and inference
    cost as the original model. No adapter overhead at inference time.
```

### Multi-adapter serving

In production, you often serve the same base model with different LoRA adapters for different tasks or customers:

```
🏭 Multi-adapter architecture:

    Base Model (14 GB, loaded once)
         │
    ┌────┼────┬────────┐
    │    │    │        │
    ▼    ▼    ▼        ▼
  LoRA₁ LoRA₂ LoRA₃  LoRA_N
  (10MB) (10MB) (10MB) (10MB)

  Task: Medical → load LoRA₁
  Task: Legal   → load LoRA₂
  Task: Code    → load LoRA₃

  Swap time: milliseconds (just swap the small adapter weights)
```

This is dramatically more efficient than keeping N separate fine-tuned models (N × 14 GB vs 14 GB + N × 10 MB).

### Adapter composition

Multiple LoRA adapters can be composed (summed) at inference time:

```
🏭 Composition:

    h = Wx + (α₁/r₁) · B₁A₁x + (α₂/r₂) · B₂A₂x

    Combine: a "style" adapter + a "domain" adapter
```

⚠️ Adapter composition works when the adapters were trained independently and their weight deltas are approximately orthogonal. When they conflict (modifying the same subspace in incompatible ways), composition can degrade quality.

---

## Staff/Principal Interview Depth

### Q1: Derive the LoRA parameter savings and explain why the low-rank assumption holds.

---

**No Hire**

*Interviewee:* "LoRA uses less parameters because it adds small matrices instead of changing the big ones. It's like 1% of the parameters."

*Interviewer:* No derivation, no formula, no explanation of why low-rank works. The "1%" number is approximately right but unsupported.

*Criteria — Met:* basic understanding / *Missing:* parameter count formula, low-rank justification, SVD connection, rank selection reasoning

---

**Weak Hire**

*Interviewee:* "LoRA factors the weight update ΔW into two matrices B and A with rank r. Instead of d × d parameters, you get r × (d + d) = 2rd parameters. For d=4096 and r=16, that's about 130K vs 16.7M — roughly 100× fewer. Low-rank works because the task-specific changes are usually simple."

*Interviewer:* Correct formula and example. But "changes are usually simple" is vague. No SVD analysis, no discussion of when low-rank might not hold, no connection to the scaling factor.

*Criteria — Met:* correct formula, worked example / *Missing:* SVD analysis, when low-rank fails, scaling factor α/r, initialization strategy

---

**Hire**

*Interviewee:* "The full update ΔW = W' − W has shape (d_out × d_in). LoRA factors this as ΔW = BA where B is (d_out × r) and A is (r × d_in). Parameters: r(d_in + d_out). For LLaMA-7B with 32 layers, applying to Q and V gives 32 × 2 × 16 × 8192 ≈ 8.4M trainable params, or 0.12% of the model. The low-rank assumption holds because the LoRA paper showed via SVD that ΔW's energy concentrates in the top few singular values — 90%+ in the top 4-8. This makes sense because fine-tuning moves parameters in a small number of task-specific directions. B is initialized to zero so training starts from the pre-trained behavior. The scaling factor α/r controls the update magnitude."

*Interviewer:* Complete derivation with worked example at model scale. SVD connection with empirical evidence. Understands initialization and scaling. Would be Strong Hire with discussion of when low-rank fails and rank selection strategy.

*Criteria — Met:* full derivation, SVD analysis, initialization, scaling factor / *Missing:* failure cases for low-rank, rank selection guidance, comparison to other PEFT methods

---

**Strong Hire**

*Interviewee:* "Starting from h = Wx, LoRA adds h = Wx + (α/r)BAx where B ∈ ℝ^(d_out×r) and A ∈ ℝ^(r×d_in). This gives r(d_in + d_out) trainable params per layer vs d_in × d_out for full FT — a factor of d/(2r) reduction for square matrices. The low-rank hypothesis is supported by the SVD analysis in the original paper: for GPT-3 175B, ΔW's top 4 singular values capture 90%+ of the Frobenius norm. This is expected from a random matrix theory perspective — fine-tuning from a good initialization moves along a low-dimensional manifold in weight space. The hypothesis breaks when there's a large domain shift — e.g., fine-tuning an English LLM for protein folding. In those cases, the required ΔW is full-rank and LoRA underperforms, which you can detect by running rank ablation: if quality keeps improving up to r=128, the task probably needs full FT. B is zero-initialized so the model starts exactly at the pre-trained point. α/r is the effective scaling factor — I typically set α = r so the factor is 1, then adjust if the loss scale is off. In production, I merge adapters into the base model for zero-overhead inference, and serve multiple task-specific adapters from the same base model."

*Interviewer:* Complete mathematical treatment. Connects to random matrix theory. Identifies when the assumption breaks and provides a practical detection method (rank ablation). Production deployment strategy. This is staff-level thinking.

*Criteria — Met:* full math, SVD analysis, random matrix theory connection, failure detection, rank ablation strategy, production deployment

---

### Q2: How do you choose the rank r, and what goes wrong if you choose poorly?

---

**No Hire**

*Interviewee:* "I usually set rank to 16. Higher rank means more parameters."

*Interviewer:* No reasoning behind the choice, no awareness of trade-offs.

*Criteria — Met:* knows r is a hyperparameter / *Missing:* trade-off analysis, how to detect wrong rank, relationship to task complexity

---

**Weak Hire**

*Interviewee:* "Higher rank means more capacity but also more parameters and risk of overfitting. I start with r=16 and increase if quality is insufficient. The LoRA paper showed r=4 works for many tasks."

*Interviewer:* Correct general trade-off. References the paper. But no quantitative guidance, no failure detection strategy, no connection to dataset size or domain distance.

*Criteria — Met:* trade-off description, paper reference / *Missing:* quantitative guidance, detection strategy, connection to data/domain, alpha interaction

---

**Hire**

*Interviewee:* "Rank controls the expressiveness of the adapter. Too low: the adapter can't capture the task, and you see a training loss plateau above the full FT baseline. Too high: overfitting, especially with small datasets, visible as a train-val gap. My process: (1) start at r=16, (2) compare to full FT on a small dataset to establish the ceiling, (3) if quality is within 2%, done. If not, double r and repeat. I stop increasing at r=64 — beyond that, you're approaching full FT cost for diminishing returns. The paper showed r=4 works for GLUE tasks, but generation tasks typically need r=16-32."

*Interviewer:* Structured process with a stopping criterion. Distinguishes between task types. Would be Strong Hire with SVD analysis of weight deltas or connection between rank and dataset size.

*Criteria — Met:* structured process, stopping criterion, task-type distinction / *Missing:* SVD-based rank selection, dataset-size interaction, alpha-rank interaction

---

**Strong Hire**

*Interviewee:* "Rank selection is fundamentally about matching adapter capacity to task complexity. I use a rank ablation sweep: train with r ∈ {1, 2, 4, 8, 16, 32, 64} on a validation set and plot the quality curve. If quality plateaus by r=8, the task is well-captured by a low-rank update (classification, sentiment). If quality keeps improving to r=64, the task needs more capacity (code generation, large domain shift). I also look at it from the data side: with N training examples and r(d_in + d_out) parameters per layer, I want the total trainable params to be well below N to avoid overfitting. For 1K examples with d=4096, r=4 gives about 1M params — reasonable. r=64 gives 16M — likely overfitting. After selecting rank, I set α = r (scaling factor = 1) as a starting point. If the LoRA contribution is too large (loss spikes), I reduce α. If too small (slow convergence), I increase it. In production, I also inspect the SVD of the trained BA to confirm the effective rank — if the bottom singular values of BA are near-zero, I could use a smaller r next time."

*Interviewer:* Rank ablation as a disciplined process. Connects rank to dataset size quantitatively. Understands α-r interaction. Post-hoc SVD inspection shows deep understanding. This is research-grade thinking applied to a practical problem.

*Criteria — Met:* rank ablation, dataset-size connection, alpha-rank interaction, SVD post-hoc analysis, quantitative reasoning

---

### Q3: Compare LoRA to adapter layers and prefix tuning. When would you choose each?

---

**No Hire**

*Interviewee:* "LoRA is the most popular one. I always use it."

*Interviewer:* No comparison, no awareness of alternatives, no decision criteria.

*Criteria — Met:* none / *Missing:* knowledge of alternatives, comparison, decision criteria

---

**Weak Hire**

*Interviewee:* "LoRA adds small matrices to existing layers. Adapters insert new layers. Prefix tuning adds tokens to the input. LoRA is usually best because it has no inference overhead."

*Interviewer:* Knows the three methods at a high level. Correctly identifies LoRA's inference advantage. But no quantitative comparison, no discussion of when alternatives win.

*Criteria — Met:* basic comparison, inference overhead point / *Missing:* quantitative comparison, when alternatives are better, parameter efficiency analysis

---

**Hire**

*Interviewee:* "LoRA modifies weight matrices with low-rank updates. After training, you merge BA into W and get zero inference overhead. Adapter layers insert bottleneck modules (down-project, nonlinearity, up-project) that add sequential latency — about 5-10% overhead on inference. Prefix tuning prepends trainable tokens that eat into the context window. I choose LoRA by default because of the merge property. I'd consider adapters if I needed to add a nonlinearity in the adaptation (LoRA is linear). I'd use prefix tuning for very parameter-efficient setups where I want less than 100K trainable params. In practice, LoRA dominates because quality is equivalent and deployment is simpler."

*Interviewer:* Good comparison with clear decision criteria. Understands the linearity limitation of LoRA. Would be Strong Hire with quantitative parameter comparison and discussion of when linearity is actually a problem.

*Criteria — Met:* clear comparison, merge property, adapter overhead, decision criteria / *Missing:* quantitative comparison, empirical quality results, linearity impact analysis

---

**Strong Hire**

*Interviewee:* "Three axes: parameter efficiency, inference overhead, and empirical quality. Parameter efficiency: LoRA with r=16 adds ~8M params for a 7B model. Adapters with bottleneck r=64 add ~50M. Prefix tuning with 100 tokens adds ~400K. So prefix is most efficient, but it reduces usable context and struggles with generation. Inference overhead: LoRA is the only method that achieves zero overhead after merge. Adapters add serial computation per layer — I measured about 8% latency increase in production. Prefix tuning adds tokens to the KV cache, increasing attention compute by n_prefix/n_seq. Quality: on GLUE benchmarks, all three match full FT within 1-2%. On generation tasks (summarization, dialogue), LoRA and adapters outperform prefix tuning. LoRA's one limitation is that it's a linear update — if the task requires a nonlinear transformation of the representation (rare but possible), adapters with a ReLU can capture this. In practice, I've never seen a task where this matters. I default to LoRA for everything and have yet to find a case where adapters or prefix tuning give meaningfully better results."

*Interviewer:* Quantitative comparison across three axes. Production measurements for inference overhead. Understands the linearity limitation theoretically but correctly notes it rarely matters in practice. Experience-backed default strategy.

*Criteria — Met:* three-axis quantitative comparison, production measurements, linearity analysis, empirical quality comparison, experience-backed strategy

---

### Q4: How would you serve 50 different LoRA-adapted models in production?

---

**No Hire**

*Interviewee:* "I'd fine-tune 50 different models and deploy them separately."

*Interviewer:* This misses the entire point of LoRA's serving advantage. 50 separate 7B models = 700 GB. One base + 50 adapters = 14 GB + 500 MB.

*Criteria — Met:* none / *Missing:* adapter-based serving, memory analysis, swap strategy, batching

---

**Weak Hire**

*Interviewee:* "Load the base model once and swap LoRA adapters per request. Each adapter is small, maybe 10-50 MB, so you keep them all in memory."

*Interviewer:* Correct high-level strategy. But no discussion of batching with mixed adapters, merging vs runtime application, or cache management.

*Criteria — Met:* shared base model concept / *Missing:* mixed batching, merge vs runtime, cache strategy, latency implications

---

**Hire**

*Interviewee:* "Architecture: one base model in GPU memory (14 GB for 7B). All 50 adapters stored in CPU memory or disk (50 × 30 MB ≈ 1.5 GB total). For each request, identify the adapter and either swap it in (if not cached) or use the cached version. For single-adapter requests, merge into base weights for zero-overhead inference. The challenge is mixed batching — if a batch contains requests for 5 different adapters, you can't merge. Instead, apply LoRA at runtime for those requests. I'd use vLLM or similar serving frameworks that support multi-adapter serving natively."

*Interviewer:* Practical architecture. Understands the merge-vs-runtime trade-off. Mentions mixed batching. Knows about serving frameworks. Would be Strong Hire with more detail on the mixed batching implementation.

*Criteria — Met:* architecture, merge-vs-runtime, mixed batching awareness, framework knowledge / *Missing:* mixed batch implementation, cache eviction, latency SLA management

---

**Strong Hire**

*Interviewee:* "I'd build a tiered system. Base model stays on GPU permanently. Popular adapters (top 10 by request volume) are pre-merged into separate model copies — this gives zero-overhead inference for 80% of traffic using maybe 140 GB total for 10 copies. Remaining 40 adapters are applied at runtime via the BA addition. For mixed batching, I'd group requests by adapter and run micro-batches. Each micro-batch applies one adapter's BA product across all matching requests, then concatenates outputs. If adapter switching cost exceeds 100μs per request, I'd batch by adapter at the router level instead. Cache policy: LRU eviction of adapter weights from GPU, with pre-fetching based on request patterns. Monitoring: track per-adapter latency and cache hit rate; promote frequently-used adapters to pre-merged tier. I'd use vLLM or S-LoRA which implement the batched LoRA kernel that applies multiple adapters within a single forward pass by tiling the BA computation."

*Interviewer:* Production-grade architecture with tiered strategy, mixed batching implementation, cache policy, and monitoring. Knows about S-LoRA's batched kernel. This is system-design-level thinking applied to ML serving.

*Criteria — Met:* tiered architecture, mixed batch implementation, cache policy, latency analysis, monitoring, S-LoRA awareness

---

## Key Takeaways

🎯 1. LoRA factors the weight update as ΔW = BA where B is (d_out × r) and A is (r × d_in). This reduces parameters from d_in × d_out to r(d_in + d_out).

🎯 2. The low-rank hypothesis holds because fine-tuning moves parameters in a small number of task-specific directions. SVD of ΔW shows 90%+ energy in the top 4-8 singular values.

3. Zero-initializing B ensures training starts from the pre-trained model's behavior. This is why LoRA preserves knowledge better than full fine-tuning.

🎯 4. After training, merge BA into W for zero inference overhead. This is LoRA's unique advantage over other PEFT methods.

⚠️ 5. Rank selection is empirical. Start at r=16, run an ablation sweep, and check if quality plateaus. If quality keeps improving at r=64+, consider full fine-tuning.

6. In production, one base model + N small adapters replaces N full model copies. Memory savings: N × 14 GB → 14 GB + N × 30 MB.

7. Alpha (α) controls the update magnitude. Set α = r as a starting point. Adjust if the loss scale is wrong.
