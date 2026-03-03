> **What this file covers**
> - 🎯 Why full fine-tuning works: gradient descent on all parameters with pre-trained initialization
> - 🧮 SGD and Adam update rules with every symbol labeled
> - 📊 Memory formula: 4× model size breakdown (weights + gradients + optimizer + activations)
> - ⚠️ Catastrophic forgetting: mechanism, detection, and 5 mitigation strategies
> - 💡 Learning rate scheduling: warmup, cosine decay, and why they matter
> - 🏭 Production: distributed training, mixed precision, gradient checkpointing
> - Staff/Principal Q&A with all four hiring levels shown

---

# Full Fine-Tuning — Interview Deep-Dive

This file assumes you have read [full-fine-tuning.md](./full-fine-tuning.md) and understand the basic idea: full fine-tuning updates every parameter in a pre-trained model using gradient descent on task-specific data. Everything here is for Staff/Principal depth.

---

## 🗺️ Concept Flow

```
Pre-trained model (billions of parameters set by pre-training)
                    │
                    ▼
    Load task-specific dataset (input, label) pairs
                    │
                    ▼
         ┌─────────────────────────────┐
         │  For each mini-batch:        │
         │                              │
         │  1. Forward pass             │
         │     → compute predictions    │
         │                              │
         │  2. Compute loss             │
         │     → how wrong is it?       │
         │                              │
         │  3. Backward pass            │
         │     → gradient for EVERY     │
         │       parameter              │
         │                              │
         │  4. Optimizer step           │
         │     → update ALL parameters  │
         │     → (SGD, Adam, AdamW)     │
         └──────────┬──────────────────┘
                    │
                    ▼ (repeat for N epochs)
                    │
         Fine-tuned model (all params changed)
```

🎯 The key insight: pre-training gives a good initialization. Full fine-tuning then moves *all* parameters toward the task-specific optimum. The quality ceiling is higher than any parameter-efficient method — but so is the cost and the risk.

---

## 🧮 The Optimization Math

### SGD (Stochastic Gradient Descent)

The simplest optimizer. One update rule:

```
🧮 SGD update:

    θ_{t+1} = θ_t - η · ∇L(θ_t)

    Where:
      θ_t    = all model parameters at step t
      η      = learning rate (scalar, typically 1e-5 to 5e-5 for fine-tuning)
      ∇L(θ_t) = gradient of the loss with respect to θ
```

Each parameter moves in the opposite direction of its gradient, scaled by the learning rate. If the gradient says "increasing this parameter increases the loss", the update decreases it.

### Adam (the standard for fine-tuning)

Adam keeps running averages of the gradient (first moment) and the squared gradient (second moment). This gives each parameter its own effective learning rate.

```
🧮 Adam update (per parameter):

    Step 1 — Update first moment (momentum):
    m_t = β₁ · m_{t-1} + (1 - β₁) · g_t

    Step 2 — Update second moment (variance):
    v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²

    Step 3 — Bias correction (compensate for zero initialization):
    m̂_t = m_t / (1 - β₁ᵗ)
    v̂_t = v_t / (1 - β₂ᵗ)

    Step 4 — Update parameter:
    θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)

    Where:
      g_t  = gradient at step t
      m_t  = first moment estimate (moving average of gradient)
      v_t  = second moment estimate (moving average of squared gradient)
      β₁   = decay rate for first moment (typically 0.9)
      β₂   = decay rate for second moment (typically 0.999)
      ε    = small constant to avoid division by zero (typically 1e-8)
      η    = learning rate
```

**Why Adam needs 2× extra memory per parameter:** It stores m_t and v_t for every parameter. For a 7B model, that is 7B × 2 = 14B extra floats.

### AdamW (Adam with weight decay)

AdamW decouples weight decay from the gradient update. This is the standard optimizer for fine-tuning language models.

```
🧮 AdamW difference from Adam:

    θ_{t+1} = θ_t - η · (m̂_t / (√v̂_t + ε) + λ · θ_t)

    Where:
      λ = weight decay coefficient (typically 0.01 to 0.1)

    The "+ λ · θ_t" term pulls weights toward zero,
    acting as regularization to prevent overfitting.
```

💡 In standard Adam, weight decay is tangled with the adaptive learning rate. AdamW separates them, which gives more predictable regularization. This matters most when fine-tuning — you want the decay to actually shrink weights uniformly, not to get rescaled by the adaptive denominator.

---

## 📊 Memory Analysis

### The 4× Rule

Full fine-tuning with Adam in mixed precision (FP16 forward/backward, FP32 optimizer states) requires:

```
📊 Memory breakdown for a model with P parameters:

    Component              Bytes per param    Formula
    ─────────────────────  ────────────────   ──────────
    Model weights (FP16)   2                  2P
    Gradients (FP16)       2                  2P
    Optimizer m_t (FP32)   4                  4P
    Optimizer v_t (FP32)   4                  4P
    Master weights (FP32)  4                  4P
    ─────────────────────  ────────────────   ──────────
    Total (excluding act.) 16 bytes/param     16P

    Plus activations:      depends on batch size, sequence length,
                           and number of layers. Typically 2-8P.
```

**Worked example — LLaMA-7B:**

| Component | Calculation | Memory |
|-----------|------------|--------|
| Model weights (FP16) | 7B × 2 bytes | 14 GB |
| Gradients (FP16) | 7B × 2 bytes | 14 GB |
| Adam m_t (FP32) | 7B × 4 bytes | 28 GB |
| Adam v_t (FP32) | 7B × 4 bytes | 28 GB |
| Master weights (FP32) | 7B × 4 bytes | 28 GB |
| **Subtotal** | | **112 GB** |
| Activations (batch=1, seq=2048) | ~10-20 GB | ~15 GB |
| **Total** | | **~127 GB** |

That is more than one A100 80GB can hold. You need at least two A100s with model parallelism, or gradient checkpointing to trade compute for memory.

### Activation memory

Activation memory scales with:

```
📊 Activation memory:

    A ≈ L × n × d_model × b × k

    Where:
      L       = number of layers
      n       = sequence length
      d_model = hidden dimension
      b       = batch size
      k       = a constant depending on the architecture
                (typically 10-12 for transformer blocks)
```

⚠️ Activation memory grows linearly with batch size and sequence length. For long sequences (8K+ tokens), activations can dominate total memory even more than optimizer states.

### Gradient checkpointing

Gradient checkpointing reduces activation memory by ~60-70% at the cost of ~30% more compute time. Instead of storing all activations, it recomputes them during the backward pass.

```
Without checkpointing:  Store activations for all L layers → memory = O(L)
With checkpointing:     Store every √L-th layer, recompute the rest → memory = O(√L)
                        But backward pass takes ~1.3× longer
```

---

## ⚠️ Catastrophic Forgetting

### The Mechanism

Catastrophic forgetting happens because the same parameters that store general knowledge are the parameters being updated for the new task.

```
⚠️ How forgetting works:

    Pre-trained weights θ₀ encode general knowledge across all parameters.

    Fine-tuning moves θ₀ → θ* to minimize the task-specific loss L_task.

    But L_task does not penalize losing general capabilities.
    The optimizer happily destroys knowledge in θ₀ that is not
    needed for the new task — it has no reason to preserve it.

    The further θ* moves from θ₀, the more general knowledge is lost.
```

### Detection

Monitor performance on a held-out general benchmark during fine-tuning. If general performance drops more than 5-10%, forgetting is occurring.

### 5 Mitigation Strategies

| Strategy | How it works | Trade-off |
|----------|-------------|-----------|
| **Low learning rate** (1e-5 to 5e-5) | Small steps keep θ* close to θ₀ | Slower convergence, may underfit |
| **Learning rate warmup** | Start near zero, ramp up over first 5-10% of steps | Prevents early large updates that cause irreversible damage |
| **Data mixing** | Include 5-20% general-purpose data in each batch | Uses training budget on non-task data |
| **Early stopping** | Stop when validation loss plateaus, before overfitting | May leave performance on the table |
| **EWC (Elastic Weight Consolidation)** | Add penalty proportional to Fisher information × (θ - θ₀)² | Extra compute for Fisher matrix; approximate only |

💡 In practice, the combination of low learning rate + warmup + early stopping handles most forgetting. EWC is theoretically elegant but rarely used in production fine-tuning because the overhead is not worth the marginal improvement.

---

## 🧮 Learning Rate Scheduling

### Why scheduling matters

A constant learning rate has two problems: (1) it may be too large initially, causing unstable updates that damage pre-trained features, and (2) it may be too large near convergence, causing the optimizer to bounce around the minimum instead of settling into it.

### Warmup + Cosine Decay

The standard schedule for fine-tuning:

```
🧮 Warmup (linear):

    For steps t = 0 to T_warmup:
    η_t = η_max × (t / T_warmup)

    Typically T_warmup = 5-10% of total steps.


🧮 Cosine decay (after warmup):

    For steps t = T_warmup to T_total:
    η_t = η_min + (η_max - η_min) × 0.5 × (1 + cos(π × (t - T_warmup) / (T_total - T_warmup)))

    Where:
      η_max = peak learning rate (e.g., 2e-5)
      η_min = minimum learning rate (e.g., 0 or 1e-6)
```

```
  Learning Rate Schedule (Warmup + Cosine Decay):

  η_max ─ ─ ─ ─ ─ ╱╲
                  ╱   ╲
                 ╱     ╲
                ╱       ╲
               ╱         ╲
              ╱            ╲
             ╱               ╲
  η_min ── ╱─────────────────────╲──
           │        │              │
           0    T_warmup       T_total
           ← warm →← cosine decay →
```

⚠️ Skipping warmup is a common mistake. The pre-trained model is in a sensitive region of the loss landscape. Large initial updates can push parameters into a bad region from which the optimizer never recovers.

---

## 🏭 Production and Scaling

### Mixed Precision Training (FP16/BF16)

Store parameters in FP16 for forward/backward passes but keep a FP32 master copy for the optimizer step. This halves activation memory and speeds up matrix multiplications on modern GPUs.

```
🏭 Mixed precision flow:

    FP32 master weights → cast to FP16 → forward pass (FP16) → loss (FP32)
                                        → backward pass (FP16) → gradients (FP16)
    → cast gradients to FP32 → optimizer step (FP32) → update FP32 master weights
```

⚠️ BF16 (bfloat16) is preferred over FP16 when available. BF16 has the same exponent range as FP32 (8 bits) so it does not need loss scaling. FP16 has a smaller range and requires a loss scaling factor to prevent gradient underflow.

### Distributed Training

For models that do not fit on a single GPU:

| Strategy | What it splits | Communication cost |
|----------|---------------|-------------------|
| **Data Parallel (DDP)** | Data across GPUs; each GPU has full model copy | Gradient all-reduce per step |
| **FSDP (Fully Sharded Data Parallel)** | Parameters, gradients, and optimizer state across GPUs | Higher communication, but each GPU holds only 1/N of model |
| **Tensor Parallel** | Individual layers split across GPUs | Requires fast interconnect (NVLink) |
| **Pipeline Parallel** | Different layers on different GPUs | Bubble overhead from sequential dependency |

💡 For fine-tuning (as opposed to pre-training), FSDP is usually the best choice: it lets you fine-tune models larger than a single GPU can hold, with minimal code changes (just wrap your model).

---

## Staff/Principal Interview Depth

### Q1: Walk me through the memory required to fully fine-tune a 7B parameter model with Adam. Where does each component come from?

---

**No Hire**

*Interviewee:* "It needs a lot of memory. The model is 7 billion parameters so probably like 28 GB or something. You need a big GPU."

*Interviewer:* The candidate gives a vague number without breaking down the components. No mention of gradients, optimizer states, or activations. No distinction between parameter storage and training memory. This suggests memorized facts without understanding.

*Criteria — Met:* none / *Missing:* memory breakdown by component, bytes-per-parameter calculation, optimizer state explanation, activation memory, practical GPU requirements

---

**Weak Hire**

*Interviewee:* "You need memory for the model weights, the gradients, and the optimizer state. Adam stores two extra values per parameter — momentum and variance. So it's roughly 4× the model size. For 7B parameters at FP16, the model is 14 GB, so total is around 56 GB. An A100 should handle it."

*Interviewer:* The candidate knows the components and gets the rough 4× rule. But the math is wrong: 4× of 14 GB = 56 GB only if everything is FP16, but Adam states must be FP32. They also missed master weights and activations. The conclusion (one A100 is enough) is incorrect.

*Criteria — Met:* knows the components, understands Adam stores m and v / *Missing:* FP16 vs FP32 distinction, master weights, activation memory, correct total (112+ GB), correct hardware requirement

---

**Hire**

*Interviewee:* "In mixed-precision training with Adam, you need: model weights in FP16 (14 GB), gradients in FP16 (14 GB), and Adam's first and second moment estimates in FP32 (28 GB each = 56 GB). You also keep FP32 master weights for the optimizer step (28 GB). That's 112 GB before activations. Activations add another 10-20 GB depending on batch size and sequence length. So you need at least two A100 80GB GPUs with FSDP, or one GPU with aggressive gradient checkpointing and batch size 1."

*Interviewer:* Solid breakdown with correct FP16/FP32 distinction. Knows about master weights. Gives the right total and the right hardware conclusion. Mentions gradient checkpointing as a mitigation. Would be a Strong Hire if they discussed how activation memory scales with sequence length or mentioned the trade-off of checkpointing (recompute cost).

*Criteria — Met:* correct component breakdown, FP16/FP32 distinction, master weights, correct total, practical hardware implications / *Missing:* activation scaling formula, gradient checkpointing trade-off details

---

**Strong Hire**

*Interviewee:* "Let me break it down precisely. With mixed precision and AdamW: FP16 weights = 2 bytes × 7B = 14 GB. FP16 gradients = 14 GB. FP32 master weights = 4 bytes × 7B = 28 GB. Adam m and v in FP32 = 2 × 28 GB = 56 GB. Total fixed cost: 112 GB. Activations scale as roughly L × n × d_model × b × k, where k ≈ 10 for a standard transformer. For LLaMA-7B (32 layers, d_model=4096, seq_len=2048, batch=1), that's roughly 10-15 GB. Total: ~127 GB. This needs at least two A100-80GB with FSDP. Gradient checkpointing can reduce activation memory to O(√L) ≈ 5-6 layers stored instead of 32, saving about 10 GB but adding ~30% wall-clock time. Alternatively, you can use DeepSpeed ZeRO Stage 3, which shards everything across GPUs. The real question is whether you even need full fine-tuning — for most tasks, LoRA gets you within 1-2% accuracy with 10× less memory."

*Interviewer:* Complete breakdown with exact numbers, correct formulas, knows the activation scaling, understands gradient checkpointing trade-off quantitatively, and spontaneously suggests LoRA as an alternative — showing systems-level judgment about when full FT is actually warranted.

*Criteria — Met:* exact memory math, FP16/FP32 breakdown, activation formula, gradient checkpointing trade-off, FSDP/DeepSpeed awareness, LoRA comparison

---

### Q2: What is catastrophic forgetting and how would you detect and mitigate it in production?

---

**No Hire**

*Interviewee:* "It's when the model forgets things after training. You should use a small learning rate to fix it."

*Interviewer:* The definition is vague. "Forgets things" does not convey the mechanism. Only one mitigation strategy mentioned, with no reasoning about why it works.

*Criteria — Met:* knows the term exists / *Missing:* mechanism explanation, detection strategy, multiple mitigations, understanding of why each works

---

**Weak Hire**

*Interviewee:* "Catastrophic forgetting is when fine-tuning on a new task makes the model lose its performance on previous tasks. This happens because you're changing all the parameters. To prevent it, use a low learning rate, early stopping, or mix in some general data."

*Interviewer:* Correct definition. Multiple mitigations listed. But no explanation of the mechanism (why changing parameters causes forgetting), no detection strategy, and no discussion of trade-offs.

*Criteria — Met:* correct definition, multiple mitigations / *Missing:* mechanism (parameter interference), detection methods, trade-off analysis, EWC or regularization-based approaches

---

**Hire**

*Interviewee:* "Catastrophic forgetting occurs because the same parameters that encode general knowledge are being moved by the optimizer to minimize the task-specific loss. The optimizer has no term in its objective that penalizes losing general capabilities — it only sees the task loss. The further the parameters move from the pre-trained initialization, the more general capability is lost. To detect it, I'd maintain a small evaluation set of general tasks and monitor it during training. For mitigation: (1) low learning rate with warmup to avoid destructive early updates, (2) mix 10-20% general data into training batches, (3) early stopping when the general eval starts degrading. There's also EWC, which adds a regularization term weighted by the Fisher information matrix — it penalizes changing parameters that were important for previous tasks — but it's expensive to compute and rarely used in practice."

*Interviewer:* Strong explanation of the mechanism. Practical detection strategy. Multiple mitigations with reasoning. Knows about EWC and correctly identifies its practical limitations. Would reach Strong Hire with a quantitative discussion of how learning rate interacts with forgetting, or production experience.

*Criteria — Met:* mechanism, detection, multiple mitigations with reasoning, EWC / *Missing:* quantitative relationship, production experience, LoRA comparison

---

**Strong Hire**

*Interviewee:* "The fundamental issue is that L_task has no term that preserves L_general. The optimizer moves θ₀ → θ* along the task-specific loss gradient, and any general knowledge encoded in directions orthogonal to the task gradient can be destroyed. For detection, I set up a general benchmark suite — typically MMLU or HellaSwag — and evaluate every 100 steps. I track the ratio: if general perf drops more than 5% relative while task perf is still improving, I flag it. Mitigation toolkit in order of effectiveness: (1) learning rate — I use 2e-5 with linear warmup over the first 10% of steps; the warmup is critical because the first few hundred steps do the most damage. (2) Data mixing at 10-20% general data. (3) Early stopping. (4) For extreme cases, EWC or L2-SP regularization toward θ₀ can help, but I've never needed them with proper LR and mixing. The cleaner solution is LoRA — it freezes θ₀ entirely and adds small trainable adapters, so forgetting is structurally impossible. In production, I'd default to LoRA and only use full FT if we need that last 1-2% accuracy and have enough data to justify it."

*Interviewer:* Deep mechanistic understanding expressed mathematically. Practical detection with specific thresholds. Ordered mitigation strategies with reasoning about why warmup matters. Connects to LoRA as the structural solution. Shows judgment about when to use full FT vs not.

*Criteria — Met:* full mechanism with math, detection with thresholds, ordered mitigations, EWC awareness, LoRA comparison, production judgment

---

### Q3: How does the learning rate for fine-tuning differ from pre-training, and why?

---

**No Hire**

*Interviewee:* "You use a smaller learning rate for fine-tuning because the model is already trained."

*Interviewer:* Correct at the surface level but no explanation of why. Does not discuss schedules, warmup, or what happens if the LR is too high or too low.

*Criteria — Met:* knows fine-tuning LR is smaller / *Missing:* explanation of why, specific ranges, warmup, schedule, consequences of wrong LR

---

**Weak Hire**

*Interviewee:* "Pre-training uses a larger learning rate, like 3e-4, and fine-tuning uses something like 2e-5. The model is already near a good solution from pre-training, so you want small steps to avoid destroying the learned features. You typically use a cosine decay schedule."

*Interviewer:* Knows the typical ranges and the basic reasoning. Mentions cosine decay. But no discussion of warmup, no explanation of why the pre-trained initialization is sensitive, and no distinction between the warmup and decay phases.

*Criteria — Met:* correct LR ranges, cosine decay, basic reasoning / *Missing:* warmup phase, loss landscape sensitivity, consequences of each phase, connection to forgetting

---

**Hire**

*Interviewee:* "Pre-training LRs are typically 1e-4 to 3e-4 because the model is starting from random initialization and needs to make large moves. Fine-tuning LRs are 1e-5 to 5e-5 because the parameters are already near a good solution — large steps would push them out of the basin of attraction found during pre-training. I always use warmup for the first 5-10% of steps. Without warmup, the very first gradient updates are unreliable — they're based on a small batch that may not be representative of the full dataset — and they can push parameters into a bad region. After warmup, I use cosine decay to zero. This lets the model settle into the minimum rather than bouncing around it."

*Interviewer:* Good explanation of the basin-of-attraction intuition. Understands warmup and can explain why it matters. Cosine decay with reasoning. Would reach Strong Hire with discussion of layer-wise LR or discriminative fine-tuning.

*Criteria — Met:* correct ranges with reasoning, warmup with explanation, cosine decay, basin-of-attraction concept / *Missing:* layer-wise LR, discriminative fine-tuning, quantitative effect of wrong LR

---

**Strong Hire**

*Interviewee:* "The pre-trained model sits at a critical point in the loss landscape — it's found a wide basin during pre-training that generalizes well. The fine-tuning learning rate controls how far we move from this point. Too high (e.g., 1e-3): we leave the basin entirely and lose pre-trained features — this manifests as catastrophic forgetting and unstable training loss. Too low (e.g., 1e-7): we barely move and the model doesn't learn the new task. The sweet spot is 1e-5 to 5e-5 for most models. I use linear warmup for the first 5-10% of steps because the initial gradient estimates are noisy (small batch, unfamiliar data distribution), and I want to let the optimizer build up accurate moment estimates before making real updates. After warmup, cosine decay to a small η_min. One refinement: discriminative fine-tuning uses different learning rates per layer — lower layers (which capture general features like syntax) get a smaller LR, upper layers (which capture task-specific features) get a larger LR. ULMFiT showed this gives 10-20% better results on small datasets."

*Interviewer:* Complete picture: loss landscape geometry, quantitative consequences of wrong LR, warmup with the moment-estimate reasoning, and discriminative fine-tuning as an advanced technique. The ULMFiT reference shows breadth.

*Criteria — Met:* loss landscape reasoning, quantitative consequences, warmup mechanics, cosine decay, discriminative fine-tuning, historical reference

---

### Q4: When would you choose full fine-tuning over LoRA, and what data requirements differ?

---

**No Hire**

*Interviewee:* "Full fine-tuning is when you update everything and LoRA is when you update a few parameters. Full fine-tuning needs more data."

*Interviewer:* Surface-level comparison. No reasoning about when each is appropriate, no quantitative data thresholds, no mention of trade-offs.

*Criteria — Met:* basic distinction / *Missing:* decision criteria, data requirements, trade-offs, quality comparison, practical scenarios

---

**Weak Hire**

*Interviewee:* "Full fine-tuning gives better results than LoRA because it updates more parameters. I'd use it when quality is the top priority and I have enough compute. LoRA is for when you have limited GPU resources. Full fine-tuning needs more data to avoid overfitting — maybe 10K+ examples."

*Interviewer:* Reasonable rules of thumb but the claim that full FT always gives better results is not true — LoRA often matches full FT quality, especially with good rank selection. No discussion of when each actually wins.

*Criteria — Met:* general comparison, data size consideration / *Missing:* nuanced quality comparison, domain shift reasoning, overfitting analysis, multi-task serving, specific scenarios

---

**Hire**

*Interviewee:* "The answer depends on three factors: domain distance, data volume, and serving requirements. Full fine-tuning wins when: (1) the target domain is very different from pre-training (e.g., English LLM → code + Japanese medical text), because the model needs to reshape many features; (2) you have 50K+ labeled examples, enough to justify updating all parameters without overfitting; (3) you only need one model variant, not multiple. LoRA wins when: (1) the task is a refinement of existing capabilities (e.g., a general chatbot → a customer support chatbot); (2) data is limited (1K-10K examples); (3) you need to serve multiple adapters on the same base model. In practice, I start with LoRA and only switch to full FT if LoRA plateaus below my quality threshold."

*Interviewer:* Good decision framework based on domain distance, data volume, and serving. The practical heuristic of starting with LoRA is sound. Would be Strong Hire with overfitting analysis and quantitative quality comparison from literature.

*Criteria — Met:* three-factor framework, practical heuristic, multi-adapter serving consideration / *Missing:* overfitting analysis, quantitative quality comparison, specific paper references

---

**Strong Hire**

*Interviewee:* "I'd frame this as a three-axis decision: domain shift, data volume, and operational requirements. For domain shift: the LoRA paper showed that low-rank updates capture most of the weight delta during fine-tuning — the fine-tuned weights are within a low-rank subspace of the pre-trained weights. This means LoRA should match full FT when the task aligns with the pre-training distribution. When there's a large domain shift (e.g., English text to protein sequences), the required weight delta may be full-rank, and LoRA will underperform. For data volume: full FT has more parameters to update, so it needs more data to avoid overfitting. The rule of thumb is 10-50× more examples than trainable parameters — for a 7B model, that's unrealistic (you'd need tens of billions of examples), which is why you always use regularization (weight decay, dropout, early stopping). In practice, full FT works with 50K+ examples for most NLP tasks. LoRA, with only millions of trainable parameters, works well with 1K-10K examples. For serving: LoRA adapters can be swapped at inference time — you serve one base model with many adapters. Full FT requires a separate model copy per task. At scale with 50+ task variants, the serving cost difference is 50×. I default to LoRA r=16 and only use full FT if the eval gap is >2% after rank ablation up to r=128."

*Interviewer:* Excellent analysis connecting LoRA theory (low-rank hypothesis) to practical decisions. Understands the overfitting dynamic quantitatively. Serving cost analysis shows production awareness. The default strategy with rank ablation shows disciplined experimentation.

*Criteria — Met:* low-rank hypothesis, domain shift reasoning, overfitting analysis, serving cost, practical default strategy with rank ablation

---

## Key Takeaways

🎯 1. Full fine-tuning updates every parameter. It gives the highest quality ceiling but costs the most in memory, compute, and risk.

🎯 2. Memory cost follows the formula: 16 bytes per parameter (FP16 weights + FP16 grads + FP32 master + FP32 Adam m + FP32 Adam v) plus activations. For 7B parameters: ~112 GB before activations.

3. Adam's m_t (momentum) and v_t (variance) are why the optimizer state costs 2× the model size in FP32. This is the dominant memory component.

4. Catastrophic forgetting happens because the loss function does not penalize losing general capabilities — only task error. The fix: low LR, warmup, data mixing, early stopping.

🎯 5. Learning rate warmup is not optional for fine-tuning. The first gradient updates on a new dataset are noisy and can irreversibly damage pre-trained features.

⚠️ 6. Start with LoRA. Only use full fine-tuning when LoRA quality is insufficient, the domain shift is large, and you have enough data and compute.

7. In production, gradient checkpointing and FSDP are the standard tools for fitting full fine-tuning into available hardware.
