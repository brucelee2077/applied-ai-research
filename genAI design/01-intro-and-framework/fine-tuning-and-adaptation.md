# Fine-Tuning and Adaptation

## Introduction

One of the most important decisions in any genAI system design is: should we fine-tune the foundation model or use it as-is with prompting? Get this wrong in either direction and you pay a price — fine-tuning when prompting would suffice wastes weeks of engineering effort, and prompting when fine-tuning is needed leaves quality on the table.

The decision isn't binary. There's a spectrum of adaptation techniques, from zero-shot prompting to full fine-tuning, and each point on the spectrum trades off cost, control, and complexity differently. This page covers the full spectrum with practical guidance on when each approach makes sense.

---

## The Adaptation Spectrum

From cheapest/simplest to most expensive/most control:

| Technique | Data Needed | Compute | Time to Deploy | Control | Quality Ceiling |
|-----------|------------|---------|---------------|---------|----------------|
| Zero-shot prompting | None | None | Minutes | Low | Depends on task overlap with pretraining |
| Few-shot prompting | 3-10 examples | None | Minutes | Medium | Better, but context window limited |
| Prompt tuning | 100-1K examples | Low | Hours | Medium | Good for simple tasks |
| LoRA / QLoRA | 1K-10K examples | Moderate | Hours-days | High | Near full fine-tuning |
| Full fine-tuning | 10K+ examples | Very high | Days-weeks | Highest | Best possible |

**Decision framework:** Start at the left (zero-shot prompting). Move right only when the simpler approach demonstrably fails to meet your quality requirements. Each step right costs significantly more time, compute, and engineering complexity.

---

## When Fine-Tuning Is Necessary

Fine-tuning is worth the cost when prompting hits a ceiling you can't engineer around:

**Domain-specific knowledge.** The model doesn't know your internal terminology, your company's product taxonomy, or the specific patterns of medical/legal/financial language in your context. No amount of prompting can teach the model patterns it has never seen.

**Consistent output format.** Prompting can enforce format most of the time, but for production systems that need 99.9%+ format compliance across thousands of requests, fine-tuning on formatted examples is more reliable.

**Latency constraints.** A long system prompt with few-shot examples adds tokens to every request. A fine-tuned model internalizes those instructions, so the prompt can be shorter. For latency-sensitive applications, this matters — fewer input tokens = faster time-to-first-token.

**Cost at scale.** A 2000-token system prompt × 1M daily requests = 2B extra tokens per day. At $0.01/1K tokens, that's $20K/day in prompt overhead. Fine-tuning a smaller model that doesn't need the long prompt can pay for itself quickly.

**Privacy.** Fine-tuning lets you embed sensitive domain knowledge into the model weights instead of passing it in prompts. Prompts may be logged, inspected, or transmitted — model weights stay on your infrastructure.

---

## When Fine-Tuning Is NOT Necessary

**The task can be solved with good prompting.** Many tasks that seem to need fine-tuning actually just need better prompts. Before committing to fine-tuning, try: structured system prompts, few-shot examples, chain-of-thought, and output format specifications. If prompting gets you to 90% of your quality target, the last 10% may not justify fine-tuning.

**You have fewer than 100-500 high-quality examples.** Fine-tuning on tiny datasets leads to memorization, not generalization. The model learns the specific examples rather than the underlying pattern.

**The domain is already well-covered by the pretrained model.** If your task is standard English text generation, summarization, or translation, the pretrained model already handles it well. Fine-tuning adds complexity without meaningful quality improvement.

**Requirements change frequently.** Fine-tuning takes days to weeks. If your task definition, output format, or domain changes monthly, you'll be constantly retraining. Prompting changes in minutes.

---

## Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning updates every parameter in the model. For a 70B parameter model, this requires enormous GPU memory and compute. PEFT methods achieve near-full-fine-tuning quality by updating only a small fraction of parameters.

### LoRA (Low-Rank Adaptation)

The most widely used PEFT method. Freeze the base model and add small, trainable low-rank matrices to the attention layers.

**How it works:**
- For each weight matrix W in the model, add a low-rank update: `W' = W + A × B`
- A has shape (d × r), B has shape (r × d), where r << d (typically r = 8-64)
- Only A and B are trainable. The original W is frozen.
- At inference, merge: `W_merged = W + A × B`. No additional latency.

**Why it works:** The weight updates during fine-tuning tend to be low-rank — they don't need the full dimensionality of W to capture task-specific patterns. LoRA exploits this structure.

| Hyperparameter | Typical Values | Effect |
|---------------|---------------|--------|
| Rank (r) | 8-64 | Higher = more capacity, more parameters |
| Alpha (α) | 16-64 | Scaling factor. Higher = larger updates |
| Target modules | q_proj, v_proj (attention) | Which layers get LoRA adapters |
| Dropout | 0.05-0.1 | Regularization |

**Advantages:** 0.1-1% of parameters trained. Can be merged into base model at inference (no latency cost). Multiple LoRA adapters can be trained for different tasks and swapped at serving time.

### QLoRA

Quantize the base model to 4-bit precision, then apply LoRA on top.

**How it works:**
- Base model weights stored in 4-bit (NormalFloat4 quantization)
- LoRA adapters trained in FP16/BF16
- During forward pass: dequantize weights on-the-fly, apply LoRA

**Why it matters:** Fits fine-tuning of a 65B model on a single 48GB GPU. Without QLoRA, this would require 4-8 GPUs. Democratizes fine-tuning of large models.

**Quality:** Slightly below standard LoRA due to quantization noise, but the gap is small (typically <1% on benchmarks).

### Prompt Tuning / Prefix Tuning

Learn a set of continuous "soft prompt" tokens that are prepended to the input. The model itself is frozen.

**How it works:**
- Learn K embedding vectors (soft tokens) that are prepended to every input
- Only these K vectors are trainable. Typically K = 10-100.
- Total trainable parameters: K × embedding_dim (e.g., 50 × 4096 = 200K parameters)

**Advantages:** Extremely parameter-efficient (<0.01% of model parameters). Multiple prompts can be served from one model.
**Disadvantages:** Quality ceiling is lower than LoRA for complex tasks. Requires careful initialization.

### Adapters

Insert small trainable layers (bottleneck layers) between the frozen layers of the model.

**How it works:**
- After each transformer layer, add: input → down-project → nonlinearity → up-project → add residual
- Down-project reduces dimension (d → r), up-project restores it (r → d)
- Only the adapter layers are trainable

**Comparison with LoRA:** Similar concept (low-rank updates) but different placement. LoRA modifies existing weights; adapters add new layers. LoRA has no inference latency cost (merge weights); adapters add small latency (extra layers).

### Method Comparison

| Method | Parameters Trained | Memory | Data Needed | Quality | Inference Cost |
|--------|-------------------|--------|-------------|---------|---------------|
| Full fine-tuning | 100% | Very high (4-8 GPUs for 70B) | 10K+ examples | Best | Same as base |
| LoRA (rank 16) | ~0.5% | Moderate (1 GPU for 70B with QLoRA) | 1K-10K | Near full FT | Same (merged) |
| QLoRA | ~0.5% | Low (1 GPU for 65B) | 1K-10K | Slightly below LoRA | Slightly higher (dequantize) |
| Prompt tuning | <0.01% | Low | 100-1K | Good for simple tasks | Same |
| Adapters | 1-5% | Moderate | 1K-10K | Near full FT | Slightly higher |

---

## Fine-Tuning Data

### Quality Over Quantity

The single most important factor in fine-tuning quality is data quality, not quantity. 1,000 carefully curated, high-quality examples consistently outperform 100,000 noisy ones.

**What "high quality" means:**
- Accurate and correct (no factual errors in the target outputs)
- Representative of the distribution you'll see at serving time
- Diverse (covers edge cases, different phrasings, different topics within your domain)
- Consistently formatted (if you want JSON output, every example should have valid JSON)

### Data Format

| Fine-Tuning Type | Data Format | Example |
|-----------------|------------|---------|
| Instruction tuning | (instruction, response) pairs | ("Summarize this article: ...", "The article discusses...") |
| Chat fine-tuning | Multi-turn conversations | [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}] |
| Completion | (prompt, completion) pairs | ("def sort_list(", "arr):\n return sorted(arr)") |

### Synthetic Data Generation

Use a stronger model to generate training data for fine-tuning a weaker model. This is a form of distillation.

**How it works:**
1. Define your target task with detailed instructions and examples
2. Use a strong model (GPT-4, Claude) to generate thousands of (input, output) pairs
3. Filter for quality (human review, automated checks)
4. Fine-tune a smaller, cheaper model on this data

**When it works:** The strong model can do the task well, and you need a cheaper/faster model for production. Common for specialized classifiers, format conversion, and domain adaptation.

**When it fails:** The strong model has the same gaps as the weak model — no new knowledge is added. Or the synthetic data lacks the diversity and edge cases of real data.

### Data Contamination

Ensure your evaluation data doesn't leak into the fine-tuning set. If the model has seen the test examples during fine-tuning, evaluation metrics are meaningless.

**Prevention:** Strict data splits before fine-tuning. Hash-based deduplication between fine-tuning and evaluation sets. Hold out evaluation data before any data processing.

---

## Common Failure Modes

### Catastrophic Forgetting

The model loses general capabilities after fine-tuning on a narrow task. Fine-tuning on medical Q&A might make the model great at medical questions but terrible at everything else.

**Symptoms:** Model can't handle basic tasks it could do before fine-tuning. General knowledge questions get worse. The model becomes a one-trick specialist.

**Prevention:**
- Use PEFT methods (LoRA, QLoRA) — only a small fraction of parameters change, preserving most general knowledge
- Mix in general-purpose data during fine-tuning (10-20% general, 80-90% task-specific)
- Use a low learning rate (1e-5 to 5e-5 for full fine-tuning, 1e-4 to 3e-4 for LoRA)
- Monitor general capability benchmarks during fine-tuning

### Overfitting

The model memorizes training examples instead of learning patterns. With small fine-tuning datasets (< 1K examples), this is a significant risk.

**Symptoms:** Training loss decreases while validation loss increases. Model outputs look like copy-pasted training examples. Performance on held-out data is poor.

**Prevention:** Early stopping based on validation loss. LoRA regularization (dropout, low rank). More diverse training data.

### Alignment Tax

Fine-tuning can degrade the safety alignment that was established through RLHF/DPO. The model may become more helpful for your specific task but also more willing to follow harmful instructions.

**Why it happens:** RLHF alignment is stored in the model's weights. Fine-tuning modifies those weights. If the fine-tuning data doesn't include safety examples, the safety behavior erodes.

**Prevention:** Include safety-relevant examples in the fine-tuning data. Run safety evaluations before and after fine-tuning. Use PEFT methods (less weight modification = less alignment degradation).

### Format Collapse

After fine-tuning on a specific output format, the model only generates in that format — even when the user asks for something different.

**Example:** Fine-tuning on JSON outputs → model produces JSON even for conversational questions.

**Prevention:** Include diverse output formats in fine-tuning data. Mix in general conversation examples.

---

## Evaluation After Fine-Tuning

Fine-tuning evaluation requires more than just checking task accuracy. You need to verify the model didn't break anything.

| Evaluation | What to Check | Why |
|-----------|--------------|-----|
| Task-specific held-out set | Accuracy on the fine-tuned task | The primary goal of fine-tuning |
| General capability benchmarks | MMLU, HellaSwag, ARC, etc. | Check for catastrophic forgetting |
| Safety evaluations | ToxiGen, BBQ, red-team prompts | Check alignment wasn't degraded |
| Format compliance | % of outputs in correct format | Verify consistent output structure |
| A/B test vs base model + prompting | User preference, task success rate | Validate fine-tuning is actually better |

The A/B test against base-model-with-prompting is the most important evaluation. If prompting achieves similar quality, fine-tuning was unnecessary — and you've added a maintenance burden (retraining pipeline, data curation, alignment monitoring) for no benefit.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should understand the difference between prompting and fine-tuning and know when each is appropriate. For a customer support chatbot, they should recognize that few-shot prompting is the starting point and fine-tuning is an option when prompting can't achieve consistent enough quality. They differentiate by mentioning that fine-tuning requires training data and can be expensive, showing awareness of the cost-benefit tradeoff.

### Senior Engineer

Senior candidates can navigate the full adaptation spectrum. They know LoRA and can explain why it's preferred over full fine-tuning for most applications (less compute, less catastrophic forgetting, swappable adapters). For a domain-specific question-answering system, a senior candidate would propose starting with RAG + prompting, and only escalating to LoRA fine-tuning if domain-specific patterns can't be captured through retrieval alone. They proactively discuss evaluation strategy: task-specific accuracy, catastrophic forgetting checks, and A/B testing against the prompting baseline.

### Staff Engineer

Staff candidates treat the fine-tuning decision as a system design problem with cost, quality, and operational implications. They recognize that fine-tuning creates an ongoing maintenance burden — data curation, retraining pipelines, alignment monitoring — and weigh that against the quality improvement. A Staff candidate might propose a multi-model architecture where a large prompted model handles complex queries while a small fine-tuned model handles common, well-defined queries — optimizing cost and latency without sacrificing quality for the hard cases. They also consider the organizational dimension: who curates the fine-tuning data, how do you prevent data quality degradation over time, and how do you detect when the fine-tuned model needs retraining.
