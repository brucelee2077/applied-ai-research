# Latency and Cost Optimization

## Introduction

GenAI models are expensive to run and slow to generate. A single GPT-4 query costs roughly $0.01-0.10. An image generation takes 2-10 seconds. These numbers seem small — until you multiply them by millions of daily users.

Every production genAI system lives inside a cost-latency-quality triangle. You can optimize for any two at the expense of the third. The candidates who stand out in interviews are those who can navigate this triangle: articulate which corner they're optimizing for, what they're trading away, and how they'd implement the tradeoffs at scale.

---

## The Cost Problem

### LLM Inference Cost

LLM cost is proportional to model size × sequence length × number of requests.

| Model Size | Cost per 1M Input Tokens | Cost per 1M Output Tokens | Typical Use |
|-----------|------------------------|-------------------------|-------------|
| Small (7-13B) | $0.05-0.20 | $0.10-0.30 | Simple tasks, classification, extraction |
| Medium (30-70B) | $0.30-1.00 | $0.60-2.00 | General assistant, moderate reasoning |
| Large (100B+) | $2.00-10.00 | $5.00-30.00 | Complex reasoning, code generation |

**At scale:** A chatbot serving 1M daily users, averaging 500 tokens per interaction:
- Small model: ~$100/day
- Large model: ~$5,000-15,000/day

This is just inference cost — not including infrastructure, engineering, or data costs.

### Image Generation Cost

Image generation cost is proportional to model size × number of diffusion steps × resolution.

| Configuration | Time per Image | GPU Cost per Image | Daily Cost at 100K Images |
|--------------|---------------|-------------------|--------------------------|
| SD 1.5, 20 steps, 512×512 | ~2s on A100 | ~$0.002 | ~$200 |
| SDXL, 30 steps, 1024×1024 | ~5s on A100 | ~$0.005 | ~$500 |
| SDXL, 50 steps, 1024×1024 | ~8s on A100 | ~$0.008 | ~$800 |

### Why Cost Optimization Is Existential

At startup scale (thousands of users), cost is manageable. At production scale (millions of users), GenAI inference can become the single largest infrastructure cost — exceeding compute for the rest of the application stack combined. Teams that can't optimize costs either run out of money or degrade their product.

---

## Model-Level Optimizations

### Quantization

Reduce numerical precision of model weights to speed up computation and reduce memory.

| Precision | Memory Reduction | Speed Improvement | Quality Impact | Effort |
|-----------|-----------------|-------------------|---------------|--------|
| FP32 → FP16/BF16 | 2x | 1.5-2x | Negligible | Trivial (one flag) |
| FP16 → INT8 (GPTQ, AWQ) | 2x (4x from FP32) | 2-3x | Small (<1% on benchmarks) | Low |
| FP16 → INT4 (GPTQ, AWQ) | 4x (8x from FP32) | 3-4x | Moderate (1-3% on benchmarks) | Low |

**FP16/BF16 is free.** Always use it. There's no reason to run inference in FP32.

**INT8 is the production standard.** The quality loss is negligible for most applications. Do this before anything else.

**INT4 is for cost-constrained settings.** Quality drops are measurable but often acceptable. Best combined with larger models (quantized 70B can outperform unquantized 13B).

### Distillation

Train a smaller model to mimic a larger model's outputs.

**How it works:**
1. Generate outputs from the large (teacher) model for a diverse set of inputs
2. Train a smaller (student) model to produce the same outputs
3. Deploy the student model at a fraction of the cost

**Results:** A well-distilled 7B model can achieve 80-90% of a 70B model's quality on the specific task it was distilled for.

**Tradeoff:** Significant upfront cost (need to generate training data from the teacher and train the student). But the ongoing inference cost reduction makes it worthwhile for high-volume applications.

### Pruning

Remove unnecessary parameters or attention heads from the model.

- **Unstructured pruning:** Zero out individual weights. Modest speedup without hardware-specific optimization.
- **Structured pruning:** Remove entire attention heads, layers, or channels. Better speedup, but more quality loss.

Pruning is less popular than quantization and distillation because the speedup-quality tradeoff is generally worse.

---

## KV-Cache Management

### What the KV-Cache Is

In autoregressive generation, each new token attends to all previous tokens. The key and value matrices for previous tokens don't change — they can be cached to avoid recomputation.

**Without KV-cache:** Each new token requires recomputing attention over the entire sequence. O(n²) per token.
**With KV-cache:** Each new token only computes attention against cached keys/values. O(n) per token.

The KV-cache is essential for efficient generation. But it consumes GPU memory proportional to:

`memory = batch_size × num_layers × 2 × num_heads × seq_len × head_dim × precision_bytes`

For a 70B model serving a batch of 32 sequences at 4096 tokens each in FP16: ~40GB just for the KV-cache.

### PagedAttention (vLLM)

The key innovation: manage KV-cache like virtual memory pages.

**The problem:** Traditional KV-cache allocates contiguous memory for the maximum possible sequence length. For a 4096-token limit, every sequence reserves 4096 tokens of memory — even if the actual sequence is only 200 tokens. This wastes 95% of the allocated memory.

**The solution:** PagedAttention allocates KV-cache in small pages (blocks of 16-64 tokens). Pages are allocated on demand as the sequence grows. No more wasted memory.

**Result:** 2-4x improvement in batch size → 2-4x improvement in throughput. This is why vLLM has become the standard serving engine for LLMs.

### Prefix Caching

Share KV-cache for common system prompts across requests.

**How it works:** If 1000 requests share the same 500-token system prompt, compute the KV-cache for the system prompt once and share it across all requests.

**Impact:** For applications with long system prompts (which is most production applications), prefix caching reduces time-to-first-token by 30-70%.

---

## Batching Strategies

Batching multiple requests together amortizes the cost of loading model weights — the weights are loaded once and applied to multiple inputs.

| Strategy | How It Works | Latency | Throughput | Best For |
|----------|-------------|---------|-----------|----------|
| No batching | Process one request at a time | Lowest individual | Lowest | Testing, single-user |
| Static batching | Wait for N requests, process together | High (waiting) | Medium | Batch processing |
| Dynamic batching | Batch requests arriving within a time window | Moderate | Good | Online serving |
| Continuous batching | Process different requests at different stages simultaneously | Low | Best | Production serving |

### Continuous Batching (The Standard)

Different requests have different sequence lengths and are at different stages of generation. Continuous batching processes them simultaneously:

- When a request finishes generating, its GPU slot is immediately filled by a new request
- No waiting for all requests in a batch to finish before starting new ones
- Maximizes GPU utilization

This is the default in vLLM, TensorRT-LLM, and other production serving engines. If you're not using continuous batching, you're leaving significant throughput on the table.

---

## Speculative Decoding

Use a small, fast model to draft tokens, then verify them with the large model in parallel.

### How It Works

1. **Draft phase:** Small model (e.g., 1B parameters) generates N candidate tokens autoregressively. This is fast because the model is small.
2. **Verify phase:** Large model (e.g., 70B parameters) processes all N tokens in a single forward pass. It checks each token against its own distribution.
3. **Accept/reject:** Where the large model agrees with the small model, accept the token. At the first disagreement, sample from the large model's distribution and discard remaining draft tokens.

### Key Properties

- **Lossless:** The output distribution is mathematically identical to running the large model alone. No quality loss.
- **Speedup:** 2-3x typical. Depends on the acceptance rate (how often the small model matches the large model).
- **When acceptance rate is high:** For simple continuations ("the" → "cat" → "sat"), the small model matches the large model frequently. High speedup.
- **When acceptance rate is low:** For complex reasoning or creative text, the models diverge more often. Lower speedup.

### Draft Model Selection

| Draft Model | Acceptance Rate | Speed | Notes |
|-------------|----------------|-------|-------|
| Same architecture, fewer layers | 70-85% | 4-8x faster per token | Best acceptance, moderate speedup |
| Same architecture, quantized | 65-80% | 2-4x faster per token | Good balance |
| Different smaller architecture | 50-70% | 10-20x faster per token | Fast drafting, lower acceptance |

---

## System Architecture for Cost

### Model Routing

Route queries to different models based on difficulty. Simple queries → cheap model. Complex queries → expensive model.

**How it works:**
1. **Classify query difficulty:** Use a lightweight classifier (or heuristic) to estimate complexity
2. **Route:** Easy queries (factual lookup, simple formatting) → 7B model. Hard queries (complex reasoning, creative writing) → 70B model.
3. **Fallback:** If the small model's response fails quality checks, retry with the large model.

**Impact:** 50-80% of queries can be handled by the small model, reducing average cost by 3-5x.

**The classification challenge:** A simple classifier that gets routing wrong 10% of the time sends 5% of easy queries to the expensive model (wasted money) and 5% of hard queries to the cheap model (degraded quality). The classifier needs to be conservative — when in doubt, route to the larger model.

### Semantic Caching

Cache responses for semantically similar queries.

**How it works:**
1. Embed the user query
2. Check if a semantically similar query (cosine similarity > threshold) has been answered recently
3. If yes, return the cached response
4. If no, run the full generation pipeline and cache the result

**Hit rate:** Depends on the application. FAQ-style queries: 30-60% hit rate. Open-ended conversations: 5-15%.

**Risks:**
- Cached responses may be stale (information changed since caching)
- Semantic similarity threshold too low → serve irrelevant cached responses
- Personalized queries → cached response from another user may not be appropriate

### Prompt Compression

Reduce the number of input tokens to reduce cost and latency.

| Technique | How It Works | Token Reduction | Quality Impact |
|-----------|-------------|-----------------|---------------|
| System prompt optimization | Rewrite system prompt to be shorter without losing meaning | 20-50% | None if done well |
| Conversation summarization | Summarize past messages instead of including full history | 50-80% | Some context loss |
| Retrieved context pruning | Only include the most relevant chunks, not all retrieved | 30-60% | Better focus, less noise |
| LLMLingua / compressors | ML-based prompt compression | 40-60% | Small quality loss |

---

## The Cost-Latency-Quality Triangle

You can optimize for any two at the expense of the third.

| Optimize For | What You Trade |
|-------------|---------------|
| Low cost + Low latency | Smaller model → lower quality |
| Low cost + High quality | Batch processing → higher latency |
| Low latency + High quality | Large model, fast hardware → higher cost |

**Interview approach:** State which corner you're optimizing for and what you're willing to trade. Different products make different choices:

| Product | Priority | Trade |
|---------|---------|-------|
| Real-time chatbot | Low latency + good quality | Higher cost |
| Batch content generation | Low cost + high quality | Higher latency |
| Interactive coding assistant | Low latency + high quality | Higher cost |
| Email draft suggestions | Low cost + low latency | Moderate quality acceptable |

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should recognize that LLM inference is expensive and latency-sensitive, and mention at least one optimization technique (quantization, smaller model, caching). For a chatbot system, they should acknowledge the latency requirements (users expect sub-second time-to-first-token) and propose basic optimizations like INT8 quantization and response caching for common queries. They differentiate by understanding the cost structure — cost scales with model size and token count.

### Senior Engineer

Senior candidates can navigate the cost-latency-quality triangle explicitly. They discuss quantization (INT8 as the production standard), KV-cache management (PagedAttention for throughput), and batching strategies (continuous batching). For a high-traffic AI assistant, a senior candidate would propose model routing (small model for simple queries, large model for complex ones), speculative decoding for latency, and prompt compression to reduce per-request token count. They bring up the tradeoff explicitly: "We can serve 3x more users by using INT4 quantization, at the cost of ~2% quality regression on benchmarks."

### Staff Engineer

Staff candidates think about cost optimization as a system design problem, not just an inference trick. They recognize that the biggest cost lever is often not model optimization — it's reducing the number of tokens generated or the number of expensive model calls. A Staff candidate might propose a multi-tier architecture: a router that handles 60% of queries with a small, cheap model and cached responses; a medium model for 30% of queries that require moderate reasoning; and the large model for the 10% of queries that genuinely need it. They also think about the economic dimension — at what daily volume does it become cheaper to fine-tune and serve your own model vs calling an API? What's the ROI on investing engineering time in serving infrastructure vs paying higher API costs?
