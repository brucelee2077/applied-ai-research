# Inference Optimization

## What Is Inference Optimization?

**Inference** is when you use a trained model to make predictions. **Inference
optimization** is making that process faster, cheaper, and more memory-efficient.

Think of it like this: you've written a 500-page book (training). Now someone
asks you a question, and you need to look up the answer quickly (inference).
Optimization is about organizing the book so you can find answers faster.

```
+-------------------------------------------------------------------+
|              Why Optimize Inference?                               |
|                                                                   |
|   Problem                    Solution                             |
|   -------                    --------                             |
|   Model is too big           Quantization (smaller numbers)       |
|   for my GPU memory          Pruning (remove parts)               |
|                              Distillation (smaller model)         |
|                                                                   |
|   Model is too slow          Batching (process together)          |
|   for real-time use          Caching (don't repeat work)          |
|                              Hardware acceleration (better chips)  |
|                                                                   |
|   GPU costs too much         All of the above!                    |
|   per month                  (smaller + faster = cheaper)         |
+-------------------------------------------------------------------+
```

---

## The Optimization Landscape

There are many techniques, but they fall into a few categories:

```
+-------------------------------------------------------------------+
|              Inference Optimization Techniques                     |
|                                                                   |
|   MODEL-LEVEL (change the model itself):                          |
|     +-- Quantization: Use smaller numbers (FP16, INT8, INT4)     |
|     +-- Pruning: Remove unnecessary weights                       |
|     +-- Distillation: Train a smaller model to mimic the big one |
|     +-- Low-rank factorization: Simplify weight matrices         |
|                                                                   |
|   COMPUTATION-LEVEL (change how you compute):                     |
|     +-- Flash Attention: Faster attention calculation             |
|     +-- Operator fusion: Combine operations together              |
|     +-- KV-Cache: Don't recompute during text generation         |
|     +-- Speculative decoding: Draft with small model, verify     |
|                                                                   |
|   SYSTEM-LEVEL (change the infrastructure):                       |
|     +-- Batching: Process multiple requests together              |
|     +-- Model parallelism: Split across GPUs                      |
|     +-- Compilation: Convert to optimized format (TensorRT, etc.)|
+-------------------------------------------------------------------+
```

---

## Model-Level Optimizations

### Quantization (Most Impactful)

The single most important optimization. Uses **fewer bits** to represent
numbers in the model.

**Detailed coverage:** [Quantization](./quantization.md)

```
Quick summary:
  FP32 (default):  32 bits per number  -->  Model size: 100%
  FP16 (half):     16 bits per number  -->  Model size: ~50%
  INT8:             8 bits per number  -->  Model size: ~25%
  INT4:             4 bits per number  -->  Model size: ~12.5%

A 7B parameter model:
  FP32: ~28 GB     (needs expensive GPU)
  FP16: ~14 GB     (fits on a single good GPU)
  INT8:  ~7 GB     (fits on a consumer GPU!)
  INT4:  ~3.5 GB   (fits on a laptop GPU!)
```

### Pruning

Remove weights (connections) that don't contribute much to the output.
Like trimming a bush -- cut the dead branches, keep the shape.

### Knowledge Distillation

Train a smaller "student" model to copy a bigger "teacher" model.
Like a student taking notes from a professor's lecture -- the notes
are smaller but capture the key ideas.

### Low-Rank Factorization

Replace large weight matrices with smaller ones that approximate the
same computation. Think of it as compressing an image -- you lose a tiny
bit of quality but save a lot of space.

**Detailed coverage:** [Model Compression](./model-compression.md)

---

## Computation-Level Optimizations

### Flash Attention

The attention mechanism in transformers is slow because it creates a huge
matrix (sequence length x sequence length). **Flash Attention** computes
the same result without ever creating this full matrix, saving memory and time.

```
+-------------------------------------------------------------------+
|              Regular vs Flash Attention                            |
|                                                                   |
|   Regular Attention:                                              |
|     1. Compute full attention matrix (N x N)    <-- HUGE!        |
|     2. Apply softmax                                              |
|     3. Multiply by values                                         |
|     Memory: O(N^2)  -- grows rapidly with sequence length        |
|                                                                   |
|   Flash Attention:                                                |
|     1. Process in small blocks (tiles)                            |
|     2. Never store the full N x N matrix                          |
|     3. Same mathematical result!                                  |
|     Memory: O(N)  -- much more efficient                         |
|                                                                   |
|   For a 4096-token sequence:                                      |
|     Regular: ~16 million entries in attention matrix              |
|     Flash: processes in manageable chunks                         |
|     Speed improvement: 2-4x faster                               |
+-------------------------------------------------------------------+
```

### KV-Cache (Key-Value Cache)

When an LLM generates text token by token, it normally recomputes attention
over ALL previous tokens for each new token. KV-Cache stores the intermediate
results so each new token only needs to compute its own attention.

```
Without KV-Cache (wasteful):
  Token 1:  compute attention for [1]
  Token 2:  compute attention for [1, 2]         (1 already done!)
  Token 3:  compute attention for [1, 2, 3]      (1,2 already done!)
  Token 4:  compute attention for [1, 2, 3, 4]   (1,2,3 already done!)

With KV-Cache (efficient):
  Token 1:  compute [1], STORE result
  Token 2:  load [1] from cache, compute only [2], STORE
  Token 3:  load [1,2] from cache, compute only [3], STORE
  Token 4:  load [1,2,3] from cache, compute only [4], STORE
```

### Speculative Decoding

Use a **small, fast model** to generate a "draft" of several tokens, then
use the **big model** to verify them all at once. If the draft is right
(which it often is for simple text), you've generated multiple tokens in
the time it takes the big model to do one pass.

```
+-------------------------------------------------------------------+
|              Speculative Decoding                                  |
|                                                                   |
|   Small model (fast): Generates draft tokens quickly              |
|     "The capital of France is Paris, which is known for"          |
|                                                                   |
|   Big model (accurate): Verifies all draft tokens at once         |
|     "The capital of France is Paris, which is known for" -- OK!  |
|                                                                   |
|   If the big model disagrees with a token, it corrects from      |
|   that point and the small model tries again.                     |
|                                                                   |
|   Result: 2-3x faster generation with identical output quality!  |
+-------------------------------------------------------------------+
```

---

## System-Level Optimizations

### Model Compilation

Convert your model from a flexible Python format into an optimized, compiled
format that runs much faster on specific hardware.

| Tool | What It Does |
|------|-------------|
| **TensorRT** | NVIDIA's optimizer for GPU inference |
| **ONNX Runtime** | Cross-platform model optimization |
| **torch.compile** | PyTorch's built-in compilation (PyTorch 2.0+) |
| **TVM** | Optimizes for various hardware targets |

```
Before compilation:          After compilation:
  Python operations           Fused, optimized operations
  Flexible but slow           Hardware-specific but fast
  ~100 ms per inference       ~20 ms per inference
```

### Continuous Batching

Traditional batching waits for a batch to fill up before processing.
**Continuous batching** starts processing new requests as soon as previous
ones finish generating tokens, keeping the GPU busy at all times.

```
Traditional batching:
  Batch 1: [Req A, Req B, Req C] --> process --> [A done, B done, C done]
  (Must wait for ALL to finish before starting next batch)
  If A generates 10 tokens but C generates 100, GPU idles after A finishes

Continuous batching:
  [Req A, B, C processing...] --> A finishes --> [add Req D in A's slot]
  GPU is always fully utilized!
```

---

## Performance Metrics

How do you measure if your optimization worked?

| Metric | What It Measures | Better = |
|--------|-----------------|----------|
| **Latency** | Time for one request (ms) | Lower |
| **Throughput** | Requests per second | Higher |
| **Memory usage** | GPU RAM consumed (GB) | Lower |
| **Quality** | Model accuracy/perplexity after optimization | Higher (or "same as before") |
| **Tokens/sec** | How fast text is generated | Higher |
| **Time to first token** | How long until the first word appears | Lower |

```
+-------------------------------------------------------------------+
|              The Optimization Tradeoff                             |
|                                                                   |
|                High Quality                                       |
|                    ^                                               |
|                    |      * FP32 (original)                        |
|                    |                                               |
|                    |     * FP16                                    |
|                    |    * INT8                                     |
|                    |                                               |
|                    |  * INT4                                       |
|                    |                                               |
|                    +------------------------------> Low Cost       |
|                                                                   |
|   Every optimization trades some quality for speed/cost.          |
|   The goal: find the sweet spot where quality is "good enough"   |
|   and cost/speed meet your requirements.                          |
+-------------------------------------------------------------------+
```

---

## Practical Recommendations

```
+-------------------------------------------------------------------+
|              What Should I Optimize First?                         |
|                                                                   |
|   Impact (highest to lowest):                                     |
|                                                                   |
|   1. QUANTIZATION (FP16 or INT8)                                  |
|      Easiest, biggest impact. Just do it.                         |
|      2-4x memory reduction, often faster too.                     |
|                                                                   |
|   2. KV-CACHE + FLASH ATTENTION                                   |
|      Most serving frameworks include these by default.            |
|      Use vLLM or TGI and you get them for free.                  |
|                                                                   |
|   3. BATCHING                                                     |
|      Group requests together. Huge throughput improvement.        |
|      Most frameworks handle this automatically.                   |
|                                                                   |
|   4. MODEL COMPILATION                                            |
|      Convert to TensorRT or use torch.compile.                    |
|      Extra 20-50% speedup on top of everything else.             |
|                                                                   |
|   5. DISTILLATION / PRUNING                                       |
|      Only if you need a fundamentally smaller model.              |
|      More work, but can give 5-10x size reduction.               |
+-------------------------------------------------------------------+
```

---

## Summary

```
+------------------------------------------------------------------+
|           Inference Optimization Cheat Sheet                      |
|                                                                  |
|  Goal: Make models faster, smaller, cheaper to run               |
|                                                                  |
|  Most impactful:                                                 |
|    Quantization:       Use fewer bits (INT8/INT4). Do this first.|
|    Flash Attention:    2-4x faster attention. Use vLLM/TGI.     |
|    KV-Cache:           Don't recompute previous tokens.          |
|    Batching:           Process multiple requests together.       |
|                                                                  |
|  Advanced:                                                       |
|    Speculative decode: Draft with small model, verify with big.  |
|    Compilation:        Convert to optimized format (TensorRT).   |
|    Distillation:       Train smaller model to mimic big one.     |
|                                                                  |
|  The tradeoff: quality vs speed/cost. Find your sweet spot.     |
+------------------------------------------------------------------+
```

---

## Further Reading

- **FlashAttention: Fast and Memory-Efficient Exact Attention** -- Dao et al., 2022
- **Efficient Transformers: A Survey** -- Tay et al., 2020
- **Speculative Decoding** -- Leviathan et al., 2023
- **PagedAttention (vLLM)** -- Kwon et al., 2023

---

[Back to Deployment](../README.md)
