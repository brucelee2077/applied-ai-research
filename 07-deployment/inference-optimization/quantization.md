# Quantization

## What Is Quantization?

Imagine you have a ruler marked in millimeters (very precise). If you switch
to a ruler marked in centimeters (less precise), you lose a tiny bit of accuracy,
but the ruler is **much simpler and faster to read**. That's quantization.

**Quantization** means representing model weights with **fewer bits** (smaller
numbers). Instead of using 32-bit floating point numbers (very precise), you
use 16-bit, 8-bit, or even 4-bit numbers (less precise, but much smaller).

```
+-------------------------------------------------------------------+
|              Quantization: The Core Idea                           |
|                                                                   |
|   Original weight (FP32):  3.14159265358979...                    |
|     Stored as: 32 bits (very precise)                             |
|                                                                   |
|   Quantized to FP16:  3.14159...                                 |
|     Stored as: 16 bits (still pretty precise)                    |
|                                                                   |
|   Quantized to INT8:  3                                           |
|     Stored as: 8 bits (less precise, but much smaller!)          |
|                                                                   |
|   Quantized to INT4:  3                                           |
|     Stored as: 4 bits (least precise, but tiny!)                 |
|                                                                   |
|   Each step: half the bits = half the memory = faster inference  |
+-------------------------------------------------------------------+
```

---

## Why Does Quantization Matter?

This is the **single most impactful** optimization for deploying AI models.

```
+-------------------------------------------------------------------+
|              LLaMA-7B Model Size at Different Precisions          |
|                                                                   |
|   FP32 (32-bit):   ~28 GB    Needs: A100 80GB or multiple GPUs  |
|   FP16 (16-bit):   ~14 GB    Needs: A100 40GB or RTX 4090       |
|   INT8 (8-bit):     ~7 GB    Needs: RTX 3090 or RTX 4080        |
|   INT4 (4-bit):    ~3.5 GB   Needs: RTX 3060 or even a laptop!  |
|                                                                   |
|   Same model, same knowledge, but 8x smaller!                    |
|   And the quality loss is often negligible.                       |
|                                                                   |
|   This is why you can run chatbots on your laptop now --         |
|   quantization made it possible.                                 |
+-------------------------------------------------------------------+
```

---

## Number Formats Explained

To understand quantization, you need to know how computers store numbers:

### Floating Point (FP32, FP16, BF16)

```
+-------------------------------------------------------------------+
|              Floating Point Numbers                                |
|                                                                   |
|   FP32 (32 bits): [1 sign][8 exponent][23 mantissa]              |
|     Range: +/- 3.4 x 10^38                                       |
|     Precision: ~7 decimal digits                                  |
|     Used in: Default model training                               |
|                                                                   |
|   FP16 (16 bits): [1 sign][5 exponent][10 mantissa]              |
|     Range: +/- 65,504                                             |
|     Precision: ~3 decimal digits                                  |
|     Used in: Mixed-precision training, inference                  |
|                                                                   |
|   BF16 (16 bits): [1 sign][8 exponent][7 mantissa]               |
|     Range: Same as FP32 (+/- 3.4 x 10^38)                        |
|     Precision: ~2 decimal digits                                  |
|     Used in: Training on modern GPUs (A100, H100)                |
|     Why: Same range as FP32 but half the memory                  |
|                                                                   |
|   Think of it like scientific notation:                           |
|     FP32: 3.1415926 x 10^2  (very precise)                       |
|     FP16: 3.14 x 10^2       (pretty good)                        |
|     BF16: 3.1 x 10^2        (less precise but handles big numbers)|
+-------------------------------------------------------------------+
```

### Integer (INT8, INT4)

```
+-------------------------------------------------------------------+
|              Integer Quantization                                  |
|                                                                   |
|   INT8 (8 bits):                                                  |
|     Range: -128 to 127 (or 0 to 255 unsigned)                    |
|     Only 256 possible values                                      |
|     Must MAP floating point weights to this small range           |
|                                                                   |
|   INT4 (4 bits):                                                  |
|     Range: -8 to 7 (or 0 to 15 unsigned)                         |
|     Only 16 possible values!                                      |
|     Very aggressive -- some quality loss expected                 |
|                                                                   |
|   How mapping works (simplified):                                 |
|     Original weights: [0.12, 0.45, -0.30, 0.88, -0.15]          |
|     Find min (-0.30) and max (0.88)                               |
|     Map this range to 0-255 (INT8)                                |
|     Quantized:  [91, 164, 0, 255, 33]                            |
|     Store the scale factor to convert back                        |
+-------------------------------------------------------------------+
```

---

## Quantization Methods

### 1. Post-Training Quantization (PTQ)

Take an already-trained model and quantize it. No retraining needed.
Like converting a high-resolution photo to a smaller file -- quick and easy.

```
+-------------------------------------------------------------------+
|              Post-Training Quantization                            |
|                                                                   |
|   [Trained Model (FP32)] --> [Quantize weights] --> [Model (INT8)]|
|                                                                   |
|   Steps:                                                          |
|     1. Load your trained FP32 model                               |
|     2. For each layer, find the range of weight values            |
|     3. Map those values to INT8 (or INT4)                         |
|     4. Store the scale/zero-point for converting back             |
|     5. Done! Model is now 2-4x smaller                            |
|                                                                   |
|   Types:                                                          |
|     Weight-only: Only quantize the stored weights                 |
|       (activations stay in FP16 during computation)               |
|     Weight + Activation: Quantize both                            |
|       (faster computation but may lose more quality)              |
+-------------------------------------------------------------------+
```

**Popular PTQ tools:**

| Tool | What It Does | Best For |
|------|-------------|----------|
| **GPTQ** | GPU-based quantization, very popular for LLMs | 4-bit LLM quantization |
| **AWQ** | Activation-aware quantization (preserves important weights) | High-quality 4-bit |
| **bitsandbytes** | Easy-to-use library by Tim Dettmers | Quick INT8/INT4 with HuggingFace |
| **llama.cpp** | CPU-friendly quantization (GGUF format) | Running models on CPU/laptop |

### 2. Quantization-Aware Training (QAT)

Train the model while **simulating** quantization, so it learns to be robust
to the precision loss. Like practicing a speech with a slightly broken
microphone -- you learn to speak clearly despite the limitations.

```
+-------------------------------------------------------------------+
|              Quantization-Aware Training                           |
|                                                                   |
|   During training:                                                |
|     Forward pass:  Simulate quantization (fake INT8 values)       |
|     Backward pass: Use full precision gradients (FP32)            |
|     The model learns to work well with quantized weights          |
|                                                                   |
|   [Model] --> [Train with fake quantization] --> [Quantize for real]|
|                                                                   |
|   Result: Better quality than PTQ, but requires retraining       |
|                                                                   |
|   When to use:                                                    |
|     - When PTQ quality is not good enough                         |
|     - When you have access to training data and compute           |
|     - When you need INT4 with minimal quality loss               |
+-------------------------------------------------------------------+
```

### PTQ vs QAT

| | Post-Training (PTQ) | Quantization-Aware (QAT) |
|--|---------------------|--------------------------|
| **Effort** | Very easy (just convert) | Requires retraining |
| **Speed** | Minutes | Hours/days |
| **Quality** | Good (especially INT8) | Best (especially INT4) |
| **Data needed** | Small calibration set (or none) | Full training dataset |
| **Best for** | INT8, quick deployment | INT4, maximum quality |

---

## Popular Quantization Formats

### GGUF (llama.cpp)

The format used by llama.cpp for running models on CPUs and laptops.
You'll see models named like `llama-7b-Q4_K_M.gguf`:

```
+-------------------------------------------------------------------+
|              GGUF Quantization Levels                              |
|                                                                   |
|   Format    Bits   Size (7B model)   Quality                     |
|   ------    ----   ---------------   -------                     |
|   Q2_K      2      ~2.7 GB          Lowest (noticeable loss)     |
|   Q3_K_M    3      ~3.3 GB          Low-medium                   |
|   Q4_K_M    4      ~4.1 GB          Good (recommended default)   |
|   Q5_K_M    5      ~4.8 GB          Very good                    |
|   Q6_K      6      ~5.5 GB          Near-original                |
|   Q8_0      8      ~7.2 GB          Almost identical to FP16     |
|   F16       16     ~14 GB           Original quality             |
|                                                                   |
|   Q4_K_M is the sweet spot for most people:                       |
|   3.5x smaller than FP16 with minimal quality loss.              |
+-------------------------------------------------------------------+
```

### GPTQ

Optimized for GPU inference. Uses clever math to decide which weights
matter most and preserve their precision.

### AWQ (Activation-Aware Weight Quantization)

Observes which weights have the biggest impact on activations and keeps
those at higher precision. Gets better quality than GPTQ at the same
bit width.

---

## The Quality vs Size Tradeoff

How much quality do you actually lose? Here's a rough guide:

```
+-------------------------------------------------------------------+
|              Quality Impact by Quantization Level                  |
|                                                                   |
|   FP32 --> FP16:   ~0% quality loss                               |
|     (Almost always safe. Just do it.)                             |
|                                                                   |
|   FP16 --> INT8:   ~0-1% quality loss                             |
|     (Usually safe. Recommended for production.)                   |
|                                                                   |
|   FP16 --> INT4:   ~1-5% quality loss                             |
|     (Noticeable on benchmarks, but often fine in practice.)       |
|     (Use GPTQ or AWQ for best results.)                          |
|                                                                   |
|   FP16 --> INT2-3: ~5-15% quality loss                            |
|     (Significant degradation. Only for memory-constrained cases.) |
|                                                                   |
|   Bigger models tolerate quantization BETTER:                     |
|     - 70B model at INT4 > 7B model at FP16 (often!)              |
|     - When in doubt, use a bigger quantized model                 |
|       rather than a smaller full-precision model                  |
+-------------------------------------------------------------------+
```

---

## Key Research

### LLM.int8() (Dettmers et al., 2022)

Discovered that a few "outlier" weights in large models are extremely
important. Their solution: keep those outlier weights in FP16 while
quantizing everything else to INT8. This made INT8 work well even for
very large models.

### SmoothQuant (Xiao et al., 2022)

Found that activations are harder to quantize than weights because they
have outliers. Solution: "smooth" the activations by shifting difficulty
from activations to weights (which are easier to quantize).

### GPTQ (Frantar et al., 2022)

A one-shot weight quantization method that processes weights layer by
layer, using a small calibration dataset to minimize the error introduced
by quantization. Can quantize a 175B model to 3-4 bits.

---

## Summary

```
+------------------------------------------------------------------+
|              Quantization Cheat Sheet                             |
|                                                                  |
|  What:     Use fewer bits to represent model weights             |
|  Why:      2-8x smaller models, faster inference, less memory    |
|  Tradeoff: Slight quality loss (often negligible)                |
|                                                                  |
|  Quick guide:                                                    |
|    FP32 --> FP16:  Always do this. Zero downside.               |
|    FP16 --> INT8:  Recommended. Minimal quality loss.            |
|    FP16 --> INT4:  Great for running on consumer hardware.       |
|                                                                  |
|  Tools:                                                          |
|    bitsandbytes:  Easy INT8/INT4 with HuggingFace               |
|    GPTQ:          Best GPU INT4 quantization                     |
|    AWQ:           Highest quality INT4                            |
|    llama.cpp:     Run on CPU/laptop (GGUF format)               |
|                                                                  |
|  Rule of thumb:                                                  |
|    A bigger model quantized to 4-bit often beats                |
|    a smaller model at full precision.                            |
+------------------------------------------------------------------+
```

---

## Further Reading

- **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** -- Dettmers et al., 2022
- **GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers** -- Frantar et al., 2022
- **AWQ: Activation-aware Weight Quantization for LLM Compression** -- Lin et al., 2023
- **SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs** -- Xiao et al., 2022
- **QLoRA: Efficient Finetuning of Quantized LLMs** -- Dettmers et al., 2023

---

[Back to Inference Optimization](./README.md) | [Back to Deployment](../README.md)
