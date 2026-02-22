# Model Compression

## What Is Model Compression?

You have a massive neural network with billions of parameters, but you need
it to run on a phone, or you need it to be 10x faster, or your GPU bill is
too high. **Model compression** is the set of techniques for making models
smaller while keeping them (nearly) as smart.

Think of it like packing for a trip. Your closet has 200 items, but your
suitcase only fits 30. You need to pick the essentials and leave behind
what you won't need. The goal: arrive with everything you need, in a much
smaller bag.

```
+-------------------------------------------------------------------+
|              Model Compression Techniques                          |
|                                                                   |
|   [Big Model: 7B params, 28GB]                                    |
|        |                                                          |
|        +---> Quantization (use fewer bits)                        |
|        |       --> 3.5 GB  (covered in quantization.md)           |
|        |                                                          |
|        +---> Pruning (remove unneeded weights)                    |
|        |       --> 2-4x smaller                                   |
|        |                                                          |
|        +---> Knowledge Distillation (train smaller model)         |
|        |       --> 500M params, same quality                      |
|        |                                                          |
|        +---> Low-Rank Factorization (simplify matrices)           |
|                --> 1-2x smaller                                   |
+-------------------------------------------------------------------+
```

---

## 1. Pruning: Removing What You Don't Need

Neural networks are **over-parameterized** -- they have more connections than
they actually need. Pruning removes the connections (weights) that contribute
the least, like pulling weeds from a garden.

### Types of Pruning

```
+-------------------------------------------------------------------+
|              Pruning Types                                         |
|                                                                   |
|   UNSTRUCTURED PRUNING (remove individual weights):               |
|                                                                   |
|     Before:                After:                                 |
|     [0.5  0.01 0.8]       [0.5  0    0.8]                       |
|     [0.3  0.7  0.02]  --> [0.3  0.7  0   ]                      |
|     [0.9  0.04 0.6]       [0.9  0    0.6]                        |
|                                                                   |
|     Remove weights close to zero (they barely matter).            |
|     Can remove 50-90% of weights!                                 |
|     Problem: Sparse matrices are hard to speed up on GPUs.        |
|                                                                   |
|   STRUCTURED PRUNING (remove entire neurons/channels/heads):      |
|                                                                   |
|     Before:                After:                                 |
|     [Layer with 256      [Layer with 128                         |
|      neurons]              neurons]                               |
|                                                                   |
|     Remove entire rows/columns of the weight matrix.              |
|     Actually makes the model smaller and faster on any hardware.  |
|     But can lose more quality than unstructured pruning.          |
+-------------------------------------------------------------------+
```

### How Pruning Works

```
+-------------------------------------------------------------------+
|              The Pruning Process                                   |
|                                                                   |
|   Step 1: TRAIN the full model normally                           |
|                                                                   |
|   Step 2: SCORE each weight by importance                         |
|     Method 1 - Magnitude: |weight| (small = unimportant)        |
|     Method 2 - Gradient: How much does loss change if removed?   |
|     Method 3 - Movement: How much did it change during training? |
|                                                                   |
|   Step 3: REMOVE the least important weights                      |
|     Set them to zero (unstructured)                               |
|     Or delete entire neurons (structured)                         |
|                                                                   |
|   Step 4: FINE-TUNE the pruned model                              |
|     Train for a few more epochs to recover quality                |
|     The remaining weights adjust to compensate                    |
|                                                                   |
|   Optional: Repeat steps 2-4 (iterative pruning)                 |
|     Prune 10% at a time, fine-tune, repeat                        |
|     Gets better results than pruning 90% all at once              |
+-------------------------------------------------------------------+
```

### The Lottery Ticket Hypothesis

One of the most fascinating discoveries in neural networks (Frankle & Carlin, 2019):

> **Inside every large neural network, there exists a small subnetwork
> (a "winning lottery ticket") that, if trained from scratch with the
> same initial weights, achieves the same performance as the full network.**

This means most of the network's capacity is "wasted" -- and if we could
find these winning tickets upfront, we could train small models directly.

```
+-------------------------------------------------------------------+
|              The Lottery Ticket Hypothesis                         |
|                                                                   |
|   Full network: 100 million parameters                            |
|     Train, then prune to find the important 10%                   |
|                                                                   |
|   "Winning ticket": 10 million parameters                         |
|     Same initial weights, same performance!                       |
|                                                                   |
|   Like a lottery: most tickets (weights) are losers,              |
|   but a few are winners. The hard part is finding them.           |
+-------------------------------------------------------------------+
```

---

## 2. Knowledge Distillation: The Student-Teacher Method

Instead of shrinking the big model, train a **new smaller model** to copy it.
The big model is the "teacher" and the small model is the "student."

### How It Works

```
+-------------------------------------------------------------------+
|              Knowledge Distillation                                |
|                                                                   |
|   Teacher (big model):                                            |
|     Input: "The capital of France is ___"                         |
|     Output probabilities:                                         |
|       Paris: 0.85                                                 |
|       Lyon:  0.08    <-- "soft" knowledge!                       |
|       Nice:  0.04    <-- The teacher knows Lyon is more          |
|       Rome:  0.02        likely than Rome (both cities,          |
|       Dog:   0.01        but Lyon is in France)                  |
|                                                                   |
|   Student (small model):                                          |
|     Learns from BOTH:                                             |
|       1. The correct answer (Paris) -- "hard" label              |
|       2. The teacher's full probability distribution -- "soft"    |
|          label (this is the KEY insight!)                         |
|                                                                   |
|   The "soft" labels contain extra knowledge:                      |
|     - Which wrong answers are "almost right"                      |
|     - Relationships between categories                            |
|     - Uncertainty about ambiguous cases                           |
|                                                                   |
|   This is MORE information than just "the answer is Paris"       |
+-------------------------------------------------------------------+
```

### The Temperature Trick

The teacher's output probabilities are "softened" using a **temperature**
parameter. Higher temperature makes the distribution more uniform, revealing
more of the teacher's internal knowledge.

```
Temperature = 1 (normal):       Temperature = 5 (softened):
  Paris: 0.85                     Paris: 0.40
  Lyon:  0.08                     Lyon:  0.22
  Nice:  0.04                     Nice:  0.18
  Rome:  0.02                     Rome:  0.12
  Dog:   0.01                     Dog:   0.08

At higher temperature, the differences between "wrong" answers
become clearer -- Lyon vs Dog is now much more obvious.
```

### Famous Distilled Models

| Teacher | Student | Size Reduction | Quality Retained |
|---------|---------|----------------|-----------------|
| BERT-base | DistilBERT | 40% smaller, 60% faster | 97% of BERT's performance |
| GPT-4 (rumored) | Various "GPT-4 quality" small models | Varies | Varies |
| Llama-70B | Many fine-tuned 7B models | 10x smaller | ~85-90% quality |

---

## 3. Low-Rank Factorization

Neural network layers are essentially **matrix multiplications**. A large matrix
can sometimes be **approximated** by the product of two smaller matrices. This
is low-rank factorization.

### The Intuition

```
+-------------------------------------------------------------------+
|              Low-Rank Factorization                                |
|                                                                   |
|   Original weight matrix: 1000 x 1000 = 1,000,000 parameters    |
|                                                                   |
|   If the matrix has rank 10 (most information lives in 10        |
|   dimensions), we can approximate it as:                          |
|                                                                   |
|     A (1000 x 10) x B (10 x 1000) = 20,000 parameters           |
|                                                                   |
|     1,000,000 --> 20,000  (50x fewer parameters!)                |
|                                                                   |
|   Like compressing an image:                                      |
|     A 1000x1000 photo might look great at 100x100 resolution.   |
|     Most of the "information" was redundant.                      |
|                                                                   |
|   This is the same math behind:                                  |
|     - SVD (Singular Value Decomposition)                          |
|     - PCA (Principal Component Analysis)                          |
|     - LoRA (Low-Rank Adaptation for fine-tuning!)                |
+-------------------------------------------------------------------+
```

### LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2021) applies this idea to **fine-tuning**. Instead of
updating all the weights of a pretrained model, add small low-rank
matrices that capture the changes.

```
+-------------------------------------------------------------------+
|              LoRA: Efficient Fine-Tuning                           |
|                                                                   |
|   Original layer: W (d x d matrix, say 4096 x 4096)              |
|   Full fine-tuning: Update all 16M parameters in W               |
|                                                                   |
|   LoRA: Freeze W. Add two small matrices:                         |
|     A (4096 x 16) and B (16 x 4096)                              |
|     New output = W*x + A*B*x                                      |
|                                                                   |
|   Parameters:                                                     |
|     Full fine-tuning: 16,777,216 per layer                        |
|     LoRA (rank 16):   131,072 per layer (128x fewer!)            |
|                                                                   |
|   Quality:                                                        |
|     Often matches full fine-tuning quality!                       |
|     Works because most of the "change" during fine-tuning        |
|     lives in a low-rank space.                                    |
+-------------------------------------------------------------------+
```

---

## Comparison of Techniques

| Technique | Size Reduction | Quality Loss | Effort | Best For |
|-----------|---------------|-------------|--------|----------|
| **Quantization** | 2-8x | Very low | Easy | Always do this first |
| **Pruning (unstructured)** | 2-10x | Low-medium | Medium | Research, sparse hardware |
| **Pruning (structured)** | 2-4x | Medium | Medium | Actual speedup on any GPU |
| **Distillation** | 3-10x | Low-medium | High (needs training) | When you need a fundamentally smaller model |
| **Low-rank factorization** | 1.5-3x | Low | Medium | Fine-tuning (LoRA) |

```
+-------------------------------------------------------------------+
|              When to Use What                                      |
|                                                                   |
|   "I need my model smaller, right now"                            |
|     --> Quantization (minutes, no training needed)               |
|                                                                   |
|   "I need a much smaller model and have training budget"          |
|     --> Knowledge distillation (best quality for the size)       |
|                                                                   |
|   "I want to fine-tune a large model cheaply"                    |
|     --> LoRA (low-rank adaptation)                                |
|                                                                   |
|   "I want to understand what my model actually needs"            |
|     --> Pruning (reveals which parts matter)                     |
|                                                                   |
|   "I want maximum compression"                                   |
|     --> Combine: Distill + Quantize + Prune                      |
+-------------------------------------------------------------------+
```

---

## Combining Techniques

These techniques are not mutually exclusive! The best results come from
combining them:

```
Example: Making a chat model run on a phone

  Start: LLaMA-70B (140 GB, needs 4 GPUs)
    |
    v
  Step 1: Distill to 7B parameters
    Now: 14 GB (fits on 1 GPU)
    |
    v
  Step 2: Quantize to INT4
    Now: 3.5 GB (fits on a good laptop GPU)
    |
    v
  Step 3: Prune 30% of weights
    Now: ~2.5 GB (fits on a phone!)
    |
    v
  Result: From 140 GB to 2.5 GB (56x smaller)
  Quality: Still surprisingly good for most tasks!
```

---

## Summary

```
+------------------------------------------------------------------+
|           Model Compression Cheat Sheet                           |
|                                                                  |
|  Quantization:   Use fewer bits. Easiest. Do first.             |
|  Pruning:        Remove unneeded weights. Structured = actual   |
|                  speedup. Lottery ticket = fascinating theory.   |
|  Distillation:   Train small model from big one. Best quality   |
|                  for the size, but expensive to do.             |
|  Low-Rank:       Approximate large matrices with smaller ones.  |
|                  LoRA makes fine-tuning 100x cheaper.           |
|                                                                  |
|  The combination: Distill + Quantize + Prune                    |
|  can achieve 50-100x compression with acceptable quality.       |
+------------------------------------------------------------------+
```

---

## Further Reading

- **The Lottery Ticket Hypothesis** -- Frankle & Carlin, 2019
  - The surprising discovery that small subnetworks inside large networks work just as well
- **DistilBERT: A Distilled Version of BERT** -- Sanh et al., 2019
  - The most famous knowledge distillation success story
- **LoRA: Low-Rank Adaptation of Large Language Models** -- Hu et al., 2021
  - Made fine-tuning large models affordable for everyone
- **A Survey of Model Compression and Acceleration for Deep Neural Networks** -- Cheng et al., 2017

---

[Back to Inference Optimization](./README.md) | [Back to Deployment](../README.md)
