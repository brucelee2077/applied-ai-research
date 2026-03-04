# Training & Experiments

## The Mystery Worth Solving

You built every piece of a transformer from scratch in the architecture module — attention, multi-head attention, positional encoding, the full block. You know *how* the parts work.

But here's a question that might be nagging you: **does it actually learn anything?**

All those weight matrices started as random numbers. Does the model figure out real patterns on its own, just from training data? And if you give the same task to an encoder and a decoder, does it matter which one you use?

This section answers both questions. You will train a small transformer on real text and watch it go from random garbage to readable output. Then you will build two different transformer types and see — with your own eyes — why architecture choice matters.

---

**Before you start, you need to know:**
- How a transformer block works (attention + FFN + residuals + layer norm) — covered in [Transformer Block](../architecture/transformer-block.md)
- Basic PyTorch syntax (tensors, `nn.Module`, `.backward()`) — if you completed the RNN module, you're ready

---

## What You'll Build

### Training a Small Transformer

You will build a tiny decoder-only transformer (the same kind of architecture behind GPT and Claude) and train it to generate text one character at a time. The model starts by producing random characters. After a few minutes of training, it produces text that looks surprisingly like the training data.

This is the same task you did with an LSTM in the RNN module — same input, same goal. The difference is the architecture. You'll see how a transformer handles the same problem.

### Encoder vs Decoder Comparison

You will build two classifiers — one encoder-only (like BERT) and one decoder-only (like GPT) — and train both on the same sentiment task. Same data, same size, different attention patterns. You'll see which one performs better and *why*, by looking at the attention maps side by side.

---

## Coverage Map

### Training & Experiments

| Topic | Depth | Files |
|-------|-------|-------|
| Training a Small Transformer — char-level decoder-only model, training loop, text generation | [Applied] | [theory](./training-a-small-transformer.md) · [notebook](./01_training_a_small_transformer.ipynb) |
| Encoder vs Decoder Comparison — same task, different architectures, attention pattern analysis | [Applied] | [theory](./encoder-vs-decoder.md) · [notebook](./02_encoder_vs_decoder.ipynb) |
| Model Scaling — how transformers grow, scaling laws, compute-optimal training | [Awareness] | [README.md#model-scaling](#model-scaling-awareness) |

---

## Study Plan

```
START HERE (after completing the architecture section)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Training a Small Transformer                                 │
│     → Read: training-a-small-transformer.md                      │
│     → Code: 01_training_a_small_transformer.ipynb                │
│     → You'll train a real model and generate text!               │
│                                                                  │
│  2. Encoder vs Decoder Comparison                                │
│     → Read: encoder-vs-decoder.md                                │
│     → Code: 02_encoder_vs_decoder.ipynb                          │
│     → Same task, two architectures — which wins and why?         │
│                                                                  │
│  3. Model Scaling (read below)                                   │
│     → Quick overview of how transformers grow from small to huge │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Scaling {#model-scaling-awareness}

### How Big Can Transformers Get?

The tiny model you train in this module has about 50,000–100,000 parameters. GPT-3 has 175 billion. That's roughly a million times larger. How does that happen, and what changes when you scale up?

### Where Parameters Live

A transformer's parameter count comes from three places:

| Component | Formula (approximate) | Example (d_model=768, 12 layers) |
|-----------|----------------------|----------------------------------|
| Embeddings | vocab_size × d_model | 50,257 × 768 ≈ 39M |
| Attention (per layer) | 4 × d_model² | 4 × 768² ≈ 2.4M |
| FFN (per layer) | 8 × d_model² | 8 × 768² ≈ 4.7M |
| Total per layer | ~12 × d_model² | ~7.1M |
| Full model | embeddings + layers × per-layer | 39M + 12 × 7.1M ≈ 124M |

To make a model bigger, you increase d_model (wider vectors), add more layers (deeper), or both. Most of the growth comes from d_model — doubling it roughly quadruples the parameter count.

### Scaling Laws: More Data + More Compute = Better Performance

In 2020, researchers at OpenAI (Kaplan et al.) discovered something surprising: if you plot a model's performance against its size, training data, or compute budget on a log-log graph, you get a nearly straight line. This means performance improves *predictably* as you scale up.

Key findings from the Kaplan scaling laws:
- **Bigger models are more sample-efficient.** A 10× larger model needs fewer training examples to reach the same performance.
- **Performance follows power laws.** Loss decreases as a power of model size, dataset size, and compute.
- **There's no sign of diminishing returns** (at the scales tested). Bigger models keep getting better.

### Chinchilla: The Compute-Optimal Rule

In 2022, DeepMind's Chinchilla paper (Hoffmann et al.) refined the scaling laws with a practical rule of thumb:

**For compute-optimal training, use roughly 20 tokens of training data per parameter.**

| Model | Parameters | Optimal training tokens | Actual training tokens |
|-------|-----------|------------------------|----------------------|
| GPT-3 | 175B | ~3.5T (optimal) | 300B (undertrained) |
| Chinchilla | 70B | ~1.4T | 1.4T (compute-optimal) |
| LLaMA 2 | 70B | ~1.4T | 2T (overtrained for inference) |

This changed how the field thinks about scaling. GPT-3 was probably too large for its training budget — a smaller model trained on more data (Chinchilla) matched its performance at a fraction of the size.

More recent models like LLaMA intentionally "overtrain" — they train on more tokens than compute-optimal, because a smaller model that is trained longer is cheaper to *deploy* (even if training costs more). The Chinchilla ratio is optimal for training cost, but deployment cost depends on model size, not training data.

### Emergent Abilities

As models grow, some abilities seem to appear suddenly rather than gradually:

- **Few-shot learning:** Small models cannot learn from a few examples in the prompt. Large models (roughly 10B+ parameters) start doing this reliably.
- **Chain-of-thought reasoning:** Asking a small model to "think step by step" doesn't help. For large models, it dramatically improves accuracy on math and logic problems.
- **Instruction following:** Small models struggle to follow natural language instructions. Large models do this well after instruction tuning.

Whether these are truly "emergent" (appearing suddenly) or just gradual improvements that cross a usefulness threshold is still debated. But the practical result is clear: some capabilities only become useful above a certain model size.

### Further Reading

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al.) | 2020 | Power-law scaling of loss with compute, data, and parameters |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al.) | 2022 | Chinchilla scaling — ~20 tokens per parameter |
| [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) (Wei et al.) | 2022 | Documenting abilities that appear at scale |

---

[Back to Transformers Overview](../README.md) | [Next: Training a Small Transformer →](./training-a-small-transformer.md)
