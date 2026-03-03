# Transformer Architecture

## The Mystery Worth Solving

In 2017, a small team at Google published a 15-page paper and changed the course of AI.

The title was deliberately humble: "Attention Is All You Need." They were proposing to throw out a decade of recurrent neural network research and replace everything with one simple mechanism.

It worked. Within three years, every state-of-the-art language model used their architecture. If you've talked to ChatGPT, Claude, or Gemini — you've been talking to a descendant of that 2017 paper.

What did they figure out? This module takes you through every piece, from the ground up.

---

## What This Module Covers

This is the architecture module of the transformers section. It covers the four [Core] topics that make up the transformer: attention, multi-head attention, positional encoding, and the transformer block. Each topic has a beginner file, an interview deep-dive, a concept notebook, and an experiments notebook.

By the end, you'll have built a complete transformer block from scratch in NumPy and understood every component at Staff/Principal interview depth.

---

## Coverage Map

### Architecture Components

| Topic | Depth | Files |
|-------|-------|-------|
| Attention Mechanisms — Q, K, V, dot product, softmax, self vs cross attention | [Core] | [attention-mechanisms.md](./attention-mechanisms.md) · [interview](./attention-mechanisms-interview.md) · [concept notebook](./01_attention_mechanisms.ipynb) · [experiments](./01_attention_mechanisms_experiments.ipynb) |
| Multi-Head Attention — parallel heads, concatenation, head specialization | [Core] | [multi-head-attention.md](./multi-head-attention.md) · [interview](./multi-head-attention-interview.md) · [concept notebook](./02_multi_head_attention.ipynb) · [experiments](./02_multi_head_attention_experiments.ipynb) |
| Positional Encoding — sinusoidal, learned, RoPE, ALiBi | [Core] | [positional-encoding.md](./positional-encoding.md) · [interview](./positional-encoding-interview.md) · [concept notebook](./03_positional_encoding.ipynb) · [experiments](./03_positional_encoding_experiments.ipynb) |
| Transformer Block — FFN, residuals, LayerNorm, Pre-LN vs Post-LN, full assembly, three transformer types | [Core] | [transformer-block.md](./transformer-block.md) · [interview](./transformer-block-interview.md) · [concept notebook](./04_transformer_block.ipynb) · [experiments](./04_transformer_block_experiments.ipynb) |

---

## Study Order

Read the theory guide first, then work through the concept notebook. After both, try the experiments notebook if you want interview-ready evidence for your claims.

| Order | Theory (Read) | Hands-on (Code) | What You'll Learn |
|-------|---------------|------------------|-------------------|
| 1 | [Attention Mechanisms](./attention-mechanisms.md) | [Concept](./01_attention_mechanisms.ipynb) · [Experiments](./01_attention_mechanisms_experiments.ipynb) | Q, K, V, dot product, softmax |
| 2 | [Multi-Head Attention](./multi-head-attention.md) | [Concept](./02_multi_head_attention.ipynb) · [Experiments](./02_multi_head_attention_experiments.ipynb) | Parallel heads, specialization |
| 3 | [Positional Encoding](./positional-encoding.md) | [Concept](./03_positional_encoding.ipynb) · [Experiments](./03_positional_encoding_experiments.ipynb) | How word order is encoded |
| 4 | [Transformer Block](./transformer-block.md) | [Concept](./04_transformer_block.ipynb) · [Experiments](./04_transformer_block_experiments.ipynb) | LayerNorm, residuals, FFN, full block |

After these four, you'll understand all the core building blocks and have built a complete transformer from scratch in NumPy.

---

## Hyperparameters Quick Reference

These are the knobs you can turn when designing a transformer. No equations here — for the mathematical reasoning behind these choices, see the [Transformer Block Interview Deep-Dive](./transformer-block-interview.md).

```
Parameter          What It Controls              Typical Values
─────────────     ──────────────────────         ──────────────
d_model            Size of word vectors           512, 768, 1024, 4096
N (num layers)     Depth of the network           6, 12, 24, 32, 96
h (num heads)      Parallel attention mechanisms   8, 12, 16, 96
d_ff               FFN inner dimension             2048, 3072, 4096, 16384
d_k                Key/query dimension per head    64, 128
dropout            Regularization rate              0.1
max_seq_len        Maximum input length             512, 2048, 4096, 128K
```

Model sizes for reference:

```
Model          Params    Layers    d_model    Heads
───────────    ───────   ──────    ───────    ─────
GPT-2 Small    117M      12        768        12
BERT Base      110M      12        768        12
GPT-2 Large    774M      36        1280       20
GPT-3          175B      96        12288      96
LLaMA 7B       7B        32        4096       32
LLaMA 70B      70B       80        8192       64
```

---

## Key Takeaways

1. **Transformers process all words simultaneously** using attention (not one at a time like RNNs)
2. **The transformer block** = multi-head attention + feed-forward network + residual connections + layer norm
3. **Multiple blocks stack** to create deep understanding (6 to 96+ layers)
4. **Three flavors:** encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5)
5. **Attention** is the key innovation — it lets every word consider every other word
6. **Residual connections** and **layer normalization** make training deep networks stable

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — visual walkthrough
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) — line-by-line code walkthrough

---

[Back to Transformers Module](../README.md)
