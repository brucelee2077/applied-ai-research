# Transformers

## What Are Transformers? (The 30-Second Version)

Imagine you're trying to understand a long sentence. Older AI models read words one at a time, left to right, like reading through a tiny keyhole -- by the time they reached the end, they'd forgotten the beginning. **Transformers** changed everything by letting the model see **all words at once** and figure out which words are important for understanding each other.

This single idea -- called **attention** -- is the foundation of ChatGPT, Claude, Google Translate, GitHub Copilot, and virtually every modern AI system that works with language.

### Prerequisites

Before starting this module, you should have completed:
- [Neural Network Fundamentals](../00-neural-networks/fundamentals/) -- especially notebooks 01-08
- Basic Python and NumPy knowledge

If terms like "weights", "loss function", or "backpropagation" are unfamiliar, go through `00-neural-networks` first.

---

## Study Plan

Follow this path from top to bottom. Each section builds on the previous one.

```
START HERE
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Understand the Architecture (Theory + Code)           │
│                                                                  │
│  1. Architecture Overview               (architecture/README)    │
│     → What is a transformer? The big picture.                   │
│     → All the building blocks explained simply.                 │
│                                                                  │
│  2. Attention Mechanisms                                         │
│     → Read: attention-mechanisms.md (theory + diagrams)          │
│     → Code: 01_attention_mechanisms.ipynb (build from scratch)   │
│                                                                  │
│  3. Multi-Head Attention                                         │
│     → Read: multi-head-attention.md (theory + diagrams)          │
│     → Code: 02_multi_head_attention.ipynb (build from scratch)   │
│                                                                  │
│  4. Positional Encoding                                          │
│     → Read: positional-encoding.md (theory + diagrams)           │
│     → Code: 03_positional_encoding.ipynb (build + visualize)     │
│                                                                  │
│  5. Complete Transformer Block                                   │
│     → Code: 04_transformer_block.ipynb (assemble everything!)    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Train & Experiment                        [Coming Soon]│
│                                                                  │
│  6. Train a small transformer on a real task                     │
│  7. Compare encoder vs. decoder architectures                    │
│  8. Explore model scaling (small → large)                        │
└─────────────────────────────────────────────────────────────────┘
```

### Recommended Reading Order

For each topic, read the `.md` file first (theory), then work through the `.ipynb` notebook (code):

| Step | Theory (Read) | Code (Hands-on) | What You'll Learn |
|------|---------------|------------------|-------------------|
| 1 | [Architecture Overview](./architecture/README.md) | — | The complete transformer, all components, three model types |
| 2 | [Attention Mechanisms](./architecture/attention-mechanisms.md) | [Notebook](./architecture/01_attention_mechanisms.ipynb) | Q, K, V, dot product, softmax, self vs cross attention |
| 3 | [Multi-Head Attention](./architecture/multi-head-attention.md) | [Notebook](./architecture/02_multi_head_attention.ipynb) | Parallel heads, concatenation, head specialization |
| 4 | [Positional Encoding](./architecture/positional-encoding.md) | [Notebook](./architecture/03_positional_encoding.ipynb) | Why order matters, sinusoidal vs learned encodings |
| 5 | (covered in Architecture Overview) | [Notebook](./architecture/04_transformer_block.ipynb) | Layer norm, residual connections, FFN, full block |

---

## Directory Structure

```
01-transformers/
├── README.md                                ← You are here (study plan)
├── architecture/                            ← Theory & hands-on code
│   ├── README.md                            ← Full architecture overview
│   ├── attention-mechanisms.md              ← Attention theory + diagrams
│   ├── 01_attention_mechanisms.ipynb        ← Build attention from scratch
│   ├── multi-head-attention.md              ← Multi-head theory + diagrams
│   ├── 02_multi_head_attention.ipynb        ← Build multi-head from scratch
│   ├── positional-encoding.md               ← Position encoding theory
│   ├── 03_positional_encoding.ipynb         ← Build + visualize encodings
│   └── 04_transformer_block.ipynb           ← Complete transformer block
├── implementations/                         ← Full implementations (coming soon)
│   └── .gitkeep
└── experiments/                             ← Experiments (coming soon)
    └── .gitkeep
```

---

## Key Concepts Glossary

New to ML? Here's a quick reference for terms you'll encounter:

| Term | Simple Explanation |
|------|-------------------|
| **Attention** | A way for each word to "look at" every other word and decide what's relevant |
| **Embedding** | Converting a word into a list of numbers that captures its meaning |
| **Token** | A piece of text (word or sub-word) that the model processes |
| **Vector** | A list of numbers, like [0.2, -0.5, 0.8] |
| **Matrix** | A grid of numbers (like a spreadsheet) |
| **Dot product** | A way to measure how similar two vectors are |
| **Softmax** | Converts any numbers into probabilities that sum to 1 |
| **Query (Q)** | What a word is "looking for" -- its question |
| **Key (K)** | What a word "advertises" about itself -- its label |
| **Value (V)** | The actual information a word carries -- its content |
| **Encoder** | Processes input and creates understanding (bidirectional) |
| **Decoder** | Generates output one word at a time (left-to-right) |
| **Layer norm** | Keeps numbers in a stable range during processing |
| **Residual connection** | A shortcut that adds the input back to the output to prevent information loss |
| **FFN** | Feed-Forward Network -- a small neural net for "private thinking" per word |
| **d_model** | The size of word vectors (e.g., 512 or 768 numbers) |

---

## Key Papers

| Paper | Year | Why It Matters |
|-------|------|----------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Introduced the transformer architecture |
| [BERT](https://arxiv.org/abs/1810.04805) | 2018 | Showed encoder-only transformers are great for understanding |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | Showed decoder-only transformers can generate coherent text |
| [GPT-3](https://arxiv.org/abs/2005.14165) | 2020 | Showed scaling transformers to 175B parameters gives emergent abilities |
| [T5](https://arxiv.org/abs/1910.10683) | 2019 | Unified NLP tasks as text-to-text using encoder-decoder |

## Further Reading

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) -- excellent visual guide
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) -- comprehensive overview
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) -- code walkthrough

---

[Back to Main](../README.md) | [Previous: Neural Networks](../00-neural-networks/README.md) | [Next: Fine-Tuning](../02-fine-tuning/README.md)
