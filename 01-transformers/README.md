# Transformers

## The Mystery Worth Solving

Here's something that genuinely puzzled researchers when it first happened:

A model trained on nothing but text — no images, no game boards, no medical scans — somehow learned to write working code, solve math problems, and beat humans at trivia across dozens of domains. Not because researchers programmed those skills in. The model discovered them on its own, just from reading enough sentences.

How does that happen?

The answer turns out to be one surprisingly simple idea. It has a boring name: **attention**. But it changed everything about how we build AI. Every large language model you've heard of — ChatGPT, Claude, Gemini, Llama, Grok — runs on this idea.

By the end of this module, you'll understand exactly how it works. Not just "attention helps words look at each other" — but *why* that matters, *how* it's computed, and *what breaks* when you scale it up.

Let's go.

---

## The 30-Second Version

Think about how you read this sentence:

> "The cat sat on the mat because **it** was tired."

When you hit the word "it", you don't panic and say "I don't know what 'it' refers to." Your brain automatically *looks back* at the whole sentence and picks out "the cat." You didn't read the sentence twice — you just know how to scan back and find the relevant piece.

Older AI models couldn't do this. They read words one at a time, left to right, like reading through a keyhole. By the time they reached "tired," the beginning of the sentence had faded from memory. **Transformers** fixed this by letting every word look at every other word at once.

**What this analogy gets right:** Your brain's ability to look back and find "the cat" is the exact mechanism of attention — each word asks "which other words help me understand my own meaning?" and then collects information from those relevant words.

**Where this analogy breaks down:** Your brain reads words one at a time and does this backward scan implicitly. A transformer processes all words simultaneously — every word looks at every other word at the same time, in parallel. There's no "reading through" the sentence at all.

You just understood the core mechanism of every major AI system released in the last seven years. GPT-4, Claude, Gemini, GitHub Copilot — they all run on this one idea: let every word look at every other word. The rest of this module unpacks what "look at" actually means, and builds it from scratch.

---

## Coverage Map

### Architecture

| Topic | Depth | Files |
|-------|-------|-------|
| Attention Mechanisms — Q, K, V, dot product, softmax, self vs cross attention | [Core] | [theory](./architecture/attention-mechanisms.md) · [interview](./architecture/attention-mechanisms-interview.md) · [concept notebook](./architecture/01_attention_mechanisms.ipynb) · [experiments](./architecture/01_attention_mechanisms_experiments.ipynb) |
| Multi-Head Attention — parallel heads, concatenation, head specialization | [Core] | [theory](./architecture/multi-head-attention.md) · [interview](./architecture/multi-head-attention-interview.md) · [concept notebook](./architecture/02_multi_head_attention.ipynb) · [experiments](./architecture/02_multi_head_attention_experiments.ipynb) |
| Positional Encoding — sinusoidal, learned, RoPE, ALiBi | [Core] | [theory](./architecture/positional-encoding.md) · [interview](./architecture/positional-encoding-interview.md) · [concept notebook](./architecture/03_positional_encoding.ipynb) · [experiments](./architecture/03_positional_encoding_experiments.ipynb) |
| Transformer Block — FFN, residuals, LayerNorm, Pre-LN vs Post-LN, encoder/decoder types | [Core] | [theory](./architecture/transformer-block.md) · [interview](./architecture/transformer-block-interview.md) · [concept notebook](./architecture/04_transformer_block.ipynb) · [experiments](./architecture/04_transformer_block_experiments.ipynb) |

### Training & Experiments [Coming Soon]

| Topic | Depth | Files |
|-------|-------|-------|
| Training a small transformer | [Applied] | Coming soon |
| Encoder vs Decoder comparison | [Applied] | Coming soon |
| Model scaling experiments | [Awareness] | Coming soon |

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
│                                                                  │
│  2. Attention Mechanisms                                         │
│     → Read: attention-mechanisms.md (theory)                     │
│     → Code: 01_attention_mechanisms.ipynb (build from scratch)   │
│                                                                  │
│  3. Multi-Head Attention                                         │
│     → Read: multi-head-attention.md (theory)                     │
│     → Code: 02_multi_head_attention.ipynb (build from scratch)   │
│                                                                  │
│  4. Positional Encoding                                          │
│     → Read: positional-encoding.md (theory)                      │
│     → Code: 03_positional_encoding.ipynb (build + visualize)     │
│                                                                  │
│  5. Transformer Block                                            │
│     → Read: transformer-block.md (theory)                        │
│     → Code: 04_transformer_block.ipynb (assemble everything!)    │
│                                                                  │
│  Ready for interviews? Read the -interview.md files and run      │
│  the _experiments.ipynb notebooks for each topic.                │
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
| 2 | [Attention Mechanisms](./architecture/attention-mechanisms.md) | [Concept](./architecture/01_attention_mechanisms.ipynb) | Q, K, V, dot product, softmax, self vs cross attention |
| 3 | [Multi-Head Attention](./architecture/multi-head-attention.md) | [Concept](./architecture/02_multi_head_attention.ipynb) | Parallel heads, concatenation, head specialization |
| 4 | [Positional Encoding](./architecture/positional-encoding.md) | [Concept](./architecture/03_positional_encoding.ipynb) | Why order matters, sinusoidal vs learned encodings |
| 5 | [Transformer Block](./architecture/transformer-block.md) | [Concept](./architecture/04_transformer_block.ipynb) | Layer norm, residual connections, FFN, full block |

---

## Directory Structure

```
01-transformers/
├── README.md                                    ← You are here (study plan)
├── PROGRESS.md                                  ← Session tracking
├── architecture/                                ← Theory & hands-on code
│   ├── README.md                                ← Architecture overview + coverage map
│   ├── attention-mechanisms.md                  ← Attention theory (Layer 1)
│   ├── attention-mechanisms-interview.md        ← Attention interview depth (Layer 2)
│   ├── 01_attention_mechanisms.ipynb            ← Build attention from scratch
│   ├── 01_attention_mechanisms_experiments.ipynb ← Benchmark + ablate attention
│   ├── multi-head-attention.md                  ← Multi-head theory (Layer 1)
│   ├── multi-head-attention-interview.md        ← Multi-head interview depth (Layer 2)
│   ├── 02_multi_head_attention.ipynb            ← Build multi-head from scratch
│   ├── 02_multi_head_attention_experiments.ipynb ← Benchmark + ablate multi-head
│   ├── positional-encoding.md                   ← Position encoding theory (Layer 1)
│   ├── positional-encoding-interview.md         ← Position encoding interview depth (Layer 2)
│   ├── 03_positional_encoding.ipynb             ← Build + visualize encodings
│   ├── 03_positional_encoding_experiments.ipynb  ← Compare PE methods
│   ├── transformer-block.md                     ← Transformer block theory (Layer 1)
│   ├── transformer-block-interview.md           ← Transformer block interview depth (Layer 2)
│   ├── 04_transformer_block.ipynb               ← Complete transformer block
│   └── 04_transformer_block_experiments.ipynb    ← Pre-LN vs Post-LN + ablations
├── implementations/                             ← Full implementations (coming soon)
│   └── .gitkeep
└── experiments/                                 ← Experiments (coming soon)
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
| **Softmax** | Converts any set of numbers into probabilities that sum to 1 |
| **Query (Q)** | What a word is "looking for" — its question |
| **Key (K)** | What a word "advertises" about itself — its label |
| **Value (V)** | The actual information a word carries — its content |
| **Encoder** | Processes input and creates understanding (bidirectional) |
| **Decoder** | Generates output one word at a time (left-to-right) |
| **Layer norm** | Keeps numbers in a stable range during processing |
| **Residual connection** | A shortcut that adds the input back to the output to prevent information loss |
| **FFN** | Feed-Forward Network — a small neural net for "private thinking" per word |
| **d_model** | The size of word vectors (e.g., 512 or 768 numbers) |

---

## Common Confusion Points

These are the misconceptions that trip up nearly everyone when first learning transformers. You're not alone — even experienced ML engineers sometimes get these wrong when they first dig in.

### 1. "Q, K, and V are three different inputs"

**Not quite.** In self-attention, Q, K, and V all come from the **same input sequence**. The word "The" produces Q_The AND K_The AND V_The — all from the same embedding vector, just multiplied by three different learned weight matrices.

The matrices W_Q, W_K, and W_V are what differ — not the input. They project the same embedding into different "question", "label", and "content" spaces.

**Exception:** Cross-attention (in encoder-decoder models) does use two different inputs — Q from the decoder, K and V from the encoder. But standard self-attention always uses one.

### 2. "Multi-head attention is h times more expensive"

**Not quite.** Multi-head attention with h heads costs roughly the **same** as single-head attention with the same d_model.

Each head works with d_k = d_model / h dimensions. The total cost for h heads is h * d_k^2 = d_model^2 / h. The output projection W_O adds d_model^2, bringing the total back to roughly the same. The trick: splitting d_model across heads makes each head cheaper, and h heads * (1/h cost each) = same total.

### 3. "Positional encoding is concatenated to word embeddings"

**Not quite.** Positional encoding is **added** to word embeddings, not concatenated. Addition keeps the vector dimension the same (d_model). Concatenation would double the size and require larger downstream weight matrices.

### 4. "Attention means the model is paying special attention to important things"

**Partially right, but misleading.** At a technical level, "attention" just means **weighted average**. The output for each word is a weighted sum of Value vectors. The "attention" metaphor is useful but don't let it mislead you — the model learns which weights to assign, and those learned weights happen to pick out meaningful relationships.

---

## Key Papers

| Paper | Year | Why It Matters |
|-------|------|----------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Introduced the transformer architecture |
| [BERT](https://arxiv.org/abs/1810.04805) | 2018 | Showed encoder-only transformers are great for understanding |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | Showed decoder-only transformers can generate coherent text |
| [GPT-3](https://arxiv.org/abs/2005.14165) | 2020 | Showed scaling transformers to 175B parameters gives emergent abilities |
| [T5](https://arxiv.org/abs/1910.10683) | 2019 | Unified NLP tasks as text-to-text using encoder-decoder |
| [Flash Attention](https://arxiv.org/abs/2205.14135) | 2022 | Made long-context attention practical via I/O optimization |

---

## Further Reading

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — excellent visual guide
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — comprehensive overview
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) — code walkthrough

---

[Back to Main](../README.md) | [Next: Architecture Overview →](./architecture/README.md)
