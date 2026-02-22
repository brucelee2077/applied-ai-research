# Transformer Architecture

## The Big Picture: What Is a Transformer?

A transformer is a type of neural network designed to process **sequences** (like sentences, code, or music). It was introduced in 2017 in the paper "Attention Is All You Need" and has become the backbone of virtually all modern AI language models (GPT, BERT, T5, LLaMA, etc.).

Before transformers, the best models for language were RNNs (Recurrent Neural Networks), which read words one at a time like reading a book left to right. Transformers broke this pattern by processing **all words simultaneously** using a mechanism called **attention** -- and the results were dramatically better.

### The Factory Analogy

Think of a transformer as a **word factory** with a series of processing stations:

```
Raw materials (words)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                     THE TRANSFORMER FACTORY                   │
│                                                               │
│  Station 1: INTAKE                                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Turn words into numbers (embeddings)                    │  │
│  │ Stamp each word with its position (positional encoding) │  │
│  └─────────────────────────────────────────────────────────┘  │
│         │                                                     │
│         ▼                                                     │
│  Station 2: PROCESSING (repeated N times)                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ a) Let words talk to each other (multi-head attention)  │  │
│  │ b) Think about what they learned (feed-forward network) │  │
│  │ c) Quality control at each step (layer norm + residual) │  │
│  └─────────────────────────────────────────────────────────┘  │
│         │                                                     │
│         ▼                                                     │
│  Station 3: OUTPUT                                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Convert processed numbers back into useful predictions  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
Finished product (predictions)
```

---

## The Components (Building Blocks)

### 1. Input Embeddings: Words to Numbers

Neural networks can only work with numbers, not words. So the first step is converting each word (or "token") into a vector -- a list of numbers that captures its meaning.

```
Vocabulary:  { "the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4, ... }

"the"  →  token ID 0  →  look up in embedding table  →  [0.2, -0.1, 0.5, ...]
"cat"  →  token ID 1  →  look up in embedding table  →  [0.8, 0.3, -0.2, ...]
"sat"  →  token ID 2  →  look up in embedding table  →  [0.1, 0.7, 0.4, ...]

Each word becomes a vector of d_model numbers (typically 512 or 768).
These vectors are LEARNED during training -- similar words end up
with similar vectors.
```

For details on how position information is added, see [Positional Encoding](./positional-encoding.md).

### 2. Multi-Head Attention: Words Talking to Each Other

This is the core innovation of transformers. Each word gets to "look at" every other word and decide what's relevant. Multiple attention "heads" run in parallel, each learning to focus on different things (grammar, meaning, position, etc.).

For the full explanation, see [Attention Mechanisms](./attention-mechanisms.md) and [Multi-Head Attention](./multi-head-attention.md).

```
Input: "The cat sat on the mat"

After attention, each word carries information from other words:

"sat" now knows:
  - WHO sat → "cat" (from one attention head)
  - WHERE → "on the mat" (from another head)
  - WHICH cat → "the" cat (from another head)
```

### 3. Feed-Forward Network (FFN): Thinking Time

After words have talked to each other (attention), each word gets "thinking time" through a small neural network. This is applied to each word **independently** -- no interaction between words here.

```
                    ┌─────────────────────────────────┐
word vector  ──→    │  Linear(d_model → d_ff)         │  ← expand (e.g., 512 → 2048)
(512 dims)         │  Activation (ReLU or GELU)       │  ← add non-linearity
                    │  Linear(d_ff → d_model)          │  ← compress back (2048 → 512)
                    └─────────────────────────────────┘
                                    │
                           word vector out (512 dims)
```

**Why is this needed?** Attention lets words share information, but FFN lets each word **process and transform** that information. Think of it as:
- Attention = **gathering information** from a group discussion
- FFN = **thinking privately** about what you heard

The FFN typically expands the vector to 4x its size (e.g., 512 → 2048), applies a non-linear activation function, then compresses it back. This expansion gives the model more "space to think."

### 4. Residual Connections: The Safety Net

A **residual connection** (also called a "skip connection") adds the input of a layer directly to its output:

```
                    ┌──────────────┐
        x ─────┬──→│  Sub-layer   │──→ output
               │   │(attention or │
               │   │    FFN)      │
               │   └──────────────┘
               │          │
               │          ▼
               └────→ x + output  ──→  (to next layer)
                     ▲
                     │
                This is the residual connection!
                It adds the original input back to the output.
```

**Why?** Imagine you're following cooking instructions, but each step slightly changes the recipe. If something goes wrong at step 5, you've lost the original recipe. A residual connection is like **keeping a copy of the recipe at each step** -- even if one step doesn't help much, you don't lose the original information.

In technical terms:
- They prevent the **vanishing gradient problem** (gradients shrinking to zero in deep networks)
- They allow the model to learn **incremental changes** rather than complete transformations
- They make very deep networks (dozens of layers) possible to train

### 5. Layer Normalization: Quality Control

**Layer normalization** adjusts the numbers in each vector so they have a consistent scale (mean ≈ 0, standard deviation ≈ 1):

```
Before LayerNorm:  [150.0, -200.0, 3.0, 50.0]     ← numbers are all over the place
After LayerNorm:   [0.5, -1.3, -0.4, 0.2]          ← numbers are normalized

Think of it like grading on a curve:
  - Raw test scores: [95, 30, 72, 88]  ← hard to compare
  - Curved scores:   [A, D, B, A-]     ← standardized scale
```

**Why?** Without normalization, numbers in the network can grow very large or very small as they pass through many layers. This makes training unstable -- like trying to balance a pencil on its tip. Layer normalization keeps everything in a reasonable range.

There are two common placements:

```
Post-LayerNorm (original paper):        Pre-LayerNorm (most modern models):
┌────────┐                               ┌────────────┐
│ Attention│                              │ LayerNorm  │
└────┬────┘                               └────┬───────┘
     │                                         │
  + residual                              ┌────┴────┐
     │                                    │Attention │
┌────┴───────┐                            └────┬────┘
│ LayerNorm  │                                 │
└────────────┘                              + residual

Pre-LayerNorm is more stable and is used by GPT, LLaMA, etc.
```

---

## The Transformer Block (One Layer)

All these components combine into a single **transformer block** (also called a "layer"). This block is the repeating unit of the transformer:

```
┌─────────────────────────────────────────────────────────────┐
│                   TRANSFORMER BLOCK                          │
│                                                              │
│  Input (word vectors)                                        │
│        │                                                     │
│        ├──────────────────────────────┐                      │
│        ▼                              │                      │
│  ┌──────────────┐                     │                      │
│  │  Layer Norm   │                    │  (residual           │
│  └──────┬───────┘                     │   connection)        │
│         │                             │                      │
│  ┌──────┴───────┐                     │                      │
│  │  Multi-Head   │                    │                      │
│  │  Attention    │                    │                      │
│  └──────┬───────┘                     │                      │
│         │                             │                      │
│         ◄─────────────────────────────┘                      │
│         │  (add input back = residual)                       │
│         │                                                    │
│         ├──────────────────────────────┐                     │
│         ▼                              │                     │
│  ┌──────────────┐                     │                      │
│  │  Layer Norm   │                    │  (residual           │
│  └──────┬───────┘                     │   connection)        │
│         │                             │                      │
│  ┌──────┴───────┐                     │                      │
│  │  Feed-Forward │                    │                      │
│  │  Network      │                    │                      │
│  └──────┬───────┘                     │                      │
│         │                             │                      │
│         ◄─────────────────────────────┘                      │
│         │  (add input back = residual)                       │
│         │                                                    │
│  Output (refined word vectors)                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘

A typical transformer stacks 6-96 of these blocks!
```

Each time word vectors pass through a block, they get **refined** -- carrying more context and understanding. Early layers tend to capture simple patterns (grammar, nearby words), while later layers capture complex patterns (meaning, relationships, reasoning).

---

## Three Types of Transformers

The original transformer had both an **encoder** and a **decoder**. Modern models typically use one or the other (or both), depending on the task.

### Encoder-Only: Understanding Text

The **encoder** processes the full input and produces rich, context-aware representations. Every word can attend to every other word (bidirectional).

```
Input: "The cat sat on the mat"
        │
        ▼
┌─────────────────────────────┐
│         ENCODER              │
│                              │
│  ┌────────────────────────┐  │
│  │  Transformer Block × N │  │     All words can see all
│  │  (self-attention +     │  │     other words (bidirectional)
│  │   feed-forward)        │  │
│  └────────────────────────┘  │
│                              │
└─────────────┬───────────────┘
              │
              ▼
  Rich representations of each word
  (useful for classification, NER, etc.)
```

**Used by:** BERT, RoBERTa, ELECTRA

**Good for:** Understanding/classifying text, finding named entities, answering questions about a passage.

### Decoder-Only: Generating Text

The **decoder** generates text one word at a time. Each word can only attend to words that came **before** it (causal/unidirectional) -- because you can't look at words you haven't generated yet!

```
Input: "The cat" → What comes next?
        │
        ▼
┌─────────────────────────────┐
│         DECODER              │
│                              │
│  ┌────────────────────────┐  │
│  │  Transformer Block × N │  │     Each word can ONLY see
│  │  (masked self-attention│  │     words BEFORE it
│  │   + feed-forward)      │  │     (causal masking)
│  └────────────────────────┘  │
│                              │
└─────────────┬───────────────┘
              │
              ▼
  Prediction: "sat" (most likely next word)
```

The "masking" prevents future words from being seen:

```
Processing "The cat sat":

                Attending to:
                The    cat    sat
                ─── ─── ───
  "The" can see: [YES]  [no]   [no]     ← can only see itself
  "cat" can see: [YES]  [YES]  [no]     ← can see "The" and itself
  "sat" can see: [YES]  [YES]  [YES]    ← can see all previous words

  [no] means the attention score is set to -infinity before softmax,
  making the weight effectively 0.
```

**Used by:** GPT-2, GPT-3, GPT-4, LLaMA, Claude, Mistral

**Good for:** Generating text, chatbots, code completion, creative writing.

### Encoder-Decoder: Transforming Text

The **encoder-decoder** model processes an input with the encoder, then generates an output with the decoder. The decoder uses **cross-attention** to look at the encoder's output.

```
Input: "The cat sat"                     Output: "Le chat s'est assis"
        │                                         ▲
        ▼                                         │
┌───────────────────┐                    ┌────────┴──────────┐
│     ENCODER        │                   │      DECODER       │
│                    │                   │                    │
│ ┌────────────────┐ │     cross-        │ ┌────────────────┐ │
│ │ Self-Attention │ │     attention     │ │ Masked Self-   │ │
│ │ + FFN          │ │────────────────→  │ │ Attention      │ │
│ │ (× N layers)  │ │  decoder looks    │ │ + Cross-Attn   │ │
│ └────────────────┘ │  at encoder       │ │ + FFN          │ │
│                    │  output           │ │ (× N layers)   │ │
└────────────────────┘                   │ └────────────────┘ │
                                         └───────────────────┘
```

**Used by:** Original Transformer, T5, BART, mBART

**Good for:** Translation, summarization, any task that transforms one sequence into another.

### Comparison

```
┌──────────────────┬───────────────┬───────────────┬──────────────────┐
│                  │ Encoder-Only  │ Decoder-Only  │ Encoder-Decoder  │
├──────────────────┼───────────────┼───────────────┼──────────────────┤
│ Attention        │ Bidirectional │ Causal (left  │ Both             │
│ direction        │ (see all)     │ to right)     │                  │
├──────────────────┼───────────────┼───────────────┼──────────────────┤
│ Best for         │ Understanding │ Generating    │ Transforming     │
│                  │ text          │ text          │ text             │
├──────────────────┼───────────────┼───────────────┼──────────────────┤
│ Example models   │ BERT          │ GPT, LLaMA    │ T5, BART         │
├──────────────────┼───────────────┼───────────────┼──────────────────┤
│ Example tasks    │ Classification│ Chat, code    │ Translation,     │
│                  │ NER, Q&A      │ completion    │ summarization    │
└──────────────────┴───────────────┴───────────────┴──────────────────┘
```

---

## Full Transformer Architecture (Original Paper)

Here's the complete architecture from the original "Attention Is All You Need" paper:

```
         INPUT                                          OUTPUT
    "The cat sat"                                  "Le chat s'est"
          │                                              │
          ▼                                              ▼
  ┌───────────────┐                              ┌───────────────┐
  │  Input         │                             │  Output        │
  │  Embedding     │                             │  Embedding     │
  └───────┬───────┘                              └───────┬───────┘
          │                                              │
          ▼                                              ▼
  ┌───────────────┐                              ┌───────────────┐
  │  + Positional  │                             │  + Positional  │
  │  Encoding      │                             │  Encoding      │
  └───────┬───────┘                              └───────┬───────┘
          │                                              │
          │              ENCODER                         │           DECODER
          │         ┌──────────────┐                     │      ┌──────────────────┐
          │         │              │                     │      │                  │
          ├────┐    │              │                     ├────┐ │                  │
          ▼    │    │              │                     ▼    │ │                  │
       ┌──────┐│   │              │                  ┌──────┐│ │                  │
       │Multi-││   │              │                  │Masked││ │                  │
       │Head  ││   │              │                  │Multi-││ │                  │
       │Attn  ││   │   ×N         │                  │Head  ││ │                  │
       └──┬───┘│   │   layers     │                  │Attn  ││ │                  │
          │    │    │              │                  └──┬───┘│ │                  │
       Add+Norm◄───┘    │              │               Add+Norm◄───┘ │       ×N        │
          │         │              │                     │      │   layers      │
          ├────┐    │              │            ┌────────┤      │                  │
          ▼    │    │              │            │        ├────┐ │                  │
       ┌──────┐│   │              │            │     ┌──────┐│ │                  │
       │Feed- ││   │              │            │     │Cross-││ │                  │
       │Fwd   ││   │              │            │     │Attn  ││ │                  │
       │Net   ││   │              │            │     └──┬───┘│ │                  │
       └──┬───┘│   │              │            │     Add+Norm◄───┘               │
          │    │    │              │            │        │      │                  │
       Add+Norm◄───┘    │              │            │        ├────┐ │                  │
          │         │              │            │        ▼    │ │                  │
          │         └──────────────┘            │     ┌──────┐│ │                  │
          │                                    │     │Feed- ││ │                  │
          │                                    │     │Fwd   ││ │                  │
          └────────────────────────────────────┘     │Net   ││ │                  │
                   (encoder output goes              └──┬───┘│ │                  │
                    to decoder cross-attention)      Add+Norm◄───┘               │
                                                        │      │                  │
                                                        │      └──────────────────┘
                                                        │
                                                        ▼
                                                 ┌──────────────┐
                                                 │    Linear     │
                                                 │   + Softmax   │
                                                 └──────┬───────┘
                                                        │
                                                        ▼
                                                  Next word
                                                  probabilities
```

---

## How Information Flows (Putting It All Together)

Let's trace how a word gets processed through the entire transformer:

```
Step 1: TOKENIZATION
  "The cat sat" → [100, 2368, 1590]   (words become token IDs)

Step 2: EMBEDDING + POSITIONAL ENCODING
  token 100  → [0.2, -0.1, ...] + [0.0, 1.0, ...] = [0.2, 0.9, ...]
  token 2368 → [0.8, 0.3, ...]  + [0.8, 0.5, ...] = [1.6, 0.8, ...]
  token 1590 → [0.1, 0.7, ...]  + [0.9, -0.4,...] = [1.0, 0.3, ...]

Step 3: TRANSFORMER BLOCKS (repeated N times)

  Each block refines the representations:

  Block 1: "cat" learns it's preceded by "The" (grammar)
  Block 2: "sat" learns the subject is "cat" (semantics)
  Block 3: "sat" learns "cat" is tired (deeper context)
  ...
  Block N: Rich, context-aware representations

Step 4: OUTPUT
  Final representations → predictions
  (next word, classification, translation, etc.)
```

---

## Hyperparameters: The Knobs You Can Turn

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

## Reading Order

We recommend studying the architecture components in this order. For each topic, read the theory guide first, then work through the interactive notebook:

| Order | Theory (Read) | Hands-on (Code) | What You'll Learn |
|-------|---------------|------------------|-------------------|
| 1 | [Attention Mechanisms](./attention-mechanisms.md) | [Notebook](./01_attention_mechanisms.ipynb) | Q, K, V, dot product, softmax |
| 2 | [Multi-Head Attention](./multi-head-attention.md) | [Notebook](./02_multi_head_attention.ipynb) | Parallel heads, specialization |
| 3 | [Positional Encoding](./positional-encoding.md) | [Notebook](./03_positional_encoding.ipynb) | How word order is encoded |
| 4 | (covered above) | [Notebook](./04_transformer_block.ipynb) | LayerNorm, residuals, FFN, full block |

After these four, you'll understand all the core building blocks and have built a complete transformer from scratch in NumPy.

---

## Key Takeaways

1. **Transformers process all words simultaneously** using attention (not one at a time like RNNs)
2. **The transformer block** = multi-head attention + feed-forward network + residual connections + layer norm
3. **Multiple blocks stack** to create deep understanding (6 to 96+ layers)
4. **Three flavors:** encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5)
5. **Attention** is the key innovation -- it lets every word consider every other word
6. **Residual connections** and **layer normalization** make training deep networks stable

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) -- the original transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) -- visual walkthrough
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) -- line-by-line code walkthrough

---

[Back to Transformers Module](../README.md)
