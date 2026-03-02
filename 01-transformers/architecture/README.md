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

### Why These Values?

The hyperparameter choices above are not arbitrary. Each has a rationale grounded in empirical results and hardware constraints.

**Why d_k = 64?**

When d_k is too small (below about 32), each attention head has insufficient capacity to represent complex query-key relationships. The attention matrix becomes underpowered. When d_k is very large, the QK^T matrix multiply dominates total FLOPs and the dot products before softmax grow in magnitude proportionally to sqrt(d_k) — this is why the original paper scales by 1/sqrt(d_k), to prevent softmax saturation. d_k = 64 was found empirically to balance head capacity and compute cost. Notably, d_k = 64 has been used from GPT-1 through GPT-3 and across LLaMA-1, 2, and 3 with almost no variation — it is remarkably robust across scales.

**Why d_ff = 4 × d_model?**

Geva et al. (2021) showed that FFN layers act as key-value memories: the first linear layer acts as a key lookup, and the second projects from value space back to d_model. The 4× expansion provides enough key-value capacity for the model to store factual associations during pre-training. Experiments consistently show: reducing to 2× costs 3-5% on downstream benchmarks. Increasing to 8× rarely helps. The relationship is not linear — there is a diminishing returns curve that peaks around 4×. LLaMA uses a variant with three matrices and SwiGLU activation instead of the standard two-matrix ReLU/GELU FFN. The three-matrix structure (gate × up-projection, then down-projection) effectively makes d_ff ≈ 2.67 × d_model in parameter-count terms, but the gating mechanism adds expressivity that compensates for the smaller expansion ratio.

**Depth vs. width tradeoff:**

For a fixed parameter budget, deeper networks (more layers, smaller d_model) generalize better. Wider networks (fewer layers, larger d_model) memorize better. In practice, depth is the more important scaling axis — GPT-3 scales to 96 layers while keeping d_model relatively modest relative to total parameters. However, very deep networks are harder to train: gradient flow through 96 layers requires Pre-LN or careful warmup. The scaling laws (Kaplan et al. 2020) showed that for a fixed compute budget, the optimal depth/width ratio favors depth, but the curve is shallow enough that within 2× of optimal depth, quality differences are small.

---

## System-Level Failure Modes

### Learning rate warmup with Post-LN

In the original transformer (Post-LN), the gradient at layer 1 depends on the product of Jacobians through all subsequent layers. At initialization, this product is highly variable, producing enormous gradient magnitudes. Without warmup, early gradient steps are huge and training diverges. The standard fix from the original paper: linear warmup for 4000 steps (from 0 to the peak learning rate), then inverse square root decay. Pre-LN architectures avoid this problem because the residual branch bypasses normalization — gradient magnitude at any layer is bounded by the identity connection in the residual path, making it approximately 1.0 at initialization regardless of depth. This is the primary reason modern large models use Pre-LN: you can set the learning rate immediately without warmup, and training stability improves across the board.

### Gradient clipping is essential

Training without gradient clipping (standard: clip_norm = 1.0) leads to occasional gradient explosions, particularly with Post-LN and long sequences. Even with Pre-LN, rare gradient spikes appear in practice — a single outlier batch can destabilize training if unclipped. Always clip gradients. The clip norm of 1.0 is nearly universal across GPT-2, GPT-3, LLaMA, and BERT. If you see training loss spike suddenly after many stable steps and then not recover, a gradient explosion that was not fully caught by clipping is the first hypothesis to investigate.

### FFN d_ff ratio sensitivity

Reducing d_ff/d_model from 4 to 2 costs 3-5% on downstream benchmarks. This seems small but is consistent across tasks and training budgets. More importantly, an undersized FFN forces the model to push more computation into the attention weights. Attention weights are harder to interpret and harder to fine-tune efficiently (LoRA targets FFN matrices because they contain more factual knowledge; an undersized FFN makes this strategy less effective). Do not cut d_ff as a cost-saving measure without validating on your target benchmark.

### Depth during fine-tuning

Very deep pre-trained models (48+ layers) can catastrophically forget general knowledge during aggressive fine-tuning on narrow datasets. The phenomenon is well-documented: lower layers learn general syntactic and phonological patterns; upper layers learn task-specific representations. Aggressive fine-tuning on a small dataset updates all layers equally but has insufficient signal to properly re-learn lower-layer representations, leading to degradation on out-of-distribution prompts. Standard mitigations: (1) freeze the bottom N layers and fine-tune only upper layers; (2) use a lower learning rate for lower layers (discriminative fine-tuning, ULMFiT style); (3) use LoRA — low-rank adaptation adds small rank-r perturbation matrices (r=8 to 64) to attention and FFN weight matrices, dramatically reducing the number of trainable parameters and limiting the magnitude of change to any individual weight.

---

## Staff/Principal Interview Depth

**Q1: What are the key differences in backpropagation between encoder-only and decoder-only transformers?**

In an encoder-only model (BERT), all tokens are visible to all other tokens. The attention matrix is fully dense (n×n). Gradients flow freely from the loss at any token to all other tokens via attention. Every token position contributes to predictions at every other position, so gradient signal is distributed uniformly across positions.

In a decoder-only model (GPT), position i can only attend to positions ≤ i. The upper triangle of the attention matrix is masked to -infinity before softmax, producing zero attention weights and zero gradient flow from future to past tokens. During backpropagation: the gradient of the loss at position j with respect to attention weights involving position i is zero whenever i > j. This means earlier tokens receive less gradient signal — position 0 only contributes to one prediction (its own output token if using a reconstruction loss), while position n-1 contributes to all n predictions. In practice, decoder-only models experience weaker gradient signal at early positions, which is partially why decoder training benefits from large batch sizes (to average over many positions and reduce variance) and why the loss on early tokens is often down-weighted in practice.

This asymmetry also affects what the model learns: BERT's bidirectional context means every position has full context for every prediction, making it excellent for understanding tasks. GPT must predict each token from only left context, which forces it to develop a strong generative model of language structure.

**Q2: Describe the KV cache and how it enables efficient autoregressive inference.**

During autoregressive generation (token-by-token), each new token needs to attend to ALL previous tokens. Without a cache, you would recompute K and V for all previous tokens at every generation step — O(n²) total compute for n tokens, and redundant recomputation of the same K and V vectors at every step.

KV cache: after computing token t, store K_t and V_t for every layer. When generating token t+1, compute only Q_{t+1} (a single vector of size d_model), load all cached K_{0..t} and V_{0..t}, compute attention(Q_{t+1}, K_{0..t}, V_{0..t}), and append K_{t+1} and V_{t+1} to the cache. Total generation compute drops from O(n²) to O(n) in the attention layers.

Memory cost: 2 (K and V) × n_layers × d_model × n_tokens × bytes_per_element. For LLaMA-2 70B (80 layers, d_model=8192, float16) at 4096 context: 2 × 80 × 8192 × 4096 × 2 bytes ≈ 10.7 GB per concurrent request. This is why memory, not compute, is the primary bottleneck for serving long-context LLMs. Serving 10 concurrent requests at 4096 context requires ~107 GB just for KV cache, on top of model weights. Techniques like grouped query attention (GQA, used in LLaMA-2 70B) reduce the KV cache by sharing K and V across groups of attention heads — LLaMA-2 70B uses 8 KV heads for 64 attention heads (8× reduction), cutting KV cache from 10.7 GB to 1.3 GB per request at 4096 context.

**Q3: Why do modern LLMs use 32-96 layers when the original paper used 6? What changed?**

Three things changed after the original 6-layer transformer.

First, compute. The original model trained on 8 P100 GPUs for 3.5 days. Modern models train on thousands of A100/H100s for weeks to months. The compute budget increased by approximately 5-6 orders of magnitude, making much larger models feasible.

Second, training stability. The original Post-LN transformer becomes harder to train past approximately 12 layers without careful warmup tuning. Pre-LN (used by GPT-2 onward) stabilizes gradient flow through arbitrarily deep networks by ensuring that gradient magnitude along the residual path is approximately 1.0 at initialization regardless of depth. This removed the practical ceiling on depth.

Third, and most importantly, scaling laws. Kaplan et al. (2020) demonstrated that model quality scales as a power law with compute, data, and parameters — and that for a fixed compute budget, the optimal model is much larger than was previously used. Specifically, for a fixed FLOP budget, the optimal parameter count is roughly proportional to C^0.5 (where C is total training compute). At modern compute scales, this optimal point falls in the range of 7B-70B+ parameters. Deeper models are parameter-efficient: a 96-layer model with d_model=12288 achieves far better perplexity per parameter than a 6-layer model of the same total size because depth enables compositional computation that width alone cannot replicate.

**Q4: What is the memory/compute tradeoff of depth vs. width, and how would you advise choosing between them for a 7B parameter model?**

Consider two architectures with approximately equal total parameters:

Option A (narrow-deep): d_model=2048, 64 layers, h=16 heads, d_ff=8192. Total params ≈ 64 × (4×2048² + 2×2048×8192) ≈ 7.2B.

Option B (wide-shallow): d_model=4096, 16 layers, h=32 heads, d_ff=16384. Total params ≈ 16 × (4×4096² + 2×4096×16384) ≈ 7.2B.

Compute is approximately equal (both proportional to total parameters × sequence length).

KV cache memory: proportional to n_layers × d_model × context_len. Option A: 64 × 2048 × context_len. Option B: 16 × 4096 × context_len. Option A uses 2× less KV cache at equal context length — a significant advantage for long-context serving.

Quality: option A (deeper) generally wins. Deeper networks form more compositional representations — each layer can build on the abstractions of previous layers. Width alone adds parameters but not representational depth. Empirically, models below 28 layers at 7B scale show consistent quality degradation on reasoning benchmarks.

LLaMA-2 7B settled on 32 layers, d_model=4096, d_k=128 (32 heads). Practical constraints shaped this choice beyond theory:

- d_model must be divisible by the number of tensor-parallel GPUs (typically 8 or 16) and by n_heads.
- d_ff must be a multiple of 64 (for Tensor Core utilization on NVIDIA hardware). LLaMA uses d_ff = 11008 ≈ 2.67 × 4096 with SwiGLU.
- d_k = 128 (rather than 64) is slightly larger than the classic default, providing more capacity per head at the cost of larger attention matrices.

For a new 7B model in 2025: default to 32-36 layers, d_model=4096, and validate d_ff between 11008 (SwiGLU) and 16384 (standard). Use GQA with 8 KV heads to cut KV cache by 4×. Use RoPE with base θ=500,000 for good long-context handling. These choices are not theoretically optimal but are empirically validated at this scale.

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
