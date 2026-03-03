# Positional Encoding

## The Mystery Worth Solving

Here's a puzzle. These two sentences contain identical words:

> "The dog chased the cat."
> "The cat chased the dog."

They have completely opposite meanings. Now here's the question: if you show both sentences to a transformer that processes all words simultaneously — rather than one at a time — how does it know which meaning is which?

Without an answer to this question, a transformer is blind to order. It can't tell "dog chases cat" from "cat chases dog."

Positional encoding is the answer. And the way it solves the problem is more elegant than you might expect.

---

**Before you start, you need to know:**
- How attention works (Q, K, V, self-attention) — covered in [Attention Mechanisms](./attention-mechanisms.md)
- What word embeddings are (a word turned into a vector of numbers) — covered in [Neural Network Fundamentals](../../00-neural-networks/fundamentals/04_neural_network_layers.ipynb)

---

## The Problem: Transformers Have No Sense of Order

Remember how attention works? Every word looks at every other word **all at once**. That's powerful, but it creates a surprising problem: **the transformer doesn't know what order the words are in.**

Think about these two sentences:

> "The dog chased the cat."
> "The cat chased the dog."

Same words, completely different meanings! But to a transformer without positional encoding, these two sentences look **identical** — just a bag of the same five words. The attention mechanism treats them the same way because it processes everything in parallel.

Older models (RNNs) didn't have this problem because they read words one at a time, left to right. The order was built into the process. But transformers traded sequential processing for speed — and lost the sense of order.

**Positional encoding** is the solution: we add information about each word's position directly into its representation.

---

## The Seat Number Analogy

Imagine you're at a movie theater. Everyone has a ticket with the movie name — but no seat number.

```
Without positional encoding (chaos!):
┌─────────────────────────────────────────────┐
│  🎬 Movie Theater                           │
│                                             │
│  Ticket: "Avengers"    Ticket: "Avengers"   │
│  Ticket: "Avengers"    Ticket: "Avengers"   │
│                                             │
│  Everyone knows WHAT movie, but nobody      │
│  knows WHERE to sit!                        │
└─────────────────────────────────────────────┘

With positional encoding (order restored!):
┌─────────────────────────────────────────────┐
│  🎬 Movie Theater                           │
│                                             │
│  Seat A1: "Avengers"   Seat A2: "Avengers"  │
│  Seat B1: "Avengers"   Seat B2: "Avengers"  │
│                                             │
│  Now everyone knows both WHAT movie         │
│  and WHERE their seat is!                   │
└─────────────────────────────────────────────┘
```

In a transformer, each word already has a vector (embedding) that says WHAT the word means. Positional encoding adds another vector that says WHERE the word is in the sentence.

**What this analogy gets right:** A seat number and a movie ticket give you two separate pieces of information that combine to tell you everything you need to know — who you are + where you sit. Word embeddings + positional encoding work the same way.

**Where this analogy breaks down:** In a theater, the management assigns seat numbers — uniqueness is guaranteed. In a transformer, the positional encoding must be designed carefully so each position is actually distinguishable. If you chose bad positional encodings, the model couldn't tell positions apart.

---

## How It Works: Adding Position to Meaning

The core idea is simple:

```
Final word representation = Word meaning + Position information

Example for the sentence "The cat sat":

Word "The" at position 0:
  word_embedding("The")     = [0.2, -0.1, 0.5, 0.3]    ← what "The" means
  position_encoding(0)      = [0.0,  0.0, 1.0, 1.0]    ← position 0
  ─────────────────────────────────────────────────
  final_embedding           = [0.2, -0.1, 1.5, 1.3]    ← meaning + position

Word "cat" at position 1:
  word_embedding("cat")     = [0.8, 0.3, -0.2, 0.6]    ← what "cat" means
  position_encoding(1)      = [0.8, 0.6,  0.6, 0.8]    ← position 1
  ─────────────────────────────────────────────────
  final_embedding           = [1.6, 0.9,  0.4, 1.4]    ← meaning + position
```

We literally just **add** the position vector to the word vector.

But the key question is: **what numbers should we use for the position vectors?**

---

## Approach 1: Sinusoidal Positional Encoding (Original Transformer)

The original "Attention Is All You Need" paper used a clever formula based on **sine and cosine waves** — the same waves that describe sound and light.

### The Clock Analogy

Think about how a clock tells time using multiple hands:

```
A clock uses MULTIPLE rotating hands at different speeds:

  Hour hand:    rotates slowly    → tells you the rough time (morning/afternoon)
  Minute hand:  rotates faster    → tells you the specific minute
  Second hand:  rotates fastest   → tells you the exact second

  Together, they give a UNIQUE time reading for every moment!

Positional encoding works the same way:

  Dimension 0-1:   slow wave    → tells you rough position (beginning/middle/end)
  Dimension 2-3:   medium wave  → tells you more specific position
  Dimension 4-5:   fast wave    → tells you exact position
  ...and so on

  Together, they give a UNIQUE position code for every word!
```

### The Formula

For a word at position `pos`, and for each dimension `i`:

```
PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))

Where:
  pos     = position of the word in the sentence (0, 1, 2, ...)
  i       = the dimension index
  d_model = total size of the encoding (e.g., 512)
```

Key ideas:
1. **Even dimensions use sine, odd dimensions use cosine** — like having two clock hands per speed
2. **Different dimensions have different frequencies** — some wave slowly, some wave quickly
3. **Each position gets a unique pattern** — no two positions have the same encoding

**What this analogy gets right:** Multiple hands at different speeds combine to give a unique fingerprint for every moment in time. Sinusoidal PE does the same for positions: multiple waves at different speeds combine to create a unique fingerprint per position.

**Where this analogy breaks down:** A real clock's hands have fixed mechanical speeds set by gears. The frequency schedule in sinusoidal PE (the formula uses 10000 as a base) was chosen by the original authors through experimentation. Also, the model doesn't literally "read a clock" — it learns to extract position information from those numbers through training.

---

## Worked Example with Real Numbers

Let's compute PE(pos=2) step by step with d_model=4.

With d_model=4, we have four dimensions (0, 1, 2, 3). The denominator becomes 10000^(i/2).

```
dim 0  (i=0, even → use sine):
  PE(2, 0) = sin(2 / 10000^0) = sin(2 / 1) = sin(2) ≈ 0.909

dim 1  (i=0, odd → use cosine):
  PE(2, 1) = cos(2 / 10000^0) = cos(2 / 1) = cos(2) ≈ -0.416

dim 2  (i=1, even → use sine):
  PE(2, 2) = sin(2 / 10000^0.5) = sin(2 / 100) = sin(0.02) ≈ 0.0200

dim 3  (i=1, odd → use cosine):
  PE(2, 3) = cos(2 / 100) = cos(0.02) ≈ 0.9998
```

So PE(pos=2) ≈ [0.909, -0.416, 0.020, 0.9998].

What's happening here:
- **Dimensions 0-1** use the fastest wave (dividing by 1). These change a lot between adjacent positions.
- **Dimensions 2-3** use a much slower wave (dividing by 100). These barely change between nearby positions, but clearly distinguish positions far apart.

Together they form a unique "fingerprint" for position 2. No other position produces exactly this combination.

---

## Why Sine and Cosine? The Distance Property

Here is something elegant: **if you know the encoding at position p, you can figure out the encoding at position p+k using just a fixed rotation** — the same rotation works no matter where you start.

Imagine you're standing at mile marker 50 on a highway. If you know your "PE address", you can compute what mile marker 53 looks like with simple math. The distance between them (3 miles) is all you need — it doesn't matter where on the highway you started.

This property means a trained transformer can learn to compute "how far apart are these two words?" without memorizing a lookup table for every possible pair of distances.

This mathematical structure is what inspired RoPE — the positional encoding used in LLaMA, Mistral, and Gemma. Every time you chat with one of those models, you're benefiting from this geometry.

---

## Approach 2: Learned Positional Embeddings

An alternative: **just let the model learn the best position encodings from data.**

```
Instead of using a mathematical formula, create a lookup table:

Position    Learned Encoding (random at first, refined by training)
────────    ────────────────────────────────────────────────────────
0           [0.12, -0.34, 0.56, ...]     ← learned from data
1           [0.78, 0.23, -0.11, ...]     ← learned from data
2           [-0.45, 0.67, 0.89, ...]     ← learned from data
...
511         [0.33, -0.22, 0.44, ...]     ← learned from data

These are regular parameters that get updated during training,
like any other weights in the network.
```

```
┌─────────────────┬──────────────────────┬──────────────────────┐
│                 │ Sinusoidal           │ Learned              │
├─────────────────┼──────────────────────┼──────────────────────┤
│ How created     │ Mathematical formula │ Learned from data    │
│ Can handle      │ Any length           │ Only lengths seen    │
│ longer texts?   │ (even unseen ones)   │ during training      │
│ Extra params?   │ No (formula-based)   │ Yes (one vector per  │
│                 │                      │ possible position)   │
│ Performance     │ Comparable           │ Comparable           │
│ Used by         │ Original Transformer │ BERT, GPT-2, GPT-3  │
└─────────────────┴──────────────────────┴──────────────────────┘
```

In practice, both work similarly well. Most modern models (BERT, GPT) use learned embeddings because they're simpler to implement and the maximum sequence length is fixed anyway.

---

## Approach 3: Relative Positional Encoding

Both approaches above encode **absolute** position: "I am the 5th word." Relative positional encoding instead encodes **distance between words**: "That word is 3 positions to my left."

Why this matters:

```
Consider these sentences:

  "The cat sat on the mat"
  "Yesterday, the cat sat on the mat"

In both sentences, "sat" is 1 position after "cat" — the relationship is the same.

But their ABSOLUTE positions are different:
  → Sentence 1: "cat" is at position 1, "sat" is at position 2
  → Sentence 2: "cat" is at position 2, "sat" is at position 3

Relative encoding captures: "sat is 1 step after cat" regardless of
where they appear. This makes patterns more portable!
```

Modern models like LLaMA and Mistral use **RoPE (Rotary Position Embeddings)**, which rotates Q and K vectors before computing attention. This means position information is baked directly into the attention scores — not added to the embeddings. ALiBi (used in BLOOM) adds a simple linear penalty to attention scores based on distance: the further apart two words, the lower their attention score.

---

## Putting It All Together

```
Input sentence: "The cat sat"
        │
        ▼
┌───────────────────┐
│ Tokenization      │   "The" → token 100
│                   │   "cat" → token 2368
│                   │   "sat" → token 1590
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Word Embeddings   │   token 100  → [0.2, -0.1, 0.5, ...]
│ (lookup table)    │   token 2368 → [0.8, 0.3, -0.2, ...]
│                   │   token 1590 → [0.1, 0.7, 0.4, ...]
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ + Positional      │   pos 0 → [0.0, 1.0, 0.0, ...]
│   Encoding        │   pos 1 → [0.8, 0.5, 0.01, ...]
│                   │   pos 2 → [0.9, -0.4, 0.02, ...]
└───────┬───────────┘
        │
        ▼
  Combined vectors     [0.2, 0.9, 0.5, ...]   ← "The" + position 0
  (meaning + position) [1.6, 0.8, -0.19, ...]  ← "cat" + position 1
                       [1.0, 0.3, 0.42, ...]   ← "sat" + position 2
        │
        ▼
┌───────────────────┐
│ Transformer       │   Now attention can consider BOTH
│ Layers            │   meaning AND position!
│ (attention, etc.) │
└───────────────────┘
```

---

## Quick Check — can you answer these?

- Why do transformers need positional encoding but RNNs don't?
- What is the key difference between sinusoidal and learned positional encoding?
- Why is position information **added** to word embeddings rather than kept separate?

If you can't answer one, go back and re-read that part. That is completely normal.

---

## Victory Lap

You just worked through the core idea behind positional encoding — and the elegant distance property that makes sinusoidal encoding special. This same mathematical structure (that a fixed rotation maps any position to any other position at a fixed distance) inspired RoPE, which is used in LLaMA, Mistral, and Gemma. Every time you chat with a modern AI assistant, you're benefiting from this geometry. The position encoding you just learned is a direct ancestor of the mechanism that lets those models handle 32K+ token contexts.

---

Ready to go deeper? → [Positional Encoding — Interview Deep-Dive](./positional-encoding-interview.md)

---

**Further Reading**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — Section 3.5
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — modern approach
- [ALiBi: Train Short, Test Long](https://arxiv.org/abs/2108.12409) — linear bias approach

---

[Previous: Multi-Head Attention](./multi-head-attention.md) | [Back to Architecture Overview](./README.md)
