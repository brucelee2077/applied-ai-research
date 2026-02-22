# Positional Encoding

## The Problem: Transformers Have No Sense of Order

Remember how attention works? Every word looks at every other word **all at once**. That's powerful, but it creates a surprising problem: **the transformer doesn't know what order the words are in.**

Think about these two sentences:

> "The dog chased the cat."
> "The cat chased the dog."

Same words, completely different meanings! But to a transformer without positional encoding, these two sentences look **identical** -- just a bag of the same five words. The attention mechanism treats them the same way because it processes everything in parallel.

Older models (RNNs) didn't have this problem because they read words one at a time, left to right. The order was built into the process. But transformers traded sequential processing for speed -- and lost the sense of order.

**Positional encoding** is the solution: we add information about each word's position directly into its representation.

---

## The Seat Number Analogy

Imagine you're at a movie theater. Everyone has a ticket with the movie name -- but no seat number.

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

Word "sat" at position 2:
  word_embedding("sat")     = [0.1, 0.7, 0.4, -0.3]    ← what "sat" means
  position_encoding(2)      = [0.9, 0.1, -0.4, 0.5]    ← position 2
  ─────────────────────────────────────────────────
  final_embedding           = [1.0, 0.8,  0.0, 0.2]    ← meaning + position
```

We literally just **add** the position vector to the word vector. Simple!

But the key question is: **what numbers should we use for the position vectors?**

---

## Approach 1: Sinusoidal Positional Encoding (Original Transformer)

The original "Attention Is All You Need" paper used a clever mathematical formula based on **sine and cosine waves** (the same waves used to describe sound and light).

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

For a word at position `pos`, and for each dimension `i` of the encoding vector:

```
PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))

Where:
  pos     = position of the word in the sentence (0, 1, 2, ...)
  i       = the dimension index
  d_model = total size of the encoding (e.g., 512)
```

Don't worry about memorizing this! The key ideas are:

1. **Even dimensions use sine, odd dimensions use cosine** -- like having two clock hands per speed
2. **Different dimensions have different frequencies** -- some wave slowly, some wave quickly
3. **Each position gets a unique pattern** -- no two positions have the same encoding

### What It Looks Like

```
Position encodings for a small example (4 dimensions):

         dim0    dim1    dim2    dim3
         (sin)   (cos)   (sin)   (cos)
         slow    slow    fast    fast
pos 0:  [ 0.00,  1.00,   0.00,  1.00 ]
pos 1:  [ 0.84,  0.54,   0.01,  1.00 ]
pos 2:  [ 0.91, -0.42,   0.02,  1.00 ]
pos 3:  [ 0.14, -0.99,   0.03,  1.00 ]
pos 4:  [-0.76, -0.65,   0.04,  1.00 ]
pos 5:  [-0.96,  0.28,   0.05,  1.00 ]

Notice:
  • Slow dimensions (0,1): change a LOT between positions → big picture
  • Fast dimensions (2,3): change a LITTLE between positions → fine detail
  • Every row (position) has a unique combination of values!

Visualized as a wave pattern:

dim 0 (slow sine):
pos:  0    1    2    3    4    5    6    7    8    9
      ·    ╱─╲       ╱
      ·   ╱   ╲     ╱
      ·──╱     ╲   ╱
      ·         ╲─╱
               period = long

dim 2 (fast sine):
pos:  0    1    2    3    4    5    6    7    8    9
      · ╱╲ ╱╲ ╱╲ ╱╲ ╱╲
      ·╱  ╳  ╳  ╳  ╳  ╲
      ·  ╱ ╲╱ ╲╱ ╲╱ ╲╱
               period = short
```

### Why Sine/Cosine? The Distance Property

A beautiful property of sinusoidal encoding: **the encoding for position `pos + k` can be expressed as a linear combination of the encoding at position `pos`**. This means the model can easily learn to compute "how far apart are these two words?"

```
Think of it this way:

If you know where you are on a clock face (say, 3 o'clock),
you can easily figure out where 2 positions later would be
(5 o'clock) using simple rotation math.

Similarly, the transformer can learn:
  "The word 3 positions to my left" or
  "The word 5 positions to my right"

This is crucial for learning patterns like:
  "Adjectives usually appear 1 position before nouns"
  "Verbs usually appear 1-2 positions after subjects"
```

---

## Approach 2: Learned Positional Embeddings

An alternative approach: **just let the model learn the best position encodings from data!**

```
Instead of using a mathematical formula, create a lookup table:

Position    Learned Encoding (random at first, refined by training)
────────    ────────────────────────────────────────────────────────
0           [0.12, -0.34, 0.56, ...]     ← learned from data
1           [0.78, 0.23, -0.11, ...]     ← learned from data
2           [-0.45, 0.67, 0.89, ...]     ← learned from data
...
511         [0.33, -0.22, 0.44, ...]     ← learned from data

These are just regular parameters that get updated during training,
like any other weights in the network.
```

### Comparison

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

In practice, both approaches work similarly well. Most modern models (BERT, GPT) use learned embeddings because they're simpler to implement and the maximum sequence length is fixed anyway.

---

## Approach 3: Relative Positional Encoding

Both approaches above encode **absolute** position: "I am the 5th word." Relative positional encoding instead encodes **distance between words**: "That word is 3 positions to my left."

### Why Relative Positions?

```
Consider these sentences:

  "The cat sat on the mat"
  "Yesterday, the cat sat on the mat"

In both sentences, the relationship between "cat" and "sat" is the same:
  → "sat" is 1 position after "cat"

But their ABSOLUTE positions are different:
  → Sentence 1: "cat" is at position 1, "sat" is at position 2
  → Sentence 2: "cat" is at position 2, "sat" is at position 3

Relative encoding captures: "sat is 1 step after cat" regardless of
where they appear in the sentence. This makes patterns more portable!
```

### How It Works (Simplified)

Instead of adding position information to the embeddings, relative positional encoding modifies the **attention scores** directly:

```
Standard attention:
  score = Q · K

With relative position:
  score = Q · K + Q · R(distance)

  where R(distance) encodes the relative distance between two words

Example: Computing attention from "sat" (pos 2) to other words:

  "The" (pos 0):  distance = 2-0 = 2  →  use R(2)
  "cat" (pos 1):  distance = 2-1 = 1  →  use R(1)
  "sat" (pos 2):  distance = 2-2 = 0  →  use R(0)
```

### Variants Used in Modern Models

```
Model               Positional Encoding Type
─────────────       ─────────────────────────────────────────
Original Transformer   Sinusoidal (absolute)
BERT                   Learned (absolute)
GPT-2/3               Learned (absolute)
Transformer-XL        Relative (Shaw et al.)
T5                     Relative bias (simplified)
RoPE (LLaMA, etc.)    Rotary (combines absolute and relative)
ALiBi (BLOOM)          Linear bias based on distance
```

---

## RoPE: Rotary Position Embeddings (Modern Approach)

Many recent large language models (LLaMA, Mistral, etc.) use **Rotary Position Embeddings (RoPE)**. The core idea is elegant: instead of adding position information, **rotate** the Q and K vectors based on position.

```
Think of it like a combination lock:

  Position 0: rotate 0 degrees     🔒 [→]
  Position 1: rotate 30 degrees    🔒 [↗]
  Position 2: rotate 60 degrees    🔒 [↑]
  Position 3: rotate 90 degrees    🔒 [←]

When computing attention (dot product) between two words:
  - Words at similar positions have similar rotations → higher dot product
  - The dot product naturally depends on RELATIVE distance

The beauty: you get relative position information for free,
just by rotating the vectors before computing attention!
```

This is a more advanced topic -- the key takeaway is that position encoding is still an active area of research, and different approaches have different tradeoffs.

---

## Putting It All Together

Here's where positional encoding fits in the full transformer pipeline:

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

## Key Takeaways

1. **Transformers process all words in parallel**, so they need explicit position information
2. **Positional encoding adds "where" to "what"** -- combining position with meaning
3. **Sinusoidal encoding** uses math (sine/cosine waves) to create unique position patterns
4. **Learned encoding** lets the model figure out the best position representation from data
5. **Relative encoding** captures distance between words rather than absolute position
6. **Most modern models** use either learned absolute positions (GPT, BERT) or rotary embeddings (LLaMA)
7. Position encoding is **added** (not concatenated) to word embeddings

---

## Prerequisites

Before reading this, you should understand:
- [Attention Mechanisms](./attention-mechanisms.md) -- why position matters for attention
- Basic trigonometry concepts (sine/cosine) are helpful but not required

## Further Reading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) -- Section 3.5
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) -- modern approach
- [ALiBi: Train Short, Test Long](https://arxiv.org/abs/2108.12409) -- linear bias approach

---

[Previous: Multi-Head Attention](./multi-head-attention.md) | [Back to Architecture Overview](./README.md)
