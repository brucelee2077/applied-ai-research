# Multi-Head Attention

## Why Not Just One Attention?

In the [previous section](./attention-mechanisms.md), we learned how attention lets each word look at every other word and decide what's relevant. But here's the thing: **a single attention mechanism can only focus on one type of relationship at a time.**

Think about this sentence:

> "The **tired** cat sat on the **soft** mat."

To fully understand the word "cat", you need to notice **multiple things** at once:
- **What is it?** → "cat" (it's an animal)
- **What's it doing?** → "sat" (the action)
- **What describes it?** → "tired" (its state)
- **Where is it?** → "on the mat" (its location)

One attention head might learn to focus on adjectives ("tired"), while another learns to focus on verbs ("sat"), and another on locations ("on the mat").

**Multi-head attention** runs several attention mechanisms **in parallel**, each one learning to look for different types of relationships.

---

## The Group Project Analogy

Imagine you're a teacher assigning a group project about a book. Instead of having one student analyze everything, you split the work:

```
Book: "Harry Potter and the Philosopher's Stone"

Student 1 (Head 1):  "Analyze the CHARACTERS"
                     → focuses on who interacts with whom

Student 2 (Head 2):  "Analyze the PLOT"
                     → focuses on cause and effect between events

Student 3 (Head 3):  "Analyze the SETTING"
                     → focuses on where and when things happen

Student 4 (Head 4):  "Analyze the THEMES"
                     → focuses on deeper meanings and motifs

Each student reads the SAME book but looks for DIFFERENT things.
At the end, they COMBINE their findings into one comprehensive report.
```

That's exactly what multi-head attention does:
1. **Split** the work across multiple "heads"
2. Each head performs attention **independently**, learning to focus on different patterns
3. **Combine** all the results back together

---

## How It Works: Step by Step

### Step 1: Split the Embedding into Heads

Remember, each word has an embedding vector (a list of numbers). In multi-head attention, we **split** this vector into smaller pieces, one for each head.

```
Example: Word embedding with 512 numbers, split across 8 heads

Full embedding for "cat": [0.2, -0.1, 0.5, 0.3, ..., 0.7, -0.4]
                           ←────────── 512 numbers ──────────────→

Split into 8 heads (each gets 512 ÷ 8 = 64 numbers):

Head 1: [0.2, -0.1, ..., 0.3]     ← 64 numbers
Head 2: [0.5,  0.3, ..., 0.1]     ← 64 numbers
Head 3: [0.7, -0.4, ..., 0.2]     ← 64 numbers
Head 4: [0.1,  0.6, ..., 0.5]     ← 64 numbers
Head 5: [0.3, -0.2, ..., 0.4]     ← 64 numbers
Head 6: [0.8,  0.1, ..., 0.6]     ← 64 numbers
Head 7: [0.4, -0.3, ..., 0.7]     ← 64 numbers
Head 8: [0.6,  0.5, ..., 0.8]     ← 64 numbers
```

Technically, each head has its own learned W_Q, W_K, W_V matrices that project the full embedding into a smaller Q, K, V for that head. The effect is similar to each head getting a different "slice" of the information.

### Step 2: Each Head Does Independent Attention

Each head performs the full attention computation we learned before (Q, K, V, dot product, scale, softmax, weighted sum) -- but in its own smaller space:

```
Head 1: "I'll focus on grammar"          Head 2: "I'll focus on meaning"
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  Q₁ × K₁ᵀ                   │         │  Q₂ × K₂ᵀ                   │
│  ──────── → softmax → × V₁  │         │  ──────── → softmax → × V₂  │
│   √d_k                      │         │   √d_k                      │
│                              │         │                              │
│  "cat" attends to:           │         │  "cat" attends to:           │
│    "The" → 0.60  (determiner)│         │    "tired" → 0.55 (adjective)│
│    "sat" → 0.30  (verb)      │         │    "cat" → 0.25 (self)       │
│    "tired" → 0.10            │         │    "sat" → 0.20 (action)     │
└──────────────────────────────┘         └──────────────────────────────┘

Head 3: "I'll focus on position"         Head 4: "I'll focus on context"
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  Q₃ × K₃ᵀ                   │         │  Q₄ × K₄ᵀ                   │
│  ──────── → softmax → × V₃  │         │  ──────── → softmax → × V₄  │
│   √d_k                      │         │   √d_k                      │
│                              │         │                              │
│  "cat" attends to:           │         │  "cat" attends to:           │
│    "sat" → 0.50  (next word) │         │    "mat" → 0.40  (related)  │
│    "The" → 0.35  (prev word) │         │    "on" → 0.30   (prepos.)  │
│    "on" → 0.15               │         │    "tired" → 0.30 (state)   │
└──────────────────────────────┘         └──────────────────────────────┘

Each head sees the same words but learns to find DIFFERENT patterns!
```

### Step 3: Concatenate All Heads

After each head computes its output, we **concatenate** (join together) all the results:

```
Head 1 output: [0.3, 0.1, ..., 0.4]     ← 64 numbers
Head 2 output: [0.5, 0.2, ..., 0.6]     ← 64 numbers
Head 3 output: [0.1, 0.7, ..., 0.3]     ← 64 numbers
Head 4 output: [0.4, 0.3, ..., 0.5]     ← 64 numbers
Head 5 output: [0.2, 0.6, ..., 0.1]     ← 64 numbers
Head 6 output: [0.7, 0.4, ..., 0.2]     ← 64 numbers
Head 7 output: [0.3, 0.5, ..., 0.8]     ← 64 numbers
Head 8 output: [0.6, 0.1, ..., 0.4]     ← 64 numbers
                                          ────────────
Concatenated:  [0.3, 0.1, ..., 0.4, 0.5, 0.2, ..., 0.4]  ← 512 numbers!
               └─── head1 ───┘└─── head2 ───┘    └─ head8 ─┘
```

### Step 4: Final Linear Projection

The concatenated result goes through one more learned weight matrix (W_O) that mixes information across heads:

```
Concatenated (512) ──→ × W_O ──→ Final output (512)

This final projection lets the model combine insights from ALL heads
into one unified representation.
```

---

## The Complete Picture

```
                     Input embedding for each word
                              │
               ┌──────────────┼──────────────┐
               │              │              │
          ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
          │  × W_Q  │   │  × W_K  │   │  × W_V  │
          └────┬────┘   └────┬────┘   └────┬────┘
               │              │              │
          ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
          │  Split   │   │  Split   │   │  Split   │
          │into heads│   │into heads│   │into heads│
          └────┬────┘   └────┬────┘   └────┬────┘
               │              │              │
    ┌──────────┼──────────────┼──────────────┼──────────┐
    │          │              │              │          │
    │    ┌─────┴─────┐  ┌────┴────┐   ┌────┴────┐     │
    │    │  Q₁ K₁ V₁ │  │ Q₂ K₂ V₂│  │ Q₃ K₃ V₃│ ... │
    │    │ Attention  │  │Attention │  │Attention │     │
    │    │  Head 1    │  │ Head 2   │  │ Head 3   │     │
    │    └─────┬─────┘  └────┬────┘   └────┬────┘     │
    │          │              │              │          │
    │          └──────────────┼──────────────┘          │
    │                         │                         │
    │                  ┌──────┴──────┐                  │
    │                  │ Concatenate  │                  │
    │                  └──────┬──────┘                  │
    │                         │                         │
    └─────────────────────────┼─────────────────────────┘
                              │
                       ┌──────┴──────┐
                       │   × W_O     │   (final projection)
                       └──────┬──────┘
                              │
                     Multi-Head Output
```

---

## The Formula

The formula for multi-head attention is:

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) × W_O

where each head is:
  headᵢ = Attention(Q × W_Qᵢ, K × W_Kᵢ, V × W_Vᵢ)
```

Breaking this down:
- `h` = number of heads (typically 8 or 16)
- `W_Qᵢ, W_Kᵢ, W_Vᵢ` = learned weight matrices for head `i`
- `W_O` = learned output projection matrix
- Each head uses the same attention formula from the [previous section](./attention-mechanisms.md)

---

## Why Does This Actually Work?

Different heads naturally learn to specialize in different tasks. Research has shown that in trained transformers:

```
What different heads learn (examples from real models):

Head Type          What It Focuses On                 Example
─────────────      ───────────────────                ──────────────────
Syntactic head     Grammar relationships              "cat" → "The" (determiner)
Positional head    Nearby words                       "cat" → "sat" (next word)
Semantic head      Meaning/topic                      "cat" → "tired" (descriptor)
Coreference head   Pronoun resolution                 "it" → "cat" (same entity)
Separator head     Sentence boundaries                "." → "[SEP]" (structure)
```

**Nobody programs these specializations.** The model discovers them during training because having diverse perspectives leads to better performance. It's like natural selection -- the heads that specialize survive because they're useful.

---

## Common Configurations

Different transformer models use different numbers of heads:

```
Model              Embedding Size    Num Heads    Head Size
─────────────      ──────────────    ─────────    ─────────
GPT-2 Small        768               12           64
GPT-2 Medium       1024              16           64
GPT-3              12288             96           128
BERT Base          768               12           64
BERT Large         1024              16           64
```

**Pattern:** Head size is typically 64 or 128. The number of heads = embedding size / head size.

**Important:** More heads doesn't always mean better. What matters is having enough heads to capture the different types of relationships in the data, and enough dimensions per head to represent those relationships well.

---

## Computational Cost

A common question: "Doesn't running 8 attention computations cost 8x more?"

Surprisingly, **no!** Here's why:

```
Single-head attention with d_model = 512:
  Q, K, V each are: 512-dimensional
  Computation per attention: proportional to 512 × 512 = 262,144

Multi-head attention with 8 heads, d_model = 512:
  Q, K, V per head: 64-dimensional (512 ÷ 8 = 64)
  Computation per head: proportional to 64 × 64 = 4,096
  Total across 8 heads: 8 × 4,096 = 32,768

  Plus the output projection: ~262,144

  Total: ~294,912 ≈ similar cost to single-head!
```

The trick: by splitting the dimensions across heads, the total computation stays roughly the same. And because the heads are independent, they can run **in parallel** on a GPU -- so it's not even slower in practice.

---

## Key Takeaways

1. **Multi-head attention = multiple attention operations in parallel**, each learning different patterns
2. **Each head** gets a smaller slice of the embedding and runs attention independently
3. **Concatenation + projection** combines all heads' findings back into one vector
4. **Different heads specialize** naturally during training (grammar, meaning, position, etc.)
5. **No extra cost** -- splitting dimensions means multi-head costs about the same as single-head
6. Typical models use **8-96 heads** with **64-128 dimensions per head**

---

## Prerequisites

Before reading this, you should understand:
- [Attention Mechanisms](./attention-mechanisms.md) -- the foundation for this section
- Vectors and matrices (from [Neural Network Fundamentals](../../00-neural-networks/fundamentals/04_neural_network_layers.ipynb))

## Further Reading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) -- Section 3.2
- [A Multiscale Visualization of Attention](https://arxiv.org/abs/1906.05714) -- what heads actually learn
- [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) -- head pruning research

---

[Previous: Attention Mechanisms](./attention-mechanisms.md) | [Back to Architecture Overview](./README.md) | [Next: Positional Encoding](./positional-encoding.md)
