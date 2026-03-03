# Multi-Head Attention

## The Mystery Worth Solving

You know how, when you read a sentence, you're simultaneously tracking grammar, meaning, who's doing what, and where things are happening — all at the same time?

That's not one skill. It's several running in parallel.

A single attention head can only look for one type of relationship at a time. So how do transformer models keep track of all those different things simultaneously?

The answer is the most elegant design decision in the whole architecture: run several attention operations in parallel, each one looking for something different, then combine their findings at the end.

That's multi-head attention.

---

**Before you start, you need to know:**
- What attention is (Q, K, V, dot product, softmax) — covered in [Attention Mechanisms](./attention-mechanisms.md)
- Vectors and matrices — covered in [Neural Network Fundamentals](../../00-neural-networks/fundamentals/04_neural_network_layers.ipynb)

---

## Why Not Just One Attention Head?

In [attention mechanisms](./attention-mechanisms.md), we learned how each word can look at every other word and decide what's relevant. But here's the thing: **a single attention head can only focus on one type of relationship at a time.**

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

**What this analogy gets right:** Each student reads the same book (same input) but produces a different analysis (different attention patterns). Their findings are genuinely combined at the end into one report (concatenate + W_O projection). The specialization emerges naturally — no one told student 1 to focus on characters. The model discovers useful specializations the same way.

**Where this analogy breaks down:** Human students can deliberately choose their specialization and coordinate. Attention heads can't. The specialization in a trained model emerges entirely from gradient descent finding useful patterns — each head lands on a specialty because it makes the loss go down, not because of any explicit coordination.

---

## How It Works: Step by Step

### Step 1: Each Head Gets Its Own Projection

Each head has its own learned weight matrices W_Qᵢ, W_Kᵢ, W_Vᵢ that project the full word embedding into a smaller Q, K, V space for that head:

```
Example: Word embedding with 512 numbers, split across 8 heads

Full embedding for "cat": [0.2, -0.1, 0.5, 0.3, ..., 0.7, -0.4]
                           ←────────── 512 numbers ──────────────→

Each head projects to 64-dimensional Q, K, V (512 ÷ 8 = 64):

Head 1: Q₁, K₁, V₁  ← 64 numbers each, learned to find one type of pattern
Head 2: Q₂, K₂, V₂  ← 64 numbers each, learned to find a different pattern
Head 3: Q₃, K₃, V₃  ← 64 numbers each
...
Head 8: Q₈, K₈, V₈  ← 64 numbers each
```

### Step 2: Each Head Does Independent Attention

Each head performs the full attention computation from the previous section — but in its own smaller space:

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

Each head sees the same words but learns to find DIFFERENT patterns!
```

### Step 3: Concatenate All Heads

After each head computes its output, we **concatenate** (join together) all the results:

```
Head 1 output: [0.3, 0.1, ..., 0.4]     ← 64 numbers
Head 2 output: [0.5, 0.2, ..., 0.6]     ← 64 numbers
...
Head 8 output: [0.6, 0.1, ..., 0.4]     ← 64 numbers
                                          ────────────
Concatenated:  [0.3, 0.1, ..., 0.4, 0.5, 0.2, ..., 0.4]  ← 512 numbers!
               └─── head1 ───┘└─── head2 ───┘    └─ head8 ─┘
```

### Step 4: Final Linear Projection (W_O)

The concatenated result goes through one more learned weight matrix (W_O) that mixes information **across** all heads:

```
Concatenated (512) ──→ × W_O ──→ Final output (512)

This final projection lets the model combine insights from ALL heads
into one unified representation.
```

Without W_O, the information from different heads would stay separate — head 1's findings at positions 0–63 could never combine with head 2's findings at positions 64–127. W_O is the meeting where all heads share their insights.

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
   ┌──────────┬┴──────────────┴──────────────┤
   │          │                              │
   │    ┌─────┴─────┐  ┌──────────┐   ┌─────┴─────┐
   │    │ Q₁ K₁ V₁  │  │ Q₂ K₂ V₂│   │ Q₃ K₃ V₃  │  ...
   │    │ Attention  │  │Attention │   │ Attention  │
   │    │  Head 1    │  │ Head 2   │   │ Head 3     │
   │    └─────┬─────┘  └────┬─────┘   └─────┬─────┘
   │          │              │               │
   │          └──────────────┼───────────────┘
   │                         │
   │                  ┌──────┴──────┐
   │                  │ Concatenate  │
   │                  └──────┬──────┘
   │                         │
   └─────────────────────────┼──────────────────────
                              │
                       ┌──────┴──────┐
                       │   × W_O     │   (final projection — where heads talk)
                       └──────┬──────┘
                              │
                     Multi-Head Output
```

---

## The Formula

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) × W_O

where each head is:
  headᵢ = Attention(Q × W_Qᵢ, K × W_Kᵢ, V × W_Vᵢ)
```

- `h` = number of heads (typically 8 or 16)
- `W_Qᵢ, W_Kᵢ, W_Vᵢ` = learned weight matrices for head `i`
- `W_O` = learned output projection matrix
- Each head uses the same attention formula from [Attention Mechanisms](./attention-mechanisms.md)

---

## Why Do Heads Specialize?

Nobody programs the specialization. It emerges from training. Research has shown that in trained transformers, heads tend to learn specific jobs:

```
Head Type          What It Focuses On                 Example
─────────────      ───────────────────                ──────────────────
Syntactic head     Grammar relationships              "cat" → "The" (determiner)
Positional head    Nearby words                       "cat" → "sat" (next word)
Semantic head      Meaning/topic                      "cat" → "tired" (descriptor)
Coreference head   Pronoun resolution                 "it" → "cat" (same entity)
Separator head     Sentence boundaries                "." → "[SEP]" (structure)
```

It's like natural selection — the heads that find genuinely useful patterns survive because they help reduce the loss. The model ends up with a diverse team of specialists, not one generalist doing everything poorly.

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

---

## Does Multi-Head Cost More?

Surprisingly, no. Here's the key insight:

```
Single-head attention with d_model = 512:
  Q, K, V each are 512-dimensional
  Attention computation cost ∝ 512 × 512

Multi-head attention with 8 heads, d_model = 512:
  Q, K, V per head: 64-dimensional (512 ÷ 8 = 64)
  Cost per head ∝ 64 × 64 = 4,096
  Total across 8 heads: 8 × 4,096 = 32,768

  Plus the output projection W_O adds similar cost back.

  Total ≈ similar cost to single-head!
```

The trick: by splitting the dimensions across heads, the total computation stays roughly the same. And because the heads are independent, they run **in parallel** on a GPU — so it's not even slower in practice.

---

## Quick Check — can you answer these?

- What is the job of W_O? What would happen if you left it out?
- Why does running 8 heads not cost 8 times more than running 1 head?
- How does a head "learn" to focus on grammar vs. meaning? Does someone program it?

If you can't answer one, go back and re-read that part. That is completely normal.

---

## Victory Lap

You now understand multi-head attention — the exact mechanism that lets GPT-4 simultaneously understand grammar, semantics, pronoun reference, and long-range dependencies in a single pass. Every major transformer model from BERT to LLaMA uses this. The "parallel heads with different specializations" insight is what makes transformers so much more expressive than a single-head model at the same parameter count. The architecture you just learned is the core of every language model that exists today.

---

Ready to go deeper? → [Multi-Head Attention — Interview Deep-Dive](./multi-head-attention-interview.md)

---

**Further Reading**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — Section 3.2
- [A Multiscale Visualization of Attention](https://arxiv.org/abs/1906.05714) — what heads actually learn
- [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) — head pruning research

---

[Previous: Attention Mechanisms](./attention-mechanisms.md) | [Back to Architecture Overview](./README.md) | [Next: Positional Encoding](./positional-encoding.md)
