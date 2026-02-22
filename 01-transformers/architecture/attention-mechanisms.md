# Attention Mechanisms

## What Problem Does Attention Solve?

Imagine you're reading this sentence:

> "The **cat** sat on the mat because **it** was tired."

When you read the word "it", your brain instantly knows "it" refers to "the cat" -- not "the mat". You **pay attention** to the right words to understand meaning.

Before transformers, neural networks processed words one at a time, left to right, like reading through a tiny keyhole. By the time the network reached "it", it might have forgotten about "the cat" way back at the beginning. This was a major problem with older models (RNNs/LSTMs).

**Attention** is the mechanism that lets a model look at **all words at once** and decide which ones are important for understanding each word.

```
Processing the word "it":

  The    cat    sat    on    the    mat   because   it    was   tired
  [0.05] [0.70] [0.02] [0.01] [0.02] [0.05] [0.03] [---] [0.05] [0.07]
           ^
           "it" pays the most attention to "cat"!
```

---

## The Core Idea: Questions, Keys, and Values (Q, K, V)

This is the heart of attention, and it uses three concepts: **Query (Q)**, **Key (K)**, and **Value (V)**. Let's build intuition with a real-world analogy.

### The Library Analogy

Imagine you walk into a library looking for a book about dinosaurs.

| Concept | Library Analogy | In a Transformer |
|---------|----------------|-------------------|
| **Query (Q)** | Your question: "I want books about dinosaurs" | What the current word is *looking for* |
| **Key (K)** | The label on each bookshelf: "Science", "History", "Fiction" | What each word *advertises* about itself |
| **Value (V)** | The actual books on each shelf | The actual information each word carries |

Here's how it works:

1. **You ask your question** (Query): "I want dinosaurs"
2. **You compare your question to every shelf label** (Keys): "Does 'Science' match? Does 'History' match?"
3. **The better the match, the more you look at those books** (Values): Science shelf = great match, so you take lots from it

```
Your Query: "dinosaurs"
                                          Attention
Shelf Key         Match Score            Weight        Books (Value) You Take
─────────────     ───────────           ──────────     ─────────────────────
"Science"         HIGH match      →     0.60      →   Take 60% from Science
"History"         Medium match    →     0.25      →   Take 25% from History
"Fiction"         Low match       →     0.10      →   Take 10% from Fiction
"Cooking"         No match        →     0.05      →   Take 5% from Cooking
                                        ────
                                        1.00 (always sums to 1)
```

The final answer is a **weighted blend** of all the values, where the weights come from how well each key matches the query.

---

## Self-Attention: Words Talking to Each Other

**Self-attention** is when words in the **same sentence** pay attention to each other. Every word gets to "ask a question" about every other word.

Let's trace through a simple example:

### Example: "The cat sat"

Each word creates its own Q, K, and V vectors (we'll see how in a moment):

```
Step 1: Each word creates Q, K, V vectors
──────────────────────────────────────────

 Word       Query (Q)         Key (K)          Value (V)
 ─────      ──────────        ──────────       ──────────
 "The"      Q_the             K_the            V_the
 "cat"      Q_cat             K_cat            V_cat
 "sat"      Q_sat             K_sat            V_sat


Step 2: Each word compares its Query to ALL Keys
────────────────────────────────────────────────

For the word "sat":

  Q_sat  ·  K_the  =  score_1  (how relevant is "The" to "sat"?)
  Q_sat  ·  K_cat  =  score_2  (how relevant is "cat" to "sat"?)
  Q_sat  ·  K_sat  =  score_3  (how relevant is "sat" to itself?)

  The "·" means dot product (we'll explain below!)


Step 3: Convert scores to weights (using softmax)
──────────────────────────────────────────────────

  scores:  [1.2,  3.5,  0.8]
                   ↓ softmax
  weights: [0.15, 0.72, 0.13]   ← these sum to 1.0


Step 4: Weighted sum of Values
──────────────────────────────

  output_sat = 0.15 × V_the  +  0.72 × V_cat  +  0.13 × V_sat

  → "sat" now carries information mostly from "cat"
    (makes sense: WHO sat? The cat!)
```

---

## The Math (Explained Simply)

### Step-by-Step: Scaled Dot-Product Attention

The full attention formula is:

```
                        Q × K^T
Attention(Q, K, V) = softmax( ─────── ) × V
                         √d_k
```

Don't panic! Let's break down every piece:

### 1. Creating Q, K, V

Each word starts as a vector (a list of numbers) called an **embedding**. We multiply it by three different weight matrices to get Q, K, and V:

```
           ┌─────────┐
word  ──→  │ × W_Q   │ ──→  Query  (Q)
embedding  │ × W_K   │ ──→  Key    (K)
           │ × W_V   │ ──→  Value  (V)
           └─────────┘

W_Q, W_K, W_V are weight matrices that the model LEARNS during training.
They learn to extract useful "question", "label", and "content" information
from each word.
```

**What's a vector?** Think of it as a list of numbers that describes something. Like how GPS coordinates [latitude, longitude] describe a location, a word vector might be [0.2, -0.5, 0.8, ...] with hundreds of numbers that describe the word's meaning.

**What's a matrix?** A grid of numbers. Multiplying a vector by a matrix transforms it -- like putting on different colored glasses that highlight different features of the word.

### 2. Dot Product: Measuring Similarity

The **dot product** measures how similar two vectors are. Think of it as a "compatibility score."

```
Simple dot product example:

  Q = [1, 0, 1]     K = [1, 0, 1]     Q · K = 1×1 + 0×0 + 1×1 = 2  (HIGH!)
  Q = [1, 0, 1]     K = [0, 1, 0]     Q · K = 1×0 + 0×1 + 1×0 = 0  (LOW!)

  Higher dot product = vectors point in similar direction = MORE relevant
```

When we compute Q × K^T ("K transpose"), we're computing dot products between **every query** and **every key** at once:

```
        K_the  K_cat  K_sat
       ┌─────────────────────┐
Q_the  │ 1.2    0.3    0.1   │   ← How much "The" attends to each word
Q_cat  │ 0.2    1.5    0.8   │   ← How much "cat" attends to each word
Q_sat  │ 0.1    2.1    0.5   │   ← How much "sat" attends to each word
       └─────────────────────┘
       This is the "attention score matrix"
```

### 3. Why Divide by √d_k? (The "Scaled" Part)

The `d_k` is the size of the key vectors (e.g., 64 numbers long). We divide by √d_k (e.g., √64 = 8) to keep the numbers from getting too big.

**Why does this matter?** Imagine a classroom where students vote on what to eat for lunch:

```
Without scaling (scores get too extreme):
  Pizza: 150.0    Salad: 2.0    Pasta: 3.0
  → softmax: [0.999, 0.0001, 0.0001]
  → Almost ALL attention goes to pizza, nothing else gets considered

With scaling (scores stay reasonable):
  Pizza: 5.0    Salad: 2.0    Pasta: 3.0
  → softmax: [0.65, 0.09, 0.26]
  → Pizza still wins, but other options are still considered
```

Without scaling, the softmax becomes too "sharp" and the model can only focus on one thing. Scaling keeps the attention "soft" so it can blend information from multiple sources.

### 4. Softmax: Turning Scores into Weights

**Softmax** converts any list of numbers into probabilities (positive numbers that sum to 1):

```
Raw scores:     [2.0,  1.0,  0.5]
                  ↓ softmax
Weights:        [0.59, 0.24, 0.17]    ← always positive, always sum to 1

How? For each score x:
  softmax(x) = e^x / (sum of all e^scores)

Don't worry about the formula -- just know:
  • Bigger scores → bigger weights
  • All weights are between 0 and 1
  • They always add up to exactly 1 (like percentages)
```

### 5. Multiply by V: The Weighted Sum

Finally, we multiply the attention weights by the Value vectors:

```
weights × V = 0.59 × V_the + 0.24 × V_cat + 0.17 × V_sat

This blends the value information, weighted by relevance.
```

---

## Self-Attention vs. Cross-Attention

There are two flavors of attention used in transformers:

### Self-Attention

**All Q, K, V come from the same sequence.** Words attend to other words in the same sentence.

```
Input: "The cat sat on the mat"

Q, K, V all come from → "The cat sat on the mat"

Each word asks: "Which OTHER words in MY sentence are important for understanding ME?"
```

**Used in:** Every transformer layer (both encoder and decoder).

### Cross-Attention

**Q comes from one sequence, K and V come from a different sequence.** This lets one sequence "look at" another.

```
Decoder input: "Le chat"          ← generating French translation
Encoder input: "The cat sat"      ← the original English sentence

Q comes from    → "Le chat"       (the decoder asks questions)
K, V come from  → "The cat sat"   (the encoder provides answers)

"Le" asks: "Which English words should I pay attention to?"
  → Pays most attention to "The"

"chat" asks: "Which English words should I pay attention to?"
  → Pays most attention to "cat"
```

**Used in:** The decoder in encoder-decoder transformers (like translation models, T5).

```
┌─────────────────────────────────────────────────────────────┐
│                    Comparison Table                          │
├──────────────────┬─────────────────┬────────────────────────┤
│                  │ Self-Attention   │ Cross-Attention        │
├──────────────────┼─────────────────┼────────────────────────┤
│ Q comes from     │ Same sequence    │ Decoder sequence       │
│ K, V come from   │ Same sequence    │ Encoder sequence       │
│ Purpose          │ Understand       │ Connect two            │
│                  │ context within   │ sequences (e.g.,       │
│                  │ a sentence       │ translation)           │
│ Example use      │ "it" → "cat"    │ "chat" → "cat"         │
└──────────────────┴─────────────────┴────────────────────────┘
```

---

## Visualizing Attention

Attention weights can be visualized as a heatmap, showing what each word "looks at":

```
Processing: "The cat sat on the mat because it was tired"

Attention weights for the word "it":

         The   cat   sat   on   the   mat  because  it   was  tired
          │     │     │     │     │     │     │      │     │     │
          ▼     ▼     ▼     ▼     ▼     ▼     ▼      ▼     ▼     ▼
  "it" → [░░░] [███] [░░░] [░░░] [░░░] [░░░] [░░░] [░▒░] [░░░] [░░░]
                 ▲
                 │
          Strongest attention!
          The model learned that "it" = "cat"

  ░░░ = low attention (light)
  ▒▒▒ = medium attention
  ███ = high attention (dark)
```

This is a powerful debugging tool -- if the model makes a mistake, you can look at the attention weights to understand what it was "looking at."

---

## Putting It All Together

Here's the complete flow of attention in one diagram:

```
                    Input: "The cat sat"
                           │
                    ┌──────┴──────┐
                    │  Embeddings  │   (words → vectors of numbers)
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         ┌────┴────┐ ┌────┴────┐ ┌────┴────┐
         │  × W_Q  │ │  × W_K  │ │  × W_V  │   (learned weight matrices)
         └────┬────┘ └────┬────┘ └────┬────┘
              │            │            │
              Q            K            V
              │            │            │
              │     ┌──────┴──────┐     │
              └────►│  Q × K^T    │     │     (dot products = similarity scores)
                    └──────┬──────┘     │
                           │            │
                    ┌──────┴──────┐     │
                    │  ÷ √d_k    │     │     (scale down)
                    └──────┬──────┘     │
                           │            │
                    ┌──────┴──────┐     │
                    │   Softmax   │     │     (convert to probabilities)
                    └──────┬──────┘     │
                           │            │
                    ┌──────┴──────┐     │
                    │  × V        │◄────┘     (weighted sum of values)
                    └──────┬──────┘
                           │
                      Output vectors
                (each word now carries context
                 from the words it attended to)
```

---

## Key Takeaways

1. **Attention lets every word look at every other word** -- no more reading through a keyhole
2. **Q, K, V** are like asking a question (Q), checking shelf labels (K), and grabbing books (V)
3. **Dot product** measures similarity between queries and keys
4. **Softmax** turns similarity scores into weights that sum to 1
5. **Scaling by √d_k** prevents attention from becoming too extreme
6. **Self-attention** = words attending to their own sentence
7. **Cross-attention** = one sequence attending to a different sequence

---

## Prerequisites

Before reading this, you should understand:
- Vectors and matrices (from [Neural Network Fundamentals](../../00-neural-networks/fundamentals/04_neural_network_layers.ipynb))
- How neural networks learn weights (from [Backpropagation](../../00-neural-networks/fundamentals/07_backpropagation.ipynb))

## Further Reading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) -- the paper that started it all
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) -- excellent visual guide
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) -- comprehensive overview

---

[Back to Architecture Overview](./README.md) | [Next: Multi-Head Attention](./multi-head-attention.md)
