# Attention Mechanisms

## The Mystery Worth Solving

Here's something that should make you curious: a model trained on nothing but text — no images, no game boards, no code — somehow learns to resolve pronouns, translate languages, write working programs, and pass standardized medical exams.

It doesn't do this because someone told it the rules. It figures them out by learning which words are relevant to which other words.

That one idea — *which words are relevant to which* — is attention. And it's simpler than you might expect.

---

**Before you start, you need to know:**
- What a vector is (a list of numbers) — covered in [Neural Network Fundamentals](../../00-neural-networks/fundamentals/04_neural_network_layers.ipynb)
- What a weight matrix does (transforms one vector into another) — covered in [Backpropagation](../../00-neural-networks/fundamentals/07_backpropagation.ipynb)
- What softmax does (converts numbers into probabilities) — covered in the same notebook

---

## What Problem Does Attention Solve?

Imagine you're reading this sentence:

> "The **cat** sat on the mat because **it** was tired."

When you read the word "it", your brain instantly knows "it" refers to "the cat" — not "the mat". You **pay attention** to the right words to understand meaning.

Before transformers, neural networks processed words one at a time, left to right, like reading through a tiny keyhole. By the time the network reached "it", it might have forgotten about "the cat" way back at the beginning. This was a major problem with older models.

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

This is the heart of attention. It uses three concepts: **Query (Q)**, **Key (K)**, and **Value (V)**. Let's build intuition with a real-world analogy.

🧒 **Kid-Friendly Explanation:** Imagine attention like three friends working together at school! Query is like the **curious friend** who asks questions ("What's fun to do at recess?"). Key is like the **helpful friend** who holds up signs about what they know ("I know about soccer!", "I know about art!"). Value is like the **knowledgeable friend** who actually has the good stuff to share (soccer tips, art supplies, etc.). The curious friend looks at all the signs, picks the most interesting ones, and gets the best information from the knowledgeable friend!

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
Shelf Key         Match Score         Weight        Books (Value) You Take
─────────────     ───────────        ──────────     ─────────────────────
"Science"         HIGH match    →    0.60      →   Take 60% from Science
"History"         Medium match  →    0.25      →   Take 25% from History
"Fiction"         Low match     →    0.10      →   Take 10% from Fiction
"Cooking"         No match      →    0.05      →   Take 5% from Cooking
                                     ────
                                     1.00 (always sums to 1)
```

The final answer is a **weighted blend** of all the values, where the weights come from how well each key matches the query.

**What this analogy gets right:** You always end up with a blend from multiple shelves — not just the single best match. Even the "Cooking" shelf contributes 5%. That's exactly how attention works: every word contributes *something* to the output, just weighted by how relevant it is. No position is ever completely ignored.

**Where this analogy breaks down:** In a library, the shelf labels (Keys) and the books (Values) are two separate things someone else created. In attention, each word creates its own Key and Value at the same time — they're both projections of the same word vector. A word is simultaneously labeling itself for others to find *and* deciding what information it carries. Real library books don't label themselves.

---

## Self-Attention: Words Talking to Each Other

**Self-attention** is when words in the **same sentence** pay attention to each other. Every word gets to "ask a question" about every other word.

Let's trace through a simple example:

### Example: "The cat sat"

Each word creates its own Q, K, and V vectors:

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

  The "·" means dot product — a way to measure how similar two vectors are.


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

## How Q, K, V Are Created

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

🧒 **Kid-Friendly Explanation:** A vector is like a recipe card! Just like a recipe for chocolate chip cookies has specific amounts (2 cups flour, 1 cup sugar, 3 eggs), a word has specific numbers that describe what it means. The word "happy" might be described by numbers like [0.8, 0.2, 0.9] where the first number says "how positive is this word?" and so on. Every word gets its own special recipe card of numbers!

**What's a matrix?** A grid of numbers. Multiplying a vector by a matrix transforms it — like putting on different colored glasses that highlight different features of the word.

🧒 **Kid-Friendly Explanation:** A matrix is like a magic filter! Imagine you have a photo and you can put different colored glasses on to see different things better. Red glasses might make all the red things pop out, blue glasses make blue things stand out. A matrix does the same thing to our word recipe cards - it's like special glasses that help us see certain "flavors" of the word's meaning more clearly. When we "multiply" a word by a matrix, we're putting those special glasses on to transform how we see that word!

---

## Measuring Similarity: Dot Product

The **dot product** measures how similar two vectors are. Think of it as a "compatibility score."

🧒 **Kid-Friendly Explanation:** The dot product is like checking how much two people have in common! Imagine you and your friend both make lists of your favorite things (ice cream flavors, hobbies, etc.). To see how similar you are, you compare each item on your lists and give points for matches. If you both love chocolate ice cream, that's a point! If you both hate brussels sprouts, that's another point! The more things you agree on, the higher your "friendship score" (dot product) gets. Words work the same way - they compare their "preference lists" to see how similar they are!

```
Simple dot product example:

  Q = [1, 0, 1]     K = [1, 0, 1]     Q · K = 1×1 + 0×0 + 1×1 = 2  (HIGH!)
  Q = [1, 0, 1]     K = [0, 1, 0]     Q · K = 1×0 + 0×1 + 1×0 = 0  (LOW!)

  Higher dot product = vectors point in similar direction = MORE relevant
```

When we compute the dot product between every query and every key, we get a score matrix:

```
        K_the  K_cat  K_sat
       ┌─────────────────────┐
Q_the  │ 1.2    0.3    0.1   │   ← How much "The" attends to each word
Q_cat  │ 0.2    1.5    0.8   │   ← How much "cat" attends to each word
Q_sat  │ 0.1    2.1    0.5   │   ← How much "sat" attends to each word
       └─────────────────────┘
       This is the "attention score matrix"
```

---

## Why Divide by √d_k? (The Scaling Step)

After computing the scores, we divide by a number called √d_k before feeding them into softmax.

Here's an analogy: imagine a classroom where students vote on what to eat for lunch.

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

Without scaling, the softmax becomes too "sharp" and the model can only focus on one thing. Dividing by √d_k (the square root of the key vector size) keeps scores in a healthy range so the model can blend information from multiple sources.

---

## Softmax: Turning Scores into Weights

**Softmax** converts any list of numbers into probabilities — positive numbers that always add up to 1:

```
Raw scores:     [2.0,  1.0,  0.5]
                  ↓ softmax
Weights:        [0.59, 0.24, 0.17]    ← always positive, always sum to 1

Key properties:
  • Bigger scores → bigger weights
  • All weights are between 0 and 1
  • They always add up to exactly 1 (like percentages)
```

---

## The Final Step: Weighted Sum

Finally, we multiply the attention weights by the Value vectors:

```
weights × V = 0.59 × V_the + 0.24 × V_cat + 0.17 × V_sat

This blends the value information, weighted by relevance.
```

Each word's output is a mixture of information from all the words it paid attention to. Words that got high attention weights contribute more to the output.

---

## Worked Example with Real Numbers

Let's trace attention all the way through with exact arithmetic. We use 2-dimensional vectors so the math fits on the page.

We'll process the sequence "cat sat" with small vectors.

**Step 1: Word embeddings**

```
"cat" embedding: x_cat = [1.0,  0.0]
"sat" embedding: x_sat = [0.0,  1.0]
```

**Step 2: Weight matrices**

```
W_Q = [[1, 0],   W_K = [[0, 1],   W_V = [[1, 1],
       [0, 1]]          [1, 0]]          [0, 1]]
```

**Step 3: Compute Q, K, V for each word**

```
Q_cat = x_cat @ W_Q = [1, 0] @ [[1,0],[0,1]] = [1.0, 0.0]
K_cat = x_cat @ W_K = [1, 0] @ [[0,1],[1,0]] = [0.0, 1.0]
V_cat = x_cat @ W_V = [1, 0] @ [[1,1],[0,1]] = [1.0, 1.0]

Q_sat = x_sat @ W_Q = [0, 1] @ [[1,0],[0,1]] = [0.0, 1.0]
K_sat = x_sat @ W_K = [0, 1] @ [[0,1],[1,0]] = [1.0, 0.0]
V_sat = x_sat @ W_V = [0, 1] @ [[1,1],[0,1]] = [0.0, 1.0]
```

**Step 4: Compute attention scores for "cat"**

```
score(cat→cat) = Q_cat · K_cat / √2
               = [1.0, 0.0] · [0.0, 1.0] / 1.414
               = 0.0 / 1.414 = 0.0

score(cat→sat) = Q_cat · K_sat / √2
               = [1.0, 0.0] · [1.0, 0.0] / 1.414
               = 1.0 / 1.414 = 0.707
```

**Step 5: Apply softmax**

```
exp(0.0)   = 1.000
exp(0.707) = 2.028
Sum        = 3.028

weight(cat→cat) = 1.000 / 3.028 = 0.330
weight(cat→sat) = 2.028 / 3.028 = 0.670
```

"cat" pays 33% attention to itself and 67% attention to "sat".

**Step 6: Weighted sum of Values**

```
output_cat = 0.330 × [1.0, 1.0]  +  0.670 × [0.0, 1.0]
           = [0.330, 0.330]  +  [0.000, 0.670]
           = [0.330, 1.000]
```

The output for "cat" is [0.330, 1.000]. It now carries blended information from both words, with more weight on "sat" because Q_cat and K_sat had a higher dot product. Every number came from explicit arithmetic — nothing magic.

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

**Used in:** The decoder in encoder-decoder transformers (like translation models).

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

This is a powerful tool — if the model makes a mistake, you can look at the attention weights to understand what it was "looking at."

---

## Putting It All Together

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

## Quick Check — can you answer these?

- In your own words: what problem does attention solve? Why did older models struggle with long sentences?
- What do Q, K, and V stand for, and what role does each one play?
- Why do we divide the attention scores by √d_k before applying softmax?

If you can't answer one, go back and re-read that part. That is completely normal.

---

## Victory Lap

You just understood the core operation inside every transformer in existence. GPT-4, Claude, Gemini, LLaMA, GitHub Copilot — they all run this computation millions of times per forward pass. The "magic" of language models is this: learned Q, K, V weight matrices that have seen enough text to know which words are relevant to which. Everything else in the architecture is in service of this one operation. The hard part is behind you.

---

Ready to go deeper? → [Attention Mechanisms — Interview Deep-Dive](./attention-mechanisms-interview.md)

---

**Further Reading**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — the paper that started it all
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — excellent visual guide
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — comprehensive overview

---

[Back to Architecture Overview](./README.md) | [Next: Multi-Head Attention](./multi-head-attention.md)
