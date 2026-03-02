# Attention Mechanisms

## The Mystery Worth Solving

Here's something that should make you curious: a model trained on nothing but text — no images, no game boards, no code — somehow learns to resolve pronouns, translate languages, write working programs, and pass standardized medical exams.

It doesn't do this because someone told it the rules. It figures them out by learning which words are relevant to which other words.

That one idea — *which words are relevant to which* — is attention. And it's simpler than you might expect.

---

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

**What this analogy gets right:** You always end up with a blend from multiple shelves — not just the single best match. Even the "Cooking" shelf contributes 5%. That's exactly how attention works: every word contributes *something* to the output, just weighted by how relevant it is. No position is ever completely ignored.

**Where this analogy breaks down:** In a library, the shelf labels (Keys) and the books (Values) are two separate things someone else created. In attention, each word creates its own Key and Value at the same time — they're both projections of the same word vector. A word is simultaneously labeling itself for others to find *and* deciding what information it carries. Real library books don't label themselves.

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

Fair warning: this is the most mathematical part of attention. Even researchers sit with this one for a while. The key thing to grab is the punchline — variance control — and the rest will make sense from there.

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

**The formal reason — variance proof:**

Think of the query vector q and key vector k as lists of random numbers. Say each number is drawn independently from a normal distribution with mean 0 and variance 1 (written N(0,1)).

The dot product q·k = q₁k₁ + q₂k₂ + ... + q_{d_k}k_{d_k}. Each term qᵢkᵢ has variance 1 (product of two independent N(0,1) variables). You sum d_k of these terms, so the total variance is d_k. The standard deviation is √d_k.

Dividing by √d_k normalizes the variance back to 1: Var(q·k / √d_k) = d_k / d_k = 1.

**Why unit variance matters:**

Softmax saturates when its inputs are large. "Saturates" means it pushes nearly all weight to one element and near-zero weight everywhere else. At saturation, gradients vanish — the model can't learn.

For a vector x where one entry is much larger than the others, softmax(x) ≈ [1, 0, 0, ...]. The gradient of softmax at this point is approximately zero everywhere except that one entry. When training signals become zero, weights stop updating — training stalls.

Dividing by √d_k keeps dot products near unit scale, where softmax gradients are healthy. Dividing by d_k would over-correct: variance would become 1/d_k → too small → attention becomes too uniform, and the model can't differentiate between relevant and irrelevant words.

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

### Victory Lap

If that clicked — you just understood the core operation inside every transformer in existence. GPT-4, Claude, Gemini, LLaMA, GitHub Copilot — they all run this computation millions of times per forward pass. The "magic" of language models is this: learned Q, K, V weight matrices that have seen enough text to know which words are relevant to which. Everything else in the architecture is in service of this one operation. The hard part is behind you.

---

## Worked Example: Step-by-Step with Real Numbers

Let's trace attention all the way through with exact arithmetic. We use 2-dimensional vectors so the math fits on the page.

We'll process the sequence "cat sat" with d_k = 2.

**Step 1: Word embeddings (pretend these were learned)**

```
"cat" embedding: x_cat = [1.0,  0.0]
"sat" embedding: x_sat = [0.0,  1.0]
```

**Step 2: Weight matrices (pretend these were learned)**

```
W_Q = [[1, 0],   W_K = [[0, 1],   W_V = [[1, 1],
       [0, 1]]          [1, 0]]          [0, 1]]
```

**Step 3: Compute Q, K, V for each word**

For "cat": multiply x_cat by each weight matrix.

```
Q_cat = x_cat @ W_Q = [1, 0] @ [[1,0],[0,1]] = [1.0, 0.0]
K_cat = x_cat @ W_K = [1, 0] @ [[0,1],[1,0]] = [0.0, 1.0]
V_cat = x_cat @ W_V = [1, 0] @ [[1,1],[0,1]] = [1.0, 1.0]

Q_sat = x_sat @ W_Q = [0, 1] @ [[1,0],[0,1]] = [0.0, 1.0]
K_sat = x_sat @ W_K = [0, 1] @ [[0,1],[1,0]] = [1.0, 0.0]
V_sat = x_sat @ W_V = [0, 1] @ [[1,1],[0,1]] = [0.0, 1.0]
```

**Step 4: Compute attention scores for "cat" attending to everything**

The scaling factor is √d_k = √2 ≈ 1.414.

```
score(cat→cat) = Q_cat · K_cat / √2
               = [1.0, 0.0] · [0.0, 1.0] / 1.414
               = (1×0 + 0×1) / 1.414
               = 0.0 / 1.414
               = 0.0

score(cat→sat) = Q_cat · K_sat / √2
               = [1.0, 0.0] · [1.0, 0.0] / 1.414
               = (1×1 + 0×0) / 1.414
               = 1.0 / 1.414
               = 0.707
```

Raw scores for "cat": [0.0, 0.707]

**Step 5: Apply softmax**

```
exp(0.0)   = 1.000
exp(0.707) = 2.028
Sum        = 3.028

attention weights for "cat":
  weight(cat→cat) = 1.000 / 3.028 = 0.330
  weight(cat→sat) = 2.028 / 3.028 = 0.670
```

"cat" pays 33% attention to itself and 67% attention to "sat".

**Step 6: Weighted sum of Values**

```
output_cat = 0.330 × V_cat  +  0.670 × V_sat
           = 0.330 × [1.0, 1.0]  +  0.670 × [0.0, 1.0]
           = [0.330, 0.330]  +  [0.000, 0.670]
           = [0.330, 1.000]
```

The output for "cat" is [0.330, 1.000]. It now carries blended information from both "cat" and "sat", with more weight on "sat" because Q_cat and K_sat had a higher dot product. Every number came from explicit arithmetic — nothing magic.

---

## Complexity Analysis

Understanding cost is critical for deploying transformers in production.

### Time Complexity

The dominant computation is the QK^T matrix multiply. For a sequence of n tokens with d_k-dimensional keys:

- Computing QK^T: each of the n queries does a dot product with each of the n keys, and each dot product costs d_k multiplications. Total: **O(n² · d_k)** operations.
- Computing the weighted sum (attention weights × V): n² weights, each multiplied against a d_v-dimensional value. Total: **O(n² · d_v)**.
- Computing Q, K, V projections: each is a matrix multiply of shape (n × d_model) × (d_model × d_k). Total: **O(n · d_model · d_k)** per projection, times 3.
- Full single-head attention: **O(n² · d_model + n · d_model²)**

For standard transformers where d_k = d_model, the n² term dominates at long sequence lengths.

### Memory Complexity

The attention weight matrix has shape (n × n). You must store this matrix during forward and backward passes.

**Memory = O(n²) per layer per head.**

To put real numbers on this: a 4096-token sequence with 32 heads has 4096 × 4096 × 32 ≈ 537 million entries per layer in float32, that is about 2GB per layer just for attention weights. With 96 layers (GPT-3 scale), attention matrices alone would exceed 190GB. This is why Flash Attention (see Q3 below) is not optional for long contexts.

At 128K tokens (GPT-4 context): 128,000² × 32 heads ≈ 524 billion entries per layer. Materializing this is impossible without tiling tricks.

### Parameter Count

The parameters live in the projection matrices, not in the attention computation itself:

- W_Q: d_model × d_k parameters
- W_K: d_model × d_k parameters
- W_V: d_model × d_v parameters (usually d_v = d_k)
- No parameters in the dot product, scale, or softmax steps — those are just arithmetic

**Total for one attention head: 3 × d_model × d_k**

For multi-head attention with h heads and d_k = d_model/h: 3 × d_model² plus the output projection d_model². Total: 4 × d_model² (see multi-head-attention.md for the full derivation).

### The n² Bottleneck in Practice

Doubling sequence length quadruples the attention cost.

```
GPT-3 (n=2048, d_model=12288):
  Attention matrix per head: 2048 × 2048 = ~4.2M entries

GPT-4 (n=128000, d_model assumed ~12288):
  Attention matrix per head: 128000 × 128000 = ~16.4B entries

Ratio: 128000² / 2048² = (128000/2048)² = 62.5² = 3906× more entries
```

This is why long-context models require architectural changes (sliding window attention, linear attention, Flash Attention) rather than just more memory.

---

## Failure Modes

### Attention Collapse (Uniform Weights)

At random initialization, Q and K are unrelated. Their dot products are near zero. Softmax of near-zero inputs produces uniform weights: every position gets weight ≈ 1/n.

The model attends equally to everything, which is equivalent to attending to nothing. The output is just an average of all value vectors — no information selection happens.

This is fine at initialization; the model learns to differentiate during training. But if it persists (due to learning rate issues or bad initialization), training stalls.

### Attention Spike (Softmax Saturation)

The opposite problem: after training, sometimes all attention concentrates on one token. The period "." and special separator tokens are infamous for this — they become "attention sinks" that absorb weight even when they carry no relevant information.

This is sometimes deliberately tolerated (it gives the model a "null" attention target), but it can hurt when important tokens get starved of attention weight.

### Future Leakage Bug (Missing Causal Mask)

Decoder models must not let token at position t attend to tokens at positions t+1, t+2, ... (they haven't been generated yet at inference time). This is enforced by adding -∞ to future positions before softmax, so exp(-∞) = 0 and those positions get zero attention weight.

If you forget the mask during training, the model learns to use future information. Training loss drops fast (it's cheating). But at inference, future tokens don't exist — the model fails. This is a silent, common bug. Always verify your mask by checking that the attention weight matrix is lower-triangular after softmax.

### Numerical Instability in Softmax

Computing exp(x) directly for large x overflows float32 (max representable value ≈ 3.4 × 10³⁸; exp(90) already exceeds this). Before scaling by √d_k was standard practice, dot products in high-dimensional spaces routinely caused NaN gradients.

Always use the numerically stable softmax:

```
softmax(x)ᵢ = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
```

Subtracting max(x) does not change the output (the max term cancels in numerator and denominator) but prevents overflow. The largest exponent computed is exp(0) = 1, and all others are smaller.

---

## Staff/Principal Interview Depth

The questions below are the kind you'd face in a Staff/Principal MLE interview. They are judgment questions, not recall questions. Try answering before reading the levels.

---

**Q1: Why do we divide by √d_k and not d_k or some other constant?**

---
**No Hire**
*Interviewee:* "We divide to make the numbers smaller so softmax works better."
*Interviewer:* The candidate knows *that* scaling happens but has no idea *why* √d_k specifically. This is the "I saw it in the formula" answer. No variance analysis, no gradient reasoning, no alternative considered.
*Criteria — Met:* Knows scaling exists / *Missing:* Variance analysis, softmax gradient reasoning, why √d_k vs d_k

---
**Weak Hire**
*Interviewee:* "If d_k is large, the dot products get large, which pushes softmax into a saturated region where gradients are near zero. Dividing by √d_k keeps the scale reasonable."
*Interviewer:* Correct and useful answer. The candidate grasps the saturation problem and knows that scaling prevents it. What's missing: why √d_k specifically? Why not just clip the values, or divide by d_k, or use a learned scale? The candidate can't derive the √d_k from first principles.
*Criteria — Met:* Saturation problem, gradient vanishing at large scale / *Missing:* Variance derivation, reason for √ vs other normalizations

---
**Hire**
*Interviewee:* "The reason is variance. If q and k are vectors with entries drawn i.i.d. from N(0,1), their dot product is a sum of d_k independent terms, each with variance 1. So Var(q·k) = d_k, and the standard deviation is √d_k. Dividing by √d_k normalizes the variance back to 1. We want unit variance because softmax gradients are maximized near zero — large inputs saturate softmax, gradients go to zero, and training stalls. Dividing by d_k would over-correct: variance becomes 1/d_k, which is very small for large d_k, making attention near-uniform and unable to differentiate relevant from irrelevant positions."
*Interviewer:* Strong. The candidate derives √d_k from the variance analysis, correctly distinguishes √d_k from d_k, and gives the gradient reasoning. What would push to Strong Hire: mentioning the assumption (N(0,1) initialization) explicitly, noting what happens if that assumption breaks (e.g., after several training steps the QK distributions shift), and awareness of alternatives like learned temperature scaling.
*Criteria — Met:* Variance derivation (Var=d_k), √d_k normalization to unit variance, softmax saturation at large input, over-correction argument for d_k / *Missing:* Assumption about initialization distribution, learned temperature as alternative

---
**Strong Hire**
*Interviewee:* "The derivation starts with the initialization assumption: W_Q and W_K are initialized such that Q and K vectors have entries ≈ N(0,1). Under this assumption, the dot product q·k = Σᵢ qᵢkᵢ. Each term qᵢkᵢ is a product of two independent N(0,1) variables, which has mean 0 and variance 1. Summing d_k such terms: Var(q·k) = d_k, standard deviation = √d_k. Dividing by √d_k gives Var(q·k / √d_k) = 1. We specifically want unit variance — not just 'small' — because that's where softmax operates in its highest-gradient regime. Too large → saturation, gradient collapse. Too small → near-uniform attention, model can't select. The √d_k choice is derived directly from the initialization distribution. Two practical caveats: first, after many training steps, Q and K distributions shift and the initialization argument no longer holds exactly — some practitioners use learned temperature scaling (replacing the fixed 1/√d_k with a learned scalar) to adapt. Second, in float16 training, you need to be careful even after scaling because exp overflow is possible — production implementations use numerically stable softmax with the max subtraction."
*Interviewer:* This is the answer. Derives from first principles, explains why unit variance is the target (not just "smaller"), gives both caveats (distribution shift after training, float16 overflow), and mentions the learned temperature alternative as a real production consideration. The level of precision — calling out the initialization distribution explicitly and noting it breaks down — is what separates staff-level reasoning from senior-level recall.
*Criteria — Met:* Full variance derivation, unit variance target reasoning, gradient regime analysis, √ vs d_k comparison, float16 overflow, learned temperature alternative, distribution shift caveat
---

---

**Q2: What's the difference between additive (Bahdanau) and dot-product attention? When would you use each?**

---
**No Hire**
*Interviewee:* "Additive attention uses addition and dot-product attention uses dot products. Transformers use dot-product because it's faster."
*Interviewer:* Technically the first sentence follows from the names, but it conveys nothing. "Faster" is asserted without any reasoning. The candidate doesn't know what additive attention actually computes, when the speed difference matters, or why dot-product has a weakness that the √d_k fix addresses.
*Criteria — Met:* None / *Missing:* Additive attention formula, when each wins, the relationship between √d_k and dot-product's large-d_k weakness

---
**Weak Hire**
*Interviewee:* "Additive attention uses a small neural network to compute scores: score(q,k) = v·tanh(W_q q + W_k k). Dot-product just does q·k / √d_k. Additive has more parameters, dot-product is faster because it's just matrix multiply. Use dot-product for transformers."
*Interviewer:* The candidate knows both formulas and the compute reason. What's missing: when does additive attention actually *outperform* dot-product? The candidate implies dot-product is always better, which isn't true at small d_k. Also no awareness of why √d_k was introduced specifically to fix dot-product's large-d_k problem.
*Criteria — Met:* Both formulas, compute speed argument / *Missing:* Small d_k case where additive wins, √d_k as the fix for dot-product's large-dimension weakness

---
**Hire**
*Interviewee:* "The formulas differ fundamentally: additive is score(q,k) = v^T tanh(W_q q + W_k k), with learned parameters v, W_q, W_k. Dot-product is score(q,k) = q·k / √d_k, with no extra parameters. At small d_k — say 16 or 32 — additive attention performs better. The tanh non-linearity is a richer compatibility function than a dot product, so additive can model more complex relationships. Dot product at small dimension has limited capacity. At large d_k, dot-product wins: it's one matrix multiply, highly optimized on GPU. The extra parameters of additive attention stop paying off because the high-dimensional dot-product space already has enough expressiveness. The √d_k fix was introduced specifically to address dot-product's failure at large d_k — without it, dot products blow up in variance and saturate softmax."
*Interviewer:* Excellent. Gives both formulas, explains the expressiveness argument for small d_k, correctly frames √d_k as the fix for dot-product's large-d_k weakness, and gives a practical decision rule. What would push to Strong Hire: knowing the crossover point more precisely, discussing how this connects to temperature annealing, or mentioning the RKHS (kernel) interpretation of attention.
*Criteria — Met:* Both formulas, small d_k expressiveness argument, large d_k GPU efficiency, √d_k as fix / *Missing:* Kernel interpretation, precise crossover analysis

---
**Strong Hire**
*Interviewee:* "The formulas: additive is score(q,k) = v^T tanh(W_q q + W_k k) — a feedforward network parameterized by v ∈ R^d, W_q, W_k ∈ R^{d×d_k}. Dot-product: score(q,k) = q·k / √d_k — no additional parameters, pure arithmetic. The theoretical framing: both can be viewed as kernel functions computing similarity between q and k. Additive uses a learned kernel (tanh MLP), dot-product uses the linear inner product kernel, which corresponds to an RKHS where features are the vectors themselves. At small d_k, the linear kernel is low-rank — it can't represent complex compatibility patterns. The MLP kernel has higher capacity and consistently outperforms at d_k ≤ 32 (Luong et al., 2015 showed this empirically). At large d_k, the dot product's advantage is: (1) it's a single batched matrix multiply that gets full cuBLAS optimization, (2) the high-dimensional inner product space is expressive enough that extra parameters don't help much. The variance problem — without √d_k, dot products grow as O(d_k) in standard deviation — made dot-product unstable at large dimension until the scaling fix was introduced. This is why Bahdanau used additive attention in 2015 (pre-scaling fix) and Vaswani switched to dot-product in 2017 (post-fix). In modern practice: dot-product everywhere except specialized cases where sequences are very short and embeddings very small."
*Interviewer:* Exactly what you want from a staff candidate. The kernel framing shows architectural thinking. Correctly explains the historical reason Bahdanau used additive attention (no √d_k fix yet). Knows the empirical crossover point. Gives a concrete production recommendation. The connection between the MLP kernel and RKHS is optional depth that signals genuine understanding of why the formulas behave the way they do.
*Criteria — Met:* Both formulas, kernel/RKHS framing, small d_k empirical evidence, large d_k efficiency argument, historical context (Bahdanau 2015 vs Vaswani 2017), production recommendation
---

---

**Q3: What is Flash Attention and why does it matter for long contexts?**

---
**No Hire**
*Interviewee:* "Flash Attention is a faster version of attention that uses less memory. It's used in GPT-4 and other long-context models."
*Interviewer:* The candidate has heard the name and the general claim. There's no mechanistic understanding — the candidate doesn't know what Flash Attention actually does differently, why it uses less memory, or whether it produces the same results.
*Criteria — Met:* Knows the name, knows it's memory-efficient / *Missing:* Mechanism, exact vs approximate, SRAM/HBM distinction, online normalization

---
**Weak Hire**
*Interviewee:* "Standard attention materializes the full n×n attention matrix, which is O(n²) memory. Flash Attention avoids materializing that matrix by computing attention in tiles, staying in fast on-chip memory. It reduces memory usage to O(n) and is faster because it reduces memory reads and writes. It's exact — same math, just computed in a different order."
*Interviewer:* The candidate gets the key ideas: tiling, O(n) memory, exactness. This is a solid answer that most senior engineers could give. What's missing for staff level: the SRAM/HBM distinction (the actual reason tiling matters), the online softmax algorithm (how you compute softmax without seeing all the values), and actual speedup numbers.
*Criteria — Met:* Tiling concept, O(n) memory, exact arithmetic / *Missing:* SRAM vs HBM framing, online softmax normalization algorithm, hardware-specific speedup numbers

---
**Hire**
*Interviewee:* "Standard attention materializes the n×n attention score matrix in HBM — GPU high-bandwidth memory. For n=4096 with 32 heads and float16, that's 4096²×32×2 bytes ≈ 1GB per layer, just for attention weights. Flash Attention's insight: the bottleneck isn't FLOPs, it's HBM memory bandwidth. Reading and writing the n×n matrix takes more wall-clock time than the actual arithmetic. Flash Attention tiles the computation: for each tile of Q, iterate over K, V tiles and accumulate using the online softmax algorithm — maintaining a running max and normalization factor so you never need the full row's scores at once. This keeps computation in SRAM (20MB on A100) instead of HBM (80GB). Result: HBM reads scale as O(n² / M) where M is SRAM size, vs O(n²) before. Memory footprint is O(n) — the n×n matrix is never stored. On A100, Flash Attention is 2–4× faster for typical context lengths. It's exact — same result as standard attention, just different computation order."
*Interviewer:* Very strong. The candidate quantifies the memory (1GB per layer example), identifies the actual bottleneck (memory bandwidth not FLOPs), explains the online softmax mechanism at a high level, uses the SRAM/HBM framing correctly, and gives real speedup numbers. What would push to Strong Hire: explaining the online normalization algorithm precisely (running max, log-sum-exp), Flash Attention 2 improvements, and the backward pass implications.
*Criteria — Met:* HBM bottleneck framing, quantified memory example, tiling mechanism, online softmax (high level), SRAM/HBM, O(n) memory footprint, 2-4x speedup / *Missing:* Online normalization precision, Flash Attention 2, backward pass tiling

---
**Strong Hire**
*Interviewee:* "Flash Attention's insight is that the attention bottleneck for long sequences is memory bandwidth, not arithmetic. For n=32K, 32 heads, float16: the n×n matrix is 32K²×32×2 bytes ≈ 65GB per layer — impossible to materialize, and even at shorter lengths, reading/writing HBM is slow. Flash Attention reorders computation to stay in SRAM using two key algorithms. First, tiling: split Q into row blocks Q_i, K and V into column blocks K_j, V_j. For each (Q_i, K_j) tile, compute partial attention scores and partial weighted sums. Second, online softmax normalization: to correctly combine partial softmax computations without seeing the full row, maintain two running statistics per query position — m_i (running max of scores seen so far) and l_i (running sum of exp(score - m_i)). When a new tile arrives with max m_new, rescale the accumulated output: O_i = (l_i × O_i_prev + exp(m_i - max(m_i, m_new)) × new_partial) / new_l_i. This produces exact softmax without materializing the full row. Result: O(n) memory instead of O(n²), HBM reads O(n·d / M) where M = SRAM size (~20MB A100). Wall-clock speedup: 2–4× on A100, more on sequences > 4K tokens. Flash Attention 2 (2023) further improves GPU utilization by splitting work across query blocks instead of K-V blocks, getting closer to theoretical peak FLOP utilization — typically 50–70% MFU vs 25–35% for Flash Attention 1. The backward pass requires recomputing the attention matrix from tiles (recomputation vs. storing), trading FLOPs for memory. For training where both activations and gradients need to be stored, this is a significant saving."
*Interviewer:* This is exactly the staff-level answer. The candidate knows the actual algorithm (running max, log-sum-exp update, the exact recurrence), derives the memory savings from first principles, gives real hardware specs (SRAM size, A100 HBM), compares Flash Attention 1 and 2, and mentions the backward pass recomputation trade-off. Volunteering the backward pass trade-off without being asked is the signal of someone who has used this in production and hit the real constraints.
*Criteria — Met:* Memory bandwidth bottleneck, tiling algorithm, online softmax recurrence (precise), O(n) memory derivation, actual hardware numbers, Flash Attention 2 improvement, backward pass recomputation trade-off
---

---

**Q4: In cross-attention, why do Q come from the decoder and K, V from the encoder?**

---
**No Hire**
*Interviewee:* "Because the decoder generates the output and needs to look at the encoder's input."
*Interviewer:* Correct by tautology. The candidate hasn't explained *why* Q maps to "looking" and K,V map to "being looked at" — they've just restated which direction the information flows.
*Criteria — Met:* Correct direction / *Missing:* Mechanistic reasoning for the Q/KV assignment, why reversal would fail

---
**Weak Hire**
*Interviewee:* "The decoder is generating output one token at a time and needs to figure out which parts of the input are relevant for the next token. The Query comes from the decoder because it's asking the question: 'which input words matter for my next output word?' The Keys and Values come from the encoder because the encoder has processed the full input and is providing the answers."
*Interviewer:* This is the correct conceptual explanation. The candidate understands the Q = "question", KV = "answer" framing and correctly maps the decoder and encoder roles. What's missing: why would reversing this fail? And what does it mean computationally that the same encoder KV can be queried multiple times?
*Criteria — Met:* Q = question from decoder, KV = answer from encoder, basic role mapping / *Missing:* Why reversal fails, KV reuse across decoder steps, computational implications

---
**Hire**
*Interviewee:* "The decoder's current state encodes 'what I'm looking for' at this generation step — that's the Query. The encoder's output encodes 'what the input contains' — that's the Keys for matching and Values for retrieval. The direction isn't arbitrary: the decoder drives the query because its state changes at every generation step (as it produces more tokens), while the encoder output is fixed once. If you reversed it — Q from encoder, KV from decoder — the encoder would be querying a target that changes with every generated token. You'd have to recompute cross-attention from the encoder side after each decoder step, which breaks the clean separation that lets the encoder run once and be cached. Concretely: in translation, at step 1 the decoder state is [BOS], and its Q focuses the attention on the first input word. At step 2 the decoder state is [BOS, 'Le'], and its Q shifts attention to the second relevant word. The encoder KV is reused identically at every step — no recomputation."
*Interviewer:* Excellent. The candidate correctly frames the asymmetry (decoder state changes, encoder is fixed), explains why reversal fails (encoder would have to requery a shifting target), and gives the concrete caching advantage. What would push to Strong Hire: discussing how the encoder's fixed KV enables KV caching during inference, the connection to memory-augmented architectures, and why the attention pattern learned in cross-attention encodes the word alignment between languages.
*Criteria — Met:* Q = changing decoder state, KV = fixed encoder, reversal failure argument, KV reuse insight / *Missing:* KV cache inference benefit, alignment interpretation, connection to memory architectures

---
**Strong Hire**
*Interviewee:* "The assignment is mechanistically necessary. The decoder state at step t encodes 'what I currently need' — this is the information that should do the querying, because a query selects what to retrieve. The encoder output encodes 'what the source contains' — this is the information that should be retrieved, making it both Keys (for matching) and Values (for content). Reversing would create two problems. First, logical: the encoder is a fixed representation of the full input — it has no 'question' to ask, because it doesn't know yet what output is being generated. Second, computational: cross-attention with Q from encoder and KV from decoder would require the encoder to re-attend to the decoder's output at every generation step. The encoder-side computation would scale with the number of generated tokens and couldn't be precomputed. The current design is efficient: encoder runs once, its output is stored as KV, and the decoder attends to those cached KV at every step. This is why cross-attention doesn't need to be included in the autoregressive KV cache — the encoder KV is computed once and held. A deeper view: cross-attention in translation models learns soft word alignments. The attention weight α(decoder_t → encoder_s) approximates 'how much output token t corresponds to input token s.' This is studied explicitly in the neural machine translation literature (Bahdanau et al., 2015 showed attention weights recover classical alignment tables). The Q=decoder, KV=encoder structure is the only one that lets this alignment be read out at each decoder step."
*Interviewer:* Staff-level. The candidate explains why reversal is logically wrong (encoder has no question to ask), derives the computational efficiency argument (encoder KV precomputed, not in autoregressive cache), and connects to the word alignment literature. Volunteering the Bahdanau alignment result and noting it "recovers classical alignment tables" shows genuine depth — this is a person who understands what cross-attention actually computes, not just how to implement it.
*Criteria — Met:* Logical argument against reversal, computational efficiency of fixed encoder KV, autoregressive cache separation, word alignment interpretation, historical context (Bahdanau alignment)
---

---

**Q5: Explain exposure bias and teacher forcing. How does this relate to attention?**

---
**No Hire**
*Interviewee:* "Teacher forcing means we give the model the right answer during training. Exposure bias is when the model doesn't see its own errors during training."
*Interviewer:* The candidate knows the terms and their basic meaning. But "the model doesn't see its own errors" could describe any supervised learning — the specific mechanism (input distribution mismatch between train and inference) isn't articulated, and the attention connection isn't made at all.
*Criteria — Met:* Definition of teacher forcing, recognition of exposure bias / *Missing:* Distribution mismatch mechanism, attention's role in propagating errors, mitigation strategies

---
**Weak Hire**
*Interviewee:* "During training with teacher forcing, the model always receives the correct token as input at each step, not what it predicted. This makes training stable and fast. Exposure bias is the gap this creates: at inference, the model receives its own predictions, including errors. If it generates a wrong token, the next token's prediction is conditioned on that wrong context, which the model was never trained on. This can cascade."
*Interviewer:* Correct description of both concepts and the cascade mechanism. What's missing: how does attention specifically amplify exposure bias compared to a simpler model? The candidate treats all parts of the model the same — but attention's soft blending over all previous tokens means that a wrong token at step t influences all future tokens' attention patterns, not just the next hidden state.
*Criteria — Met:* Teacher forcing definition, exposure bias definition, distribution mismatch, error cascade / *Missing:* Attention-specific amplification mechanism, mitigation strategies with trade-offs

---
**Hire**
*Interviewee:* "Teacher forcing: at training step t, the decoder receives the ground-truth token at position t-1, not the model's prediction. This means the training input distribution is the ground-truth corpus. At inference, the input is the model's own predictions. If the model generates wrong token w at step t, token t+1 sees w as context. The model was never trained on the error distribution — this is exposure bias. Attention amplifies this in a specific way. In attention, every token at every step looks back at all previous tokens via the attention mechanism. A wrong token at step t becomes part of the Key-Value pool for all subsequent tokens. Tokens that attend heavily to recent context will heavily reference the error. The attention weights will distribute over the wrong token just as they would over a correct one, because the weights are computed from similarity, not from correctness. This is unlike an RNN where the error affects only the hidden state chain — attention makes the error universally available. Mitigations: scheduled sampling gradually replaces gold tokens with model predictions during training; beam search reduces single-error dominance; RLHF directly trains on the model's own output distribution."
*Interviewer:* Strong. The candidate correctly articulates the distribution mismatch, explains attention's specific role (wrong token becomes KV pool entry available to all subsequent tokens), distinguishes this from the RNN case, and gives three mitigations. What would push to Strong Hire: quantitative framing of scheduled sampling (the annealing schedule), why RLHF is the most principled fix, and why beam search only partially mitigates (it doesn't fix the distribution mismatch).
*Criteria — Met:* Teacher forcing definition, distribution mismatch, attention-specific amplification (wrong KV pool), comparison to RNN, three mitigations / *Missing:* Scheduled sampling annealing, why RLHF is principled, beam search partial mitigation argument

---
**Strong Hire**
*Interviewee:* "Teacher forcing trains the decoder with p(y_t | y_{<t}^*, x) where y^* is the ground truth. Inference evaluates p(y_t | y_{<t}^model, x) where y^model is the model's own generations. The training and inference distributions over input sequences differ — this gap is exposure bias (Ranzato et al., 2016 formalized it). Attention makes exposure bias structurally different from the RNN case. In an RNN, errors propagate through a single hidden state chain: h_t = f(h_{t-1}, y_{t-1}). A wrong y_{t-1} corrupts h_t but only influences subsequent states through that one pathway. In a transformer decoder, the wrong token y_{wrong} at position t is stored as a Key and Value entry accessible to all subsequent queries at steps t+1, t+2, .... Attention heads that specialize in recent context — and many do (Voita et al., 2019 showed induction heads specialize in local context) — will heavily weight y_{wrong} for the next several tokens, propagating the error broadly. The softmax attention ensures y_{wrong} can never be completely ignored (attention weights are strictly positive). Three mitigations with precise trade-offs: scheduled sampling (Bengio et al., 2015) anneals the mixing ratio from 100% teacher-forced to 100% model-generated over training; this fixes distribution mismatch but introduces non-differentiability and training instability. Beam search hedges against single errors by exploring multiple beams, but doesn't fix the distribution — the model still never trained on its own errors, beam search just post-hoc reduces their impact. RLHF (Ouyang et al., 2022) directly optimizes the model on its own output distribution via a reward signal, eliminating the distribution mismatch at the cost of training complexity and reward hacking risk. For production language models, RLHF is the dominant approach because it's the only one that fundamentally addresses the gap."
*Interviewer:* This is exactly what staff-level looks like. The candidate formalizes both distributions with notation, brings in the research literature at exactly the right moments (Ranzato, Bengio, Voita, Ouyang), explains attention's amplification through induction heads, and gives a nuanced comparison of mitigations that includes the trade-offs, not just the names. Volunteering "softmax attention ensures y_{wrong} can never be completely ignored" — that's not from a textbook, that's from understanding the math.
*Criteria — Met:* Formal distribution notation, research literature citations, RNN vs transformer comparison, induction head mechanism, scheduled sampling annealing, beam search limitation, RLHF as principled fix with trade-offs
---

## Key Takeaways

1. **Attention lets every word look at every other word** -- no more reading through a keyhole
2. **Q, K, V** are like asking a question (Q), checking shelf labels (K), and grabbing books (V)
3. **Dot product** measures similarity between queries and keys
4. **Softmax** turns similarity scores into weights that sum to 1
5. **Scaling by √d_k** prevents attention from becoming too extreme — formally, it normalizes dot-product variance back to 1
6. **Self-attention** = words attending to their own sentence
7. **Cross-attention** = one sequence attending to a different sequence
8. **Time complexity is O(n² · d_model)** — doubling sequence length quadruples cost
9. **Memory is O(n²) per head per layer** — this is the bottleneck for long contexts
10. **Failure modes** — attention collapse, saturation spikes, causal mask bugs, and softmax overflow are the four common failure patterns to guard against

---

## Prerequisites

Before reading this, you should understand:
- Vectors and matrices (from [Neural Network Fundamentals](../../00-neural-networks/fundamentals/04_neural_network_layers.ipynb))
- How neural networks learn weights (from [Backpropagation](../../00-neural-networks/fundamentals/07_backpropagation.ipynb))

## Further Reading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) -- the paper that started it all
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) -- excellent visual guide
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) -- comprehensive overview
- [Flash Attention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) -- IO-aware exact attention
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) (Dao, 2023) -- improved parallelism

---

[Back to Architecture Overview](./README.md) | [Next: Multi-Head Attention](./multi-head-attention.md)
