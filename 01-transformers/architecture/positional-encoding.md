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

---

## Worked Numerical Example

Let's compute PE(pos=2) step by step with d_model=4. This is small enough to hold in your head.

With d_model=4, we have four dimensions (0, 1, 2, 3). The denominator formula becomes:

```
10000^(2i/d_model) = 10000^(2i/4) = 10000^(i/2)
```

Now compute each dimension for position 2:

```
dim 0  (i=0, even → use sine):
  PE(2, 0) = sin(2 / 10000^(0/2))
           = sin(2 / 10000^0)
           = sin(2 / 1)
           = sin(2)
           ≈ 0.909

dim 1  (i=0, odd → use cosine):
  PE(2, 1) = cos(2 / 10000^(0/2))
           = cos(2 / 1)
           = cos(2)
           ≈ -0.416

dim 2  (i=1, even → use sine):
  PE(2, 2) = sin(2 / 10000^(1/2))
           = sin(2 / 100)
           = sin(0.02)
           ≈ 0.0200

dim 3  (i=1, odd → use cosine):
  PE(2, 3) = cos(2 / 10000^(1/2))
           = cos(2 / 100)
           = cos(0.02)
           ≈ 0.9998
```

So PE(pos=2) ≈ [0.909, -0.416, 0.020, 0.9998].

Notice what's happening:
- Dimensions 0-1 use the **fastest wave** (dividing by 1). Period ≈ 2π ≈ 6.28 positions. These dimensions change dramatically between adjacent positions.
- Dimensions 2-3 use a **much slower wave** (dividing by 100). Period ≈ 2π × 100 ≈ 628 positions. These dimensions barely change between positions 0 and 5, but distinguish positions 0 and 300.

Together they form a unique "fingerprint" for position 2. No other position produces exactly this combination.

Here's the full table for positions 0 through 3 with d_model=4:

```
         dim0      dim1      dim2      dim3
         sin(p/1)  cos(p/1)  sin(p/100) cos(p/100)

pos 0:  [ 0.000,   1.000,    0.000,    1.000 ]
pos 1:  [ 0.841,   0.540,    0.010,    1.000 ]
pos 2:  [ 0.909,  -0.416,    0.020,    0.9998]
pos 3:  [ 0.141,  -0.990,    0.030,    0.9996]
```

Every row is unique. A model seeing these vectors can unambiguously distinguish the four positions.

---

## The Distance Property: Why Sinusoidal Encoding Is Special

Here is the most important mathematical property of sinusoidal encoding. Imagine you are standing at mile marker 50 on a highway. If you know the "PE address" of mile marker 50, can you figure out what mile marker 53 looks like? With sinusoidal encoding, yes — because there is a fixed rotation that takes you from any position to the same position +k, regardless of where you started.

**Formal proof:**

Write PE(pos) for a single frequency as a 2D vector:

```
PE(pos) = [sin(w·pos), cos(w·pos)]

where w = 1 / 10000^(2i/d_model)  (the frequency for dimension pair i)
```

Now write PE(pos+k):

```
PE(pos+k) = [sin(w·(pos+k)), cos(w·(pos+k))]
```

Apply the angle addition formulas:

```
sin(w·pos + w·k) = sin(w·pos)·cos(w·k) + cos(w·pos)·sin(w·k)
cos(w·pos + w·k) = cos(w·pos)·cos(w·k) - sin(w·pos)·sin(w·k)
```

Write this as a matrix equation:

```
PE(pos+k) = R(w·k) · PE(pos)

where R(θ) = [[cos(θ), -sin(θ)],
              [sin(θ),  cos(θ)]]

is a 2D rotation matrix.
```

The key insight: **R(w·k) depends only on the distance k, not on pos.**

This means the transformation from PE(pos) to PE(pos+k) is the same rotation no matter where you start. The model can learn "look k steps ahead" as a linear function of the current PE — it just needs to learn to apply the right rotation matrix, which is the same everywhere in the sequence.

This is the mathematical reason why transformers trained with sinusoidal encoding can, in principle, generalize to relative position reasoning. The structure is not arbitrary — it is baked into the geometry of the encoding.

---

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

## RoPE: Rotary Position Embeddings (Full Math)

RoPE does **not** add a position vector to the embedding. Instead, it rotates the Q and K vectors before computing the dot product. Position information lives in the angle of the vectors, not in their magnitude or direction.

**The rotation operation:**

For a 2D subvector (q_{2i}, q_{2i+1}) at position pos, RoPE applies:

```
RoPE_q(pos) = R(θ_i · pos) · [q_{2i}, q_{2i+1}]^T

where R(θ) = [[cos(θ), -sin(θ)],
              [sin(θ),  cos(θ)]]

and θ_i = 10000^(-2i/d_model)   (same frequency schedule as sinusoidal PE)
```

The same rotation is applied to the K vector at its own position pos'.

**Why this makes dot products encode relative distance:**

Compute the dot product of the rotated Q and K:

```
(RoPE_q(pos))^T · (RoPE_k(pos'))
  = [q_{2i}, q_{2i+1}] · R(-θ_i·pos) · R(θ_i·pos') · [k_{2i}, k_{2i+1}]^T
  = [q_{2i}, q_{2i+1}] · R(θ_i·(pos' - pos)) · [k_{2i}, k_{2i+1}]^T
```

The rotation matrix depends only on (pos' - pos), the **relative distance**. The absolute positions cancel out. Position information is baked directly into the attention score, not the stored embeddings.

**Why this matters:**

- The Q and K vectors are fully dedicated to content. Position is handled entirely through the rotation, so there is no "positional noise" added to the embedding that the model must learn to filter out.
- Relative position is exact and automatic — the model does not need to learn to extract distance from additive position vectors.
- RoPE requires no extra parameters. The rotation angles θ_i are computed from the same formula as sinusoidal PE.
- Every attention layer applies the rotation independently. This means position information is re-injected at every layer, rather than only at the input.

The tradeoff: RoPE must be applied at every attention layer (not just once at input), adding compute proportional to n_layers × n × d_model.

---

## Failure Modes

### Learned encoding extrapolation failure

BERT and GPT-2 use learned positional embeddings. If you train on sequences up to length 512, position 513 has never been seen — its embedding is randomly initialized and was never updated. The model does not degrade gracefully; it degrades catastrophically at unseen lengths because the entire downstream computation was trained assuming the position embeddings carry meaningful signal. This is why you cannot simply use BERT for documents longer than 512 tokens without workarounds (chunking, hierarchical approaches, or re-training with ALiBi/RoPE).

### Sinusoidal at extreme lengths

Sinusoidal encoding is mathematically defined for any length — you can compute PE(10000) without any modification. However, at very long positions, the slow-frequency waves have barely moved (e.g., at position 10000, the slowest dimension has completed about 0.16 cycles), while the fast-frequency waves have cycled many times and may have similar values at positions separated by the wave period. The model may confuse very distant positions that share similar fast-wave values. In practice, pre-LN transformers degrade more gracefully than post-LN on out-of-distribution lengths.

### Position embedding interference at initialization

If positional embeddings are initialized with much larger values than word embeddings, the model ignores word content early in training and focuses only on position. This can create a bad local minimum that is hard to escape. Standard initialization: positional embeddings at the same scale as word embeddings (approximately 0.02 standard deviation for GPT-2-style models). Monitoring the ratio of positional embedding norms to word embedding norms at initialization is a practical debugging step.

### RoPE's graceful degradation and the fix

At unseen sequence lengths, RoPE's rotation angles continue to extrapolate mathematically — but the model has never trained on those rotation values, so attention scores become out-of-distribution. The fix is **Position Interpolation** (Chen et al. 2023): scale all position indices by (train_len / inference_len) before computing the rotation. This "compresses" the position scale so that position 32768 maps to the same rotation angle as position 2048 did during training. LLaMA-2 uses this approach to extend from 4K to 32K context without full retraining. For larger extensions, **NTK-aware scaling** (Su 2023) adjusts the base θ rather than scaling positions directly, which better preserves high-frequency information lost by naive interpolation.

---

## Complexity Analysis

```
Method          Parameters          Compute             Memory
──────────────  ──────────────────  ──────────────────  ──────────────────────
Sinusoidal      0 (formula-based)   O(n × d_model)      O(n × d_model)
                                    to compute;         to store (can recompute
                                    computed once       on-the-fly)
                                    at input

Learned PE      max_len × d_model   O(1) at inference   O(max_len × d_model)
                (e.g., BERT:        (just a lookup)     always in memory
                512 × 768 =
                393,216 ≈ 0.4M
                params — tiny
                vs 110M total)

RoPE            0 (formula-based)   O(n × d_model)      O(n × d_model) per
                                    per attention       attention layer
                                    layer; applied      (not reusable
                                    at every layer,     across layers)
                                    not just once

ALiBi           0 (formula-based)   O(1) to compute     O(h × n) for the
                                    (linear bias        bias matrix
                                    added directly      (negligible)
                                    to attention
                                    scores)
```

Key tradeoff: ALiBi is the cheapest at inference (zero parameters, O(1) compute per attention score). RoPE is more expensive than additive PE because rotation must be re-applied at every attention layer. Learned PE is the simplest to implement but trades flexibility for a fixed parameter table.

---

## Staff/Principal Interview Depth

**Q1: Why is positional encoding added to word embeddings rather than concatenated?**

Concatenation would require every downstream weight matrix to be larger — the Q, K, V projection matrices would need to be (d_model + d_pos) × d_k instead of d_model × d_k. More importantly, addition lets position and content interact immediately: the Q projection W_Q receives a vector that is already a mixture of content and position, so it can learn to extract position-sensitive features in a single linear operation. With concatenation, you would need explicit cross terms between the content block and the position block of W_Q to achieve the same interaction — effectively requiring the weight matrix to have off-diagonal block structure, which it would learn eventually but at the cost of more parameters and slower convergence. Addition is equivalent to concatenation followed by a weight matrix with a specific constrained structure: it is a lower-dimensional but empirically sufficient parameterization. The original paper validated this empirically and found no significant performance difference between sinusoidal addition and learned addition.

**Q2: Compare RoPE vs ALiBi. When would you use each, and what are the tradeoffs?**

ALiBi (Press et al. 2022) adds a linear penalty to attention scores: score(i,j) = q_i·k_j - m·|i-j|, where m is a per-head slope (different slopes for different heads, fixed not learned). Zero extra parameters. Extrapolates to longer sequences well because the linear penalty grows smoothly — longer sequences just get larger penalties for attending far back, but there is no out-of-distribution embedding lookup. Does not require retraining for longer sequences. Downside: the penalty is not learned, so it cannot capture non-monotonic position effects (e.g., attending strongly to something exactly 10 tokens back but weakly to things 9 or 11 tokens back). Very long-range attention is always penalized regardless of content.

RoPE provides a learned interaction between position and content through the rotation. The model can learn complex position-content dependencies (e.g., "this word matters when it is exactly 3 positions before a verb"). Does not extrapolate to unseen lengths without position interpolation. Higher compute cost (applied at every attention layer). Quality advantage grows with context length and task complexity.

In practice: ALiBi for fast deployment with variable-length inference and situations where simplicity and robustness matter more than peak quality. RoPE for maximum quality in fixed-context production models (LLaMA, Mistral, Gemma). ALiBi trains faster in early training because there is no positional interference at the input; RoPE converges to better final quality on most benchmarks.

**Q3: Why was 10000 chosen as the base in the sinusoidal formula? What happens if you change it?**

10000 was chosen empirically in Vaswani et al. 2017. The base controls the range of wavelengths across dimensions. With base B, the wavelengths span from 2π (fastest dimension, i=0) to 2π × B (slowest dimension, i=d_model/2 - 1). With base=10000, the slowest period is approximately 62,832 positions — covering typical sequence lengths of up to ~10K tokens with useful discriminability.

If you use base=100: slowest period ≈ 628 positions. All positions beyond ~600 get nearly identical encodings in the slow dimensions, reducing the model's ability to distinguish distant positions. If you use base=1,000,000: slowest period ≈ 6.3M positions — better for very long contexts but poor discrimination at short range because the slow waves barely move between adjacent positions.

LLaMA 3 uses a modified RoPE with θ=500,000 specifically to improve long-context handling. The base is a hyperparameter controlling the tradeoff between short-range discriminability (small base, fast waves dominate) and long-range discriminability (large base, slow waves distinguish distant positions). Changing the base without retraining is equivalent to applying a distribution shift to every attention layer simultaneously — the model will degrade immediately. Any base change requires retraining or at minimum continued training on relevant context lengths.

**Q4: A model trained on 512-length sequences struggles on 600-length inputs. What are your options?**

The answer depends on which positional encoding was used at training time.

If using **learned absolute PE** (original BERT): Option 1 — interpolate the embedding table from 512 to 600 positions using bilinear or linear interpolation along the position axis, then fine-tune on longer sequences for a small number of steps (~1-5% of original training). Option 2 — initialize positions 513-600 as random and fine-tune directly on longer sequences. Option 3 — replace with a relative PE scheme (RoPE or ALiBi) and fine-tune from the pre-trained weights, which typically works well because the attention weights themselves transfer well even if the position scheme changes.

If using **sinusoidal PE**: PE(513) through PE(600) are mathematically defined — just compute them. The encoding is not the problem. The attention patterns may not generalize because the model never trained on those positions, but degradation is typically gradual rather than catastrophic.

If using **RoPE**: position interpolation (Chen et al. 2023) — scale all position indices by (512/600) so that the maximum rotation angle seen at inference matches the maximum seen during training. This avoids out-of-distribution rotations. Works well for moderate extensions (up to approximately 4× training length). For larger extensions, NTK-aware scaling adjusts the base θ: new_base = θ × (inference_len / train_len)^(d_model / (d_model - 2)). This preserves high-frequency information that is destroyed by naive position scaling. LLaMA's extended context variants use variants of this approach.

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
