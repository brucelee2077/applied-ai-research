# Multi-Head Attention

## The Mystery Worth Solving

You know how, when you read a sentence, you're simultaneously tracking grammar, meaning, who's doing what, and where things are happening — all at the same time?

That's not one skill. It's several running in parallel.

A single attention head can only look for one type of relationship at a time. So how do transformer models keep track of all those different things simultaneously?

The answer is the most elegant design decision in the whole architecture: run several attention operations in parallel, each one looking for something different, then combine their findings at the end.

That's multi-head attention.

---

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

**What this analogy gets right:** Each student reads the same book (same input) but produces a different analysis (different attention patterns). Their findings are genuinely combined at the end into one report (concatenate + W_O projection). The specialization emerges naturally — no one told student 1 to focus on characters. The model discovers useful specializations the same way.

**Where this analogy breaks down:** Human students can deliberately choose their specialization and coordinate. Attention heads can't. The specialization in a trained model emerges entirely from gradient descent finding useful patterns — each head lands on a specialty because it makes the loss go down, not because of any explicit coordination. Also, real students could share notes mid-project; attention heads cannot share information during the forward pass through that layer.

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

### Victory Lap

You now understand multi-head attention — the exact mechanism that lets GPT-4 simultaneously understand grammar, semantics, pronoun reference, and long-range dependencies in a single pass. Every major transformer model from BERT to LLaMA uses this. The "parallel heads with different specializations" insight is what makes transformers so much more expressive than a single-head model at the same parameter count. You've earned the rest of this section.

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

## The Parameter Count Formula

Let's derive the exact parameter count step by step. This number matters because parameter count directly determines model size, memory usage, and training cost.

For a model with h heads and d_model total embedding size, each head uses d_k = d_model / h dimensions.

**Per head:**
- W_Qᵢ projects from d_model to d_k: **d_model × d_k parameters**
- W_Kᵢ projects from d_model to d_k: **d_model × d_k parameters**
- W_Vᵢ projects from d_model to d_v = d_k: **d_model × d_k parameters**

**Across all h heads for Q, K, V:**

```
3 × h × (d_model × d_k)
= 3 × h × d_model × (d_model / h)
= 3 × d_model²
```

The h cancels. The total Q, K, V parameters are always 3 × d_model², regardless of how many heads you use.

**Output projection W_O:**

W_O maps from the concatenated head outputs (h × d_k = d_model) back to d_model: **d_model × d_model = d_model² parameters**.

**Total parameters in one attention block:**

```
3 × d_model²   (Q, K, V projections)
+   d_model²   (output projection)
─────────────
4 × d_model²
```

**Worked example with real models:**

```
BERT Base: d_model = 768
  One attention block: 4 × 768² = 4 × 589,824 = 2,359,296 ≈ 2.36M parameters
  BERT Base has 12 layers: 12 × 2.36M = 28.3M parameters just in attention
  Total BERT Base is ~110M — attention is about 26% of the model

GPT-3: d_model = 12,288
  One attention block: 4 × 12,288² = 4 × 150,994,944 ≈ 604M parameters
  GPT-3 has 96 layers: 96 × 604M ≈ 58B parameters just in attention
  GPT-3 total is ~175B — attention is about 33% of the model
```

Note: the other large parameter contributor is the feedforward layer in each transformer block, which contributes another ~8 × d_model² per layer (two matrices of size d_model × 4·d_model).

---

## FLOPs Analysis

Floating point operations (FLOPs) tell you how much compute is needed to run the model. This determines inference latency, training time, and hardware requirements.

For a sequence of n tokens with d_model dimensions, h heads, and d_k = d_model/h:

**Computing Q, K, V projections (3 matrix multiplies):**
Each projection is (n × d_model) @ (d_model × d_k), costing 2 × n × d_model × d_k per projection.
With h heads: 3 × h × 2 × n × d_model × d_k = 3 × 2 × n × d_model × (h × d_k) = **6 × n × d_model²**

**Computing QK^T (attention scores) per head:**
Shape is (n × d_k) @ (d_k × n), costing 2 × n² × d_k per head.
Across h heads: h × 2 × n² × d_k = 2 × n² × (h × d_k) = **2 × n² × d_model**

**Softmax:**
O(n² × h) — negligible compared to the matrix multiplies above.

**Weighted sum (attention weights × V) per head:**
Shape is (n × n) @ (n × d_k), costing 2 × n² × d_k per head.
Across h heads: **2 × n² × d_model**

**Output projection W_O:**
Shape is (n × d_model) @ (d_model × d_model), costing **2 × n × d_model²**

**Total FLOPs for one attention layer:**

```
6n·d_model²  +  2n²·d_model  +  2n²·d_model  +  2n·d_model²
= 8n·d_model²  +  4n²·d_model
```

**When does n² dominate vs d_model² dominate?**

Set the two terms equal: 8n·d_model² = 4n²·d_model → n = 2·d_model. The crossover is around n ≈ 2 × d_model.

```
n=512,  d_model=512:
  Projections:   8 × 512 × 512²  = 1.07B FLOPs
  Attention QKV: 4 × 512² × 512  = 536M FLOPs
  → Projections dominate (2:1 ratio)

n=1024, d_model=512:
  Projections:   8 × 1024 × 512² = 2.15B FLOPs
  Attention QKV: 4 × 1024² × 512 = 2.15B FLOPs
  → Equal (crossover near n = 2 × d_model ✓)

n=8192, d_model=512:
  Projections:   8 × 8192 × 512² = 17.2B FLOPs
  Attention QKV: 4 × 8192² × 512 = 137B FLOPs
  → Attention dominates (8:1 ratio)
```

This is why long-context models require special attention algorithms (Flash Attention, sliding window, linear attention) — at large n, the n² attention terms swamp everything else.

---

## Why W_O Is Not Optional

After the concatenation step, each position in the output vector came from exactly one head. The first d_k positions came from head 1's output. The next d_k positions from head 2. And so on.

Without W_O, the heads are completely isolated from each other. Head 1's output can never influence head 2's portion of the representation. The model can't combine insights from multiple heads. You effectively have h small independent models that never talk to each other.

W_O is the only layer where heads interact. It learns to mix head outputs in whatever combination is useful: "take the semantic insight from head 3, combine it with the positional signal from head 1, and use the result to produce a single coherent representation."

**The mathematics of why mixing is necessary:**

After concatenation, the output has shape (n × d_model) where the first d_k columns are exclusively head 1's values, the next d_k columns are exclusively head 2's values, etc.

Without W_O: output[i] = [head_1[i] ; head_2[i] ; ... ; head_h[i]] — the j-th position carries exactly one head's information. No cross-head combination is possible downstream.

With W_O ∈ R^{d_model × d_model}: output[i] = concat(heads) @ W_O — now every output dimension is a learned linear combination over all h heads. The model can amplify useful head combinations and suppress unhelpful ones.

Without W_O, you also lose the ability to down-weight poor heads. W_O can learn to zero out the contribution of a head that has collapsed (learned the same pattern as another head).

**An analogy:** imagine 8 expert consultants each write a separate one-page report. W_O is the final synthesis meeting where a coordinator reads all 8 reports and writes one integrated document. Without that meeting, you have 8 separate reports but no synthesis.

---

## Failure Modes

### Head Collapse

Multiple heads learn identical or near-identical attention patterns. This wastes capacity — you effectively have fewer distinct heads than you paid for.

Cause: insufficient diversity in initialization, or the task genuinely only benefits from one type of pattern (rare). The model takes the easy path of learning the same useful pattern h times rather than investing in finding h different patterns.

Detection: compute the pairwise cosine similarity between attention weight matrices from different heads across a validation batch. Similarity above ~0.8 between two heads suggests collapse. Also examine the row norms of W_O partitioned by head block — collapsed heads will have large norms (the model compensates by amplifying their signal) while diverse heads share the load.

Fix during training: attention dropout (standard: 0.1) randomly zeroes out attention weights during training, forcing other heads to compensate for the dropped head. This creates pressure toward diversity. Alternatively, add an explicit diversity loss penalizing high cosine similarity between heads.

### One Head Monopolizing W_O

A single head can dominate the output projection. If head 3 learns the most universally useful pattern (e.g., local syntactic context), W_O may allocate most of its weight to the head-3 block. Other heads contribute near-zero to the final output.

This is a milder form of collapse — the attention patterns may be diverse, but the information from most heads is discarded by W_O. The fix is the same: attention dropout during training, and head pruning post-training.

### Batched Masking Bugs

In practice, sequences in a batch have different lengths and must be padded to the same length. Implementing the attention mask correctly is surprisingly error-prone.

Common bug: mask shape (batch_size, 1, seq_len, seq_len) is misaligned with the attention score tensor (batch_size, n_heads, seq_len, seq_len) during broadcasting. This can cause future tokens to leak through (model cheats), or valid tokens to be masked out (model can't use information it should have).

Always verify masks by running a batch of known examples through the model and checking that the attention weight matrix (after softmax) is zero for all padded positions and all future positions in decoder layers.

### Attention to [PAD] Tokens

Without masking padding positions, the model computes attention weights for padding tokens the same as for real tokens. The model wastes attention capacity on meaningless padding and the Value vectors from padding positions contaminate real token representations.

Fix: before softmax, add -∞ (implemented as a very large negative number, e.g., -1e9 or -1e4 in float16) to the attention scores at all padding positions. After softmax, exp(-∞) = 0 — padding positions receive exactly zero attention weight.

---

## Staff/Principal Interview Depth

**Q1: Why is W_O necessary? Couldn't you just average or sum the head outputs?**

Averaging or summing would lose information about which head contributed what. More critically, it makes cross-head interaction impossible in a structured way.

With summation: output[i] = Σ_h head_h[i], where each head_h[i] = (Σ_j w_{h,i,j} V_{h,j}). The h-th head's influence on position i is bounded by its own value projection W_{V,h}. You cannot form a learned combination like "feature from head 2 times feature from head 5" — summation is a linear operation over heads, and both features would need to be in the same vector position.

With W_O: output = concat(heads) @ W_O. The matrix W_O can express any linear combination of any element from any head. Head-3 dimension 7 can be combined with head-7 dimension 12 in a single output dimension. This is qualitatively richer.

Empirically: removing W_O and replacing with mean-pooling of head outputs degrades performance significantly across tasks. W_O also has the same parameter count as a single Q/K/V projection (d_model²), so the cost-to-benefit ratio is excellent.

---

**Q2: What are Grouped Query Attention (GQA) and Multi-Query Attention (MQA)? Why are they used in production?**

In standard Multi-Head Attention (MHA), each of the h query heads has its own K and V projection. At inference, to avoid recomputing K and V for every new token, they are cached in HBM (the KV cache). KV cache size per token per layer = 2 × h × d_k × bytes_per_element = 2 × d_model × bytes_per_element. For a 70B parameter model serving a 32K context to 100 concurrent users, the KV cache alone requires tens of GB.

Multi-Query Attention (Shazeer, 2019): all h query heads share a single K projection and a single V projection. KV cache size shrinks by a factor of h (the number of heads). With h=32, this is a 32× memory reduction. Quality degrades slightly, especially on tasks requiring diverse attention patterns. Used in: original PaLM, Falcon.

Grouped Query Attention (Ainslie et al., 2023): instead of one shared K/V pair, use g groups of query heads, each group sharing one K/V pair. n_kv_heads = h / g. Quality is between MHA (g=h) and MQA (g=1). GQA with g=8 (e.g., 32 query heads, 8 KV heads) recovers most of MHA quality while reducing KV cache by 4×.

Used in production: LLaMA-2 (70B uses GQA), Mistral-7B (uses GQA), Gemma (uses MQA or GQA). The trade-off is straightforward: training quality vs inference memory and throughput. For models serving many concurrent users on long contexts, KV cache size is often the binding constraint, making GQA/MQA the right choice even at a small quality cost.

---

**Q3: What causes head collapse and how would you diagnose and fix it in a production model?**

Head collapse means multiple heads learn similar attention distributions — they are attending to the same pattern multiple times, wasting model capacity.

Causes: (1) insufficient regularization — gradient descent finds the locally easy path of learning one useful pattern h times rather than finding h distinct patterns; (2) learning rate too high early in training — heads converge to the same attractor before they can diversify; (3) the task may genuinely favor one attention type (rare: most NLP tasks benefit from diverse heads).

Diagnosis:
1. Collect attention weight matrices from all heads on a representative validation batch.
2. For each pair of heads (i, j), compute mean cosine similarity between their flattened attention matrices across the batch. If similarity > 0.8, those heads have collapsed.
3. Partition W_O by head block and compute the L2 norm of each block. Collapsed heads often have outsized norms (the model compensates by amplifying their contribution).
4. Prune one head from each high-similarity pair and fine-tune. If quality is unchanged, collapse was the cause.

Fixes during training: attention dropout (0.1 is the standard value from the original transformer paper). This randomly zeroes entire rows of the attention weight matrix during training. A head that is zeroed out forces the model to rely on other heads, creating diversity pressure.

Post-training: head pruning with retraining. Michel et al. (2019) showed that up to 50% of heads can be pruned with <1% quality loss, confirming widespread collapse in production models. After identifying redundant heads (using gradient-based importance scores), remove them and fine-tune the remaining heads for a few thousand steps to recover quality.

---

**Q4: Prove that multi-head attention has the same total FLOPs as single-head attention with the same d_model.**

Single-head attention with d_model dimensions:

- QK^T per step: (n × d_model) @ (d_model × n) costs 2 × n² × d_model FLOPs.
- Attention × V: (n × n) @ (n × d_model) costs 2 × n² × d_model FLOPs.
- Q, K, V projections (three d_model × d_model matrices): 3 × 2 × n × d_model² FLOPs.
- No output projection needed (or it's a d_model × d_model projection): 2 × n × d_model².

Total: 4 × n² × d_model + 8 × n × d_model².

Multi-head attention with h heads and d_k = d_model / h:

- QK^T per head: 2 × n² × d_k. Across h heads: h × 2 × n² × d_k = 2 × n² × (h × d_k) = 2 × n² × d_model.
- Attention × V per head: 2 × n² × d_k. Across h heads: 2 × n² × d_model. Same.
- Q, K, V projections per head: each is (d_model × d_k), costing 2 × n × d_model × d_k. Three projections, h heads: 3 × h × 2 × n × d_model × d_k = 6 × n × d_model × (h × d_k) = 6 × n × d_model².
- Output projection W_O (d_model × d_model): 2 × n × d_model².

Total: 4 × n² × d_model + 8 × n × d_model².

Identical. The key insight: splitting d_model into h pieces of size d_model/h and processing them independently does not change the total volume of computation. h × (d_model/h)² = d_model²/h per head, times h heads = d_model². The split reorganizes the computation into smaller, parallelizable units on GPU, but the total number of multiply-accumulate operations is unchanged. This is why multi-head attention is free — you get specialization at no extra compute cost.

---

## Key Takeaways

1. **Multi-head attention = multiple attention operations in parallel**, each learning different patterns
2. **Each head** gets a smaller slice of the embedding and runs attention independently
3. **Concatenation + projection** combines all heads' findings back into one vector
4. **Different heads specialize** naturally during training (grammar, meaning, position, etc.)
5. **No extra cost** -- splitting dimensions means multi-head costs the same FLOPs as single-head (proven above)
6. Typical models use **8-96 heads** with **64-128 dimensions per head**
7. **Total parameters per attention block = 4 × d_model²** regardless of the number of heads
8. **FLOPs = 8n·d_model² + 4n²·d_model** — projections dominate for short sequences, attention dominates for long ones
9. **W_O is mandatory** — it is the only layer where heads interact and cross-head combinations are learned
10. **Head collapse, W_O monopolization, mask bugs, and PAD leakage** are the four failure modes to monitor in production

---

## Prerequisites

Before reading this, you should understand:
- [Attention Mechanisms](./attention-mechanisms.md) -- the foundation for this section
- Vectors and matrices (from [Neural Network Fundamentals](../../00-neural-networks/fundamentals/04_neural_network_layers.ipynb))

## Further Reading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) -- Section 3.2
- [A Multiscale Visualization of Attention](https://arxiv.org/abs/1906.05714) -- what heads actually learn
- [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) -- head pruning research
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (Shazeer, 2019) -- MQA

---

[Previous: Attention Mechanisms](./attention-mechanisms.md) | [Back to Architecture Overview](./README.md) | [Next: Positional Encoding](./positional-encoding.md)
