> **What this file covers**
> - 🎯 Why √d_k normalizes dot-product variance to exactly 1 (derived, not guessed)
> - 🧮 Full attention formula with worked example and every symbol labeled
> - ⚠️ 4 failure modes: collapse, spike, causal mask bug, softmax overflow
> - 📊 O(n²·d_k) time, O(n²) memory per layer per head — exact formulas with real numbers
> - 💡 Dot-product vs additive attention: when each wins, historical context
> - 🏭 Flash Attention: why it's mandatory at long contexts, how tiling works
> - 🗺️ Cross-attention design: why Q from decoder and KV from encoder
> - Staff/Principal Q&A with all four hiring levels shown

---

# Attention Mechanisms — Interview Deep-Dive

This file assumes you have read [attention-mechanisms.md](./attention-mechanisms.md) and have the intuition for Q, K, V, softmax, and the basic flow. Everything here is for Staff/Principal depth.

---

## 🧮 The Full Formula

```
🧮 Scaled dot-product attention:

    Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

    Where:
      Q  = query matrix  (n × d_k)   — what each position is looking for
      K  = key matrix    (n × d_k)   — what each position advertises
      V  = value matrix  (n × d_v)   — what each position actually carries
      d_k = dimension of keys — controls the scale factor
      n  = sequence length
```

Step-by-step:
- **Q·Kᵀ** — dot product between every query and every key. Produces an n×n matrix. Entry (i,j) = "how much does position i want to attend to position j?"
- **/ √d_k** — normalize. Without this, large d_k causes scores to grow large, pushing softmax into a near-zero-gradient region. Details below.
- **softmax(...)** — each row becomes a probability distribution over all positions. Every row sums to 1.
- **· V** — weighted average of Value vectors using those weights.

---

## 🗺️ Concept Flow

```
           word has ambiguous meaning (e.g., "it" in "the cat was tired, it slept")
                              │
                              ▼
                 query: "what context helps me understand 'it'?"
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
      check "cat"        check "tired"       check "slept"
      (high match)       (medium match)      (low match)
           │                  │                  │
           └──────────────────┼──────────────────┘
                              ▼
                 blend value info, weighted by match scores
                              │
                              ▼
                 "it" now has a context-aware representation
                 → model resolves pronoun to "cat"
```

The key insight: the weight assigned to each source word is **learned** (through W_Q and W_K), not hand-coded.

---

## 🔬 Why √d_k? The Variance Derivation

This is one of the most common Staff/Principal interview questions. Here is the complete derivation.

**Assumption:** At initialization, entries of W_Q and W_K are drawn from N(0,1), so q and k vectors have entries ≈ N(0,1).

**Dot product variance:**

The dot product q·k = q₁k₁ + q₂k₂ + ... + q_{d_k}k_{d_k}.

Each term qᵢkᵢ is a product of two independent N(0,1) variables:
- E[qᵢkᵢ] = 0
- Var[qᵢkᵢ] = E[qᵢ²]·E[kᵢ²] = 1·1 = 1

Summing d_k independent terms:
- **Var[q·k] = d_k**
- **Std[q·k] = √d_k**

Dividing by √d_k: Var[q·k / √d_k] = d_k / d_k = **1**. Variance normalized to 1.

**Why unit variance matters:**

Softmax saturates when inputs are large. "Saturates" means it assigns near-1 weight to one element and near-0 to everything else.

For vector x where one entry dominates: softmax(x) ≈ [1, 0, 0, ...].

The gradient of softmax at saturation is approximately zero everywhere. When gradients go to zero, weights stop updating — **training stalls**.

Dividing by √d_k keeps dot products near unit scale, where softmax gradients are healthy.

**Why not divide by d_k instead?**

Dividing by d_k would give Var = 1/d_k. For large d_k, this is very small — attention scores become near-uniform, and the model can't differentiate relevant from irrelevant positions. You need variance near 1, not near zero.

**The assumption breaks down after training:**

After many gradient steps, Q and K distributions shift away from N(0,1). The √d_k factor is only exactly correct at initialization. Some practitioners use a **learned temperature** (a trainable scalar replacing the fixed 1/√d_k) to adapt as training progresses. This is seen in some production implementations but rarely changes results significantly.

---

## 📊 Complexity Analysis

### Time Complexity

The dominant operation is the QKᵀ matrix multiply.

- **QKᵀ**: n queries, each doing a dot product with n keys, each dot product costs d_k multiplications → **O(n² · d_k)**
- **Attention weights × V**: n² weights, each multiplied against d_v-dimensional value → **O(n² · d_v)**
- **Q, K, V projections**: each is (n × d_model) × (d_model × d_k) → **O(n · d_model · d_k)** per projection, ×3
- **Full single-head**: O(n² · d_model + n · d_model²)

The n² term dominates at long sequence lengths.

### Memory Complexity

The attention weight matrix has shape (n × n). Must be stored during both forward and backward passes.

**Memory = O(n²) per layer per head.**

Real numbers:
- 4,096-token sequence, 32 heads, float32: 4096 × 4096 × 32 ≈ 537M entries → ~2 GB per layer for attention weights alone
- With 96 layers (GPT-3 scale): attention matrices alone would exceed 190 GB
- At 128K tokens (GPT-4 context): 128K² × 32 ≈ 524 billion entries per layer — impossible to materialize without tiling

This is why **Flash Attention is not optional** for long contexts.

### Parameter Count

Parameters live in the projection matrices, not in the attention computation:
- W_Q: d_model × d_k
- W_K: d_model × d_k
- W_V: d_model × d_v (usually d_v = d_k)
- No parameters in dot product, scale, or softmax

**Total for one attention head: 3 × d_model × d_k**

For multi-head attention with h heads and d_k = d_model/h: 3 × d_model² plus the output projection d_model². **Total: 4 × d_model²**.

---

## ⚠️ Failure Modes

### Attention Collapse (Uniform Weights)

At random initialization, Q and K are unrelated. Dot products are near zero. Softmax of near-zero inputs produces uniform weights: every position gets weight ≈ 1/n.

The output is just an average of all value vectors — no information selection. This is acceptable at initialization; the model learns to differentiate during training. But if it persists (learning rate issues, bad initialization), training stalls.

### Attention Spike (Softmax Saturation)

The opposite: after training, all attention concentrates on one token. Period tokens "." and separator tokens are infamous "attention sinks" — they absorb weight even when carrying no relevant information.

Sometimes tolerated (gives the model a "null" attention target), but can starve important tokens of attention weight.

### Future Leakage Bug (Missing Causal Mask)

Decoder models must not allow position t to attend to positions t+1, t+2, ... (they don't exist at inference). Enforced by adding −∞ to future positions before softmax, so exp(−∞) = 0.

If you forget the mask during training: the model learns to use future information. Training loss drops fast (it's cheating). But at inference, future tokens don't exist — the model fails. **This is a silent, common bug.** Always verify by checking that the attention weight matrix is lower-triangular after softmax in a decoder model.

### Numerical Instability in Softmax

Computing exp(x) directly for large x overflows float32 (exp(90) already exceeds the maximum). Pre-√d_k scaling, dot products in high-dimensional spaces routinely caused NaN gradients.

Always use numerically stable softmax:

```
softmax(x)ᵢ = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
```

Subtracting max(x) does not change the output (the max term cancels) but prevents overflow. The largest exponent computed is exp(0) = 1.

---

## 💡 Design Trade-offs: Dot-Product vs Additive Attention

| | Dot-product attention | Additive (Bahdanau) attention |
|---|---|---|
| Formula | Q·Kᵀ / √d_k | v·tanh(W_q·Q + W_k·K) |
| Extra parameters | None | v, W_q, W_k ∈ R^{d×d_k} |
| Best when | d_k ≥ 32 (high expressiveness + GPU efficiency) | d_k < 32 (additive MLP more expressive) |
| GPU efficiency | ✅ Single matrix multiply, fully optimized | ❌ Sequential MLP, harder to parallelize |
| Historical use | Transformers (Vaswani et al. 2017+) | Bahdanau NMT (2015) |

**Why Bahdanau used additive in 2015:** The √d_k scaling fix didn't exist yet. Without it, dot-product attention was unstable at higher dimensions. Additive attention avoided this problem. Once Vaswani introduced √d_k, dot-product became dominant.

**Kernel interpretation:** Both are similarity functions between Q and K. Dot-product uses the linear kernel (inner product). Additive uses a learned tanh MLP kernel, which is richer but slower. At small d_k, the linear kernel is low-rank and the MLP kernel consistently outperforms (Luong et al., 2015 showed this empirically).

---

## 🏭 Flash Attention: Why It's Mandatory at Long Contexts

Standard attention materializes the n×n matrix in HBM (GPU high-bandwidth memory). For n=4096 with 32 heads and float16: 4096² × 32 × 2 bytes ≈ 1 GB per layer — and that's just for attention weights.

**The bottleneck is memory bandwidth, not arithmetic.** Reading and writing the n×n matrix to/from HBM takes more wall-clock time than the actual multiply-accumulate operations.

**Flash Attention's solution: tile the computation.**

```
Standard attention:                    Flash Attention:

Load full Q from HBM                  For each tile of Q_i:
Load full K from HBM                    For each tile of K_j, V_j:
Compute QK^T  ← n×n matrix               Load tile from HBM
  (materializes in HBM)                  Compute partial scores
Load V from HBM                          Update running max m_i
Compute softmax(QK^T)·V                  Update running sum l_i
Write output to HBM                      Accumulate partial output O_i
                                       Finalize with correct normalization
                                       Write output (size n×d_v) to HBM
```

**Online softmax algorithm:** To compute softmax without seeing the full row, maintain two running statistics per query position: m_i (running max of scores) and l_i (running sum of exp(score − m_i)). When a new tile arrives, rescale the accumulated output using the updated normalization. This produces exact softmax without materializing the full row.

**Result:**
- Memory footprint: O(n) instead of O(n²) — the n×n matrix is never stored
- HBM reads: O(n²/M) where M = SRAM size (~20MB on A100), vs O(n²) before
- Wall-clock speedup: 2–4× on A100, larger at longer contexts
- Output: **exact same result as standard attention** — not approximate

Flash Attention 2 (2023) further improves GPU utilization by splitting work across query blocks, getting 50–70% model FLOP utilization vs 25–35% for Flash Attention 1.

---

## Staff/Principal Interview Depth

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
*Interviewer:* Correct and useful answer. The candidate grasps the saturation problem and knows that scaling prevents it. What's missing: why √d_k specifically? Why not clip the values, or divide by d_k, or use a learned scale? The candidate can't derive the √d_k from first principles.
*Criteria — Met:* Saturation problem, gradient vanishing at large scale / *Missing:* Variance derivation, reason for √ vs other normalizations

---
**Hire**
*Interviewee:* "The reason is variance. If q and k are vectors with entries drawn i.i.d. from N(0,1), their dot product is a sum of d_k independent terms, each with variance 1. So Var(q·k) = d_k, and the standard deviation is √d_k. Dividing by √d_k normalizes the variance back to 1. We want unit variance because softmax gradients are maximized near zero — large inputs saturate softmax, gradients go to zero, and training stalls. Dividing by d_k would over-correct: variance becomes 1/d_k, which is very small for large d_k, making attention near-uniform and unable to differentiate relevant from irrelevant positions."
*Interviewer:* Strong. The candidate derives √d_k from the variance analysis, correctly distinguishes √d_k from d_k, and gives the gradient reasoning. What would push to Strong Hire: mentioning the assumption (N(0,1) initialization) explicitly, noting what happens after training steps when distributions shift, and awareness of learned temperature scaling.
*Criteria — Met:* Variance derivation (Var=d_k), √d_k normalization to unit variance, softmax saturation at large input, over-correction argument for d_k / *Missing:* Assumption about initialization distribution, learned temperature as alternative

---
**Strong Hire**
*Interviewee:* "The derivation starts with the initialization assumption: W_Q and W_K are initialized such that Q and K vectors have entries ≈ N(0,1). Under this assumption, the dot product q·k = Σᵢ qᵢkᵢ. Each term qᵢkᵢ is a product of two independent N(0,1) variables, which has mean 0 and variance 1. Summing d_k such terms: Var(q·k) = d_k, standard deviation = √d_k. Dividing by √d_k gives Var(q·k / √d_k) = 1. We specifically want unit variance — not just 'small' — because that's where softmax operates in its highest-gradient regime. Too large → saturation, gradient collapse. Too small → near-uniform attention, model can't select. The √d_k choice is derived directly from the initialization distribution. Two practical caveats: first, after many training steps, Q and K distributions shift and the initialization argument no longer holds exactly — some practitioners use learned temperature scaling (replacing the fixed 1/√d_k with a learned scalar) to adapt. Second, in float16 training, you need to be careful even after scaling because exp overflow is possible — production implementations use numerically stable softmax with the max subtraction."
*Interviewer:* This is the answer. Derives from first principles, explains why unit variance is the target (not just "smaller"), gives both caveats (distribution shift after training, float16 overflow), and mentions the learned temperature alternative as a real production consideration. The level of precision — calling out the initialization distribution explicitly and noting it breaks down — is what separates staff-level reasoning from senior-level recall.
*Criteria — Met:* Full variance derivation, unit variance target reasoning, gradient regime analysis, √ vs d_k comparison, float16 overflow, learned temperature alternative, distribution shift caveat

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

🎯 1. **√d_k normalizes dot-product variance to 1** — this is derived from the N(0,1) initialization assumption, not arbitrary
2. **Attention is O(n²·d_model) time and O(n²) memory** per layer per head — the n² term is the scaling bottleneck
🎯 3. **Flash Attention reduces memory to O(n)** by tiling and using online softmax normalization — same result, different computation order
4. **Causal mask must be verified**: attention matrix should be lower-triangular after softmax in a decoder model
⚠️ 5. **Attention spike (softmax saturation) and attention collapse (uniform weights) are opposite failure modes** — both indicate problems worth diagnosing
6. **Dot-product attention beats additive at d_k ≥ 32**; additive beats it at very small d_k due to richer MLP kernel
7. **Cross-attention Q=decoder, KV=encoder** is mechanistically necessary — encoder KV is fixed and reused at every decoder step
8. **Teacher forcing creates exposure bias** — attention amplifies this more than RNNs because wrong tokens enter the KV pool universally
9. **Parameters in attention: 4 × d_model²** total (three input projections + output projection), independent of head count
🎯 10. **Float16 overflow in softmax is a silent killer** — always subtract max(x) before computing exp

---

**Further Reading**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Flash Attention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) — IO-aware exact attention
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) (Dao, 2023) — improved parallelism
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — comprehensive overview with additive vs dot-product comparison

---

[← Back to Attention Mechanisms (Layer 1)](./attention-mechanisms.md) | [Next: Multi-Head Attention](./multi-head-attention.md)
