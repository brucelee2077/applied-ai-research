> **What this file covers**
> - 🎯 Why W_O is not optional — the block partitioning proof, cross-head interaction
> - 🧮 Exact parameter count: 4 × d_model² — derived term by term
> - 📊 Full FLOPs analysis: 8n·d_model² + 4n²·d_model — with crossover analysis
> - ⚠️ 4 failure modes: head collapse, W_O monopolization, mask bugs, PAD leakage
> - 💡 Multi-head FLOPs = single-head FLOPs — algebraic proof with GPU parallelism implications
> - 🏭 GQA and MQA: KV cache reduction in production — exact formulas and model examples
> - Staff/Principal Q&A with all four hiring levels shown

---

# Multi-Head Attention — Interview Deep-Dive

This file assumes you have read [multi-head-attention.md](./multi-head-attention.md) and [attention-mechanisms-interview.md](./attention-mechanisms-interview.md). Everything here is for Staff/Principal depth.

---

## 🧮 The Full Formula

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) × W_O

headᵢ = Attention(Q·W_Qᵢ, K·W_Kᵢ, V·W_Vᵢ)
      = softmax( (Q·W_Qᵢ)(K·W_Kᵢ)ᵀ / √d_k ) · (V·W_Vᵢ)

Where:
  h = number of heads
  d_k = d_model / h   (head dimension)
  W_Qᵢ ∈ R^{d_model × d_k}
  W_Kᵢ ∈ R^{d_model × d_k}
  W_Vᵢ ∈ R^{d_model × d_v}  (d_v = d_k in standard implementations)
  W_O  ∈ R^{d_model × d_model}
```

---

## 🗺️ Information Flow

```
      same input embedding (n × d_model)
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
  ×W_Q₁     ×W_K₁     ×W_V₁       Head 1 projections
    │          │          │
    └──────────┼──────────┘
               │
         attention(Q₁, K₁, V₁)    Head 1 output: n × d_k
               │
    (same pattern for h heads, in parallel)
               │
               ▼
    Concat(h₁, h₂, ..., hₕ)       n × d_model  (block structure: no head mixing yet)
               │
              ×W_O                 learned cross-head mixture
               │
               ▼
          final output             n × d_model  (all heads combined)
```

**Key: before W_O, each position in the output is exclusively owned by one head's output.
After W_O, every output dimension is a linear combination over all heads.**

---

## 🔬 Why W_O Is Not Optional

After concatenation, the output has shape (n × d_model), but it has a strict block structure:

```
output[i] = [head_1(i) ; head_2(i) ; ... ; head_h(i)]

            ← d_k →  ← d_k →          ← d_k →
            [  head1  |  head2  | ... |  headh  ]
```

Without W_O: position 0–d_k in the output carries exclusively head-1 information. Position d_k–2d_k carries exclusively head-2 information. There is no way to form a feature that combines head-3 dimension 5 with head-7 dimension 12. Cross-head combination is impossible.

With W_O ∈ R^{d_model × d_model}: output = concat(heads) @ W_O. Now every output dimension is a learned linear combination over all h heads. The model can amplify useful combinations and suppress unhelpful ones.

**Summation would be even worse than W_O:** averaging the h head outputs would give equal weight to every head and destroy all diversity — a constrained version of W_O where all weights are fixed to 1/h. W_O can learn to down-weight collapsed or redundant heads.

**Secondary role:** W_O can learn to zero out the contribution of a collapsed head. The d_k × d_model block of W_O corresponding to a redundant head will have small Frobenius norm. Michel et al. (2019) confirmed this empirically — W_O block norms correlate with head importance.

---

## 📊 Parameter Count: Exact Derivation

For h heads and d_model total embedding size, each head uses d_k = d_model / h.

**Per head (three projections):**

```
W_Qᵢ: d_model × d_k = d_model × (d_model/h)
W_Kᵢ: d_model × d_k = d_model × (d_model/h)
W_Vᵢ: d_model × d_k = d_model × (d_model/h)
```

**Across all h heads:**

```
3 × h × (d_model × d_k)
= 3 × h × d_model × (d_model / h)
= 3 × d_model²
```

The h cancels. Q/K/V parameters are always **3 × d_model²**, regardless of head count.

**Output projection W_O:**

```
W_O: (h × d_k) × d_model = d_model × d_model = d_model²
```

**Total for one attention block:**

```
3 × d_model²   (Q, K, V projections)
+   d_model²   (output projection)
─────────────
4 × d_model²
```

**Worked examples:**

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

---

## 📊 FLOPs Analysis

For n tokens, d_model dimensions, h heads, d_k = d_model/h:

**Q, K, V projections (3 matrix multiplies):**
Each projection is (n × d_model) @ (d_model × d_k), costing 2 × n × d_model × d_k per projection per head.
With h heads: 3 × h × 2 × n × d_model × d_k = 3 × 2 × n × d_model × (h × d_k) = **6 × n × d_model²**

**QKᵀ (attention scores) per head:**
(n × d_k) @ (d_k × n) = 2 × n² × d_k per head.
Across h heads: **2 × n² × d_model**

**Attention weights × V per head:**
(n × n) @ (n × d_k) = 2 × n² × d_k per head.
Across h heads: **2 × n² × d_model**

**W_O projection:**
(n × d_model) @ (d_model × d_model) = **2 × n × d_model²**

**Total FLOPs for one attention layer:**

```
6n·d_model²  +  2n²·d_model  +  2n²·d_model  +  2n·d_model²
= 8n·d_model²  +  4n²·d_model
```

**Crossover analysis:**

Set 8n·d_model² = 4n²·d_model → n = 2·d_model.

```
n=512,  d_model=512:    projection=1.07B, attention=536M   → projections dominate (2:1)
n=1024, d_model=512:    projection=2.15B, attention=2.15B  → equal (n ≈ 2×d_model ✓)
n=8192, d_model=512:    projection=17.2B, attention=137B   → attention dominates (8:1)
```

Long-context models require special attention algorithms (Flash Attention, sliding window, linear attention) because the n² attention terms swamp everything else at large n.

---

## ⚠️ Failure Modes

### Head Collapse

Multiple heads learn identical or near-identical attention patterns. This wastes capacity — you effectively have fewer distinct heads than you paid for.

**Cause:** Gradient descent finds the path of least resistance. If one attention pattern is useful, learning it h times is easier than finding h genuinely different useful patterns.

**Detection:**
- Compute pairwise cosine similarity between attention weight matrices from different heads on a validation batch. Similarity > ~0.8 suggests collapse.
- Check W_O block norms partitioned by head — collapsed heads tend to get large norms (the model compensates by amplifying their signal).

**Fix:** Attention dropout (standard: 0.1) randomly zeroes out attention weight rows during training, forcing other heads to compensate. Explicit diversity losses are used in some research settings but rarely in production due to hyperparameter sensitivity.

### One Head Monopolizing W_O

A single head dominates the output projection. Other heads contribute near-zero to the final output. The attention patterns may be diverse, but W_O discards most of them.

Detection: examine the Frobenius norm of each d_k × d_model block of W_O corresponding to each head. Disproportionately large blocks indicate monopolization.

Fix: same as head collapse — attention dropout, and post-training head pruning (Michel et al. 2019 showed 50% of heads can be pruned with < 1% quality loss).

### Batched Masking Bugs

In practice, sequences in a batch have different lengths and must be padded. Implementing the attention mask correctly is surprisingly error-prone.

Common bug: mask shape (batch_size, 1, seq_len, seq_len) is misaligned with attention score tensor (batch_size, n_heads, seq_len, seq_len) during broadcasting. This can cause future tokens to leak through (model cheats on training) or valid tokens to be masked out.

Always verify by running a batch of known examples and checking the attention weight matrix is zero for all padded positions and all future positions in decoder layers.

### Attention to [PAD] Tokens

Without masking padding positions, the model computes attention weights for padding tokens the same as for real tokens. The model wastes attention capacity on meaningless padding and Value vectors from padding positions contaminate real token representations.

Fix: add −∞ to attention scores at all padding positions before softmax. exp(−∞) = 0 — padding positions receive exactly zero attention weight.

---

## 🏭 GQA and MQA: KV Cache Reduction in Production

At inference, every forward pass requires reading K and V for all previous tokens from HBM. The KV cache size is:

```
Standard MHA: 2 × L × h × d_k × S × B × element_size

Where:
  L = number of layers
  h = number of heads
  d_k = head dimension
  S = sequence length
  B = batch size
  element_size = 2 bytes (float16)

LLaMA-2 70B example: 2 × 80 × 64 × 128 × 32000 × 32 × 2 bytes ≈ 214 GB
— more than the model weights themselves
```

**Multi-Query Attention (MQA)** (Shazeer, 2019): All h query heads share one K head and one V head. KV cache shrinks by factor h. Quality degrades because all heads must share the same K/V subspace — the model loses the ability to have diverse K/V patterns. Used in original PaLM and Falcon.

**Grouped Query Attention (GQA)** (Ainslie et al., 2023): n_kv_heads groups, each shared by h/n_kv_heads query heads. Interpolates between MHA (n_kv_heads = h) and MQA (n_kv_heads = 1).

```
| Variant | KV heads | Cache reduction | Quality vs MHA |
|---------|----------|-----------------|----------------|
| MHA     | h = 32   | 1×              | Baseline       |
| GQA     | 8        | 4×              | < 1% loss      |
| MQA     | 1        | 32×             | Notable loss   |
```

LLaMA-2 70B uses GQA with n_kv_heads=8, h=64 → 8× cache reduction. Mistral-7B also uses GQA. LLaMA-3 uses n_kv_heads=8 with h=32 → 4× cache reduction.

**Why GQA preserves quality better than MQA:** More K/V pairs means more degrees of freedom in the K/V subspaces, allowing heads within a group to specialize their queries relative to a shared but less constrained K/V structure.

---

## Staff/Principal Interview Depth

---

**Q1: Why is W_O necessary? Couldn't you just average or sum the head outputs?**

---
**No Hire**
*Interviewee:* "You need W_O to combine the heads and bring the dimensions back to the right size."
*Interviewer:* The candidate knows W_O exists and roughly what it does. But "bring dimensions back to the right size" misses the point — concatenation already restores d_model. The candidate can't explain *why* a learned projection is better than a simpler aggregation.
*Criteria — Met:* Knows W_O exists, vaguely knows it combines heads / *Missing:* Why learned projection vs sum/mean, cross-head interaction argument, parameter count reasoning

---
**Weak Hire**
*Interviewee:* "If you just summed the heads, you'd lose the structure — each head's output would just be added together and you couldn't tell which head contributed what. W_O is a learned projection that can weight each head's contribution differently and mix information across heads."
*Interviewer:* Correct intuition. The candidate gets that W_O is a learned combination. What's missing: the mathematical argument for why summation prevents cross-head combination, and the fact that without W_O heads are completely partitioned in the output.
*Criteria — Met:* Learned weighting argument, mixing intuition / *Missing:* Mathematical partitioning argument, cross-head combination impossibility without W_O

---
**Hire**
*Interviewee:* "After concatenation, the output vector has shape (n, d_model) but the first d_k dimensions exclusively come from head 1, the next d_k from head 2, and so on. Without W_O, those partitions are permanent — downstream layers see a concatenated vector but can't mix features across head boundaries. Head 3's insight at dimension 192 can never be combined with head 7's insight at dimension 448 in a single output feature. W_O ∈ R^{d_model × d_model} is the only place where cross-head combination happens. With W_O, every output dimension is a learned linear combination over all h heads. Summation would be even worse than W_O: you'd be constraining the combination to equal weights, discarding the diversity that heads built up. W_O also has the same parameter count as one Q/K/V projection (d_model²), so the cost-to-benefit ratio is excellent."
*Interviewer:* Strong. The candidate gives the partitioning argument precisely, explains why cross-head combination is impossible without W_O, and correctly notes that summation is a constrained special case. What would push to Strong Hire: noting that W_O can zero out collapsed heads (setting their block to near-zero), and the connection to why multi-head FLOPs equals single-head FLOPs.
*Criteria — Met:* Partitioning argument, cross-head combination impossibility, summation is constrained, W_O parameter cost / *Missing:* W_O as head pruning mechanism, connection to FLOP equivalence

---
**Strong Hire**
*Interviewee:* "The concatenation after multi-head attention creates a block structure: output[i] = [h_1(i) ; h_2(i) ; ... ; h_H(i)] where h_k(i) ∈ R^{d_k}. Without W_O, this representation is permanently partitioned — downstream weight matrices in the next layer see a full d_model vector, but dimensions [0, d_k) only contain head-1 information and dimensions [(H-1)d_k, Hd_k) only contain head-H information. You cannot form a feature that combines head-3 dimension 5 with head-7 dimension 12 without a cross-head mixing layer. W_O is that layer. W_O ∈ R^{d_model × d_model}: output = concat(heads) @ W_O, where now any output dimension can be any linear combination over all heads. Two secondary roles that are often overlooked: first, W_O can zero out collapsed heads — if head 3 learned the same pattern as head 1, the model can learn to set the head-3 block of W_O near zero, effectively pruning it. This is why you see sparse blocks in W_O when you analyze trained models (Michel et al., 2019 confirmed this empirically). Second, without W_O, multi-head would not be equivalent to single-head attention in FLOPs — the output projection is what closes the FLOP budget to 4 × d_model² total. Averaging would cost nothing but would also destroy the cross-head combination capacity permanently."
*Interviewer:* This is staff-level. The candidate gives the formal partitioning argument, explains both the primary (cross-head combination) and secondary (head pruning, FLOP budget) roles, and references the Michel et al. empirical finding. Volunteering "this is why you see sparse blocks in W_O when you analyze trained models" is the sign of someone who has actually inspected model internals, not just read the paper.
*Criteria — Met:* Block partitioning formalism, cross-head impossibility proof, two secondary roles, head pruning mechanism, FLOP budget argument, empirical reference

---

**Q2: What are Grouped Query Attention (GQA) and Multi-Query Attention (MQA)? Why are they used in production?**

---
**No Hire**
*Interviewee:* "GQA is a way to make attention faster by grouping queries. I think it's used in LLaMA."
*Interviewer:* The candidate knows the name and a usage, but has no mechanistic understanding. "Grouping queries" could mean many things. The candidate can't explain what changes, why it helps, or what's lost.
*Criteria — Met:* Knows GQA exists, association with LLaMA / *Missing:* Mechanism, KV cache motivation, quality trade-off, comparison with MQA

---
**Weak Hire**
*Interviewee:* "GQA and MQA reduce the number of K and V projection heads while keeping the same number of Q heads. In MQA, all query heads share one K and V. In GQA, queries are grouped and each group shares a K and V. This reduces the KV cache size, which is important for memory during inference."
*Interviewer:* Correct and concise. The candidate understands both variants and the KV cache motivation. What's missing: the specific reduction factors, when to use each, production examples with numbers, and the quality trade-off.
*Criteria — Met:* MQA and GQA mechanisms, KV cache motivation / *Missing:* Reduction factors, quality trade-off, production examples with numbers

---
**Hire**
*Interviewee:* "Standard MHA has h query heads, h key heads, and h value heads. The KV cache at inference stores K and V for all previous tokens: 2 × h × d_k per token per layer = 2 × d_model per token per layer. For a 70B model with 80 layers, 32K context, and batch_size=64 in float16, that's 80 × 2 × 12288 × 32000 × 64 × 2 bytes ≈ several hundred GB — often more than the model weights. MQA (Shazeer, 2019): all h query heads share one K and one V projection. KV cache shrinks by factor h — a 32× reduction for h=32. Quality degrades, especially on long-context tasks, because all heads must share the same K/V structure. GQA (Ainslie et al., 2023): n_kv_heads groups, each shared by h/n_kv_heads query heads. With n_kv_heads=8 (and h=32), KV cache is 4× smaller than MHA, quality is close to MHA. LLaMA-2 70B uses GQA with n_kv_heads=8, Mistral-7B uses GQA too. The production trade-off: GQA gives most of MHA quality with a significant KV cache reduction, making it the default for models serving long contexts."
*Interviewer:* Strong. The candidate derives the KV cache formula, gives a real example with numbers, explains the quality trade-off correctly, and gives production model examples. What would push to Strong Hire: discussing why MQA degrades more than GQA (fewer unique K/V patterns = less diversity), how to choose n_kv_heads in practice, and awareness of MLA (Multi-head Latent Attention) as the next step.
*Criteria — Met:* KV cache formula derivation, concrete size example, MQA/GQA mechanism, reduction factors, quality trade-off, production examples / *Missing:* Why MQA degrades more than GQA, n_kv_heads selection, MLA

---
**Strong Hire**
*Interviewee:* "The KV cache bottleneck: at inference, every forward pass requires reading K and V for all previous tokens from HBM. For standard MHA, KV cache = 2 × L × h × d_k × S × B × element_size, where L=layers, h=heads, d_k=head_dim, S=sequence_length, B=batch_size. For LLaMA-2 70B: 2 × 80 × 64 × 128 × 32000 × 32 × 2 bytes ≈ 214GB — impossible on a single A100. Multi-Query Attention (Shazeer, 2019): reduce to 1 K head and 1 V head. Cache shrinks by h=64×. This was used in original PaLM and Falcon. Quality drop is real: with one K/V pair, all query heads must project onto the same key-value subspace. The model loses the ability to have diverse K/V patterns, which hurts tasks requiring multiple attention modes. Grouped Query Attention (Ainslie et al., 2023): n_kv_heads groups. Each group's queries share one K/V pair. GQA interpolates between MHA (n_kv_heads = h) and MQA (n_kv_heads = 1). With n_kv_heads = h/4 (LLaMA-3 uses n_kv_heads=8 with h=32), cache reduction is 4× with typically < 1% quality loss on most benchmarks. Why does GQA preserve quality better than MQA? More K/V pairs means more degrees of freedom in the K/V subspaces, allowing heads within a group to specialize their queries relative to a shared but not overly constrained K/V structure. In practice: choose n_kv_heads to fit the KV cache within memory budget, then verify quality on your specific task distribution. DeepSeek-V2 takes this further with MLA (Multi-head Latent Attention), compressing the KV cache via low-rank projection — achieving further reduction while preserving more diversity than GQA by keeping the full set of Q heads with diverse projections."
*Interviewer:* Staff-level. The candidate derives the exact KV cache formula, explains the quality degradation mechanism for MQA at the subspace level, gives the practical GQA selection heuristic, and knows MLA as the current frontier. The explanation of *why* GQA preserves quality better than MQA — more degrees of freedom in K/V subspaces — is the mathematical insight that distinguishes this from a Hire answer.
*Criteria — Met:* Full KV cache formula, MQA/GQA mechanisms, subspace diversity argument for quality, production model examples, n_kv_heads selection heuristic, MLA awareness

---

**Q3: What causes head collapse and how would you diagnose and fix it in a production model?**

---
**No Hire**
*Interviewee:* "Head collapse is when the heads all learn the same thing. You'd use more heads or more regularization."
*Interviewer:* The candidate knows the term and guesses at solutions. Neither "more heads" nor "more regularization" (unspecified) is a principled response to head collapse. There's no discussion of causes, diagnosis, or the specific regularization technique that addresses it.
*Criteria — Met:* Definition of head collapse / *Missing:* Causes, diagnostic procedure, specific fixes (attention dropout, diversity loss, head pruning)

---
**Weak Hire**
*Interviewee:* "Head collapse is when multiple attention heads learn similar patterns, wasting model capacity. You can detect it by comparing the attention weight matrices of different heads — if they're similar, heads have collapsed. Using attention dropout during training can help force diversity. You can also prune redundant heads after training."
*Interviewer:* Correct diagnosis approach and correct fixes. The candidate knows attention dropout and head pruning. What's missing: why gradient descent causes collapse in the first place (the path of least resistance), how to quantify the similarity metric precisely, and the specific diagnostic steps for a production model.
*Criteria — Met:* Definition, similarity-based detection, attention dropout, head pruning / *Missing:* Root cause in gradient dynamics, precise similarity metric, production diagnostic procedure

---
**Hire**
*Interviewee:* "Cause: gradient descent finds the path of least resistance. If one attention pattern is useful, learning it once is hard, so learning it h times is easier than finding h genuinely different useful patterns. Especially early in training with high learning rates, heads tend to converge to the same attractor. Detection: collect attention weight matrices from all heads on a validation batch. For each pair (i, j), compute mean cosine similarity between flattened attention matrices. Similarity > 0.8 suggests collapse. Also check W_O's block norms partitioned by head — collapsed heads tend to get large norms because the model compensates by amplifying their contribution, leaving other heads underweighted. Fix during training: attention dropout (0.1, from the original transformer paper) randomly zeroes out attention weight rows, forcing the model to rely on other heads and creating diversity pressure. Fix post-training: head pruning. Identify heads with high pairwise similarity, remove one from each pair, fine-tune the remaining heads for a few thousand steps. Michel et al. (2019) showed up to 50% of heads can be pruned with < 1% quality loss."
*Interviewer:* Strong. Root cause correctly explained (gradient path of least resistance), detection procedure is precise (cosine similarity threshold, W_O block norms), fixes are specific. What would push to Strong Hire: discussing how to set the similarity threshold empirically, the diversity loss as an explicit training objective, and the difference between attention collapse (attention patterns) vs W_O collapse (head contribution to output).
*Criteria — Met:* Root cause, cosine similarity detection, W_O block norm analysis, attention dropout mechanism, pruning with reference / *Missing:* Diversity loss training objective, two types of collapse distinction

---
**Strong Hire**
*Interviewee:* "Head collapse has two distinct forms that require different diagnostics. Attention pattern collapse: multiple heads produce similar attention weight matrices — they are attending to the same patterns (e.g., both focusing on local syntax). W_O contribution collapse: attention patterns may be diverse, but W_O has learned to zero out most heads' contributions, effectively making the output dominated by one or two heads. Both waste capacity but via different mechanisms. Diagnosis: First, compute Jensen-Shannon divergence (or cosine similarity on flattened softmax outputs) between all head pairs on 100+ validation examples. JSD < 0.1 (or cosine sim > 0.8) between a pair indicates pattern collapse. Second, compute the Frobenius norm of each d_k × d_model block of W_O corresponding to each head. Disproportionately large blocks indicate W_O collapse — that head dominates the output. Third, a gradient-based importance score: compute the gradient of the loss with respect to each head's output, average across the validation set — heads with near-zero gradient sensitivity are effectively pruned already. Training-time fix: attention dropout (0.1) is standard. An explicit diversity loss adds L_div = Σ_{i≠j} sim(A_i, A_j) to the training objective, penalizing correlated heads. Used in some research models but rarely in production because the hyperparameter sensitivity makes it hard to tune. Post-training: gradient-based pruning (Voita et al., 2019, Michel et al., 2019). Use importance scores to rank heads, remove below-threshold heads, fine-tune for 3–5K steps. In practice, 40–60% of heads in large models can be pruned without measurable quality degradation on standard benchmarks — a strong signal that collapse is endemic."
*Interviewer:* This is exactly staff-level. Two distinct forms of collapse, three distinct diagnostic methods (JSD, W_O block norms, gradient sensitivity), principled fix with explicit diversity loss and calibrated judgment about its production viability. Citing specific papers with their conclusions demonstrates that the candidate has read and synthesized the research, not just paraphrased a blog post.
*Criteria — Met:* Two collapse forms, three diagnostic methods, training-time fixes with trade-offs, post-training pruning with research references, calibrated judgment about production viability

---

**Q4: Prove that multi-head attention has the same total FLOPs as single-head attention with the same d_model.**

---
**No Hire**
*Interviewee:* "Because each head is smaller, so they add up to the same amount."
*Interviewer:* The intuition is right but the answer is not a proof. There's no notation, no derivation, and no explanation of *why* the computation factoring works out. A staff interview expects a derivation, not an appeal to intuition.
*Criteria — Met:* Correct intuition / *Missing:* Any mathematical derivation

---
**Weak Hire**
*Interviewee:* "Single-head attention computes QK^T which costs O(n² × d_model) FLOPs. Multi-head splits into h heads of size d_model/h, and each head costs O(n² × d_model/h). Across h heads, that's O(h × n² × d_model/h) = O(n² × d_model). Same."
*Interviewer:* Correct at the O() level for the attention computation. The candidate correctly shows the h cancellation. What's missing: the full FLOPs derivation covering Q/K/V projections, W_O, and exact constants (not just O()), and the extension to the projection terms which also cancel.
*Criteria — Met:* Attention QK^T FLOP cancellation / *Missing:* Projection terms, W_O, exact constants, complete derivation

---
**Hire**
*Interviewee:* "Single-head attention with d_k = d_model: three projections cost 3 × 2 × n × d_model² FLOPs. QK^T costs 2 × n² × d_model. Attention × V costs 2 × n² × d_model. Output projection W_O costs 2 × n × d_model². Total: 8n·d_model² + 4n²·d_model. Multi-head with h heads and d_k = d_model/h: Q/K/V projections: each projection is (d_model × d_k), h heads, 3 projections → 3 × h × 2 × n × d_model × d_k = 6n × d_model × (h × d_k) = 6n × d_model². QK^T per head: 2 × n² × d_k, across h heads → h × 2 × n² × d_k = 2n² × (h × d_k) = 2n² × d_model. Attention × V: same → 2n² × d_model. W_O: 2n × d_model². Total: 8n·d_model² + 4n²·d_model. Identical. The key step: h × d_k = h × (d_model/h) = d_model. All the h factors cancel exactly."
*Interviewer:* Clean, complete derivation. All terms are covered, exact constants are given, and the key algebraic step is called out. What would push to Strong Hire: an intuitive explanation of *why* this works (splitting computation into smaller independent units doesn't change total work if the problem structure is linear), and discussing what this means for GPU parallelism benefit — multi-head is not just "same cost" but "same cost with better GPU utilization."
*Criteria — Met:* Full derivation of both cases, all terms (projections, QKV, W_O), exact constants, algebraic cancellation shown / *Missing:* Intuitive explanation, GPU parallelism benefit

---
**Strong Hire**
*Interviewee:* "Let me derive both cases with exact FLOPs and then give the algebraic insight. Single-head, d_k = d_model, n tokens: Q/K/V projections: 3 × (2nd_model²) = 6nd_model². QK^T: 2n²d_model. Softmax: O(n²) negligible. AV: 2n²d_model. W_O: 2nd_model². Total = 8nd_model² + 4n²d_model. Multi-head, h heads, d_k = d_model/h: Q/K/V per head: 2nd_model·d_k per projection. Three projections, h heads: 3h × 2nd_model(d_model/h) = 6nd_model². QK^T per head: 2n²(d_model/h). Across h heads: 2n²d_model. AV per head: 2n²(d_model/h). Across h heads: 2n²d_model. W_O: 2nd_model² (unchanged). Total = 8nd_model² + 4n²d_model. Algebraically identical in every term. The algebraic reason: the operation is bilinear in the dimensions. Splitting a d_model operation into h operations of size d_model/h each costs h × (d_model/h)² = d_model²/h per head, times h heads = d_model². The h cancels in any term where the dimensions scale quadratically. The practical implication beyond 'same FLOPs': multi-head has better GPU utilization. Each head's computation is smaller and more likely to fit in SRAM (useful for Flash Attention), and h smaller matrix multiplies can be batched and run in parallel across GPU SMs more efficiently than one large matrix multiply that might serialize across the memory hierarchy. So multi-head doesn't just cost the same — in practice on modern hardware, it tends to be slightly faster at equivalent d_model."
*Interviewer:* Exactly right. Full derivation, clean algebraic argument, and the insight that goes beyond the proof: GPU utilization benefits. The observation that multi-head FLOPs equivalence doesn't tell the full story because of hardware effects — and that multi-head tends to be *slightly faster* due to parallelism — is the kind of nuance you'd only know if you've profiled this in practice.
*Criteria — Met:* Complete derivation both cases, all terms exact, algebraic cancellation explained, bilinear structure insight, GPU parallelism benefit beyond FLOP equivalence

---

## Key Takeaways

🎯 1. **W_O is mandatory** — without it, heads are partitioned and can never mix information across head boundaries
2. **Total parameters = 4 × d_model²** regardless of head count — the h cancels algebraically in every term
🎯 3. **Multi-head FLOPs = single-head FLOPs** — same math, but multi-head has better GPU utilization
4. **FLOPs = 8n·d_model² + 4n²·d_model** — projection terms dominate for short sequences (n < 2·d_model), attention terms dominate for long sequences
5. **Head collapse has two forms**: attention pattern collapse and W_O contribution collapse — diagnose with JSD/cosine similarity and W_O block norms respectively
⚠️ 6. **GQA is the production standard** for long-context models — n_kv_heads=8 gives 4–8× KV cache reduction with < 1% quality loss
7. **Batched masking bugs are silent** — always verify that attention weights are zero for padded positions and future positions
8. **Head pruning reveals the redundancy**: 40–60% of heads can be removed in large models, confirming endemic collapse

---

**Further Reading**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — Section 3.2
- [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) (Michel et al., 2019) — head pruning
- [Analyzing Multi-Head Self-Attention](https://arxiv.org/abs/1905.09418) (Voita et al., 2019) — what heads actually do
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (Shazeer, 2019) — MQA

---

[← Back to Multi-Head Attention (Layer 1)](./multi-head-attention.md) | [Next: Positional Encoding](./positional-encoding.md)
