> **What this file covers**
> - 🎯 Full derivation of perplexity from cross-entropy: PPL = exp(H)
> - 🧮 Chain rule decomposition of joint probability and its connection to teacher forcing
> - ⚠️ 4 failure modes: vocabulary incomparability, BPE tokenization bias, domain mismatch, overfitting detection
> - 📊 O(N·V) computation per token, memory requirements for large vocabularies
> - 💡 Perplexity vs bits-per-character vs bits-per-byte — when each is appropriate
> - 🏭 Production monitoring: perplexity as a drift detector, train/val gap analysis
> - Staff/Principal Q&A with all four hiring levels shown

---

# Perplexity — Interview Deep-Dive

This file assumes you have read [perplexity.md](./perplexity.md) and have the intuition that perplexity measures "how many words the model is choosing between at each step." Everything here is for Staff/Principal depth.

---

## 🗺️ Concept Flow

```
              Training corpus
                    │
                    ▼
            Model learns P(word | context)
                    │
                    ▼
            Held-out test corpus: w₁, w₂, ..., wₙ
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    P(w₁|ctx)   P(w₂|ctx)   P(wₙ|ctx)
        │           │           │
        ▼           ▼           ▼
    log P(w₁)   log P(w₂)   log P(wₙ)
        │           │           │
        └───────────┼───────────┘
                    ▼
            Average log probability
            = negative cross-entropy H
                    │
                    ▼
            Perplexity = exp(H)
            = "effective number of choices"
```

The key insight: perplexity is the exponential of cross-entropy. Cross-entropy measures average surprise in log-space. Exponentiation converts it back to an interpretable count.

---

## 🧮 The Full Derivation

### Step 1: Joint Probability of the Text

A language model defines a probability distribution over sequences. For a text with N tokens w₁, w₂, ..., wₙ:

```
🧮 Joint probability via chain rule:

    P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁,...,wₙ₋₁)

                        = ∏ᵢ₌₁ᴺ P(wᵢ | w₁, ..., wᵢ₋₁)
```

Each factor P(wᵢ | w₁, ..., wᵢ₋₁) is the probability the model assigns to the actual token wᵢ given all preceding tokens.

### Step 2: Per-Token Log Probability (Cross-Entropy)

The joint probability is a product of many small numbers — it quickly becomes tiny. Taking the log converts products to sums.

```
🧮 Average negative log probability (cross-entropy):

    H = -(1/N) × Σᵢ₌₁ᴺ log P(wᵢ | w₁, ..., wᵢ₋₁)

    Where:
      N    = total number of tokens in the test corpus
      wᵢ   = the i-th token
      log  = natural logarithm (ln)
      H    = average surprise per token, in nats
```

This is the cross-entropy between the true data distribution and the model's distribution. Lower H means the model assigns higher probability to the actual tokens.

**Why "cross-entropy"?** In information theory, entropy H(p) measures the average surprise of a distribution p. Cross-entropy H(p, q) measures the average surprise when using distribution q (the model) to encode data from distribution p (reality). The cross-entropy is always ≥ entropy: H(p, q) ≥ H(p). Equality holds only when q = p.

### Step 3: Exponentiate to Get Perplexity

```
🧮 Perplexity:

    PPL = exp(H) = exp( -(1/N) × Σᵢ₌₁ᴺ log P(wᵢ | w₁, ..., wᵢ₋₁) )

    Equivalently:

    PPL = ( ∏ᵢ₌₁ᴺ P(wᵢ | w₁, ..., wᵢ₋₁) )^(-1/N)

    = geometric mean of (1 / P(wᵢ | context))
```

**Interpretation:** PPL is the geometric mean of the inverse probabilities. If the model assigns probability p to each token, PPL = 1/p, which is "how many equally-likely choices that probability corresponds to."

### Worked Example

```
Sentence: "The cat sat" (3 tokens)

Model's conditional probabilities:
  P("The" | <start>)      = 0.10
  P("cat" | "The")        = 0.05
  P("sat" | "The cat")    = 0.20

Step 1: Log probabilities
  ln(0.10) = -2.303
  ln(0.05) = -2.996
  ln(0.20) = -1.609

Step 2: Average negative log probability
  H = -(1/3) × (-2.303 + -2.996 + -1.609)
    = -(1/3) × (-6.908)
    = 2.303

Step 3: Exponentiate
  PPL = exp(2.303) = 10.0

Interpretation: on average, the model was choosing between
~10 equally likely tokens at each step.
```

---

## 🧮 Connection to Bits-Per-Character and Bits-Per-Byte

Different communities report perplexity in different units. The core quantity is the same; only the base of the logarithm and the tokenization unit differ.

```
🧮 Unit conversions:

    Cross-entropy in nats:   H_nat = -(1/N) × Σ ln P(wᵢ | ctx)
    Cross-entropy in bits:   H_bit = H_nat / ln(2)  = -(1/N) × Σ log₂ P(wᵢ | ctx)
    Perplexity:              PPL = exp(H_nat) = 2^(H_bit)

    Bits-per-character (BPC): H_bit computed with characters as tokens
    Bits-per-byte (BPB):      H_bit computed with raw bytes as tokens
```

| Unit | Token unit | Base | Used by |
|------|-----------|------|---------|
| Perplexity (PPL) | Subword tokens (BPE/WordPiece) | e | Most LLM papers |
| Bits-per-character (BPC) | Characters | 2 | Character-level models |
| Bits-per-byte (BPB) | Raw bytes | 2 | Cross-tokenizer comparison |

**🎯 Key insight:** BPB is the only unit that allows fair comparison across models with different tokenizers. A model with BPE vocabulary 32K and one with 64K produce different token counts for the same text, making token-level perplexity incomparable. BPB normalizes by the raw byte count, which is tokenizer-independent.

---

## ⚠️ Failure Modes

### 1. Vocabulary Incomparability

**What happens:** two models with different tokenizers produce different token counts for the same text. Their per-token perplexities are not comparable.

**Example:** "unhappiness" might be 1 token in model A (large vocabulary) and 3 tokens ["un", "happi", "ness"] in model B (BPE). Model B has 3 chances to be surprised by the same word. Even if both models understand the word equally well, model B's perplexity will differ.

**How to detect:** check if the models use the same tokenizer and vocabulary size. If not, token-level PPL is invalid for comparison.

**How to fix:** compare using bits-per-byte (BPB) or bits-per-character (BPC), which normalize by a tokenizer-independent unit.

### 2. BPE Fertility Bias

**What happens:** BPE tokenizers split rare words into more tokens than common words. This means rare words contribute more terms to the perplexity sum, weighting them disproportionately.

**Example:** the word "serendipitous" might split into 4 BPE tokens. The model's perplexity is now partly measuring how well it predicts BPE subword boundaries, not just language understanding.

**Impact:** domains with many rare or technical words (medical, legal, scientific text) will have inflated perplexity compared to casual text, even if the model handles both equally well.

**How to detect:** compare word-normalized perplexity (divide total log probability by number of words, not tokens) to token-normalized perplexity. If they diverge significantly, BPE fertility is distorting the metric.

### 3. Domain Mismatch

**What happens:** a model trained on news articles has low perplexity on news but high perplexity on Reddit comments. This does not mean the model is bad — it means the evaluation domain does not match the training domain.

**How to detect:** always report the evaluation dataset alongside the perplexity number. A perplexity of 25 on WikiText-103 and 150 on code are not contradictory — they test different things.

**How to fix:** evaluate on the domain you care about. If the model must work across domains, report perplexity on each domain separately.

### 4. Overfitting Detection via Train/Val Gap

**What happens:** training perplexity keeps dropping, but validation perplexity stops improving or increases. The model is memorizing training data rather than learning generalizable patterns.

**How to detect:** plot training PPL and validation PPL over time. The gap between them is the overfitting signal.

**How to fix:** early stopping (stop training when validation PPL stops improving), regularization (dropout, weight decay), or more training data.

---

## 📊 Computational Complexity

### Per-Token Cost

Computing the model's probability distribution at each position requires a forward pass through the entire model.

```
🧮 Cost per token:

    Forward pass: O(L × d² + L × n × d)
      L = number of layers
      d = model dimension
      n = sequence length (due to attention)

    Softmax over vocabulary: O(V)
      V = vocabulary size (typically 32K–128K)

    Log probability lookup: O(1)
```

For a full corpus of N tokens, total cost = N forward passes (or fewer with batching and KV caching).

### Memory

The dominant memory cost is the model itself. Perplexity evaluation does not require gradients, so memory is roughly half of training (no optimizer states, no activation storage for backprop).

For a 7B parameter model in float16: ~14 GB model weights. Evaluation can run on a single GPU.

### Perplexity Computation Time

| Model Size | Tokens/sec (A100) | Time for 1M tokens |
|-----------|-------------------|-------------------|
| 125M | ~50,000 | ~20 seconds |
| 1.3B | ~10,000 | ~100 seconds |
| 7B | ~2,000 | ~8 minutes |
| 70B | ~200 | ~80 minutes |

These are approximate throughput numbers for evaluation (no gradient computation).

---

## 💡 Design Trade-offs

### Perplexity vs Task-Specific Evaluation

| | Perplexity | Task-specific metrics |
|---|---|---|
| Measures | Next-token prediction quality | Actual downstream performance |
| Requires | Raw text corpus | Labeled task data |
| Generality | One number for all text domains | Specific to one task |
| Weakness | Low PPL does not guarantee good task performance | Task metrics miss general language quality |
| When to use | Model pre-training, architecture comparison | Fine-tuning, production deployment |

**🎯 Key insight:** perplexity is a necessary but not sufficient condition for a good language model. A model can have low perplexity and still fail at reasoning, factuality, or instruction following. This is why modern LLM evaluation combines perplexity with task benchmarks (MMLU, HumanEval) and human evaluation.

### Perplexity vs Loss

In practice, what you actually compute during training is the cross-entropy loss, not perplexity directly.

```
🧮 Relationship:

    Cross-entropy loss = H = -(1/N) × Σ log P(wᵢ | ctx)
    Perplexity = exp(H)

    Loss = 3.0  →  PPL = exp(3.0) = 20.1
    Loss = 2.0  →  PPL = exp(2.0) = 7.4
    Loss = 1.0  →  PPL = exp(1.0) = 2.7
```

Perplexity is reported for interpretability (it maps to "number of choices"). But the loss is what the optimizer actually minimizes, and it has nicer numerical properties for gradient computation.

---

## 🏭 Production Considerations

### Perplexity as a Drift Detector

In production, you can monitor perplexity on incoming data as a proxy for distribution shift. If the perplexity of your production traffic suddenly increases, the model is encountering text that is unlike its training data.

Pipeline: compute rolling-window perplexity on production inputs. Set alerts when it exceeds 2-3x the evaluation-set baseline.

### Evaluation Protocol Best Practices

1. **Stride matters:** for models with limited context windows, the stride (how much you advance between evaluation windows) affects the result. Non-overlapping windows miss cross-window dependencies. Stride-1 is most accurate but most expensive.

2. **BOS/EOS handling:** whether you include beginning-of-sequence and end-of-sequence tokens in the count changes the result. Always document which convention you use.

3. **Conditional vs unconditional:** some papers condition on a prefix (e.g., the first 512 tokens) and only measure PPL on the remaining tokens. This gives the model context, producing lower PPL than unconditional evaluation.

---

## Staff/Principal Interview Depth

### Q1: Explain the relationship between perplexity and cross-entropy. Why do we exponentiate?

---
**No Hire**
*Interviewee:* "Perplexity measures how well the model predicts text. Lower is better. I think it's related to the loss function somehow."
*Interviewer:* Vague connection, no mathematical relationship, no understanding of the exponentiation.
*Criteria — Met:* basic purpose / *Missing:* PPL = exp(H), why exponentiate, information-theoretic connection

**Weak Hire**
*Interviewee:* "Perplexity is the exponential of the cross-entropy loss. Cross-entropy is the average negative log probability of the tokens. We exponentiate to make it more interpretable."
*Interviewer:* Correct relationship stated. But "more interpretable" is vague — does not explain what the exponentiation achieves.
*Criteria — Met:* PPL = exp(H), correct formula / *Missing:* information-theoretic interpretation, why the result is "number of choices"

**Hire**
*Interviewee:* "Cross-entropy H is the average negative log probability per token, measured in nats. It represents the average surprise. Perplexity = exp(H) converts this back from log-space to a linear count: it's the effective number of equally-likely tokens the model is choosing between. If PPL = 20, the model's uncertainty is equivalent to a uniform distribution over 20 tokens. We exponentiate because the log-space number is not intuitive — 'cross-entropy of 3.0 nats' means nothing to most people, but 'the model is choosing between 20 options' is immediately clear."
*Interviewer:* Clear explanation of what exponentiation achieves and why it gives "number of choices." Would be stronger with the connection to information-theoretic entropy.
*Criteria — Met:* PPL = exp(H), interpretation as choice count, why exponentiate / *Missing:* H(p,q) ≥ H(p) relationship, connection to KL divergence

**Strong Hire**
*Interviewee:* "Cross-entropy H(p, q) between the true distribution p and the model q measures how many nats are needed to encode data from p using the code optimized for q. The gap H(p, q) - H(p) = KL(p || q) is the inefficiency of the model. Perplexity = exp(H) converts this to the effective vocabulary size: if PPL = 20, the model's average uncertainty is equivalent to uniform over 20 tokens. The minimum achievable perplexity is exp(H(p)) — the entropy of the true language distribution, estimated at about 1.1 bits per character for English (Shannon 1951), which corresponds to PPL ≈ 2 per character. For subword tokens, the minimum depends on the tokenizer. One subtlety: exponentiation is monotonic, so PPL and H always rank models the same way. The only reason to use PPL over H is human interpretability."
*Interviewer:* Full information-theoretic framing. Connects to KL divergence, cites the theoretical lower bound, notes that rankings are preserved. Staff-level answer.
*Criteria — Met:* everything / *Missing:* nothing

---

### Q2: Can you compare the perplexity of two models that use different tokenizers? If not, how do you fix it?

---
**No Hire**
*Interviewee:* "Sure, just compare their perplexity scores. Lower is better."
*Interviewer:* Does not recognize the tokenizer incomparability problem.
*Criteria — Met:* none / *Missing:* tokenizer dependence, vocabulary size effect, BPB

**Weak Hire**
*Interviewee:* "Different tokenizers produce different numbers of tokens for the same text, so perplexity isn't directly comparable. You'd need to normalize somehow."
*Interviewer:* Identifies the problem but does not know the solution.
*Criteria — Met:* identifies incomparability / *Missing:* specific normalization (BPB/BPC), why it works

**Hire**
*Interviewee:* "No, token-level perplexity is not comparable across tokenizers. A BPE tokenizer with 32K vocabulary splits 'unhappiness' into 3 tokens, while a 100K vocabulary might keep it as 1 token. The model with more tokens gets more chances to be surprised, inflating its perplexity. The fix is to normalize by bytes: compute total log probability of the corpus, then divide by the byte count (not the token count). This gives bits-per-byte (BPB), which is tokenizer-independent."
*Interviewer:* Correctly identifies the problem, explains why it occurs, and provides the standard fix. Solid answer.
*Criteria — Met:* identifies problem, explains mechanism, knows BPB / *Missing:* BPE fertility effect, word-normalized PPL alternative

**Strong Hire**
*Interviewee:* "Token-level PPL is fundamentally tokenizer-dependent. Two issues: (1) different total token counts for the same text, and (2) BPE fertility bias — rare words are split into more subwords, giving them disproportionate weight in the loss. The standard fix is bits-per-byte (BPB): total_log_prob / (num_bytes × ln(2)). This normalizes by a tokenizer-independent quantity. BPC (bits-per-character) also works but is encoding-dependent (UTF-8 vs UTF-16). An alternative is word-normalized PPL: divide total log prob by word count rather than token count. This partially controls for tokenizer differences but introduces the question of what counts as a 'word' (language-dependent). For truly rigorous comparison, the GPT-4 technical report used BPB specifically because they changed tokenizers between GPT-3 and GPT-4. If someone compares per-token PPL across different tokenizers, their comparison is invalid."
*Interviewer:* Complete analysis. Identifies both the count mismatch and the fertility bias. Provides multiple solutions with trade-offs. Cites a real-world example. Staff-level depth.
*Criteria — Met:* everything / *Missing:* nothing

---

### Q3: Your language model has training perplexity of 12 but validation perplexity of 45. What is happening and what do you do?

---
**No Hire**
*Interviewee:* "The model is doing well on training. Validation is higher but that's normal — validation is always harder."
*Interviewer:* Does not recognize overfitting. Dismisses the gap as normal.
*Criteria — Met:* none / *Missing:* overfitting diagnosis, remedies, expected gap size

**Weak Hire**
*Interviewee:* "That's a big gap — the model is overfitting. I'd add more regularization like dropout."
*Interviewer:* Correct diagnosis but only one remedy. No quantification of what gap size is concerning.
*Criteria — Met:* overfitting diagnosis / *Missing:* expected gap size, multiple remedies, systematic debugging

**Hire**
*Interviewee:* "A 3.75x ratio (45/12) is a strong overfitting signal. The model has memorized the training set rather than learning generalizable patterns. My debugging steps: (1) check training set size — is it too small relative to model parameters? Rule of thumb: you want at least 10-20 tokens per parameter. (2) Check for data contamination — is any validation data accidentally in the training set? (3) Apply regularization: increase dropout, add weight decay, reduce model size. (4) Early stopping — the validation PPL probably started increasing at some point; stop training there. (5) Add more diverse training data if possible."
*Interviewer:* Good systematic approach. The tokens-per-parameter rule of thumb and contamination check show practical experience.
*Criteria — Met:* diagnosis, multiple remedies, systematic approach / *Missing:* learning rate analysis, specific remediation priority order

**Strong Hire**
*Interviewee:* "This is overfitting. First, I'd verify it's not a data issue: check for train/val leakage by searching for exact n-gram overlaps. If the split is clean, the 45/12 = 3.75x ratio suggests severe overfitting. I'd plot both curves over training steps — the validation PPL probably started increasing at some epoch while training PPL kept falling. The divergence point is where I'd apply early stopping. For remediation, in priority order: (1) more training data (most impactful but may not be feasible), (2) increase dropout (standard is 0.1 for transformers; try 0.2-0.3), (3) weight decay (1e-2 to 1e-1), (4) reduce model capacity (fewer layers or smaller hidden dim), (5) data augmentation (back-translation, paraphrasing). I'd also check the learning rate schedule — if using a constant LR without warmup and decay, the model may be continuing to fit training data long after validation has plateaued. One subtlety: for large pre-trained models, some train/val gap is normal because the model has capacity to memorize portions of training data. A ratio of 1.1-1.3x is typical. 3.75x is far outside that range."
*Interviewer:* Prioritized remediation, quantified what a normal gap looks like, checked for data leakage, identified learning rate as a factor. Complete analysis.
*Criteria — Met:* everything / *Missing:* nothing

---

## Key Takeaways

🎯 1. Perplexity = exp(cross-entropy) — it converts average log-space surprise into "number of choices"
🎯 2. Token-level perplexity is tokenizer-dependent — never compare across different tokenizers without normalizing to BPB
   3. The minimum achievable perplexity is bounded by the entropy of the language itself
⚠️ 4. BPE fertility biases perplexity toward rare/technical words — domain-specific text will always have higher PPL
   5. Training/validation PPL gap is the primary overfitting diagnostic — ratios above 1.3x warrant investigation
🎯 6. Perplexity is necessary but not sufficient — low PPL does not guarantee good task performance, factuality, or safety
   7. In production, rolling-window perplexity on incoming data serves as a distribution shift detector
   8. Always report the evaluation dataset and tokenization details alongside the PPL number
