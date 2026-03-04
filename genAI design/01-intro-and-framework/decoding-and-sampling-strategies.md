# Decoding and Sampling Strategies

## Introduction

Every autoregressive genAI system faces the same question at inference time: "The model has computed probabilities for the next token — now what?" The answer to this question affects output quality, diversity, and controllability more than most candidates realize.

The decoding strategy is the bridge between a trained model and its actual outputs. A great model with a bad decoding strategy produces repetitive, degenerate text. A decent model with a well-tuned decoding strategy produces natural, useful outputs. In interviews, candidates who can explain when to use greedy vs beam search vs nucleus sampling — and why — demonstrate they've actually built systems that generate text, not just trained models.

---

## Greedy Decoding

At each step, pick the token with the highest probability. Deterministic — same input always produces same output.

**How it works:**
```
At step t: token_t = argmax(P(token | context))
```

**When it works:**
- Short, factual outputs (classification labels, entity extraction, yes/no answers)
- Tasks with one correct answer
- When determinism is required (reproducible outputs)

**When it fails:**
- Long text generation — leads to repetitive, degenerate output. The model gets stuck in loops ("The cat sat on the mat. The cat sat on the mat. The cat...")
- Creative tasks — no diversity, always picks the "safest" continuation
- Misses globally optimal sequences — the locally best token at step 5 might lead to a bad sequence overall

**Why greedy fails on long text:** Each greedy choice narrows the search space. After a few steps, the model is locked into a path that may not be globally optimal. There's no mechanism to backtrack or explore alternatives.

---

## Beam Search

Maintain the top-B partial sequences (beams) at each step. At the end, pick the complete sequence with the highest total probability.

**How it works:**
1. Start with B copies of the prompt
2. At each step, expand each beam by all possible next tokens
3. Keep only the top-B sequences by total log-probability
4. Continue until all beams produce an end-of-sequence token
5. Return the beam with the highest total probability (or a variant — see below)

**Beam width tradeoff:**

| Beam width | Quality | Latency | Memory | Notes |
|-----------|---------|---------|--------|-------|
| 1 | = greedy | Baseline | Baseline | Equivalent to greedy decoding |
| 4-8 | Good | 4-8x slower | 4-8x more | Standard for translation/summarization |
| 16-32 | Slightly better | 16-32x slower | 16-32x more | Diminishing returns beyond 8 |

**Length normalization:** Without it, beam search favors shorter sequences (shorter = fewer multiplied probabilities = higher total probability). The fix:

`score = (1/length^α) × log P(sequence)`

α controls the length penalty. α = 0.6-0.8 is typical. Without this, beam search tends to generate truncated outputs.

**When to use:** Machine translation, summarization, speech recognition — tasks where there's a specific "correct" output.

**When NOT to use:** Chatbots, creative writing, brainstorming. Beam search outputs feel generic and bland because it always selects the highest-probability sequences, which tend to be safe, common phrases.

---

## Temperature Scaling

Divide logits by a temperature parameter T before applying softmax:

`P(token_i) = exp(z_i / T) / Σ exp(z_j / T)`

Temperature controls the sharpness of the probability distribution:

| Temperature | Effect | Distribution | Best For |
|------------|--------|-------------|----------|
| T → 0 | Equivalent to greedy | All mass on top token | When you want determinism |
| T = 0.1-0.3 | Very focused | Top tokens dominate | Factual QA, code completion |
| T = 0.7-0.9 | Balanced | Moderate diversity | Chatbot conversation, general text |
| T = 1.0 | Original distribution | As trained | Baseline reference |
| T = 1.0-1.5 | Flattened | More uniform | Creative writing, brainstorming |
| T > 2.0 | Nearly uniform | Random | Rarely useful (too chaotic) |

**Intuition:** Low temperature makes the model more "confident" — it strongly prefers high-probability tokens. High temperature makes the model more "creative" — it's willing to pick lower-probability tokens.

**How to choose T:** There's no universal answer. It depends on the task and user expectations. Start at 0.7 for general text, adjust based on output quality. For production systems, A/B test different temperatures.

---

## Top-k Sampling

Only sample from the top-k most probable tokens. All other tokens are set to probability 0.

**How it works:**
1. Compute the probability distribution over the vocabulary
2. Keep only the k tokens with the highest probability
3. Renormalize the remaining probabilities
4. Sample from this restricted distribution

**The problem with fixed k:**

When the model is confident (probability concentrated on a few tokens), k=50 includes many low-quality options that shouldn't be considered. When the model is uncertain (probability spread across many tokens), k=50 might exclude valid options.

**Example:**
- Model is confident: P = [0.9, 0.05, 0.02, 0.01, ...]. Top-k=50 includes 47 tokens that are almost certainly wrong.
- Model is uncertain: P = [0.05, 0.04, 0.04, 0.03, ...]. Top-k=50 might exclude token #51 which had probability 0.02 — nearly as good as token #1.

This inflexibility is why top-p (nucleus) sampling was developed.

---

## Top-p (Nucleus) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds p.

**How it works:**
1. Sort tokens by probability (descending)
2. Add tokens until cumulative probability ≥ p
3. Renormalize probabilities within this set
4. Sample from this set

**Why it's better than top-k:** The set size adapts to the distribution shape.
- When confident: cumulative probability reaches p after just a few tokens → small, focused set
- When uncertain: cumulative probability spreads across many tokens → large, diverse set

**Typical values:** p = 0.9-0.95 for general text generation. p = 0.99 for maximum diversity. p = 0.5-0.7 for more focused generation.

**Combining temperature and top-p:** You can apply both. First scale logits by temperature, then apply top-p filtering. Temperature adjusts how spread the distribution is; top-p cuts off the tail.

---

## Advanced Techniques

### Repetition Penalty

Penalize tokens that have appeared recently to prevent degenerate loops.

**Implementation:** Divide the logit of any token that appeared in the last N tokens by a penalty factor R:

`logit_adjusted = logit / R  (if token appeared in last N tokens)`

Typical values: R = 1.1-1.3, N = 64-256. Too high → unnatural avoidance of common words.

### Typical Sampling

Instead of sampling the most probable tokens (top-k/top-p), sample tokens whose information content is close to the expected information content (entropy).

**Intuition:** Human text tends to be "typically surprising" — not too predictable, not too random. Very high-probability tokens make text boring. Very low-probability tokens make it incoherent. Typical sampling targets the sweet spot.

### Speculative Decoding

Use a small, fast draft model to generate N candidate tokens, then verify all N tokens in a single forward pass of the large model.

**How it works:**
1. Small model generates N tokens autoregressively (fast — small model)
2. Large model processes all N tokens in one forward pass (parallel — one pass for N tokens)
3. Compare: where the large model agrees with the small model, accept. At the first disagreement, sample from the large model's distribution and discard the rest.

**Key property:** The output distribution is exactly the same as the large model alone. This is not an approximation — it's mathematically lossless. You get the quality of the large model at closer to the speed of the small model.

**Speedup:** 2-3x in practice. Depends on how often the draft model matches the large model (acceptance rate). Higher acceptance rate → more speedup.

### Classifier-Free Guidance (CFG)

Primarily used for diffusion models, but the concept applies broadly.

`output = unconditional + guidance_scale × (conditional - unconditional)`

- `guidance_scale = 1`: standard conditional generation
- `guidance_scale = 7-15`: stronger prompt adherence (typical for image generation)
- `guidance_scale > 20`: over-saturated, artifact-prone

**Tradeoff:** Higher guidance → images that more closely match the prompt, but less diverse and more likely to have artifacts. Lower guidance → more diverse, more natural, but may not match the prompt well.

---

## Choosing the Right Strategy

| Task | Best Strategy | Temperature | Why |
|------|---------------|-------------|-----|
| Machine translation | Beam search (width 4-8) | — | There's a correct answer; beam search finds it |
| Factual QA | Greedy or low temperature | 0.0-0.3 | One correct answer; minimize randomness |
| Code generation | Nucleus sampling (p=0.95) | 0.2-0.4 | Correctness matters more than creativity |
| Chatbot conversation | Nucleus sampling (p=0.9) | 0.7-0.9 | Natural, non-repetitive, slightly creative |
| Creative writing | Nucleus sampling (p=0.95) | 0.8-1.2 | Diversity and surprise are desirable |
| Brainstorming | High temperature + top-p | 1.0-1.5 | Want unusual, unexpected ideas |
| Image generation (diffusion) | CFG scale 7-15 | — | Balance prompt adherence with quality |
| Summarization | Beam search (width 4) or low temperature | 0.3-0.5 | Faithful to source, not creative |

### The Production Decision

In production systems, you often can't use a single strategy:
- Different tasks within the same application may need different settings
- Users may have preferences ("more creative" vs "more precise")
- The same model may serve multiple products

**Best practice:** Make decoding parameters configurable per request. Default to reasonable values (temperature=0.7, top_p=0.9), but allow overrides for specific use cases.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should understand the difference between greedy decoding and sampling, and know that temperature controls randomness. For a chatbot system, they should propose sampling (not greedy or beam search) and explain why — greedy produces repetitive text, sampling produces natural conversation. They differentiate by mentioning top-p or top-k as a way to prevent low-probability garbage tokens from being sampled.

### Senior Engineer

Senior candidates can explain the full toolkit: greedy, beam search, temperature, top-k, top-p — and when each is appropriate. They know that beam search is right for translation but wrong for chatbots, and can explain why. For a code generation system, a senior candidate would propose nucleus sampling with low temperature (correctness over creativity) and discuss repetition penalties for avoiding degenerate loops. They bring up the tradeoff between temperature and top-p and how they interact.

### Staff Engineer

Staff candidates think about decoding strategy as a system design choice, not just a model parameter. They recognize that the decoding strategy should be configurable per use case and that A/B testing different decoding configurations is often more impactful than A/B testing model changes. A Staff candidate might propose speculative decoding for latency optimization and explain the lossless speedup property. They also understand the cost implications — at scale, beam search with width 8 means 8x the compute per request — and discuss how to balance quality, latency, and cost across different product surfaces.
