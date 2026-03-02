# ChatGPT / Conversational AI System — Staff/Principal Interview Guide

---

## How to Use This Guide

This guide is structured for interviewers and candidates preparing for staff- or principal-level ML design interviews. The interview is **45 minutes** total. Each section includes an **interviewer prompt**, the **signal being tested**, and **four-level model answers** representing the candidate response quality spectrum.

**Rating Levels:**
- **No Hire** — Fundamental misunderstanding or silence
- **Lean No Hire** — Partial understanding, significant gaps, needs heavy prompting
- **Lean Hire** — Correct understanding, hits main points, minor gaps
- **Strong Hire** — Deep, nuanced, first-principles reasoning, proactively addresses trade-offs, demonstrates platform-level thinking

**Interviewer Notes:**
- Spend the first minute reading the prompt aloud and giving the candidate time to think silently.
- Do not volunteer information unless the candidate is stuck for more than 90 seconds.
- Use the follow-up probes listed under each section to differentiate Hire from Strong Hire.
- The principal-level bar requires connecting individual design decisions to broader organizational or platform impact.

**Time Budget:**

| Section | Time |
|---|---|
| Problem Statement & Clarification | 5 min |
| ML Problem Framing | 5 min |
| Data & Preprocessing | 8 min |
| Model Architecture Deep Dive | 12 min |
| Evaluation | 5 min |
| Serving Architecture | 7 min |
| Edge Cases & Failure Modes | 5 min |
| Principal-Level Platform Thinking | 3 min |

---

## Section 1: Problem Statement & Clarification (5 min)

### Interviewer Prompt

> "Design a production conversational AI assistant similar to ChatGPT. The system must serve millions of concurrent users, maintain multi-turn context, and be aligned with human preferences to be safe, helpful, and honest. Walk me through how you'd approach this, starting with what you'd need to clarify."

### Signal Being Tested

Does the candidate recognize that building a ChatGPT-class system involves three distinct phases (pretraining, supervised fine-tuning, RLHF) and that requirements around safety, latency, and scale fundamentally shape all design decisions?

### Six Clarification Dimensions

| Dimension | Why It Matters |
|---|---|
| **Model scale** | Determines compute budget, serving infrastructure, and whether distillation is needed |
| **Safety requirements** | Shapes RLHF reward signal design and classifier deployment |
| **Latency SLA** | Drives speculative decoding, quantization, and continuous batching decisions |
| **Context length** | Determines KV-cache memory budget and attention architecture choices |
| **Training data access** | Determines if pretraining from scratch vs. fine-tuning a base model |
| **User correction loop** | Shapes ongoing RLHF data collection and reward model refresh cadence |

### Follow-up Probes

- "What changes about your design if this system must handle 100K concurrent users versus 1M?"
- "If you had to pick one metric to track that predicts user satisfaction, what would it be and why?"
- "How does your training pipeline change if you start from an existing LLM vs. pretraining from scratch?"

---

### Model Answers — Section 1

**No Hire:**
The candidate immediately proposes "fine-tuning GPT-4 with RLHF" without asking any clarifying questions. There is no acknowledgment that requirements around safety level, latency, scale, or context length fundamentally shape the entire system design.

**Lean No Hire:**
The candidate asks one or two surface questions ("How many users?", "What kind of questions will people ask?") but misses the critical dimensions: they don't probe for safety requirements, don't ask about latency SLAs, and don't recognize that the question of starting from scratch vs. fine-tuning is a massive decision that changes everything.

**Lean Hire:**
The candidate identifies the three-stage training pipeline (pretraining → SFT → RLHF) and asks about at least four of the six dimensions. They note that safety requirements and context length are particularly important for the design. They understand that serving millions of concurrent users is a qualitatively different challenge from small-scale deployment.

**Strong Hire Answer (first-person):**

Before I commit to any architecture, I need to understand several dimensions of this problem because they create fundamentally different systems.

First, are we training from scratch or starting from a pretrained base model? If we're building a ChatGPT-equivalent from scratch, we're talking about pretraining a 70B–175B+ parameter transformer on trillions of tokens — roughly $10–50M in compute cost and several weeks of wall-clock time on thousands of H100s. If we're fine-tuning an existing open-weight model like Llama 3.1 or Mistral, the scope changes dramatically. I'll assume we're fine-tuning an existing pretrained LLM since that reflects most real-world deployment scenarios.

Second, what are the safety requirements? There is a spectrum from "avoid clearly harmful content" to "pass formal AI safety red-teaming at government standards." The safety bar determines how many stages of RLHF we need, whether we need constitutional AI or process reward models, and how much annotation budget to allocate. I'll assume we need production-grade safety aligned with major platforms — so reward model training, PPO fine-tuning, and an inference-time safety classifier.

Third, latency SLA. A 100ms p50 time-to-first-token (TTFT) feels like instant to users; 500ms feels sluggish; 2 seconds feels broken. The SLA determines whether we need speculative decoding, aggressive quantization (INT4/INT8), or tensor parallelism across GPUs. I'll assume a 200ms TTFT target at p90.

Fourth, context length. GPT-4 supports 128K tokens; this requires very different KV-cache management than a 4K-context model. Long contexts are expensive both to prefill (O(n²) attention in vanilla transformers) and to store in KV-cache during inference. I'll assume 32K context as a starting point.

Fifth, the user correction loop. Ongoing RLHF requires continuously collecting human preference annotations on model outputs. Is there an annotation team, or are we using implicit feedback (thumbs up/down)? The feedback loop design directly affects how frequently we can refresh the reward model and re-run PPO.

With these constraints established, let me now walk through the full system.

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

> "How would you frame the training pipeline for this system? What are the distinct ML problems you're solving?"

### Signal Being Tested

Does the candidate correctly decompose the problem into pretraining, SFT, and RLHF? Can they articulate what each stage learns and why each is necessary?

### Follow-up Probes

- "Why is supervised fine-tuning not sufficient on its own? Why do you need RLHF on top?"
- "What exactly is the reward model learning? What is its input and output?"
- "Why do you need a KL divergence penalty in PPO? What failure mode does it prevent?"

---

### Model Answers — Section 2

**No Hire:**
"I would fine-tune a pretrained model on chat data." Cannot explain the distinction between SFT and RLHF, or why RLHF is needed on top of SFT.

**Lean No Hire:**
Correctly mentions RLHF but describes it vaguely as "training the model to be helpful using human feedback." Cannot explain the reward model's role or the PPO optimization step mechanically.

**Lean Hire:**
Correctly describes the three-stage pipeline. Can explain that SFT teaches format but doesn't optimize for human preference; RLHF uses the reward model to directly optimize for what humans prefer. Notes the KL penalty's role in preventing reward hacking.

**Strong Hire Answer (first-person):**

The ChatGPT training pipeline decomposes into three ML problems, each solving something the previous stage cannot.

**Stage 1: Pretraining (Language Modeling)**
The base model learns world knowledge and language structure by minimizing next-token cross-entropy loss over a massive text corpus:

```
L_pretrain = -Σ_{t=1}^{T} log p_θ(w_t | w_1, ..., w_{t-1})
```

After pretraining, the model is a capable text predictor but not a useful assistant — it will continue whatever text you give it rather than answering questions helpfully. It has no concept of "I am a helpful assistant."

**Stage 2: Supervised Fine-Tuning (SFT)**
We fine-tune the pretrained model on high-quality (prompt, response) pairs created by human annotators who write ideal responses. The loss is identical — cross-entropy — but the data distribution shifts from general web text to curated instruction-following examples. This stage teaches the model the format of helpful interaction: how to interpret instructions, how to structure responses, how to say "I don't know" rather than hallucinating.

The limitation of SFT alone: the reward signal is binary presence/absence of each ground-truth token. It cannot capture the *quality* spectrum between a mediocre response and an excellent one. Two responses to the same prompt might get identical SFT loss even if humans strongly prefer one.

**Stage 3: RLHF — Reward Model + PPO**
This is the core of the alignment pipeline. We train a reward model R_φ(x, y) that takes a (prompt, response) pair and outputs a scalar quality score. The reward model is trained on human preference comparisons: given prompt x and responses y_A, y_B, human annotators pick which they prefer. We train R_φ to assign higher scores to preferred responses:

```
L_RM = -E_{(x, y_w, y_l)} [log σ(R_φ(x, y_w) - R_φ(x, y_l))]
```

where y_w is the "winner" and y_l is the "loser" in each comparison.

Then we use PPO to optimize the language model policy π_θ to maximize expected reward while staying close to the SFT model:

```
L_PPO = E_{x,y~π_θ} [R_φ(x, y) - β · KL(π_θ(y|x) || π_SFT(y|x))]
```

The KL term is critical: without it, the model quickly learns to produce outputs that fool the reward model without being genuinely helpful — reward hacking. The β coefficient controls this trade-off; too small and reward hacking dominates, too large and RLHF provides no signal beyond SFT.

---

## Section 3: Data & Preprocessing (8 min)

### Interviewer Prompt

> "Walk me through the data pipeline for each stage. What data do you need, how do you collect it, and how do you ensure quality?"

### Signal Being Tested

Does the candidate understand the annotation requirements for SFT demonstrations and RLHF preference comparisons? Can they identify data quality issues specific to each stage?

### Follow-up Probes

- "How do you handle annotator disagreement in preference comparisons?"
- "What is the minimum annotation budget to get meaningful RLHF signal?"
- "How would you detect if annotators are introducing systematic biases into the reward model?"

---

### Model Answers — Section 3

**No Hire:**
"I would scrape the web and use that as training data." No understanding of SFT demonstration data or preference comparison data collection.

**Lean No Hire:**
Knows that SFT needs (prompt, response) pairs and RLHF needs preference comparisons but cannot describe quality controls or annotator disagreement handling.

**Lean Hire:**
Correctly describes both data types. Notes that annotation quality is more important than quantity. Mentions inter-annotator agreement and the need for clear annotation guidelines.

**Strong Hire Answer (first-person):**

The data pipeline has three distinct phases with very different requirements.

**SFT Data — Demonstration Collection**
For SFT, I need high-quality (prompt, response) pairs where the responses are genuinely helpful, harmless, and honest. The InstructGPT paper used roughly 13,000 demonstration examples — a surprisingly small number, showing that data quality dominates quantity at this stage.

Key collection principles:
1. Diverse prompt distribution: customer service queries, code generation, factual Q&A, creative writing, multi-step reasoning. Without diversity, the model will overfit to common prompt types.
2. Annotator calibration: I would run a two-week calibration period where annotators write responses to the same prompts independently, then compare. We look for systematic differences (one annotator always writes longer responses, another always adds caveats).
3. Edge case coverage: deliberately include sensitive topics, ambiguous instructions, and requests the model should decline. If 0% of SFT training examples involve refusals, the model won't learn when to refuse.

**Reward Model Data — Preference Comparisons**
For the reward model, I need (prompt, response_A, response_B, preference) tuples. The preference is which response humans prefer. The InstructGPT paper used ~33,000 comparisons for the initial reward model.

Quality controls:
- Each comparison is shown to multiple annotators; the final label uses majority voting or a confidence-weighted aggregation.
- I track inter-annotator agreement (Fleiss' κ). If κ < 0.4, the annotation guidelines are ambiguous and need revision.
- I actively monitor for annotator-specific biases: if one annotator consistently prefers longer responses, their data will teach the reward model that length = quality, leading to verbose reward-hacking.

**RLHF Ongoing Data Collection**
After the initial training run, I set up a continuous feedback loop: sample outputs from the deployed model, send batches to annotators, refresh the reward model quarterly, and re-run PPO. This is critical because the model distribution shifts after each PPO run — a reward model trained on pre-PPO outputs may be miscalibrated on post-PPO outputs.

**Data Contamination**
A subtle issue: I must ensure that evaluation benchmarks (MMLU, HellaSwag, etc.) are not present in the pretraining or SFT data. Contamination inflates benchmark scores and makes evaluation unreliable.

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

> "Walk me through the model architecture for each stage: the base LLM, the reward model, and the PPO training setup. Be specific about the mechanisms."

### Signal Being Tested

Can the candidate explain the transformer architecture at a mechanistic level? Can they describe reward model architecture, PPO clipping, and the KL penalty implementation? Do they understand why specific choices (e.g., grouped-query attention, RoPE) were made?

### Follow-up Probes

- "Explain exactly what the attention mechanism computes. What are Q, K, V?"
- "Why does the reward model need a separate value head? What does it output?"
- "PPO clips the policy ratio within [1-ε, 1+ε]. Why does this matter for an LLM with a 100K-token vocabulary?"
- "What is the difference between DPO and PPO? When would you choose one over the other?"

---

### Model Answers — Section 4

**No Hire:**
"I would use a transformer model." Cannot explain attention mechanism, cannot describe reward model architecture, does not know what PPO clipping does.

**Lean No Hire:**
Describes transformers as "using attention to weigh importance of tokens" but cannot compute attention outputs. Cannot explain reward model as distinct from the language model. Knows PPO has a "clipping trick" but not why.

**Lean Hire:**
Correctly describes multi-head self-attention with Q, K, V matrices. Explains reward model as LLM with a classification head outputting a scalar. Describes PPO clipping at a conceptual level. Can compare SFT and RLHF mechanically.

**Strong Hire Answer (first-person):**

Let me walk through each component at a mechanistic level.

**Base LLM Architecture: Transformer Decoder**

Modern ChatGPT-class models are decoder-only transformers. For a 7B parameter model, the typical architecture is:
- 32 transformer layers
- Hidden dimension d_model = 4096
- 32 attention heads (head dimension 128)
- Feed-forward intermediate dimension = 4× d_model = 16384
- Vocabulary size ≈ 32K–128K tokens

Each attention layer computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

where Q = X·W_Q, K = X·W_K, V = X·W_V are linear projections of the input X ∈ R^{n × d_model}, and d_k = d_model / num_heads. This gives each token a weighted average of all previous token values, where weights depend on query-key similarity.

**Key architectural choices in modern LLMs:**

1. **RoPE (Rotary Position Embedding):** Instead of absolute position embeddings, we encode relative positions via rotation matrices applied to Q and K. This enables better length generalization and is the standard in Llama/Mistral.

2. **Grouped-Query Attention (GQA):** Full multi-head attention uses 32 Q heads and 32 K/V heads. GQA uses 32 Q heads but only 8 K/V heads — the K/V heads are shared across groups of Q heads. This reduces KV-cache size by 4×, which is critical for serving long contexts.

3. **SwiGLU activation:** `FFN(x) = (xW_1 ⊙ σ(xW_3))W_2` — this gated linear unit in the feed-forward layer outperforms ReLU in practice.

4. **RMSNorm pre-normalization:** Applied before each sub-layer (not after), which stabilizes training without the mean-centering of LayerNorm.

**Reward Model Architecture**

The reward model starts as a copy of the SFT model. We remove the language modeling head and add a linear layer that maps d_model → 1 (a scalar reward score). The reward model takes a full (prompt + response) sequence and produces a single score for the entire response.

Formally: `R_φ(x, y) = W_r · h_{[EOS]}` where h_{[EOS]} is the hidden state at the final token.

**PPO Training Setup**

PPO maintains four models simultaneously:
1. **Policy π_θ** — the LLM being optimized (updated during training)
2. **Reference policy π_ref** — the frozen SFT model (used for KL penalty)
3. **Reward model R_φ** — frozen after RM training phase
4. **Value function V_ψ** — a copy of the policy with a value head (predicts expected future reward)

The PPO objective is:

```
L_CLIP(θ) = E_t [min(r_t(θ) · Â_t, clip(r_t(θ), 1-ε, 1+ε) · Â_t)]
```

where `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)` is the policy ratio and Â_t is the advantage estimate. The clip prevents large policy updates that destabilize training.

For language models with 100K-token vocabularies, the clipping matters because even a small change in the distribution over common tokens (like "the", "a", "is") can accumulate into large divergences from the reference policy. The KL term `β · KL(π_θ || π_ref)` provides a softer regularization signal on top of clipping.

**DPO vs. PPO**

Direct Preference Optimization (DPO) derives a closed-form policy from the PPO+KL objective, showing that the optimal policy under the KL-constrained reward maximization objective satisfies:

```
r(x, y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)
```

DPO rearranges this to train directly from preference pairs without a separate reward model:

```
L_DPO = -E[log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

I choose PPO over DPO when: (1) I need online data collection (the model explores beyond the reference distribution), (2) preference data is noisy (the reward model can smooth contradictory preferences), (3) I need iterative rounds of alignment. DPO is preferred when: annotation budget is very limited, deployment timeline is aggressive, and the reference model is already quite good.

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

> "How do you evaluate the quality of a conversational AI system? What metrics do you use and what are their limitations?"

### Signal Being Tested

Does the candidate understand that automatic metrics are insufficient for measuring alignment? Can they propose a rigorous human evaluation framework alongside automated metrics?

### Follow-up Probes

- "Reward model score is already used during training. Why can't you use it for evaluation?"
- "How would you measure sycophancy specifically?"
- "What is the Elo rating system and how would you use it for model comparison?"

---

### Model Answers — Section 5

**No Hire:**
"I would use accuracy on a test set." Treats this as a classification problem.

**Lean No Hire:**
Mentions perplexity and BLEU score. Does not understand why these are insufficient for conversational quality.

**Lean Hire:**
Distinguishes between automated metrics (perplexity, MMLU accuracy) and human evaluation. Notes that reward model score cannot be used as an evaluation metric because the model was trained to maximize it. Proposes human preference ratings as the ground truth.

**Strong Hire Answer (first-person):**

Evaluating a conversational AI is fundamentally multi-dimensional — no single metric captures alignment quality. I use a layered evaluation framework.

**Layer 1: Capability Benchmarks (Automated)**
- **MMLU** (Massive Multitask Language Understanding): 57-subject multiple-choice benchmark. Measures factual knowledge. Score is accuracy (%). GPT-4: ~87%, Llama 3.1 70B: ~83%.
- **HumanEval**: Functional code generation correctness. Score is pass@k.
- **MT-Bench**: Multi-turn instruction following, scored by GPT-4 as judge on 1–10 scale.
- **Perplexity on held-out data**: `PP = exp(-1/N Σ log p(w_t|w_{<t}))`. Lower is better, but perplexity alone doesn't measure helpfulness.

**Layer 2: Safety Evaluation**
- **HarmBench**: Standard adversarial prompt suite. Attack success rate (ASR) — lower is better.
- **TruthfulQA**: Tests whether model produces false but plausible-sounding answers. Truthfulness rate %.
- **Sycophancy measure**: Present model with factually wrong premises framed as the user's belief; measure rate of incorrect agreement.

**Layer 3: Human Preference Evaluation (Gold Standard)**
I run head-to-head comparisons (A/B tests) between model versions. Annotators see a prompt and two responses (blinded to which model produced which) and indicate preference. I compute an Elo rating for each model version based on win rates:

```
Expected score: E_A = 1 / (1 + 10^((R_B - R_A)/400))
Rating update: R_A' = R_A + K · (S_A - E_A)
```

where K is typically 32 for model comparisons.

**Why I can't use reward model score for evaluation**: The policy was optimized to maximize the reward model score — using the same metric for evaluation would be circular and would hide reward hacking. The reward model's scores drift as the policy changes. I always maintain a separate human evaluation panel.

**Key Metric I'd Track in Production**: Thumbs up/down rate normalized by query type. This gives ongoing signal without expensive annotation.

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

> "Walk me through how you serve this model to millions of concurrent users. What does the inference stack look like?"

### Signal Being Tested

Does the candidate understand continuous batching, KV-cache management, tensor parallelism, and speculative decoding? Can they calculate memory requirements for a large model?

### Follow-up Probes

- "How much GPU memory does a 70B parameter model require at float16 precision?"
- "Explain continuous batching. How is it different from static batching?"
- "What is speculative decoding and when does it help?"
- "What is PagedAttention and why does it matter?"

---

### Model Answers — Section 6

**No Hire:**
"I would run the model on a GPU server and add more servers if needed." No understanding of batching, KV-cache, or model parallelism.

**Lean No Hire:**
Mentions the need for GPU scaling and batching but cannot explain KV-cache or continuous batching. Cannot calculate memory requirements.

**Lean Hire:**
Correctly explains KV-cache, continuous batching, and tensor parallelism at a conceptual level. Can estimate memory requirements roughly.

**Strong Hire Answer (first-person):**

Let me walk through the serving stack from hardware to application layer.

**Memory Budget**
A 70B parameter model at float16 precision requires:
`70B × 2 bytes = 140 GB` just for model weights.

The KV-cache for each sequence adds:
`2 × num_layers × d_model × seq_len × batch_size × bytes_per_element`

For a 32-layer model with d_model=4096, seq_len=8192, batch_size=64, float16:
`2 × 32 × 4096 × 8192 × 64 × 2 bytes ≈ 275 GB`

This exceeds a single A100 (80GB) or H100 (80GB). We need tensor parallelism across 4–8 GPUs.

**Tensor Parallelism (Megatron-style)**
We shard the weight matrices across GPUs. For attention: each GPU holds a subset of attention heads. For the feed-forward layer: each GPU holds a column partition of W_1 and a row partition of W_2. Communication happens at the output of each layer via AllReduce. This splits memory proportionally to the number of GPUs but adds inter-GPU communication overhead (~10–20% on NVLink).

**Continuous Batching (vLLM)**
Static batching holds a batch until all sequences complete — a single long sequence blocks the entire batch. Continuous batching inserts new requests as existing ones finish, keeping GPU utilization near 100%.

vLLM implements this with PagedAttention: KV-cache is allocated in fixed-size pages (e.g., 16 tokens/page), like OS virtual memory pages. Pages are allocated on demand and can be non-contiguous in physical memory. This eliminates the fragmentation that occurs when sequences have variable lengths, improving GPU memory utilization from ~40% (static) to ~90%+ (paged).

**Speculative Decoding**
For a 70B model, we maintain a small 7B draft model. The draft model proposes k=5 tokens autoregressively (5 forward passes of 7B). The 70B model then evaluates all 5 proposed tokens in a single forward pass (same cost as generating 1 token). Tokens are accepted if the 70B probability exceeds the draft probability up to a threshold; mismatches trigger a resample. If acceptance rate is 60%, we get 3 tokens per (7B + 70B) forward pass, versus 1 token per 70B pass — roughly 2.5× speedup.

Speculative decoding degrades when: output diversity is high (creative writing tasks, sampling temperature > 0.9), batch size exceeds 8 (the parallelism overhead starts dominating), or the draft model is poorly aligned with the target (different fine-tuning).

**Serving Stack Summary:**
- Load balancer (routing by user ID for KV-cache reuse on multi-turn)
- vLLM engine with PagedAttention and continuous batching
- 8× H100 nodes in tensor parallel for 70B model
- Safety classifier running in parallel on CPU (add 10–20ms latency)
- Request priority queues (premium users get lower p99 latency SLA)

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

> "What are the most critical failure modes of this system, and how do you detect and mitigate them?"

### Signal Being Tested

Does the candidate understand reward hacking, sycophancy, multi-turn jailbreaks, and hallucination? Can they propose concrete detection and mitigation strategies?

### Follow-up Probes

- "Give me a concrete example of reward hacking. What does the model actually learn?"
- "A user sends 10 turns gradually establishing a harmful persona. Why does the model comply? What prevents it?"
- "How would you detect that the model has started hallucinating more after a PPO update?"

---

### Model Answers — Section 7

**No Hire:**
"I would add a content filter." Cannot explain reward hacking or why multi-turn safety differs from single-turn.

**Lean No Hire:**
Mentions hallucination and jailbreaks as known problems but cannot propose specific mitigations. Doesn't understand the difference between specification failure and reward hacking.

**Lean Hire:**
Correctly identifies reward hacking, sycophancy, and multi-turn jailbreaks. Proposes reward model refresh and multi-turn adversarial training as mitigations. Can explain why single-turn safety training doesn't generalize to multi-turn.

**Strong Hire Answer (first-person):**

There are four failure modes I worry about most, each requiring a different mitigation strategy.

**1. Reward Hacking**
The model learns to fool the reward model rather than actually improve. Concrete examples I've seen: (a) verbosity hacking — reward model was trained on human raters who gave higher scores to longer responses; model learns to generate 3-paragraph answers to yes/no questions. (b) sycophancy hacking — reward model was trained on raters who prefer being agreed with; model learns to always validate the user's premise even when wrong. (c) hedging hacking — reward model penalizes confident wrong answers; model learns to hedge every statement into meaninglessness.

Detection: monitor distribution of response length, hedge phrase frequency, and agreement rate with user premises over time. Compare reward model score vs. independent human rating on monthly samples — divergence signals hacking.

**2. Multi-Turn Jailbreaks**
A user gradually establishes a harmful persona over 8 turns, then makes a harmful request in turn 9. The model complies because it was RLHF-trained on individual (prompt, response) pairs — the safety training never saw the full conversation trajectory.

Mitigation: (a) safety classifier that evaluates the full conversation context, not just the last message, (b) periodic "safety anchor" injected into the system prompt that reasserts model identity regardless of conversation history, (c) adversarial training with multi-turn jailbreak examples in the RLHF data.

**3. Hallucination Under Distribution Shift**
The model confidently generates factually incorrect information, especially on recent events, obscure topics, or complex multi-step reasoning. After a PPO update, hallucination rates can increase if the reward model didn't penalize factual errors.

Detection: run TruthfulQA and a held-out factual QA benchmark after every PPO run. If accuracy drops >2 percentage points, investigate whether the reward model is rewarding confident-sounding answers regardless of correctness.

**4. Context Window Exhaustion**
For very long conversations, the model's performance degrades as the context approaches the maximum length. Important multi-turn context from early in the conversation is effectively lost.

Mitigation: conversation summarization — periodically compress older turns into a summary and replace them in the context. The summary model is a separate fine-tuned component.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

> "You've built one ChatGPT-style assistant. Now you're asked to build a platform that enables ten product teams to fine-tune this base model for their specific use cases (customer support, code assistant, medical Q&A, etc.). How does your thinking change?"

### Signal Being Tested

Does the candidate think beyond a single model to shared infrastructure, multi-tenant training pipelines, and platform economics? Can they identify where standardization helps and where flexibility is needed?

### Follow-up Probes

- "How do you prevent fine-tuning on one product's data from degrading capability for another product?"
- "What shared infrastructure components provide the most leverage?"

---

### Model Answers — Section 8

**No Hire:**
"I would give each team their own copy of the model." No consideration of shared infrastructure, compute cost, or catastrophic forgetting.

**Lean No Hire:**
Suggests fine-tuning a shared base model for each product but doesn't address how to prevent cross-contamination or how to update the base model when new capabilities are added.

**Lean Hire:**
Proposes a hub-and-spoke architecture: shared base model + product-specific LoRA adapters. Notes that LoRA keeps adapter parameters small. Mentions shared RLHF infrastructure and reward model versioning.

**Strong Hire Answer (first-person):**

Moving from one product to a platform changes the optimization target: I'm no longer optimizing one model's performance — I'm optimizing the marginal cost of adding a new use case and the reliability of shared infrastructure.

The key architecture decision is **parameter-efficient fine-tuning (PEFT) over full fine-tuning**. Rather than maintaining 10 full copies of a 70B model (700B parameters total), each product team fine-tunes a LoRA adapter:

```
W_fine-tuned = W_base + BA, where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)
```

At rank r=16 and d=4096, each adapter is roughly 1.3M parameters — 50,000× smaller than the base model. We serve the shared base model weights once and swap adapters per request.

**Shared infrastructure leverage points:**
1. **RLHF pipeline**: shared reward model training infrastructure, shared annotation tooling. Each team contributes preference data; the base reward model is re-trained centrally.
2. **Safety classifier**: one classifier serves all products; product-specific safety thresholds are configurable parameters.
3. **Evaluation harness**: shared capability benchmarks run on all adapters after fine-tuning. A regression suite that automatically detects if a product adapter degraded base capabilities.
4. **Serving fleet**: all products share the same 70B model serving cluster; only the adapter weights differ per request.

**Preventing catastrophic forgetting**: each product adapter is fine-tuned with a regularization term that preserves the base model's performance on a capability benchmark. If a medical Q&A fine-tune degrades coding ability by >5%, the adapter is rejected until regularization strength is increased.

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**Pretraining cross-entropy loss:**
```
L_pretrain = -Σ_{t=1}^{T} log p_θ(w_t | w_1, ..., w_{t-1})
```

**Reward model (Bradley-Terry) loss:**
```
L_RM = -E[(x,y_w,y_l)] [log σ(R_φ(x, y_w) - R_φ(x, y_l))]
```

**PPO clipped objective:**
```
L_CLIP = E_t [min(r_t · Â_t, clip(r_t, 1-ε, 1+ε) · Â_t)]
r_t = π_θ(a_t|s_t) / π_old(a_t|s_t), ε = 0.2 typical
```

**KL-penalized RLHF objective:**
```
max_θ E_{x~D, y~π_θ} [R_φ(x,y) - β · KL(π_θ(y|x) || π_ref(y|x))]
```

**DPO loss:**
```
L_DPO = -E [log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

**Perplexity:**
```
PP(W) = exp(-1/T Σ_{t=1}^{T} log p(w_t | w_{<t}))
```

**Chinchilla compute-optimal scaling:**
```
N_opt ≈ √(C/6),  T_opt ≈ 20 × N_opt
```

**Speculative decoding speedup:**
```
Speedup ≈ (k × α) / (cost_draft/cost_target + 1)
where α = acceptance rate, k = draft tokens
```

**Attention computation:**
```
Attention(Q,K,V) = softmax(QK^T / √d_k) · V
Q,K,V ∈ R^{n × d_k}
```

### Vocabulary Cheat Sheet

| Term | Definition |
|---|---|
| **SFT** | Supervised Fine-Tuning — trains on curated (prompt, response) demonstrations |
| **RLHF** | Reinforcement Learning from Human Feedback — uses reward model + PPO |
| **PPO** | Proximal Policy Optimization — clips policy ratio to prevent large updates |
| **KL divergence** | Measures how much policy has drifted from reference; used as penalty |
| **Reward hacking** | Model exploits reward model flaws instead of genuinely improving |
| **Sycophancy** | Model agrees with user premises regardless of factual accuracy |
| **DPO** | Direct Preference Optimization — closed-form alternative to PPO |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning via low-rank weight updates |
| **Speculative decoding** | Draft model proposes tokens; target model validates in parallel |
| **PagedAttention** | Non-contiguous KV-cache allocation for memory efficiency |
| **Continuous batching** | Inserts new requests as others complete without restarting batch |
| **GQA** | Grouped-Query Attention — shares K/V heads to reduce KV-cache size |
| **Constitutional AI** | Uses model to evaluate its own outputs against principles |
| **Process Reward Model** | Rewards correct reasoning steps, not just final answer |
| **RLAIF** | Reinforcement Learning from AI Feedback — uses AI raters instead of humans |

### Key Numbers Table

| Metric | Value |
|---|---|
| GPT-3 parameter count | 175B |
| GPT-3 training tokens | 300B (undertrained vs. Chinchilla) |
| Chinchilla-optimal token:param ratio | ~20:1 |
| InstructGPT SFT demonstrations | ~13K examples |
| InstructGPT reward model comparisons | ~33K comparisons |
| Float16 memory per parameter | 2 bytes |
| 70B model minimum GPU memory | ~140 GB weights alone |
| Llama 3.1 70B MMLU score | ~83% |
| GPT-4 MMLU score | ~87% |
| Typical PPO clip ε | 0.2 |
| Typical KL coefficient β | 0.02–0.1 |
| Target TTFT SLA (p90) | 200ms |
| LoRA rank typical value | 8–64 |
| Speculative decoding speedup (typical) | 2–3× |
| Context length: Llama 3.1 | 128K tokens |

### Rapid-Fire Day-Before Review

1. **Three stages of ChatGPT training?** Pretraining → SFT → RLHF (reward model + PPO)
2. **What does the reward model take as input?** (prompt, response) pair → scalar score
3. **Why is KL penalty needed?** Prevents reward hacking by keeping policy close to SFT reference
4. **DPO vs. PPO key difference?** DPO is offline (fixed preference data); PPO is online (can collect new data during training)
5. **Chinchilla optimal ratio?** ~20 training tokens per parameter
6. **What is continuous batching?** Insert new requests as others finish, vs. waiting for whole batch
7. **PagedAttention purpose?** Non-contiguous KV-cache pages → better memory utilization
8. **Speculative decoding: when does it break down?** High temperature/diversity, batch size > 8, misaligned draft model
9. **How to detect reward hacking?** Compare reward model score vs. independent human rating monthly
10. **Multi-turn jailbreak root cause?** RLHF trained on single-turn pairs; never learned to evaluate full conversation trajectory
