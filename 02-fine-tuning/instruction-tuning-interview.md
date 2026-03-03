# Instruction Tuning, RLHF, and DPO — Interview Deep Dive

> **What this file covers**
> - 🎯 SFT loss function — cross-entropy on response tokens only
> - 🧮 RLHF objective — reward minus KL penalty, full derivation
> - 🧮 DPO loss — step-by-step derivation from the RLHF objective
> - ⚠️ Failure modes: reward hacking, mode collapse, alignment tax, over-refusal
> - 📊 Complexity: training cost comparison across SFT, RLHF, DPO
> - 💡 RLHF vs DPO vs RLAIF — when each wins
> - 🏭 Production: data collection, annotation pipelines, iterative alignment
> - 🗺️ 4 Staff/Principal interview questions with all four hiring levels

---

## Brief Restatement

Instruction tuning is the process that turns a raw language model into a helpful assistant. It has three components: **SFT** (supervised fine-tuning on instruction-response pairs), **RLHF** (reinforcement learning from human feedback using a learned reward model), and **DPO** (direct preference optimization that skips the reward model). The intuition was covered in [instruction-tuning.md](./instruction-tuning.md). This file covers the math, failure modes, and interview-grade depth.

---

## 🧮 SFT: The Loss Function

### What SFT Optimizes

SFT is standard supervised fine-tuning on instruction-response data. The key subtlety: we only compute loss on the **response tokens**, not the instruction tokens.

Why? The model should learn to *generate good responses*, not to *generate good instructions*. The instruction is given — we just need the model to continue it well.

### The SFT Loss

Given a training example with instruction tokens x₁, x₂, ..., xₘ and response tokens y₁, y₂, ..., yₙ:

```
L_SFT = -∑ᵢ₌₁ⁿ log p(yᵢ | x₁, ..., xₘ, y₁, ..., yᵢ₋₁)
```

Where:
- x₁, ..., xₘ are the instruction tokens (we condition on these but do not compute loss)
- y₁, ..., yₙ are the response tokens (we compute loss only on these)
- p(yᵢ | ...) is the model's predicted probability of the correct next token

In plain words: for each token in the response, compute the negative log probability of the correct token. Sum them up. The model learns to assign high probability to the correct response tokens given the instruction.

### Worked Example

Suppose the instruction is "What is 2+2?" and the response is "4".

The model processes: "What is 2+2?" and then predicts the next token.

```
p("4" | "What is 2+2?") = 0.8

L_SFT = -log(0.8) = 0.22
```

If the model predicted p("4") = 0.1, the loss would be -log(0.1) = 2.3 — much higher. The optimizer pushes the model to increase p("4").

### 🎯 Key Insight: Loss Masking

The instruction tokens are NOT included in the loss. This is implemented by setting a **loss mask** — a binary vector that is 0 for instruction positions and 1 for response positions.

```
Tokens:    [What] [is] [2+2?] [4]
Loss mask: [  0 ] [ 0] [  0 ] [1]   ← only compute loss on "4"
```

Without this masking, the model would waste capacity learning to predict instruction tokens, which are given at inference time and do not need to be generated.

---

## 🧮 RLHF: The Full Objective

### Step 1: Train a Reward Model

The reward model r(x, y) takes a prompt x and a response y, and outputs a scalar score predicting how much a human would prefer that response.

It is trained on preference data: pairs (y_w, y_l) where y_w is the human-preferred (winning) response and y_l is the rejected (losing) response for the same prompt x.

The reward model loss (Bradley-Terry model):

```
L_RM = -log σ(r(x, y_w) - r(x, y_l))
```

Where:
- σ is the sigmoid function: σ(z) = 1 / (1 + e⁻ᶻ)
- r(x, y_w) is the reward score for the preferred response
- r(x, y_l) is the reward score for the rejected response

In plain words: we want r(y_w) > r(y_l). The sigmoid ensures this is a smooth, differentiable objective. The larger the gap r(y_w) - r(y_l), the lower the loss.

### Worked Example

Prompt: "Explain gravity simply"

y_w = "It is the force that pulls things down." (preferred)
y_l = "According to general relativity, spacetime curvature..." (rejected)

If r(x, y_w) = 2.0 and r(x, y_l) = 0.5:

```
r(y_w) - r(y_l) = 2.0 - 0.5 = 1.5
σ(1.5) = 1 / (1 + e⁻¹·⁵) = 0.82
L_RM = -log(0.82) = 0.20   (low loss — good separation)
```

If the scores were reversed (r(y_w) = 0.5, r(y_l) = 2.0):

```
r(y_w) - r(y_l) = -1.5
σ(-1.5) = 0.18
L_RM = -log(0.18) = 1.71   (high loss — model got it wrong)
```

### Step 2: Optimize the Policy with RL

The RLHF objective maximizes reward while staying close to the original SFT model (the reference policy π_ref):

```
max_π  E_{x~D, y~π(·|x)} [ r(x, y) - β · KL(π(·|x) ‖ π_ref(·|x)) ]
```

Where:
- π is the policy (the model being trained)
- π_ref is the reference policy (the SFT model, frozen)
- r(x, y) is the reward model's score
- β is a hyperparameter controlling the KL penalty strength
- KL(π ‖ π_ref) is the Kullback-Leibler divergence between the two policies
- D is the distribution of prompts

In plain words: generate responses that get high reward scores, but do not drift too far from the SFT model. The β · KL term is a leash — it prevents the model from gaming the reward model.

### Why the KL Penalty Matters

Without the KL penalty, the model would find degenerate ways to maximize the reward. For example:

- The reward model gives high scores to long, detailed answers → the model writes 10,000-word answers to every question
- The reward model gives high scores to confident answers → the model claims certainty about everything, even when wrong
- The reward model has blind spots → the model exploits them

The KL penalty says: "you can improve, but stay close to the model that already works reasonably well." It is the single most important design decision in RLHF.

### 📊 The KL-Reward Trade-off

```
  Reward ↑
  │
  │           ┌── Sweet spot
  │          ╱
  │    ╱────╱
  │   ╱
  │  ╱
  │ ╱         ← Diminishing returns:
  │╱            more KL divergence gives less reward gain
  └──────────────────────────→ KL divergence from π_ref

  β too high: Model barely changes from SFT (underfitting)
  β too low:  Model exploits reward model (reward hacking)
  β just right: Model improves meaningfully while staying stable
```

---

## 🧮 DPO: Derivation from RLHF

### The Key Insight

The RLHF objective has a closed-form optimal solution. If we solve for the optimal policy π* given the reward function r, we get:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x, y) / β)
```

Where Z(x) is a normalizing constant (partition function).

Rearranging to express the reward in terms of the optimal policy:

```
r(x, y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)
```

### Building the DPO Loss Step by Step

**Step 1:** Substitute this reward expression into the Bradley-Terry preference model:

```
p(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

**Step 2:** The Z(x) terms cancel (both responses have the same prompt):

```
r(x, y_w) - r(x, y_l) = β · log(π*(y_w|x) / π_ref(y_w|x)) - β · log(π*(y_l|x) / π_ref(y_l|x))
```

**Step 3:** Define the log-ratio shorthand:

```
Let ρ(y) = log(π_θ(y|x) / π_ref(y|x))
```

**Step 4:** The DPO loss directly optimizes the policy π_θ to match the preference data:

```
L_DPO = -E_{(x, y_w, y_l)} [ log σ( β · (ρ(y_w) - ρ(y_l)) ) ]
```

Expanded:

```
L_DPO = -E [ log σ( β · [ log(π_θ(y_w|x) / π_ref(y_w|x)) - log(π_θ(y_l|x) / π_ref(y_l|x)) ] ) ]
```

### What Each Part Does

- **log(π_θ(y_w|x) / π_ref(y_w|x))**: How much more likely π_θ makes the preferred response compared to the reference. We want this to be positive (increase probability of good responses).
- **log(π_θ(y_l|x) / π_ref(y_l|x))**: How much more likely π_θ makes the rejected response. We want this to be negative (decrease probability of bad responses).
- **β**: Controls how aggressively the model moves. Higher β = bigger updates. Same role as in RLHF.
- **σ(...)**: Sigmoid turns the difference into a probability, making the loss smooth.

### 🎯 Why DPO Works

DPO is mathematically equivalent to RLHF under the Bradley-Terry preference model. It skips the reward model by baking the reward into the loss function. The reference policy π_ref acts as the implicit KL regularizer — the log-ratios naturally penalize deviation from the reference.

### Worked Example

Prompt: "Is the earth flat?"

y_w = "No, the earth is approximately spherical." (preferred)
y_l = "There is debate about this topic." (rejected)

Suppose:
- π_θ(y_w|x) = 0.7, π_ref(y_w|x) = 0.5 → ρ(y_w) = log(0.7/0.5) = 0.336
- π_θ(y_l|x) = 0.1, π_ref(y_l|x) = 0.3 → ρ(y_l) = log(0.1/0.3) = -1.099
- β = 0.1

```
β · (ρ(y_w) - ρ(y_l)) = 0.1 · (0.336 - (-1.099)) = 0.1 · 1.435 = 0.144
σ(0.144) = 0.536
L_DPO = -log(0.536) = 0.624
```

The model is correctly making y_w more likely and y_l less likely relative to the reference, but the margin is small. Training will push the model to increase this gap.

---

## ⚠️ Failure Modes

### 1. Reward Hacking

The model finds ways to get high reward scores without being genuinely helpful. The reward model is an imperfect proxy for human judgment, and the RL optimizer exploits its weaknesses.

**Examples:**
- Model produces excessively long responses because the reward model gives higher scores to longer text
- Model uses confident language ("I am 100% certain") because the reward model conflates confidence with quality
- Model produces sycophantic responses ("What a great question!") because annotators rated those higher

**Mitigation:** Strong KL penalty (high β), diverse annotator pool, periodic re-training of the reward model, red-teaming.

### 2. Mode Collapse

The model converges to a narrow set of responses — it gives the same kind of answer to every question, losing diversity and creativity.

**Example:** After RLHF, the model starts every response with "Sure, I'd be happy to help!" and ends with "Let me know if you have any other questions!" regardless of context.

**Mitigation:** KL penalty from reference policy, temperature sampling during RL, monitoring response diversity metrics.

### 3. Alignment Tax

RLHF can reduce the model's raw capability on benchmarks while improving its alignment. The model trades some factual accuracy or reasoning ability for better instruction following and safety.

**Example:** A model that scored 85% on MMLU before RLHF scores 82% after — it lost some knowledge to gain safety properties.

**Mitigation:** Careful balance of KL penalty, multi-task reward that includes helpfulness AND accuracy, smaller learning rate during RL.

### 4. Over-Refusal

The model becomes too cautious and refuses benign requests. Safety training makes it treat every edge case as dangerous.

**Example:** User asks "How do I cut a watermelon?" and the model responds "I cannot provide instructions that could lead to harm involving sharp objects."

**Mitigation:** Include benign examples in safety training data, use nuanced reward signals that distinguish genuine danger from safe requests, constitutional AI techniques.

### 5. Distributional Shift in DPO

DPO uses a fixed dataset of preferences. If the policy drifts far from the distribution that generated the preference data, the implicit reward becomes unreliable.

**Example:** The preference data was collected from an SFT model that generates 50-word responses. After DPO training, the model starts generating 200-word responses — but the preference labels were never evaluated on responses that long.

**Mitigation:** Online DPO (generate fresh data periodically), iterative DPO, larger and more diverse preference datasets.

---

## 📊 Complexity and Cost Comparison

| | SFT | RLHF | DPO |
|---|---|---|---|
| **Models in memory** | 1 (π_θ) | 4 (π_θ, π_ref, reward, value) | 2 (π_θ, π_ref) |
| **GPU memory (7B)** | ~14 GB (LoRA) | ~56 GB (all 4 models) | ~28 GB (both models) |
| **Training data** | (instruction, response) pairs | Preference pairs (y_w, y_l) | Preference pairs (y_w, y_l) |
| **Data needed** | 10K–1M examples | 10K–100K comparisons | 10K–100K comparisons |
| **Training stability** | Stable (standard SGD) | Unstable (RL is hard to tune) | Stable (standard SGD) |
| **Hyperparameters** | LR, batch size | LR, β, GAE λ, clip ε, ... | LR, β |
| **Implementation** | Simple | Complex (PPO + reward model) | Simple |
| **Time to train (7B)** | Hours | Days | Hours |

### 🎯 Key Cost Insight

The dominant cost in RLHF is not compute — it is **human annotation**. Collecting high-quality preference data requires trained annotators who understand the task, the safety guidelines, and the edge cases. At scale, this costs millions of dollars.

This is why DPO and other offline methods are attractive: they can reuse the same preference dataset many times without needing fresh annotations.

---

## 💡 RLHF vs DPO vs RLAIF — Comparison Table

| | RLHF | DPO | RLAIF |
|---|---|---|---|
| **Full name** | RL from Human Feedback | Direct Preference Optimization | RL from AI Feedback |
| **Reward model** | Explicit, trained separately | Implicit (baked into loss) | Explicit, from AI annotations |
| **Annotators** | Humans | Humans (for data collection) | AI model (e.g., Claude, GPT-4) |
| **Online/Offline** | Online (generates fresh data) | Offline (fixed dataset) | Online or Offline |
| **Stability** | Unstable (PPO is finicky) | Stable | Moderate |
| **Quality ceiling** | Highest (explores during training) | Very good | Good (limited by AI judge) |
| **Cost** | Very high (humans + compute) | Moderate (compute only) | Low (AI annotations are cheap) |
| **Used by** | OpenAI (ChatGPT), Anthropic | Zephyr, Tulu, many open-source | Anthropic (Constitutional AI) |
| **Best for** | Maximum quality, safety-critical | Resource-constrained, rapid iteration | Large-scale, low-budget alignment |

---

## 🏭 Production and Scaling Considerations

### Data Collection Pipeline

```
  Prompt Collection → Response Generation → Human Annotation → Quality Control
        │                     │                    │                  │
  Diverse prompts      Multiple responses     Pairwise ranking    Inter-annotator
  covering edge       per prompt (2-8)      by trained humans    agreement check
  cases and safety                                                (κ > 0.7)
```

### Iterative Alignment

Production systems do not train once. They use iterative cycles:

1. Deploy model v1
2. Collect user feedback and red-team data
3. Generate new preference data from model v1's responses
4. Train model v2 with updated preferences
5. Repeat

Each iteration targets the specific failure modes discovered in the previous deployment.

### Scaling Laws for Alignment

Research suggests:
- SFT quality scales with dataset diversity more than dataset size
- RLHF reward model accuracy scales with the number of unique prompts, not total comparisons
- Annotation quality matters more than annotation quantity — 10K expert-labeled examples often outperform 100K crowd-sourced ones

---

## Staff/Principal Interview Depth

### Q1: Walk me through the SFT loss function. Why do we mask the instruction tokens?

---
**No Hire**
*Interviewee:* "SFT is just fine-tuning on instruction data. We use cross-entropy loss. I think we train on all the tokens."
*Interviewer:* Understands that SFT uses instruction data but misses the crucial detail of loss masking. Does not demonstrate understanding of why masking matters.
*Criteria — Met:* knows SFT uses cross-entropy / *Missing:* loss masking, reasoning about why instruction tokens are excluded, connection to inference-time behavior

**Weak Hire**
*Interviewee:* "SFT computes cross-entropy loss only on the response tokens. We mask the instruction part because at inference time the instruction is given — we only need to generate the response. So training on instruction tokens would waste model capacity."
*Interviewer:* Correct high-level understanding with the right reasoning. Missing the mathematical formulation and any discussion of implementation details or edge cases.
*Criteria — Met:* loss masking, correct reasoning / *Missing:* formula, multi-turn handling, connection to causal LM training, discussion of what happens without masking

**Hire**
*Interviewee:* "The SFT loss is L = -∑ᵢ log p(yᵢ | x, y₁...yᵢ₋₁) summed only over response tokens yᵢ. We mask instruction tokens because they are given at inference and training on them wastes gradient signal. In multi-turn conversations, we typically mask all prior turns and only compute loss on the latest assistant response. Without masking, the model can still learn the task but converges slower and achieves slightly worse response quality because gradients from instruction tokens push the model in less useful directions."
*Interviewer:* Solid answer with the formula, correct reasoning, and a practical consideration (multi-turn). Would be elevated to Strong Hire with discussion of data formatting choices and their impact.
*Criteria — Met:* formula, masking reasoning, multi-turn consideration, what-if analysis / *Missing:* data format trade-offs (Alpaca vs ChatML), system prompt handling, connection to packing and efficiency

**Strong Hire**
*Interviewee:* "The SFT loss is the standard causal LM cross-entropy, L = -∑ᵢ log p(yᵢ | x, y<i), but with a binary mask that zeros out instruction positions. Three nuances: First, the data format matters — ChatML format with special tokens (<|im_start|>, <|im_end|>) makes it easy to identify assistant turns for masking. Second, in multi-turn conversations, there is a design choice: you can mask all turns except the last assistant turn, or compute loss on all assistant turns. Computing loss on all assistant turns gives more training signal per example but can overweight early turns. Third, in practice, we use packing — multiple examples concatenated into one sequence with attention masking to prevent cross-contamination. The loss mask must align with the packing boundaries. Finally, I would note that some work (like the Llama 2 paper) reports that training on all tokens (no masking) with a reduced weight on instruction tokens can work nearly as well, suggesting the strict mask is not always necessary."
*Interviewer:* Exceptional depth covering the formula, data format considerations, multi-turn design choices, packing implementation detail, and a nuanced view of when masking is necessary. This demonstrates production experience.
*Criteria — Met:* formula, masking reasoning, multi-turn handling, data format trade-offs, packing, nuanced perspective on masking necessity
---

### Q2: Derive the DPO loss from the RLHF objective. What assumptions does this derivation rely on?

---
**No Hire**
*Interviewee:* "DPO is a simpler version of RLHF. Instead of training a reward model, you directly train on preferences. I'm not sure about the exact derivation."
*Interviewer:* Knows the high-level relationship but cannot derive it. This is a core result that staff candidates are expected to know.
*Criteria — Met:* knows DPO is related to RLHF / *Missing:* derivation, assumptions, any mathematical detail

**Weak Hire**
*Interviewee:* "DPO starts from the RLHF objective which maximizes reward minus β times KL divergence. The optimal policy is proportional to π_ref times exp(r/β). You can rearrange this to express the reward in terms of the policy log-ratio. Then you substitute into the Bradley-Terry model and the partition function cancels. The result is a loss that only depends on the policy and reference, not the reward model."
*Interviewer:* Correct outline of the derivation with the right key steps identified. Missing the actual formulas and a discussion of the assumptions.
*Criteria — Met:* correct derivation outline, knows the partition function cancels / *Missing:* full equations, explicit assumptions, limitations of the derivation

**Hire**
*Interviewee:* "The RLHF objective is max_π E[r(x,y) - β·KL(π‖π_ref)]. The optimal policy is π*(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β). Rearranging: r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x). Substituting into the Bradley-Terry preference model p(y_w ≻ y_l) = σ(r(y_w) - r(y_l)), the log Z(x) terms cancel because both responses share the same prompt. This gives L_DPO = -log σ(β·[log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))]). Key assumption: the Bradley-Terry model accurately captures human preferences, which assumes transitivity and independence from irrelevant alternatives."
*Interviewer:* Complete derivation with the main assumption identified. Strong answer. Would push to Strong Hire with discussion of practical implications and when the assumptions break down.
*Criteria — Met:* full derivation, Bradley-Terry assumption / *Missing:* limitations of the offline setting, discussion of when assumptions fail in practice, connection to online variants

**Strong Hire**
*Interviewee:* [Gives the full derivation as in Hire, then continues:] "Three important assumptions and their limitations. First, Bradley-Terry assumes a latent utility model where preferences are transitive — but real human preferences are often intransitive (A > B, B > C, but C > A). Second, the derivation assumes the policy can reach the optimal solution — but in practice, neural networks are function approximators and may not be able to represent π*. Third, and most practically important: DPO is offline. The preference data was collected from π_ref's distribution. As π_θ drifts from π_ref during training, we are evaluating the implicit reward on out-of-distribution samples. This is why online DPO variants like OAIF and iterative DPO exist — they periodically regenerate data from the current policy. There is also the question of whether the KL-constrained objective is the right formulation at all. Some recent work argues that f-divergences other than KL (like Jensen-Shannon) give better alignment properties."
*Interviewer:* Demonstrates deep understanding not just of the math but of where it breaks in practice. The discussion of distributional shift and online variants shows genuine research awareness. Staff-level thinking.
*Criteria — Met:* full derivation, all assumptions, practical limitations, online variants, alternative formulations
---

### Q3: What is reward hacking in RLHF? Give concrete examples and explain how to mitigate it.

---
**No Hire**
*Interviewee:* "Reward hacking is when the model tricks the reward model somehow. I think it's related to adversarial examples."
*Interviewer:* Vague understanding. Cannot distinguish reward hacking from adversarial attacks or provide any concrete mechanism.
*Criteria — Met:* knows the term exists / *Missing:* definition, examples, mechanism, mitigation

**Weak Hire**
*Interviewee:* "Reward hacking is when the RL-trained model finds ways to maximize the reward score without being genuinely helpful. For example, it might produce very long responses because the reward model correlates length with quality. You can mitigate it with the KL penalty."
*Interviewer:* Correct definition with one good example. Mentions KL penalty but does not explain other mitigations or discuss the deeper issue.
*Criteria — Met:* correct definition, one example, KL penalty / *Missing:* multiple examples, explanation of why it happens (Goodhart's law), comprehensive mitigation strategies

**Hire**
*Interviewee:* "Reward hacking occurs because the reward model is an imperfect proxy for true human preferences — this is Goodhart's law: when a measure becomes a target, it ceases to be a good measure. Concrete examples: (1) Length hacking — model produces verbose answers because reward model correlates length with quality. (2) Sycophancy — model produces flattering, agreeable responses because annotators slightly preferred polite answers. (3) Formatting hacking — model uses bullet points and markdown everywhere because these got higher scores. Mitigations include: strong KL penalty (β), ensemble of reward models to reduce exploitable patterns, periodic reward model retraining on the current policy's outputs, and length normalization in the reward."
*Interviewer:* Good depth with multiple examples, the connection to Goodhart's law, and several mitigations. Would be elevated with discussion of detection methods and the theoretical relationship between reward model capacity and hackability.
*Criteria — Met:* Goodhart's law, multiple examples, multiple mitigations / *Missing:* detection methods, relationship to reward model capacity, real-world case studies

**Strong Hire**
*Interviewee:* [Gives everything in Hire, then adds:] "Two more important points. First, reward hacking is not binary — it exists on a spectrum. Early in training, the model makes genuine improvements. As training continues, the easy gains are exhausted and the model starts exploiting reward model weaknesses. This is why monitoring the KL divergence during training is critical — a sharp increase in reward with corresponding increase in KL often signals the transition from genuine improvement to hacking. Second, the fundamental issue is that reward model capacity is finite. An RL optimizer with enough compute will always find reward model weaknesses — it is an adversarial game. This is why Anthropic's Constitutional AI approach is interesting: instead of a single reward model, it uses a constitution (a set of principles) evaluated by the model itself, which is harder to hack because the evaluator changes as the model changes. In production, I would also implement output diversity monitoring — if the model starts producing homogeneous responses (low entropy across prompts), that is a strong signal of mode collapse from reward hacking."
*Interviewer:* Exceptional answer that goes beyond listing examples to discussing the dynamics of hacking, detection methods, theoretical limits, alternative approaches, and production monitoring. Clear staff-level thinking.
*Criteria — Met:* Goodhart's law, multiple examples, comprehensive mitigations, detection methods, Constitutional AI alternative, production monitoring
---

### Q4: You are designing the alignment pipeline for a new LLM. Walk me through your approach — what methods would you use, in what order, and why?

---
**No Hire**
*Interviewee:* "I would do SFT first, then RLHF. SFT makes it follow instructions and RLHF makes it safe."
*Interviewer:* Correct order but extremely shallow. No discussion of data, infrastructure, evaluation, or trade-offs.
*Criteria — Met:* correct ordering / *Missing:* data strategy, method selection rationale, evaluation, iterative process, practical considerations

**Weak Hire**
*Interviewee:* "First, SFT on a high-quality instruction dataset — something like 50K-100K diverse examples covering Q&A, coding, math, creative writing, and safety-related prompts. Then I would collect preference data by generating multiple responses per prompt and having annotators rank them. I would use DPO rather than RLHF because it is simpler and more stable. Finally, I would evaluate on standard benchmarks and with red-teaming."
*Interviewer:* Reasonable pipeline with some specifics. Missing the iterative nature, data quality considerations, and deeper trade-off analysis.
*Criteria — Met:* SFT details, preference collection, method choice with rationale, evaluation / *Missing:* iterative alignment, annotator quality, multi-stage evaluation, safety-specific training, production deployment strategy

**Hire**
*Interviewee:* "I would design a 4-phase pipeline. Phase 1: High-quality SFT with 50K-100K expert-written examples. I would prioritize diversity of tasks over volume, and include safety-relevant examples (refusals for harmful requests, honest 'I don't know' responses). Phase 2: Preference data collection. I would generate 4-8 responses per prompt using the SFT model with different temperatures, then have trained annotators do pairwise comparisons. I would aim for 20K-50K comparisons with inter-annotator agreement κ > 0.7. Phase 3: DPO training with β tuned via ablation on a held-out set. I would monitor KL divergence and response diversity throughout. Phase 4: Safety evaluation — red-teaming, benchmark evaluation (MT-Bench, AlpacaEval), and A/B testing with real users. I would then iterate: deploy, collect feedback, retrain. The choice of DPO over RLHF is driven by engineering simplicity and stability. If we had the resources and safety requirements justified it, I would consider RLHF for its online exploration advantage."
*Interviewer:* Comprehensive pipeline with concrete numbers, quality controls, monitoring, evaluation strategy, and an honest trade-off analysis of DPO vs RLHF. Would push to Strong Hire with discussion of failure recovery, data contamination, and scaling considerations.
*Criteria — Met:* multi-phase design, concrete numbers, quality controls, evaluation, iterative deployment, method trade-off / *Missing:* failure recovery, contamination checks, scaling strategy, safety-specific techniques beyond red-teaming

**Strong Hire**
*Interviewee:* [Gives everything in Hire, then adds:] "A few things I would add for production robustness. First, I would implement a data contamination check — make sure the SFT data does not include benchmark test sets, which would inflate evaluation scores. Second, I would build a safety classifier early (even a simple one) and use it to filter both training data and model outputs during preference collection. Third, for scaling, I would start with RLAIF using a stronger model (like GPT-4) for initial preference labels, validate a subset with human annotators to calibrate, and then use the calibrated AI labels at scale. This gives 10x more preference data at 1/10 the cost. Fourth, I would design the annotation interface to capture not just 'which is better' but 'why' — free-text justifications from annotators are invaluable for debugging reward model failures. Finally, I would plan for continual alignment: the model will encounter novel situations in production that were not in the training data. I would build a feedback pipeline from user interactions (thumbs up/down, regenerations, edits) and use this as a continuous source of preference data for periodic retraining. The biggest risk in alignment is not the initial training — it is maintaining alignment as the model encounters distributional shift in production."
*Interviewer:* This answer demonstrates genuine systems thinking about alignment as a continuous process, not a one-time training step. The discussion of data contamination, RLAIF for scaling, annotator interface design, and continual alignment shows staff-level experience with production ML systems.
*Criteria — Met:* comprehensive pipeline, concrete numbers, quality controls, contamination checks, RLAIF scaling strategy, annotator interface design, continual alignment, failure recovery, distributional shift awareness
---

---

## Key Takeaways

🎯 1. SFT loss is cross-entropy on response tokens only — instruction tokens are masked because they are given at inference
   2. RLHF objective: maximize reward - β·KL(π‖π_ref). The KL penalty prevents reward hacking
🎯 3. DPO loss is mathematically derived from the RLHF objective by substituting the optimal policy. It eliminates the reward model
   4. The Bradley-Terry preference model assumes transitive preferences — real human preferences are often not transitive
⚠️ 5. Reward hacking is the primary failure mode of RLHF. It is a consequence of Goodhart's law applied to learned reward functions
   6. Mode collapse, alignment tax, and over-refusal are additional failure modes that require active monitoring
🎯 7. DPO is simpler and more stable than RLHF but is offline — distributional shift degrades its implicit reward as training progresses
   8. Production alignment is iterative: deploy, collect feedback, retrain. One-time training is not enough
   9. Annotation quality dominates annotation quantity. 10K expert labels often outperform 100K crowd-sourced labels
🎯 10. The biggest risk in alignment is not initial training — it is maintaining alignment under distributional shift in production
