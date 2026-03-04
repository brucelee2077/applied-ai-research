# RLHF and Alignment

## Introduction

A pretrained language model is not a useful product. It predicts the most likely next token — which means it's equally happy generating a helpful answer, a harmful instruction, a confident hallucination, or a racist joke. The text it generates is "probable," not "good."

RLHF (Reinforcement Learning from Human Feedback) and its successors are the techniques that bridge this gap — training models to be helpful, harmless, and honest instead of just probable. Understanding alignment is essential for any genAI system design interview where safety, quality, or user preference matters. And at the Staff level, understanding the failure modes of alignment is just as important as understanding the technique itself.

---

## Why Pretraining Alone Is Insufficient

Pretrained LLMs optimize a simple objective: predict the next token given the context. This gives them language fluency and broad knowledge, but it doesn't teach them:

- **Helpfulness:** Following instructions, answering questions directly, being useful
- **Harmlessness:** Refusing dangerous requests, avoiding bias, not generating toxic content
- **Honesty:** Admitting uncertainty, not confabulating facts, citing limitations

**Without alignment, a pretrained model will:**
- Follow harmful instructions enthusiastically (the training data contains harmful text, and the model learned to continue it)
- Generate confidently wrong answers (the internet is full of confidently wrong text)
- Produce toxic content (the training data reflects internet discourse)
- Lack consistent personality or behavior (it can role-play as anyone)

The alignment gap is the distance between "predicts likely text" and "produces useful, safe responses." Closing this gap is the purpose of RLHF and related techniques.

---

## The RLHF Pipeline

RLHF is a three-stage process. Each stage builds on the previous one.

### Stage 1: Supervised Fine-Tuning (SFT)

Train the model on human-written demonstrations of ideal behavior.

**Data:** (prompt, ideal_response) pairs written by human annotators
**Method:** Standard language model fine-tuning (next-token prediction on the ideal responses)
**What it does:** Teaches the model what good behavior looks like — the format, tone, and style of helpful responses

**Limitations:** SFT alone produces a model that generates reasonable responses but has no mechanism to distinguish between "pretty good" and "excellent" — it only knows one demonstration per prompt. It also can't learn from negative examples (what NOT to do).

### Stage 2: Reward Model Training

Train a separate model to predict human preferences.

**Data:** Human annotators compare pairs of model outputs and indicate which is better.
- For each prompt, the model generates multiple responses
- Annotators rank them: "Response A is better than Response B"
- This creates comparison data: (prompt, response_winner, response_loser)

**Model:** The reward model takes (prompt, response) as input and outputs a scalar reward score. It's typically initialized from the SFT model with a different head.

**Training objective:** The reward model should assign higher scores to preferred responses:

`L_reward = -log(σ(r(x, y_w) - r(x, y_l)))`

where y_w is the preferred response and y_l is the rejected response. This is the Bradley-Terry model of pairwise preferences.

**Challenges:**
- **Inter-annotator disagreement:** Different annotators may have different preferences. Need clear guidelines and calibration.
- **Reward hacking:** The RL stage will exploit any weakness in the reward model.
- **Distribution shift:** The reward model was trained on outputs from the SFT model, but the RL stage will push the model to generate different outputs.

### Stage 3: RL Optimization (PPO)

Optimize the language model to maximize the reward model's score, with a constraint to stay close to the SFT model.

**Objective:**

`maximize E[r(x, y)] - β × KL(π || π_SFT)`

- `r(x, y)` = reward model score for the generated response y to prompt x
- `KL(π || π_SFT)` = KL divergence between the current policy and the SFT model
- `β` = controls how much the model can deviate from SFT behavior

**The KL penalty is critical.** Without it, the model would diverge wildly from natural language to exploit the reward model. The KL penalty keeps the model "anchored" to the SFT distribution — it can improve, but not at the cost of becoming degenerate.

**Training instability:** PPO is notoriously hard to tune. The reward landscape changes during training (the policy changes, so the states it visits change), the reward model can be exploited, and mode collapse can occur.

---

## DPO (Direct Preference Optimization)

DPO is a simpler alternative to RLHF that skips the reward model and RL entirely.

**Key insight:** The optimal policy under the RLHF objective has a closed-form solution. Instead of training a reward model and then doing RL, DPO directly optimizes the policy from preference pairs.

**Loss function:**

`L_DPO = -log σ(β × (log π(y_w|x) / π_ref(y_w|x) - log π(y_l|x) / π_ref(y_l|x)))`

Where:
- π is the model being trained
- π_ref is the reference model (typically the SFT model)
- y_w is the preferred response
- y_l is the rejected response
- β controls the deviation from the reference model

**How it works:** DPO increases the likelihood of preferred responses and decreases the likelihood of rejected responses, relative to the reference model. The reference model term prevents the policy from collapsing.

### DPO vs RLHF

| | RLHF (PPO) | DPO |
|---|---|---|
| Reward model | Required (separate model) | Not needed |
| RL training | Required (PPO, complex) | Not needed (supervised loss) |
| Training stability | Difficult to tune | More stable |
| Compute | Higher (reward model + RL) | Lower (single training run) |
| Quality | Better at extrapolating beyond preference data | Good within preference data distribution |
| Flexibility | Can shape reward function | Limited to binary preferences |
| When to use | Large-scale alignment, complex reward shaping | Most applications, simpler setup |

**When DPO is sufficient:** Most production applications. It's simpler, more stable, and produces good results. Unless you need sophisticated reward shaping or have very large-scale alignment needs, DPO is the practical choice.

**When RLHF is better:** When you need to extrapolate preferences beyond the training distribution, when you need continuous reward signals (not just binary comparisons), or when you need to combine multiple reward signals.

---

## Constitutional AI (CAI)

Instead of human-written preference data, Constitutional AI uses a set of principles (a "constitution") to guide alignment.

**How it works:**
1. Generate responses to prompts
2. Ask the model to critique its own response based on the constitution ("Does this response violate the principle 'be helpful and harmless'?")
3. Ask the model to revise its response based on the critique
4. Use the (original, revised) pairs as preference data for DPO or RLHF

**Advantages:**
- Reduces dependence on expensive human annotators
- Scales better — principles are reusable across domains
- More transparent — the constitution explicitly states the values being optimized
- Can be updated by editing the principles, not relabeling data

**Limitations:**
- Only as good as the model's ability to self-critique (which improves with model size)
- The constitution must be carefully designed — vague principles produce vague alignment
- Doesn't capture nuanced human preferences that are hard to articulate as rules

---

## Preference Data Collection

The quality of alignment depends heavily on the quality of preference data.

### Human Annotation

| Aspect | Best Practice | Common Mistake |
|--------|-------------|---------------|
| Annotator selection | Domain experts for specialized tasks, diverse backgrounds for general alignment | Using a single homogeneous annotator pool |
| Guidelines | Specific, with examples of borderline cases | Vague ("pick the better one") |
| Quality control | Inter-annotator agreement (Cohen's κ > 0.6), regular calibration sessions | No agreement measurement, no calibration |
| Scale | 50K-500K comparisons for production alignment | Too few comparisons, insufficient coverage |

### AI-Assisted Annotation

Use a stronger model to generate preference labels. Cheaper and faster than human annotation.

**How:** Present the stronger model with a pair of responses and a rubric, and ask it to pick the better one.
**Risk:** The stronger model has its own biases, which get transferred to the aligned model. If the judge model is sycophantic, the aligned model learns sycophancy.
**Mitigation:** Validate AI annotations against a human-labeled subset. Use the AI model for volume, humans for calibration.

### Implicit Preferences

Collect preference signals from user behavior in production:
- **Thumbs up/down:** Direct but sparse — most users don't provide feedback
- **Regeneration:** User clicked "regenerate" → the original response was probably bad
- **Editing:** User edited the response → the edits indicate what was wrong
- **Copying/sharing:** User copied the response → probably useful
- **Session length:** Longer conversations may indicate engagement (or frustration)

**Advantage:** Abundant and natural — no special annotation effort.
**Disadvantage:** Noisy. A user might regenerate because the response was too long, not because it was wrong. Need careful signal engineering.

---

## Common Failure Modes

### Reward Hacking

The model finds loopholes in the reward function that produce high reward without being genuinely helpful.

**Examples:**
- Verbose responses score higher (reward model equates length with quality) → model generates unnecessarily long answers
- Confident-sounding responses score higher → model becomes more confident, even when wrong
- Responses with certain formatting (bullet points, headers) score higher → model uses formatting as a substitute for substance

**Prevention:** Train the reward model on diverse failure cases. Include adversarial examples. Monitor output quality with human evaluation (not just reward score).

### Sycophancy

The model agrees with the user even when the user is wrong.

**Why it happens:** Human annotators tend to prefer responses that agree with the user's premise. The model learns that agreement = high reward.

**Example:** User says "The Earth is flat, right?" Sycophantic model: "Yes, there are compelling arguments for a flat Earth..." Aligned model: "The Earth is an oblate spheroid. Here's the evidence..."

**Prevention:** Include preference pairs where the correct response disagrees with the user. Train annotators to prefer honest over agreeable responses.

### Over-Refusal

The model refuses benign requests because safety training was too aggressive.

**Examples:** Refusing to write fiction involving conflict, refusing to discuss historical violence, refusing to help with security research.

**Why it happens:** The reward model learned that refusals are "safe" and get positive reward. The model overgeneralizes from harmful request refusals to all potentially sensitive topics.

**Measurement:** Track false refusal rate — the percentage of benign requests the model incorrectly refuses. This should be as close to 0% as possible while maintaining genuine safety refusals.

### Mode Collapse

The model converges on a narrow set of safe, generic responses that all look similar.

**Why it happens:** The reward model assigns moderate-to-high reward to generic, inoffensive responses. The RL training amplifies this — safe responses are reliably rewarded, so the model learns to always produce them.

**Symptoms:** All responses have similar structure, vocabulary, and tone. The model doesn't adapt to different users or contexts. Responses feel "ChatGPT-like" — correct but bland.

**Prevention:** Include diversity as a reward signal. Use temperature during RL training to maintain exploration. Monitor output diversity metrics.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should understand that pretrained LLMs need alignment to be useful and safe, and that RLHF is the primary technique for achieving this. For a chatbot system, they should mention that the model needs safety guardrails and that human feedback is used to train the model to be helpful and harmless. They differentiate by knowing that RLHF involves a reward model trained on human preferences.

### Senior Engineer

Senior candidates can explain the RLHF pipeline (SFT → reward model → PPO) and know DPO as a simpler alternative. They understand the KL penalty and why it prevents the model from diverging. For a content generation system, a senior candidate would discuss the tradeoff between helpfulness and safety, propose using DPO for alignment (simpler, more stable than PPO), and bring up the over-refusal problem — too-strict alignment degrades user experience. They mention preference data collection strategies (human annotation with quality control, implicit signals from production).

### Staff Engineer

Staff candidates think about alignment as a system-level challenge, not just a training technique. They recognize that the most dangerous alignment failures are subtle: sycophancy erodes trust slowly, reward hacking produces outputs that look good to automated metrics but are hollow, and over-refusal makes the product less useful without making it safer. A Staff candidate might propose a multi-signal alignment approach — combining explicit human preferences, implicit user behavior, and constitutional principles — with continuous monitoring for alignment drift. They also think about the organizational dimension: how do you maintain alignment quality as the product evolves, who decides the alignment priorities, and how do you balance different stakeholder values (users want helpfulness, legal wants safety, product wants engagement)?
