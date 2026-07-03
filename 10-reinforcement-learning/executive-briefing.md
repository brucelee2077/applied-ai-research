# Reinforcement Learning — Executive Briefing

A 10-minute reference for understanding RL well enough to talk to your research team.

---

## What is Reinforcement Learning?

Reinforcement learning (RL) is a type of AI where a program learns by trial and error rather than from labeled examples. It tries actions, gets feedback (rewards), and gradually figures out the best strategy. This is how AlphaGo beat the world champion, how robots learn to walk, and how ChatGPT learned to be helpful.

---

## How RL Fits With Other AI

| Approach | How it learns | Example |
|----------|--------------|---------|
| **Supervised learning** | From labeled examples ("this photo is a cat") | Image classification, spam filters |
| **Unsupervised learning** | Finds patterns in data without labels | Customer segmentation, anomaly detection |
| **Reinforcement learning** | By trial and error with reward feedback | Game playing, LLM alignment, robotics |

**Key difference:** Supervised learning needs someone to provide the right answers upfront. RL discovers good behavior on its own through experience.

---

## The RL-to-LLM Pipeline

How modern language models are built, stage by stage:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  PRETRAINING          FINE-TUNING           RL ALIGNMENT        │
│                                                                 │
│  Read the internet    Practice on curated   Learn what humans   │
│  Learn language       high-quality Q&A      actually prefer     │
│  patterns                                                       │
│                                                                 │
│  Result: knows        Result: can follow    Result: gives       │
│  a lot, but raw       instructions          helpful, safe,      │
│                                             well-judged answers │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         Stage 1               Stage 2              Stage 3
       (expensive)           (moderate)         (your researchers)
```

**Supervised learning teaches facts. RL teaches judgment.**

Stage 3 is where your researchers operate. The techniques below are all methods for doing Stage 3 effectively.

---

## Key Methods Your Team Uses

### RLHF (Reinforcement Learning from Human Feedback)

The general approach for Stage 3:
1. The model generates multiple responses to a question
2. Humans (or a trained judge model) rank which response is better
3. Those rankings train a "reward model" that predicts human preferences
4. The LLM adjusts its behavior to score higher with that reward model

### PPO (Proximal Policy Optimization)

The standard RL algorithm used in ChatGPT's training. It adjusts the model's behavior based on reward feedback, but with a safety constraint: never change too much in one step. Small, careful updates prevent the model from breaking.

**Limitation:** Requires a separate "critic" model, which is expensive to run.

### GRPO (Group Relative Policy Optimization)

A simpler alternative to PPO, used in DeepSeek-R1. Instead of a critic, it generates a group of responses to the same question and compares them against each other. Responses that are better than the group average get reinforced; worse ones get discouraged.

**Limitation:** When every response in the group is wrong, GRPO learns nothing from that batch.

### SGPO (Stepwise Guided Policy Optimization) — Paper 1

Your researchers' improvement to GRPO. Fixes the "all-negative group" problem by using a step-by-step judge to find which parts of wrong answers were on the right track. Even when every response fails, some reasoning steps were better than others. This gives the model something to learn from, especially early in training when failures are common.

### DPO (Direct Preference Optimization)

A shortcut that skips the reward model entirely. Instead of training a judge and then using RL, DPO directly adjusts the model from human preference pairs ("Response A is better than Response B"). Simpler but less flexible.

---

## Where Your Researchers' Papers Fit

### Paper 1: SGPO (arxiv 2505.11595)

**Problem solved:** When training LLMs to reason (e.g., solve math problems), the standard method (GRPO) throws away all learning opportunity whenever every attempt in a batch fails. Early in training, this happens a lot — the model hasn't learned much yet, so it often gets everything wrong.

**What SGPO does:** Uses a step-by-step judge to evaluate individual reasoning steps within failed attempts. Even if the final answer is wrong, some steps were better than others. This creates diversity of quality that GRPO can learn from.

**Why it matters:** Accelerates learning in early and mid-training. Works across model sizes (7B to 32B parameters). Does not require the judge to know the correct answer — it only needs to compare steps.

### Paper 2: MIRIX (arxiv 2507.07957)

**Problem solved:** AI agents forget everything between sessions. They cannot build up knowledge about a user over time or learn from past interactions.

**What MIRIX does:** Gives AI agents a structured memory system with 6 types: Core (identity), Episodic (past events), Semantic (facts), Procedural (skills), Resource (tools/files), and Knowledge Vault (deep reference). A multi-agent system coordinates what to remember and how to recall it.

**Why it matters:** Achieves state-of-the-art performance on conversation memory benchmarks. Handles images and screenshots (not just text). Reduces storage by 99.9% compared to brute-force approaches while improving accuracy by 35%.

---

## Jargon Decoder

| Term | Plain English |
|------|--------------|
| Agent | The AI program that is learning or acting |
| Environment | The world the agent interacts with |
| Reward / reward signal | Feedback that says "good" or "bad" |
| Policy | The agent's strategy for choosing actions |
| Reward model | A separate AI that predicts what humans would prefer |
| Reward hacking | The model finds loopholes — scores high but gives bad answers |
| Rollout | One complete attempt (generate a full response) |
| On-policy | Learning from fresh attempts the model just made |
| Off-policy | Learning from stored past attempts |
| KL penalty / KL divergence | A safety leash preventing the model from changing too much |
| Exploration vs. exploitation | Try new things vs. stick with what already works |
| All-negative group | A batch where every attempt failed (GRPO's blind spot) |
| Step-wise judge | An AI that evaluates reasoning step-by-step, not just the final answer |
| Episodic memory | Memory of specific past events |
| Semantic memory | Memory of general facts and concepts |
| Alignment | Making an AI behave the way humans want |
| Distillation | Transferring knowledge from a large model to a smaller one |
| Ablation | Testing what happens when you remove one component |
| Benchmark | A standardized test for comparing AI systems |
| Parameters (7B, 14B, 32B) | Model size — billions of learned numbers. Bigger = more capable but more expensive |
| Online training | Model generates fresh data as it trains |
| Offline training | Model trains on pre-collected data |

---

## Questions You Can Ask Your Researchers

These signal that you understand the landscape:

1. **"How often do all-negative groups occur in early training, and how much does SGPO reduce wasted compute on those batches?"** — Shows you understand the core problem SGPO solves and care about efficiency.

2. **"Does the step-wise judge need to be retrained as the main model improves, or does it generalize?"** — Shows you're thinking about ongoing maintenance cost.

3. **"How does SGPO compare to just using a stronger reward model with regular GRPO?"** — Shows you understand alternative approaches and want to know why this path was chosen.

4. **"For MIRIX, what's the failure mode — when does the memory system retrieve the wrong thing, and how bad is that?"** — Shows you think about production reliability.

5. **"Are we seeing SGPO and MIRIX as complementary — could an agent with better memory also improve its reasoning training loop?"** — Shows you see connections across the team's work.

6. **"What's the compute cost tradeoff? Does the step-wise judge add significant training overhead?"** — Shows you care about resource efficiency (always relevant for executives).

7. **"Which benchmarks are we most confident about, and where do we think there's still room to improve?"** — Shows you understand that benchmarks don't tell the whole story.

---

## One-Sentence Summaries (for when you need to explain it to others)

**RL in general:** "AI that learns by trial and error instead of from labeled examples."

**RLHF:** "The technique that taught ChatGPT the difference between a technically correct answer and a genuinely helpful one."

**SGPO (your team's paper):** "Our researchers found a way to help AI models learn from their mistakes even when every attempt fails, by evaluating individual reasoning steps rather than just final answers."

**MIRIX (your team's paper):** "Our researchers built a memory system that lets AI agents remember users and past interactions across sessions, using six types of structured memory — similar to how human memory works."
