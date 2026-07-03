# GRPO — Group Relative Policy Optimization

Imagine you are a teacher grading essays. Instead of giving each essay a score out of 100, you read a batch of essays together and rank them: "This one is the best, this one is average, this one needs work." You do not need a rubric — you just compare them to each other within the group.

That is what GRPO does. Instead of training a separate "reward judge" (like PPO does), GRPO generates multiple answers to the same question and compares them to each other. The best answers in the group get reinforced. The worst get discouraged. No separate critic network needed.

---

**Before you start, you need to know:**
- What a policy is — a strategy for choosing actions ([policies-and-value-functions.md](../fundamentals/policies-and-value-functions.md))
- What PPO is — the clipping mechanism and why it stabilizes training ([ppo-from-scratch.md](../advanced-algorithms/ppo-from-scratch.md))
- What RLHF is — training language models from human preferences ([what-is-rlhf.md](./what-is-rlhf.md))

---

## The Analogy: A Cooking Competition Within Each Round

Imagine a cooking show where, in each round, one chef makes 8 different dishes from the same recipe. A judge tastes all 8 and ranks them. The chef learns from the differences: "Dish 3 was the best — use more of that technique. Dish 7 was the worst — avoid that."

The chef does not need a separate food critic to rate each dish on an absolute scale. The comparison within the group is enough to improve.

**What the analogy gets right:**
- Multiple outputs from the same input (same prompt, multiple responses)
- Learning by comparing within the group, not against an absolute score
- The best-in-group answer gets reinforced, the worst gets discouraged
- No separate "critic" or "reward model" is needed for the advantage calculation

**The concept in plain words:**

**GRPO** (Group Relative Policy Optimization) is a reinforcement learning algorithm for training language models. For each prompt, it generates a group of G responses (typically 8-64). It scores each response using a reward function (or rule-based check). Then it computes an advantage for each response by comparing its reward to the group average: responses above average get positive advantages (reinforced), and responses below average get negative advantages (discouraged).

The key innovation: GRPO eliminates the critic (value function) network entirely. In PPO-based RLHF, you need a separate value network to estimate advantages — this doubles the memory cost and introduces its own training instabilities. GRPO replaces the critic with a simple group-level comparison.

**Where the analogy breaks down:** In the cooking show, the judge is separate from the chef. In GRPO, the reward signal can come from any source — a trained reward model, a rule-based checker, or even a simple heuristic like "did the code compile?"

---

**Quick check — can you answer these?**
- How does GRPO compute advantages without a critic network?
- Why does generating multiple responses per prompt help with training?
- What is the difference between GRPO and standard PPO for RLHF?

If you cannot answer one, re-read the section above. That is completely normal.

---

## What You Just Unlocked

GRPO is the algorithm behind DeepSeek-R1, one of the most capable reasoning models. It is gaining popularity because it is simpler than PPO (no critic network), uses less memory, and works especially well for tasks where rewards are easy to verify (math, code, factual questions). You now understand the core idea that powers a major frontier model.

---

Ready to go deeper? → [grpo-interview.md](./grpo-interview.md)
