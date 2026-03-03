# DPO and Alternatives

RLHF works — it gave us ChatGPT. But it requires training a reward model, running PPO (which can be unstable), and juggling four models in memory at the same time. What if there were a shortcut? What if you could go straight from preference data to an aligned model in one step?

---

**Before you start, you need to know:**
- What RLHF is and its three stages — covered in `what-is-rlhf.md`
- How a reward model learns from comparisons — covered in `reward-modeling.md`
- How PPO fine-tunes a language model — covered in `ppo-for-language-models.md`

---

## The analogy: the shortcut

Imagine you want to travel from your house (preference data) to a mountain top (an aligned model).

The RLHF route goes through a valley first. You hike down to the valley (train a reward model), cross a rickety bridge (run PPO), and then climb up the other side to the mountain top. The bridge is tricky — sometimes you fall off and have to start over. The whole journey takes days.

DPO found a trail that goes straight from your house to the mountain top. No valley, no bridge. Just a direct path. You arrive at the same place, but the hike is shorter, simpler, and you do not risk falling off a bridge.

**DPO (Direct Preference Optimization) skips the reward model and PPO entirely. It trains directly on preference data using a simple supervised loss.**

### What the analogy gets right

- RLHF and DPO start from the same place (preference data) and end at the same place (an aligned model)
- The RLHF route is longer and has a risky step (PPO instability)
- The DPO route is shorter and more stable
- Both routes can reach the mountain top — the destination is the same

### The concept in plain words

DPO starts with a mathematical insight: the RLHF objective (maximize reward minus KL penalty) has a closed-form solution. That means we can write down exactly what the optimal policy looks like, without running any RL at all.

Here is the key idea in four steps.

**Step 1:** RLHF says: "Find a policy that maximizes reward while staying close to the reference model." This is an optimization problem.

**Step 2:** Mathematicians can solve this optimization problem on paper. The answer is: the optimal policy should increase the probability of generating each response in proportion to how much reward it gets. More reward → higher probability.

**Step 3:** If we know the optimal policy, we can work backwards and figure out what the reward must have been. The reward is hidden inside the log-probability ratio between the policy and the reference model.

**Step 4:** When we compare a preferred response and a rejected response, the tricky parts of the math cancel out. What is left is a simple formula that says: "Make the preferred response more likely (relative to the reference) and the rejected response less likely."

The DPO loss function looks like this: for each preference pair, compute how much more likely the preferred response is under the policy versus the reference, compared to the rejected response. Then push that gap wider. That is it — standard supervised learning with a special loss function.

### Where the analogy breaks down

The shortcut trail only works if you have a good map (high-quality preference data). With RLHF, the reward model can score new responses that the model generates during training — it can explore and discover new paths. DPO is offline: it only learns from the preference data you already have. If the training data does not cover some types of responses, DPO cannot learn about them.

---

**Quick check — can you answer these?**
- What does DPO skip that RLHF needs?
- Why can DPO skip the reward model?
- What is the main limitation of DPO compared to RLHF?

If you cannot answer one, re-read that part. That is completely normal.

---

## DPO vs RLHF: when to use which

DPO is simpler, but it is not always better. Here is how to choose.

**Use DPO when:**
- You have good preference data and do not need to collect more
- You want stable, predictable training
- You have limited compute (DPO needs only two models in memory: the policy and the reference)
- Simplicity matters — fewer things can go wrong

**Use RLHF (PPO) when:**
- You need online learning — generate new responses, get feedback, and improve in a loop
- You need the reward model for other purposes (like filtering or scoring)
- You want to shape the reward in complex ways (for example, combining multiple objectives)
- You have the engineering resources to handle the complexity

In practice, many teams now start with DPO. If it works well enough, they stop there. If they need more control or iterative improvement, they switch to RLHF.

## The alternatives

DPO was not the last word. Several other methods have been developed, each with its own twist.

**RLAIF (RL from AI Feedback)** replaces human annotators with an AI model. Instead of humans comparing responses, a strong language model ranks them. This is much cheaper and faster — you can generate millions of preference pairs instead of thousands. The risk is that the AI annotator might have its own biases.

**Constitutional AI** gives the model a set of principles (a "constitution") and asks it to critique and revise its own outputs. The model reads its own response, checks it against the principles, and rewrites anything that violates them. This is self-improvement without human labels.

**ORPO (Odds Ratio Preference Optimization)** combines supervised fine-tuning and preference learning into a single step. It does not even need a reference model during training. Even simpler than DPO.

**KTO (Kahneman-Tversky Optimization)** works with unpaired feedback — just thumbs up or thumbs down on individual responses, without needing matched pairs. This makes data collection even easier.

The trend is clear: each new method is simpler, cheaper, and requires less data than the last.

---

You just learned the biggest shortcut in language model alignment. DPO takes the same mathematical objective as RLHF and solves it directly, skipping the reward model and PPO entirely. The result is a training procedure that is simpler, more stable, and often just as effective.

**Ready to go deeper?** Head to [dpo-and-alternatives-interview.md](./dpo-and-alternatives-interview.md) for the full derivation, failure modes, and interview-grade depth.
