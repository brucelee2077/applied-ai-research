# REINFORCE — The Simplest Way to Learn What to Do

> You already know the big idea behind policy gradients: instead of rating every possible action, the agent just tries things and does more of what works. But how, exactly, does the agent adjust? What is the actual recipe? REINFORCE is the answer — and it is surprisingly simple. It is the first algorithm that turns the policy gradient idea into working code.

---

**Before you start, you need to know:**
- What a policy gradient is and why it skips value functions — covered in [Policy Gradient Intuition](./policy-gradient-intuition.md)
- What a return is (the total discounted reward from a point forward) — covered in [Rewards and Returns](../fundamentals/rewards-and-returns.md)

---

## The analogy: the talent show performer

Imagine you perform in a weekly talent show. Every week, you go on stage, do your full act, and at the end the judges give you a single score for the whole performance.

Here is how you improve:

1. **Perform the whole act.** You cannot stop in the middle to ask "how am I doing?" You have to finish.
2. **See the score.** After the act is done, you get one number — your total score.
3. **Remember what you did.** You think back through every move you made during the performance.
4. **Adjust.** If the score was high, you tell yourself: "Do more of those moves next time." If the score was low: "Do less of those."
5. **Perform again next week.** Each week, your act gets a little better.

That is REINFORCE. The entire algorithm.

## What the analogy gets right

The parallel is exact. In REINFORCE:

- "Perform the whole act" means run a complete episode — from start state to terminal state. You cannot update the policy in the middle.
- "See the score" means compute the return — the total discounted reward for the episode.
- "Remember what you did" means you saved every state, action, and reward along the way.
- "Adjust" means you change the policy network's weights so that actions with high returns become more likely, and actions with low returns become less likely.
- "Perform again next week" means you throw away the old episode and collect a brand new one using the updated policy. This is called **on-policy** learning — you always train on fresh data from your current policy.

The name REINFORCE literally means "reinforce the good behavior." Actions that led to good outcomes get reinforced — made more probable.

## The concept in plain words

### The four steps

REINFORCE has four steps, repeated for every episode:

**Step 1 — Collect an episode.** The agent follows its current policy from start to finish. At each step, it picks an action by sampling from its probability distribution (not by picking the best one — sampling is important for exploration). It saves every state, action, and reward.

**Step 2 — Compute returns.** After the episode ends, the agent works backward through the saved rewards to compute the return at each time step. The return at time step t is: "the reward I got at step t, plus the discounted reward from step t+1, plus the discounted reward from step t+2, and so on until the end." Earlier steps have higher returns because they have more future rewards ahead of them.

**Step 3 — Compute the loss.** For each step in the episode, the agent multiplies two things together: how surprising the action was (the log probability) and how good the outcome was (the return). This product tells the optimizer: "make good actions more likely, make bad actions less likely."

**Step 4 — Update the policy.** The agent runs gradient descent on the loss. The network weights change slightly, making good actions a bit more probable in similar states next time.

Then the agent throws away the entire episode and starts fresh. It never reuses old episodes. This is what makes REINFORCE a **Monte Carlo** method — it waits for the full episode to finish before learning anything.

### Why it is called Monte Carlo

Monte Carlo methods are named after the casino. The idea is: if you want to know the average outcome, just run the experiment many times and take the average. REINFORCE does exactly this. It does not estimate the return from a partial episode — it runs the whole thing and uses the actual return. This makes the gradient estimate **unbiased** (correct on average), but it comes at a cost: high variance.

### The variance problem

Here is the big weakness. Imagine two episodes where the agent takes the exact same action in the exact same state. In episode 1, things go well afterward and the return is 300. In episode 2, things go badly afterward and the return is 50. The agent updates in opposite directions for the same action! This is the **variance problem** — the learning signal is noisy because the return depends on everything that happens after the action, not just the action itself.

This noise makes REINFORCE slow. The agent needs many episodes to average out the randomness and find the true signal. It works, but it takes a while.

### A simple fix: normalize the returns

One trick that helps a lot is **return normalization**. Instead of using the raw return, the agent subtracts the average return and divides by the spread. This way, returns above average become positive ("do more of this") and returns below average become negative ("do less of this"). It does not eliminate the variance, but it makes the learning signal much more stable.

## Where the analogy breaks down

The talent show performer gets one score for the whole act and adjusts everything equally. In REINFORCE, the agent gets a different return for each time step — early actions get credit for all future rewards, while late actions only get credit for what comes after them. This is better than one score for everything, but still imperfect. The agent cannot tell which specific action was responsible for a high reward five steps later. This "credit assignment problem" is what later methods (baselines, actor-critic) try to fix.

---

**Quick check — can you answer these?**
- What are the four steps of REINFORCE?
- Why is REINFORCE called a "Monte Carlo" method?
- Why does REINFORCE have high variance, and what is one way to reduce it?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## Victory lap

You just learned the first real policy gradient algorithm — the one that turns the abstract idea of "do more of what works" into actual code. REINFORCE is the ancestor of every modern policy gradient method. PPO, the algorithm behind ChatGPT's alignment? It is a descendant of REINFORCE with smarter variance reduction and update constraints. A2C, the workhorse for many game-playing agents? It replaces REINFORCE's Monte Carlo returns with a learned value estimate. Every improvement starts from here. You now have the foundation.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [reinforce-algorithm-interview.md](./reinforce-algorithm-interview.md).
