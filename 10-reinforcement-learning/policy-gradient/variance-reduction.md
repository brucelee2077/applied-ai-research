# Variance Reduction — Making Policy Gradients Actually Work

> REINFORCE has a beautiful idea: try things, do more of what works. But there is a problem. Imagine you perform the same dance routine every night, and some nights the audience loves it (score: 300) and other nights they are lukewarm (score: 50). Same routine, wildly different feedback. How do you know which moves to keep? That randomness in the signal is called **variance**, and it makes REINFORCE painfully slow. This file is about the single most important trick to fix it — and it is surprisingly simple.

---

**Before you start, you need to know:**
- How REINFORCE works — four steps, Monte Carlo returns, the variance problem — covered in [REINFORCE Algorithm](./reinforce-algorithm.md)
- What a value function V(s) is — the expected return from a state — covered in [Policies and Value Functions](../fundamentals/policies-and-value-functions.md)

---

## The analogy: grading on a curve

Imagine you take an exam and score 75 out of 100. Is that good or bad? You have no idea — it depends on the exam.

- If the exam was easy and everyone scored 90 or above, then 75 is bad.
- If the exam was hard and the average score was 50, then 75 is excellent.

The raw score (75) is not useful by itself. What you need is the **curved score**: your score minus the class average. If the average was 60, your curved score is 75 - 60 = +15. That is clearly above average. If the average was 90, your curved score is 75 - 90 = -15. That is clearly below average.

The curved score gives you a clear signal: positive means "better than expected," negative means "worse than expected." The raw score does not.

## What the analogy gets right

In REINFORCE, the return G is like the raw exam score. It tells you the total reward from an episode, but it does not tell you whether that was good or bad for that particular state. A return of 200 might be great in a hard state and terrible in an easy state.

The **baseline** is the class average. It is a number that represents "how well do things usually go from this state?" When you subtract the baseline from the return, you get a clear signal:

- G - baseline > 0: "This episode went better than usual. Do more of what I did."
- G - baseline < 0: "This episode went worse than usual. Do less of what I did."

This is called **baseline subtraction**, and it is the core variance reduction technique in policy gradients.

## The concept in plain words

### The problem with raw returns

In vanilla REINFORCE, the agent multiplies the log probability of each action by the return G. The return is almost always positive (you usually get some reward). So every action gets reinforced — even bad ones. The bad actions just get reinforced slightly less than the good ones.

This is like giving every student an A or A+ and hoping they figure out who actually did well. It works eventually, but it is slow and noisy.

### The fix: subtract a baseline

The fix is to subtract a number — the baseline — from the return before multiplying. The policy gradient formula changes from:

"make action more likely by an amount proportional to **G**"

to:

"make action more likely by an amount proportional to **G minus the baseline**"

If G is above the baseline, the action gets reinforced (pushed up). If G is below the baseline, the action gets discouraged (pushed down). Now the agent has a clear signal about which actions are above average and which are below.

The remarkable thing is: this does not change the direction the agent learns on average. The expected gradient is the same with or without the baseline. But the variance — the noisiness of the signal — drops dramatically. The math proves this, but the intuition is simple: centering the signal around zero gives a cleaner contrast between good and bad.

### The best baseline: the value function

Any number could serve as a baseline, but some are better than others. The best baseline is **V(s)** — the expected return from the current state. This is the value function from earlier in the module.

Why is V(s) the best? Because it represents "how well things usually go from here." Subtracting V(s) from the return tells you exactly how much better or worse this specific episode was compared to what you would normally expect. This difference has a name: the **advantage**.

### The advantage function

The advantage function A(s, a) answers a simple question: "How much better is this action than the average action in this state?"

- A(s, a) > 0 means action a is better than average. Reinforce it.
- A(s, a) < 0 means action a is worse than average. Discourage it.
- The advantages across all actions in a state add up to zero — some are above average, some are below.

In practice, the agent estimates the advantage as: G (the actual return) minus V(s) (the expected return). This gives the clearest possible signal about which actions are worth repeating.

### How the agent learns V(s)

The agent needs to know V(s) to compute advantages. It learns V(s) by adding a second output to its neural network — one head predicts the action probabilities (the policy), and a second head predicts V(s) (the value). Both heads share the same hidden layers, so learning one helps the other. The value head is trained with a simple rule: minimize the squared difference between its prediction V(s) and the actual return G.

This two-headed architecture is the bridge to actor-critic methods, which are covered next.

## Where the analogy breaks down

The exam grading analogy assumes the class average is known and fixed. In RL, the baseline V(s) is itself learned and starts out wrong. Early in training, V(s) might be a poor estimate, which means the advantage signal is noisy too. The baseline gets better as training continues, so the variance reduction improves over time — but it is not perfect from the start.

---

**Quick check — can you answer these?**
- Why does subtracting a baseline not change the expected gradient?
- What is the advantage function, and what does a positive advantage mean?
- Why is V(s) a better baseline than a constant number?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## Victory lap

You just learned the trick that makes policy gradients practical. Without baseline subtraction, REINFORCE is elegant but slow. With it, learning becomes dramatically faster and more stable. Every modern policy gradient algorithm — PPO, A2C, SAC — uses advantages, and you now understand why. The advantage function is not just a mathematical convenience. It is the clear signal that tells the agent: "was this action actually good, or did it just happen to be in a good episode?" That distinction is everything.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [variance-reduction-interview.md](./variance-reduction-interview.md).
