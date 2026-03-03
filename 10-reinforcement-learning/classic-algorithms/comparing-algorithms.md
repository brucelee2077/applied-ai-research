# Comparing Classic RL Algorithms

You now know four different ways to teach an agent: Monte Carlo, TD learning, Q-learning, and SARSA. They all solve the same problem — learning to make good decisions — but they do it in different ways. So how do you choose? That is what this topic is about, and the answer matters more than you might think.

---

**Before you start, you need to know:**
- What Monte Carlo methods do — learn by playing full episodes and looking back at what happened (covered in [monte-carlo-methods.md](./monte-carlo-methods.md))
- What TD learning does — learn after every single step by guessing what comes next (covered in [temporal-difference-learning.md](./temporal-difference-learning.md))
- The difference between Q-learning and SARSA — Q-learning assumes perfect future actions, SARSA uses actual future actions (covered in [q-learning.md](./q-learning.md) and [sarsa.md](./sarsa.md))

---

## The Analogy: Tools in a Toolbox

Imagine you have a toolbox with four tools:

- **A ruler** (Monte Carlo). It measures things exactly, but you have to measure the whole length before you get any answer. If the thing you are measuring is very long, you wait a long time.
- **A laser distance meter** (TD learning). It gives you a quick answer right away by bouncing a beam off the nearest surface. It is fast, but the reading might be slightly off because it is estimating, not measuring the whole thing.
- **A GPS navigator** (Q-learning). It always shows you the fastest route to your destination, even if you are currently driving on a different road. It assumes you will follow its directions perfectly.
- **A careful friend in the passenger seat** (SARSA). They give you directions that account for the fact that you sometimes miss turns or take wrong exits. Their route might be a bit longer, but it works better for how you actually drive.

## What the analogy gets right

- **Monte Carlo is exact but slow.** It uses real results (no guessing), but you have to wait until the episode ends. For long episodes, this means waiting a long time to learn anything.
- **TD learning is fast but biased.** It updates after every step using an estimate of the future. This means faster learning, but the estimates might be wrong early on.
- **Q-learning finds the best route.** It learns the optimal policy regardless of what the agent is currently doing. But it ignores the risk of exploration mistakes.
- **SARSA finds a practical route.** It learns a policy that accounts for the agent's actual behavior, including random exploration. This makes it safer near dangerous states.

## The concept in plain words

Here are the four key choices you face when picking an algorithm:

**1. When do you learn — after the episode or after each step?**
Monte Carlo waits until the episode ends. TD learning, Q-learning, and SARSA all learn after every step. If your episodes are long (or never end), Monte Carlo is a bad choice.

**2. Do you use real results or estimates?**
Monte Carlo uses the actual total reward it collected. This means no guessing errors, but more randomness from episode to episode. TD-based methods use estimates (bootstrapping), which reduces randomness but introduces a small error from the estimate.

**3. Do you learn about the best possible behavior or your actual behavior?**
Q-learning learns about the best possible policy (off-policy). SARSA learns about the policy you are actually following (on-policy). If your current policy includes random exploration, SARSA's value estimates include the cost of that randomness.

**4. Does safety matter?**
If mistakes during learning are expensive (a robot falling, a trade losing money), SARSA is safer because it avoids states where exploration could cause harm. Q-learning will happily walk along a cliff edge because it assumes you will never accidentally step off.

## Where the analogy breaks down

Real tools are either exact or not — there is no middle ground. In RL, you can control the trade-off. You can adjust the learning rate, the exploration rate, and the discount factor to balance speed, accuracy, and safety. The algorithms are not rigid tools — they are flexible methods that you tune.

---

**Quick check — can you answer these?**
- Why can Monte Carlo not be used for tasks that never end?
- What is the trade-off between Monte Carlo (unbiased) and TD learning (lower variance)?
- In one sentence, why is SARSA safer than Q-learning near dangerous states?

If you cannot answer one, go back and re-read that part. That is completely normal.

---

## Victory lap

You now have a complete picture of the four classic RL algorithms and when to use each one. These are not just historical curiosities — they are the building blocks of everything that came after. Deep Q-networks (DQN) are Q-learning with a neural network. PPO is an actor-critic method that evolved from SARSA-style on-policy thinking. AlphaGo uses Monte Carlo tree search. Every modern RL system traces back to one of these four ideas. You now understand all of them.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [comparing-algorithms-interview.md](./comparing-algorithms-interview.md)
