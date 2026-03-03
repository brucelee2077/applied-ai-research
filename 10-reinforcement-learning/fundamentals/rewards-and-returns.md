# Rewards and Returns

Imagine two jobs. Job A pays you $100 today. Job B pays you $100 one year from now. Which is better?

Most people pick Job A. Money now is worth more than money later. RL agents think the same way — and the math behind *how* they think about it is one of the most important ideas in reinforcement learning.

---

**Before you start, you need to know:**
- What an MDP is (states, actions, transitions, rewards) — covered in [markov-decision-processes.md](./markov-decision-processes.md)

---

## The Analogy: Collecting Coins in a Video Game

You are playing a game where coins appear at different times. Some coins are right in front of you. Others are far away, behind enemies and traps. You could grab the easy coins now, or risk everything for a bigger pile of coins later.

**What the analogy gets right:**
- The agent gets rewards at different moments in time
- Nearby rewards are more certain — you know you can grab them
- Far-away rewards are risky — you might not make it there
- A good player thinks about the total coins they will collect, not just the next one

**The concept in plain words:**

A **reward** is the number the environment gives the agent after each step. It can be positive (good) or negative (bad). But the agent does not care about just one reward — it cares about the **return**, which is the total reward it collects over time.

Here is the key idea: future rewards are worth less than immediate rewards. We shrink future rewards using a number called **gamma** (γ), which is between 0 and 1. A reward two steps from now gets multiplied by γ twice. A reward three steps away gets multiplied by γ three times. The further away a reward is, the smaller it becomes.

This is called **discounting**, and it does two important things:
1. It makes the agent prefer rewards sooner rather than later
2. It keeps the math from blowing up when games go on forever

**Where the analogy breaks down:** In the game, you can see where the coins are. In RL, the agent does not know what future rewards will be — it has to estimate them. That estimation is what makes RL hard.

---

**Quick check — can you answer these?**
- What is the difference between a reward and a return?
- Why do we discount future rewards?
- If gamma is 0.9 and you get a reward of 10 in three steps, how much is it worth now?

If you cannot answer one, re-read the section above. That is completely normal.

---

## What You Just Unlocked

The return — the discounted sum of future rewards — is what every RL agent is trying to maximize. Value functions estimate it. Bellman equations decompose it. Policy gradients optimize it. Every technique you will learn from here builds on this one idea. You now have the target that the entire field is aiming at.

---

Ready to go deeper? → [rewards-and-returns-interview.md](./rewards-and-returns-interview.md)
