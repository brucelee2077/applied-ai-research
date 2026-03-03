# Markov Decision Processes

Here is a puzzle: how do you write down a game so that a computer can play it? Not a specific game — any game. Any situation where someone makes choices, things happen, and there are consequences. Is there a single way to describe all of that?

Yes. It is called a **Markov Decision Process**, and it is the language that every RL algorithm speaks.

---

**Before you start, you need to know:**
- What reinforcement learning is (agent, environment, reward) — covered in [what-is-reinforcement-learning.md](./what-is-reinforcement-learning.md)

---

## The Analogy: A Board Game

Think about a simple board game like Snakes and Ladders.

At any moment, you are on a specific square. That is your **state**. You roll the dice and move — that is your **action**. Where you land depends partly on what you did and partly on luck — that is the **transition**. And some squares are better than others (ladders are good, snakes are bad) — that is the **reward**.

**What the analogy gets right:**
- The game has a clear set of possible positions (states)
- You make choices (actions) that affect where you end up
- The outcome can involve some randomness (the dice)
- The reward depends on where you land, not on what happened ten turns ago — what matters is where you are *now*

**The concept in plain words:**

A Markov Decision Process (MDP) is a way to describe any decision-making problem using four things:
1. **States (S)** — all the situations the agent can be in
2. **Actions (A)** — all the things the agent can do
3. **Transitions** — the rules for what happens when the agent takes an action (which new state does it go to?)
4. **Rewards (R)** — the feedback the agent gets after each transition

The special thing about an MDP is the **Markov property**: the future depends only on where you are right now, not on how you got there. Your current state contains all the information you need to make a good decision.

**Where the analogy breaks down:** In Snakes and Ladders, you have no real choices — you just roll the dice. In a real MDP, the agent chooses its actions on purpose. That is the whole point — the agent is trying to find the best action for each state.

---

**Quick check — can you answer these?**
- What are the four parts of an MDP?
- What does the Markov property mean in your own words?
- Why is the Markov property useful for an RL agent?

If you cannot answer one, re-read the section above. That is completely normal.

---

## What You Just Unlocked

Every RL algorithm you will ever learn — Q-learning, policy gradients, PPO, even RLHF — works inside an MDP. The MDP is the shared language. Once you can describe a problem as an MDP, you can apply any RL algorithm to it. You now have the framework that holds everything together.

---

Ready to go deeper? → [markov-decision-processes-interview.md](./markov-decision-processes-interview.md)
