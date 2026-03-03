# Bellman Equations

Here is a clever shortcut. Suppose you want to know how far it is from your house to the airport. You could drive the whole way and measure. Or — you could ask your neighbor how far *they* are from the airport, and just add the distance from your house to their house.

That is the Bellman equation: instead of measuring the total value from scratch, you break it into one step plus the value of where you end up. It is the most important equation in reinforcement learning.

---

**Before you start, you need to know:**
- What a policy and value function are — covered in [policies-and-value-functions.md](./policies-and-value-functions.md)
- What returns and discounting are — covered in [rewards-and-returns.md](./rewards-and-returns.md)

---

## The Analogy: Asking Your Neighbor for Directions

Imagine you want to know the best route from your house to the airport. You do not need to figure out the whole route yourself. Instead:

1. Look at your immediate options: you can go to your neighbor on the left or the neighbor on the right
2. Ask each neighbor: "How far is the airport from *your* house?"
3. Pick the neighbor who is closer, and go there

You turned one big problem (find the best route from here) into a small problem (one step) plus someone else's answer (the value of the next state).

**What the analogy gets right:**
- You break a big problem into one step plus a smaller version of the same problem
- The answer at each house depends on the answers at the neighboring houses
- Once every house knows its value, you can trace the best path by always going to the best neighbor
- This is exactly how the Bellman equation works: value at this state = reward for one step + discounted value of the next state

**The concept in plain words:**

The **Bellman equation** says: the value of being in a state equals the reward you get right now, plus the discounted value of where you end up next.

There are two versions:
- **Bellman expectation equation** — tells you the value of a state *for a given policy*. "If I follow this strategy, how good is this state?"
- **Bellman optimality equation** — tells you the value of a state *under the best possible policy*. "If I play perfectly, how good is this state?"

The optimality equation is what RL algorithms are really trying to solve. If you could solve it, you would know the best action in every state — and the game would be won.

**Where the analogy breaks down:** In the neighbor analogy, you can just ask and get the right answer. In RL, the agent does not know the true values — it has to estimate them by trying things and updating its guesses over many episodes. That process of updating guesses is what algorithms like Q-learning and TD learning are all about.

---

**Quick check — can you answer these?**
- In your own words: what does the Bellman equation break the value into?
- What is the difference between the expectation and optimality versions?
- Why can you not just solve the Bellman optimality equation directly for most real problems?

If you cannot answer one, re-read the section above. That is completely normal.

---

## What You Just Unlocked

The Bellman equation is the engine inside almost every RL algorithm. Q-learning uses it. TD learning uses it. Dynamic programming uses it. Even modern deep RL algorithms like DQN are just ways to approximately solve the Bellman equation with neural networks. You now have the core equation that powers the entire field.

---

Ready to go deeper? → [bellman-equations-interview.md](./bellman-equations-interview.md)
