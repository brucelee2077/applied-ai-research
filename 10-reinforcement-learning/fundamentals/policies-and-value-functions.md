# Policies and Value Functions

If you had a map that told you exactly how good every position on a chessboard is — not just whether you are winning now, but how likely you are to win from that position if you play well — you would be unstoppable. You would always move to the position with the highest value.

That map is called a **value function**, and the strategy you use to pick moves is called a **policy**. Together, they are the heart of reinforcement learning.

---

**Before you start, you need to know:**
- What an MDP is (states, actions, transitions, rewards) — covered in [markov-decision-processes.md](./markov-decision-processes.md)
- What returns and discounting are — covered in [rewards-and-returns.md](./rewards-and-returns.md)

---

## The Analogy: A GPS and a Map

You are driving to a new city. You have two tools:

**The GPS** tells you what to do at each intersection: turn left, go straight, turn right. It gives you a rule for every situation. That is a **policy** — a rule that maps each state to an action.

**The map with traffic colors** shows how good each road is. Green means fast, red means slow. You can look at any point on the map and know how long it will take to get to your destination from there. That is a **value function** — a number for each state that says how good it is to be there.

**What the analogy gets right:**
- A policy tells you what to do (the GPS directions)
- A value function tells you how good each position is (the traffic map)
- You can use the value function to build a better policy: always go toward the greener roads
- Different policies give different value functions: a bad GPS gives routes through red roads

**The concept in plain words:**

A **policy** is the agent's strategy. It takes a state and tells the agent what action to pick. Some policies are simple rules ("always go right"). Some are complex neural networks. The agent's goal is to find the best policy — the one that collects the most total reward.

A **value function** tells the agent how much reward it can expect from a given state (or state-action pair) if it follows a specific policy. There are two kinds:
- **V(s)** — the state value: "how good is it to be in state s?"
- **Q(s, a)** — the action value: "how good is it to take action a in state s?"

The trick is that policies and value functions help each other. If you know the value function, you can build a better policy (just pick the action that leads to the highest value). If you have a policy, you can calculate its value function (just see how much reward that policy collects from each state).

**Where the analogy breaks down:** A GPS knows the full map ahead of time. An RL agent does not — it has to learn the value function and policy by exploring, making mistakes, and updating its estimates.

---

**Quick check — can you answer these?**
- What is the difference between a policy and a value function?
- What do V(s) and Q(s, a) measure?
- Why does knowing Q(s, a) immediately give you a policy?

If you cannot answer one, re-read the section above. That is completely normal.

---

## What You Just Unlocked

You now understand the two things every RL algorithm is trying to learn: a policy (what to do) and a value function (what to expect). Some algorithms learn the value function first and derive the policy from it (value-based methods like Q-learning). Others learn the policy directly (policy gradient methods). Some do both at once (actor-critic methods). Every approach you will study is a different strategy for learning these two things.

---

Ready to go deeper? → [policies-and-value-functions-interview.md](./policies-and-value-functions-interview.md)
