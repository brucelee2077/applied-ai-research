# SARSA

Have you ever noticed that a careful driver and an aggressive driver can both know the fastest route — but only the careful driver adjusts their behavior because they know they sometimes make mistakes? That is the difference between SARSA and Q-learning, and it turns out to be one of the most important ideas in reinforcement learning.

---

**Before you start, you need to know:**
- What Q-learning does — it learns the value of each action in each state by assuming you will always pick the best action in the future (covered in [q-learning.md](./q-learning.md))
- What a policy is — the rule that tells the agent what to do (covered in [../fundamentals/policies-and-value-functions.md](../fundamentals/policies-and-value-functions.md))
- What epsilon-greedy means — most of the time pick the best action, but sometimes try a random one (covered in [q-learning.md](./q-learning.md))

---

## The Analogy: Two Drivers on a Mountain Road

Imagine two drivers learning a route along a mountain road with sharp cliffs on one side.

**Driver A (Q-learning)** figures out that the fastest lane is right next to the cliff edge. The travel time is shortest there. So Driver A's mental map says: "the edge lane is the best."

**Driver B (SARSA)** knows the same thing — the edge lane is technically fastest. But Driver B also knows something about themselves: "Sometimes I jerk the wheel by accident. Sometimes I get distracted. If I am in the edge lane when that happens, I go off the cliff." So Driver B's mental map says: "stay one lane away from the edge. It is a bit slower, but much safer given how I actually drive."

Both drivers have the same goal. But they learn different strategies because they account for different things.

## What the analogy gets right

- **Q-learning (Driver A) assumes perfect future behavior.** When Q-learning calculates the value of a state, it asks: "what if I take the best possible action from here?" It ignores the fact that exploration means the agent will sometimes take random actions.
- **SARSA (Driver B) accounts for actual behavior.** When SARSA calculates the value of a state, it asks: "what if I take the action I would actually take from here — including the random ones?" This makes SARSA more cautious near dangerous situations.
- **The same information, different conclusions.** Both algorithms see the same states, rewards, and transitions. But SARSA includes the risk of exploration in its value estimates, while Q-learning does not.

## The concept in plain words

SARSA is a reinforcement learning algorithm that learns how good each action is in each state — just like Q-learning. The difference is in one small detail of the update rule.

Here is what happens at every step:

1. The agent is in some **state** (S).
2. The agent picks an **action** (A) using its policy (usually epsilon-greedy).
3. The environment gives back a **reward** (R) and a new **state** (S').
4. The agent picks a new **action** (A') using the same policy.
5. The agent updates its Q-value using all five pieces: S, A, R, S', A'.

That is where the name comes from: **S-A-R-S-A** — State, Action, Reward, State, Action.

The key difference from Q-learning is in step 4. Q-learning does not pick a new action. Instead, it looks at all possible actions in the new state and uses the best one for the update. SARSA uses the action it would actually take — which might be a random exploration action.

This one difference changes everything. Near a cliff, Q-learning says "this state is great because the best action avoids the cliff." SARSA says "this state is risky because 10% of the time my epsilon-greedy policy will randomly send me off the cliff."

This is called **on-policy** learning. SARSA learns about the policy it is actually following. Q-learning is **off-policy** — it learns about the optimal policy while following a different (exploratory) policy.

## Where the analogy breaks down

In the driving analogy, both drivers are humans with the same car. In RL, the "mistakes" are not real mistakes — they are deliberate exploration. The agent chooses to take random actions because it needs to discover new things. SARSA treats this exploration as part of its reality and learns accordingly.

---

**Quick check — can you answer these?**
- What do the five letters in S-A-R-S-A stand for?
- What is the one-step difference between SARSA's update and Q-learning's update?
- Why does SARSA learn to stay away from cliffs while Q-learning does not?

If you cannot answer one, go back and re-read that part. That is completely normal.

---

## Victory lap

You just learned the idea that separates "what is theoretically best" from "what is best given how I actually behave." This distinction — on-policy vs off-policy — shows up everywhere in modern RL. When a robot learns to walk in a simulator, it uses off-policy methods because falls are free. When a robot learns in the real world, on-policy methods like SARSA (and its descendants like A2C and PPO) are preferred because real falls break real hardware. Understanding when safety matters more than optimality is one of the most practical skills in reinforcement learning.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [sarsa-interview.md](./sarsa-interview.md)
