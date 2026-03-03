# Actor-Critic — The Best of Both Worlds

> Here is the problem with REINFORCE: the agent performs an entire episode, then looks back and asks "was that good?" But by then, it has taken dozens or hundreds of actions, and it cannot tell which ones mattered. What if, instead of waiting until the end, someone watched every single step and said "that was better than I expected" or "that was worse than I expected" — right there, in the moment? That is exactly what actor-critic methods do. They combine two ideas — a policy that chooses actions and a value function that judges them — into one system that learns faster than either could alone.

---

**Before you start, you need to know:**
- How REINFORCE works and why it has high variance — covered in [REINFORCE Algorithm](./reinforce-algorithm.md)
- What a baseline is and why subtracting it reduces variance — covered in [Variance Reduction](./variance-reduction.md)

---

## The analogy: the actor and the coach

Imagine you are learning to act in a play. There are two approaches to getting feedback.

**Approach A (REINFORCE):** You perform the entire play — every scene, start to finish. After the curtain falls, the audience gives you a score. "7 out of 10." You try to figure out which scenes were good and which were bad, but all you have is one number for the whole thing. Was Scene 3 great and Scene 7 terrible? Or was it the other way around? Hard to tell.

**Approach B (Actor-Critic):** A coach sits in the front row and watches every scene. After Scene 1, the coach whispers: "That was better than I expected — keep doing that." After Scene 2: "That was about what I expected." After Scene 3: "That was worse than I expected — try something different." You adjust your performance scene by scene, in real time.

The actor is you — the one making decisions about how to perform. The critic is the coach — the one judging whether each scene went better or worse than expected. Together, they learn faster than either could alone.

## What the analogy gets right

In an actor-critic system, there are literally two components:

- **The actor** is the policy network. It takes a state as input and outputs action probabilities. It decides what to do. This is the same policy network from REINFORCE.
- **The critic** is the value network. It takes a state as input and outputs a single number: V(s), the expected return from that state. It judges how good things are.

The critic's feedback is called the **TD error** (temporal difference error). It answers a simple question: "Did things go better or worse than I expected?"

- TD error > 0: "Better than expected!" — the actor should do more of this.
- TD error < 0: "Worse than expected!" — the actor should do less of this.
- TD error near 0: "About what I expected." — no strong signal either way.

The key difference from REINFORCE is timing. REINFORCE waits until the entire episode is over to learn. Actor-critic learns after every single step. This is called **online learning**, and it is much faster.

## The concept in plain words

### Two networks, one goal

The actor and critic are usually two heads on the same neural network. They share the same hidden layers (which extract useful features from the state), but they have different output layers:

- The actor head outputs a probability for each action.
- The critic head outputs a single number — V(s).

Sharing hidden layers helps because both the actor and the critic need to understand the state. A feature that is useful for choosing actions is often useful for estimating value too.

### How the critic gives feedback

After the actor takes an action and sees the result, the critic computes the TD error:

"What I actually got (the reward, plus what I expect from the next state) minus what I expected from this state."

If the actual outcome is better than what the critic predicted, the TD error is positive — the action was better than average. If the outcome is worse than predicted, the TD error is negative — the action was worse than average.

This TD error serves the same role as the advantage from variance reduction, but it is computed instantly — no need to wait for the episode to end.

### How both learn

The actor and the critic learn at the same time, but they learn different things:

- **The actor** learns which actions are good. It uses the TD error as a signal: make actions with positive TD error more likely, make actions with negative TD error less likely.
- **The critic** learns to predict returns accurately. It updates its value estimate to be closer to what actually happened: the reward received plus the estimated value of the next state.

They help each other. A better critic gives clearer feedback to the actor. A better actor generates better episodes for the critic to learn from. This mutual improvement is what makes actor-critic methods powerful.

### Why it beats REINFORCE

Actor-critic wins on two fronts:

- **More updates.** REINFORCE updates once per episode. If an episode has 200 steps, actor-critic makes 200 updates in the same time. More updates means faster learning.
- **Lower variance.** REINFORCE uses the full return, which includes all future randomness. The TD error only looks one step ahead and uses the critic's estimate for everything after that. This is noisier in a different way — the critic's estimate might be wrong — but the overall variance is much lower.

The trade-off is a small amount of **bias**. The critic's estimate of V(s) is not perfect, so the TD error is not exactly right. But in practice, the variance reduction more than makes up for this small bias.

## Where the analogy breaks down

A real acting coach has years of experience and gives reliable feedback from the start. The critic in actor-critic starts with random weights — its early feedback is nearly useless. The actor has to learn from a critic that is also still learning. This bootstrapping problem means that early training can be unstable. As both networks improve, the feedback gets better and learning accelerates — but the first few hundred episodes can be rough.

---

**Quick check — can you answer these?**
- What does the actor do? What does the critic do?
- What is the TD error, and what does a positive TD error mean?
- Why does actor-critic have lower variance than REINFORCE?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## Victory lap

You just learned the architecture behind nearly every modern RL algorithm. PPO? It is an actor-critic method with a clipped objective. SAC? Actor-critic with entropy regularization. A3C? Actor-critic with parallel environments. RLHF for language models? The policy model is the actor, and the reward model acts like a critic. Every time you read about a new RL algorithm, look for the actor and the critic — they are almost always there. You now understand the pattern.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [actor-critic-interview.md](./actor-critic-interview.md).
