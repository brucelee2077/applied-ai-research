# A2C and A3C — Training Faster with Parallel Worlds

> Actor-critic learns after every step, which is faster than REINFORCE. But there is a way to make it even faster: instead of running one environment, run many environments at the same time. Four environments means four times as much experience in the same amount of time. This simple idea — parallel environments — is what turns actor-critic from a research algorithm into a practical training method. A2C and A3C are the two ways to do it.

---

**Before you start, you need to know:**
- How actor-critic works — the actor chooses actions, the critic judges them using TD error — covered in [Actor-Critic](./actor-critic.md)
- What variance is and why reducing it helps learning — covered in [Variance Reduction](./variance-reduction.md)

---

## The analogy: learning to cook from parallel kitchens

Imagine you are learning to cook a new dish. Every attempt teaches you something, but each attempt takes 30 minutes. That is slow.

Now imagine you have four kitchens. In each kitchen, a copy of you tries a slightly different approach at the same time. Kitchen 1 tries more salt. Kitchen 2 tries higher heat. Kitchen 3 tries a different oil. Kitchen 4 follows the original recipe exactly. After 30 minutes, all four results come in. You learn four lessons in the time it would have taken to learn one.

But how do you combine the lessons? There are two approaches:

**The synchronous approach (A2C):** All four kitchens finish at the same time. You look at all four results, decide what works best, and give every kitchen the same updated recipe. Everyone starts the next round with the same instructions.

**The asynchronous approach (A3C):** Each kitchen finishes whenever it finishes. Kitchen 2 might finish first and immediately get an updated recipe, while Kitchen 4 is still cooking. The updates happen continuously, but some kitchens might be working with slightly old recipes.

## What the analogy gets right

In A2C and A3C, "kitchens" are parallel environments. The agent runs the same policy in multiple copies of the environment at the same time. Each environment produces its own stream of states, actions, and rewards. The agent learns from all of them.

The benefits are exactly like the cooking analogy:

- **Speed.** Four environments produce four times as much experience per second.
- **Diversity.** Each environment follows a different trajectory, so the agent sees a wider range of situations. This makes the gradient estimates more stable.
- **Exploration.** Different environments naturally end up in different states, which helps the agent discover things it might miss with a single environment.

The synchronous (A2C) vs asynchronous (A3C) distinction is also accurate. A2C waits for all environments to finish their steps, collects all the data, and does one big update. A3C lets each environment update the shared model as soon as it has data, without waiting for the others.

## The concept in plain words

### A2C: everyone waits, then one big update

A2C stands for **Advantage Actor-Critic**. The "advantage" refers to the advantage function from variance reduction — the signal that tells the actor whether an action was better or worse than average.

Here is how A2C works:

1. **Run all environments forward.** All N environments take a few steps (say, 5 steps each) using the current policy. This produces N separate streams of experience.
2. **Compute advantages.** For each environment, compute the advantage at each time step — how much better or worse things went compared to what the critic expected.
3. **Batch the data.** Combine all the experience from all environments into one big batch.
4. **Update once.** Run one gradient descent step on the combined batch. Both the actor and critic update.
5. **Repeat.** All environments now use the freshly updated policy for the next round.

Because everyone waits and updates together, every environment always uses the most up-to-date policy. This is clean and simple.

### A3C: update whenever ready

A3C stands for **Asynchronous Advantage Actor-Critic**. The key word is "asynchronous" — each environment updates the model on its own schedule.

In A3C, each environment runs on its own thread with its own copy of the model. When a thread finishes collecting experience, it computes gradients and immediately updates the shared global model. Then it copies the latest global model back to its local copy and starts collecting again.

The problem is that some threads might be using slightly outdated parameters — they started collecting before the latest update happened. These **stale gradients** can slow learning slightly. In practice, A3C still works well, but it is messier than A2C.

### Why A2C won

A3C was designed in 2016 when researchers trained on CPUs. Asynchronous updates made sense because CPU threads are independent. But today, most training happens on GPUs, which are much faster at processing large batches of data all at once. A2C's synchronous batching fits GPUs perfectly. A3C's asynchronous design does not.

The result: A2C is simpler, just as effective, and better suited to modern hardware. Most practitioners use A2C (or its descendant, PPO).

### GAE: tuning the bias-variance knob

A2C uses a technique called **Generalized Advantage Estimation (GAE)** to compute advantages. GAE has a parameter called lambda that controls a trade-off:

- Lambda near 0: The advantage only looks one step ahead. Low variance, but the critic's estimate adds some bias.
- Lambda near 1: The advantage looks many steps ahead, closer to the full return. High variance, but less bias.
- Lambda around 0.95: A sweet spot that works well in practice.

GAE is like a weighted average of different lookahead distances. Close steps get high weight, distant steps get low weight. This gives a cleaner advantage signal than either extreme.

### The entropy bonus

A2C adds one more trick: an **entropy bonus**. This is a small reward for keeping the policy spread out — not putting all probability on one action. Without it, the policy can collapse to always choosing the same action, which stops exploration. The entropy bonus keeps the agent curious enough to try different things, especially early in training.

## Where the analogy breaks down

In the cooking analogy, each kitchen is truly independent — they can experiment with completely different recipes. In A2C, all environments share the same policy. They explore different trajectories because of the randomness in action sampling and environment dynamics, but they are all following the same set of instructions. The diversity comes from randomness, not from intentional variation.

---

**Quick check — can you answer these?**
- What is the difference between A2C and A3C?
- Why is A2C preferred over A3C on modern hardware?
- What does GAE lambda control?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## Victory lap

You just completed the entire policy gradient section. Starting from the basic idea of "try things, do more of what works," you built up through REINFORCE, variance reduction, actor-critic, and now parallel training. A2C is the method that takes all these ideas and makes them practical. It is the direct ancestor of PPO — the most widely used RL algorithm today, the one behind ChatGPT's alignment, game-playing agents, and robotic control systems. Everything in the next section (advanced algorithms) builds directly on what you just learned.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [a2c-a3c-interview.md](./a2c-a3c-interview.md).
