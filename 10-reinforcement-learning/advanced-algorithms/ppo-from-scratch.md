# PPO From Scratch — The Algorithm That Trained ChatGPT

> Here is a fact that sounds too simple to be true. The algorithm that turned GPT into ChatGPT — that made language models helpful, harmless, and honest — is not some exotic new invention. It is a small, clever trick added on top of the policy gradient methods you already know. That trick is called **clipping**, and the algorithm built around it is PPO. It is the most widely used reinforcement learning algorithm in the world today.

---

**Before you start, you need to know:**
- What trust regions are and why policy collapse is dangerous — covered in [Trust Region Methods](./trust-region-methods.md)
- How actor-critic works (actor chooses actions, critic estimates value) — covered in [Actor-Critic](../policy-gradient/actor-critic.md)
- What advantage estimation does — covered in [Variance Reduction](../policy-gradient/variance-reduction.md)

---

## The analogy: the speed limit

You are learning to drive. Your driving instructor sits beside you and gives advice after each lesson: "Turn a little more smoothly. Brake a little earlier." You improve each time.

But one day the instructor says "You should go faster through turns." You take this too literally — you floor it through the next corner and crash the car.

**Without PPO:** The instructor's advice is the gradient. Sometimes the gradient says "make this action much more likely!" and you do. You change the policy dramatically. If the change was too big, the policy crashes — it becomes terrible, collects bad data, and cannot recover.

**With PPO:** There is a speed limit. No matter what the gradient says, you are not allowed to change the policy by more than 20%. "Make this action more likely? Sure — but only 20% more likely at most." You can still improve, but you cannot crash. If the advice was good, you take a moderate step. If the advice was bad, the damage is tiny.

The speed limit is the clip parameter. It caps how much the policy can change in a single update.

## What the analogy gets right

The parallel is direct. In PPO, the "speed" is the probability ratio — how much more (or less) likely an action becomes under the new policy compared to the old one. The "speed limit" is the clip range, typically [0.8, 1.2]. If the ratio tries to go above 1.2 or below 0.8, PPO cuts off the gradient. There is no incentive to push further.

The analogy also captures the key benefit: you can still learn from every lesson (every batch of data), and you can even replay the same lesson multiple times (multiple epochs). The speed limit ensures that even with aggressive reuse of data, you never veer off course.

## The concept in plain words

### The probability ratio

PPO starts with a simple question: "How much did the policy change?"

It answers this by computing a ratio. For each action the agent took, divide the new policy's probability by the old policy's probability:

ratio = new probability / old probability

If the ratio is 1.0, the policy has not changed. If the ratio is 2.0, the action is now twice as likely. If the ratio is 0.5, the action is now half as likely.

### The clipping trick

Here is PPO's core idea. Instead of letting the ratio be anything, PPO clips it to a narrow range. With the standard clip parameter of 0.2:

- The ratio cannot go above 1.2
- The ratio cannot go below 0.8

This means the policy can change by at most 20% per update. If the gradient wants to push further, PPO says "no — you have changed enough."

### The pessimistic objective

PPO computes two versions of the objective:

1. The unclipped objective: ratio times advantage
2. The clipped objective: clipped ratio times advantage

Then it takes whichever is **smaller** (more pessimistic).

Why the minimum? Think about it case by case:

- **Good action (positive advantage), ratio going up:** The unclipped version keeps rewarding you for making this action more likely. The clipped version caps the reward at ratio = 1.2. The minimum picks the cap. "You have increased this action enough."

- **Bad action (negative advantage), ratio going down:** The unclipped version keeps penalizing you for making this action less likely. The clipped version caps the penalty at ratio = 0.8. The minimum picks the cap. "You have decreased this action enough."

In both directions, PPO limits the change. Stable progress in every update.

### Multiple epochs

Vanilla policy gradient methods (REINFORCE, A2C) use each batch of data exactly once, then throw it away. This is wasteful — collecting data is expensive.

PPO can safely reuse the same batch of data multiple times (typically 10 epochs). Why? Because the clip prevents the policy from drifting too far from where it was when the data was collected. After 10 passes, the ratios are still close to 1.0, so the updates are still valid.

This makes PPO much more data-efficient than vanilla methods.

### The full PPO update

A single PPO update looks like this:

1. **Collect data.** Run the current policy in the environment for many steps. Store states, actions, log probabilities, rewards, and values.
2. **Compute advantages.** Use Generalized Advantage Estimation (GAE) to figure out which actions were better or worse than expected.
3. **Normalize advantages.** Subtract the mean and divide by the standard deviation. This centers the signal and makes training more stable.
4. **Update the network** for multiple epochs. In each epoch, shuffle the data into mini-batches and compute the clipped policy loss, the value loss, and an entropy bonus. Update the weights with standard gradient descent.
5. **Clear the buffer** and repeat from step 1.

### Actor-critic architecture

PPO uses a single neural network with two heads:

- **Actor head** — outputs action probabilities (the policy). This is what the clip controls.
- **Critic head** — outputs the state value V(s). This is used to compute advantages.

The two heads share hidden layers because understanding the state is useful for both tasks: deciding what to do and estimating how good things are.

### The entropy bonus

PPO adds a small bonus to the loss for entropy — a measure of how spread out the action probabilities are. High entropy means the agent is exploring many actions. Low entropy means it has committed to one action.

Without the entropy bonus, the policy can collapse early: it finds one action that works okay and stops trying anything else. The entropy bonus encourages the agent to keep exploring, at least a little.

## Where the analogy breaks down

A speed limit is the same everywhere — on highways, in towns, through turns. PPO's clip is also the same everywhere (0.2 is 0.2 for all actions in all states). But in practice, some state-action pairs might benefit from larger updates while others need more caution. PPO applies the same conservative bound uniformly, which is safe but sometimes slower than necessary.

---

**Quick check — can you answer these?**
- What does the probability ratio measure, and what happens when PPO clips it?
- Why does PPO take the minimum of the clipped and unclipped objectives?
- How does clipping allow PPO to reuse data for multiple epochs?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## Victory lap

You just learned the most important reinforcement learning algorithm in use today. PPO is what makes RLHF work — the process that turns a language model into a helpful assistant. When someone says "we fine-tuned the model with PPO," they mean exactly what you just learned: collect data, compute advantages, clip the ratio, update for multiple epochs. The same algorithm that balances a CartPole is the same algorithm that aligned ChatGPT. The only difference is the environment: a physics simulation versus a conversation.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [ppo-from-scratch-interview.md](./ppo-from-scratch-interview.md).
