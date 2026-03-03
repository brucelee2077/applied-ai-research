# Policy Gradient Intuition — Learning What to Do, Not What Things Are Worth

> Here is a strange idea: what if an RL agent never learned Q-values at all? What if, instead of figuring out "how good is each action?" and then picking the best one, the agent just learned "what should I do?" directly? That is exactly what policy gradient methods do. They skip the middleman — no Q-table, no Q-network, no argmax. The agent learns a policy that maps states directly to actions. This one shift unlocks something DQN cannot do: handle continuous actions like steering angles, joint torques, and throttle levels.

---

**Before you start, you need to know:**
- What DQN does and why it works — covered in [DQN From Scratch](../deep-rl/dqn-from-scratch.md)
- What a probability distribution is — a way to describe how likely each outcome is

---

## The analogy: the dance instructor

Imagine you are learning to dance. There are two very different instructors you could have.

**Instructor A (the value-based approach):** Before you make any move, this instructor rates every possible move you could make. "Stepping left is worth 7 points. Spinning is worth 9 points. Jumping is worth 3 points." You calculate which move has the highest score and do that one. This works fine when there are only a few moves to choose from. But what if the instructor has to rate every possible angle your arm could be at? Every possible speed your foot could move? There are infinitely many options. Rating them all is impossible.

**Instructor B (the policy gradient approach):** This instructor says "just dance! Try things. I will tell you if it was good or bad." You try a move. The instructor says "that was great — do more of that!" So next time, you do more of that move. You try another move. The instructor says "that was terrible." So you do less of that. Over time, you naturally start doing the good moves more often and the bad moves less often. You never had to rate every possible move — you just tried things and adjusted.

## What the analogy gets right

The parallel is precise. In value-based methods (DQN), the agent learns Q-values for every action and picks the highest one. This requires comparing all possible actions, which is impossible when actions are continuous (like steering angles or joint torques). In policy gradient methods, the agent learns a policy — a function that directly outputs what to do. It tries actions, observes the results, and adjusts: actions that led to high rewards become more likely, actions that led to low rewards become less likely.

The dance instructor analogy also captures the stochastic nature of policy gradients. The agent does not always do the same thing in the same situation. It samples from a probability distribution over actions, which means it naturally explores different options. This built-in randomness is like the dancer trying slightly different moves each time — sometimes a variation turns out better than the original.

## The concept in plain words

### The key difference: learn actions, not values

DQN learns a value function: "how good is each action?" Then it picks the best one using argmax. Policy gradient methods skip the value function entirely and learn a policy: "what should I do?" The policy is a neural network that takes a state as input and outputs either:

- **For discrete actions** (like left/right): a probability for each action. The agent samples from these probabilities. If the network says "70% left, 30% right," the agent goes left 70% of the time.

- **For continuous actions** (like a steering angle): the mean and spread of a bell curve (Gaussian distribution). The agent samples a number from this bell curve. If the mean is 15 degrees and the spread is small, the agent usually steers around 15 degrees, with small random variations.

### Why DQN cannot handle continuous actions

DQN needs to compute argmax — it checks Q-values for every possible action and picks the highest one. With two actions (left/right), this is trivial. With 18 Atari actions, it is still fast. But with a continuous action like "steering angle from -30 to +30 degrees," there are infinitely many possible values. You cannot check them all. Policy gradients avoid this problem entirely because they never need argmax — they just output the action directly.

### The policy gradient update rule

The update rule has a beautiful intuition. After the agent finishes an episode, it looks at what happened:

- For each action it took: was the outcome good (high total reward) or bad (low total reward)?
- If the outcome was good: make that action more likely in that state.
- If the outcome was bad: make that action less likely in that state.

The mathematical formula uses a trick with logarithms that makes this work with gradient descent. But the intuition is simple: reinforce good behavior, weaken bad behavior. This is why the simplest policy gradient algorithm is literally called REINFORCE.

### Stochastic vs deterministic policies

DQN's policy is deterministic — in any given state, it always picks the action with the highest Q-value. Policy gradient methods typically learn stochastic policies — in any given state, they output a probability distribution and sample from it.

Stochastic policies have three advantages. First, built-in exploration: the agent tries slightly different actions each time, which helps it discover better strategies. Second, they can represent mixed strategies — in rock-paper-scissors, the best strategy is to play each option with equal probability, which a deterministic policy cannot do. Third, smooth optimization: small changes to the network weights cause small changes to the action probabilities, making gradient descent work well.

## Where the analogy breaks down

The dance instructor gives feedback after every move. In RL, the agent often has to wait until the end of an entire episode to find out how well it did. A single bad move in the middle of a great episode still gets positive feedback, because the total reward was high. This "credit assignment problem" — figuring out which specific actions were responsible for the outcome — is the main weakness of basic policy gradient methods. It causes high variance in the learning signal, making training noisy and slow. Later methods (actor-critic, PPO) address this.

---

**Quick check — can you answer these?**
- Why can DQN not handle continuous action spaces?
- What does a policy gradient agent output for discrete actions? For continuous actions?
- Why is a stochastic policy better than a deterministic one for exploration?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## Victory lap

You just understood the fundamental shift from value-based to policy-based reinforcement learning. This is not a minor variation — it is a completely different way of thinking about the problem. Instead of "evaluate everything, pick the best," policy gradients say "try things, do more of what works." This simple idea is the foundation of every modern RL algorithm used in production: PPO (the algorithm behind ChatGPT's RLHF), SAC (the go-to for robotics), and A3C (used for distributed training). When someone asks "how does RLHF work?" — the answer starts here, with policy gradients.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [policy-gradient-intuition-interview.md](./policy-gradient-intuition-interview.md).
