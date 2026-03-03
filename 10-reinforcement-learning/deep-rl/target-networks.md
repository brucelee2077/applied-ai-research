# Target Networks — Freezing the Goal So You Can Hit It

> Imagine learning to throw darts at a dartboard. You throw, miss slightly to the left, adjust your aim, and throw again. Each throw gets closer. Now imagine that every time you throw, someone picks up the dartboard and moves it to a new spot on the wall. You can never improve because the target keeps changing. That is what happens when you train a Q-network without a target network — and freezing a copy of the board in place for a while is what fixes it.

---

**Before you start, you need to know:**
- How DQN uses a neural network to approximate Q-values — covered in [DQN From Scratch](./dqn-from-scratch.md)
- How experience replay breaks correlation in training data — covered in [Experience Replay](./experience-replay.md)

---

## The analogy: the practice exam with frozen answers

You are studying with a friend who quizzes you. They have an answer sheet. After every question you answer, they grade you and you adjust your understanding.

**Problem: the answer sheet keeps changing.** Your friend is also learning. Every time you answer a question, they update their own understanding, which changes the answer sheet. Now you are trying to match a target that shifts with every interaction. You both drift together without converging.

**Fix: freeze the answer sheet.** Your friend makes a photocopy of their answer sheet and uses the photocopy to grade you for the next 100 questions. During those 100 questions, the answers are fixed. You can make real progress because the target is stable. After 100 questions, your friend makes a new photocopy (incorporating what they have learned) and the cycle repeats.

This is a target network. The main Q-network (your brain) updates every step. The target network (the photocopied answer sheet) stays frozen and is only replaced periodically.

## What the analogy gets right

The parallel is precise. In DQN, the training target is r + γ max Q(s', a'). This target depends on the Q-network's own weights. When the weights update, the target changes. The agent is chasing a target that moves every time it takes a step. A target network fixes this by providing a frozen copy of the Q-network for computing targets. The frozen copy only updates every N steps (hard update) or blends slowly toward the live network each step (soft update, also called Polyak averaging).

## The concept in plain words

DQN uses two copies of the same neural network:

1. The **Q-network** (also called the online network). This is the one being trained. It is updated every training step through gradient descent. The agent uses it to select actions.

2. The **target network**. This is a frozen copy. It is used only for computing the TD target: r + γ max Q_target(s', a'). Because its weights do not change during training, the target stays stable.

Periodically, the target network is updated to match the Q-network. There are two ways to do this:

- **Hard update**: every N steps (for example, every 1,000 steps), copy all the weights from the Q-network to the target network. Between copies, the target network is completely frozen. This is what the original DQN paper (Mnih et al., 2015) used.

- **Soft update** (Polyak averaging): every step, blend a tiny fraction of the Q-network's weights into the target network. The formula is θ_target ← τ θ_online + (1 − τ) θ_target, where τ is small (typically 0.005). This gives smoother updates with no sudden jumps. Modern algorithms like SAC and TD3 use this approach.

Both approaches achieve the same goal: the target changes slowly enough that the Q-network can learn to match it before it moves again.

## Where the analogy breaks down

In the practice exam analogy, the answers eventually converge to a single correct answer sheet. In DQN, there is no guarantee of convergence with function approximation. The target network reduces instability but does not eliminate it. The combination of function approximation, bootstrapping, and off-policy learning (the deadly triad) means DQN can still diverge in some cases — the target network makes this rare, not impossible.

---

**Quick check — can you answer these?**
- Why does the training target move when you update the Q-network?
- What is the difference between a hard update and a soft update for the target network?
- Why does freezing the target for a while help training?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## Why both innovations matter together

Experience replay and target networks solve different problems:

| Problem | Solution |
|---------|----------|
| Correlated training data (consecutive experiences look alike) | Experience replay: sample random mini-batches from a buffer |
| Moving training target (target changes every weight update) | Target network: freeze the target computation |

Remove either one and DQN fails. Mnih et al. showed that both are necessary: experience replay alone is not enough (targets still move), and a target network alone is not enough (data is still correlated). The two innovations complement each other.

---

## Victory lap

You now understand both pillars that make DQN stable: experience replay breaks correlation, and target networks freeze the target. Together, they tame the deadly triad enough for a neural network to learn Q-values from raw pixels. Every deep RL algorithm since DQN uses one or both of these ideas. SAC and TD3 use soft target updates. PPO avoids target networks by being on-policy (no bootstrapping from a separate network), but it uses a similar idea with clipped surrogate objectives to prevent the policy from changing too much in one step.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [target-networks-interview.md](./target-networks-interview.md).
