# Experience Replay — Learning from a Shuffled Notebook

> Think about how you study. If you read the same chapter ten times in a row, you will know that chapter well — but you will forget everything else. A better approach: write the key ideas from every chapter on separate cards, shuffle the cards, and study a random handful each session. Every study session covers a mix of old and new material, so nothing gets forgotten. Experience replay does exactly this for a reinforcement learning agent.

---

**Before you start, you need to know:**
- How DQN trains a neural network to approximate Q-values — covered in [DQN From Scratch](./dqn-from-scratch.md)
- Why correlated training data is a problem for neural networks — they expect each training sample to be independent

---

## The analogy: the flashcard notebook

You are preparing for a test with ten chapters. You attend class every day and take notes. Here are two ways to study:

**Bad strategy: study in order.** On Monday you review only Monday's notes. On Tuesday you review only Tuesday's notes. By Friday, Monday's material is gone from memory.

**Good strategy: the shuffled notebook.** Every day you tear out your notes and drop them into a box. When it is time to study, you reach into the box and grab a random handful. Monday's notes sit next to Thursday's notes sit next to Tuesday's notes. Every study session covers a diverse mix.

This is experience replay. The agent interacts with the environment and collects experiences — each one is a tuple (state, action, reward, next state, done). Instead of training on each experience once and throwing it away, the agent drops it into a **replay buffer** (the box). When it is time to train, the agent samples a random mini-batch from the buffer.

## What the analogy gets right

The core mechanism is exact. Sequential experiences in RL are highly correlated — if the cart is at position 1.0 at time t, it is probably at 1.01 at time t+1. Training on correlated data causes the neural network to overfit to the current situation and forget how to handle older situations. Random sampling from the buffer breaks this correlation, giving the network a diverse training set that looks approximately independent and identically distributed (i.i.d.), which is what gradient descent expects.

The analogy also captures **data reuse**. In real studying, you can review the same flashcard many times. Similarly, an experience in the replay buffer can be sampled and trained on multiple times. DQN replays each experience roughly 8 times on average, extracting more learning from each interaction with the environment.

## The concept in plain words

A replay buffer is a fixed-size memory bank. Every time the agent takes a step in the environment, it stores the experience (state, action, reward, next state, done) in the buffer. When the buffer is full, the oldest experience is removed to make room for the newest one.

During training, the agent does not use the most recent experience directly. Instead, it samples a random mini-batch of 32 or 64 experiences from the buffer. These might come from different episodes, different time steps, and different parts of the state space. The diversity of this mini-batch is what makes training stable.

Without experience replay, DQN fails on most Atari games. With it, training becomes stable enough to learn from raw pixels. Mnih et al. (2015) showed this experimentally: removing replay from DQN caused performance to collapse on the majority of games.

## Where the analogy breaks down

In real studying, your understanding of Chapter 3 does not change your understanding of Chapter 7. But in a neural network, all states share the same parameters. Training on one mini-batch changes Q-values for every state, not just the ones in the batch. This is why experience replay helps but does not fully solve the problem — you also need a target network (covered in the next topic).

---

**Quick check — can you answer these?**
- Why are consecutive RL experiences correlated, and why is that a problem?
- How does random sampling from a replay buffer break this correlation?
- Why is it useful that the same experience can be sampled multiple times?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## Prioritized experience replay: study what you got wrong

Not all flashcards are equally useful. If you already know the answer to a card, reviewing it again does not help much. But if you got a card wrong, reviewing it is very valuable.

Prioritized experience replay applies this idea. Each experience has a **priority** based on its TD error — the difference between what the agent predicted and what actually happened. A large TD error means the agent was surprised, which means there is a lot to learn from that experience. Experiences with large TD errors are sampled more often.

The standard formula: the probability of sampling experience i is proportional to (|TD error_i| + ε)^α, where ε is a small constant (so no experience has zero probability) and α controls how much to prioritize (α = 0 means uniform sampling, α = 1 means full prioritization).

Prioritized replay can speed up learning by 2–5x on Atari games. The cost is extra bookkeeping: you need to update priorities after each training step and use importance sampling weights to correct for the non-uniform sampling.

---

## Victory lap

You now understand one of the two innovations that made deep reinforcement learning work. Experience replay is used in every major off-policy deep RL algorithm: DQN, Double DQN, Rainbow, SAC, TD3. It is also the reason off-policy methods are 10–100x more sample-efficient than on-policy methods like PPO — they can reuse past data, while on-policy methods must throw it away after each update.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [experience-replay-interview.md](./experience-replay-interview.md).
