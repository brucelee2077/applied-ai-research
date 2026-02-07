# Deep Reinforcement Learning

This section covers combining deep neural networks with reinforcement learning to handle complex, high-dimensional problems.

## What You'll Learn

- Why we need function approximation
- Deep Q-Networks (DQN) from scratch
- Experience replay and target networks
- DQN improvements (Double, Dueling DQN)
- Playing Atari games

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Function Approximation](01_function_approximation.ipynb) | Why tabular methods fail at scale |
| 2 | [DQN From Scratch](02_dqn_from_scratch.ipynb) | Implementing DQN in PyTorch |
| 3 | [Experience Replay](03_experience_replay.ipynb) | Breaking correlation with replay buffers |
| 4 | [Target Networks](04_target_networks.ipynb) | Stabilizing DQN training |
| 5 | [DQN Improvements](05_dqn_improvements.ipynb) | Double DQN, Dueling DQN |
| 6 | [Atari Games](06_atari_games.ipynb) | Playing games with Stable-Baselines3 |

## Prerequisites

- Complete [Fundamentals](../fundamentals/) and [Classic Algorithms](../classic-algorithms/)
- Neural network basics from `00-neural-networks/`
- PyTorch basics

## Key Concepts

| Concept | Purpose |
|---------|---------|
| Function Approximation | Handle large/continuous state spaces |
| Experience Replay | Break correlation, improve sample efficiency |
| Target Network | Stabilize training by fixing Q-target |
| Double DQN | Reduce overestimation bias |
| Dueling DQN | Separate state value and advantage |

## What's Next?

After this section:
- [Policy Gradient](../policy-gradient/) - Learn policies directly
- [Advanced Algorithms](../advanced-algorithms/) - PPO, SAC, and more
