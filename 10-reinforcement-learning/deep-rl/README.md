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

## Coverage Map

| Topic | Depth | Files |
|-------|-------|-------|
| Function Approximation — why tabular methods fail at scale | [Core] | [md](./function-approximation.md) · [interview](./function-approximation-interview.md) · [notebook](./01_function_approximation.ipynb) · [experiments](./01_function_approximation_experiments.ipynb) |
| DQN from Scratch — implementing DQN in PyTorch | [Core] | [md](./dqn-from-scratch.md) · [interview](./dqn-from-scratch-interview.md) · [notebook](./02_dqn_from_scratch.ipynb) · [experiments](./02_dqn_from_scratch_experiments.ipynb) |
| Experience Replay — breaking correlation with replay buffers | [Core] | [md](./experience-replay.md) · [interview](./experience-replay-interview.md) · [notebook](./03_experience_replay.ipynb) · [experiments](./03_experience_replay_experiments.ipynb) |
| Target Networks — stabilizing DQN training | [Core] | [md](./target-networks.md) · [interview](./target-networks-interview.md) · [notebook](./04_target_networks.ipynb) · [experiments](./04_target_networks_experiments.ipynb) |
| DQN Improvements — Double DQN, Dueling DQN, Rainbow | [Core] | [md](./dqn-improvements.md) · [interview](./dqn-improvements-interview.md) · [notebook](./05_dqn_improvements.ipynb) · [experiments](./05_dqn_improvements_experiments.ipynb) |
| Atari Games — playing games with learned agents | [Core] | [md](./atari-games.md) · [interview](./atari-games-interview.md) · [notebook](./06_atari_games.ipynb) · [experiments](./06_atari_games_experiments.ipynb) |

## What's Next?

After this section:
- [Policy Gradient](../policy-gradient/) - Learn policies directly
- [Advanced Algorithms](../advanced-algorithms/) - PPO, SAC, and more
