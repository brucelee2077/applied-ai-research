# Policy Gradient Methods

This section covers policy-based methods that directly learn a policy without computing value functions.

## What You'll Learn

- Why learn policies directly
- The REINFORCE algorithm
- Variance reduction techniques
- Actor-Critic methods
- A2C and A3C

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Policy Gradient Intuition](01_policy_gradient_intuition.ipynb) | Why optimize policies directly |
| 2 | [REINFORCE Algorithm](02_reinforce_algorithm.ipynb) | Monte Carlo policy gradient |
| 3 | [Variance Reduction](03_variance_reduction.ipynb) | Baselines and advantages |
| 4 | [Actor-Critic](04_actor_critic.ipynb) | Combining value and policy learning |
| 5 | [A2C and A3C](05_a2c_a3c.ipynb) | Advantage Actor-Critic |

## Prerequisites

- Complete [Classic Algorithms](../classic-algorithms/)
- Neural network basics from `00-neural-networks/`
- PyTorch basics

## Key Concepts

| Method | Description |
|--------|-------------|
| REINFORCE | Monte Carlo policy gradient |
| Baseline | Subtract value to reduce variance |
| Actor-Critic | Use critic to bootstrap |
| A2C | Advantage Actor-Critic (synchronous) |
| A3C | Asynchronous Advantage Actor-Critic |

## Coverage Map

| Topic | Depth | Files |
|-------|-------|-------|
| Policy Gradient Intuition — why optimize policies directly | [Core] | [md](./policy-gradient-intuition.md) · [interview](./policy-gradient-intuition-interview.md) · [notebook](./01_policy_gradient_intuition.ipynb) · [experiments](./01_policy_gradient_intuition_experiments.ipynb) |
| REINFORCE Algorithm — Monte Carlo policy gradient | [Core] | [md](./reinforce-algorithm.md) · [interview](./reinforce-algorithm-interview.md) · [notebook](./02_reinforce_algorithm.ipynb) · [experiments](./02_reinforce_algorithm_experiments.ipynb) |
| Variance Reduction — baselines and advantages | [Core] | [md](./variance-reduction.md) · [interview](./variance-reduction-interview.md) · [notebook](./03_variance_reduction.ipynb) · [experiments](./03_variance_reduction_experiments.ipynb) |
| Actor-Critic — combining value and policy learning | [Core] | [md](./actor-critic.md) · [interview](./actor-critic-interview.md) · [notebook](./04_actor_critic.ipynb) · [experiments](./04_actor_critic_experiments.ipynb) |
| A2C and A3C — advantage actor-critic methods | [Core] | [md](./a2c-a3c.md) · [interview](./a2c-a3c-interview.md) · [notebook](./05_a2c_a3c.ipynb) · [experiments](./05_a2c_a3c_experiments.ipynb) |

## What's Next?

After this section:
- [Advanced Algorithms](../advanced-algorithms/) - PPO, SAC
- [RLHF](../rlhf/) - RL for LLM alignment
