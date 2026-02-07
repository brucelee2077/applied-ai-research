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

## What's Next?

After this section:
- [Advanced Algorithms](../advanced-algorithms/) - PPO, SAC
- [RLHF](../rlhf/) - RL for LLM alignment
