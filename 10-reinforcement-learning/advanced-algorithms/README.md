# Advanced RL Algorithms

This section covers state-of-the-art RL algorithms: PPO, TRPO, and SAC.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Trust Region Methods](01_trust_region_methods.ipynb) | TRPO intuition |
| 2 | [PPO From Scratch](02_ppo_from_scratch.ipynb) | Implementing PPO |
| 3 | [PPO with Stable-Baselines](03_ppo_with_stable_baselines.ipynb) | Production PPO |
| 4 | [SAC for Continuous Control](04_sac_continuous_control.ipynb) | Soft Actor-Critic |
| 5 | [Comparing Methods](05_comparing_advanced_methods.ipynb) | PPO vs SAC |

## Key Algorithms

| Algorithm | Type | Best For |
|-----------|------|----------|
| **TRPO** | On-policy | Theoretical guarantees |
| **PPO** | On-policy | General-purpose, RLHF |
| **SAC** | Off-policy | Continuous control |

## Prerequisites

- Complete [Policy Gradient](../policy-gradient/)
- PyTorch experience

## What's Next?

- [RLHF](../rlhf/) - Using PPO to align LLMs
