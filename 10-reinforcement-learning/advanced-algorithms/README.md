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

## Coverage Map

| Topic | Depth | Files |
|-------|-------|-------|
| Trust Region Methods — TRPO and constrained optimization | [Core] | [md](./trust-region-methods.md) · [interview](./trust-region-methods-interview.md) · [notebook](./01_trust_region_methods.ipynb) · [experiments](./01_trust_region_methods_experiments.ipynb) |
| PPO from Scratch — implementing Proximal Policy Optimization | [Core] | [md](./ppo-from-scratch.md) · [interview](./ppo-from-scratch-interview.md) · [notebook](./02_ppo_from_scratch.ipynb) · [experiments](./02_ppo_from_scratch_experiments.ipynb) |
| PPO with Stable-Baselines3 — production PPO usage | [Core] | [md](./ppo-with-stable-baselines.md) · [interview](./ppo-with-stable-baselines-interview.md) · [notebook](./03_ppo_with_stable_baselines.ipynb) · [experiments](./03_ppo_with_stable_baselines_experiments.ipynb) |
| SAC Continuous Control — entropy-regularized RL | [Core] | [md](./sac-continuous-control.md) · [interview](./sac-continuous-control-interview.md) · [notebook](./04_sac_continuous_control.ipynb) · [experiments](./04_sac_continuous_control_experiments.ipynb) |
| Comparing Advanced Methods — PPO vs SAC vs TRPO | [Core] | [md](./comparing-advanced-methods.md) · [interview](./comparing-advanced-methods-interview.md) · [notebook](./05_comparing_advanced_methods.ipynb) · [experiments](./05_comparing_advanced_methods_experiments.ipynb) |

## What's Next?

- [RLHF](../rlhf/) - Using PPO to align LLMs
