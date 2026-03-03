# Classic RL Algorithms

This section covers tabular reinforcement learning methods - the foundational algorithms that all modern RL builds upon.

## What You'll Learn

- Monte Carlo methods for learning from episodes
- Temporal Difference (TD) learning
- Q-Learning (off-policy TD control)
- SARSA (on-policy TD control)
- Comparing algorithm performance

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Monte Carlo Methods](01_monte_carlo_methods.ipynb) | Learning from complete episodes |
| 2 | [Temporal Difference Learning](02_temporal_difference_learning.ipynb) | Bootstrapping and TD(0) |
| 3 | [Q-Learning](03_q_learning.ipynb) | Off-policy TD control |
| 4 | [SARSA](04_sarsa.ipynb) | On-policy TD control |
| 5 | [Comparing Algorithms](05_comparing_algorithms.ipynb) | Empirical comparisons |

## Prerequisites

- Complete the [Fundamentals](../fundamentals/) section
- Understand Bellman equations and value functions

## Key Concepts

| Algorithm | Type | Update Rule |
|-----------|------|-------------|
| Monte Carlo | Model-free | V(s) += alpha * (G - V(s)) |
| TD(0) | Model-free | V(s) += alpha * (r + gamma*V(s') - V(s)) |
| Q-Learning | Off-policy | Q(s,a) += alpha * (r + gamma*max_a' Q(s',a') - Q(s,a)) |
| SARSA | On-policy | Q(s,a) += alpha * (r + gamma*Q(s',a') - Q(s,a)) |

## Coverage Map

| Topic | Depth | Files |
|-------|-------|-------|
| Monte Carlo Methods — learning from complete episodes | [Core] | [md](./monte-carlo-methods.md) · [interview](./monte-carlo-methods-interview.md) · [notebook](./01_monte_carlo_methods.ipynb) · [experiments](./01_monte_carlo_methods_experiments.ipynb) |
| Temporal Difference Learning — bootstrapping and TD(0) | [Core] | [md](./temporal-difference-learning.md) · [interview](./temporal-difference-learning-interview.md) · [notebook](./02_temporal_difference_learning.ipynb) · [experiments](./02_temporal_difference_learning_experiments.ipynb) |
| Q-Learning — off-policy TD control | [Core] | [md](./q-learning.md) · [interview](./q-learning-interview.md) · [notebook](./03_q_learning.ipynb) · [experiments](./03_q_learning_experiments.ipynb) |
| SARSA — on-policy TD control | [Core] | [md](./sarsa.md) · [interview](./sarsa-interview.md) · [notebook](./04_sarsa.ipynb) · [experiments](./04_sarsa_experiments.ipynb) |
| Comparing Algorithms — empirical comparisons of tabular methods | [Core] | [md](./comparing-algorithms.md) · [interview](./comparing-algorithms-interview.md) · [notebook](./05_comparing_algorithms.ipynb) · [experiments](./05_comparing_algorithms_experiments.ipynb) |

## What's Next?

After completing this section, proceed to:
- [Deep RL](../deep-rl/) - Using neural networks for function approximation
- [Policy Gradient](../policy-gradient/) - Learning policies directly
