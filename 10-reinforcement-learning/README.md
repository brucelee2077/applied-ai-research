# 10 - Reinforcement Learning

Welcome to the comprehensive Reinforcement Learning section! This section covers RL from fundamentals through advanced algorithms and RLHF for LLM alignment.

## What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. Unlike supervised learning (which learns from labeled examples), RL learns from **trial and error** - receiving rewards for good actions and penalties for bad ones.

**Key idea:** An agent takes actions, observes outcomes, and learns a policy to maximize cumulative rewards.

## Learning Path

```
fundamentals/ (Start here!)
       ↓
classic-algorithms/
       ↓
   ┌───┴───┐
   ↓       ↓
deep-rl/  policy-gradient/  (can be done in parallel)
   ↓       ↓
   └───┬───┘
       ↓
advanced-algorithms/
       ↓
    rlhf/
       ↓
  applications/
```

## Section Overview

### 1. Fundamentals (Start Here)
Core RL concepts that everything else builds on.
- What is Reinforcement Learning
- Markov Decision Processes (MDPs)
- Rewards and Returns
- Policies and Value Functions
- Bellman Equations

### 2. Classic Algorithms
Tabular methods - the foundation of RL algorithms.
- Monte Carlo Methods
- Temporal Difference Learning
- Q-Learning
- SARSA
- Algorithm Comparison

### 3. Deep RL
Combining neural networks with RL for complex problems.
- Function Approximation
- Deep Q-Networks (DQN)
- Experience Replay
- Target Networks
- DQN Improvements (Double, Dueling)
- Playing Atari Games

### 4. Policy Gradient
Learning policies directly instead of value functions.
- Policy Gradient Intuition
- REINFORCE Algorithm
- Variance Reduction
- Actor-Critic Methods
- A2C and A3C

### 5. Advanced Algorithms
State-of-the-art RL methods.
- Trust Region Methods (TRPO)
- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)
- Algorithm Comparisons

### 6. RLHF (Reinforcement Learning from Human Feedback)
Using RL to align language models with human preferences.
- What is RLHF
- Reward Modeling
- PPO for Language Models
- DPO and Alternatives
- TRL Library Tutorial
- End-to-End RLHF Pipeline

### 7. Applications
Real-world RL applications and case studies.
- Game Playing
- Robotics Simulation
- Recommendation Systems
- LLM Alignment Case Study
- Multi-Agent Systems

## Prerequisites

- **Required:** Completion of `00-neural-networks/fundamentals/` (for deep RL sections)
- **Recommended:** `02-fine-tuning/` (for RLHF section)
- **Python skills:** NumPy, PyTorch basics

## Installation

```bash
# Core RL libraries
pip install gymnasium
pip install stable-baselines3

# For RLHF section
pip install trl

# PyTorch, transformers, peft should already be installed from other sections
```

## Implementation Philosophy

This section follows the same philosophy as the rest of the repository:

1. **From-scratch first:** Understand algorithms by implementing them in NumPy/PyTorch
2. **Then frameworks:** Use production-ready libraries (Stable-Baselines3, TRL) for real applications
3. **Visual learning:** Matplotlib visualizations to build intuition
4. **Real examples:** Apply algorithms to actual games, robotics simulations, and LLMs

## Key Libraries Used

| Library | Purpose |
|---------|---------|
| `gymnasium` | RL environments (successor to OpenAI Gym) |
| `stable-baselines3` | Production-ready RL implementations |
| `trl` | Transformer Reinforcement Learning for LLMs |
| `numpy` | From-scratch implementations |
| `torch` | Deep RL implementations |

## Quick Start

New to RL? Start here:
1. [What is Reinforcement Learning?](fundamentals/01_what_is_reinforcement_learning.ipynb)
2. [Markov Decision Processes](fundamentals/02_markov_decision_processes.ipynb)
3. [Q-Learning](classic-algorithms/03_q_learning.ipynb)

Coming from other sections? Jump to what interests you:
- From `08-ai-agents`: Go to RLHF section to understand how agents learn from human feedback
- From `02-fine-tuning`: Go to RLHF section for reward-based fine-tuning

## Coverage Map

### fundamentals/

| Topic | Depth | Files |
|-------|-------|-------|
| What is Reinforcement Learning — core RL concepts | [Core] | [md](./fundamentals/what-is-reinforcement-learning.md) · [interview](./fundamentals/what-is-reinforcement-learning-interview.md) · [notebook](./fundamentals/01_what_is_reinforcement_learning.ipynb) · [experiments](./fundamentals/01_what_is_reinforcement_learning_experiments.ipynb) |
| Markov Decision Processes — formal framework for RL | [Core] | [md](./fundamentals/markov-decision-processes.md) · [interview](./fundamentals/markov-decision-processes-interview.md) · [notebook](./fundamentals/02_markov_decision_processes.ipynb) · [experiments](./fundamentals/02_markov_decision_processes_experiments.ipynb) |
| Rewards and Returns — cumulative reward formulations | [Core] | [md](./fundamentals/rewards-and-returns.md) · [interview](./fundamentals/rewards-and-returns-interview.md) · [notebook](./fundamentals/03_rewards_and_returns.ipynb) · [experiments](./fundamentals/03_rewards_and_returns_experiments.ipynb) |
| Policies and Value Functions — decision-making structures | [Core] | [md](./fundamentals/policies-and-value-functions.md) · [interview](./fundamentals/policies-and-value-functions-interview.md) · [notebook](./fundamentals/04_policies_and_value_functions.ipynb) · [experiments](./fundamentals/04_policies_and_value_functions_experiments.ipynb) |
| Bellman Equations — recursive value decomposition | [Core] | [md](./fundamentals/bellman-equations.md) · [interview](./fundamentals/bellman-equations-interview.md) · [notebook](./fundamentals/05_bellman_equations.ipynb) · [experiments](./fundamentals/05_bellman_equations_experiments.ipynb) |

### classic-algorithms/

| Topic | Depth | Files |
|-------|-------|-------|
| Monte Carlo Methods — learning from complete episodes | [Core] | [md](./classic-algorithms/monte-carlo-methods.md) · [interview](./classic-algorithms/monte-carlo-methods-interview.md) · [notebook](./classic-algorithms/01_monte_carlo_methods.ipynb) · [experiments](./classic-algorithms/01_monte_carlo_methods_experiments.ipynb) |
| Temporal Difference Learning — bootstrapped value updates | [Core] | [md](./classic-algorithms/temporal-difference-learning.md) · [interview](./classic-algorithms/temporal-difference-learning-interview.md) · [notebook](./classic-algorithms/02_temporal_difference_learning.ipynb) · [experiments](./classic-algorithms/02_temporal_difference_learning_experiments.ipynb) |
| Q-Learning — off-policy value learning | [Core] | [md](./classic-algorithms/q-learning.md) · [interview](./classic-algorithms/q-learning-interview.md) · [notebook](./classic-algorithms/03_q_learning.ipynb) · [experiments](./classic-algorithms/03_q_learning_experiments.ipynb) |
| SARSA — on-policy value learning | [Core] | [md](./classic-algorithms/sarsa.md) · [interview](./classic-algorithms/sarsa-interview.md) · [notebook](./classic-algorithms/04_sarsa.ipynb) · [experiments](./classic-algorithms/04_sarsa_experiments.ipynb) |
| Comparing Algorithms — TD vs MC vs Q-Learning vs SARSA | [Core] | [md](./classic-algorithms/comparing-algorithms.md) · [interview](./classic-algorithms/comparing-algorithms-interview.md) · [notebook](./classic-algorithms/05_comparing_algorithms.ipynb) · [experiments](./classic-algorithms/05_comparing_algorithms_experiments.ipynb) |

### deep-rl/

| Topic | Depth | Files |
|-------|-------|-------|
| Function Approximation — neural networks as value estimators | [Core] | [md](./deep-rl/function-approximation.md) · [interview](./deep-rl/function-approximation-interview.md) · [notebook](./deep-rl/01_function_approximation.ipynb) · [experiments](./deep-rl/01_function_approximation_experiments.ipynb) |
| DQN from Scratch — Deep Q-Network implementation | [Core] | [md](./deep-rl/dqn-from-scratch.md) · [interview](./deep-rl/dqn-from-scratch-interview.md) · [notebook](./deep-rl/02_dqn_from_scratch.ipynb) · [experiments](./deep-rl/02_dqn_from_scratch_experiments.ipynb) |
| Experience Replay — breaking correlation in training data | [Core] | [md](./deep-rl/experience-replay.md) · [interview](./deep-rl/experience-replay-interview.md) · [notebook](./deep-rl/03_experience_replay.ipynb) · [experiments](./deep-rl/03_experience_replay_experiments.ipynb) |
| Target Networks — stabilizing Q-learning with neural nets | [Core] | [md](./deep-rl/target-networks.md) · [interview](./deep-rl/target-networks-interview.md) · [notebook](./deep-rl/04_target_networks.ipynb) · [experiments](./deep-rl/04_target_networks_experiments.ipynb) |
| DQN Improvements — Double DQN, Dueling DQN, Rainbow | [Core] | [md](./deep-rl/dqn-improvements.md) · [interview](./deep-rl/dqn-improvements-interview.md) · [notebook](./deep-rl/05_dqn_improvements.ipynb) · [experiments](./deep-rl/05_dqn_improvements_experiments.ipynb) |
| Atari Games — applying DQN to visual environments | [Core] | [md](./deep-rl/atari-games.md) · [interview](./deep-rl/atari-games-interview.md) · [notebook](./deep-rl/06_atari_games.ipynb) · [experiments](./deep-rl/06_atari_games_experiments.ipynb) |

### policy-gradient/

| Topic | Depth | Files |
|-------|-------|-------|
| Policy Gradient Intuition — why learn policies directly | [Core] | [md](./policy-gradient/policy-gradient-intuition.md) · [interview](./policy-gradient/policy-gradient-intuition-interview.md) · [notebook](./policy-gradient/01_policy_gradient_intuition.ipynb) · [experiments](./policy-gradient/01_policy_gradient_intuition_experiments.ipynb) |
| REINFORCE Algorithm — Monte Carlo policy gradient | [Core] | [md](./policy-gradient/reinforce-algorithm.md) · [interview](./policy-gradient/reinforce-algorithm-interview.md) · [notebook](./policy-gradient/02_reinforce_algorithm.ipynb) · [experiments](./policy-gradient/02_reinforce_algorithm_experiments.ipynb) |
| Variance Reduction — baselines and control variates | [Core] | [md](./policy-gradient/variance-reduction.md) · [interview](./policy-gradient/variance-reduction-interview.md) · [notebook](./policy-gradient/03_variance_reduction.ipynb) · [experiments](./policy-gradient/03_variance_reduction_experiments.ipynb) |
| Actor-Critic — combining policy and value learning | [Core] | [md](./policy-gradient/actor-critic.md) · [interview](./policy-gradient/actor-critic-interview.md) · [notebook](./policy-gradient/04_actor_critic.ipynb) · [experiments](./policy-gradient/04_actor_critic_experiments.ipynb) |
| A2C and A3C — synchronous and asynchronous advantage actor-critic | [Core] | [md](./policy-gradient/a2c-a3c.md) · [interview](./policy-gradient/a2c-a3c-interview.md) · [notebook](./policy-gradient/05_a2c_a3c.ipynb) · [experiments](./policy-gradient/05_a2c_a3c_experiments.ipynb) |

### advanced-algorithms/

| Topic | Depth | Files |
|-------|-------|-------|
| Trust Region Methods — TRPO and constrained optimization | [Core] | [md](./advanced-algorithms/trust-region-methods.md) · [interview](./advanced-algorithms/trust-region-methods-interview.md) · [notebook](./advanced-algorithms/01_trust_region_methods.ipynb) · [experiments](./advanced-algorithms/01_trust_region_methods_experiments.ipynb) |
| PPO from Scratch — Proximal Policy Optimization | [Core] | [md](./advanced-algorithms/ppo-from-scratch.md) · [interview](./advanced-algorithms/ppo-from-scratch-interview.md) · [notebook](./advanced-algorithms/02_ppo_from_scratch.ipynb) · [experiments](./advanced-algorithms/02_ppo_from_scratch_experiments.ipynb) |
| PPO with Stable-Baselines3 — production PPO | [Core] | [md](./advanced-algorithms/ppo-with-stable-baselines.md) · [interview](./advanced-algorithms/ppo-with-stable-baselines-interview.md) · [notebook](./advanced-algorithms/03_ppo_with_stable_baselines.ipynb) · [experiments](./advanced-algorithms/03_ppo_with_stable_baselines_experiments.ipynb) |
| SAC Continuous Control — entropy-regularized RL | [Core] | [md](./advanced-algorithms/sac-continuous-control.md) · [interview](./advanced-algorithms/sac-continuous-control-interview.md) · [notebook](./advanced-algorithms/04_sac_continuous_control.ipynb) · [experiments](./advanced-algorithms/04_sac_continuous_control_experiments.ipynb) |
| Comparing Advanced Methods — PPO vs SAC vs TRPO | [Core] | [md](./advanced-algorithms/comparing-advanced-methods.md) · [interview](./advanced-algorithms/comparing-advanced-methods-interview.md) · [notebook](./advanced-algorithms/05_comparing_advanced_methods.ipynb) · [experiments](./advanced-algorithms/05_comparing_advanced_methods_experiments.ipynb) |

### rlhf/

| Topic | Depth | Files |
|-------|-------|-------|
| What is RLHF — aligning models with human preferences | [Core] | [md](./rlhf/what-is-rlhf.md) · [interview](./rlhf/what-is-rlhf-interview.md) · [notebook](./rlhf/01_what_is_rlhf.ipynb) · [experiments](./rlhf/01_what_is_rlhf_experiments.ipynb) |
| Reward Modeling — learning human preferences | [Core] | [md](./rlhf/reward-modeling.md) · [interview](./rlhf/reward-modeling-interview.md) · [notebook](./rlhf/02_reward_modeling.ipynb) · [experiments](./rlhf/02_reward_modeling_experiments.ipynb) |
| PPO for Language Models — adapting PPO for text generation | [Core] | [md](./rlhf/ppo-for-language-models.md) · [interview](./rlhf/ppo-for-language-models-interview.md) · [notebook](./rlhf/03_ppo_for_language_models.ipynb) · [experiments](./rlhf/03_ppo_for_language_models_experiments.ipynb) |
| DPO and Alternatives — Direct Preference Optimization | [Core] | [md](./rlhf/dpo-and-alternatives.md) · [interview](./rlhf/dpo-and-alternatives-interview.md) · [notebook](./rlhf/04_dpo_and_alternatives.ipynb) · [experiments](./rlhf/04_dpo_and_alternatives_experiments.ipynb) |
| TRL Library Tutorial — Transformer Reinforcement Learning | [Core] | [md](./rlhf/trl-library-tutorial.md) · [interview](./rlhf/trl-library-tutorial-interview.md) · [notebook](./rlhf/05_trl_library_tutorial.ipynb) · [experiments](./rlhf/05_trl_library_tutorial_experiments.ipynb) |
| Fine-tuning with RLHF — end-to-end RLHF pipeline | [Core] | [md](./rlhf/fine-tuning-with-rlhf.md) · [interview](./rlhf/fine-tuning-with-rlhf-interview.md) · [notebook](./rlhf/06_fine_tuning_with_rlhf.ipynb) · [experiments](./rlhf/06_fine_tuning_with_rlhf_experiments.ipynb) |

### applications/

| Topic | Depth | Files |
|-------|-------|-------|
| Game Playing — RL for board games and video games | [Core] | [md](./applications/game-playing.md) · [interview](./applications/game-playing-interview.md) · [notebook](./applications/01_game_playing.ipynb) · [experiments](./applications/01_game_playing_experiments.ipynb) |
| Robotics Simulation — RL for continuous control | [Core] | [md](./applications/robotics-simulation.md) · [interview](./applications/robotics-simulation-interview.md) · [notebook](./applications/02_robotics_simulation.ipynb) · [experiments](./applications/02_robotics_simulation_experiments.ipynb) |
| Recommendation Systems — RL for sequential recommendations | [Core] | [md](./applications/recommendation-systems.md) · [interview](./applications/recommendation-systems-interview.md) · [notebook](./applications/03_recommendation_systems.ipynb) · [experiments](./applications/03_recommendation_systems_experiments.ipynb) |
| LLM Alignment Case Study — applying RLHF in practice | [Core] | [md](./applications/llm-alignment-case-study.md) · [interview](./applications/llm-alignment-case-study-interview.md) · [notebook](./applications/04_llm_alignment_case_study.ipynb) · [experiments](./applications/04_llm_alignment_case_study_experiments.ipynb) |
| Multi-Agent Systems — cooperative and competitive RL | [Core] | [md](./applications/multi-agent-systems.md) · [interview](./applications/multi-agent-systems-interview.md) · [notebook](./applications/05_multi_agent_systems.ipynb) · [experiments](./applications/05_multi_agent_systems_experiments.ipynb) |

## Connection to Other Sections

- **00-neural-networks:** Deep RL uses neural networks as function approximators
- **02-fine-tuning:** RLHF is an advanced fine-tuning technique using RL
- **08-ai-agents:** RL provides the learning foundation for autonomous agents
