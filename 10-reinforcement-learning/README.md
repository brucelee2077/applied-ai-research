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

## Connection to Other Sections

- **00-neural-networks:** Deep RL uses neural networks as function approximators
- **02-fine-tuning:** RLHF is an advanced fine-tuning technique using RL
- **08-ai-agents:** RL provides the learning foundation for autonomous agents
