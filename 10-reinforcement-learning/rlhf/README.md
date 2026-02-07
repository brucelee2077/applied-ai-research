# RLHF: Reinforcement Learning from Human Feedback

This section covers how RL is used to align language models with human preferences - the key technique behind ChatGPT.

## What You'll Learn

- The RLHF pipeline
- Reward modeling from preferences
- PPO for language models
- Alternatives like DPO
- Using the TRL library

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [What is RLHF?](01_what_is_rlhf.ipynb) | Overview of RLHF and why it matters |
| 2 | [Reward Modeling](02_reward_modeling.ipynb) | Training reward models from preferences |
| 3 | [PPO for Language Models](03_ppo_for_language_models.ipynb) | Adapting PPO for text generation |
| 4 | [DPO and Alternatives](04_dpo_and_alternatives.ipynb) | Direct Preference Optimization |
| 5 | [TRL Library Tutorial](05_trl_library_tutorial.ipynb) | Using Hugging Face TRL |
| 6 | [Full RLHF Pipeline](06_fine_tuning_with_rlhf.ipynb) | End-to-end implementation |

## Prerequisites

- [Advanced Algorithms](../advanced-algorithms/) (especially PPO)
- [02-fine-tuning](../../02-fine-tuning/) (LoRA/PEFT)
- Transformers and PyTorch

## Key Concepts

| Stage | Description |
|-------|-------------|
| **SFT** | Supervised Fine-Tuning on demonstrations |
| **Reward Model** | Train to predict human preferences |
| **PPO** | Optimize policy to maximize reward |
| **DPO** | Direct optimization without reward model |

## The RLHF Pipeline

```
1. Pretrained LLM
       ↓
2. SFT (Supervised Fine-Tuning)
       ↓
3. Reward Model Training
       ↓
4. PPO Fine-Tuning
       ↓
5. Aligned LLM
```

## Required Libraries

```bash
pip install trl transformers peft accelerate
```
