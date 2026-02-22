# Instruction Tuning

## 🎯 Overview

Instruction tuning is how raw language models are transformed into helpful assistants like ChatGPT. Instead of just predicting the next word, instruction-tuned models learn to **follow instructions** and **have conversations**.

```
  Before Instruction Tuning:           After Instruction Tuning:
  ┌────────────────────────────┐      ┌────────────────────────────┐
  │  User: "What is Python?"   │      │  User: "What is Python?"   │
  │                            │      │                            │
  │  Model: "Python is Python  │      │  Model: "Python is a       │
  │  is a python snake is a    │      │  popular programming       │
  │  large snake found in..."  │      │  language used for web      │
  │                            │      │  development, data science, │
  │  (just continues text!)    │      │  and AI. Here are some      │
  │                            │      │  key features:..."          │
  └────────────────────────────┘      └────────────────────────────┘
```

## 📂 Contents

| Notebook | Description |
|----------|-------------|
| [01_instruction_tuning.ipynb](./01_instruction_tuning.ipynb) | Complete guide to instruction tuning, dataset formats, and RLHF |

## 🔑 Key Concepts Covered

- [x] What instruction tuning is and why it matters
- [x] How ChatGPT was made (the 3-step process)
- [x] Instruction dataset formats with examples
- [x] RLHF (Reinforcement Learning from Human Feedback) explained simply
- [x] DPO (Direct Preference Optimization) as an alternative to RLHF
- [x] How to create your own instruction datasets

## 📋 Prerequisites

- Completed [01_what_is_fine_tuning.ipynb](../01_what_is_fine_tuning.ipynb)
- LoRA/QLoRA notebooks (recommended)
- Basic Python knowledge

---

[Back to Fine-Tuning](../README.md) | [Previous: LoRA & QLoRA](../lora-qlora/README.md)
