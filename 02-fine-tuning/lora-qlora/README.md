# LoRA & QLoRA

## 🎯 Overview

LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) are clever techniques that let you fine-tune massive AI models **without needing massive resources**. Instead of updating all billions of parameters, they add and train tiny "adapter" layers.

```
  The LoRA Idea (Simplified):

  Instead of this:                    Do this:
  ┌──────────────────────┐           ┌──────────────────────┐
  │  Update ALL 7 billion│           │  Freeze 7 billion    │
  │  parameters          │           │  parameters (don't   │
  │  (needs HUGE GPU)    │           │  touch them!)        │
  │                      │           │                      │
  │  Cost: $$$$$         │           │  + Add tiny adapter  │
  │  Memory: 100+ GB    │           │    (only ~0.1% new)  │
  └──────────────────────┘           │                      │
                                     │  Cost: $             │
                                     │  Memory: ~6 GB       │
                                     └──────────────────────┘
```

## 📂 Contents

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [01_understanding_lora.ipynb](./01_understanding_lora.ipynb) | LoRA explained from scratch with analogies, math made simple, and hands-on code |
| 2 | [02_qlora.ipynb](./02_qlora.ipynb) | QLoRA: combining LoRA with quantization for maximum efficiency |

## 🔑 Key Concepts Covered

- [x] Why we need parameter-efficient fine-tuning
- [x] Matrix math explained visually (no PhD required!)
- [x] Low-rank decomposition with real-world analogies
- [x] LoRA architecture and how adapters work
- [x] What quantization is and why it helps
- [x] QLoRA = LoRA + 4-bit quantization
- [x] Hands-on code examples

## 📋 Prerequisites

- Completed [01_what_is_fine_tuning.ipynb](../01_what_is_fine_tuning.ipynb)
- Completed [Full Fine-Tuning](../full-fine-tuning/01_full_fine_tuning.ipynb) (recommended)
- Basic Python knowledge

---

[Back to Fine-Tuning](../README.md) | [Previous: Full Fine-Tuning](../full-fine-tuning/README.md) | [Next: Instruction Tuning](../instruction-tuning/README.md)
