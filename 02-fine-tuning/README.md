# Fine-Tuning

## The Mystery Worth Solving

GPT-3 has 175 billion parameters and was trained on most of the internet. It knows more facts than any human who ever lived. But ask it to write a medical report, summarize legal documents in your company's style, or classify your customer support tickets — and it stumbles. It gives vague, generic answers.

Then something remarkable happens. You show it just 1,000 examples of your specific task — tiny compared to the trillions of words it already saw — and suddenly it becomes an expert. A model that took months and millions of dollars to train can be reshaped for your exact needs in a few hours on a single GPU.

How can such a small amount of data change such a massive model? What exactly changes inside when you fine-tune? And why do some methods change every single parameter while others change less than 1% and get the same results?

That's what this module is about.

---

## The 30-Second Version

Imagine you hire a brilliant new employee who graduated top of their class. They know a lot about the world — history, science, language, math. But they have never worked at your company before.

**Fine-tuning** is like giving that employee a few weeks of on-the-job training. You are not teaching them everything from scratch — you are just helping them apply what they already know to your specific needs.

```
    Pre-trained Model              Fine-Tuning                Fine-tuned Model
  ┌─────────────────┐         ┌──────────────────┐        ┌──────────────────┐
  │  Knows general  │         │  Train on YOUR   │        │  Expert at YOUR  │
  │  language and   │  ──────>│  specific data   │──────> │  specific task!  │
  │  world facts    │         │  and examples    │        │                  │
  └─────────────────┘         └──────────────────┘        └──────────────────┘
   "I know English"           "Here are 1000              "I can diagnose
                               medical reports"            medical conditions"
```

**What this analogy gets right:** The employee already has general knowledge (the pre-trained weights). Training them on your company's processes (fine-tuning data) makes them useful fast, without starting from zero.

**Where this analogy breaks down:** A real employee keeps all their old skills when they learn new ones. A neural network can forget old abilities during fine-tuning — a real problem called catastrophic forgetting.

---

## Coverage Map

| Topic | Depth | Files |
|-------|-------|-------|
| What is Fine-Tuning — transfer learning, types of fine-tuning, when to use it | [Applied] | [concept notebook](./01_what_is_fine_tuning.ipynb) |
| Full Fine-Tuning — updating all parameters, memory cost, catastrophic forgetting | [Core] | [theory](./full-fine-tuning.md) · [interview](./full-fine-tuning-interview.md) · [concept notebook](./02_full_fine_tuning.ipynb) · [experiments](./02_full_fine_tuning_experiments.ipynb) |
| LoRA — low-rank adapters, frozen weights, rank selection | [Core] | [theory](./lora.md) · [interview](./lora-interview.md) · [concept notebook](./03_lora.ipynb) · [experiments](./03_lora_experiments.ipynb) |
| QLoRA — quantized LoRA, NF4, double quantization | [Applied] | [theory](./qlora.md) · [concept notebook](./04_qlora.ipynb) |
| Instruction Tuning — SFT, RLHF, DPO, reward modeling | [Core] | [theory](./instruction-tuning.md) · [interview](./instruction-tuning-interview.md) · [concept notebook](./05_instruction_tuning.ipynb) · [experiments](./05_instruction_tuning_experiments.ipynb) |

---

## Study Plan

Follow this path from top to bottom. Each section builds on the previous one.

```
START HERE
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1: Understand the Basics                                   │
│                                                                    │
│  1. What is Fine-Tuning?             (01_what_is_fine_tuning)     │
│     → Why fine-tuning exists, transfer learning, types             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 2: Fine-Tuning Methods                                     │
│                                                                    │
│  2. Full Fine-Tuning                                               │
│     → Read: full-fine-tuning.md (theory)                           │
│     → Code: 02_full_fine_tuning.ipynb (build from scratch)         │
│                                                                    │
│  3. LoRA                                                           │
│     → Read: lora.md (theory)                                       │
│     → Code: 03_lora.ipynb (build from scratch)                     │
│                                                                    │
│  4. QLoRA                                                          │
│     → Read: qlora.md (theory)                                      │
│     → Code: 04_qlora.ipynb (quantized fine-tuning)                 │
│                                                                    │
│  Ready for interviews? Read the -interview.md files and run        │
│  the _experiments.ipynb notebooks for each topic.                  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 3: Advanced Topics                                         │
│                                                                    │
│  5. Instruction Tuning + RLHF                                      │
│     → Read: instruction-tuning.md (theory)                         │
│     → Code: 05_instruction_tuning.ipynb (SFT, RLHF, DPO)         │
└──────────────────────────────────────────────────────────────────┘
```

### Recommended Reading Order

For each topic, read the `.md` file first (theory), then work through the `.ipynb` notebook (code):

| Step | Theory (Read) | Code (Hands-on) | What You'll Learn |
|------|---------------|------------------|-------------------|
| 1 | — | [What is Fine-Tuning?](./01_what_is_fine_tuning.ipynb) | Transfer learning, types of fine-tuning |
| 2 | [Full Fine-Tuning](./full-fine-tuning.md) | [Concept](./02_full_fine_tuning.ipynb) | Update all weights, memory cost, when to use |
| 3 | [LoRA](./lora.md) | [Concept](./03_lora.ipynb) | Low-rank adapters, frozen weights, rank selection |
| 4 | [QLoRA](./qlora.md) | [Concept](./04_qlora.ipynb) | Quantized LoRA, NF4, double quantization |
| 5 | [Instruction Tuning](./instruction-tuning.md) | [Concept](./05_instruction_tuning.ipynb) | SFT, RLHF, DPO, reward modeling |

---

## Prerequisites

Before starting this module, you should understand:

- **What a neural network is** — covered in [Module 0: Neural Networks](../00-neural-networks/README.md)
- **Basic Python** — variables, loops, functions
- **What a "model" is** — a program that learned patterns from data

Don't worry if you don't know:
- Linear algebra (we explain what we need)
- Calculus (we keep the math minimal)
- How transformers work (helpful but not required)

---

## Key Vocabulary

| Term | Plain English Definition |
|------|------------------------|
| **Pre-trained model** | An AI that already learned from lots of data (like GPT, LLaMA, BERT) |
| **Fine-tuning** | Extra training on specific data to make the model better at one task |
| **Transfer learning** | Using knowledge from one task to help with a different task |
| **Parameters / Weights** | The numbers inside a model that determine its behavior |
| **LoRA** | A trick to fine-tune only a tiny part of the model instead of everything |
| **QLoRA** | LoRA + compression to use even less computer memory |
| **PEFT** | Parameter-Efficient Fine-Tuning — umbrella term for "cheap" fine-tuning methods |
| **Instruction tuning** | Teaching a model to follow instructions (how ChatGPT was made) |
| **RLHF** | Using human feedback to teach a model what "good" answers look like |
| **DPO** | Direct Preference Optimization — a simpler alternative to RLHF |
| **Epoch** | One complete pass through all training data |
| **Learning rate** | How big of a step the model takes when learning |
| **Catastrophic forgetting** | When fine-tuning destroys the model's original abilities |

---

## Directory Structure

```
02-fine-tuning/
├── README.md                                    ← You are here (study plan)
├── PROGRESS.md                                  ← Session tracking
├── full-fine-tuning.md                          ← Full FT theory (Layer 1)
├── full-fine-tuning-interview.md                ← Full FT interview depth (Layer 2)
├── lora.md                                      ← LoRA theory (Layer 1)
├── lora-interview.md                            ← LoRA interview depth (Layer 2)
├── qlora.md                                     ← QLoRA theory (Layer 1)
├── instruction-tuning.md                        ← Instruction tuning theory (Layer 1)
├── instruction-tuning-interview.md              ← Instruction tuning interview depth (Layer 2)
├── 01_what_is_fine_tuning.ipynb                 ← Start here
├── 02_full_fine_tuning.ipynb                    ← Full fine-tuning from scratch
├── 02_full_fine_tuning_experiments.ipynb         ← Benchmark + ablate full FT
├── 03_lora.ipynb                                ← LoRA from scratch
├── 03_lora_experiments.ipynb                     ← Rank ablation + memory comparison
├── 04_qlora.ipynb                               ← QLoRA in practice
├── 05_instruction_tuning.ipynb                  ← SFT, RLHF, DPO
└── 05_instruction_tuning_experiments.ipynb       ← Loss masking + KL sweep
```

---

## Key Papers

| Paper | Year | What It Introduced |
|-------|------|--------------------|
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | 2021 | The LoRA technique |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | 2023 | QLoRA (LoRA + 4-bit quantization) |
| [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) | 2022 | Instruction tuning + RLHF |
| [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) | 2021 | Instruction tuning at scale |
| [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) | 2023 | Simpler alternative to RLHF |

---

[Back to Main](../README.md) | [Previous: Transformers](../01-transformers/README.md) | [Next: RAG](../03-rag/README.md)
