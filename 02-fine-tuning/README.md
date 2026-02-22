# 2️⃣ Fine-Tuning

## 🎯 What is Fine-Tuning? (The 30-Second Version)

Imagine you hire a **brilliant new employee** who graduated top of their class. They know a LOT about the world - history, science, language, math. But they've never worked at YOUR company before.

**Fine-tuning** is like giving that employee a few weeks of on-the-job training. You're not teaching them everything from scratch - you're just helping them apply what they already know to YOUR specific needs.

In AI terms:
- **Pre-trained model** = The brilliant new hire (knows general language/patterns)
- **Fine-tuning** = On-the-job training (teaching it YOUR specific task)
- **Fine-tuned model** = A specialist who's great at your specific job

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

---

## 🗺️ Study Plan (Start Here!)

This module is designed for **complete beginners**. Follow the notebooks in order - each one builds on the last.

### Phase 1: Understanding the Basics
| # | Notebook | What You'll Learn | Time |
|---|----------|-------------------|------|
| 1 | [What is Fine-Tuning?](./01_what_is_fine_tuning.ipynb) | Why fine-tuning exists, transfer learning, types of fine-tuning | ~30 min |

### Phase 2: Fine-Tuning Methods
| # | Notebook | What You'll Learn | Time |
|---|----------|-------------------|------|
| 2 | [Full Fine-Tuning](./full-fine-tuning/01_full_fine_tuning.ipynb) | Updating ALL model weights, when and why | ~30 min |
| 3 | [Understanding LoRA](./lora-qlora/01_understanding_lora.ipynb) | The clever shortcut that makes fine-tuning affordable | ~40 min |
| 4 | [QLoRA](./lora-qlora/02_qlora.ipynb) | Making LoRA even more memory-efficient with quantization | ~30 min |

### Phase 3: Advanced Topics
| # | Notebook | What You'll Learn | Time |
|---|----------|-------------------|------|
| 5 | [Instruction Tuning](./instruction-tuning/01_instruction_tuning.ipynb) | How ChatGPT-style models learn to follow instructions | ~35 min |

```
  Recommended Path:

  ┌──────────────────────┐
  │ 1. What is           │
  │    Fine-Tuning?      │──── Start here! Understand the big picture
  └──────────┬───────────┘
             │
             v
  ┌──────────────────────┐
  │ 2. Full Fine-Tuning  │──── The "classic" approach (update everything)
  └──────────┬───────────┘
             │
             v
  ┌──────────────────────┐
  │ 3. LoRA              │──── The "smart shortcut" (update only a little)
  └──────────┬───────────┘
             │
             v
  ┌──────────────────────┐
  │ 4. QLoRA             │──── LoRA + compression = even cheaper!
  └──────────┬───────────┘
             │
             v
  ┌──────────────────────┐
  │ 5. Instruction       │──── How chatbots learn to be helpful
  │    Tuning + RLHF     │
  └──────────────────────┘
```

---

## 📋 Prerequisites

Before starting this module, you should understand:

- **What a neural network is** - If not, check out [Module 0: Neural Networks](../00-neural-networks/README.md)
- **Basic Python** - Variables, loops, functions
- **What a "model" is** - A program that learned patterns from data

Don't worry if you don't know:
- Linear algebra (we'll explain what we need!)
- Calculus (we keep the math minimal!)
- How transformers work (helpful but not required)

---

## 🔑 Key Vocabulary (Cheat Sheet)

| Term | Plain English Definition |
|------|------------------------|
| **Pre-trained model** | An AI that already learned from tons of data (like GPT, LLaMA, BERT) |
| **Fine-tuning** | Extra training on specific data to make the model better at one task |
| **Transfer learning** | Using knowledge from one task to help with a different task |
| **Parameters / Weights** | The numbers inside a model that determine its behavior (like knobs on a radio) |
| **LoRA** | A trick to fine-tune only a tiny part of the model instead of everything |
| **QLoRA** | LoRA + compression to use even less computer memory |
| **PEFT** | Parameter-Efficient Fine-Tuning - umbrella term for "cheap" fine-tuning methods |
| **Instruction tuning** | Teaching a model to follow instructions (how ChatGPT was made) |
| **RLHF** | Using human feedback to teach a model what "good" answers look like |
| **Epoch** | One complete pass through all training data |
| **Learning rate** | How big of a step the model takes when learning (too big = chaos, too small = slow) |
| **Overfitting** | When a model memorizes training data but can't generalize (like memorizing answers instead of learning concepts) |

---

## 📂 Directory Structure

```
02-fine-tuning/
├── README.md                          <── You are here!
├── 01_what_is_fine_tuning.ipynb       <── Start here
│
├── full-fine-tuning/                  <── Full fine-tuning deep dive
│   ├── README.md
│   └── 01_full_fine_tuning.ipynb
│
├── lora-qlora/                        <── Parameter-efficient methods
│   ├── README.md
│   ├── 01_understanding_lora.ipynb
│   └── 02_qlora.ipynb
│
├── instruction-tuning/                <── Instruction & RLHF
│   ├── README.md
│   └── 01_instruction_tuning.ipynb
│
└── experiments/                       <── Hands-on experiments
    └── (coming soon)
```

---

## 📖 Key Papers

If you want to go deeper, here are the landmark papers that introduced these ideas:

| Paper | Year | What It Introduced |
|-------|------|--------------------|
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | 2021 | The LoRA technique |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | 2023 | QLoRA (LoRA + 4-bit quantization) |
| [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) | 2022 | Instruction tuning + RLHF |
| [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) | 2021 | Instruction tuning at scale |

---

## 🚀 Why Should You Care About Fine-Tuning?

Fine-tuning is one of the most **practical skills** in modern AI. Here's why:

1. **It's how real AI products are built** - Almost every production AI system uses fine-tuning
2. **It's surprisingly accessible** - With LoRA/QLoRA, you can fine-tune on a single GPU
3. **It's in huge demand** - Companies need people who can customize AI models
4. **It bridges the gap** - Between "I can use ChatGPT" and "I can build AI systems"

```
  The AI Skills Ladder:

  Level 1: Use AI tools (ChatGPT, etc.)           <── Most people stop here
  Level 2: Prompt engineering                       <── Getting smarter
  Level 3: Fine-tune models for specific tasks      <── YOU WILL BE HERE ⭐
  Level 4: Train models from scratch                <── Research territory
  Level 5: Design new architectures                 <── PhD territory
```

---

[Back to Main](../README.md) | [Previous: Transformers](../01-transformers/README.md) | [Next: RAG](../03-rag/README.md)
