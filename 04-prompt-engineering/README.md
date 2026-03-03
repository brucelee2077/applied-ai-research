# 4. Prompt Engineering

The same AI model scores 30% or 90% on the same test — depending on six words you add to your prompt. No retraining. No new data. Just different instructions.

How is that possible? That is what this module is about.

---

**Before you start, you need to know:**
- What an LLM is at a high level (a model trained on text that predicts the next word) — if this is new, read the box below
- How to type a prompt into a chatbot like ChatGPT or Claude — that is all the technical skill you need

---

## Wait, What is an LLM?

**LLM (Large Language Model)** is a type of AI that has read billions of pages of text — books, websites, articles — and learned patterns in language. Think of it like a student who has read the entire library. They can write essays, answer questions, translate languages, and more.

```
┌──────────────────────────────────────────────────────────────────┐
│                     How an LLM Works (Simplified)                │
│                                                                  │
│   You write a prompt          The LLM thinks about               │
│   (your instructions)   -->   patterns it learned       -->  Answer│
│                               from all that reading              │
│                                                                  │
│   "Explain gravity            Searches its knowledge             │
│    like I'm 5"           -->   for simple explanations  -->  "Gravity│
│                               of gravity                    is like│
│                                                             a big │
│                                                             magnet│
│                                                             ..."  │
└──────────────────────────────────────────────────────────────────┘
```

**Key terms explained simply:**

| Term | Simple Explanation |
|------|-------------------|
| **Prompt** | The text you type to tell the AI what to do (your instructions) |
| **Completion / Response** | The text the AI writes back to you |
| **Token** | A small piece of a word. "Hamburger" = ["Ham", "bur", "ger"] = 3 tokens. LLMs read and write in tokens, not full words |
| **Context window** | The AI's "short-term memory" — how much text it can look at at once (e.g., 8,000 tokens ~ 6,000 words) |
| **Temperature** | A "creativity dial." Low (0.0) = focused, predictable answers. High (1.0) = creative, surprising answers |
| **System prompt** | Behind-the-scenes instructions that set the AI's personality or rules (e.g., "You are a helpful math tutor") |
| **Zero-shot** | Asking the AI to do something with NO examples |
| **Few-shot** | Giving the AI a FEW examples before your actual question |

---

## What is Prompt Engineering?

**Prompt engineering** is the skill of writing good instructions to get the best possible answers from AI language models.

**The analogy: a librarian.** Imagine you walk into a huge library and ask the librarian for help. If you say "Books?", the librarian has no idea what you want. If you say "Can you recommend a mystery book for a 12-year-old who liked Harry Potter?", the librarian knows exactly what to look for.

```
Bad question to a librarian:    "Books?"
Better question:                "Can you recommend a book?"
Best question:                  "Can you recommend a mystery book for a 12-year-old
                                 who liked Harry Potter?"
```

**What the analogy gets right:** the more specific and clear your instructions, the better the answer you get back. Both the librarian and the LLM have tons of knowledge — the quality of the result depends on how well you describe what you need.

**The concept in plain words:** prompt engineering is learning how to ask clearly, give helpful context, and structure your instructions so the AI understands exactly what you want.

**Where the analogy breaks down:** a librarian can ask you follow-up questions if your request is unclear. An LLM just does its best with whatever you gave it — it will not stop and say "I need more details." That is why getting the prompt right matters even more with AI.

---

## Why Does Prompt Engineering Matter?

The same AI model can give wildly different answers depending on how you write your prompt.

```
┌───────────────────────────────────────────────────────────────┐
│                    Same AI, Different Prompts                  │
│                                                               │
│  Prompt: "Summarize this"                                     │
│  Result: ⭐⭐ (okay, but vague — summarize how? how long?)    │
│                                                               │
│  Prompt: "Summarize this article in 3 bullet points           │
│           for a high school student"                          │
│  Result: ⭐⭐⭐⭐⭐ (clear, specific, great output)            │
└───────────────────────────────────────────────────────────────┘
```

Good prompt engineering helps you:
- **Get more accurate answers** (fewer mistakes)
- **Save time** (less back-and-forth)
- **Unlock advanced reasoning** (the AI can solve harder problems)
- **Get consistent results** (same format every time)

---

## Key Concepts at a Glance

Here is a quick cheat sheet of the main techniques you will learn:

| Technique | One-Line Summary | When to Use |
|-----------|-----------------|-------------|
| **Zero-shot** | Just ask — no examples | Simple, straightforward tasks |
| **Few-shot** | Give a few examples first | When you need a specific format or style |
| **Chain-of-Thought** | Ask the AI to think step-by-step | Math, logic, complex reasoning |
| **Prompt Templates** | Reusable prompt "recipes" with blanks to fill in | Repeated tasks with the same structure |
| **Self-Consistency** | Ask the same question multiple times, pick the most common answer | When accuracy really matters |

```
┌───────────────────────────────────────────────────────────────────────┐
│                     When to Use Which Technique                       │
│                                                                       │
│   Task Type              Best Technique          Why                   │
│   ─────────────────────────────────────────────────────────────────── │
│   Simple question        Zero-shot (just ask)    AI already knows how │
│                                                                       │
│   Need specific format   Few-shot               Examples show the     │
│   or style               (give examples)         format you want      │
│                                                                       │
│   Math, logic, or        Chain-of-thought        Step-by-step reduces │
│   multi-step problem     ("think step by step")  errors               │
│                                                                       │
│   Repeated similar       Prompt template          Write once, reuse   │
│   tasks                  (fill-in-the-blank)     with different inputs│
│                                                                       │
│   Really hard problem    CoT + Few-shot +        Combine techniques   │
│   needing high accuracy  Self-consistency         for best results    │
└───────────────────────────────────────────────────────────────────────┘
```

**Combining techniques:** the real power comes from mixing techniques together. For example, you can put few-shot examples inside a template, add "Let's think step by step" for reasoning, and run it multiple times for self-consistency. Each technique file explains how it connects to the others.

---

## Study Plan

Here is the recommended order. Each topic builds on the one before it.

```
    START HERE
        │
        ▼
┌───────────────────────┐
│  1. Few-Shot Learning │  Learn how giving examples helps the AI
│                       │  understand what you want
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  2. Chain-of-Thought  │  Learn how to make the AI "show its work"
│                       │  for harder problems
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  3. Prompt Templates  │  Learn to build reusable prompt "recipes"
│                       │  you can use again and again
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  4. Prompt Evaluation │  Learn how to measure whether your
│                       │  prompts are actually working well
└───────────────────────┘
```

---

**Quick check — can you answer these?**
- In your own words: why does the same AI give different quality answers to different prompts?
- What is the difference between zero-shot and few-shot?
- Why might you want to use a template instead of writing a new prompt each time?

If you cannot answer one, re-read that section above. That is completely normal.

---

## You Just Learned the Foundation

You now understand the core idea behind prompt engineering: the words you choose change the result. This single insight is behind every technique in this module — few-shot examples, chain-of-thought reasoning, templates, and evaluation. Each one is a way to make your instructions clearer and more effective.

Companies like Google, OpenAI, and Anthropic have entire teams dedicated to prompt engineering. The techniques you are about to learn are the same ones they use. Let's get started.

---

## Coverage Map

| Topic | Depth | Files |
|-------|-------|-------|
| Few-shot learning — teaching the AI by example | [Applied] | [few-shot-learning.md](./few-shot-learning.md) · [01_few_shot_learning.ipynb](./01_few_shot_learning.ipynb) |
| Chain-of-thought — making the AI show its work | [Applied] | [chain-of-thought.md](./chain-of-thought.md) · [02_chain_of_thought.ipynb](./02_chain_of_thought.ipynb) |
| Prompt templates — reusable prompt recipes | [Applied] | [prompt-templates.md](./prompt-templates.md) · [03_prompt_templates.ipynb](./03_prompt_templates.ipynb) |
| Prompt evaluation — measuring prompt quality | [Applied] | [prompt-evaluation.md](./prompt-evaluation.md) · [04_prompt_evaluation.ipynb](./04_prompt_evaluation.ipynb) |

---

## Key Papers

If you want to go deeper, these are the landmark research papers behind the techniques:

- **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** — Wei et al., 2022
  - The paper that showed adding "Let's think step by step" dramatically improves AI reasoning
- **Self-Consistency Improves Chain of Thought Reasoning** — Wang et al., 2022
  - Showed that asking the same question multiple times and picking the majority answer improves accuracy
- **Language Models are Few-Shot Learners** — Brown et al., 2020
  - The GPT-3 paper that demonstrated LLMs can learn new tasks from just a few examples in the prompt

---

[Back to Main](../README.md) | [Previous: RAG](../03-rag/README.md) | [Next: Multimodal](../05-multimodal/README.md)
