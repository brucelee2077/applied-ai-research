# 4. Prompt Engineering

## What is Prompt Engineering?

Imagine you have a super-smart robot friend who can answer any question and do almost any
writing task. But here's the catch: **how you ask** matters just as much as **what you ask**.

**Prompt engineering** is the skill of writing good instructions (called "prompts") to get
the best possible answers from AI language models.

Think of it like this:

```
Bad question to a librarian:    "Books?"
Better question:                "Can you recommend a book?"
Best question:                  "Can you recommend a mystery book for a 12-year-old
                                 who liked Harry Potter?"
```

The more specific and clear your instructions, the better the answer. That's prompt
engineering in a nutshell.

---

## Wait, What is an LLM?

Before we dive in, let's cover some terms you'll see everywhere:

**LLM (Large Language Model)** is a type of AI that has read billions of pages of text
(books, websites, articles) and learned patterns in language. Think of it like a student
who has read the entire library -- they can write essays, answer questions, translate
languages, and more.

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
| **Context window** | The AI's "short-term memory" -- how much text it can look at at once (e.g., 8,000 tokens ~ 6,000 words) |
| **Temperature** | A "creativity dial." Low (0.0) = focused, predictable answers. High (1.0) = creative, surprising answers |
| **System prompt** | Behind-the-scenes instructions that set the AI's personality or rules (e.g., "You are a helpful math tutor") |
| **Zero-shot** | Asking the AI to do something with NO examples |
| **Few-shot** | Giving the AI a FEW examples before your actual question |

---

## Why Does Prompt Engineering Matter?

The same AI model can give wildly different answers depending on how you write your prompt.

```
┌───────────────────────────────────────────────────────────────┐
│                    Same AI, Different Prompts                  │
│                                                               │
│  Prompt: "Summarize this"                                     │
│  Result: ⭐⭐ (okay, but vague -- summarize how? how long?)   │
│                                                               │
│  Prompt: "Summarize this article in 3 bullet points           │
│           for a high school student"                          │
│  Result: ⭐⭐⭐⭐⭐ (clear, specific, great output)           │
└───────────────────────────────────────────────────────────────┘
```

Good prompt engineering helps you:
- **Get more accurate answers** (fewer mistakes)
- **Save time** (less back-and-forth)
- **Unlock advanced reasoning** (the AI can solve harder problems)
- **Get consistent results** (same format every time)

---

## Study Plan

Here's the recommended order to learn the topics in this module. Start from the top
and work your way down -- each topic builds on the one before it.

```
    START HERE
        │
        ▼
┌───────────────────────┐
│  1. Few-Shot Learning │  Learn how giving examples helps the AI
│     (techniques/)     │  understand what you want
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  2. Chain-of-Thought  │  Learn how to make the AI "show its work"
│     (techniques/)     │  for harder problems
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  3. Prompt Templates  │  Learn to build reusable prompt "recipes"
│     (techniques/)     │  you can use again and again
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  4. Evaluation        │  Learn how to measure whether your
│     (evaluation/)     │  prompts are actually working well
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  5. Experiments       │  Try it yourself! Build and test
│     (experiments/)    │  your own prompts
└───────────────────────┘
```

**Prerequisites:** No ML knowledge needed! Just basic familiarity with typing prompts
into an AI chatbot (like ChatGPT or Claude).

---

## Directory Structure

```
04-prompt-engineering/
├── README.md                          # You are here
├── techniques/                        # Core prompting techniques
│   ├── README.md                      #   Overview & comparison of all techniques
│   ├── few-shot-learning.md           #   Learning from examples in your prompt
│   ├── chain-of-thought.md            #   Getting the AI to reason step-by-step
│   └── prompt-templates.md            #   Reusable prompt patterns & recipes
├── evaluation/                        # Measuring prompt quality
│   └── README.md                      #   Metrics, A/B testing, scoring methods
└── experiments/                       # Hands-on practice
    └── (your experiments go here!)
```

---

## Key Concepts at a Glance

Here's a quick cheat sheet of the main techniques you'll learn:

| Technique | One-Line Summary | When to Use |
|-----------|-----------------|-------------|
| **Zero-shot** | Just ask -- no examples | Simple, straightforward tasks |
| **Few-shot** | Give a few examples first | When you need a specific format or style |
| **Chain-of-Thought** | Ask the AI to think step-by-step | Math, logic, complex reasoning |
| **Prompt Templates** | Reusable prompt "recipes" with blanks to fill in | Repeated tasks with the same structure |
| **Self-Consistency** | Ask the same question multiple times, pick the most common answer | When accuracy really matters |

---

## Key Papers

If you want to go deeper, these are the landmark research papers behind the techniques:

- **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** -- Wei et al., 2022
  - The paper that showed adding "Let's think step by step" dramatically improves AI reasoning
- **Self-Consistency Improves Chain of Thought Reasoning** -- Wang et al., 2022
  - Showed that asking the same question multiple times and picking the majority answer improves accuracy
- **Language Models are Few-Shot Learners** -- Brown et al., 2020
  - The GPT-3 paper that demonstrated LLMs can learn new tasks from just a few examples in the prompt

---

[Back to Main](../README.md) | [Previous: RAG](../03-rag/README.md) | [Next: Multimodal](../05-multimodal/README.md)
