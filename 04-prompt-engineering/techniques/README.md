# Prompting Techniques

## Overview

This folder covers the core techniques for writing effective prompts. Think of these
as tools in your toolbox -- each one is best suited for different types of tasks.

You don't need to use every technique for every prompt. Start simple, and reach for
more advanced techniques only when the simple approach doesn't give you good enough
results.

---

## Techniques Covered

| # | Technique | File | What It Does |
|---|-----------|------|-------------|
| 1 | **Few-Shot Learning** | [few-shot-learning.md](./few-shot-learning.md) | Give the AI a few examples so it learns the pattern you want |
| 2 | **Chain-of-Thought** | [chain-of-thought.md](./chain-of-thought.md) | Make the AI "show its work" and think step by step |
| 3 | **Prompt Templates** | [prompt-templates.md](./prompt-templates.md) | Build reusable prompt "recipes" with fill-in-the-blank variables |

---

## Quick Comparison

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

---

## Recommended Reading Order

1. **[Few-Shot Learning](./few-shot-learning.md)** -- Start here. Understanding how
   examples change AI behavior is the foundation of everything else.

2. **[Chain-of-Thought](./chain-of-thought.md)** -- Next, learn how to unlock the AI's
   reasoning ability for harder problems.

3. **[Prompt Templates](./prompt-templates.md)** -- Finally, learn how to package your
   best prompts into reusable templates.

---

## Combining Techniques

The real power comes from **mixing techniques** together. Here's an example that
combines few-shot + chain-of-thought + template:

```
Template (Few-Shot CoT):

  "You are a {role}. Solve the following problem step by step.

   Example:
   Problem: {example_problem}
   Solution:
   {example_solution}

   Now solve this:
   Problem: {actual_problem}
   Solution:"
```

As you read through each technique, think about how they could work together for
your specific use case.

---

[Back to Prompt Engineering](../README.md)
