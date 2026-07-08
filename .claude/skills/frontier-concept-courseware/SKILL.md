---
name: frontier-concept-courseware
description: Turn any Frontier Lab concept into beginner-friendly, deep, bilingual HTML courseware with diagrams, examples, quiz, and an optional D3.js interactive visualization.
---

# Frontier Concept Courseware

Create an interactive HTML mini-course for one technical concept.

Use this for new courseware. If modifying existing `sessions/**/lesson.html`, prefer `frontier-lesson-humanizer` or `frontier-curriculum-refactor`.

## Required goal

Lower the bar without lowering the ceiling:

- first 20% understandable to a beginner
- middle 60% explains the actual mechanism
- final 20% connects to frontier-lab research engineering

## Output files

Always create or update:

```text
courseware/<topic-slug>/index.html
courseware/<topic-slug>/README.md
courseware/<topic-slug>/lesson-notes.md
```

If JavaScript is nontrivial, also create:

```text
courseware/<topic-slug>/main.js
courseware/<topic-slug>/style.css
```

## HTML course structure

The HTML page must contain:

1. Hero section
   - topic title
   - one-line intuition
   - difficulty meter: Beginner → Staff Engineer

2. Story / analogy section
   - vivid everyday analogy
   - where the analogy breaks

3. Step-by-step mechanism
   - pipeline
   - inputs / outputs
   - formulas only after intuition

4. Interactive visual
   - D3.js for graphs, flows, matrices, routing, curves, timelines, or tokens
   - slider/button/toggle only when meaningful

5. Math Ladder if needed
   - problem
   - intuition
   - symbols
   - tiny example
   - formula
   - sanity check
   - misconception

6. Frontier-lab relevance
   - why this matters for pretraining, post-training, inference, agents, or systems

7. Mini lab
   - a small experiment the learner can implement

8. Quiz
   - 5 questions with hidden or collapsible answers if feasible

9. Research log
   - 3 reflection prompts

## D3 visualization patterns

Use D3.js for:

- attention matrix heatmap
- token-to-token attention arrows
- scaling laws curves
- KV cache timeline
- MoE expert routing
- quantization precision comparison
- batch/prefill/decode serving pipeline
- GPU memory bandwidth bottleneck diagram

Use local D3 if present. Otherwise use CDN only if acceptable for this repo.

## Bilingual style

Use Chinese for intuition and emotional scaffolding.

Use English for terms, formulas, and interview phrasing.

Example:

> 这里的 intuition 是：attention 不是“模型有意识地理解你”，而是每个 token 在每一层里重新计算 relevance。In English: each token asks, “Who should I listen to right now?”

## Quality bar

The result should feel like a small internal course page, not a random blog post.

It should be visually clean, runnable locally, and useful for review.
