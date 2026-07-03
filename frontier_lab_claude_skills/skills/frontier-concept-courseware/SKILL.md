---
name: frontier-concept-courseware
description: Turn any Frontier Lab concept into beginner-friendly, deep, bilingual HTML courseware with diagrams, examples, quiz, and an optional D3.js interactive visualization. Use for topics like attention, scaling laws, KV cache, MoE, quantization, Pallas kernels, JAX, inference systems, RLHF, DPO, or agent loops.
---

# Frontier Concept Courseware

Create an interactive HTML mini-course for one technical concept.

## Required goal
Lower the bar **without lowering the ceiling**:
- The first 20% should be understandable to a beginner.
- The middle 60% should explain the actual mechanism.
- The final 20% should connect to frontier-lab research engineering.

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
   - Topic title
   - One-line intuition
   - Difficulty meter: Beginner → Staff Engineer

2. Story/analogy section
   - Use a vivid everyday analogy.
   - Example: attention is a meeting where every token decides who to listen to.

3. Step-by-step mechanism
   - Explain the pipeline.
   - Include formulas only after intuition.

4. Interactive visual
   - Prefer D3.js for graphs, flows, matrices, routing, curves, timelines, or tokens.
   - Include slider/button/toggle when meaningful.

5. Frontier-lab relevance
   - Why this matters for pretraining, post-training, inference, agents, or systems.

6. Mini lab
   - A small experiment the learner can implement.

7. Quiz
   - 5 questions with hidden or collapsible answers if feasible.

8. Research log
   - 3 reflection prompts.

## D3 visualization patterns
Use D3.js for:
- Attention matrix heatmap
- Token-to-token attention arrows
- Scaling laws curves
- KV cache timeline
- MoE expert routing
- Quantization precision comparison
- Batch/prefill/decode serving pipeline
- GPU memory bandwidth bottleneck diagram

Use local D3 if present:

```html
<script src="./vendor/d3.v7.min.js"></script>
```

Otherwise use:

```html
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
```

Include a no-D3 fallback message or static SVG.

## Bilingual style
Use Chinese for intuition and emotional scaffolding, English for terms/interview phrasing.

Example:

> 这里的 intuition 是：attention 不是“模型有意识地理解你”，而是每个 token 在每一层里重新计算 relevance。In English: each token asks, “Who should I listen to right now?”

## Quality bar
The result should feel like a small internal course page, not a random blog post. It should be visually clean, runnable locally, and useful for review.
