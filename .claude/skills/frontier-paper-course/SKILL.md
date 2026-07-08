---
name: frontier-paper-course
description: Convert a paper, blog post, transcript, or technical article into an interactive Frontier Lab learning module with reading map, intuition, experiments, HTML/D3 visuals, and a concrete artifact plan.
---

# Frontier Paper Course

Use this when the user gives a paper, blog, transcript, arXiv link, PDF, or paper title and wants to learn it deeply.

## Learning objective

Do not merely summarize.

Convert the source into:

```text
Paper → mental model → mechanism → visualization → experiment → frontier-lab relevance → portfolio artifact
```

## Output files

Create:

```text
notes/papers/<paper-slug>.md
courseware/<paper-slug>/index.html
experiments/<paper-slug>/README.md
```

Optional:

```text
courseware/<paper-slug>/main.js
courseware/<paper-slug>/style.css
experiments/<paper-slug>/src/
experiments/<paper-slug>/tests/
```

## Paper reading map

Tell the learner:

- what to read carefully
- what to skim
- what to skip for now
- which figure/table matters most
- which claim should be tested with a mini experiment
- which terms must be understood before reading deeply

## Paper note structure

```markdown
# <Paper Title>

## 1. One-sentence takeaway

## 2. Why this paper exists
What problem forced this paper to exist?

## 3. Beginner intuition
Explain with a story or analogy.

## 4. Core mechanism
Step by step.

## 5. Key equations / algorithms
Use Math Ladder:
problem, intuition, symbols, tiny example, formula, sanity check, misconception.

## 6. What changed after this paper
Why it matters historically or practically.

## 7. Limitations
What the paper does not solve.

## 8. Frontier-lab relevance
Why OpenAI / Anthropic / DeepMind-style teams would care.

## 9. Mini experiment
A small reproduction or toy demo.

## 10. Quiz
5 questions.

## 11. Research taste note
What makes this paper important, overhyped, under-tested, or reusable?
```

## Visualization requirement

Create at least one visual explanation when the paper has:

- curves
- matrices
- routing
- timelines
- token flows
- distributed-system pipelines
- before/after comparisons

Use D3.js only when it teaches the concept better than a static diagram.

## Experiment requirement

Always propose one small experiment.

Examples:

- implement attention from scratch
- fit a toy scaling law
- benchmark batching latency
- compare dense vs sparse routing
- simulate quantization error
- visualize KV cache growth

## Tone

Warm and concrete.

The learner should feel:

> “I can actually understand this paper.”

Not:

> “I read a fancy abstract and blacked out.”
