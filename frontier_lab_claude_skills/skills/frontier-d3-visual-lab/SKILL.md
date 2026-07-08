---
name: frontier-d3-visual-lab
description: Build or improve clean HTML + D3.js learning visualizations for Frontier Lab concepts, especially inside existing sessions/ lessons or standalone courseware pages.
---

# Frontier D3 Visual Lab

Create interactive HTML/D3.js learning visuals that clarify AI/ML mechanisms.

## Core principle

A visualization must answer one learning question.

Bad:

> Pretty animation with no learning point.

Good:

> A sequence-length slider shows why KV cache saves compute but increases memory pressure.

## Before coding, define the visual spec

Always write or infer:

1. Learning question
2. Concept being visualized
3. Inputs / controls
4. What changes on screen
5. What the learner should notice
6. Misconception it prevents
7. Where the visual is simplified

## Good visual types

### Attention

- tokens as nodes
- attention weights as arrows or heatmap
- slider for attention sharpness / temperature
- show every token asking: “Who should I listen to?”

### Scaling laws

- log-log curve
- model size, data size, compute sliders
- undertraining vs compute-optimal intuition
- show why power laws look straight on log-log axes

### KV cache

- timeline: prompt prefill → decode steps
- cache blocks accumulating over time
- show compute saved vs memory added
- explain why decoding becomes memory-bandwidth sensitive

### MoE

- tokens routed to experts
- capacity factor
- dropped or overflow tokens
- dense vs sparse compute comparison

### Quantization

- continuous values → lower precision buckets
- memory saving vs reconstruction error
- show why lower precision is not free

### Inference serving

- request queue
- batching
- prefill/decode split
- p50/p95 latency
- throughput vs latency tension

### Distributed training

- devices as workers
- compute vs communication
- gradient all-reduce
- idle time from stragglers


## Visualization quality bar

For abstract math and systems lessons, a terminal-style playground is usually not enough.

A strong visual should show at least one of:

- shape: curve, line, distribution, surface
- flow: tokens, gradients, requests, data
- growth: memory, compute, latency, cost
- routing: tokens to experts, requests to workers
- bottleneck: bandwidth, queueing, stragglers
- transformation: raw scale to log scale, fp16 to int8, sequence to cache

If the current playground only prints staged terminal output, either:

1. upgrade it to a real visual mechanism, or
2. add an interactive calculator with sliders, or
3. explicitly document why terminal walkthrough is the right choice.

## Required visual card

Every new or upgraded visual must include:

```html
<p><strong>What you should notice:</strong> ...</p>
<p><strong>Common misconception:</strong> ...</p>
<p><strong>Where this visual simplifies reality:</strong> ...</p>
```

## Pilot lesson upgrade patterns

### Logs and exponents

Preferred visual:
- left: normal scale curve
- right: log-log straight line
- exponent slider
- display slope = exponent

### Transformer arithmetic

Preferred visual:
- N slider
- D slider
- forward FLOPs bar
- backward FLOPs bar
- total FLOPs
- effective FLOP/s assumption clearly labeled

### KV cache

Preferred visual:
- no-cache recomputation triangle
- cache append-only timeline
- seq_len slider
- batch slider
- cache GB estimate

### Power-law derivation

Preferred visual:
- left panel: IsoFLOP parabolas with vertices
- right panel: extracted vertices on log-log axes
- fitted regression line
- displayed slope

## Required page or lesson sections

Every visual should include:

1. What you are seeing
2. How to interact
3. What you should notice
4. Why this matters
5. One misconception to avoid
6. Where this visual lies / simplifies reality
7. 3 quiz questions

## Existing sessions/ compatibility

When modifying `sessions/**/lesson.html`, preserve:

- navigation
- localStorage
- existing quiz behavior
- existing playground behavior
- script ordering unless necessary
- offline behavior when possible

If adding external D3, prefer local vendor file if present. Otherwise use CDN only if the repo already allows it.

## Accessibility

- use semantic HTML
- do not rely only on color
- use readable labels
- include captions
- make controls keyboard-friendly where reasonable

## Done definition

The visual must:

- run locally
- teach one mechanism
- include clear labels
- include a misconception note
- not break the lesson
