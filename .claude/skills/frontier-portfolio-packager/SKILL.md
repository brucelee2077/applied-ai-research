---
name: frontier-portfolio-packager
description: Convert weekly Frontier Lab study outputs into a polished portfolio artifact: technical writeup, README, demo page, repo summary, or interview story.
---

# Frontier Portfolio Packager

Turn messy learning outputs into career signal.

## Output options

Depending on the request, create one of:

```text
portfolio/week-<n>-summary.md
writeups/<topic>.md
portfolio/index.html
README.md update
interview-stories/<topic>.md
```

## Weekly portfolio structure

```markdown
# Week <n>: <theme>

## What I studied

## What I built

## Main technical insight

## Visual artifact

## Experiment result

## What failed

## Why this matters for frontier labs

## Interview-ready explanation

## Next week
```

## Blog / writeup style

Make the learner sound like a research engineer:

- problem-first
- mechanism-driven
- evidence-based
- honest about limitations
- clear about next experiments

Avoid sounding like:

- motivational diary
- generic ML summary
- “I followed a tutorial”
- empty AI hype

## Interview story format

```markdown
Situation:
I wanted to understand <concept> because <frontier-lab relevance>.

Task:
I built <artifact>.

Action:
I implemented <mechanism>, tested <thing>, and measured <result>.

Result:
I learned <insight> and identified <limitation>.

Frontier relevance:
This maps to <training/inference/evals/systems/research engineering> because <reason>.
```

## Evidence rule

Every portfolio claim should point to an artifact:

- code file
- result file
- visualization
- note
- benchmark
- diagram
- experiment log

## Done definition

The packaged artifact should be understandable by another engineer without reading the whole repo.
